import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.model.module import MegatronModule, float16_to_fp32, fp32_to_float16
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.utils import get_linear_layer, init_method_normal, scaled_init_method_normal, attention_mask_func
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.language_model import Pooler

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import PipelineModule, LayerSpec

from .transformer import LoRAColumnParallelLinear, LoRARowParallelLinear


QWEN3_8B_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "head_dim": 128,
    "use_qkv_bias": True,
    "use_sliding_window": False,
    "sliding_window": 32768,
}


class Qwen3RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 40960, 
        base: float = 1000000.0, 
        device=None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3ParallelMLP(MegatronModule):
    def __init__(
        self, 
        init_method, 
        config, 
        output_layer_init_method, 
        moe: bool = False, 
        enable_expert_tensor_parallelism: bool = False
    ):
        super(Qwen3ParallelMLP, self).__init__()
        args = get_args()
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        if args.enable_lora:
            self.gate_proj = LoRAColumnParallelLinear(
                args.hidden_size,
                args.ffn_hidden_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )
        else:
            self.gate_proj = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.ffn_hidden_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

        if args.enable_lora:
            self.up_proj = LoRAColumnParallelLinear(
                args.hidden_size,
                args.ffn_hidden_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )
        else:
            self.up_proj = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.ffn_hidden_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

        self.activation_func = F.silu

        if args.enable_lora:
            self.down_proj = LoRARowParallelLinear(
                args.ffn_hidden_size,
                args.hidden_size,
                config=config,
                input_is_parallel=True,
                init_method=self.output_layer_init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )
        else:
            self.down_proj = tensor_parallel.RowParallelLinear(
                args.ffn_hidden_size,
                args.hidden_size,
                config=config,
                input_is_parallel=True,
                init_method=self.output_layer_init_method,
                skip_bias_add=True,
                bias=False,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

    def forward(self, hidden_states):
        gate_output = self.gate_proj(hidden_states)[0]
        up_output = self.up_proj(hidden_states)[0]
        
        intermediate = self.activation_func(gate_output) * up_output
        output, _ = self.down_proj(intermediate)
        
        return output


class Qwen3ParallelAttention(MegatronModule):
    def __init__(
        self, 
        init_method, 
        config,
        output_layer_init_method, 
        layer_number: int,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.causal
    ):
        super(Qwen3ParallelAttention, self).__init__()

        assert attention_type == AttnType.self_attn
        assert attn_mask_type == AttnMaskType.causal

        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
            
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = getattr(args, 'num_key_value_heads', args.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = args.kv_channels

        self.q_projection_size = self.head_dim * self.num_attention_heads
        self.kv_projection_size = self.head_dim * self.num_key_value_heads

        world_size = mpu.get_tensor_model_parallel_world_size()
        self.num_attention_heads_per_partition = core.utils.divide(
            self.num_attention_heads, world_size
        )
        self.num_key_value_heads_per_partition = core.utils.divide(
            self.num_key_value_heads, world_size
        )
        self.hidden_size_per_partition = self.head_dim * self.num_attention_heads_per_partition
        self.kv_hidden_size_per_partition = self.head_dim * self.num_key_value_heads_per_partition

        self.use_qkv_bias = getattr(args, 'use_qkv_bias', True)

        if args.enable_lora:
            self.q_proj = LoRAColumnParallelLinear(
                args.hidden_size,
                self.q_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )
        else:
            self.q_proj = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                self.q_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )

        if args.enable_lora:
            self.k_proj = LoRAColumnParallelLinear(
                args.hidden_size,
                self.kv_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )
        else:
            self.k_proj = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                self.kv_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )

        if args.enable_lora:
            self.v_proj = LoRAColumnParallelLinear(
                args.hidden_size,
                self.kv_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )
        else:
            self.v_proj = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                self.kv_projection_size,
                config=config,
                gather_output=False,
                init_method=self.init_method,
                bias=self.use_qkv_bias
            )

        coeff = None
        self.norm_factor = math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff
        )

        rope_theta = getattr(args, 'rope_theta', 1000000.0)
        max_position_embeddings = getattr(args, 'max_position_embeddings', 40960)
        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta
        )

        self.use_sliding_window = getattr(args, 'use_sliding_window', False)
        self.sliding_window = getattr(args, 'sliding_window', 32768)

        self.o_proj = tensor_parallel.RowParallelLinear(
            self.q_projection_size,
            args.hidden_size,
            config=config,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            bias=False
        )

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        layer_past=None,
        get_key_value: bool = False
    ):
        seq_len, batch_size, _ = hidden_states.size()

        query_layer, _ = self.q_proj(hidden_states)
        key_layer, _ = self.k_proj(hidden_states)
        value_layer, _ = self.v_proj(hidden_states)

        query_layer = query_layer.view(
            seq_len, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )
        key_layer = key_layer.view(
            seq_len, batch_size, self.num_key_value_heads_per_partition, self.head_dim
        )
        value_layer = value_layer.view(
            seq_len, batch_size, self.num_key_value_heads_per_partition, self.head_dim
        )

        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)

        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat([past_key.type_as(key_layer), key_layer], dim=2)
            value_layer = torch.cat([past_value.type_as(value_layer), value_layer], dim=2)
        
        if get_key_value:
            present = (key_layer, value_layer)

        q_seq_len = query_layer.size(2)
        kv_seq_len = key_layer.size(2)

        key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
        value_layer = self.repeat_kv(value_layer, self.num_key_value_groups)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / self.norm_factor

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.use_sliding_window:
            sliding_window_mask = torch.ones(
                q_seq_len, kv_seq_len, 
                dtype=torch.bool, 
                device=attention_scores.device
            )
            for i in range(q_seq_len):
                start = max(0, kv_seq_len - q_seq_len + i - self.sliding_window + 1)
                end = kv_seq_len - q_seq_len + i + 1
                sliding_window_mask[i, start:end] = False
            sliding_window_mask = sliding_window_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(sliding_window_mask, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        context_layer = context_layer.view(q_seq_len, batch_size, self.hidden_size_per_partition)

        output, _ = self.o_proj(context_layer)

        if get_key_value:
            output = [output, present]

        return output


class Qwen3ParallelTransformerLayer(MegatronModule):
    def __init__(
        self, 
        init_method, 
        config, 
        output_layer_init_method,
        layer_number: int,
        self_attn_mask_type=AttnMaskType.causal
    ):
        args = get_args()
        super(Qwen3ParallelTransformerLayer, self).__init__()
        
        self.layer_number = layer_number
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        rms_norm_eps = getattr(args, 'layernorm_epsilon', 1e-6)
        self.input_layernorm = Qwen3RMSNorm(args.hidden_size, eps=rms_norm_eps)

        self.self_attn = Qwen3ParallelAttention(
            self.init_method,
            config,
            self.output_layer_init_method,
            layer_number,
            attn_mask_type=self_attn_mask_type
        )

        self.post_attention_layernorm = Qwen3RMSNorm(args.hidden_size, eps=rms_norm_eps)

        self.mlp = Qwen3ParallelMLP(
            self.init_method, 
            config, 
            self.output_layer_init_method
        )

    def forward(
        self, 
        hidden_states, 
        attention_mask=None,
        layer_past=None, 
        get_key_value: bool = False
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value
        )

        if get_key_value:
            hidden_states, presents = hidden_states

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if get_key_value:
            hidden_states = [hidden_states, presents]
            
        return hidden_states


class Qwen3ParallelTransformerLayerPipe(Qwen3ParallelTransformerLayer):
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        
        if torch.is_tensor(inputs) or len(inputs) == 1:
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than expected.')


class Qwen3Embedding(MegatronModule):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config,
        init_method
    ):
        super(Qwen3Embedding, self).__init__()
        args = get_args()

        self.hidden_size = hidden_size
        self.init_method = init_method

        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, 
            self.hidden_size, 
            config=config,
            init_method=self.init_method
        )

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)


class Qwen3EmbeddingPipe(Qwen3Embedding):
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        embeddings = super().forward(input_ids)
        
        if hasattr(self._args, 'attn_mask'):
            return embeddings
        else:
            return embeddings, attention_mask


class Qwen3LMHead(MegatronModule):
    def __init__(
        self,
        config,
        hidden_size: int,
        vocab_size: int,
        init_method,
        parallel_output: bool = True
    ):
        super(Qwen3LMHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=vocab_size,
            config=config,
            bias=False,
            gather_output=not self.parallel_output,
            skip_bias_add=True,
            init_method=self.init_method
        )

    def forward(self, hidden_states):
        logits, _ = self.lm_head(hidden_states)
        return logits


class Qwen3LMHeadPipe(Qwen3LMHead):
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        logits = super().forward(hidden_states)

        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask


class Qwen3ParallelTransformer(MegatronModule):
    def __init__(
        self, 
        init_method, 
        config, 
        output_layer_init_method,
        self_attn_mask_type=AttnMaskType.causal,
        pre_process: bool = True, 
        post_process: bool = True
    ):
        super(Qwen3ParallelTransformer, self).__init__()
        args = get_args()
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        def build_layer(layer_number):
            return Qwen3ParallelTransformerLayer(
                self.init_method,
                config,
                self.output_layer_init_method,
                layer_number
            )

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size
            ) + (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList([
            build_layer(i + 1 + offset) for i in range(self.num_layers)
        ])

        if self.post_process:
            rms_norm_eps = getattr(args, 'layernorm_epsilon', 1e-6)
            self.final_layernorm = Qwen3RMSNorm(args.hidden_size, eps=rms_norm_eps)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask):

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask=attention_mask)
                return x_
            return custom_forward

        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask
            )
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        layer_past=None, 
        get_key_value: bool = False
    ):
        if layer_past is not None:
            assert get_key_value
        if get_key_value:
            assert not self.checkpoint_activations

        if not self.ds_inference:
            if self.pre_process:
                if self.fp32_residual_connection:
                    hidden_states = hidden_states.transpose(0, 1).contiguous().float()
                else:
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
            else:
                hidden_states = self.input_tensor

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states, attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None if layer_past is None else layer_past[index]
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_past=past,
                    get_key_value=get_key_value
                )
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        if self.post_process:
            if not self.ds_inference:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states

        if get_key_value:
            output = [output, presents]

        return output


def Qwen3CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class Qwen3Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, labels):
        return Qwen3CrossEntropy(output, labels)


class Qwen3ModelPipe(PipelineModule, MegatronModule):
    def __init__(
        self,
        config,
        parallel_output: bool = True,
        use_embedding: bool = True,
        use_transformer: bool = True,
        use_last: bool = True,
        layers_per_stage=None
    ):
        args = get_args()

        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(
            args.init_method_std, args.num_layers
        )
        self.parallel_output = parallel_output

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        if use_embedding:
            self.specs.append(_to_float16)
            self.specs.append(LayerSpec(
                Qwen3EmbeddingPipe,
                hidden_size=args.hidden_size,
                vocab_size=args.padded_vocab_size,
                init_method=self.init_method,
                config=config
            ))

            if args.fp32_residual_connection:
                self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
            else:
                self.specs.append(lambda x: x.transpose(0, 1).contiguous())

            if layers_per_stage:
                layers_per_stage[0] += 2

        if use_transformer:
            for layer_idx in range(args.num_layers):
                self.specs.append(LayerSpec(
                    Qwen3ParallelTransformerLayerPipe,
                    init_method=self.init_method,
                    config=config,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=layer_idx
                ))

            self.specs.append(lambda x: x.transpose(0, 1).contiguous())
            if layers_per_stage:
                layers_per_stage[-1] += 1

        if use_last:
            rms_norm_eps = getattr(args, 'layernorm_epsilon', 1e-6)
            self.specs.append(LayerSpec(Qwen3RMSNorm, args.hidden_size, eps=rms_norm_eps))
            if layers_per_stage:
                layers_per_stage[-1] += 1

            self.specs.append(LayerSpec(
                Qwen3LMHeadPipe,
                config=config,
                hidden_size=args.hidden_size,
                vocab_size=args.padded_vocab_size,
                init_method=self.init_method,
                parallel_output=self.parallel_output
            ))

            if args.fp16 or args.bf16:
                self.specs.append(float16_to_fp32)
                if layers_per_stage:
                    layers_per_stage[-1] += 1

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(
            num_pp=mpu.get_pipeline_model_parallel_world_size(),
            num_mp=mpu.get_tensor_model_parallel_world_size(),
            num_dp=mpu.get_data_parallel_world_size()
        )

        if layers_per_stage:
            layers_per_stage[-1] += 1

        print(f"======================= layers_per_stage is {layers_per_stage}")

        if layers_per_stage:
            layer_partitioning = [0] * (len(layers_per_stage) + 1)
            for i in range(len(layers_per_stage)):
                layer_partitioning[i+1] = layer_partitioning[i] + layers_per_stage[i]
        else:
            layer_partitioning = None

        print(f"============================================ layer_partitioning is {layer_partitioning}")

        super().__init__(
            layers=self.specs,
            loss_fn=Qwen3Loss(),
            topology=topo,
            activation_checkpoint_interval=interval,
            partition_method='uniform',
            layer_partitioning=layer_partitioning
        )

    def get_additional_losses(self):
        return None


class Qwen3Model(MegatronModule):
    def __init__(
        self, 
        config, 
        pre_process: bool, 
        post_process: bool, 
        parallel_output: bool = True, 
        add_pooler: bool = False
    ):
        super(Qwen3Model, self).__init__()
        args = get_args()
        
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.hidden_size = args.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.add_pooler = add_pooler
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(
            args.init_method_std, args.num_layers
        )
        self.self_attn_mask_type = AttnMaskType.causal
        self.padded_vocab_size = args.padded_vocab_size

        if self.pre_process:
            self.embedding = Qwen3Embedding(
                hidden_size=args.hidden_size,
                init_method=self.init_method,
                vocab_size=self.padded_vocab_size,
                config=config
            )

        self.transformer = Qwen3ParallelTransformer(
            self.init_method,
            config,
            self.output_layer_init_method,
            self_attn_mask_type=self.self_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process
        )

        if self.post_process:
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)

            self.lm_head = Qwen3LMHead(
                config=config,
                hidden_size=args.hidden_size,
                vocab_size=self.padded_vocab_size,
                init_method=self.init_method,
                parallel_output=self.parallel_output
            )

    def set_input_tensor(self, input_tensor):
        self.transformer.set_input_tensor(input_tensor)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels=None, 
        layer_past=None, 
        get_key_value: bool = False,
        pooling_sequence_index: int = 0
    ):
        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = input_ids

        hidden_states = self.transformer(
            hidden_states, 
            attention_mask, 
            layer_past=layer_past,
            get_key_value=get_key_value
        )

        if self.post_process:
            if get_key_value:
                hidden_states, presents = hidden_states

            if self.add_pooler:
                hidden_states = self.pooler(hidden_states, pooling_sequence_index)

            hidden_states = self.lm_head(hidden_states)

            if labels is None:
                if get_key_value:
                    return [hidden_states, presents]
                return hidden_states
            else:
                if self.fp16_lm_cross_entropy:
                    assert hidden_states.dtype == torch.half
                    loss = mpu.vocab_parallel_cross_entropy(hidden_states, labels)
                else:
                    loss = mpu.vocab_parallel_cross_entropy(hidden_states.float(), labels)
                return loss

        return hidden_states
