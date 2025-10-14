import os

import torch
import torch.nn.functional as F

from megatron import get_args, core
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from .module import MegatronModule, fp32_to_float16, float16_to_fp32
from .utils import init_method_normal, scaled_init_method_normal, attention_mask_func

from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe, get_num_experts_per_layer, LoRAColumnParallelLinear, LoRARowParallelLinear
from .enums import AttnMaskType, AttnType
from megatron.model import LayerNorm, RMSNorm

from .language_model import parallel_lm_logits
from megatron.core import mpu, tensor_parallel, sequence_parallel
from .utils import init_method_normal, scaled_init_method_normal, gather_and_init
from deepspeed.accelerator import get_accelerator

from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from collections import OrderedDict
import functools


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, labels):
        return CrossEntropy(output, labels)

################# LAYERS ##################

class OPTLMHead(MegatronModule):

    def __init__(self,
                 config,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True):
        args = get_args()
        super(OPTLMHead, self).__init__()

        self.embed_dim = hidden_size
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output
        self.do_layer_norm_before = True #args.do_layer_norm_before

        # TODO: project_out might be needed here

        if self.do_layer_norm_before and not args.remove_final_layer_norm:
            self.final_layer_norm = LayerNorm(self.embed_dim)


        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                config=config,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=self.init_method, )

    def forward(self, inputs):
        if self.do_layer_norm_before:
            inputs = self.final_layer_norm(inputs)

        logits, _ = self.lm_head(inputs)
        return logits


class OPTLMHeadPipe(OPTLMHead):

    def forward(self, inputs, **kwargs):
        #print(f"AT LMHead, INPUTS IS {inputs.shape}, type is {inputs.dtype}")

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

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask

#############################################################################


class OPTEmbedding(MegatronModule):
    """OPT Embedding, adapted from language_model.py"""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 embedding_weights_in_fp32=False):

        super(OPTEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = config.init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()
        self.add_position_embedding = args.add_position_embedding

        assert self.num_tokentypes == 0
        assert self.add_position_embedding

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config, init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # on HF, this is a 'OPTLearnedPositionalEmbedding' - how do they compare?
        self._position_embeddings_key = 'position_embeddings'
        if args.sequence_parallel:
            self.position_embeddings = tensor_parallel.layers.SequenceParallelPositionEmbedding(
                max_sequence_length, self.hidden_size)
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings.local_embeddings.weight)
        else:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, self.hidden_size)
            # Initialize the position embeddings.
            if args.perform_initialization:
                if args.zero_stage == 3:
                    gather_and_init(self.position_embeddings.weight, self.init_method)
                else:
                    self.init_method(self.position_embeddings.weight)

        # No need for project in/out at OPT-2.7B, but TODO for other models
        self.project_in = None
        self.project_out = None

        self.sequence_parallel = args.sequence_parallel


    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True


    @classmethod
    def from_pretrained(cls, model_path, config=None):
        module = torch.nn.utils.skip_init(cls, config).eval()  # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module


    def forward(self, input_ids, position_ids):

        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        words_embeddings = self.word_embeddings(input_ids)

        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)

        position_embeddings = self.position_embeddings(position_ids)

        # TODO: not sure if this is needed
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # embeddings = embeddings.transpose(0, 1).contiguous()

        embeddings = words_embeddings + position_embeddings

        #print(f"AFTER EMBEDDINGS, size is {embeddings.shape}")

        return embeddings


class OPTEmbeddingPipe(OPTEmbedding):

    def forward(self, inputs, **kwargs):

        # print(f"AT EMBEDDING:")
        # for x in inputs:
        #     print(x, x.shape, x.dtype)

        if not hasattr(self, '_args'):
            self._args = get_args()

        input_ids = inputs[0]
        position_ids = inputs[1]
        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[2]

        embeddings = super().forward(input_ids, position_ids)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return embeddings
        else:
            assert False
            return embeddings, attention_mask


    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

#############################################################################

class OPTParallelAttention(MegatronModule):
    def __init__(self, init_method, config,
                 output_layer_init_method,
                 attention_type=AttnType.self_attn):
        super(OPTParallelAttention, self).__init__()

        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.attn_mask_type = AttnMaskType.causal

        self.embed_dim = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.dropout = args.attention_dropout
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        world_size = mpu.get_tensor_model_parallel_world_size()

        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            self.embed_dim, args.num_attention_heads)
        self.hidden_size_per_partition = core.utils.divide(self.embed_dim, world_size)

        if (self.head_dim * args.num_attention_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {args.num_attention_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = True

         # Strided linear layer.
        if attention_type == AttnType.self_attn:
            if args.enable_lora and 'qkv' in args.lora_target_modules:
                self.query_key_value = LoRAColumnParallelLinear(
                    self.embed_dim, 3 * self.embed_dim,
                    lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    config=config, gather_output=False, init_method=self.init_method
                )
            else:
                self.query_key_value = tensor_parallel.ColumnParallelLinear(
                    self.embed_dim, 3 * self.embed_dim,
                    config=config, gather_output=False, init_method=self.init_method
                )

        coeff = None
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Output.
        if args.enable_lora and 'dense' in args.lora_target_modules:
            self.dense = LoRARowParallelLinear(
                self.embed_dim, self.embed_dim,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                config=config, input_is_parallel=True, init_method=self.output_layer_init_method, skip_bias_add=True
            )
        else:
            self.dense = tensor_parallel.RowParallelLinear(
                self.embed_dim, self.embed_dim,
                config=config, input_is_parallel=True, init_method=self.output_layer_init_method, skip_bias_add=True
            )


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

         # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                             (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,key_layer,value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)

        # TODO: what about cross-attention?
        #print(f"[S1] key_layer shape is {key_layer.shape}, value_layer shape is {value_layer.shape}")

###############################################################################################################

         # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=get_accelerator().current_device_name())

        # TODO: what to use here? (also for alpha and beta)
        self.norm_factor = 1.0
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0 / self.norm_factor))

        #torch.bmm(query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2), out=matmul_result)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]


###############################################################################################################

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)


        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, _ = self.dense(context_layer)

        # if get_key_value:
        #     output = [output, present]

        return output


class OPTParallelTransformerLayer(MegatronModule):
    def __init__(self, init_method, config, output_layer_init_method,
                 layer_number, moe=False, enable_expert_tensor_parallelism=False) -> None:

        args = get_args()
        super(OPTParallelTransformerLayer, self).__init__()

        self.embed_dim = args.hidden_size
        self.layer_number = layer_number

        self.attention = OPTParallelAttention(init_method, config, output_layer_init_method)

        self.do_layer_norm_before = True #args.do_layer_norm_before
        self.dropout = args.attention_dropout
        self.activation_fn = F.relu

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        
        if args.enable_lora and 'fc1' in args.lora_target_modules:
            self.fc1 = LoRAColumnParallelLinear(
                self.embed_dim, args.ffn_hidden_size,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                config=config, gather_output=False, init_method=init_method, skip_bias_add=True
            )
        else:
            self.fc1 = tensor_parallel.ColumnParallelLinear(
                self.embed_dim, args.ffn_hidden_size,
                config=config, gather_output=False, init_method=init_method, skip_bias_add=True
            )

        if args.enable_lora and 'fc2' in args.lora_target_modules:
            self.fc2 = LoRARowParallelLinear(
                args.ffn_hidden_size, self.embed_dim,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                config=config, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True
            )
        else:
            self.fc2 = tensor_parallel.RowParallelLinear(
                args.ffn_hidden_size, self.embed_dim,
                config=config, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True
            )

        self.final_layer_norm = LayerNorm(self.embed_dim)



    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=None,
            use_cache=None,
            past_key_value=None,):


        # if self.layer_number==0:
        #     print(f"AFTER LNORM, Hidden states shape is {hidden_states.shape}")

        residual = hidden_states

        # TODO: check
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.attention(
            hidden_states=hidden_states,
            layer_past=past_key_value,
            attention_mask=attention_mask,
        )

        #hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # TODO: check
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)[0]
        hidden_states = self.activation_fn(hidden_states)

        #print(f"After activation, hidden states shape is {hidden_states.shape}")

        hidden_states = self.fc2(hidden_states)[0]
        #hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        outputs=hidden_states
        #outputs = (hidden_states,)
        # if use_cache:
        #     outputs += (present_key_value,)

        return outputs



class OPTParallelTransformerLayerPipe(OPTParallelTransformerLayer):
    def forward(self, inputs, **kwargs):

        #print(f"AT TRANSFORMER, INPUTS IS {inputs.shape}, type is {inputs.dtype}")

        # TODO: from LLAMA, not sure if this would work
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


def half(v):
    return v.half()

def bfloat(v):
    return v.bfloat16()

def _to_float16(fp16, bf16, inputs):
    if fp16:
        return fp32_to_float16(inputs, half)
    elif bf16:
        return fp32_to_float16(inputs, bfloat)
    else:
        return inputs

def transpose_to_float(x):
    return x.transpose(0, 1).contiguous().float()

def transpose(x):
    return x.transpose(0, 1).contiguous()

class OPTModelPipe(PipelineModule,MegatronModule):
    """OPT Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 use_embedding=True,
                 use_transformer=True,
                 use_last=True,
                 layers_per_stage=None
        ):
        args = get_args()
        self.parallel_output = parallel_output

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                        config.num_layers)

        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method

        self.specs = []

        if use_embedding:
            self.specs.append(functools.partial(_to_float16, args.fp16, args.bf16))
            # Embedding layer
            self.specs.append(LayerSpec(OPTEmbeddingPipe,
                args.hidden_size,
                args.padded_vocab_size,
                args.max_position_embeddings,
                args.hidden_dropout,
                config,
                num_tokentypes=num_tokentypes,
                embedding_weights_in_fp32=args.embedding_weights_in_fp32,))

            if args.fp32_residual_connection:
                self.specs.append(transpose_to_float)
            else:
                self.specs.append(transpose)

            if layers_per_stage:
                layers_per_stage[0]+=2 # for the embedding

        # TODO: what to do with that?
        # # # LNorm
        # if args.do_layer_norm_before and not args.remove_final_layer_norm:
        #     self.specs.append(LayerSpec(LayerNorm, args.hidden_size))

        if use_transformer:
            for layer_idx in range(args.num_layers):
                #print(f"Add layer with idx {layer_idx}")
                self.specs.append(
                    LayerSpec(OPTParallelTransformerLayerPipe, self.init_method, config, self.output_layer_init_method, layer_idx))

        if use_last:
            self.specs.append(
                LayerSpec(OPTLMHeadPipe, config=config, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
                        init_method=self.init_method, parallel_output=self.parallel_output)
            )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)
            if layers_per_stage:
                layers_per_stage[-1]+=1

        # Cache losses
        self.moe_loss = None
        self.last_lm_loss = None    # detached, for display only
        self.last_moe_loss = None   # detached, for display only

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        elif args.recompute_granularity == "full" and args.recompute_method == 'uniform':
            # deepspeed's pipeline doesn't support the block recompute method
            interval = args.recompute_num_layers
        else:
            interval = 0

        #print(f"checkpoint_activations stuff: {args.checkpoint_activations}, {args.recompute_granularity}, {args.recompute_method}, {interval}")

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, PipeHeterogeneousTopology
        if args.distributed_config_file:
            topo = PipeHeterogeneousTopology(
                num_pp=mpu.get_pipeline_model_parallel_world_size(),
                num_mp=mpu.get_tensor_model_parallel_world_size(),
                num_dp=mpu.get_data_parallel_world_size(),
                input_file=args.distributed_config_file
            )
        else:
            topo = PipeModelDataParallelTopology(
                num_pp=mpu.get_pipeline_model_parallel_world_size(),
                num_mp=mpu.get_tensor_model_parallel_world_size(),
                num_dp=mpu.get_data_parallel_world_size()
            )

        if layers_per_stage:
            layers_per_stage[-1] += 1 # for the loss

        print(f"======================= layers_per_stage is {layers_per_stage}")

        if layers_per_stage:
            layer_partitioning = [0] * (len(layers_per_stage) + 1)
            for i in range(len(layers_per_stage)):
                layer_partitioning[i+1] = layer_partitioning[i] + layers_per_stage[i]
        else:
            layer_partitioning = None

        print(f"============================================ layer_partitioning is {layer_partitioning}")

        super().__init__(layers=self.specs,
                         loss_fn=Loss(),
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='uniform',
                         layer_partitioning=layer_partitioning)

    def get_additional_losses(self):
        return None
