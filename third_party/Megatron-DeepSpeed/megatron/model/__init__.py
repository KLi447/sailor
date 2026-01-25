# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Adapted: https://github.com/microsoft/Megatron-DeepSpeed/blob/8822a5ced6ce74d926fbe0f49cdca6bb3389bef8/megatron/model/__init__.py

from deepspeed.accelerator.real_accelerator import get_accelerator
if get_accelerator().device_name() == 'cuda':
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
    from apex.normalization import MixedFusedRMSNorm as RMSNorm
else:
    from .rmsnorm import RMSNorm
    from torch.nn import LayerNorm

from .distributed import DistributedDataParallel
from .bert_model import BertModel
from .gpt_model import GPTModel, GPTModelPipe
from .gpt_neo_model import GPTNeoModelPipe
from .opt_model import OPTModelPipe
from .llama_model import LlamaModelPipe
from .t5_model import T5Model
from .language_model import get_language_model
from .module import Float16Module
from .qwen_model import Qwen3ModelPipe
