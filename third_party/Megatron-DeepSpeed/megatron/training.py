# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
import json
import socket
import numpy as np
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from collections import OrderedDict
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import os
from pathlib import Path
import re
import gc
import copy

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0, is_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint, get_additional_state
from megatron.model import Float16Module
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron, destroy_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator, update_rotary_pos_emb
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model import LlamaModelPipe, GPTModelPipe, Qwen3ModelPipe
from deepspeed.utils.timer import STEP_MICRO_TIMER

from deepspeed import comm as dist
from sailor.profiling.profile_utils import (
    add_hooks, take_time_fwd_pre, take_time_and_mem_fwd, take_time_bwd_pre, take_time_bwd,
    time_fwd, time_bwd, hook_params, collect_mem_info, take_mem_oobleck, take_mem_oobleck_pre,
    alloc_res_mem
)
from sailor.Planner.baselines.Galvatron.core import GalvatronProfiler
from sailor.Worker.checkpoint.chk_manager import Chk_manager

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

'''
Since v0.9.0, deepspeed.initialize() has forbidden simultaneous setting of args.deepspeed_config (Path) and ds_config dict.
So, we use ds_config dict which is the more flexible option.
'''
def _create_ds_config_dict():
    args = get_args()
    if isinstance(args.deepspeed_config, dict) :
        ds_config_dict = args.deepspeed_config
    else:
        with open(args.deepspeed_config, 'r', encoding='utf-8') as config_file:
            ds_config_dict = json.load(config_file)

    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}

    # Clear config path
    args.deepspeed_config = None

    return ds_config_dict

def get_lora_params(model_chunks):
    lora_params = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
    return lora_params

def load_lora_config(args):
    args = get_args()
    if args.enable_lora and args.lora_config_file:
        print_rank_0(f"> Loading LoRA configuration from {args.lora_config_file}")
        with open(args.lora_config_file, 'r') as f:
            lora_config = json.load(f)
        for key, value in lora_config.items():
            setattr(args, key, value)
    else:
        setattr(args, 'enable_lora', False)

# called every time a restart happens
def restart(
    all_args,
    model_provider,
    model_type,
    train_valid_test_dataset_provider,
    extra_args_provider=None,
    args_defaults={},
    data_post_process=None,
    external_args={}
):
    initialize_megatron(
        args = all_args,
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        external_args=external_args
    )

    args = get_args()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        if "curriculum_learning" in args.deepspeed_config_dict and \
            "enabled" in args.deepspeed_config_dict["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_config_dict[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_config_dict["curriculum_learning"])
        if "compression_training" in args.deepspeed_config_dict:
            args.compression_training = True

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)

    if args.rank == 0:
        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)

    if args.save:
        chk_manager = Chk_manager(
            args.save,
            model[0].module.parts[model[0].stage_id],
            args.rank,
            mpu.get_pipeline_model_parallel_rank(),
            mpu.get_tensor_model_parallel_rank(),
        )
        assert args.deepspeed, "checkpointing only supported with deepspeed"
    else:
        chk_manager = None

    if args.load:
        start_iter = load_simple(model[0].module.parts[model[0].stage_id], model, optimizer)
    else:
        start_iter = 0

    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)


    return model, optimizer, opt_param_scheduler, chk_manager, start_iter, \
        train_data_iterator, valid_data_iterator, test_data_iterator


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             all_args=None,
             cleanup_event=None,
             restart_event=None,
             arg_dict=None,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None,
             external_args={}
    ):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    if cleanup_event:
        # for NCCL error handling - only if run with controller/agent support
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "70"
        os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"

    reconf_start_time = time.time()
    # Initalize and get arguments, timers, and Tensorboard writer.

    all_args_copy = copy.deepcopy(all_args)
    initialize_megatron(
        args = all_args,
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        external_args=external_args
    )
    load_lora_config(all_args)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    if get_accelerator().device_name() == 'cuda':
        set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = get_accelerator().DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    rank, world_size, master_ip, master_port, local_rank = args.rank, args.world_size, args.master_ip, args.master_port, args.local_rank
    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        if "curriculum_learning" in args.deepspeed_config_dict and \
            "enabled" in args.deepspeed_config_dict["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_config_dict[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_config_dict["curriculum_learning"])
        if "compression_training" in args.deepspeed_config_dict:
            args.compression_training = True


    rank = torch.distributed.get_rank()

    torch.cuda.synchronize()
    print(f"[RECONFIGURATION] Megatron + Distributed Init time took {time.time()-reconf_start_time} sec")
    reconf_start_model_time = time.time()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)

    if args.rank == 0:
        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)

    if args.save:
        chk_manager = Chk_manager(
            args.save,
            model[0].module.parts[model[0].stage_id],
            args.rank,
            mpu.get_pipeline_model_parallel_rank(),
            mpu.get_tensor_model_parallel_rank(),
        )
        assert args.deepspeed, "checkpointing only supported with deepspeed"
    else:
        chk_manager = None

    if args.sailor_profile:
        if isinstance(model[0].module, GPTModelPipe):
    	    hook_params.set_params('tied_modules', 'loss_fn')
        else:
    	    hook_params.set_params('1', 'loss_fn')
        add_hooks(model[0].module, take_time_fwd_pre, take_time_and_mem_fwd, take_time_bwd_pre, take_time_bwd)

    timers('model-and-optimizer-setup').stop()
    print('after model, optimizer, and learning rate '
                   'scheduler are built', flush=True)

    if args.load:
        start_iter = load_simple(model[0].module.parts[model[0].stage_id], model, optimizer)
    else:
        start_iter = 0

    torch.cuda.synchronize()
    print(f"[RECONFIGURATION] Model reloading took {time.time()-reconf_start_model_time} sec")

    reconf_start_data_time = time.time()
    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)


    timers('train/valid/test-data-iterators-setup').stop()
    print('after dataloaders are built', flush=True)

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None
    if args.mos or args.kd: # Set up teacher model
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print('done with setup ...', flush=True)
    # timers.log(['model-and-optimizer-setup',
    #             'train/valid/test-data-iterators-setup'], barrier=True)
    print('after timers.log ...', flush=True)

    ####### for profiling

    tmp_world_size = mpu.get_tensor_model_parallel_world_size()
    pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = torch.distributed.get_world_size()
    gbs = args.global_batch_size
    mbs = args.micro_batch_size

    num_transformer_layers_original = args.num_transformer_layers_original

    if args.galvatron_profile:
        galvatron_profiler = GalvatronProfiler(args, pp_world_size, tmp_world_size, gbs, mbs, world_size)
        gconfig =  [
            {
                'hidden_size': args.hidden_size,
                'seq_len': args.max_position_embeddings,
                'layer_num': args.num_layers
            }
        ]
        galvatron_profiler.set_profiler_dist(args.results_dir, gconfig, args.model_name)
    else:
        galvatron_profiler = None

    if args.oobleck_profile:
        add_hooks(model[0].module, take_mem_oobleck_pre, take_mem_oobleck)
        input = next(train_data_iterator)
        batch = model[0].module._megatron_batch_fn(input)
        loaded = []
        for x in batch[0]:
            assert torch.is_tensor(x)
            mine = x.clone().detach().cuda()
            loaded.append(mine)
        loaded = tuple(loaded)
        with torch.no_grad():
            output = model[0].module(loaded)
            torch.cuda.synchronize()

    if args.galvatron_profile:
        galvatron_profiler.profile_memory(0, "After creating model")

    # needed for Varuna
    last_layer = args.num_transformer_layers_original + 1 # TODO: might need to change for other models
    if args.varuna_profile:
        inputs_dict = {
            0: (
                torch.ones((args.micro_batch_size, args.seq_length), dtype=torch.int64).cuda(),
                torch.ones((args.micro_batch_size, args.seq_length), dtype=torch.int64).cuda(),
                torch.randint(low=0, high=1, size=(1, 1, args.seq_length, args.seq_length), dtype=torch.bool).cuda()
            ),
            1: torch.rand((args.seq_length, args.micro_batch_size, args.hidden_size), dtype=torch.float32).cuda(),
            last_layer: torch.rand((args.seq_length, args.micro_batch_size, args.hidden_size), dtype=torch.float32).cuda(),
        }

        if args.use_embedding:
            layer_id = 0
        elif args.use_transformer:
            layer_id = 1
        elif args.use_last:
            layer_id = last_layer

        batch = inputs_dict[layer_id]
        num_runs = 10
        avg_mem_usage = 0
        # adapted from: https://github.com/microsoft/varuna/blob/master/varuna/profiler.py#L823
        for i in range(num_runs):
            torch.cuda.reset_max_memory_allocated()

            fwd_out = model[0].module(batch)
            if isinstance(fwd_out, tuple):
                fwd_out = fwd_out[0]
            grads = 0.00001 * torch.ones(list(fwd_out.size())).cuda()
            fwd_out.backward(grads)
            optimizer.step()
            model[0].module.zero_grad()
            optimizer.zero_grad()

            mem_usage = torch.cuda.max_memory_allocated()
            print(f"Iteration {i}, mem_usage is {mem_usage}")
            avg_mem_usage += mem_usage

        mem_usage = avg_mem_usage / num_runs
        print(f"AVG mem_usage is {mem_usage}")
        if mpu.get_tensor_model_parallel_rank() == 0:
            res_file =f'{args.results_dir}/varuna_profile_{args.model_name}.json'
            if os.path.exists(res_file):
                with open(res_file, 'r') as f:
                    profile = json.load(f)
            else:
                profile = {}
            if str(args.micro_batch_size) not in profile:
                profile[str(args.micro_batch_size)] = {}
            if str(tmp_world_size) not in profile[str(args.micro_batch_size)]:
                profile[str(args.micro_batch_size)][str(tmp_world_size)] = {}
            if layer_id != 1:
                profile[str(args.micro_batch_size)][str(tmp_world_size)][str(layer_id)] = mem_usage
            else:
                for i in range(1, args.num_transformer_layers_original+1):
                    profile[str(args.micro_batch_size)][str(tmp_world_size)][str(i)] = mem_usage
            with open(res_file, 'w') as f:
                json.dump(profile, f, indent=2)

    torch.cuda.synchronize()
    total_reconf_time = time.time()-reconf_start_time

    print(f"[RECONFIGURATION] Data setup took {time.time()-reconf_start_data_time} sec")

    print(f"[RECONFIGURATION] Total reconf time took {total_reconf_time} sec")

    if args.enable_lora:
        optimizer_params = get_lora_params(model)

        print_rank_0(f"Found {len(optimizer_params)} trainable LoRA parameters.")
        for param in optimizer_params:
            param.requires_grad = True

    # in a while loop for restarts
    exception = False
    while True:

        if not args.skip_train and not args.oobleck_profile and not args.varuna_profile:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_add_retriever:
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = start_iter
            if args.do_train and args.train_iters > 0:
                iteration, exception = train(start_iter, forward_step_func,
                                model, optimizer, opt_param_scheduler,
                                train_data_iterator, valid_data_iterator,
                                process_non_loss_data_func, chk_manager, galvatron_profiler, cleanup_event)

            #print_datetime('after training is done')
            # Clean the model
            if args.compression_training:
                model = [redundancy_clean(model[0], args.deepspeed_config_dict, mpu)]

            # if args.save and iteration != 0:
            #     save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        else:
            print_rank_0('skipping training (--skip-train is on) ...')

            iteration = args.iteration

        if args.galvatron_profile and torch.distributed.get_rank()==0:
            galvatron_profiler.process_profiled_data("memory")

        # these are only for profiling
        # 1. Time for forward and backward
        if args.sailor_profile:
            time_fwd_avg = {}
            for k, v in time_fwd.items():
                v_metrics = v[5:]
                time_fwd_avg[k] = round(np.average(np.asarray(v_metrics)), 6)

            time_bwd_avg = {}
            for k, v in time_bwd.items():
                v_metrics = v[5:]
                time_bwd_avg[k] = round(np.average(np.asarray(v_metrics)), 6)

            print('FORWARD: ', time_fwd_avg)
            print('BACKWARD: ', time_bwd_avg)

            mean_opt_time_sec = round(model[0].timers(STEP_MICRO_TIMER).mean()/1000, 6)
            print(f"MEAN OPTIMIZER TIME IS {mean_opt_time_sec}")

            # TODO: adjust
            last_idx = args.num_layers + 3 # transformer start from 3 for some reason
            if isinstance(model[0].module, LlamaModelPipe) or isinstance(model[0].module, Qwen3ModelPipe):
                last_idx += 2

            res_dict_time = {}
            res_dict_mem = {}

            if isinstance(model[0].module, GPTModelPipe):
                time_fwd_avg["1"] = time_fwd_avg["tied_modules"]
                time_bwd_avg["1"] = time_bwd_avg["tied_modules"]

            res_dict_time["0"] = [time_fwd_avg["1"], time_bwd_avg["1"], mean_opt_time_sec]
            if isinstance(model[0].module, GPTModelPipe):
                res_dict_mem["0"] = collect_mem_info(["tied_modules"], 1, 1)
            else:
                res_dict_mem["0"] = collect_mem_info(["1"], 1, 1)

            if isinstance(model[0].module, GPTModelPipe):
                res_dict_time["1"] = [time_fwd_avg["2"], time_bwd_avg["2"], mean_opt_time_sec]
                res_dict_mem["1"] = collect_mem_info(["2"], 1, 1)
                for i in range(1, num_transformer_layers_original):
                    res_dict_time[str(i+1)] = [time_fwd_avg["3"], time_bwd_avg["3"], mean_opt_time_sec]
                    res_dict_mem[str(i+1)] = collect_mem_info(["3"], 1, 1)
            else:
                res_dict_time["1"] = [time_fwd_avg["3"], time_bwd_avg["3"], mean_opt_time_sec]
                res_dict_mem["1"] = collect_mem_info(["3"], 1, 1)
                for i in range(1, num_transformer_layers_original):
                    res_dict_time[str(i+1)] = [time_fwd_avg["4"], time_bwd_avg["4"], mean_opt_time_sec]
                    res_dict_mem[str(i+1)] = collect_mem_info(["4"], 1, 1)

            if isinstance(model[0].module, LlamaModelPipe) or isinstance(model[0].module, Qwen3ModelPipe):
                rms_norm = str(num_transformer_layers_original+1)
                head_loss = str(num_transformer_layers_original+2)

                res_dict_time[rms_norm] =  [time_fwd_avg["6"], time_bwd_avg["6"], mean_opt_time_sec]
                res_dict_mem[rms_norm] = collect_mem_info(["6"], 1, 1)

                res_dict_time[head_loss] =  [
                    time_fwd_avg["7"]+ time_fwd_avg["loss_fn"],
                    time_bwd_avg["7"] + time_bwd_avg["loss_fn"],
                    mean_opt_time_sec
                ]
                res_dict_mem[head_loss] = collect_mem_info(["7", "loss_fn"], 1, 1)
            elif isinstance(model[0].module, GPTModelPipe):
                head_loss = str(num_transformer_layers_original+1)
                res_dict_time[head_loss] =  [
                time_fwd_avg["4"]+ time_fwd_avg["loss_fn"],
                time_bwd_avg["4"] + time_bwd_avg["loss_fn"],
                mean_opt_time_sec
                ]
                res_dict_mem[head_loss] = collect_mem_info(["4", "loss_fn"], 1, 1)
            else:
                head_loss = str(num_transformer_layers_original+1)
                res_dict_time[head_loss] =  [
                time_fwd_avg["5"]+ time_fwd_avg["loss_fn"],
                time_bwd_avg["5"] + time_bwd_avg["loss_fn"],
                mean_opt_time_sec
                ]
                res_dict_mem[head_loss] = collect_mem_info(["5", "loss_fn"], 1, 1)


            res_dict = {
                "timing": res_dict_time,
                "memory": res_dict_mem
            }

            Path(args.results_dir).mkdir(parents=True, exist_ok=True)
            if mpu.get_tensor_model_parallel_rank() == 0:
                with open(f'{args.results_dir}/profile_{args.model_name}_{args.gpu_type}_{tmp_world_size}_{args.micro_batch_size}.json', 'w') as f:
                    json.dump(res_dict, f, indent=2)


        if args.oobleck_profile:
            act_mem_res = {}
            act_mem_res["0"] = alloc_res_mem["1"][0]
            act_mem_res["1"] = alloc_res_mem["3"][0]
            for i in range(1, num_transformer_layers_original):
                act_mem_res[str(i+1)] = alloc_res_mem["4"][0]
            if isinstance(model[0].module, LlamaModelPipe):
                rms_norm = str(num_transformer_layers_original+1)
                head_loss = str(num_transformer_layers_original+2)
                act_mem_res[rms_norm] = alloc_res_mem["6"][0]
                act_mem_res[head_loss] = alloc_res_mem["7"][0] + alloc_res_mem["loss_fn"][0]
            else:
                head_loss = str(num_transformer_layers_original+1)
                act_mem_res[head_loss] = alloc_res_mem["5"][0] + alloc_res_mem["loss_fn"][0]
            if mpu.get_tensor_model_parallel_rank() == 0:
                res_file = f'{args.results_dir}/oobleck_profile_{args.model_name}_{args.gpu_type}.json'
                if os.path.exists(res_file):
                    with open(res_file, 'r') as f:
                        profile = json.load(f)
                else:
                    profile = {}
                profile[str(tmp_world_size)] = act_mem_res
                with open(res_file, 'w') as f:
                    json.dump(profile, f, indent=2)

        if (cleanup_event is None) or (not cleanup_event.is_set() and not exception):
            print(f"Training done!")
            if chk_manager:
                chk_manager.kill_checkpoint()

            print_memory = int(os.getenv("PRINT_MEMORY", 0))
            if print_memory:
                # device_id = torch.cuda.current_device()
                # tp = mpu.get_tensor_model_parallel_rank()
                # dp = mpu.get_data_parallel_rank()
                # pp = mpu.get_pipeline_model_parallel_rank()
                # source = f"memory_log_{device_id}"
                # dest = f"memory_log_{dp}_{pp}_{tp}"
                # os.system(f"mv {source} {dest}")

                Path(args.results_dir).mkdir(parents=True, exist_ok=True)
                if mpu.get_tensor_model_parallel_rank() == 0:
                    with open(f'{args.results_dir}/profile_{args.model_name}_{args.gpu_type}_{tmp_world_size}_{args.micro_batch_size}.json', 'w') as f:
                        json.dump(res_dict, f, indent=2)
            break
        else:
            print(f"TIME TO RESTART")
            old_rank = torch.distributed.get_rank()
            print(f"************************************************* RANK {old_rank}, BEFORE CLEANUP: {torch.cuda.memory.memory_allocated()}, {torch.cuda.memory.memory_reserved()}")

            ############### cleanup
            if exception:
                print(f"WAITING FOR CLEANUP SIGNAL")
                cleanup_event.wait()

            # 1. process groups
            destroy_megatron()

            if chk_manager:
                chk_manager.kill_checkpoint()
            del chk_manager

            # 2. model and deepspeed engine
            model[0].destroy()
            model[0].batch_timer = None
            del model[0].module
            model[0] = None
            model = None
            optimizer = None
            opt_param_scheduler = None
            del optimizer
            del model
            torch.cuda.empty_cache()
            print(f"************************************************* RANK {old_rank}, AFTER PARTIAL CLEANUP: {torch.cuda.memory.memory_allocated()}, {torch.cuda.memory.memory_reserved()}")

            # 3. data iterators

            train_data_iterator = None
            valid_data_iterator = None
            test_data_iterator = None

            print(f"************************************************* RANK {old_rank}, AFTER DATA CLEANUP: {torch.cuda.memory.memory_allocated()}, {torch.cuda.memory.memory_reserved()}")
            cleanup_event.clear()
            print(f"RANK {old_rank}, CLEANUP DONE")

            restart_event.wait()
            restart_event.clear()

            # update args and reinitialize
            all_args_copy = arg_dict['args']
            model, optimizer, opt_param_scheduler, chk_manager, start_iter, train_data_iterator, valid_data_iterator, test_data_iterator = restart(
                all_args_copy,
                model_provider,
                model_type,
                train_valid_test_dataset_provider,
                extra_args_provider=extra_args_provider,
                args_defaults=args_defaults,
                data_post_process=data_post_process,
                external_args=external_args
            )


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def setup_teacher_model(args, model_provider):

    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    print_rank_0('***>>>>> Setting up the teacher model')

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model

def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    use_embedding = True
    use_transformer = True
    use_last = True
    if args.varuna_profile:
        use_embedding = args.use_embedding
        use_transformer = args.use_transformer
        use_last = args.use_last

    if args.layers_per_stage:
        layers_per_stage = json.loads(args.layers_per_stage)
    else:
        layers_per_stage = None

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                use_embedding=use_embedding,
                use_transformer=use_transformer,
                use_last=use_last,
                layers_per_stage=layers_per_stage
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process,
            use_embedding=use_embedding,
            use_transformer=use_transformer,
            use_last=use_last,
            layers_per_stage=layers_per_stage
        )
        model.model_type = model_type

    num_params_list = list((p.numel()) for p in model.parameters())
    num_params = sum(num_params_list)
    print(f"---------------------------------- RANK {torch.distributed.get_rank()}, NUM_PARAMS IS {num_params}, num_params_list size is {len(num_params_list)}")
    # for name, layer in model.named_children():
    #     print(name, sum(p.numel() for p in layer.parameters()))


    if not isinstance(model, list):
        model = [model]

    #torch.cuda.empty_cache()
    # print(f"RANK {torch.distributed.get_rank()}, ALLOCATED MEM IS {torch.cuda.memory_allocated()/1e9} GB, RESERVED IS {torch.cuda.memory_reserved()/1e9} GB, MAX RESERVED IS {torch.cuda.max_memory_reserved()/1e9} GB")
    #time.sleep(100)

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    # args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    # assert args.allow_transformer_engine or args.transformer_impl == 'local', \
    #     'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.to(get_accelerator().current_device_name())


    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = get_accelerator().current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler

def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        # When loading just the model weights, ZeRO can be disabled.
        if 'zero_optimization' in args.deepspeed_config_dict:
            del args.deepspeed_config_dict['zero_optimization']

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            config=args.deepspeed_config_dict,
        )

        assert not isinstance(model, deepspeed.PipelineEngine), \
            'Weight loading only mode is not supported in pipeline parallelism yet.'

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration = load_checkpoint_by_layer(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)
    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              teacher=False,
                              data_post_process=None,
                              build_train_valid_test_datasets_provider=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
                model=model[0],
                args=args,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        print_rank_0('***>>>>> Student model, global step:{}'.format(student_global_steps))


    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config_dict, mpu)]

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        opt_param_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        # opt_param_scheduler is the old lr_scheduler plus weight decay scheduling
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = mpu.get_pipeline_model_parallel_world_size()
        if args.data_efficiency_curriculum_learning and build_train_valid_test_datasets_provider is not None:
            train_ds = None
            # Only need to build dataset on tp rank 0 since Megatron has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if mpu.get_tensor_model_parallel_rank() == 0:
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                    update_train_iters(args)
                else:
                    train_samples = args.train_iters * args.global_batch_size
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (args.train_iters // args.eval_interval + 1) * \
                            args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [train_samples,
                                            eval_iters * args.global_batch_size,
                                            test_iters * args.global_batch_size]
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples)
            model, optimizer, args.deepspeed_dataloader, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                training_data=train_ds,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    # Compression has its own checkpoint loading path (e.g, loading both teacher and student models). So if compression is enabled, we skip the following checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        if args.load is not None:
            timers = get_timers()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)

    return model, optimizer, opt_param_scheduler

def get_opt_state(state_dict):
    num_opt_params = 0
    num_opt_bytes = 0
    for _,v in state_dict['state'].items():
        for _, vi in v.items():
            num_opt_params += vi.numel()
            num_opt_bytes += vi.numel() * vi.element_size()

    #print(f"OPTIMIZER: num_opt_params {num_opt_params}, num_opt_bytes {num_opt_bytes}")


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config, iteration, galvatron_profiler=None):

    """Single training step."""
    args = get_args()
    timers = get_timers()

    try:
        skipped_iter = 0
        num_zeros_in_grad = 0
        assert isinstance(model[0], deepspeed.PipelineEngine)

        torch.cuda.synchronize()
        start_step = time.time()

        print(f"RANK {torch.distributed.get_rank()}, BEFORE TRAIN BATCH")
        loss = model[0].train_batch(data_iter=data_iterator, galvatron_profiler=galvatron_profiler, iteration=iteration)

        torch.cuda.synchronize()
        end_step = time.time()

        log_cmd = f"Iteration took {end_step-start_step} sec\n"
        print(f"******************* {log_cmd}", flush=True)
        print(type(model[0]))
        model[0].log_file.write(log_cmd)

        get_opt_state(optimizer.state_dict())

        additional_losses = model[0].get_additional_losses()
        loss_key = 'lm loss' if additional_losses is None else 'loss'  # use "lm loss" for backward compatibility
        loss_dict = OrderedDict({loss_key: loss})
        if additional_losses is not None:
            loss_dict.update(additional_losses)
        grad_norm = model[0].get_global_grad_norm()
        return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad, False
    except RuntimeError as e:
        print(f"-------------- GOT EXCEPTION {e}")
        return None, None, None, None, True


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            writer.add_scalar('batch-size/batch-size vs tokens', batch_size,
                              args.consumed_train_tokens)
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                              args.consumed_train_tokens)
        if args.fp16 and loss_scale and args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size/world-size', args.world_size, iteration)
            writer.add_scalar('world-size/world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            writer.add_scalar('world-size/world-size vs tokens', args.world_size,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if hasattr(args, 'actual_seq_length'):
            writer.add_scalar('seqlen/actual_seq_length', args.actual_seq_length,
                              iteration)
            writer.add_scalar('seqlen/actual_seq_length vs samples', args.actual_seq_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/actual_seq_length vs tokens', args.actual_seq_length,
                              args.consumed_train_tokens)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar('seqlen/curriculum_seqlen', args.curriculum_seqlen,
                              iteration)
            writer.add_scalar('seqlen/curriculum_seqlen vs samples', args.curriculum_seqlen,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/curriculum_seqlen vs tokens', args.curriculum_seqlen,
                              args.consumed_train_tokens)
        if args.random_ltd:
            writer.add_scalar('seqlen/random_ltd_reserved_length', args.random_ltd_reserved_length,
                              iteration)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs samples', args.random_ltd_reserved_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs tokens', args.random_ltd_reserved_length,
                              args.consumed_train_tokens)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group['params']):
                    opt_stats[0] += (torch.norm(optimizer.state[param]['exp_avg_sq']).item())**2
                    opt_stats[1] += (torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt()).item())**2
                    opt_stats[2] += (torch.norm(optimizer.state[param]['exp_avg']).item())**2
                    opt_stats[3] += (torch.norm(param).item())**2
                    opt_stats[4] += torch.norm(optimizer.state[param]['exp_avg_sq'],p=1).item()
                    opt_stats[5] += torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt(),p=1).item()
                    opt_stats[6] += torch.norm(optimizer.state[param]['exp_avg'],p=1).item()
                    opt_stats[7] += torch.norm(param,p=1).item()
                    opt_stats_2[0] = max(opt_stats_2[0], abs(optimizer.state[param]['exp_avg_sq'].max().item()), abs(optimizer.state[param]['exp_avg_sq'].min().item()))
                    opt_stats_2[1] = max(opt_stats_2[1], optimizer.state[param]['exp_avg_sq'].sqrt().abs_().max().item())
                    opt_stats_2[2] = max(opt_stats_2[2], abs(optimizer.state[param]['exp_avg'].max().item()), abs(optimizer.state[param]['exp_avg'].min().item()))
                    opt_stats_2[3] = max(opt_stats_2[3], abs(param.max().item()), abs(param.min().item()))
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_sequence_data_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_sequence_data_parallel_group())

            if args.tensor_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_tensor_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group())

            if args.pipeline_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_pipeline_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group())

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar('optimizer/variance_l2 vs tokens', opt_stats[0]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l2 vs tokens', opt_stats[1]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l2 vs tokens', opt_stats[2]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l2 vs tokens', opt_stats[3]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_l1 vs tokens', opt_stats[4], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l1 vs tokens', opt_stats[5], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l1 vs tokens', opt_stats[6], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l1 vs tokens', opt_stats[7], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_abs_max vs tokens', opt_stats_2[0], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_abs_max vs tokens', opt_stats_2[1], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_abs_max vs tokens', opt_stats_2[2], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_abs_max vs tokens', opt_stats_2[3], args.consumed_train_tokens)

                writer.add_scalar('optimizer/variance_l2', opt_stats[0]**0.5, iteration)
                writer.add_scalar('optimizer/variance_sqrt_l2', opt_stats[1]**0.5, iteration)
                writer.add_scalar('optimizer/momentum_l2', opt_stats[2]**0.5, iteration)
                writer.add_scalar('optimizer/weight_l2', opt_stats[3]**0.5, iteration)
                writer.add_scalar('optimizer/variance_l1', opt_stats[4], iteration)
                writer.add_scalar('optimizer/variance_sqrt_l1', opt_stats[5], iteration)
                writer.add_scalar('optimizer/momentum_l1', opt_stats[6], iteration)
                writer.add_scalar('optimizer/weight_l1', opt_stats[7], iteration)
                writer.add_scalar('optimizer/variance_abs_max', opt_stats_2[0], iteration)
                writer.add_scalar('optimizer/variance_sqrt_abs_max', opt_stats_2[1], iteration)
                writer.add_scalar('optimizer/momentum_abs_max', opt_stats_2[2], iteration)
                writer.add_scalar('optimizer/weight_abs_max', opt_stats_2[3], iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length
        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
            model,
            args,
            elapsed_time,
            total_iterations
        )
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        tokens_per_gpu_per_second = tokens_per_sec / args.world_size
        tokens_per_gpu_per_second_per_replica = tokens_per_gpu_per_second / args.data_parallel_size
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            assert wandb.run is not None
            wandb_metrics = {
                'throughput/iteration-time': elapsed_time_per_iteration,  # 1000 ms / s
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/samples_per_sec_per_replica': samples_per_sec_per_replica,
                'throughput/tokens_per_sec': tokens_per_sec,
                'throughput/tokens_per_sec_per_replica': tokens_per_sec_per_replica,
                'throughput/tokens_per_gpu_per_sec': tokens_per_gpu_per_second,
                'throughput/tokens_per_gpu_per_sec_per_replica': tokens_per_gpu_per_second_per_replica,
                'throughput/tflops': tflops,
                'throughput/approx_params_in_billions': approx_parameters_in_billions,
                'throughput/elapsed_ms_per_iteration': elapsed_time_per_iteration,
                'throughput/iteration': iteration,
            }
            if loss_dict is not None:
                wandb_metrics |= {
                    f'loss/{k}': v for k, v in loss_dict.items()
                }
                wandb_metrics |= {'loss/iteration': iteration}
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        # log_string += ' consumed samples: {:12d} |'.format(
        #     args.consumed_train_samples)
        # log_string += ' consumed tokens: {:12d} |'.format(
        #     args.consumed_train_tokens)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        # log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        # log_string += ' global batch size: {:5d} |'.format(batch_size)
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb_metrics |= {
                'training/iteration': iteration,
                'training/iteration_time': elapsed_time_per_iteration,
                'training/iteration_time_vs_tokens': (
                    (elapsed_time_per_iteration
                        / args.consumed_train_tokens)
                ),
                'training/iteration_time_vs_samples': (
                    (elapsed_time_per_iteration
                        / args.consumed_train_samples),
                ),
                'training/consumed_samples': args.consumed_train_samples,
                'training/consumed_tokens': args.consumed_train_tokens,
            }
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = get_accelerator().FloatTensor([0.0])
        # if wandb is not None and getattr(wandb, 'run', None) is not None:
        #     wandb.log(wandb_metrics)
        # if loss_scale is not None:
        #     log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        # if grad_norm is not None:
        #     log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        # if num_zeros_in_grad is not None:
        #     log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        # if params_norm is not None:
        #     log_string += ' params norm: {:.3f} |'.format(params_norm)
        # if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        #     log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        # if args.random_ltd:
        #     log_string += ' random ltd reserved length: {:5d} |'.format(args.random_ltd_reserved_length)
        # log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        # log_string += ' number of skipped iterations: {:3d} |'.format(
        #     total_loss_dict[skipped_iters_key])
        # log_string += ' number of nan iterations: {:3d} |'.format(
        #     total_loss_dict[nan_iters_key])
        # log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        # log_string += ' tokens per gpu per second (tgs): {:.3f} |'.format(tokens_per_gpu_per_second)
        # log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        #print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    checkpoint_throughput_calculator(model, timers('save-checkpoint').elapsed(reset=False))
    timers.log(['save-checkpoint'])


def get_latest_step_and_metadata(directory):
    pattern = re.compile(r"^check_(\d+)$")
    max_x = None
    metadata=None

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            x = int(match.group(1))
            try:
                print(f"Loading: {directory}/{filename}")
                metadata_i = torch.load(f"{directory}/{filename}")
                if max_x is None or x > max_x:
                    max_x = x
                    metadata = metadata_i
            except Exception as e:
                print(f"Skipping {filename}, Exception is {e}")
    return max_x, metadata

def load_simple(start_layer, model, optimizer):
    args = get_args()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    iteration, metadata = get_latest_step_and_metadata(args.load)
    if iteration is None:
        return 0
    index = metadata['index']
    model[0].global_samples = metadata['global_samples']
    model[0].global_steps = metadata['global_steps']
    print(f"-------------- DO LOAD, iteration is {iteration}")


    # TODO: fix for fp16
    for i,module in enumerate(model[0].forward_funcs):
        key = i+start_layer
        if not hasattr(module, 'state_dict'):
            continue
        if hasattr(module, 'parameters'):
            cpu_buffer = torch.from_numpy(np.fromfile(f"{args.load}/module_{key}_{tp_rank}_index_{index}", dtype=np.float32))

            idx = 0
            # 1. load model
            model_state_dict = module.state_dict()
            for name, value in model_state_dict.items():
                if torch.is_tensor(value):
                    sz = torch.numel(value)
                    model_state_dict[name].copy_(cpu_buffer[idx:idx+sz].view(value.size()))
                    idx+=sz

            # 2. load optimizer
            for name, p in module.named_parameters():
                if not ('weight' in name or 'bias' in name):
                    continue
                if isinstance(optimizer.state[p], dict):
                    # TODO: generalize to other optimizers
                    optimizer.state[p] = {}
                    optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    optimizer.state[p]['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    sz = torch.numel(p.data)
                    optimizer.state[p]['exp_avg'].copy_(cpu_buffer[idx:idx+sz].view(optimizer.state[p]['exp_avg'].size()))
                    idx += sz
                    optimizer.state[p]['exp_avg_sq'].copy_(cpu_buffer[idx:idx+sz].view(optimizer.state[p]['exp_avg'].size()))
                    idx += sz
    print(f"------------- LOAD DONE")
    return iteration

def train(start_iteration, forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, chk_manager, galvatron_profiler=None, cleanup_event=None):
    """Train the model function."""

    start_train_time = time.time()
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    if args.random_ltd:
        # random-ltd requires different randomness on each rank
        import random
        random.seed(args.seed + torch.distributed.get_rank())

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = start_iteration
    rank = torch.distributed.get_rank()

    # Translate args to core configuration
    config = core_transformer_config_from_args(args)
    if not args.deepspeed:
        config.grad_scale_func = optimizer.scale_loss
    config.timers = timers

    #timers('interval-time', log_level=0).start(barrier=False)
    #print_datetime('before the start of training step')
    report_memory_flag = True
    if args.random_ltd:
        assert model[0].random_ltd_enabled()
        args.random_ltd_layer_num = model[0].random_ltd_scheduler.get_random_ltd_layer_num()

    print(f"[RECONFIGURATION] Time in train, before start is {time.time()-start_train_time}")

    while iteration < args.train_iters and (args.train_tokens is None or \
        args.consumed_train_tokens < args.train_tokens):

        if cleanup_event:
            if cleanup_event.is_set():
                print(f"[RECONFIGURATION] Time to restart!")
                return iteration, False

        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            #model[0].set_train_batch_size(global_batch_size)

        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                    args.iteration + 1)
            if iteration == 0 or curriculum_seqlen != args.curriculum_seqlen:
                if args.use_rotary_position_embeddings:
                    update_rotary_pos_emb(curriculum_seqlen)
            args.curriculum_seqlen = curriculum_seqlen
        args.curr_iteration = iteration

        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad, exception = \
            train_step(forward_step_func,
                    train_data_iterator,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    config,
                    iteration,
                    galvatron_profiler)
        if exception:
            return iteration, True

        if galvatron_profiler:
            galvatron_profiler.post_profile_memory(iteration)

        iteration += 1
        args.iteration = iteration
        new_samples = mpu.get_data_parallel_world_size() * \
                                    args.micro_batch_size * \
                                    get_num_microbatches()
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[0].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (args.actual_seq_length * (args.num_layers - args.random_ltd_layer_num) + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                        get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length

        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        # report_memory_flag = training_log(loss_dict, total_loss_dict,
        #                                 optimizer.param_groups[0]['lr'],
        #                                 iteration, loss_scale,
        #                                 report_memory_flag, skipped_iter,
        #                                 grad_norm, params_norm, num_zeros_in_grad,
        #                                 model, optimizer)

        # Autoresume
        if args.adlr_autoresume and \
        (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                            opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
        args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                    valid_data_iterator, model,
                                    iteration, process_non_loss_data_func,
                                    config, False)


        if args.save and args.save_interval and \
        iteration % args.save_interval == 0 and mpu.get_data_parallel_rank()==0: # only DP rank 0 saves
            print(f"-------------- DO SAVE global steps is {model[0].global_steps}")
            if chk_manager:
                if not chk_manager.initialized:
                    chk_manager.init_buffer_and_writer(model[0].forward_funcs, optimizer)
                    model[0].chk_manager = chk_manager

                while chk_manager.checkpoint_in_progress():
                    continue

                res_dict = get_additional_state(iteration)
                if isinstance(model[0], deepspeed.PipelineEngine):
                    # we keep only what is useful from the deepspeed state
                    deepspeed_state = {
                        "lr_scheduler": model[0].lr_scheduler.state_dict(),
                        "global_samples": model[0].global_samples,
                        "global_steps": model[0].global_steps
                    }
                    res_dict.update(deepspeed_state)
                chk_manager.save_checkpoint(res_dict)

                # end_checkp = time.time()
                # print(f"Checkpoint time was {end_checkp-start_checkp} sec")
        gc.collect() 
        torch.cuda.empty_cache()

    return iteration, False


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if 'moe' not in key:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    return total_loss_dict, collected_non_loss_data

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True, test=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            data_type = 'test' if test else 'validation'
            writer.add_scalar(f'lm-loss-validation/{key} {data_type}',
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs samples',
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs tokens',
                              total_loss_dict[key].item(),
                              args.consumed_train_tokens)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl', ppl,
                                  iteration)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs samples',
                                  ppl, args.consumed_train_samples)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs tokens',
                                  ppl, args.consumed_train_tokens)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    ds_sequence_parallel = mpu.get_sequence_parallel_world_size() > 1 or args.force_ds_sequence_parallel
    rank_in_parallel_group = mpu.get_sequence_parallel_rank() if ds_sequence_parallel else mpu.get_tensor_model_parallel_rank()

    if rank_in_parallel_group == 0:
        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = get_accelerator().LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = get_accelerator().LongTensor([0, 0, 0])


    # # Broadcast num tokens.
    if ds_sequence_parallel:
        torch.distributed.broadcast(flags,
                                    mpu.get_sequence_parallel_src_rank(),
                                    group=mpu.get_sequence_parallel_group())
    else:
        tmp_group = mpu.get_tensor_model_parallel_group()
        torch.distributed.broadcast(flags,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=tmp_group)
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
