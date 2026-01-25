import os
import argparse
import json
from os.path import expanduser
import os

class ModelConfig():
    def __init__(self, num_layers, hidden_size, heads, seq_length, prof_id) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.heads = heads
        self.seq_length = seq_length
        self.prof_id = prof_id

model_configs = {
    "OPT-350": ModelConfig(24, 1024, 16, 2048, "OPT"),
    "GPT-Neo-2.7": ModelConfig(32, 2560, 20, 2048, "GPT-Neo"),
    "OPT-30": ModelConfig(48, 7168, 56, 2048, "OPT"),
    "LLAMA-3-8": ModelConfig(32, 4096, 32, 1024, "LLAMA"),
    "QWEN-3-8" : ModelConfig(32, 4096, 8, 1024, "QWEN")
}

def run(bs, tp_size, pp_size, model_name, results_dir, num_prof_layers, profile=None, use_embedding=False, use_transformer=False, use_last=False):

    devices = list(range(tp_size*pp_size))
    devices_str = ",".join(str(x) for x in devices)

    print(devices_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str

    print(f"Run with bs {bs}, tp_size {tp_size}, pp_size {pp_size}")
    with open("ds_config.json", "r") as f:
        ds_conf = json.load(f)

    ds_conf["train_batch_size"] = bs
    ds_conf["train_micro_batch_size_per_gpu"] = bs
    ds_conf["gradient_accumulation_steps"] = 1

    with open("ds_config.json", "w") as f:
        json.dump(ds_conf, f, indent=2)

    config = model_configs[model_name]
    home_dir = "/projects/beis/kli44"

    cmd = (
        f"deepspeed pretrain_gpt.py "
        f"--tensor-model-parallel-size {tp_size} "
        f"--pipeline-model-parallel-size {pp_size} "
        f"--num-layers {num_prof_layers} "
        f"--num-transformer-layers-original {config.num_layers} "
        f"--hidden-size {config.hidden_size} "
        f"--num-attention-heads {config.heads} "
        f"--seq-length {config.seq_length} "
        f"--loss-scale 12 "
        f"--max-position-embeddings {config.seq_length} "
        f"--micro-batch-size {bs} "
        f"--lr 6.0e-5 "
        f"--min-lr 6.0e-6 "
        f"--lr-decay-style cosine "
        f"--log-interval 1 "
        f"--eval-iters 40 "
        f"--eval-interval 1000 "
        f"--data-path {home_dir}/sailor/third_party/Megatron-DeepSpeed/data/meg-gpt2-gsm8k-1k_text_document "
        f"--vocab-file {home_dir}/sailor/third_party/Megatron-DeepSpeed/data/gpt2-vocab.json "
        f"--merge-file {home_dir}/sailor/third_party/Megatron-DeepSpeed/data/gpt2-merges.txt "
        f"--save-interval 1000 "
        f"--split 98,2,0 "
        f"--clip-grad 1.0 "
        f"--weight-decay 0.1 "
        f"--adam-beta1 0.9 "
        f"--adam-beta2 0.95 "
        f"--init-method-std 0.006 "
        f"--deepspeed "
        f"--deepspeed_config ds_config.json "
        f"--model-name {config.prof_id} "
        f"--gpu-type A100 "
        f"--train-iters 10 "
	    f"--results-dir {results_dir} "
        f"--lora-config-file {home_dir}/sailor/lora_config.json "
        f"--use-triton-lora True "
        f"--enable-lora True "
    )
    if profile == "sailor":
        cmd += "--sailor-profile "
    elif profile == "oobleck":
        cmd += "--oobleck-profile "
    elif profile == "galvatron":
        cmd += "--galvatron-profile "
    elif profile == "varuna":
        cmd += "--varuna-profile "
        if use_embedding:
            cmd += "--use-embedding"
        elif use_transformer:
            cmd += "--use-transformer"
        elif use_last:
            cmd += "--use-last"
    else:
        raise NotImplementedError

    if config.prof_id == "GPT-Neo":
        cmd += "--window_size 256"

    os.system(cmd)

def run_all(bs_list, tp_size, pp_size, model_name, results_dir, num_prof_layers, profile=None, use_embedding=False, use_transformer=False, use_last=False):
    for bs in bs_list:
        run(bs, tp_size,pp_size, model_name, results_dir, num_prof_layers, profile, use_embedding, use_transformer, use_last)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to profile with different microbatch sizes"
    )
    parser.add_argument(
        "--tp", type=int, required=True, help="Tensor model parallelism"
    )
    parser.add_argument(
        "--pp", type=int, required=True, help="Pipeline parallelism"
    )
    parser.add_argument(
        "--bs", type=int, required=True, help="Micro batch size"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--num_prof_layers", type=int, required=True, help="Number or transformer layers for profile"
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="Type of profiling. Can be one of: (sailor, oobleck, galvatron, varuna)"
    )
    parser.add_argument('--use-embedding', action='store_true', help='Varuna-specific. Use the embedding layer')
    parser.add_argument('--use-transformer', action='store_true', help='Varuna-specific. Use the transformer layer')
    parser.add_argument('--use-last', action='store_true', help='Varuna-specific. Use the last layer')

    args = parser.parse_args()

    tp = args.tp
    pp = args.pp
    bs = args.bs
    model_name = args.model_name
    results_dir = args.results_dir
    num_prof_layers = args.num_prof_layers

    bs_list = [args.bs]

    run_all(
        bs_list,
        args.tp,
        args.pp,
        args.model_name,
        args.results_dir,
        args.num_prof_layers,
        args.profile,
        args.use_embedding,
        args.use_transformer,
        args.use_last
    )
