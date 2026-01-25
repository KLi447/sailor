#!/bin/bash

# profiles up to a large batch size - if OOM, it will just fail and do not generate the profile

mkdir -p logs
export SAILOR_LOGS_DIR=logs

python profile_all.py --tp 1 --pp 1 --bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor
# python profile_all.py --tp 2 --pp 1 --bs 8 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor
# python profile_all.py --tp 4 --pp 1 --bs 8 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor
