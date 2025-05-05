#!/bin/bash

# example usages
# bash run_train.sh config/mono_alphabetic_sub.yaml 1 single-gpu 0 [Single GPU, no log file]
# bash run_train.sh config/mono_alphabetic_sub.yaml 1 single-gpu 4 logs/test.log [Single GPU, with log file]
# bash run_train.sh config/mono_alphabetic_sub.yaml 2 multi-gpu 0,1 logs/multi_gpu_run.log [Multi-GPU, with log file]

CONFIG_FILE=$1
NUM_GPUS=${2:-1}                  # default to 1 if not provided
MODE=${3:-single-gpu}             # default to 'single-gpu' if not provided
CUDA_DEVICES=${4:-""}             # optional: GPU IDs, e.g., "0" or "0,1"
LOG_FILE=${5:-""}                 # optional: log file path

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist."
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES if specified
if [ -n "$CUDA_DEVICES" ]; then
    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
fi

# Prepare the command
if [ "$MODE" == "multi-gpu" ] && [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS train.py --config \"$CONFIG_FILE\""
    echo "Launching multi-GPU with torchrun on $NUM_GPUS GPUs"
else
    CMD="python3 train.py --config \"$CONFIG_FILE\""
    echo "Launching single-GPU (or default mode)"
fi

# Add log redirection if LOG_FILE is specified
if [ -n "$LOG_FILE" ]; then
    echo "Logging output to $LOG_FILE"
    CMD="$CMD > \"$LOG_FILE\" 2>&1 &"
else
    echo "No log file specified; running in foreground"
fi

# Execute the command
eval $CMD

