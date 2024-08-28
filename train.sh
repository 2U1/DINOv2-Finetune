#!/bin/bash

# Set the number of GPUs to use (world_size)
WORLD_SIZE=4  # 예: 4개의 GPU를 사용

# Path to the Python script
SCRIPT_PATH="path/to/your/script.py"  # 여기에 실제 Python 스크립트 경로를 입력

# Run the training script with torchrun
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr="127.0.0.1" \
    --master_port="29500" \
    $SCRIPT_PATH