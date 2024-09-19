#!/bin/bash

# 기본 변수 설정
MODEL_PATH="/path/to/your/checkpoint/directory" 
DATA_PATH="/path/to/your/data/directory"
BATCH_SIZE=128 
TOP_K=5

# Python 스크립트 실행
python -m tools.eval \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --top_k $TOP_K \
    --use_dp