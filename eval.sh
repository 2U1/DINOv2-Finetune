#!/bin/bash

# 기본 변수 설정
MODEL_PATH="/path/to/checkpoint" 
DATA_PATH="/path/to/data"
BATCH_SIZE=128 
TOP_K=3

# Python 스크립트 실행
python -m tools.eval \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --top_k $TOP_K \
    --use_dp