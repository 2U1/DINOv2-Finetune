#!/bin/bash

# 기본 변수 설정
MODEL_PATH="/home/workspace/dinov2-ft/output/0906_CBCELoss/best_checkpoint-79886" 
DATA_PATH="/home/workspace/reid/test"
BATCH_SIZE=1024 
TOP_K=5

# Python 스크립트 실행
python -m tools.eval \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --top_k $TOP_K \
    --use_dp