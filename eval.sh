#!/bin/bash

# 기본 변수 설정
MODEL_PATH="/home/workspace/dinov2-ft/output/0905/final_model"  # 모델 체크포인트 경로
DATA_PATH="/home/workspace/reid/test"  # 데이터 경로
BATCH_SIZE=1024  # 배치 크기
TOP_K=5  # Top-K 설정

# Python 스크립트 실행
python eval.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --top_k $TOP_K \
    --use_dp