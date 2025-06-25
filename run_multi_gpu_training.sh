#!/bin/bash

# GaussMarker 多GPU训练启动脚本
# 使用方法: bash run_multi_gpu_training.sh

echo "=========================================="
echo "GaussMarker 多GPU训练启动脚本"
echo "=========================================="

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 2 ]; then
    echo "警告: GPU数量少于2个，建议使用单GPU训练"
    exit 1
fi

# 设置环境变量优化
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_DEBUG=INFO

echo "开始多GPU训练..."

# 方案1: 使用DataParallel (简单)
echo "方案1: DataParallel训练"
python train_GNR.py \
    --multi_gpu \
    --train_steps 50000 \
    --r 180 \
    --s_min 1.0 \
    --s_max 1.2 \
    --fp 0.35 \
    --neg_p 0.5 \
    --model_nf 128 \
    --batch_size 32 \
    --num_workers 16 \
    --lr 2e-4 \
    -ed multi_gpu_dp \
    --w_info_path w1_256.pth

echo "DataParallel训练完成！"

# 方案2: 使用DistributedDataParallel (高性能)
echo "方案2: DistributedDataParallel训练"
python -m torch.distributed.launch \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_GNR_ddp.py \
    --train_steps 50000 \
    --r 180 \
    --s_min 1.0 \
    --s_max 1.2 \
    --fp 0.35 \
    --neg_p 0.5 \
    --model_nf 128 \
    --batch_size 32 \
    --num_workers 16 \
    --lr 2e-4 \
    -ed multi_gpu_ddp \
    --w_info_path w1_256.pth

echo "DistributedDataParallel训练完成！"
echo "=========================================="
