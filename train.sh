#!/bin/bash

# Training script for VideoAnomalyCLIP

# Set device
DEVICE=7

# Training configurations
CONFIG_DIR="configs"
CHECKPOINT_DIR="checkpoints"

# Create directories if they don't exist
mkdir -p ${CHECKPOINT_DIR}
mkdir -p logs

echo "Starting VideoAnomalyCLIP training..."

# Train on ShanghaiTech dataset for general temporal patterns
echo "=== Training on ShanghaiTech Dataset ==="
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --config ${CONFIG_DIR}/shanghaitech_train.yaml \
    --seed 111 \
    --image_size 336 \
    --features_list 6 12 18 24 \
    2>&1 | tee logs/shanghaitech_train.log

if [ $? -eq 0 ]; then
    echo "ShanghaiTech training completed successfully!"
else
    echo "ShanghaiTech training failed!"
    exit 1
fi

echo "All training completed successfully!"