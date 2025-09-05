#!/bin/bash

# Testing script for VideoAnomalyCLIP

# Set device
DEVICE=0

# Configuration directories
CONFIG_DIR="configs"
RESULTS_DIR="results"

# Create results directory
mkdir -p ${RESULTS_DIR}
mkdir -p logs

echo "Starting VideoAnomalyCLIP zero-shot testing..."

# Test on UCSD Ped2 dataset using trained ShanghaiTech model
echo "=== Testing on UCSD Dataset ==="
CUDA_VISIBLE_DEVICES=${DEVICE} python test.py \
    --config ${CONFIG_DIR}/ucsd_test.yaml \
    --seed 111 \
    --image_size 336 \
    --features_list 6 12 18 24 \
    2>&1 | tee logs/ucsd_test.log

if [ $? -eq 0 ]; then
    echo "UCSD testing completed successfully!"
    echo "Results saved in: ${RESULTS_DIR}/ucsd/"
else
    echo "UCSD testing failed!"
    exit 1
fi

# Evaluate results independently
echo "=== Running independent evaluation ==="
python evaluate.py \
    --gt_path /path/to/ucsd/ground_truth \
    --pred_path ${RESULTS_DIR}/ucsd/frame_scores.npy \
    --dataset ucsd \
    --output_dir ${RESULTS_DIR}/ucsd/evaluation \
    --plot_curves \
    2>&1 | tee logs/evaluation.log

echo "All testing and evaluation completed successfully!"
echo "Check results in: ${RESULTS_DIR}/"