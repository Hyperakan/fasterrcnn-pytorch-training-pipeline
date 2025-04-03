#!/bin/bash

# Script to continue training a Faster R-CNN model from a checkpoint
# Usage: ./continue_training.sh [checkpoint_file] [data_config] [model_name] [additional_epochs]

# Default values
CHECKPOINT=${1:-"last_model.pth"}
DATA_CONFIG=${2:-"data_configs/custom_data.yaml"}
MODEL_NAME=${3:-"fasterrcnn_resnet50_fpn_v2"}
ADDITIONAL_EPOCHS=${4:-15}

# Check if the checkpoint file exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT' not found!"
    echo "Available checkpoint files:"
    ls -1 *.pth 2>/dev/null || echo "No .pth files found in the current directory"
    exit 1
fi

# Run the continue_training.py script
echo "Continuing training from checkpoint: $CHECKPOINT"
echo "Data config: $DATA_CONFIG"
echo "Model: $MODEL_NAME"
echo "Additional epochs: $ADDITIONAL_EPOCHS"
echo ""

python continue_training.py \
    --weights "$CHECKPOINT" \
    --data "$DATA_CONFIG" \
    --model "$MODEL_NAME" \
    --epochs "$ADDITIONAL_EPOCHS" \
    --name "continued_$(basename "$CHECKPOINT" .pth)"

echo "Training completed!" 