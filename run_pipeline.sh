#!/bin/bash

# Default values
OUTPUT_DIR="./runs"
EPOCHS=20
BATCH_SIZE=32
INITIAL_RATIO=0.2
MAX_RATIO=0.8
PRUNE_STEPS=4
CONTINUE_ON_ERROR=false

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging setup
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    if [ "$CONTINUE_ON_ERROR" = false ]; then
        exit 1
    fi
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --initial-ratio)
            INITIAL_RATIO="$2"
            shift 2
            ;;
        --max-ratio)
            MAX_RATIO="$2"
            shift 2
            ;;
        --prune-steps)
            PRUNE_STEPS="$2"
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        *)
            error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        error "$1 failed"
        return 1
    fi
    success "$1 completed successfully"
    return 0
}

# Start pipeline
log "Starting YOLOv3 optimization pipeline"
log "Output directory: $OUTPUT_DIR"
log "Configuration:"
log "  - Epochs: $EPOCHS"
log "  - Batch Size: $BATCH_SIZE"
log "  - Initial Pruning Ratio: $INITIAL_RATIO"
log "  - Maximum Pruning Ratio: $MAX_RATIO"
log "  - Pruning Steps: $PRUNE_STEPS"

# Step 1: Download weights
log "\nStep 1: Downloading YOLOv3 weights"
if [ ! -f "weights/yolov3.weights" ]; then
    python3 download_weights.py
    check_status "Weight download" || exit 1
else
    log "Weights already downloaded, skipping..."
fi

# Step 2: Download and prepare dataset
log "\nStep 2: Downloading and preparing dataset"
if [ ! -d "datasets/coco" ]; then
    python3 download_dataset.py
    check_status "Dataset download" || exit 1
else
    log "Dataset directory exists, skipping download..."
fi

# Create dataset files
log "\nCreating dataset files"
python3 create_dataset_files.py
check_status "Dataset file creation" || exit 1

# Step 3: Fine-tune model
log "\nStep 3: Fine-tuning model for person detection"
FINETUNE_DIR="$OUTPUT_DIR/finetune"
mkdir -p "$FINETUNE_DIR"

python3 finetune.py \
    --weights weights/yolov3.weights \
    --data data/coco.yaml \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --save-dir "$FINETUNE_DIR"
check_status "Model fine-tuning" || exit 1

# Step 4: Prune model
log "\nStep 4: Pruning the fine-tuned model"
PRUNE_DIR="$OUTPUT_DIR/prune"
mkdir -p "$PRUNE_DIR"

python3 prune.py \
    --weights "$FINETUNE_DIR/best.pt" \
    --data data/coco.yaml \
    --save-dir "$PRUNE_DIR" \
    --initial-ratio "$INITIAL_RATIO" \
    --max-ratio "$MAX_RATIO" \
    --steps "$PRUNE_STEPS"
check_status "Model pruning" || exit 1

# Step 5: Analyze results
log "\nStep 5: Analyzing results"

# Calculate model sizes
ORIGINAL_SIZE=$(ls -l weights/yolov3.weights | awk '{print $5}')
FINETUNED_SIZE=$(ls -l "$FINETUNE_DIR/best.pt" | awk '{print $5}')
PRUNED_SIZE=$(ls -l "$PRUNE_DIR/best.pt" | awk '{print $5}')

log "\nModel Size Comparison:"
log "Original Model:   $(echo "scale=2; $ORIGINAL_SIZE/1048576" | bc) MB"
log "Fine-tuned Model: $(echo "scale=2; $FINETUNED_SIZE/1048576" | bc) MB"
log "Pruned Model:     $(echo "scale=2; $PRUNED_SIZE/1048576" | bc) MB"
log "Size Reduction:   $(echo "scale=2; ($ORIGINAL_SIZE-$PRUNED_SIZE)/$ORIGINAL_SIZE*100" | bc)%"

success "\nPipeline completed successfully!"
log "Results saved in: $OUTPUT_DIR"
log "Log file: $LOG_FILE"
