# YOLOv3 Person Detection Optimization Pipeline

This project provides a comprehensive pipeline for optimizing YOLOv3 specifically for person detection through fine-tuning and model pruning. The pipeline includes downloading weights, preparing a person-specific dataset, fine-tuning, pruning, and performance evaluation.

## Features

- 🎯 Specialized person detection optimization
- 📊 Progressive model pruning with performance tracking
- 📈 Comprehensive performance visualization
- 🔄 Automated pipeline execution
- 📝 Detailed logging and reporting

## Project Structure

```
yolov3-pruning/
├── data/
│   └── coco.yaml           # Dataset configuration
├── utils/
│   └── visualization.py    # Visualization utilities
├── download_weights.py     # YOLOv3 weights downloader
├── download_dataset.py     # Person dataset preparation
├── finetune.py            # Model fine-tuning
├── prune.py               # Model pruning
└── run_pipeline.py        # Main pipeline script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yolov3-pruning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the entire pipeline with default settings:
```bash
python run_pipeline.py
```

### Advanced Usage

Run with custom parameters:
```bash
python run_pipeline.py \
    --output-dir ./runs \
    --epochs 20 \
    --batch-size 32 \
    --initial-ratio 0.2 \
    --max-ratio 0.8 \
    --prune-steps 4
```

### Pipeline Steps

1. **Weight Download**
   - Downloads official YOLOv3 weights
   - Saves to `weights/yolov3.weights`

2. **Dataset Preparation**
   - Downloads COCO dataset
   - Filters for person-only annotations
   - Creates optimized dataset structure

3. **Fine-tuning**
   - Specializes the model for person detection
   - Saves best model to `runs/finetune/best.pt`

4. **Pruning**
   - Progressively reduces model size
   - Maintains person detection performance
   - Generates comparison metrics

5. **Evaluation**
   - Compares model sizes
   - Analyzes performance metrics
   - Generates visualization reports

## Configuration

### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output-dir` | Output directory | `./runs` |
| `--epochs` | Fine-tuning epochs | 10 |
| `--batch-size` | Training batch size | 16 |
| `--initial-ratio` | Initial pruning ratio | 0.1 |
| `--max-ratio` | Maximum pruning ratio | 0.9 |
| `--prune-steps` | Number of pruning steps | 5 |
| `--continue-on-error` | Continue if step fails | False |

### Dataset Configuration

Edit `data/coco.yaml` to modify:
- Dataset paths
- Training/validation split
- Class configurations

## Output Structure

```
runs/
├── finetune/
│   ├── best.pt           # Best fine-tuned model
│   └── finetune.log      # Fine-tuning logs
├── prune/
│   ├── best.pt          # Best pruned model
│   ├── pruning.log      # Pruning logs
│   └── results/
│       ├── metrics.csv   # Performance metrics
│       └── plots/        # Visualization plots
└── pipeline.log         # Complete pipeline log
```

## Results Analysis

The pipeline generates:
- Performance comparison tables
- Model size reduction metrics
- mAP and precision/recall curves
- Visual comparisons of model performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
