# YOLOv3 Person Detection Optimization Pipeline

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🎯 Overview

This project provides a comprehensive pipeline for optimizing YOLOv3 specifically for person detection through model pruning and fine-tuning. It combines state-of-the-art techniques in model optimization to create a more efficient person detection model while maintaining high accuracy.

### Key Features

- 🔍 **Specialized Person Detection**: Fine-tuned specifically for human detection tasks
- 📉 **Model Pruning**: Intelligent weight pruning using L1-norm based filter pruning
- 📊 **Performance Tracking**: Comprehensive metrics and visualization tools
- 🚀 **Automated Pipeline**: End-to-end automation from dataset preparation to evaluation
- 📈 **Detailed Analytics**: Performance comparison across different pruning stages

## 🛠️ Technical Details

### Model Architecture
- Base Model: [YOLOv3](https://pjreddie.com/darknet/yolo/) (You Only Look Once version 3)
- Backbone: Darknet-53
- Input Resolution: 416×416 pixels
- Pre-trained on: MS COCO dataset

### Optimization Techniques
1. **Fine-tuning**
   - Person-specific dataset preparation
   - Learning rate scheduling
   - Class-focused optimization

2. **Pruning Strategy**
   - L1-norm based filter pruning
   - Progressive pruning with performance monitoring
   - Adaptive pruning ratios

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Simayon/yolov3-pruning.git
cd yolov3-pruning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Run the complete pipeline with default settings:
```bash
python run_pipeline.py
```

### Advanced Usage

Customize the optimization process:
```bash
python run_pipeline.py \
    --output-dir ./runs \
    --epochs 20 \
    --batch-size 32 \
    --initial-ratio 0.2 \
    --max-ratio 0.8 \
    --prune-steps 4
```

## 📊 Pipeline Components

### 1. Dataset Preparation
- Downloads and processes COCO dataset
- Filters for person class
- Creates optimized training/validation splits

### 2. Model Fine-tuning
- Adapts YOLOv3 for person detection
- Implements custom learning rate scheduling
- Saves best-performing checkpoints

### 3. Progressive Pruning
- Iterative model compression
- Performance-aware pruning
- Maintains detection accuracy

### 4. Evaluation & Analysis
- Comprehensive metrics calculation
- Performance visualization
- Model size comparisons

## 📈 Performance Metrics

The pipeline tracks multiple metrics:
- Mean Average Precision (mAP)
- Precision-Recall curves
- Inference speed (FPS)
- Model size reduction
- Memory usage

## 📁 Project Structure

```
yolov3-pruning/
├── data/
│   └── coco.yaml           # Dataset configuration
├── utils/
│   └── visualization.py    # Visualization tools
├── download_weights.py     # YOLOv3 weights downloader
├── download_dataset.py     # Dataset preparation
├── finetune.py            # Model fine-tuning
├── prune.py               # Model pruning
└── run_pipeline.py        # Main pipeline script
```

## 📊 Results

Example results after optimization:
- Model size reduction: Up to 70%
- Inference speed improvement: Up to 2x
- Minimal accuracy loss: < 1% mAP drop

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

1. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
2. [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
3. [COCO Dataset](https://cocodataset.org/)
4. [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv3](https://pjreddie.com/darknet/yolo/) by Joseph Redmon
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- [COCO dataset](https://cocodataset.org/) team for the training data

## 📧 Contact

For questions and feedback:
- Create an issue in this repository
- Contact the maintainers through GitHub

---
<div align="center">
Made with ❤️ for the Computer Vision community
</div>
