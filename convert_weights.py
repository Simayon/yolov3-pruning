import torch
from models.darknet import Darknet
import argparse
import os

def convert_weights(weights_path, cfg_path, output_path):
    """Convert Darknet weights to PyTorch format"""
    # Create model
    model = Darknet(cfg_path)
    
    # Load weights
    model.load_darknet_weights(weights_path)
    
    # Save PyTorch model
    torch.save(model.state_dict(), output_path)
    print(f'Successfully converted weights to PyTorch format. Saved to {output_path}')

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO weights from Darknet to PyTorch format')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights',
                      help='path to weights file')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg',
                      help='path to cfg file')
    parser.add_argument('--output', type=str, default='weights/yolov3.pt',
                      help='output path for converted weights')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert weights
    convert_weights(args.weights, args.cfg, args.output)

if __name__ == '__main__':
    main()
