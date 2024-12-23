import torch
import argparse
from models.darknet import Darknet

def convert_weights(weights_path, cfg_path, output_path=None):
    """Convert Darknet weights to PyTorch format"""
    # Initialize model
    model = Darknet(cfg_path)
    
    # Load weights
    if weights_path.endswith('.weights'):
        # Load from darknet weights
        model.load_darknet_weights(weights_path)
        
        # Create checkpoint
        chkpt = {
            'epoch': -1,
            'best_fitness': None,
            'training_results': None,
            'model': model.state_dict(),
            'optimizer': None
        }
        
        # Save PyTorch checkpoint
        if output_path is None:
            output_path = weights_path.rsplit('.', 1)[0] + '.pt'
        
        torch.save(chkpt, output_path)
        print(f"Success: converted '{weights_path}' to '{output_path}'")
        return output_path
    else:
        print('Error: weights file must end with .weights')
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO weights from Darknet to PyTorch format')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights',
                      help='path to weights file')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg',
                      help='path to cfg file')
    parser.add_argument('--output', type=str, default=None,
                      help='output path for converted weights (default: same as input with .pt extension)')
    
    args = parser.parse_args()
    
    convert_weights(args.weights, args.cfg, args.output)

if __name__ == '__main__':
    main()
