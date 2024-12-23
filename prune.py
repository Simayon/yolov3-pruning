import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import time
from pathlib import Path
import logging
from models.yolo import Model
from utils.metrics import bbox_iou, compute_ap
from utils.datasets import LoadImagesAndLabels

class ModelPruner:
    def __init__(self, model_path, data_yaml, save_dir='./runs/prune'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.data_config = self.load_data_config(data_yaml)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_df = pd.DataFrame(columns=['pruning_ratio', 'mAP', 'precision', 'recall', 'model_size_mb'])
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'pruning.log'),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def load_model(model_path):
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict):
            model = model['model']
        return model

    @staticmethod
    def load_data_config(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        return metrics

    def prune_model(self, model, pruning_ratio):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Calculate L1-norm for each filter
                importance = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
                num_filters = int(module.out_channels * (1 - pruning_ratio))
                threshold = torch.sort(importance)[0][num_filters]
                
                # Create binary mask
                mask = importance > threshold
                module.weight.data *= mask.view(-1, 1, 1, 1)

        return model

    def iterative_pruning(self, initial_ratio=0.1, max_ratio=0.9, steps=5):
        current_model = self.model
        ratios = np.linspace(initial_ratio, max_ratio, steps)
        
        for step, ratio in enumerate(ratios):
            logging.info(f"\nStep {step + 1}/{steps} - Pruning ratio: {ratio:.2f}")
            
            # Prune the model
            pruned_model = self.prune_model(current_model.copy(), ratio)
            
            # Evaluate and save results
            metrics = self.evaluate_model(pruned_model, self.val_dataloader)
            model_size = self.get_model_size(pruned_model)
            
            # Save results
            self.save_results(ratio, metrics, model_size)
            self.save_model(pruned_model, f"pruned_model_{ratio:.2f}")
            
            current_model = pruned_model

        self.save_comparison_table()

    def save_results(self, ratio, metrics, model_size):
        new_row = {
            'pruning_ratio': ratio,
            'mAP': metrics['mAP'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'model_size_mb': model_size
        }
        self.results_df = self.results_df.append(new_row, ignore_index=True)

    def save_comparison_table(self):
        # Save to CSV
        self.results_df.to_csv(self.save_dir / 'pruning_results.csv', index=False)
        
        # Create a formatted markdown table
        with open(self.save_dir / 'pruning_results.md', 'w') as f:
            f.write("# YOLOv3 Pruning Results\n\n")
            f.write(self.results_df.to_markdown(index=False))

    @staticmethod
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pt")
        size_mb = Path("temp.pt").stat().st_size / (1024 * 1024)
        Path("temp.pt").unlink()
        return size_mb

    def save_model(self, model, name):
        torch.save({
            'model': model.state_dict(),
            'epoch': -1,
            'optimizer': None,
        }, self.save_dir / f"{name}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to initial weights file')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--save-dir', type=str, default='./runs/prune', help='Directory to save results')
    parser.add_argument('--initial-ratio', type=float, default=0.1, help='Initial pruning ratio')
    parser.add_argument('--max-ratio', type=float, default=0.9, help='Maximum pruning ratio')
    parser.add_argument('--steps', type=int, default=5, help='Number of pruning steps')
    args = parser.parse_args()

    pruner = ModelPruner(args.weights, args.data, args.save_dir)
    pruner.iterative_pruning(args.initial_ratio, args.max_ratio, args.steps)
