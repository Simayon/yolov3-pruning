#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import time
from datetime import datetime

class YOLOPipeline:
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.base_dir = Path(args.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_command(self, command, desc=None):
        if desc:
            self.logger.info(f"\n{'='*50}\n{desc}\n{'='*50}")
        
        self.logger.info(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Real-time output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    self.logger.info(output.strip())

            returncode = process.poll()
            
            # Get any remaining output
            _, stderr = process.communicate()
            
            if returncode != 0:
                self.logger.error(f"Command failed with error:\n{stderr}")
                if not self.args.continue_on_error:
                    self.logger.error("Stopping pipeline due to error.")
                    sys.exit(1)
                return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            if not self.args.continue_on_error:
                sys.exit(1)
            return False

    def run_pipeline(self):
        start_time = time.time()
        self.logger.info("Starting YOLOv3 optimization pipeline")

        steps = [
            (self.download_weights, "Downloading YOLOv3 weights"),
            (self.download_dataset, "Downloading and preparing person dataset"),
            (self.finetune_model, "Fine-tuning model for person detection"),
            (self.prune_model, "Pruning the fine-tuned model"),
            (self.evaluate_results, "Evaluating final results")
        ]

        for step_func, desc in steps:
            step_start = time.time()
            self.logger.info(f"\nStarting: {desc}")
            
            success = step_func()
            
            if not success and not self.args.continue_on_error:
                self.logger.error(f"Pipeline failed at: {desc}")
                break
                
            step_duration = time.time() - step_start
            self.logger.info(f"Completed: {desc} (Duration: {step_duration:.2f}s)")

        total_duration = time.time() - start_time
        self.logger.info(f"\nPipeline completed in {total_duration:.2f} seconds")

    def download_weights(self):
        return self.run_command(
            ['python', 'download_weights.py'],
            "Downloading YOLOv3 weights"
        )

    def download_dataset(self):
        return self.run_command(
            ['python', 'download_dataset.py'],
            "Downloading and preparing COCO person dataset"
        )

    def finetune_model(self):
        return self.run_command([
            'python', 'finetune.py',
            '--weights', 'weights/yolov3.weights',
            '--data', 'data/coco.yaml',
            '--epochs', str(self.args.epochs),
            '--batch-size', str(self.args.batch_size),
            '--save-dir', str(self.base_dir / 'finetune')
        ], "Fine-tuning YOLOv3 for person detection")

    def prune_model(self):
        return self.run_command([
            'python', 'prune.py',
            '--weights', str(self.base_dir / 'finetune/best.pt'),
            '--data', 'data/coco.yaml',
            '--save-dir', str(self.base_dir / 'prune'),
            '--initial-ratio', str(self.args.initial_ratio),
            '--max-ratio', str(self.args.max_ratio),
            '--steps', str(self.args.prune_steps)
        ], "Pruning the fine-tuned model")

    def evaluate_results(self):
        """Analyze and display the results"""
        self.logger.info("\nFinal Results Summary:")
        
        # Log model sizes
        original_size = os.path.getsize('weights/yolov3.weights') / (1024 * 1024)
        finetuned_size = os.path.getsize(self.base_dir / 'finetune/best.pt') / (1024 * 1024)
        pruned_size = os.path.getsize(self.base_dir / 'prune/best.pt') / (1024 * 1024)
        
        self.logger.info(f"\nModel Sizes:")
        self.logger.info(f"Original Model: {original_size:.2f} MB")
        self.logger.info(f"Fine-tuned Model: {finetuned_size:.2f} MB")
        self.logger.info(f"Pruned Model: {pruned_size:.2f} MB")
        self.logger.info(f"Size Reduction: {((original_size - pruned_size) / original_size * 100):.2f}%")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='YOLOv3 Optimization Pipeline')
    parser.add_argument('--output-dir', type=str, default='./runs',
                        help='Directory to save all outputs')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--initial-ratio', type=float, default=0.1,
                        help='Initial pruning ratio')
    parser.add_argument('--max-ratio', type=float, default=0.9,
                        help='Maximum pruning ratio')
    parser.add_argument('--prune-steps', type=int, default=5,
                        help='Number of pruning steps')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')
    
    args = parser.parse_args()
    
    pipeline = YOLOPipeline(args)
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()
