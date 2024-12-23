#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import time
from datetime import datetime
import platform
import colorama  # For Windows color support

# Initialize colorama for Windows
colorama.init()

class Colors:
    """ANSI color codes"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

class YOLOPipeline:
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.base_dir = Path(args.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.is_windows = platform.system() == 'Windows'

    def setup_logging(self):
        """Setup logging with both file and console output"""
        log_file = f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def colored_log(self, color, level, message):
        """Log with color"""
        colored_message = f"{color}{message}{Colors.RESET}"
        if level == 'ERROR':
            self.logger.error(colored_message)
        elif level == 'WARNING':
            self.logger.warning(colored_message)
        else:
            self.logger.info(colored_message)

    def run_command(self, command, desc=None):
        """Run a command and handle its output"""
        if desc:
            self.logger.info(f"\n{'='*50}\n{desc}\n{'='*50}")
        
        self.logger.info(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=self.is_windows  # Use shell on Windows
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
            
            if returncode != 0:
                _, stderr = process.communicate()
                self.colored_log(Colors.RED, 'ERROR', f"Command failed with error:\n{stderr}")
                if not self.args.continue_on_error:
                    sys.exit(1)
                return False
            return True
            
        except Exception as e:
            self.colored_log(Colors.RED, 'ERROR', f"Error executing command: {e}")
            if not self.args.continue_on_error:
                sys.exit(1)
            return False

    def run_pipeline(self):
        """Run the complete pipeline"""
        start_time = time.time()
        self.logger.info("Starting YOLOv3 optimization pipeline")

        # Step 1: Download weights
        if not Path('weights/yolov3.weights').exists():
            self.run_command(
                ['python', 'download_weights.py'],
                "Downloading YOLOv3 weights"
            )
        else:
            self.colored_log(Colors.YELLOW, 'WARNING', "Weights already exist, skipping download...")

        # Step 2: Download and prepare dataset
        if not Path('datasets/coco').exists():
            self.run_command(
                ['python', 'download_dataset.py'],
                "Downloading and preparing COCO dataset"
            )
        else:
            self.colored_log(Colors.YELLOW, 'WARNING', "Dataset exists, skipping download...")

        # Create dataset files
        self.run_command(
            ['python', 'create_dataset_files.py'],
            "Creating dataset files"
        )

        # Step 3: Fine-tune model
        finetune_dir = self.base_dir / 'finetune'
        finetune_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_command([
            'python', 'finetune.py',
            '--weights', 'weights/yolov3.weights',
            '--data', 'data/coco.yaml',
            '--epochs', str(self.args.epochs),
            '--batch-size', str(self.args.batch_size),
            '--save-dir', str(finetune_dir)
        ], "Fine-tuning YOLOv3 for person detection")

        # Step 4: Prune model
        prune_dir = self.base_dir / 'prune'
        prune_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_command([
            'python', 'prune.py',
            '--weights', str(finetune_dir / 'best.pt'),
            '--data', 'data/coco.yaml',
            '--save-dir', str(prune_dir),
            '--initial-ratio', str(self.args.initial_ratio),
            '--max-ratio', str(self.args.max_ratio),
            '--steps', str(self.args.prune_steps)
        ], "Pruning the fine-tuned model")

        # Step 5: Analyze results
        self.analyze_results()

        total_duration = time.time() - start_time
        self.colored_log(Colors.GREEN, 'INFO', f"\nPipeline completed in {total_duration:.2f} seconds")

    def analyze_results(self):
        """Analyze and display the results"""
        self.logger.info("\nFinal Results Summary:")
        
        # Calculate model sizes
        original_size = Path('weights/yolov3.weights').stat().st_size / (1024 * 1024)
        finetuned_size = (self.base_dir / 'finetune/best.pt').stat().st_size / (1024 * 1024)
        pruned_size = (self.base_dir / 'prune/best.pt').stat().st_size / (1024 * 1024)
        
        self.logger.info("\nModel Size Comparison:")
        self.logger.info(f"Original Model:   {original_size:.2f} MB")
        self.logger.info(f"Fine-tuned Model: {finetuned_size:.2f} MB")
        self.logger.info(f"Pruned Model:     {pruned_size:.2f} MB")
        self.logger.info(f"Size Reduction:   {((original_size - pruned_size) / original_size * 100):.2f}%")

def main():
    parser = argparse.ArgumentParser(description='YOLOv3 Optimization Pipeline')
    parser.add_argument('--output-dir', type=str, default='./runs',
                        help='Directory to save all outputs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--initial-ratio', type=float, default=0.2,
                        help='Initial pruning ratio')
    parser.add_argument('--max-ratio', type=float, default=0.8,
                        help='Maximum pruning ratio')
    parser.add_argument('--prune-steps', type=int, default=4,
                        help='Number of pruning steps')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')
    
    args = parser.parse_args()
    
    pipeline = YOLOPipeline(args)
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()
