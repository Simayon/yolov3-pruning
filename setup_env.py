#!/usr/bin/env python3
import os
import subprocess
import argparse
import shutil
import sys
from pathlib import Path

def get_size(start_path):
    """Get the size of a directory and its contents"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):  # Skip if it's a symbolic link
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes):
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def install_pytorch(install_path):
    """Install PyTorch separately using pip"""
    print("\nInstalling PyTorch and torchvision...")
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except:
        # Try to detect NVIDIA GPU
        try:
            subprocess.run(['nvidia-smi'], check=True, capture_output=True)
            cuda_available = True
        except:
            cuda_available = False
    
    # PyTorch installation command
    if cuda_available:
        pytorch_cmd = "torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:
        pytorch_cmd = "torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    
    try:
        subprocess.run([
            'pip', 'install', 
            '--target', str(install_path),
            '--no-cache-dir'
        ] + pytorch_cmd.split(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        return False

def setup_custom_environment(install_path, requirements_file):
    """Setup a custom Python environment and install packages"""
    install_path = Path(install_path).absolute()
    
    # Create directories
    install_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Setting up custom package location at: {install_path}")
    
    # First install PyTorch
    if not install_pytorch(install_path):
        print("Warning: PyTorch installation failed. Continuing with other packages...")
    
    print("\n2. Installing other packages...")
    try:
        subprocess.run([
            'pip', 'install',
            '-r', requirements_file,
            '--target', str(install_path),
            '--no-cache-dir'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing other packages: {e}")
        return False
    
    print("\n3. Analyzing installed packages...")
    total_size = get_size(install_path)
    print(f"\nTotal installation size: {format_size(total_size)}")
    
    # Analyze individual package sizes
    print("\nPackage sizes:")
    print("-" * 50)
    print(f"{'Package':<30} {'Size':<15}")
    print("-" * 50)
    
    for item in sorted(install_path.glob('*')):
        if item.is_dir() and not item.name.startswith('.'):
            size = get_size(item)
            print(f"{item.name:<30} {format_size(size):<15}")
    
    return True

def cleanup_environment(install_path):
    """Remove the custom environment"""
    try:
        shutil.rmtree(install_path)
        print(f"\nSuccessfully removed environment at: {install_path}")
    except Exception as e:
        print(f"Error cleaning up environment: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup custom Python package environment')
    parser.add_argument('--install-path', type=str, required=True,
                      help='Custom installation path for packages')
    parser.add_argument('--requirements', type=str, default='requirements.txt',
                      help='Path to requirements.txt file')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up existing installation')
    parser.add_argument('--skip-pytorch', action='store_true',
                      help='Skip PyTorch installation')
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_environment(args.install_path)
    
    if not args.skip_pytorch:
        success = setup_custom_environment(args.install_path, args.requirements)
    else:
        print("Skipping PyTorch installation...")
        success = True
    
    if success:
        print(f"""
Environment setup complete!

To use this environment:
1. Add to PYTHONPATH:
   export PYTHONPATH={args.install_path}:$PYTHONPATH

2. Add to PATH:
   export PATH={args.install_path}/bin:$PATH
""")