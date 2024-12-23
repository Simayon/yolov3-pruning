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
        subprocess.run(['nvidia-smi'], check=True, capture_output=True)
        cuda_available = True
    except:
        cuda_available = False
    
    # PyTorch installation command for Python 3.8
    if cuda_available:
        pytorch_cmd = "torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html"
    else:
        pytorch_cmd = "torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    
    try:
        # Split the command properly
        cmd_parts = pytorch_cmd.split()
        # Separate the -f argument
        index_url = cmd_parts[-1]
        packages = cmd_parts[:-2]  # Exclude the -f and URL
        
        subprocess.run([
            'pip', 'install',
            '--target', str(install_path),
            '--no-cache-dir',
            '-f', index_url,
            *packages
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        return False

def install_package_group(install_path, packages):
    """Install a group of packages"""
    try:
        cmd = [
            'pip', 'install',
            '--target', str(install_path),
            '--no-cache-dir'
        ] + packages.split()
        
        process = subprocess.Popen(
            cmd,
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
        
        returncode = process.poll()
        
        if returncode != 0:
            _, stderr = process.communicate()
            print(f"Warning: Some packages might not have installed correctly:\n{stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False

def setup_custom_environment(install_path, requirements_file, args):
    """Setup a custom Python environment and install packages"""
    install_path = Path(install_path).absolute()
    
    # Create directories
    install_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Setting up custom package location at: {install_path}")
    
    # Read requirements file
    with open(requirements_file) as f:
        requirements = f.read()
    
    # Install base dependencies first
    print("\n2. Installing base dependencies...")
    base_deps = "numpy==1.18.5 protobuf==3.20.2 scipy==1.7.3 networkx==2.6.3"
    if not install_package_group(install_path, base_deps):
        print("Warning: Base dependencies installation had issues. Continuing...")
    
    # Install PyTorch if requested
    if not args.skip_pytorch:
        if not install_pytorch(install_path):
            print("Warning: PyTorch installation failed. Continuing with other packages...")
    
    # Install remaining packages
    print("\n3. Installing remaining packages...")
    try:
        subprocess.run([
            'pip', 'install',
            '-r', requirements_file,
            '--target', str(install_path),
            '--no-cache-dir'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Some packages might not have installed correctly: {e}")
    
    print("\n4. Analyzing installed packages...")
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
        success = setup_custom_environment(args.install_path, args.requirements, args)
    else:
        print("Skipping PyTorch installation...")
        success = setup_custom_environment(args.install_path, args.requirements, args)
    
    if success:
        print(f"""
Environment setup complete!

To use this environment:
1. Add to PYTHONPATH:
   export PYTHONPATH={args.install_path}:$PYTHONPATH

2. Add to PATH:
   export PATH={args.install_path}/bin:$PATH
""")
