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

def check_directory(path):
    """Check if directory exists and is writable"""
    path = Path(path)
    try:
        if path.exists() and not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception as e:
        print(f"Error with directory {path}: {e}")
        return False

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

def install_package_group(install_path, packages, upgrade=True):
    """Install a group of packages"""
    try:
        cmd = [
            'pip', 'install',
            '--target', str(install_path),
            '--no-cache-dir'
        ]
        
        if upgrade:
            cmd.append('--upgrade')
            
        cmd.extend(packages.split())
        
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
            if "ERROR: launchpadlib" in stderr:
                # Ignore launchpadlib error as it's not critical
                return True
            print(f"Warning: Some packages might not have installed correctly:\n{stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False

def setup_custom_environment(install_path, requirements_file, args):
    """Setup a custom Python environment and install packages"""
    install_path = Path(install_path).absolute()
    
    # Check directory permissions
    if not check_directory(install_path):
        return False
    
    print(f"\n1. Setting up custom package location at: {install_path}")
    
    # Install packages in order of dependencies
    installation_order = [
        ("Base ML", "numpy==1.18.5 scipy==1.4.1"),  # Match tensorflow requirement
        ("Image Processing", "Pillow==9.5.0 opencv-python==4.7.0.72"),
        ("Utilities", "PyYAML==6.0.1 tqdm==4.65.0"),
        ("Data Analysis", "pandas==1.3.5 matplotlib==3.5.3"),
        ("Visualization", "seaborn==0.11.2"),
        ("Deep Learning", "protobuf==3.20.2 tensorboard==2.11.2"),
        ("Graph Processing", "networkx==2.6.3")
    ]
    
    success_count = 0
    for package_group, packages in installation_order:
        print(f"\nInstalling {package_group} packages...")
        if install_package_group(install_path, packages, upgrade=True):
            success_count += 1
        else:
            print(f"Warning: Some {package_group} packages might not have installed correctly. Continuing...")
    
    # Install PyTorch if requested
    if not args.skip_pytorch:
        if install_pytorch(install_path):
            success_count += 1
        else:
            print("Warning: PyTorch installation failed. Continuing with other packages...")
    
    print("\nAnalyzing installed packages...")
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
    
    # Return True if at least some packages were installed successfully
    return success_count > 0

def cleanup_environment(install_path):
    """Remove the custom environment"""
    try:
        if Path(install_path).exists():
            shutil.rmtree(install_path)
            print(f"\nSuccessfully removed environment at: {install_path}")
        else:
            print(f"\nNothing to clean up at: {install_path}")
    except Exception as e:
        print(f"Error cleaning up environment: {e}")
        return False
    return True

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
    
    # Handle cleanup first
    if args.cleanup and not cleanup_environment(args.install_path):
        sys.exit(1)
    
    # Setup environment
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
    else:
        print("\nEnvironment setup failed!")
        sys.exit(1)
