#!/usr/bin/env python3
import os
import subprocess
import argparse
import shutil
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

def setup_custom_environment(install_path, requirements_file):
    """Setup a custom Python environment and install packages"""
    install_path = Path(install_path).absolute()
    
    # Create directories
    install_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Setting up custom package location at: {install_path}")
    
    # Create pip.conf to set default install location
    pip_config = install_path / "pip.conf"
    pip_config.write_text(f"""[global]
target={install_path}
install-option=--prefix={install_path}
""")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(install_path)
    os.environ['PIP_CONFIG_FILE'] = str(pip_config)
    
    print("\n2. Installing packages...")
    try:
        subprocess.run([
            'pip', 'install', 
            '-r', requirements_file,
            '--target', str(install_path),
            '--no-cache-dir'  # Don't use pip cache
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
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
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_environment(args.install_path)
    
    success = setup_custom_environment(args.install_path, args.requirements)
    
    if success:
        print(f"""
Environment setup complete!

To use this environment:
1. Add to PYTHONPATH:
   export PYTHONPATH={args.install_path}:$PYTHONPATH

2. Add to PATH:
   export PATH={args.install_path}/bin:$PATH
""")
