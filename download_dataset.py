import os
import subprocess
import json
from pathlib import Path

def download_file(url, output_path):
    try:
        subprocess.run(['wget', '-O', output_path, url], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_file(file_path):
    try:
        subprocess.run(['unzip', file_path], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error extracting {file_path}: {e}")
        return False

def filter_person_annotations(ann_file, output_file):
    """Filter COCO annotations to keep only person class (category_id=1)"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Filter categories
    data['categories'] = [cat for cat in data['categories'] if cat['id'] == 1]
    
    # Filter annotations
    data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] == 1]
    
    # Get image IDs with person annotations
    image_ids = set(ann['image_id'] for ann in data['annotations'])
    
    # Filter images
    data['images'] = [img for img in data['images'] if img['id'] in image_ids]
    
    # Save filtered annotations
    with open(output_file, 'w') as f:
        json.dump(data, f)

def main():
    # Create dataset directory
    dataset_dir = Path('datasets/coco')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for COCO dataset
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'train_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    print("Downloading COCO dataset (person class only)...")
    
    # Download and extract training images
    print("\nDownloading training images...")
    if download_file(urls['train_images'], 'datasets/coco/train2017.zip'):
        print("Extracting training images...")
        extract_file('datasets/coco/train2017.zip')
    
    # Download and extract validation images
    print("\nDownloading validation images...")
    if download_file(urls['val_images'], 'datasets/coco/val2017.zip'):
        print("Extracting validation images...")
        extract_file('datasets/coco/val2017.zip')
    
    # Download and extract annotations
    print("\nDownloading annotations...")
    if download_file(urls['train_annotations'], 'datasets/coco/annotations.zip'):
        print("Extracting annotations...")
        extract_file('datasets/coco/annotations.zip')
        
        # Filter annotations to keep only person class
        print("\nFiltering annotations for person class...")
        filter_person_annotations(
            'annotations/instances_train2017.json',
            'annotations/instances_train2017_person.json'
        )
        filter_person_annotations(
            'annotations/instances_val2017.json',
            'annotations/instances_val2017_person.json'
        )
    
    print("\nDataset download and preparation complete!")
    print("The dataset is filtered to contain only person-class annotations.")

if __name__ == '__main__':
    main()
