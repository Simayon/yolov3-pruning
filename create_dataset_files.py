#!/usr/bin/env python3
import os
from pathlib import Path
import json

def create_image_list(image_dir, annotation_file, output_file):
    """Create a text file listing all images that have person annotations"""
    # Read annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get all image filenames
    image_files = {img['id']: img['file_name'] for img in data['images']}
    
    # Get image IDs that have person annotations
    image_ids = set(ann['image_id'] for ann in data['annotations'])
    
    # Create image list
    with open(output_file, 'w') as f:
        for img_id in sorted(image_ids):
            img_path = os.path.join(image_dir, image_files[img_id])
            f.write(f'{img_path}\n')

def main():
    dataset_root = Path('datasets/coco')
    
    # Create training image list
    create_image_list(
        'train2017',
        dataset_root / 'annotations/instances_train2017_person.json',
        dataset_root / 'train2017.txt'
    )
    
    # Create validation image list
    create_image_list(
        'val2017',
        dataset_root / 'annotations/instances_val2017_person.json',
        dataset_root / 'val2017.txt'
    )
    
    print("Dataset files created successfully!")

if __name__ == '__main__':
    main()
