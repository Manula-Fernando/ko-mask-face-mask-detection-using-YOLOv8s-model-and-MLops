"""
Data processing module for training pipeline
Handles data loading, preprocessing, and augmentation for YOLO-based face mask detection.
"""

import os
import sys
from pathlib import Path
import random
import shutil

# --- Add this block at the top, before any src imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# ---------------------------------------------------------

import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# from src.common.logger import get_logger
from src.common.utils import FileUtils, ImageUtils

# --- CONFIGURATION ---
DATA_DIR = Path("data")
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
RAW_ANN_DIR = DATA_DIR / "raw" / "annotations"
YOLO_DIR = DATA_DIR / "processed" / "yolo_dataset"
SPLITS = ['train', 'val', 'test']
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
CLASS_MAPPING = {name: i for i, name in enumerate(CLASS_NAMES)}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

def create_yolo_structure():
    for split in SPLITS:
        (YOLO_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def parse_xml_annotation(xml_path: Path) -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASS_MAPPING:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        center_x = (xmin + xmax) / 2.0 / img_width
        center_y = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        annotations.append({
            'class_id': CLASS_MAPPING[class_name],
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        })
    return annotations

def collect_valid_pairs() -> List[Tuple[Path, Path, List[Dict]]]:
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in RAW_IMAGES_DIR.iterdir() if f.suffix.lower() in image_extensions]
    valid_pairs = []
    for img_path in image_files:
        xml_path = RAW_ANN_DIR / (img_path.stem + '.xml')
        if xml_path.exists():
            try:
                annotations = parse_xml_annotation(xml_path)
                if annotations:
                    valid_pairs.append((img_path, xml_path, annotations))
            except Exception as e:
                print(f"Error parsing {xml_path}: {e}")
    return valid_pairs

def split_dataset(valid_pairs: List[Tuple[Path, Path, List[Dict]]]):
    random.shuffle(valid_pairs)
    train_data, temp_data = train_test_split(valid_pairs, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42)
    return {'train': train_data, 'val': val_data, 'test': test_data}

def convert_to_yolo_format(splits: Dict[str, List[Tuple[Path, Path, List[Dict]]]]):
    for split_name, split_data in splits.items():
        split_images_dir = YOLO_DIR / split_name / 'images'
        split_labels_dir = YOLO_DIR / split_name / 'labels'
        for img_path, xml_path, annotations in split_data:
            # Copy image
            dst_img_path = split_images_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)
            # Write YOLO label
            label_name = img_path.stem + '.txt'
            dst_label_path = split_labels_dir / label_name
            with open(dst_label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['center_x']:.6f} {ann['center_y']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")

def create_dataset_yaml():
    yaml_content = {
        'path': str(YOLO_DIR.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Created dataset.yaml at {yaml_path}")

def main():
    print("🚀 Starting YOLO dataset preparation...")
    create_yolo_structure()
    valid_pairs = collect_valid_pairs()
    print(f"Found {len(valid_pairs)} valid image-annotation pairs.")
    if not valid_pairs:
        print("❌ No valid pairs found. Please check your data.")
        return
    splits = split_dataset(valid_pairs)
    for split in SPLITS:
        print(f"{split}: {len(splits[split])} samples")
    convert_to_yolo_format(splits)
    create_dataset_yaml()
    print("✅ YOLO dataset preparation complete!")
    print(f"YOLO dataset directory: {YOLO_DIR}")

class DataProcessor:
    """
    DataProcessor class for handling data processing tasks.
    """
    @staticmethod
    def prepare_yolo_dataset():
        main()

class YOLODataset:
    """
    Placeholder YOLODataset class for import compatibility.
    Implement dataset logic as needed.
    """
    def __init__(self, *args, **kwargs):
        # Example attribute for test compatibility
        self.image_files = []

if __name__ == "__main__":
    main()
    
