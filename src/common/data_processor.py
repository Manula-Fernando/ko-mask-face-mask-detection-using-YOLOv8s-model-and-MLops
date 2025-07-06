"""
Data Processing Pipeline for Face Mask Detection
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from tqdm import tqdm

from .config import TrainingConfig

class DataProcessor:
    """Professional data processing pipeline for face mask detection"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.class_mapping = {cls: idx for idx, cls in enumerate(config.classes)}
        
    def setup_data_directories(self):
        """Setup data directories for YOLO format"""
        base_path = Path(self.config.processed_data_path)
        
        directories = [
            base_path / "images" / "train",
            base_path / "images" / "val", 
            base_path / "labels" / "train",
            base_path / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info("✅ Data directories setup complete")
        return directories
    
    def convert_xml_to_yolo(self, xml_file: Path, images_dir: Path, output_dir: Path):
        """Convert XML annotations to YOLO format with error handling"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image information
            filename_element = root.find('filename')
            if filename_element is None:
                self.logger.warning(f"No filename found in {xml_file}")
                return False
                
            image_filename = filename_element.text
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return False
                
            # Load image to get dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Process annotations
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in self.class_mapping:
                    self.logger.warning(f"Unknown class: {class_name}")
                    continue
                    
                class_id = self.class_mapping[class_name]
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save annotations
            label_filename = image_path.stem + ".txt"
            label_path = output_dir / "labels" / label_filename
            
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
            
            # Copy image
            output_image_path = output_dir / "images" / image_filename
            shutil.copy2(image_path, output_image_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {xml_file}: {e}")
            return False
    
    def process_dataset(self, annotations_dir: str, images_dir: str, train_split: float = 0.8):
        """Process complete dataset with train/val split"""
        self.logger.info("🚀 Starting dataset processing...")
        
        # Setup directories
        self.setup_data_directories()
        
        annotations_path = Path(annotations_dir)
        images_path = Path(images_dir)
        base_output_path = Path(self.config.processed_data_path)
        
        # Get all XML files
        xml_files = list(annotations_path.glob("*.xml"))
        self.logger.info(f"Found {len(xml_files)} annotation files")
        
        if not xml_files:
            self.logger.warning("No XML annotation files found!")
            return {"train_processed": 0, "val_processed": 0, "train_total": 0, "val_total": 0}
        
        # Shuffle and split
        np.random.shuffle(xml_files)
        split_idx = int(len(xml_files) * train_split)
        train_files = xml_files[:split_idx]
        val_files = xml_files[split_idx:]
        
        # Process training set
        self.logger.info(f"Processing {len(train_files)} training files...")
        train_success = 0
        for xml_file in tqdm(train_files, desc="Training set"):
            if self.convert_xml_to_yolo(xml_file, images_path, base_output_path / "train"):
                train_success += 1
        
        # Process validation set
        self.logger.info(f"Processing {len(val_files)} validation files...")
        val_success = 0
        for xml_file in tqdm(val_files, desc="Validation set"):
            if self.convert_xml_to_yolo(xml_file, images_path, base_output_path / "val"):
                val_success += 1
        
        self.logger.info(f"✅ Dataset processing complete!")
        self.logger.info(f"📊 Training: {train_success}/{len(train_files)} files processed")
        self.logger.info(f"📊 Validation: {val_success}/{len(val_files)} files processed")
        
        return {
            "train_processed": train_success,
            "val_processed": val_success,
            "train_total": len(train_files),
            "val_total": len(val_files)
        }
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        import yaml
        
        yaml_content = {
            "path": str(Path(self.config.processed_data_path).absolute()),
            "train": "train/images",
            "val": "val/images", 
            "names": {i: cls for i, cls in enumerate(self.config.classes)}
        }
        
        yaml_path = Path(self.config.dataset_yaml)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        self.logger.info(f"✅ Dataset YAML created: {yaml_path}")
        return yaml_path
