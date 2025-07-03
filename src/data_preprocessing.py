# Face Mask Detection - Production Data Processor
import os
import zipfile
import logging
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import cv2

class DataProcessor:
    """Production-ready data processing pipeline for face mask detection."""
    
    CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    IMAGE_SIZE = (224, 224)
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.images_dir = raw_data_dir / "images"
        self.annotations_dir = raw_data_dir / "annotations"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def extract_dataset(self) -> bool:
        """Extract dataset from zip archive with validation."""
        archive_path = self.raw_data_dir / "images.zip"
        
        if not archive_path.exists():
            self.logger.error(f"Dataset archive not found: {archive_path}")
            self.logger.info("Download from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data")
            return False
            
        # Check if already extracted
        if self.images_dir.exists() and self.annotations_dir.exists():
            image_count = len(list(self.images_dir.glob("*.jpg")))
            annotation_count = len(list(self.annotations_dir.glob("*.xml")))
            if image_count > 800 and annotation_count > 800:
                self.logger.info(f"Dataset already extracted: {image_count} images, {annotation_count} annotations")
                return True
        
        self.logger.info("Extracting dataset...")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
            self.logger.info("✅ Dataset extracted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate if image can be loaded and is valid."""
        try:
            image = cv2.imread(str(image_path))
            return image is not None and image.size > 0
        except Exception:
            return False
    
    def load_annotations(self) -> pd.DataFrame:
        """Load and parse XML annotations into DataFrame with validation."""
        annotations = []
        
        for xml_file in self.annotations_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                filename = root.find('filename').text
                image_path = self.images_dir / filename
                
                # Validate image exists and is readable
                if not image_path.exists() or not self.validate_image(image_path):
                    continue
                    
                # Extract object annotations
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in self.CLASSES:
                        annotations.append({
                            'filename': filename,
                            'image_path': str(image_path),
                            'class': class_name,
                            'class_id': self.CLASSES.index(class_name)
                        })
                        break  # Take first valid annotation per image
                        
            except Exception as e:
                self.logger.warning(f"Failed to parse {xml_file}: {e}")
                continue
        
        df = pd.DataFrame(annotations)
        self.logger.info(f"Loaded {len(df)} valid annotations")
        
        # Log class distribution
        if len(df) > 0:
            dist = df['class'].value_counts()
            self.logger.info(f"Class distribution: {dict(dist)}")
        
        return df
    
    def create_splits(self, df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits."""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['class']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state,
            stratify=train_val['class']
        )
        
        self.logger.info(f"Dataset splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save split metadata and file lists."""
        splits_dir = self.processed_data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save DataFrames
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'classes': self.CLASSES,
            'train_distribution': train_df['class'].value_counts().to_dict(),
            'val_distribution': val_df['class'].value_counts().to_dict(),
            'test_distribution': test_df['class'].value_counts().to_dict()
        }
        
        with open(splits_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Splits saved to {splits_dir}")
