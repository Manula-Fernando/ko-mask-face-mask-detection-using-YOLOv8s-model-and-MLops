"""
Data preprocessing module for face mask detection.
This module handles data loading, XML annotation parsing, and data augmentation.
"""

import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import yaml
import logging
import json
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import shutil
from zipfile import ZipFile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing pipeline for face mask detection with XML annotations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data preprocessor with configuration."""
        self.config_path = config_path
        self.load_config()
        self.setup_paths()
        logger.info(f"DataPreprocessor initialized with image size: {self.image_size}")
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.paths_config = self.config['paths']
            self.data_config = self.config['data']
            self.image_size = tuple(self.data_config['image_size'])
            self.batch_size = self.data_config['batch_size']
            self.classes = self.data_config['classes']
            self.class_mapping = self.data_config['class_mapping']
            self.validation_split = self.data_config['validation_split']
            self.test_split = self.data_config['test_split']
            
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using default configuration.")
            self.image_size = (224, 224)
            self.batch_size = 32
            self.classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
            self.class_mapping = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
            self.validation_split = 0.2
            self.test_split = 0.1
            
            # Default paths
            self.paths_config = {
                'raw_data_zip': 'data/raw/images.zip',
                'unzip_dir': 'data/raw/temp_unzip',
                'images_dir': 'data/raw/images',
                'annotations_dir': 'data/raw/annotations',
                'train_dir': 'data/processed/train',
                'val_dir': 'data/processed/val',
                'test_dir': 'data/processed/test'
            }
    
    def setup_paths(self):
        """Set up paths for the current project structure."""
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create all necessary directories (skip files)
        skip_keys = ['raw_data_zip', 'model_name', 'haarcascade']  # Skip file paths
        for key, path in self.paths_config.items():
            if key not in skip_keys:
                full_path = os.path.join(self.base_path, path)
                os.makedirs(full_path, exist_ok=True)
                
        logger.info("Directory structure created successfully")
    
    def extract_dataset(self):
        """Extract the dataset zip file."""
        zip_path = os.path.join(self.base_path, self.paths_config['raw_data_zip'])
        extract_path = os.path.join(self.base_path, self.paths_config['unzip_dir'])
        
        if not os.path.exists(zip_path):
            logger.error(f"Dataset zip file not found at {zip_path}")
            return False
        
        logger.info("Extracting dataset...")
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Move images and annotations to correct locations
            images_src = os.path.join(extract_path, 'images')
            annotations_src = os.path.join(extract_path, 'annotations')
            
            images_dst = os.path.join(self.base_path, self.paths_config['images_dir'])
            annotations_dst = os.path.join(self.base_path, self.paths_config['annotations_dir'])
            
            if os.path.exists(images_src):
                if os.path.exists(images_dst):
                    shutil.rmtree(images_dst)
                shutil.move(images_src, images_dst)
                
            if os.path.exists(annotations_src):
                if os.path.exists(annotations_dst):
                    shutil.rmtree(annotations_dst)
                shutil.move(annotations_src, annotations_dst)
            
            # Clean up temporary directory
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
                
            logger.info("Dataset extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            return False
    
    def parse_xml_annotation(self, xml_path: str) -> Dict:
        """Parse PASCAL VOC XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotation = {
                'filename': root.find('filename').text,
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'objects': []
            }
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                obj_info = {
                    'class': class_name,
                    'xmin': int(bbox.find('xmin').text),
                    'ymin': int(bbox.find('ymin').text),
                    'xmax': int(bbox.find('xmax').text),
                    'ymax': int(bbox.find('ymax').text)
                }
                annotation['objects'].append(obj_info)
            
            return annotation
            
        except Exception as e:
            logger.warning(f"Error parsing XML {xml_path}: {e}")
            return None
    
    def load_annotations(self) -> pd.DataFrame:
        """Load all XML annotations and create a DataFrame."""
        annotations_dir = os.path.join(self.base_path, self.paths_config['annotations_dir'])
        images_dir = os.path.join(self.base_path, self.paths_config['images_dir'])
        
        if not os.path.exists(annotations_dir):
            logger.error(f"Annotations directory not found: {annotations_dir}")
            return pd.DataFrame()
        
        data = []
        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        logger.info(f"Processing {len(xml_files)} XML annotation files...")
        
        for xml_file in tqdm(xml_files, desc="Loading annotations"):
            xml_path = os.path.join(annotations_dir, xml_file)
            annotation = self.parse_xml_annotation(xml_path)
            
            if annotation is None:
                continue
            
            image_path = os.path.join(images_dir, annotation['filename'])
            
            # Check if image exists
            if not os.path.exists(image_path):
                # Try with different extensions
                base_name = os.path.splitext(annotation['filename'])[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    logger.warning(f"Image not found: {annotation['filename']}")
                    continue
            
            # For each object in the annotation (handling multiple faces per image)
            for obj in annotation['objects']:
                class_name = obj['class']
                
                # Map class names to our standard names
                if class_name.lower() in ['with_mask', 'mask']:
                    mapped_class = 'with_mask'
                elif class_name.lower() in ['without_mask', 'no_mask']:
                    mapped_class = 'without_mask'
                elif class_name.lower() in ['mask_weared_incorrect', 'incorrect_mask']:
                    mapped_class = 'mask_weared_incorrect'
                else:
                    logger.warning(f"Unknown class: {class_name}")
                    continue
                
                data.append({
                    'image_path': image_path,
                    'filename': annotation['filename'],
                    'class': mapped_class,
                    'class_id': self.class_mapping[mapped_class],
                    'xmin': obj['xmin'],
                    'ymin': obj['ymin'],
                    'xmax': obj['xmax'],
                    'ymax': obj['ymax'],
                    'width': annotation['width'],
                    'height': annotation['height']
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} annotations with {len(df['filename'].unique())} unique images")
        
        return df
    
    def create_class_directories(self):
        """Create directories for each class in train/val/test splits."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = os.path.join(self.base_path, self.paths_config[f'{split}_dir'])
            
            for class_name in self.classes:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
    
    def copy_images_to_split_dirs(self, df: pd.DataFrame):
        """Copy images to appropriate train/val/test directories by class."""
        self.create_class_directories()
        
        # Group by image filename to handle multiple objects per image
        grouped = df.groupby('filename')
        
        # Split data
        filenames = list(grouped.groups.keys())
        
        # First split: separate test set
        train_val_files, test_files = train_test_split(
            filenames, 
            test_size=self.test_split, 
            random_state=42
        )
        
        # Second split: separate train and validation sets
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=self.validation_split / (1 - self.test_split),
            random_state=42
        )
        
        # Create file mappings
        file_splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        logger.info(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Copy files to respective directories
        for split, files in file_splits.items():
            split_dir = os.path.join(self.base_path, self.paths_config[f'{split}_dir'])
            
            for filename in tqdm(files, desc=f"Copying {split} files"):
                # Get all annotations for this image
                image_annotations = grouped.get_group(filename)
                
                # For simplicity, use the first annotation's class
                # In practice, you might want to handle multiple classes per image differently
                primary_class = image_annotations.iloc[0]['class']
                
                src_path = image_annotations.iloc[0]['image_path']
                dst_dir = os.path.join(split_dir, primary_class)
                dst_path = os.path.join(dst_dir, filename)
                
                try:
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                except Exception as e:
                    logger.warning(f"Error copying {src_path}: {e}")
        
        return file_splits
    
    def create_data_generators(self):
        """Create data generators for training and validation."""
        train_dir = os.path.join(self.base_path, self.paths_config['train_dir'])
        val_dir = os.path.join(self.base_path, self.paths_config['val_dir'])
        
        # Data augmentation for training (same as original)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.25,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def visualize_dataset(self, df: pd.DataFrame):
        """Visualize sample images from the dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Sample one image from each class
        for i, class_name in enumerate(self.classes):
            class_data = df[df['class'] == class_name]
            if len(class_data) > 0:
                sample = class_data.iloc[0]
                
                try:
                    image = cv2.imread(sample['image_path'])
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Draw bounding box if available
                        if all(col in sample for col in ['xmin', 'ymin', 'xmax', 'ymax']):
                            cv2.rectangle(image, 
                                        (sample['xmin'], sample['ymin']), 
                                        (sample['xmax'], sample['ymax']), 
                                        (255, 0, 0), 2)
                        
                        axes[i].imshow(image)
                        axes[i].set_title(f'{class_name}\n{sample["filename"]}')
                        axes[i].axis('off')
                    else:
                        axes[i].text(0.5, 0.5, f'Image not found\n{class_name}', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].axis('off')
                except Exception as e:
                    logger.warning(f"Error visualizing {sample['filename']}: {e}")
                    axes[i].text(0.5, 0.5, f'Error loading\n{class_name}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No images\n{class_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(self.classes), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_data_statistics(self, df: pd.DataFrame, file_splits: Dict):
        """Save data statistics to JSON file."""
        stats = {
            'total_images': len(df['filename'].unique()),
            'total_annotations': len(df),
            'classes': {},
            'splits': {}
        }
        
        # Class distribution
        for class_name in self.classes:
            class_count = len(df[df['class'] == class_name])
            stats['classes'][class_name] = class_count
        
        # Split distribution
        for split, files in file_splits.items():
            stats['splits'][split] = len(files)
        
        # Save to file
        stats_path = os.path.join(self.base_path, 'data', 'processed', 'data_stats.json')
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Data statistics saved to {stats_path}")
        return stats
    
    def process_full_pipeline(self):
        """Execute the complete data preprocessing pipeline."""
        logger.info("Starting complete data preprocessing pipeline...")
        
        # Step 1: Extract dataset
        if not self.extract_dataset():
            logger.error("Dataset extraction failed")
            return None, None, None
        
        # Step 2: Load annotations
        df = self.load_annotations()
        if df.empty:
            logger.error("No annotations loaded")
            return None, None, None
        
        # Step 3: Visualize dataset
        logger.info("Visualizing dataset samples...")
        self.visualize_dataset(df)
        
        # Step 4: Split and copy images
        logger.info("Splitting and organizing images...")
        file_splits = self.copy_images_to_split_dirs(df)
        
        # Step 5: Create data generators
        logger.info("Creating data generators...")
        train_generator, val_generator = self.create_data_generators()
        
        # Step 6: Save statistics
        stats = self.save_data_statistics(df, file_splits)
        
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Training samples: {train_generator.n}")
        logger.info(f"Validation samples: {val_generator.n}")
        
        return train_generator, val_generator, self.classes


def main():
    """Run the data preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    train_gen, val_gen, classes = preprocessor.process_full_pipeline()
    
    if train_gen is not None:
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Classes: {classes}")
        logger.info(f"Training samples: {train_gen.n}")
        logger.info(f"Validation samples: {val_gen.n}")
    else:
        logger.error("Data preprocessing failed!")


if __name__ == "__main__":
    main()
