"""
Data preprocessing module for face mask detection.
This module handles data loading, preprocessing, and augmentation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import yaml
import logging
from typing import Tuple, List
import mlflow
import mlflow.tensorflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing pipeline for face mask detection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data preprocessor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data']
        self.image_size = tuple(self.data_config['image_size'])
        self.batch_size = self.data_config['batch_size']
        self.classes = self.data_config['classes']
        
        logger.info(f"DataPreprocessor initialized with image size: {self.image_size}")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the face mask dataset."""
        logger.info("Loading face mask dataset...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_config['raw_data_path'], class_name)
            
            if not os.path.exists(class_path):
                logger.warning(f"Class path {class_path} does not exist. Creating directory.")
                os.makedirs(class_path, exist_ok=True)
                continue
                
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, filename)
                    image = self._load_and_preprocess_image(image_path)
                    
                    if image is not None:
                        images.append(image)
                        labels.append(class_idx)
        
        if len(images) == 0:
            logger.error("No images found in the dataset directories!")
            raise ValueError("No images found in the dataset directories!")
        
        logger.info(f"Loaded {len(images)} images with {len(set(labels))} classes")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Normalize pixel values
        X = X.astype('float32') / 255.0
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=len(self.classes))
        
        return X, y
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        test_size = self.data_config['test_split']
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation sets
        val_size = self.data_config['validation_split'] / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray):
        """Create data generators for training with augmentation."""
        logger.info("Creating data generators with augmentation...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def save_processed_data(self, X_train: np.ndarray, X_val: np.ndarray, 
                           X_test: np.ndarray, y_train: np.ndarray, 
                           y_val: np.ndarray, y_test: np.ndarray):
        """Save processed data to disk."""
        logger.info("Saving processed data...")
        
        processed_path = self.data_config['processed_data_path']
        os.makedirs(processed_path, exist_ok=True)
        
        # Save training data
        np.save(os.path.join(processed_path, 'X_train.npy'), X_train)
        np.save(os.path.join(processed_path, 'y_train.npy'), y_train)
        
        # Save validation data
        np.save(os.path.join(processed_path, 'X_val.npy'), X_val)
        np.save(os.path.join(processed_path, 'y_val.npy'), y_val)
        
        # Save test data
        np.save(os.path.join(processed_path, 'X_test.npy'), X_test)
        np.save(os.path.join(processed_path, 'y_test.npy'), y_test)
        
        logger.info(f"Processed data saved to {processed_path}")
        
        # Log data statistics to MLflow
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("image_size", self.image_size)
        mlflow.log_param("num_classes", len(self.classes))

def main():
    """Main function to run data preprocessing pipeline."""
    # Initialize MLflow
    mlflow.set_experiment("face_mask_detection")
    
    with mlflow.start_run(run_name="data_preprocessing"):
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        X, y = preprocessor.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Save processed data
        preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
