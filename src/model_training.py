"""
Model training pipeline for face mask detection.
This module implements the training pipeline using MobileNetV2.
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import yaml
import logging
import json
from datetime import datetime
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, AveragePooling2D, Flatten, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .data_preprocessing import DataPreprocessor
except ImportError:
    from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceMaskDetector:
    """Face mask detection model using MobileNetV2."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model with configuration."""
        self.config_path = config_path
        self.load_config()
        self.model = None
        self.history = None
        
        logger.info("FaceMaskDetector initialized")
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.model_config = self.config['model']
            self.training_config = self.config['training']
            self.paths_config = self.config['paths']
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using default configuration.")
            self.model_config = {
                'input_shape': [224, 224, 3],
                'num_classes': 3,
                'dropout_rate': 0.5
            }
            self.training_config = {
                'epochs': 30,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'patience': 10
            }
            self.paths_config = {
                'models_dir': 'models',
                'model_name': 'mask_detector.h5'
            }
    
    def build_model(self) -> Model:
        """Build the face mask detection model (same architecture as original)."""
        logger.info("Building MobileNetV2-based face mask detection model...")
        
        # Model parameters from original code
        INIT_LR = self.training_config['learning_rate']
        input_shape = tuple(self.model_config['input_shape'])
        
        # Input tensor
        input_tensor = Input(shape=input_shape)
        
        # Base model - MobileNetV2
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor
        )
        
        # Add custom layers exactly as per original implementation
        x = base_model.output
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Flatten(name="flatten")(x)
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.model_config['num_classes'], activation="softmax")(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        logger.info("Model architecture built successfully")
        return model
    
    def compile_model(self, model: Model) -> Model:
        """Compile the model (same as original)."""
        logger.info("Compiling model...")
        
        INIT_LR = self.training_config['learning_rate']
        
        # Optimizer with learning rate as in original code
        opt = Adam(learning_rate=INIT_LR)
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"]
        )
        
        logger.info("Model compiled successfully")
        return model
    
    def get_callbacks(self):
        """Get training callbacks."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_path, self.paths_config['models_dir'])
        os.makedirs(models_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(models_dir, 'best_mask_detector.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_generator, val_generator):
        """Train the face mask detection model."""
        logger.info("Starting model training...")
        
        # Build and compile model
        self.model = self.build_model()
        self.model = self.compile_model(self.model)
        
        # Training parameters
        EPOCHS = self.training_config['epochs']
        
        # Calculate steps per epoch
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        
        logger.info(f"Training for {EPOCHS} epochs")
        logger.info(f"Training steps per epoch: {STEP_SIZE_TRAIN}")
        logger.info(f"Validation steps per epoch: {STEP_SIZE_VALID}")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train the model using fit (updated from fit_generator)
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def visualize_training_history(self):
        """Visualize model training performance (same as original)."""
        if self.history is None:
            logger.warning("No training history available for visualization")
            return
        
        logger.info("Visualizing training history...")
        
        # Extract metrics
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs_range = range(len(acc))
        
        # Create plots exactly as in original
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 12))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            logger.error("No model to save. Train the model first.")
            return
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_path, self.paths_config['models_dir'])
        model_path = os.path.join(models_dir, self.paths_config['model_name'])
        
        logger.info(f"Saving mask detector model to {model_path}...")
        self.model.save(model_path)
        logger.info("Model saved successfully")
    
    def save_training_metrics(self):
        """Save training metrics to JSON file."""
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_path, self.paths_config['models_dir'])
        
        # Training history for plots
        history_data = {
            'epoch': list(range(len(self.history.history['accuracy']))),
            'accuracy': self.history.history['accuracy'],
            'val_accuracy': self.history.history['val_accuracy'],
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss']
        }
        
        # Save training history
        history_path = os.path.join(models_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Final metrics
        final_metrics = {
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
            'total_epochs': len(self.history.history['accuracy'])
        }
        
        # Save metrics
        metrics_path = os.path.join(models_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {models_dir}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        
        # Initialize data preprocessor
        preprocessor = DataPreprocessor(self.config_path)
        
        # Process data
        train_generator, val_generator, classes = preprocessor.process_full_pipeline()
        
        if train_generator is None:
            logger.error("Data preprocessing failed. Cannot proceed with training.")
            return None
        
        # Train model
        history = self.train_model(train_generator, val_generator)
        
        # Visualize training results
        self.visualize_training_history()
        
        # Save model and metrics
        self.save_model()
        self.save_training_metrics()
        
        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info("Training Pipeline Complete!")
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        logger.info(f"Final Training Loss: {final_train_loss:.4f}")
        logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
        
        return {
            'train_accuracy': final_train_acc,
            'val_accuracy': final_val_acc,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'classes': classes
        }


def main():
    """Main function to run the training pipeline."""
    logger.info("Starting Face Mask Detection Model Training...")
    
    # Initialize detector
    detector = FaceMaskDetector()
    
    # Run training pipeline
    results = detector.run_training_pipeline()
    
    if results:
        logger.info("Training completed successfully!")
        for key, value in results.items():
            if key != 'classes':
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    else:
        logger.error("Training failed!")


if __name__ == "__main__":
    main()
