"""
Model training pipeline for face mask detection.
This module implements the training pipeline with MLflow integration.
"""

import os
import numpy as np
import yaml
import logging
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceMaskDetector:
    """Face mask detection model using MobileNetV2."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.mlflow_config = self.config['mlflow']
        
        self.model = None
        
        logger.info("FaceMaskDetector initialized")
    
    def build_model(self) -> Model:
        """Build the face mask detection model."""
        logger.info("Building MobileNetV2-based model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=tuple(self.model_config['input_shape']),
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=tuple(self.model_config['input_shape']))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.model_config['dropout_rate'])(x)
        outputs = Dense(
            self.model_config['num_classes'],
            activation='softmax',
            name='predictions'
        )(x)
        
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.model_config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def create_callbacks(self, model_save_path: str) -> list:
        """Create training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['reduce_lr_factor'],
                patience=self.training_config['reduce_lr_patience'],
                min_lr=self.training_config['min_lr'],
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_generator, val_generator, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the face mask detection model."""
        logger.info("Starting model training...")
        
        # Create model save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/face_mask_detector_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=self.training_config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Make predictions for detailed metrics
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Calculate detailed metrics
        class_report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=['No Mask', 'Mask'],
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        results = {
            'history': history.history,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'model_path': model_save_path
        }
        
        logger.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
        return results
    
    def log_results_to_mlflow(self, results: Dict[str, Any], 
                             run_name: str = None):
        """Log training results to MLflow."""
        logger.info("Logging results to MLflow...")
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model_architecture", self.model_config['architecture'])
            mlflow.log_param("input_shape", self.model_config['input_shape'])
            mlflow.log_param("learning_rate", self.model_config['learning_rate'])
            mlflow.log_param("dropout_rate", self.model_config['dropout_rate'])
            mlflow.log_param("epochs", self.training_config['epochs'])
            mlflow.log_param("batch_size", self.config['data']['batch_size'])
            
            # Log metrics
            mlflow.log_metric("val_loss", results['val_loss'])
            mlflow.log_metric("val_accuracy", results['val_accuracy'])
            
            # Log classification metrics
            class_report = results['classification_report']
            mlflow.log_metric("precision_mask", class_report['Mask']['precision'])
            mlflow.log_metric("recall_mask", class_report['Mask']['recall'])
            mlflow.log_metric("f1_score_mask", class_report['Mask']['f1-score'])
            mlflow.log_metric("precision_no_mask", class_report['No Mask']['precision'])
            mlflow.log_metric("recall_no_mask", class_report['No Mask']['recall'])
            mlflow.log_metric("f1_score_no_mask", class_report['No Mask']['f1-score'])
            
            # Plot and log training history
            self._plot_training_history(results['history'])
            
            # Plot and log confusion matrix
            self._plot_confusion_matrix(results['confusion_matrix'])
            
            # Log model
            mlflow.tensorflow.log_model(
                self.model,
                "model",
                registered_model_name="face_mask_detector"
            )
            
            # Log model file
            mlflow.log_artifact(results['model_path'])
    
    def _plot_training_history(self, history: Dict[str, list]):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        mlflow.log_artifact('training_history.png')
        plt.close()
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Mask', 'Mask'],
            yticklabels=['No Mask', 'Mask']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

def load_processed_data() -> Tuple[np.ndarray, ...]:
    """Load preprocessed data."""
    logger.info("Loading preprocessed data...")
    
    data_path = "data/processed"
    
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    logger.info("Data loaded successfully")
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train face mask detection model')
    parser.add_argument('--experiment-name', type=str, 
                       default='face_mask_detection',
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, 
                       default=None,
                       help='MLflow run name')
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    try:
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
        
        # Initialize preprocessor for data generators
        preprocessor = DataPreprocessor()
        train_generator, val_generator = preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Initialize and build model
        detector = FaceMaskDetector()
        detector.build_model()
        
        # Train model
        results = detector.train_model(train_generator, val_generator, X_val, y_val)
        
        # Log results to MLflow
        run_name = args.run_name or f"mask_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        detector.log_results_to_mlflow(results, run_name)
        
        logger.info("Training pipeline completed successfully!")
        
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Please run data preprocessing first.")
        logger.info("Run: python src/data_preprocessing.py")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
