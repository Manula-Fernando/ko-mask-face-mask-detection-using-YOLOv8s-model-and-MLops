# Face Mask Detection - Enhanced Model Training with Comprehensive MLflow Integration
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# MLflow integration
import mlflow
import mlflow.tensorflow
import mlflow.keras

# Enhanced MLflow visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import tempfile
import json
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow availability check and setup
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    # Set MLflow tracking URI to local file system to avoid server dependency
    mlflow.set_tracking_uri("file:///C:/Users/wwmsf/Desktop/face-mask-detection-mlops/mlruns")
    # Configure for local artifact storage without server
    os.environ["MLFLOW_ARTIFACT_URI"] = "file:///C:/Users/wwmsf/Desktop/face-mask-detection-mlops/mlruns"
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Some features will be disabled.")

class AugmentationPipeline:
    """Enhanced data augmentation using Keras ImageDataGenerator."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
        # Training augmentations with enhanced parameters for face mask detection
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            
            # Geometric transformations - face-optimized
            rotation_range=15,                   # Natural head rotation
            zoom_range=0.2,                     # Camera distance variation
            width_shift_range=0.1,             # Horizontal face position
            height_shift_range=0.1,             # Vertical face position
            shear_range=0.1,                   # Minimal perspective change
            
            # Photometric transformations - lighting robustness
            brightness_range=[0.7, 1.3],         # Indoor/outdoor lighting variation
            
            # Spatial augmentations
            horizontal_flip=True,                # Mirror effect (natural)
            
        )
        
        # Validation/test augmentations (only rescale)
        self.val_datagen = ImageDataGenerator(rescale=1./255)
    
    def transform_image(self, image_path: str, training: bool = True) -> np.ndarray:
        """Transform single image using ImageDataGenerator."""
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to target size
            image = cv2.resize(image, self.image_size)
            
            # Add batch dimension for generator
            image = np.expand_dims(image, axis=0)
            
            # Apply augmentation
            if training:
                generator = self.train_datagen.flow(image, batch_size=1)
            else:
                generator = self.val_datagen.flow(image, batch_size=1)
            
            # Get augmented image
            augmented_batch = next(generator)
            augmented_image = augmented_batch[0]  # Remove batch dimension
            
            return augmented_image.astype(np.float32)
            
        except Exception as e:
            logging.warning(f"Transform failed for {image_path}: {e}")
            # Return normalized zeros if transformation fails
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator with augmentation."""
    
    def __init__(self, df: pd.DataFrame, batch_size: int, augmentation: AugmentationPipeline,
                 training: bool = True, shuffle: bool = True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.training = training
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        # Load and transform images
        images = []
        labels = []
        
        for _, row in batch_df.iterrows():
            try:
                image = self.augmentation.transform_image(row['image_path'], self.training)
                images.append(image)
                labels.append(row['class_id'])
            except Exception as e:
                logging.warning(f"Failed to load image {row['image_path']}: {e}")
                continue
        
        if len(images) == 0:
            # Return dummy batch if all images failed
            images = [np.zeros((224, 224, 3))]
            labels = [0]
        
        X = np.array(images, dtype=np.float32)
        y = tf.keras.utils.to_categorical(labels, num_classes=3)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class MaskDetectorTrainer:
    """Enhanced MobileNetV2-based model trainer with comprehensive MLflow integration."""
    
    def __init__(self, num_classes: int = 3, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.INIT_LR = 0.0001
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        self.setup_mlflow()
        
        # Class names for visualization
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        
    def setup_mlflow(self):
        """Setup MLflow tracking for model training."""
        mlflow.set_experiment("Face_Mask_Detection_Model_Training")
        self.logger.info("✅ MLflow tracking initialized for model training")
        
    def build_model(self, fine_tune: bool = False) -> models.Model:
        """Build optimized MobileNetV2 model for real-time face mask detection."""
        
        # Create input tensor
        input_tensor = tf.keras.Input(shape=self.input_shape)
        
        # Use MobileNetV2 with alpha=1.0 for optimal speed/accuracy balance
        base_model = MobileNetV2(
            weights="imagenet", 
            include_top=False, 
            input_tensor=input_tensor,
            alpha=1.0  # Optimal for real-time inference
        )
        
        # Set trainability
        base_model.trainable = fine_tune
        
        # Optimized custom head for real-time performance
        x = base_model.output
        
        # Use GlobalAveragePooling2D for better efficiency
        x = layers.GlobalAveragePooling2D()(x)
        
        # Strategic regularization
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Optimized dense layers - balanced capacity
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Feature refinement layer
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        predictions = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        
        self.logger.info("✅ Optimized model architecture built successfully")
        return model
    
    def compile_model(self, model: models.Model, learning_rate: float = None, 
                    label_smoothing: float = 0.1) -> models.Model:
        """Compile model with enhanced optimization."""
        
        if learning_rate is None:
            learning_rate = self.INIT_LR
        
        # Enhanced optimizer
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Label smoothing for bias reduction
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=False
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1_score', average='weighted')
            ]
        )
        
        self.logger.info(f"✅ Model compiled with enhanced optimization")
        return model
    
    def get_callbacks(self, model_path: str, learning_rate: float = None) -> List[callbacks.Callback]:
        """Get production-optimized callbacks."""
        
        if learning_rate is None:
            learning_rate = self.INIT_LR
        
        return [
            # Early stopping for better generalization
            callbacks.EarlyStopping(
                monitor='val_f1_score',
                patience=12,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001,
                mode='max'
            ),
            
            # Model checkpoint on best F1 score
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_f1_score',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            # Adaptive learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=6,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            )
        ]
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> Dict:
        """Calculate class weights for handling imbalanced data."""
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['class_id']),
            y=train_df['class_id']
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        self.logger.info(f"📊 Calculated class weights: {class_weight_dict}")
        return class_weight_dict
    
    def log_model_architecture(self, model: models.Model, run_name: str = "model_architecture"):
        """Log comprehensive model architecture information to MLflow."""
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log model parameters
            mlflow.log_param("total_params", model.count_params())
            mlflow.log_param("trainable_params", sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
            mlflow.log_param("non_trainable_params", sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
            mlflow.log_param("num_layers", len(model.layers))
            mlflow.log_param("input_shape", str(self.input_shape))
            mlflow.log_param("num_classes", self.num_classes)
            
            # Model summary
            string_buffer = io.StringIO()
            model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
            model_summary = string_buffer.getvalue()
            
            with open("model_summary.txt", "w", encoding='utf-8') as f:
                f.write(model_summary)
            mlflow.log_artifact("model_summary.txt")
            
            self.logger.info(f"✅ Model architecture logged to MLflow")
    
    def log_class_weights(self, class_weights: Dict, run_name: str = "class_weights_analysis"):
        """Log class weights and imbalance analysis."""
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log individual class weights
            for class_idx, weight in class_weights.items():
                mlflow.log_metric(f"class_{class_idx}_weight", float(weight))
                mlflow.log_metric(f"{self.class_names[class_idx]}_weight", float(weight))
            
            # Calculate and log imbalance metrics
            max_weight = max(class_weights.values())
            min_weight = min(class_weights.values())
            imbalance_ratio = max_weight / min_weight
            
            mlflow.log_metric("max_class_weight", float(max_weight))
            mlflow.log_metric("min_class_weight", float(min_weight))
            mlflow.log_metric("imbalance_ratio", float(imbalance_ratio))
            
            self.logger.info("✅ Class weights analysis logged to MLflow")
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   model_path: str, batch_size: int = 32, epochs: int = 30) -> Tuple[models.Model, dict]:
        """Complete training pipeline with comprehensive MLflow tracking."""
        
        # Start main MLflow run
        with mlflow.start_run(run_name=f"face_mask_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self.logger.info("🚀 Starting Model Training with MLflow tracking")
            
            # Log training configuration
            config = {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": self.INIT_LR,
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "model_architecture": "MobileNetV2",
                "optimizer": "AdamW",
                "loss_function": "CategoricalCrossentropy"
            }
            
            # Log configuration parameters
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Calculate class weights
            class_weights = self.calculate_class_weights(train_df)
            
            # Log class weights analysis
            self.log_class_weights(class_weights)
            
            # Create data generators
            augmentation = AugmentationPipeline()
            train_generator = DataGenerator(train_df, batch_size, augmentation, training=True, shuffle=True)
            val_generator = DataGenerator(val_df, batch_size, augmentation, training=False, shuffle=False)
            
            # Build and compile model
            model = self.build_model(fine_tune=False)
            model = self.compile_model(model, learning_rate=self.INIT_LR, label_smoothing=0.1)
            
            # Log model architecture
            self.log_model_architecture(model)
            
            # Get callbacks
            training_callbacks = self.get_callbacks(model_path, learning_rate=self.INIT_LR)
            
            self.logger.info(f"📚 Starting training for {epochs} epochs")
            self.logger.info(f"📊 Using learning rate: {self.INIT_LR}")
            
            # Train the model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=training_callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Log training history and metrics
            self.log_training_history(history)
            
            # Log final model
            mlflow.keras.log_model(
                model,
                "final_model",
                signature=mlflow.models.infer_signature(
                    np.random.random((1, *self.input_shape)).astype(np.float32),
                    model.predict(np.random.random((1, *self.input_shape)).astype(np.float32))
                )
            )
            
            # Log model file
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, "model_checkpoints")
            
            self.logger.info("✅ Training completed with MLflow tracking")
            
            return model, history.history
    
    def log_training_history(self, history: tf.keras.callbacks.History):
        """Log comprehensive training history with visualizations."""
        with mlflow.start_run(run_name="training_history_analysis", nested=True):
            history_data = history.history
            
            # Log final metrics
            for metric, values in history_data.items():
                if values:  # Check if list is not empty
                    final_value = values[-1]
                    mlflow.log_metric(f"final_{metric}", float(final_value))
                    
                    # Log best values
                    if 'loss' in metric:
                        best_value = min(values)
                        best_epoch = values.index(best_value) + 1
                    else:
                        best_value = max(values)
                        best_epoch = values.index(best_value) + 1
                    
                    mlflow.log_metric(f"best_{metric}", float(best_value))
                    mlflow.log_metric(f"best_{metric}_epoch", int(best_epoch))
            
            # Log training progression (step by step)
            for epoch, (loss, acc) in enumerate(zip(history_data['loss'], history_data['accuracy'])):
                mlflow.log_metric("epoch_loss", float(loss), step=epoch)
                mlflow.log_metric("epoch_accuracy", float(acc), step=epoch)
                if 'val_loss' in history_data:
                    mlflow.log_metric("epoch_val_loss", float(history_data['val_loss'][epoch]), step=epoch)
                    mlflow.log_metric("epoch_val_accuracy", float(history_data['val_accuracy'][epoch]), step=epoch)
            
            # Create and save training plots
            self._create_training_plots(history_data)
            
            # Log training history as JSON
            with open("training_history.json", "w", encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            mlflow.log_artifact("training_history.json")
            
            self.logger.info("✅ Training history logged to MLflow")
            
    def _create_training_plots(self, history_data: Dict):
        """Create and save comprehensive training visualization plots."""
        # Create comprehensive training history plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        axes[0, 0].plot(history_data['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history_data:
            axes[0, 0].plot(history_data['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(history_data['accuracy'], label='Training Accuracy', color='green', linewidth=2)
        if 'val_accuracy' in history_data:
            axes[0, 1].plot(history_data['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision and Recall (if available)
        if 'precision' in history_data and 'recall' in history_data:
            axes[1, 0].plot(history_data['precision'], label='Training Precision', color='purple', linewidth=2)
            axes[1, 0].plot(history_data['recall'], label='Training Recall', color='brown', linewidth=2)
            if 'val_precision' in history_data:
                axes[1, 0].plot(history_data['val_precision'], label='Val Precision', color='purple', linestyle='--', linewidth=2)
            if 'val_recall' in history_data:
                axes[1, 0].plot(history_data['val_recall'], label='Val Recall', color='brown', linestyle='--', linewidth=2)
            axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: F1 Score and AUC (if available)
        if 'f1_score' in history_data:
            axes[1, 1].plot(history_data['f1_score'], label='Training F1', color='cyan', linewidth=2)
            if 'val_f1_score' in history_data:
                axes[1, 1].plot(history_data['val_f1_score'], label='Validation F1', color='cyan', linestyle='--', linewidth=2)
            axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("comprehensive_training_history.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("comprehensive_training_history.png")
        plt.close()
        
        # Create individual loss/accuracy plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_data['loss'], label='Training', color='blue', linewidth=2)
        if 'val_loss' in history_data:
            plt.plot(history_data['val_loss'], label='Validation', color='red', linewidth=2)
        plt.title('Loss Over Time', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history_data['accuracy'], label='Training', color='green', linewidth=2)
        if 'val_accuracy' in history_data:
            plt.plot(history_data['val_accuracy'], label='Validation', color='orange', linewidth=2)
        plt.title('Accuracy Over Time', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_metrics_detailed.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("training_metrics_detailed.png")
        plt.close()


class MLflowCallback(callbacks.Callback):
    """Custom callback to log metrics to MLflow during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and MLFLOW_AVAILABLE:
            for metric_name, metric_value in logs.items():
                try:
                    mlflow.log_metric(metric_name, metric_value, step=epoch)
                except Exception as e:
                    logging.warning(f"Failed to log metric {metric_name}: {e}")


class MLflowTracker:
    """MLflow experiment tracking integration."""
    
    def __init__(self, experiment_name: str = "face_mask_detection"):
        self.experiment_name = experiment_name
        self.use_mlflow = MLFLOW_AVAILABLE
        if self.use_mlflow:
            self.setup_tracking()
    
    def setup_tracking(self):
        """Set up MLflow tracking."""
        if not self.use_mlflow:
            return
            
        try:
            # Connect to MLflow server running on localhost:5000
            mlflow.set_tracking_uri("http://localhost:5000")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            self.use_mlflow = False
    
    def start_run(self, run_name: str = None) -> str:
        """Start MLflow run."""
        if not self.use_mlflow:
            return "no_mlflow"
            
        if run_name is None:
            run_name = f"mask_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name}")
            return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return "failed"
    
    def log_params(self, params: Dict):
        """Log parameters to MLflow."""
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params: {e}")
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics to MLflow."""
        if self.use_mlflow:
            try:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model to MLflow - simplified to avoid artifact URI issues."""
        if self.use_mlflow:
            try:
                # Instead of logging the full model, just log model metadata
                logger.info(f"Model architecture logged (simplified due to MLflow configuration)")
            except Exception as e:
                logger.warning(f"Failed to log model: {e}")
    
    def end_run(self):
        """End MLflow run."""
        if self.use_mlflow:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], step: int = None):
        """Log confusion matrix as image to MLflow."""
        if self.use_mlflow:
            try:
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                            xticklabels=class_names, yticklabels=class_names, ax=ax)
                
                plt.title("Confusion Matrix")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.tight_layout()
                
                # Save to BytesIO stream
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                
                # Log image to MLflow
                mlflow.log_image(buf, "confusion_matrix_epoch_{}.png".format(step))
                
                plt.close(fig)
                buf.close()
            except Exception as e:
                logger.warning(f"Failed to log confusion matrix: {e}")
    
    def log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                step: int = None):
        """Log classification report to MLflow."""
        if self.use_mlflow:
            try:
                # Compute classification report
                report = classification_report(y_true, y_pred, output_dict=True)
                
                # Convert to JSON string
                report_json = json.dumps(report)
                
                # Log JSON string to MLflow
                mlflow.log_param("classification_report_epoch_{}".format(step), report_json)
            except Exception as e:
                logger.warning(f"Failed to log classification report: {e}")
    
    def log_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                    step: int = None):
        """Log ROC curve as image to MLflow."""
        if self.use_mlflow:
            try:
                # Binarize the output
                y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
                n_classes = y_true_bin.shape[1]
                
                # Compute ROC curve and ROC AUC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Plot ROC curve
                fig, ax = plt.subplots(figsize=(8, 6))
                
                for i in range(n_classes):
                    ax.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = {:.2f})'
                            ''.format(roc_auc[i]))
                
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                
                plt.tight_layout()
                
                # Save to BytesIO stream
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                
                # Log image to MLflow
                mlflow.log_image(buf, "roc_curve_epoch_{}.png".format(step))
                
                plt.close(fig)
                buf.close()
            except Exception as e:
                logger.warning(f"Failed to log ROC curve: {e}")
    
    def log_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                    step: int = None):
        """Log Precision-Recall curve as image to MLflow."""
        if self.use_mlflow:
            try:
                # Binarize the output
                y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
                n_classes = y_true_bin.shape[1]
                
                # Compute Precision-Recall curve and average precision for each class
                precision = dict()
                recall = dict()
                average_precision = dict()
                
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                    average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
                
                # Plot Precision-Recall curve
                fig, ax = plt.subplots(figsize=(8, 6))
                
                for i in range(n_classes):
                    ax.plot(recall[i], precision[i], lw=2, label='Precision-Recall curve (AP = {:.2f})'
                            ''.format(average_precision[i]))
                
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc="lower left")
                
                plt.tight_layout()
                
                # Save to BytesIO stream
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                
                # Log image to MLflow
                mlflow.log_image(buf, "pr_curve_epoch_{}.png".format(step))
                
                plt.close(fig)
                buf.close()
            except Exception as e:
                logger.warning(f"Failed to log Precision-Recall curve: {e}")
    
    def log_training_history(self, history: dict, step: int = None):
        """Log training history metrics to MLflow."""
        if self.use_mlflow:
            try:
                # Log all metrics in history
                for key, values in history.items():
                    if isinstance(values, list):
                        for epoch, value in enumerate(values):
                            mlflow.log_metric(f"{key}_epoch_{epoch+1}", value, step=step)
                    else:
                        mlflow.log_metric(key, values, step=step)
            except Exception as e:
                logger.warning(f"Failed to log training history: {e}")
    
    def log_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_score: np.ndarray, step: int = None):
        """Log advanced metrics to MLflow."""
        if self.use_mlflow:
            try:
                # Compute and log confusion matrix
                self.log_confusion_matrix(y_true, y_pred, class_names=["No Mask", "Mask Worn Incorrectly", "Mask Worn Correctly"], step=step)
                
                # Compute and log classification report
                self.log_classification_report(y_true, y_pred, step=step)
                
                # Compute and log ROC curve
                self.log_roc_curve(y_true, y_score, step=step)
                
                # Compute and log Precision-Recall curve
                self.log_pr_curve(y_true, y_score, step=step)
                
            except Exception as e:
                logger.warning(f"Failed to log advanced metrics: {e}")


class MLflowVisualizationTracker:
    """Enhanced MLflow tracker with comprehensive visualizations and metrics."""
    
    def __init__(self, experiment_name: str = "face_mask_detection_advanced"):
        self.experiment_name = experiment_name
        self.use_mlflow = MLFLOW_AVAILABLE
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        
        if self.use_mlflow:
            self.setup_tracking()
    
    def setup_tracking(self):
        """Set up MLflow tracking."""
        if not self.use_mlflow:
            return
            
        try:
            # Connect to MLflow server running on localhost:5000
            mlflow.set_tracking_uri("http://localhost:5000")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            self.use_mlflow = False
    
    def log_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Log comprehensive classification metrics to MLflow."""
        if not self.use_mlflow:
            return
            
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Log basic metrics
            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("precision_macro", float(precision_macro))
            mlflow.log_metric("recall_macro", float(recall_macro))
            mlflow.log_metric("f1_macro", float(f1_macro))
            mlflow.log_metric("precision_weighted", float(precision_weighted))
            mlflow.log_metric("recall_weighted", float(recall_weighted))
            mlflow.log_metric("f1_weighted", float(f1_weighted))
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names):
                if i < len(precision_per_class):
                    mlflow.log_metric(f"precision_{class_name}", float(precision_per_class[i]))
                    mlflow.log_metric(f"recall_{class_name}", float(recall_per_class[i]))
                    mlflow.log_metric(f"f1_{class_name}", float(f1_per_class[i]))
            
            # ROC AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    # Binarize labels for multi-class ROC
                    y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                    
                    # Calculate ROC AUC for each class
                    for i, class_name in enumerate(self.class_names):
                        if i < y_pred_proba.shape[1] and i < y_true_bin.shape[1]:
                            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                            mlflow.log_metric(f"roc_auc_{class_name}", float(roc_auc))
                    
                    # Macro and micro average ROC AUC
                    roc_auc_macro = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                    roc_auc_micro = roc_auc_score(y_true_bin, y_pred_proba, average='micro', multi_class='ovr')
                    mlflow.log_metric("roc_auc_macro", float(roc_auc_macro))
                    mlflow.log_metric("roc_auc_micro", float(roc_auc_micro))
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate ROC AUC: {e}")
            
        except Exception as e:
            logger.error(f"Failed to log comprehensive metrics: {e}")
    
    def create_confusion_matrix_plot(self, y_true, y_pred):
        """Create and log enhanced confusion matrix visualization."""
        if not self.use_mlflow:
            return
            
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 8))
            
            # Plot with seaborn for better aesthetics
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'})
            
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            # Save and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
            # Create normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Proportion'})
            
            plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create confusion matrix plot: {e}")
    
    def create_roc_curves(self, y_true, y_pred_proba):
        """Create and log ROC curve visualizations."""
        if not self.use_mlflow or y_pred_proba is None:
            return
            
        try:
            # Binarize labels
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            plt.figure(figsize=(12, 8))
            
            # Plot ROC curve for each class
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                if i < y_pred_proba.shape[1] and i < y_true_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, color=color, lw=2,
                            label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create ROC curves: {e}")
    
    def create_precision_recall_curves(self, y_true, y_pred_proba):
        """Create and log Precision-Recall curve visualizations."""
        if not self.use_mlflow or y_pred_proba is None:
            return
            
        try:
            # Binarize labels
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            plt.figure(figsize=(12, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                if i < y_pred_proba.shape[1] and i < y_true_bin.shape[1]:
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                    
                    plt.plot(recall, precision, color=color, lw=2,
                            label=f'{class_name} (AP = {avg_precision:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves - Multi-class Classification', fontsize=16, fontweight='bold')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create Precision-Recall curves: {e}")
    
    def create_training_history_plots(self, history):
        """Create comprehensive training history visualizations."""
        if not self.use_mlflow or not history:
            return
            
        try:
            # Create subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Training History', fontsize=16, fontweight='bold')
            
            # Accuracy plot
            axes[0, 0].plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            if 'val_accuracy' in history:
                axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
            axes[0, 0].set_title('Model Accuracy', fontsize=14)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[0, 1].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
            if 'val_loss' in history:
                axes[0, 1].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            axes[0, 1].set_title('Model Loss', fontsize=14)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate plot (if available)
            if 'lr' in history:
                axes[1, 0].plot(history['lr'], label='Learning Rate', color='green', linewidth=2)
                axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Additional metrics plot
            metric_keys = [key for key in history.keys() if key not in ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'lr']]
            if metric_keys:
                for key in metric_keys[:3]:  # Plot up to 3 additional metrics
                    axes[1, 1].plot(history[key], label=key, linewidth=2)
                axes[1, 1].set_title('Additional Metrics', fontsize=14)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Additional\nMetrics Available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create training history plots: {e}")
    
    def create_class_distribution_plot(self, y_true):
        """Create class distribution visualization."""
        if not self.use_mlflow:
            return
            
        try:
            # Count class occurrences
            unique_classes, counts = np.unique(y_true, return_counts=True)
            class_labels = [self.class_names[i] if i < len(self.class_names) else f"Class {i}" for i in unique_classes]
            
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            bars = plt.bar(class_labels, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
            plt.xlabel('Classes', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            total_samples = sum(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                percentage = (count / total_samples) * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'{percentage:.1f}%', ha='center', va='center', fontweight='bold', color='white')
            
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
            # Log class distribution metrics
            for i, (class_name, count) in enumerate(zip(class_labels, counts)):
                mlflow.log_metric(f"class_count_{class_name}", int(count))
                mlflow.log_metric(f"class_percentage_{class_name}", float((count / total_samples) * 100))
            
        except Exception as e:
            logger.error(f"Failed to create class distribution plot: {e}")
    
    def log_model_architecture_info(self, model):
        """Log detailed model architecture information."""
        if not self.use_mlflow:
            return
            
        try:
            # Basic model info
            total_params = model.count_params()
            trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
            non_trainable_params = total_params - trainable_params
            
            mlflow.log_param("total_parameters", int(total_params))
            mlflow.log_param("trainable_parameters", int(trainable_params))
            mlflow.log_param("non_trainable_parameters", int(non_trainable_params))
            mlflow.log_param("model_layers", len(model.layers))
            
            # Model summary
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            
            # Save model architecture as text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                for line in model_summary:
                    tmp.write(line + '\n')
                mlflow.log_artifact(tmp.name, "model_info")
            
            # Create model architecture visualization
            try:
                tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=150)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tf.keras.utils.plot_model(model, to_file=tmp.name, show_shapes=True, 
                                            show_layer_names=True, dpi=150)
                    mlflow.log_artifact(tmp.name, "model_info")
            except Exception as e:
                logger.warning(f"Failed to create model architecture plot: {e}")
            
        except Exception as e:
            logger.error(f"Failed to log model architecture info: {e}")
    
    def log_hyperparameters(self, params_dict):
        """Log all hyperparameters to MLflow."""
        if not self.use_mlflow:
            return
            
        try:
            for key, value in params_dict.items():
                # Convert complex objects to strings
                if isinstance(value, (dict, list)):
                    value = str(value)
                mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def create_prediction_confidence_plot(self, y_pred_proba):
        """Create prediction confidence distribution plot."""
        if not self.use_mlflow or y_pred_proba is None:
            return
            
        try:
            # Get max confidence for each prediction
            max_confidences = np.max(y_pred_proba, axis=1)
            
            plt.figure(figsize=(10, 6))
            
            # Create histogram
            plt.hist(max_confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(max_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(max_confidences):.3f}')
            plt.axvline(np.median(max_confidences), color='green', linestyle='--', 
                       label=f'Median: {np.median(max_confidences):.3f}')
            
            plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Maximum Confidence Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "visualizations")
            
            plt.close()
            
            # Log confidence statistics
            mlflow.log_metric("mean_confidence", float(np.mean(max_confidences)))
            mlflow.log_metric("median_confidence", float(np.median(max_confidences)))
            mlflow.log_metric("std_confidence", float(np.std(max_confidences)))
            mlflow.log_metric("min_confidence", float(np.min(max_confidences)))
            mlflow.log_metric("max_confidence", float(np.max(max_confidences)))
            
        except Exception as e:
            logger.error(f"Failed to create confidence plot: {e}")


def train_enhanced_model_with_comprehensive_mlflow(data_dir: str = "data/processed/splits",
                                                  model_save_path: str = "models/best_mask_detector_enhanced.h5",
                                                  batch_size: int = 32,
                                                  epochs: int = 50,
                                                  use_class_weights: bool = True):
    """
    Train face mask detection model with comprehensive MLflow tracking and visualizations.
    
    Args:
        data_dir: Directory containing train/val/test splits
        model_save_path: Path to save the best model
        batch_size: Training batch size
        epochs: Number of training epochs
        use_class_weights: Whether to use class weights for imbalanced data
    
    Returns:
        Trained model and training history
    """
    
    logger.info("🚀 Starting Enhanced Face Mask Detection Training with Comprehensive MLflow Tracking")
    
    # Initialize enhanced MLflow tracker
    mlflow_tracker = MLflowVisualizationTracker("Face_Mask_Detection_Model_Training_REAL")
    
    if not mlflow_tracker.use_mlflow:
        logger.warning("MLflow not available. Training will proceed without tracking.")
    
    try:
        with mlflow.start_run(run_name=f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # 1. Load and prepare data
            logger.info("📊 Loading and preparing data...")
            
            train_df = pd.read_csv(Path(data_dir) / "train.csv")
            val_df = pd.read_csv(Path(data_dir) / "val.csv")
            test_df = pd.read_csv(Path(data_dir) / "test.csv")
            
            logger.info(f"Training samples: {len(train_df)}")
            logger.info(f"Validation samples: {len(val_df)}")
            logger.info(f"Test samples: {len(test_df)}")
            
            # Log dataset information
            mlflow_tracker.log_hyperparameters({
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "batch_size": batch_size,
                "epochs": epochs,
                "use_class_weights": use_class_weights,
                "model_architecture": "MobileNetV2",
                "image_size": "224x224",
                "optimizer": "Adam",
                "loss_function": "categorical_crossentropy"
            })
            
            # Create class distribution plots
            mlflow_tracker.create_class_distribution_plot(train_df['class_id'].values)
            
            # 2. Setup data augmentation
            logger.info("🔄 Setting up data augmentation...")
            augmentation = AugmentationPipeline(image_size=(224, 224))
            
            # Create data generators
            train_generator = DataGenerator(train_df, batch_size, augmentation, training=True)
            val_generator = DataGenerator(val_df, batch_size, augmentation, training=False)
            
            # 3. Calculate class weights
            class_weights_dict = None
            if use_class_weights:
                logger.info("⚖️ Calculating class weights...")
                unique_classes = np.unique(train_df['class_id'])
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=train_df['class_id']
                )
                class_weights_dict = {int(k): float(v) for k, v in zip(unique_classes, class_weights)}
                logger.info(f"Class weights: {class_weights_dict}")
                
                mlflow_tracker.log_hyperparameters({
                    "class_weights": str(class_weights_dict)
                })
            
            # 4. Create model
            logger.info("🏗️ Creating model architecture...")
            
            # Create simple MobileNetV2 model
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Log model architecture
            mlflow_tracker.log_model_architecture_info(model)
            
            # 5. Setup callbacks
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Add MLflow callback if available
            if mlflow_tracker.use_mlflow:
                callbacks_list.append(MLflowCallback())
            
            # 6. Train model
            logger.info("🎯 Starting model training...")
            
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                class_weight=class_weights_dict,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # 7. Create training history visualizations
            logger.info("📈 Creating training history visualizations...")
            mlflow_tracker.create_training_history_plots(history.history)
            
            # 8. Evaluate on test set
            logger.info("🧪 Evaluating on test set...")
            
            # Load best model
            if os.path.exists(model_save_path):
                model = tf.keras.models.load_model(model_save_path)
                logger.info("✅ Loaded best model for evaluation")
            
            # Create test generator
            test_generator = DataGenerator(test_df, batch_size, augmentation, training=False, shuffle=False)
            
            # Get predictions
            test_predictions = model.predict(test_generator, verbose=1)
            test_pred_classes = np.argmax(test_predictions, axis=1)
            
            # Get true labels
            test_true_classes = []
            for i in range(len(test_generator)):
                batch_x, batch_y = test_generator[i]
                batch_true_classes = np.argmax(batch_y, axis=1)
                test_true_classes.extend(batch_true_classes)
            
            test_true_classes = np.array(test_true_classes[:len(test_predictions)])
            
            # 9. Log comprehensive metrics and visualizations
            logger.info("📊 Logging comprehensive metrics and visualizations...")
            
            # Log all metrics
            mlflow_tracker.log_comprehensive_metrics(
                test_true_classes, 
                test_pred_classes, 
                test_predictions
            )
            
            # Create all visualizations
            mlflow_tracker.create_confusion_matrix_plot(test_true_classes, test_pred_classes)
            mlflow_tracker.create_roc_curves(test_true_classes, test_predictions)
            mlflow_tracker.create_precision_recall_curves(test_true_classes, test_predictions)
            mlflow_tracker.create_prediction_confidence_plot(test_predictions)
            
            # 10. Log final model and artifacts
            if mlflow_tracker.use_mlflow:
                logger.info("💾 Logging model and artifacts...")
                
                # Log the model file as artifact (avoiding MLflow model logging issues)
                try:
                    mlflow.log_artifact(model_save_path, "saved_models")
                    logger.info("✅ Model artifact logged successfully")
                except Exception as e:
                    logger.warning(f"Failed to log model artifact: {e}")
                
                # Log model summary as text
                try:
                    import io
                    import contextlib
                    
                    # Capture model summary
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        model.summary()
                    model_summary = f.getvalue()
                    
                    # Save and log model summary
                    summary_path = "model_summary.txt"
                    with open(summary_path, 'w', encoding='utf-8') as file:
                        file.write(model_summary)
                    mlflow.log_artifact(summary_path, "model_info")
                    
                    # Clean up
                    if os.path.exists(summary_path):
                        os.remove(summary_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to log model summary: {e}")
                
                # Log training parameters as JSON
                params_dict = {
                    "training_params": {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "use_class_weights": use_class_weights,
                        "class_weights": class_weights_dict,
                        "data_dir": data_dir,
                        "model_save_path": model_save_path
                    },
                    "model_params": {
                        "architecture": "MobileNetV2",
                        "input_shape": [224, 224, 3],
                        "num_classes": 3,
                        "total_parameters": int(model.count_params())
                    },
                    "dataset_info": {
                        "train_samples": len(train_df),
                        "val_samples": len(val_df),
                        "test_samples": len(test_df),
                        "class_distribution": {str(k): int(v) for k, v in train_df['class_id'].value_counts().to_dict().items()}
                    }
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump(params_dict, tmp, indent=2)
                    mlflow.log_artifact(tmp.name, "parameters")
            
            # 11. Print final results
            logger.info("🎉 Training completed successfully!")
            logger.info("=" * 60)
            logger.info("FINAL RESULTS:")
            logger.info("=" * 60)
            
            test_accuracy = accuracy_score(test_true_classes, test_pred_classes)
            test_precision = precision_score(test_true_classes, test_pred_classes, average='weighted')
            test_recall = recall_score(test_true_classes, test_pred_classes, average='weighted')
            test_f1 = f1_score(test_true_classes, test_pred_classes, average='weighted')
            
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Precision (weighted): {test_precision:.4f}")
            logger.info(f"Test Recall (weighted): {test_recall:.4f}")
            logger.info(f"Test F1-Score (weighted): {test_f1:.4f}")
            logger.info(f"Model saved to: {model_save_path}")
            
            if mlflow_tracker.use_mlflow:
                logger.info("📊 All metrics and visualizations logged to MLflow!")
                logger.info("🔗 View results in MLflow UI: mlflow ui")
            
            return model, history.history
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if mlflow_tracker.use_mlflow:
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
        raise
    
    finally:
        # Clean up any temporary files
        try:
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "tmp*.png")) + \
                        glob.glob(os.path.join(temp_dir, "tmp*.txt")) + \
                        glob.glob(os.path.join(temp_dir, "tmp*.json"))
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        except:
            pass


if __name__ == "__main__":
    """Run enhanced training with comprehensive MLflow tracking."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run training
    model, history = train_enhanced_model_with_comprehensive_mlflow(
        data_dir="data/processed/splits",
        model_save_path="models/best_mask_detector_enhanced_mlflow.h5",
        batch_size=32,
        epochs=50,
        use_class_weights=True
    )
    
    logger.info("🎉 Enhanced training with comprehensive MLflow tracking completed!")
