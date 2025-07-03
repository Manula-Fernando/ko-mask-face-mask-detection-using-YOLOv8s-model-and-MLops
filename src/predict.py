# Face Mask Detection - Production Prediction with Enhanced MLflow Tracking
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import mlflow
import mlflow.tensorflow
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FaceMaskPredictor:
    """Production face mask detection predictor with comprehensive MLflow tracking."""
    
    CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize prediction tracking
        self.prediction_history = []
        self.performance_metrics = {
            'total_predictions': 0,
            'inference_times': [],
            'confidence_scores': [],
            'class_predictions': defaultdict(int)
        }
        
        # Class-wise confidence tracking
        self.class_confidence_history = {
            'with_mask': [],
            'without_mask': [],
            'mask_weared_incorrect': []
        }
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Note: Model loading is now manual - call load_model() when needed
    
    def setup_mlflow(self):
        """Setup MLflow tracking for predictions."""
        mlflow.set_experiment("Face_Mask_Detection_Predictions")
        
    def log_prediction_session(self, session_info: Dict, run_name: str = "prediction_session"):
        """Log comprehensive prediction session information."""
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log session parameters
            for key, value in session_info.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            
            # Log performance metrics
            if self.performance_metrics['total_predictions'] > 0:
                avg_inference_time = np.mean(self.performance_metrics['inference_times'])
                avg_confidence = np.mean(self.performance_metrics['confidence_scores'])
                
                mlflow.log_metric("total_predictions", self.performance_metrics['total_predictions'])
                mlflow.log_metric("avg_inference_time_ms", avg_inference_time * 1000)
                mlflow.log_metric("avg_confidence", avg_confidence)
                mlflow.log_metric("min_confidence", np.min(self.performance_metrics['confidence_scores']))
                mlflow.log_metric("max_confidence", np.max(self.performance_metrics['confidence_scores']))
                
                # Log class distribution
                for class_name, count in self.performance_metrics['class_predictions'].items():
                    mlflow.log_metric(f"predictions_{class_name}", count)
                    mlflow.log_metric(f"predictions_{class_name}_percentage", 
                                    (count / self.performance_metrics['total_predictions']) * 100)
                
                # Create and log prediction analytics
                self._create_prediction_analytics()
                
            self.logger.info("✅ Prediction session logged to MLflow")
    
    def _create_prediction_analytics(self):
        """Create comprehensive prediction analytics visualizations."""
        # Confidence distribution
        if self.performance_metrics['confidence_scores']:
            plt.figure(figsize=(15, 10))
            
            # Confidence histogram
            plt.subplot(2, 3, 1)
            plt.hist(self.performance_metrics['confidence_scores'], bins=50, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            plt.axvline(np.mean(self.performance_metrics['confidence_scores']), 
                       color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.performance_metrics["confidence_scores"]):.3f}')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Prediction Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Inference time distribution
            plt.subplot(2, 3, 2)
            inference_times_ms = np.array(self.performance_metrics['inference_times']) * 1000
            plt.hist(inference_times_ms, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.axvline(np.mean(inference_times_ms), color='red', linestyle='--',
                       label=f'Mean: {np.mean(inference_times_ms):.1f}ms')
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Inference Time Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Class prediction distribution
            plt.subplot(2, 3, 3)
            class_names = list(self.performance_metrics['class_predictions'].keys())
            class_counts = list(self.performance_metrics['class_predictions'].values())
            colors = ['green', 'red', 'orange'][:len(class_names)]
            
            plt.bar(class_names, class_counts, color=colors)
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')
            plt.title('Class Prediction Distribution')
            plt.xticks(rotation=45)
            
            # Add count labels on bars
            for i, count in enumerate(class_counts):
                plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
            
            # Confidence by prediction class
            plt.subplot(2, 3, 4)
            for i, (class_name, predictions) in enumerate(self.class_confidence_history.items()):
                if predictions:
                    plt.hist(predictions, alpha=0.6, label=class_name, bins=20, color=colors[i % len(colors)])
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution by Class')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Time series of confidence scores
            plt.subplot(2, 3, 5)
            plt.plot(self.performance_metrics['confidence_scores'], alpha=0.7, linewidth=1)
            plt.xlabel('Prediction Number')
            plt.ylabel('Confidence Score')
            plt.title('Confidence Scores Over Time')
            plt.grid(True, alpha=0.3)
            
            # Performance summary
            plt.subplot(2, 3, 6)
            summary_metrics = [
                f"Total Predictions: {self.performance_metrics['total_predictions']}",
                f"Avg Confidence: {np.mean(self.performance_metrics['confidence_scores']):.3f}",
                f"Avg Inference Time: {np.mean(inference_times_ms):.1f}ms",
                f"High Confidence (>0.9): {len([c for c in self.performance_metrics['confidence_scores'] if c > 0.9])}",
                f"Low Confidence (<0.7): {len([c for c in self.performance_metrics['confidence_scores'] if c < 0.7])}"
            ]
            
            plt.text(0.05, 0.95, '\n'.join(summary_metrics), transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.axis('off')
            plt.title('Prediction Session Summary')
            
            plt.tight_layout()
            plt.savefig("prediction_analytics.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("prediction_analytics.png")
            plt.close()
        
    def load_model(self):
        """Load the trained model with comprehensive error handling and MLflow tracking."""
        with mlflow.start_run(run_name="model_loading", nested=True):
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model not found: {self.model_path}")
                mlflow.log_metric("model_load_success", 0)
                return
            
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info(f"Model loaded successfully: {self.model_path}")
                
                # Log model info
                total_params = self.model.count_params()
                self.logger.info(f"Model parameters: {total_params:,}")
                
                # Log model metadata to MLflow
                mlflow.log_param("model_path", self.model_path)
                mlflow.log_param("model_name", Path(self.model_path).name)
                mlflow.log_param("classes", self.CLASSES)
                mlflow.log_metric("total_parameters", total_params)
                mlflow.log_metric("model_load_success", 1)
                
                # Log model architecture summary
                model_summary = []
                self.model.summary(print_fn=lambda x: model_summary.append(x))
                summary_text = '\n'.join(model_summary)
                # Write to file first with proper encoding
                with open("model_architecture.txt", 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                mlflow.log_artifact("model_architecture.txt")
                
                # Log model as artifact
                mlflow.tensorflow.log_model(self.model, "model")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                mlflow.log_metric("model_load_success", 0)
                # Write error to file first with proper encoding
                with open("model_load_error.txt", 'w', encoding='utf-8') as f:
                    f.write(str(e))
                mlflow.log_artifact("model_load_error.txt")
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for prediction with enhanced normalization."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, (224, 224))
            
            # Normalize using ImageNet statistics (same as training)
            image = image.astype(np.float32) / 255.0
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {image_path}: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess video frame for real-time prediction."""
        try:
            # Frame should already be in RGB format from OpenCV webcam
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError("Invalid frame shape")
            
            # Resize to model input size
            frame = cv2.resize(frame, (224, 224))
            
            # Normalize using ImageNet statistics
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            # Add batch dimension
            frame = np.expand_dims(frame, axis=0)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {e}")
            return None
    
    def predict(self, image_path: str) -> Dict:
        """Make prediction on single image with comprehensive output and MLflow tracking."""
        start_time = datetime.now()
        
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return {
                'error': 'Failed to preprocess image',
                'prediction': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }
        
        try:
            # Measure inference time
            inference_start = datetime.now()
            predictions = self.model.predict(processed_image, verbose=0)[0]
            inference_time = (datetime.now() - inference_start).total_seconds()
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.CLASSES[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
            # All class probabilities
            all_probabilities = {
                self.CLASSES[i]: float(predictions[i]) for i in range(len(self.CLASSES))
            }
            
            # Update performance tracking
            self.performance_metrics['total_predictions'] += 1
            self.performance_metrics['inference_times'].append(inference_time)
            self.performance_metrics['confidence_scores'].append(confidence)
            self.performance_metrics['class_predictions'][predicted_class] += 1
            
            # Track class-wise confidence
            self.class_confidence_history[predicted_class].append(confidence)
            
            # Store prediction history
            prediction_record = {
                'image_path': image_path,
                'prediction': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'timestamp': start_time.isoformat(),
                'all_probabilities': all_probabilities
            }
            self.prediction_history.append(prediction_record)
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'inference_time_ms': inference_time * 1000,
                'timestamp': start_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }
    
    def predict_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """Make prediction on video frame - optimized for real-time."""
        if self.model is None:
            return "unknown", 0.0
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        if processed_frame is None:
            return "error", 0.0
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_frame, verbose=0)[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.CLASSES[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
            return predicted_class, confidence
            
        except Exception as e:
            self.logger.error(f"Frame prediction failed: {e}")
            return "error", 0.0
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions on multiple images."""
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            return {
                'model_path': self.model_path,
                'total_parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'classes': self.CLASSES,
                'num_classes': len(self.CLASSES)
            }
        except Exception as e:
            return {'error': str(e)}


    def log_analytics(self):
        """Generate and log prediction analytics visualization."""
        if not self.performance_metrics['confidence_scores']:
            self.logger.warning("No prediction data available for analytics")
            return
            
        # Create analytics visualization
        plt.figure(figsize=(18, 12))
        
        # Confidence score distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.performance_metrics['confidence_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(self.performance_metrics['confidence_scores']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.performance_metrics["confidence_scores"]):.3f}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Inference time distribution
        plt.subplot(2, 3, 2)
        inference_times_ms = [t * 1000 for t in self.performance_metrics['inference_times']]
        plt.hist(inference_times_ms, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(inference_times_ms), color='red', linestyle='--',
                   label=f'Mean: {np.mean(inference_times_ms):.1f}ms')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Class prediction distribution
        plt.subplot(2, 3, 3)
        class_names = list(self.performance_metrics['class_predictions'].keys())
        class_counts = list(self.performance_metrics['class_predictions'].values())
        colors = ['green', 'red', 'orange'][:len(class_names)]
        
        plt.bar(class_names, class_counts, color=colors)
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Class Prediction Distribution')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(class_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # Confidence by prediction class
        plt.subplot(2, 3, 4)
        for i, (class_name, predictions) in enumerate(self.class_confidence_history.items()):
            if predictions:
                plt.hist(predictions, alpha=0.6, label=class_name, bins=20, color=colors[i % len(colors)])
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Time series of confidence scores
        plt.subplot(2, 3, 5)
        plt.plot(self.performance_metrics['confidence_scores'], alpha=0.7, linewidth=1)
        plt.xlabel('Prediction Number')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Scores Over Time')
        plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(2, 3, 6)
        summary_metrics = [
            f"Total Predictions: {self.performance_metrics['total_predictions']}",
            f"Avg Confidence: {np.mean(self.performance_metrics['confidence_scores']):.3f}",
            f"Avg Inference Time: {np.mean(inference_times_ms):.1f}ms",
            f"High Confidence (>0.9): {len([c for c in self.performance_metrics['confidence_scores'] if c > 0.9])}",
            f"Low Confidence (<0.7): {len([c for c in self.performance_metrics['confidence_scores'] if c < 0.7])}"
        ]
        
        plt.text(0.05, 0.95, '\n'.join(summary_metrics), transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        plt.title('Prediction Session Summary')
        
        plt.tight_layout()
        plt.savefig("prediction_analytics.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("prediction_analytics.png")
        plt.close()
        
        self.logger.info("Prediction analytics visualization logged to MLflow")


class BatchProcessor:
    """Batch processing utility for multiple images."""
    
    def __init__(self, predictor: FaceMaskPredictor):
        self.predictor = predictor
        self.logger = logging.getLogger(__name__)
    
    def process_directory(self, directory_path: str, output_file: str = None) -> List[Dict]:
        """Process all images in a directory."""
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(file) for file in directory.rglob('*') 
            if file.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            self.logger.warning(f"No images found in: {directory_path}")
            return []
        
        self.logger.info(f"Processing {len(image_paths)} images...")
        
        # Process all images
        results = self.predictor.batch_predict(image_paths)
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save batch processing results to file."""
        import json
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


# Utility functions for easy import
def load_predictor(model_path: str = "models/best_mask_detector_imbalance_optimized.h5") -> FaceMaskPredictor:
    """Load the face mask predictor with default model path."""
    return FaceMaskPredictor(model_path)

def predict_image(image_path: str, model_path: str = "models/best_mask_detector_imbalance_optimized.h5") -> Dict:
    """Quick prediction on a single image."""
    predictor = load_predictor(model_path)
    return predictor.predict(image_path)

def predict_images(image_paths: List[str], model_path: str = "models/best_mask_detector_imbalance_optimized.h5") -> List[Dict]:
    """Quick batch prediction on multiple images."""
    predictor = load_predictor(model_path)
    return predictor.batch_predict(image_paths)
