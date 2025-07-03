# Face Mask Detection - Production Prediction with Real-time Capabilities
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class FaceMaskPredictor:
    """Production face mask detection predictor with comprehensive capabilities."""
    
    CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model with comprehensive error handling."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found: {self.model_path}")
            return
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info(f"✅ Model loaded successfully: {self.model_path}")
            
            # Log model info
            total_params = self.model.count_params()
            self.logger.info(f"📊 Model parameters: {total_params:,}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
    
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
        """Make prediction on single image with comprehensive output."""
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
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.CLASSES[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
            # All class probabilities
            all_probabilities = {
                self.CLASSES[i]: float(predictions[i]) for i in range(len(self.CLASSES))
            }
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'timestamp': datetime.now().isoformat()
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
