"""
Prediction module for face mask detection.
This module handles model loading and inference.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import yaml
import logging
from typing import Tuple, List, Dict, Any
import mlflow
import mlflow.tensorflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaskPredictor:
    """Face mask detection predictor."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the predictor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        self.mlflow_config = self.config['mlflow']
        
        self.model = None
        self.class_names = ['No Mask', 'Mask']
        
        logger.info("MaskPredictor initialized")
    
    def load_model(self, model_path: str = None) -> tf.keras.Model:
        """Load the trained model."""
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            # Load from MLflow model registry
            logger.info("Loading model from MLflow registry")
            model_name = "face_mask_detector"
            stage = self.config['deployment']['model_stage']
            
            try:
                model_uri = f"models:/{model_name}/{stage}"
                self.model = mlflow.tensorflow.load_model(model_uri)
                logger.info(f"Loaded model {model_name} from {stage} stage")
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {str(e)}")
                # Fallback to latest model file
                models_dir = "models"
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                    if model_files:
                        latest_model = max(model_files)
                        model_path = os.path.join(models_dir, latest_model)
                        logger.info(f"Loading fallback model: {model_path}")
                        self.model = tf.keras.models.load_model(model_path)
                    else:
                        raise ValueError("No trained model found!")
                else:
                    raise ValueError("Models directory not found!")
        
        return self.model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction."""
        # Resize image to model input size
        target_size = tuple(self.model_config['input_shape'][:2])
        image_resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """Make prediction on a single image."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get probabilities for both classes
        probabilities = {
            self.class_names[i]: float(predictions[0][i]) 
            for i in range(len(self.class_names))
        }
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities
        }
        
        return result
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Make predictions on a batch of images."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        
        return results
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """Make prediction on an image file."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.predict_single(image)
    
    def detect_faces_and_predict(self, image: np.ndarray, 
                                face_cascade_path: str = "models/haarcascade_frontalface_default.xml") -> List[Dict[str, Any]]:
        """Detect faces in image and predict mask for each face."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load face cascade classifier
        if not os.path.exists(face_cascade_path):
            logger.warning(f"Face cascade file not found: {face_cascade_path}")
            # Return prediction for entire image
            return [self.predict_single(image)]
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            # Make prediction on face
            prediction = self.predict_single(face_region)
            
            # Add bounding box coordinates
            prediction['bbox'] = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            
            results.append(prediction)
        
        # If no faces detected, analyze entire image
        if len(results) == 0:
            logger.info("No faces detected, analyzing entire image")
            results = [self.predict_single(image)]
        
        return results

def main():
    """Example usage of the predictor."""
    # Initialize predictor
    predictor = MaskPredictor()
    
    # Load model
    predictor.load_model()
    
    # Example: predict on a sample image
    # image_path = "path/to/your/image.jpg"
    # result = predictor.predict_from_path(image_path)
    # print(f"Prediction: {result}")
    
    logger.info("Predictor ready for inference")

if __name__ == "__main__":
    main()
