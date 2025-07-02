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
import json
import argparse
from typing import Tuple, List, Dict, Any
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

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
        self.class_names = self.config['data']['classes']
        
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

    def evaluate_model(self, test_dir: str = None) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if test_dir is None:
            test_dir = self.config['paths']['test_dir']
        
        logger.info(f"Evaluating model on test data: {test_dir}")
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=tuple(self.model_config['input_shape'][:2]),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Make predictions
        predictions = self.model.predict(test_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true classes
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # Calculate metrics
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Calculate overall accuracy
        accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        
        evaluation_results = {
            'accuracy': float(accuracy),
            'test_samples': len(true_classes),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels
        }
        
        # Save evaluation results
        os.makedirs('models', exist_ok=True)
        eval_path = os.path.join('models', 'evaluation_metrics.json')
        
        # Convert numpy types to native Python types for JSON serialization
        eval_results_json = json.loads(json.dumps(evaluation_results, default=str))
        
        with open(eval_path, 'w') as f:
            json.dump(eval_results_json, f, indent=2)
        
        # Create evaluation plots data
        plots_data = {
            'accuracy': float(accuracy),
            'precision_by_class': {},
            'recall_by_class': {},
            'f1_by_class': {}
        }
        
        for class_name in class_labels:
            if class_name in report:
                plots_data['precision_by_class'][class_name] = report[class_name]['precision']
                plots_data['recall_by_class'][class_name] = report[class_name]['recall']
                plots_data['f1_by_class'][class_name] = report[class_name]['f1-score']
        
        plots_path = os.path.join('models', 'evaluation_plots.json')
        with open(plots_path, 'w') as f:
            json.dump(plots_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        logger.info(f"Evaluation plots saved to {plots_path}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return evaluation_results

# Real-time prediction functions for Phase 3 MLOps Implementation

def get_model(model_path):
    """Loads and returns the trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def predict_frame(frame, face_cascade, model):
    """Processes a single frame, detects faces, and predicts masks."""
    if frame is None or face_cascade is None or model is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Define labels and colors for Sri Lankan context
    labels = ["With Mask", "Mask Worn Incorrectly", "Without Mask"]
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]  # Green, Orange, Red

    for (x, y, w, h) in faces_detected:
        try:
            face_frame = frame[y:y+h, x:x+w]
            face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame_resized = cv2.resize(face_frame_rgb, (224, 224))
            face_frame_array = np.expand_dims(face_frame_resized, axis=0)
            face_frame_preprocessed = face_frame_array / 255.0  # Normalize to [0,1]

            preds = model.predict(face_frame_preprocessed, verbose=0)[0]
            
            # Get the class with the highest probability
            prediction_idx = np.argmax(preds)
            confidence = preds[prediction_idx]

            label_text = f"{labels[prediction_idx]}: {confidence:.2f}"
            color = colors[prediction_idx]
            
            # Displaying the labels and bounding boxes
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
        except Exception as e:
            logger.error(f"Error processing face detection: {e}")
            # Draw a simple rectangle if processing fails
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
    return frame

def main():
    """Main function with command line argument handling."""
    parser = argparse.ArgumentParser(description='Face Mask Detection Prediction')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate model on test data')
    parser.add_argument('--image', type=str, 
                       help='Path to image for prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MaskPredictor()
    
    # Load model
    predictor.load_model()
    
    if args.evaluate:
        # Run evaluation
        logger.info("Running model evaluation...")
        results = predictor.evaluate_model()
        logger.info(f"Evaluation completed. Test Accuracy: {results['accuracy']:.4f}")
        
    elif args.image:
        # Predict on single image
        result = predictor.predict_from_path(args.image)
        print(f"Prediction: {result}")
        
    else:
        logger.info("Predictor ready for inference")
        logger.info("Use --evaluate to run evaluation or --image <path> to predict on an image")

if __name__ == "__main__":
    main()
