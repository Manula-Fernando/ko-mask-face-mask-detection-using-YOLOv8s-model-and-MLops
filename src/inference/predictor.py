#!/usr/bin/env python3
"""
Face Mask Predictor - Medical Grade Detection System
Real-time face mask detection using YOLO models with medical accuracy standards.
"""

import os
import sys
from pathlib import Path

# --- Add this block at the top, before any src imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# ---------------------------------------------------------
from src.common.logger import get_logger
from src.common.utils import FileUtils
# --- End of block ---

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed. Please install: pip install ultralytics")
    YOLO = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceMaskPredictor:
    """Medical-grade face mask detection predictor using YOLO models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the face mask predictor.
        
        Args:
            model_path: Path to the YOLO model file. If None, auto-detects.
        """
        self.model = None
        self.model_path = model_path
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.45
        
        # Load model at initialization
        self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load YOLO model from specified or auto-detected path.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if YOLO is None:
            logger.error("YOLO not available - ultralytics not installed")
            return False
            
        # Define possible model paths
        project_root = Path(__file__).parent.parent.parent
        possible_paths = [
            # User provided path
            Path(model_path) if model_path else None,
            # Standard model locations
            project_root / "models" / "best.pt",
            project_root / "models" / "yolov8_real_face_mask_detection" / "weights" / "best.pt",
            project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            project_root / "models" / "train" / "weights" / "best.pt",
        ]
        
        # Remove None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        # Try to load model from each path
        for path in possible_paths:
            if path.exists():
                try:
                    self.model = YOLO(str(path))
                    self.model_path = str(path)
                    logger.info(f"✅ YOLO model loaded from: {path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")
                    continue
        
        # If no custom model found, try pretrained fallback
        try:
            logger.warning("No trained model found, using YOLOv8n pretrained model")
            self.model = YOLO('yolov8n.pt')
            self.model_path = 'yolov8n.pt (pretrained)'
            return True
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict face mask status for given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            dict: Prediction results with confidence and class
        """
        if self.model is None:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Run YOLO inference
            results = self.model(image, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get the detection with highest confidence
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                # Find best detection
                best_idx = confidences.argmax()
                best_confidence = float(confidences[best_idx])
                best_class_idx = int(classes[best_idx])
                
                # Map class index to name
                if best_class_idx < len(self.class_names):
                    prediction = self.class_names[best_class_idx]
                else:
                    prediction = f'class_{best_class_idx}'
                
                return {
                    'prediction': prediction,
                    'confidence': best_confidence,
                    'class_id': best_class_idx,
                    'bbox': boxes.xyxy[best_idx].cpu().numpy().tolist()
                }
            else:
                return {
                    'prediction': 'no_detection',
                    'confidence': 0.0,
                    'error': 'No face detected'
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict face mask status for multiple images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.predict(image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        return results
    
    def detect_all_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect all faces in image and classify each one.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            list: List of all detections with bounding boxes
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(image, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_idx = int(boxes.cls[i].cpu().numpy())
                    
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_idx}'
                    
                    detections.append({
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_idx
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'model_type': 'YOLOv8' if self.model else None
        }

# Compatibility functions for backward compatibility
def load_model(model_path: Optional[str] = None) -> FaceMaskPredictor:
    """Load face mask detection model.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        FaceMaskPredictor: Initialized predictor instance
    """
    return FaceMaskPredictor(model_path)

def predict_image(image: np.ndarray, predictor: FaceMaskPredictor) -> Dict:
    """Predict face mask for single image.
    
    Args:
        image: Input image
        predictor: Predictor instance
        
    Returns:
        dict: Prediction results
    """
    return predictor.predict(image)

if __name__ == "__main__":
    # Test the predictor
    predictor = FaceMaskPredictor()
    print("Model info:", predictor.get_model_info())
