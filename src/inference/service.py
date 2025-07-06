"""
Face Mask Detection Inference Service

This module provides a high-level service for face mask detection inference,
integrating model loading, preprocessing, prediction, and monitoring.
"""

import os
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import json
import mlflow
from ultralytics import YOLO
from mlflow.tracking import MlflowClient

from common.logger import get_logger
from common.utils import load_config

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results"""
    image_path: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    execution_time: float
    model_version: str
    timestamp: float


class FaceMaskInferenceService:
    """High-level service for face mask detection inference"""
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the inference service
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model: Optional[YOLO] = None
        self.model_version: Optional[str] = None
        self.prediction_stats: Dict[str, Any] = {
            'total_predictions': 0,
            'with_mask': 0,
            'without_mask': 0,
            'avg_confidence': 0.0,
            'avg_inference_time': 0.0
        }
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model, with MLflow fallback if needed."""
        try:
            model_path = self.config['model']['path']
            if not os.path.exists(model_path):
                self._load_from_mlflow()
            else:
                self.model = YOLO(model_path)
                self.model_version = self._get_model_version(model_path)
            logger.info(f"Model loaded successfully: {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_from_mlflow(self) -> None:
        """Load YOLO model weights from MLflow registry and initialize YOLO."""
        try:
            model_name = self.config['model']['name']
            model_stage = self.config['model'].get('stage', 'Production')
            client = MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[model_stage])
            if not versions:
                raise RuntimeError(f"No model found in MLflow for {model_name} at stage {model_stage}")
            model_version = versions[0]
            model_uri = f"models:/{model_name}/{model_stage}"
            # Download artifact to a temp dir, then load with YOLO
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            # Find YOLO weights file (assume .pt)
            pt_files = list(Path(local_path).rglob('*.pt'))
            if not pt_files:
                raise RuntimeError(f"No .pt file found in MLflow artifact at {local_path}")
            self.model = YOLO(str(pt_files[0]))
            self.model_version = f"{model_name}:{model_version.version}"
            logger.info(f"Model loaded from MLflow: {self.model_version}")
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")
            self._load_default_model()

    def _load_default_model(self) -> None:
        """Load a default YOLOv8 model as fallback."""
        try:
            self.model = YOLO('yolov8n.pt')  # Nano model as fallback
            self.model_version = "yolov8n-fallback"
            logger.info("Loaded fallback YOLOv8 model")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise

    def _get_model_version(self, model_path: str) -> str:
        """Extract model version from path or metadata."""
        try:
            filename = Path(model_path).stem
            if 'v' in filename:
                return filename
            mtime = os.path.getmtime(model_path)
            return f"model-{int(mtime)}"
        except Exception:
            return "unknown"

    def predict_single(self, image_path: str, save_results: bool = True) -> InferenceResult:
        """Perform inference on a single image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save prediction results
        Returns:
            InferenceResult object containing predictions and metadata
        """
        start_time = time.time()
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            results = self.model(image_path, conf=self.config['inference']['confidence_threshold'])
            predictions: List[Dict[str, Any]] = []
            confidence_scores: List[float] = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = 'with_mask' if cls == 0 else 'without_mask'
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        prediction = {
                            'class': label,
                            'confidence': conf,
                            'bbox': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            }
                        }
                        predictions.append(prediction)
                        confidence_scores.append(conf)
                        if label == 'with_mask':
                            self.prediction_stats['with_mask'] += 1
                        else:
                            self.prediction_stats['without_mask'] += 1
            execution_time = time.time() - start_time
            result = InferenceResult(
                image_path=image_path,
                predictions=predictions,
                confidence_scores=confidence_scores,
                execution_time=execution_time,
                model_version=self.model_version or "unknown",
                timestamp=time.time()
            )
            self._update_stats(result)
            if save_results:
                self._save_prediction_result(result, image)
            logger.info(f"Inference completed for {image_path}: {len(predictions)} detections")
            return result
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            raise

    def predict_batch(self, image_paths: List[str], save_results: bool = True) -> List[InferenceResult]:
        """Perform inference on multiple images.
        
        Args:
            image_paths: List of paths to input images
            save_results: Whether to save prediction results
        Returns:
            List of InferenceResult objects
        """
        results: List[InferenceResult] = []
        logger.info(f"Starting batch inference on {len(image_paths)} images")
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path, save_results)
                results.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        logger.info(f"Batch inference completed: {len(results)}/{len(image_paths)} successful")
        return results

    def predict_realtime(self, source: int = 0) -> None:
        """Perform real-time inference using webcam.
        
        Args:
            source: Camera source index (0 for default camera)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        logger.info("Starting real-time inference (press 'q' to quit)")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, conf=self.config['inference']['confidence_threshold'])
                annotated_frame = results[0].plot()
                cv2.imshow('Face Mask Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Real-time inference stopped")

    def _update_stats(self, result: InferenceResult) -> None:
        """Update prediction statistics."""
        self.prediction_stats['total_predictions'] += 1
        if result.confidence_scores:
            avg_conf = np.mean(result.confidence_scores)
            total = self.prediction_stats['total_predictions']
            current_avg = self.prediction_stats['avg_confidence']
            self.prediction_stats['avg_confidence'] = (
                (current_avg * (total - 1) + avg_conf) / total
            )
        total = self.prediction_stats['total_predictions']
        current_avg = self.prediction_stats['avg_inference_time']
        self.prediction_stats['avg_inference_time'] = (
            (current_avg * (total - 1) + result.execution_time) / total
        )

    def _save_prediction_result(self, result: InferenceResult, image: np.ndarray) -> None:
        """Save prediction results to disk."""
        try:
            output_dir = Path(self.config['inference']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(result.timestamp * 1000)
            base_name = f"{timestamp}_{Path(result.image_path).stem}"
            if len(result.predictions) > 0:
                annotated_image = image.copy()
                for pred in result.predictions:
                    bbox = pred['bbox']
                    label = pred['class']
                    conf = pred['confidence']
                    color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
                    cv2.rectangle(
                        annotated_image,
                        (bbox['x1'], bbox['y1']),
                        (bbox['x2'], bbox['y2']),
                        color, 2
                    )
                    label_text = f"{label}: {conf:.3f}"
                    cv2.putText(
                        annotated_image, label_text,
                        (bbox['x1'], bbox['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                img_path = output_dir / f"{base_name}_annotated.jpg"
                cv2.imwrite(str(img_path), annotated_image)
            metadata = {
                'image_path': result.image_path,
                'predictions': result.predictions,
                'model_version': result.model_version,
                'execution_time': result.execution_time,
                'timestamp': result.timestamp
            }
            json_path = output_dir / f"{base_name}_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prediction result: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current prediction statistics."""
        return self.prediction_stats.copy()

    def reset_statistics(self) -> None:
        """Reset prediction statistics."""
        self.prediction_stats = {
            'total_predictions': 0,
            'with_mask': 0,
            'without_mask': 0,
            'avg_confidence': 0.0,
            'avg_inference_time': 0.0
        }
        logger.info("Prediction statistics reset")


def main() -> None:
    """CLI entry point for inference service."""
    import argparse
    parser = argparse.ArgumentParser(description='Face Mask Detection Inference Service')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--batch', type=str, help='Path to directory containing images')
    parser.add_argument('--realtime', action='store_true', help='Start real-time inference')
    parser.add_argument('--camera', type=int, default=0, help='Camera source index')
    args = parser.parse_args()
    service = FaceMaskInferenceService(args.config)
    if args.image:
        result = service.predict_single(args.image)
        print(f"Predictions: {len(result.predictions)}")
        print(f"Execution time: {result.execution_time:.3f}s")
    elif args.batch:
        batch_dir = Path(args.batch)
        # Support more image types
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_paths.extend(batch_dir.glob(ext))
        results = service.predict_batch([str(p) for p in image_paths])
        print(f"Processed {len(results)} images")
    elif args.realtime:
        service.predict_realtime(args.camera)
    else:
        print("Please specify --image, --batch, or --realtime")
        return
    stats = service.get_statistics()
    print(f"\nPrediction Statistics:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"With mask: {stats['with_mask']}")
    print(f"Without mask: {stats['without_mask']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Average inference time: {stats['avg_inference_time']:.3f}s")


if __name__ == "__main__":
    main()
