"""
Model definition and management for Face Mask Detection
Provides YOLOv8 wrapper, MLflow integration, and model management utilities.
"""

import os
import sys
from pathlib import Path

# --- Add this block at the top, before any src imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# ---------------------------------------------------------

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import numpy as np
import shutil  # <-- Needed for copying weights

from src.common.logger import get_logger
from src.common.utils import FileUtils

logger = get_logger("training.model")

class FaceMaskYOLO:
    """Face Mask Detection model wrapper using YOLOv8."""
    def __init__(self, model_name: str = "yolov8s.pt", num_classes: int = 3) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing model: {model_name} on device: {self.device}")

    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """Load YOLO model from weights or pretrained."""
        try:
            if weights_path and Path(weights_path).exists():
                self.model = YOLO(weights_path)
                logger.info(f"Loaded model from: {weights_path}")
            else:
                self.model = YOLO(self.model_name)
                logger.info(f"Loaded pretrained model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(self, 
              data_yaml: str,
              epochs: int = 35,
              batch_size: int = 16,
              img_size: int = 640,
              patience: int = 15,
              save_dir: str = "runs/train",
              **kwargs) -> Dict[str, Any]:
        """Train the model."""
        try:
            if self.model is None:
                self.load_model()
            
            # Training parameters
            train_params = {
                'data': data_yaml,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'patience': patience,
                'project': save_dir,
                'name': 'face_mask_detection',
                'save': True,
                'plots': True,
                'exist_ok': True,
                'verbose': True,
                **kwargs
            }
            
            logger.info(f"Starting training with parameters: {train_params}")
            
            # Start training
            results = self.model.train(**train_params)
            
            # Copy best.pt to DVC-tracked location
            try:
                output_weights = Path(results.save_dir) / 'weights' / 'best.pt'
                dvc_weights = Path("models/yolov8_real_face_mask_detection/weights/best.pt")
                dvc_weights.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_weights, dvc_weights)
                logger.info(f"Copied best.pt to {dvc_weights}")
            except Exception as e:
                logger.error(f"Failed to copy best.pt to DVC location: {e}")

            # Get training metrics
            metrics = self._extract_training_metrics(results)
            
            logger.info(f"Training completed. Best mAP50: {metrics.get('best_map50', 'N/A')}")
            
            return {
                'success': True,
                'results': results,
                'metrics': metrics,
                'model_path': str(dvc_weights)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {},
                'model_path': None
            }
    
    def validate(self, data_yaml: str, **kwargs) -> Dict[str, Any]:
        """Validate the model."""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return {'success': False, 'error': 'Model not loaded'}
            
            logger.info("Starting validation...")
            results = self.model.val(data=data_yaml, **kwargs)
            
            metrics = self._extract_validation_metrics(results)
            
            logger.info(f"Validation completed. mAP50: {metrics.get('map50', 'N/A')}")
            
            return {
                'success': True,
                'results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def predict(self, source, **kwargs) -> Dict[str, Any]:
        """Make predictions."""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return {'success': False, 'error': 'Model not loaded'}
            
            results = self.model.predict(source=source, **kwargs)
            
            return {
                'success': True,
                'results': results,
                'predictions': self._format_predictions(results)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def export_model(self, 
                     format: str = "onnx",
                     save_dir: str = "exports",
                     **kwargs) -> Dict[str, Any]:
        """Export model to different formats."""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return {'success': False, 'error': 'Model not loaded'}
            
            FileUtils.ensure_dir(save_dir)
            
            logger.info(f"Exporting model to {format} format...")
            export_path = self.model.export(format=format, **kwargs)
            
            logger.info(f"Model exported to: {export_path}")
            
            return {
                'success': True,
                'export_path': export_path,
                'format': format
            }
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'export_path': None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            info = {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'device': str(self.device),
                'model_size': self._get_model_size(),
                'parameters': self._count_parameters(),
                'architecture': str(type(self.model))
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def _extract_training_metrics(self, results) -> Dict[str, Any]:
        """Extract metrics from training results."""
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            # Extract key metrics
            extracted = {
                'best_map50': metrics.get('metrics/mAP50(B)', 0.0),
                'best_map95': metrics.get('metrics/mAP50-95(B)', 0.0),
                'final_loss': metrics.get('train/box_loss', 0.0),
                'precision': metrics.get('metrics/precision(B)', 0.0),
                'recall': metrics.get('metrics/recall(B)', 0.0)
            }
            
            return extracted
            
        except Exception as e:
            logger.warning(f"Failed to extract training metrics: {e}")
            return {}
    
    def _extract_validation_metrics(self, results) -> Dict[str, Any]:
        """Extract metrics from validation results."""
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            extracted = {
                'map50': metrics.get('metrics/mAP50(B)', 0.0),
                'map95': metrics.get('metrics/mAP50-95(B)', 0.0),
                'precision': metrics.get('metrics/precision(B)', 0.0),
                'recall': metrics.get('metrics/recall(B)', 0.0)
            }
            
            return extracted
            
        except Exception as e:
            logger.warning(f"Failed to extract validation metrics: {e}")
            return {}
    
    def _format_predictions(self, results) -> List[Dict[str, Any]]:
        """Format prediction results."""
        formatted_predictions = []
        
        try:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        prediction = {
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                            'confidence': float(boxes.conf[i].cpu().numpy()),
                            'class_id': int(boxes.cls[i].cpu().numpy()),
                            'class_name': result.names[int(boxes.cls[i].cpu().numpy())]
                        }
                        formatted_predictions.append(prediction)
                        
        except Exception as e:
            logger.warning(f"Failed to format predictions: {e}")
        
        return formatted_predictions
    
    def _get_model_size(self) -> str:
        """Get model file size."""
        try:
            if hasattr(self.model, 'ckpt_path') and self.model.ckpt_path:
                size_mb = Path(self.model.ckpt_path).stat().st_size / (1024 * 1024)
                return f"{size_mb:.2f} MB"
            return "Unknown"
        except:
            return "Unknown"
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        try:
            if hasattr(self.model, 'model'):
                total_params = sum(p.numel() for p in self.model.model.parameters())
                return total_params
            return 0
        except:
            return 0

class ModelManager:
    """Manage model versions and MLflow integration."""
    def __init__(self, mlflow_tracking_uri: str = "file:///C:/Users/wwmsf/Desktop/face-mask-detection-mlops/mlruns") -> None:
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

    def log_model(self, 
                  model: FaceMaskYOLO,
                  experiment_name: str,
                  run_name: str,
                  metrics: Dict[str, Any],
                  params: Dict[str, Any],
                  model_path: str) -> str:
        """Log model to MLflow."""
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.pytorch.log_model(
                    pytorch_model=model.model,
                    artifact_path="model",
                    registered_model_name="face_mask_detector"
                )
                
                # Log model file
                mlflow.log_artifact(model_path, "weights")
                
                # Get run ID
                run_id = mlflow.active_run().info.run_id
                
                logger.info(f"Model logged to MLflow with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
            return ""
    
    def load_model_from_mlflow(self, run_id: str) -> Optional[FaceMaskYOLO]:
        """Load model from MLflow."""
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pytorch.load_model(model_uri)
            
            # Wrap in FaceMaskYOLO
            face_mask_model = FaceMaskYOLO()
            face_mask_model.model = model
            
            logger.info(f"Model loaded from MLflow run: {run_id}")
            return face_mask_model
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            return None
    
    def compare_models(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple model runs."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            comparison = {}
            for run_id in run_ids:
                run = client.get_run(run_id)
                comparison[run_id] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time
                }
            
            logger.info(f"Compared {len(run_ids)} model runs")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
    
    def get_best_model(self, experiment_name: str, metric_name: str = "map50") -> Optional[str]:
        """Get the best model run ID based on a metric."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=1
            )
            
            if not runs.empty:
                best_run_id = runs.iloc[0]['run_id']
                logger.info(f"Best model run ID: {best_run_id}")
                return best_run_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None

# --- Add this block to allow running as a script for DVC ---
if __name__ == "__main__":
    # Example usage for DVC pipeline
    # Adjust the config/paths as needed for your project
    DATA_YAML = "data/processed/yolo_dataset/dataset.yaml"
    model = FaceMaskYOLO()
    model.train(data_yaml=DATA_YAML)