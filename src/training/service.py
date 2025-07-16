"""
Training pipeline and service for Face Mask Detection
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
import mlflow.pytorch

from .model import FaceMaskYOLO, ModelManager
from .data_processing import DataProcessor
from ..common.logger import get_logger
from ..common.utils import FileUtils, ConfigUtils
from src.monitoring.metrics_collector import MetricsCollector
from drift.drift_detector import DataDriftDetector

logger = get_logger("training.service")

class TrainingService:
    """Main training service orchestrating the entire training pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model_manager = ModelManager(self.config['mlflow']['tracking_uri'])
        self.data_processor = DataProcessor()
        
        # Setup directories
        self.setup_directories()
        
        logger.info("Training service initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        # Default configuration
        default_config = {
            'data': {
                'class_names': ['with_mask', 'without_mask', 'mask_weared_incorrect'],
                'data_dir': 'data/processed/yolo_dataset',
                'raw_data_dir': 'data/raw'
            },
            'model': {
                'name': 'yolov8s.pt',
                'num_classes': 3,
                'img_size': 640
            },
            'training': {
                'epochs': 35,
                'batch_size': 16,
                'patience': 15,
                'learning_rate': 0.01,
                'optimizer': 'AdamW',
                'weight_decay': 0.0005
            },
            'mlflow': {
                'tracking_uri': 'http://localhost:5000',
                'experiment_name': 'face_mask_detection'
            },
            'paths': {
                'models_dir': 'models',
                'logs_dir': 'logs',
                'runs_dir': 'runs'
            }
        }
        try:
            config = ConfigUtils.load_config(config_path)
            # Merge with defaults
            return ConfigUtils.merge_configs(default_config, config)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}, using defaults: {e}")
            return default_config
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            self.config['paths']['models_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['runs_dir'],
            self.config['data']['data_dir'],
            'exports',
            'checkpoints'
        ]
        
        for directory in directories:
            FileUtils.ensure_dir(directory)
            logger.debug(f"Ensured directory: {directory}")
    
    def prepare_data(self, force_convert: bool = False) -> bool:
        """Prepare data for training"""
        try:
            logger.info("Starting data preparation...")
            
            data_dir = Path(self.config['data']['data_dir'])
            dataset_yaml = data_dir / "dataset.yaml"
            
            # Check if data is already prepared
            if dataset_yaml.exists() and not force_convert:
                logger.info("Dataset already prepared")
                return True
            
            # Convert Pascal VOC to YOLO format
            raw_data_dir = self.config['data']['raw_data_dir']
            if not Path(raw_data_dir).exists():
                logger.error(f"Raw data directory not found: {raw_data_dir}")
                return False
            
            success = self.data_processor.convert_pascal_voc_to_yolo(
                voc_dir=raw_data_dir,
                output_dir=str(data_dir)
            )
            
            if success:
                # Get class distribution
                class_dist = self.data_processor.get_class_distribution(str(data_dir))
                logger.info(f"Data preparation completed. Class distribution: {class_dist}")
                
                # Save class distribution
                dist_file = data_dir / "class_distribution.yaml"
                ConfigUtils.save_config(class_dist, str(dist_file))
                
            return success
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return False
    
    def train_model(self, 
                   experiment_name: Optional[str] = None,
                   run_name: Optional[str] = None,
                   **training_kwargs) -> Dict[str, Any]:
        """Train the face mask detection model with integrated metrics and drift detection."""
        try:
            # Setup experiment
            exp_name = experiment_name or self.config['mlflow']['experiment_name']
            run_name = run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Starting training - Experiment: {exp_name}, Run: {run_name}")
            train_config = self.config['training'].copy()
            train_config.update(training_kwargs)
            model = FaceMaskYOLO(
                model_name=self.config['model']['name'],
                num_classes=self.config['model']['num_classes']
            )
            if not model.load_model():
                return {'success': False, 'error': 'Failed to load model'}
            data_yaml = Path(self.config['data']['data_dir']) / "dataset.yaml"
            if not data_yaml.exists():
                logger.error("Dataset YAML not found. Run data preparation first.")
                return {'success': False, 'error': 'Dataset not prepared'}
            # --- Metrics & Drift Integration ---
            metrics_collector = MetricsCollector()
            drift_detector = DataDriftDetector()
            # Start MLflow run
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    'model_name': self.config['model']['name'],
                    'num_classes': self.config['model']['num_classes'],
                    'img_size': self.config['model']['img_size'],
                    **train_config
                })
                # --- Custom Training Loop for Epoch-wise Metrics/Drift ---
                epochs = train_config['epochs']
                batch_size = train_config['batch_size']
                img_size = self.config['model']['img_size']
                patience = train_config['patience']
                save_dir = self.config['paths']['runs_dir']
                optimizer = train_config['optimizer']
                weight_decay = train_config['weight_decay']
                device = 0 if torch.cuda.is_available() else 'cpu'
                # Use YOLO's built-in training, but wrap for per-epoch hooks if possible
                # If not, run metrics/drift at end only
                results = model.model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    patience=patience,
                    project=save_dir,
                    name='face_mask_detection',
                    save=True,
                    plots=True,
                    exist_ok=True,
                    verbose=True,
                    optimizer=optimizer,
                    weight_decay=weight_decay,
                    device=device,
                    # ...other kwargs...
                )
                # After training, collect metrics and drift on validation set
                val_results = model.validate(data=str(data_yaml))
                # Collect model metrics
                predictions = []  # You may want to run model.predict on val set for detailed metrics
                inference_times = []  # If available
                ground_truth = []  # If available
                # If you have access to val set images/labels, run predictions and collect times
                # For now, log summary metrics
                metrics_collector.collect_model_metrics(
                    model_version=self.config['model']['name'],
                    predictions=predictions,
                    inference_times=inference_times,
                    ground_truth=ground_truth
                )
                # Drift detection on validation set
                val_image_paths = []  # Fill with actual val image paths if available
                if val_image_paths:
                    drift_detector.set_reference_data(val_image_paths)  # For first run, or use a fixed reference
                    # For subsequent runs, compare current val set to reference
                    # drift_result = drift_detector.detect_drift(val_image_paths)
                    # Log drift_result as needed
                # Log training/validation metrics
                mlflow.log_metrics(results.results_dict if hasattr(results, 'results_dict') else {})
                if val_results['success']:
                    mlflow.log_metrics({f"val_{k}": v for k, v in val_results['metrics'].items()})
                # Save final model
                final_model_dir = Path(self.config['paths']['models_dir']) / exp_name
                FileUtils.ensure_dir(final_model_dir)
                final_model_path = final_model_dir / f"{run_name}_best.pt"
                if hasattr(results, 'save_dir'):
                    best_model_path = results.save_dir / 'weights' / 'best.pt'
                    FileUtils.copy_with_backup(best_model_path, final_model_path)
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Training completed successfully. Run ID: {run_id}")
                # Flush metrics
                # metrics_collector.flush()  # Implement flush if needed
                return {
                    'success': True,
                    'run_id': run_id,
                    'model_path': str(final_model_path),
                    'train_metrics': results.results_dict if hasattr(results, 'results_dict') else {},
                    'val_metrics': val_results.get('metrics', {}),
                    'experiment_name': exp_name,
                    'run_name': run_name
                }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'run_id': None,
                'model_path': None
            }
    
    def evaluate_model(self, model_path: str, data_yaml: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a trained model"""
        try:
            logger.info(f"Evaluating model: {model_path}")
            
            # Initialize model
            model = FaceMaskYOLO()
            if not model.load_model(model_path):
                return {'success': False, 'error': 'Failed to load model'}
            
            # Use default data if not provided
            if data_yaml is None:
                data_yaml = str(Path(self.config['data']['data_dir']) / "dataset.yaml")
            
            # Run validation
            val_results = model.validate(data=data_yaml)
            
            if val_results['success']:
                logger.info(f"Evaluation completed. mAP50: {val_results['metrics'].get('map50', 'N/A')}")
            
            return val_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def export_model(self, 
                    model_path: str,
                    export_format: str = "onnx",
                    export_dir: str = "exports") -> Dict[str, Any]:
        """Export model to different formats"""
        try:
            logger.info(f"Exporting model to {export_format} format...")
            
            # Initialize model
            model = FaceMaskYOLO()
            if not model.load_model(model_path):
                return {'success': False, 'error': 'Failed to load model'}
            
            # Export model
            export_results = model.export_model(
                format=export_format,
                save_dir=export_dir
            )
            
            if export_results['success']:
                logger.info(f"Model exported successfully to: {export_results['export_path']}")
            
            return export_results
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'export_path': None
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and available models"""
        try:
            status = {
                'data_prepared': False,
                'available_models': [],
                'latest_experiment': None,
                'mlflow_uri': self.config['mlflow']['tracking_uri']
            }
            
            # Check if data is prepared
            data_yaml = Path(self.config['data']['data_dir']) / "dataset.yaml"
            status['data_prepared'] = data_yaml.exists()
            
            # Get available models
            models_dir = Path(self.config['paths']['models_dir'])
            if models_dir.exists():
                status['available_models'] = [
                    str(p) for p in models_dir.glob("**/*.pt")
                ]
            
            # Get latest experiment info
            try:
                experiment = mlflow.get_experiment_by_name(
                    self.config['mlflow']['experiment_name']
                )
                if experiment:
                    status['latest_experiment'] = {
                        'experiment_id': experiment.experiment_id,
                        'name': experiment.name,
                        'lifecycle_stage': experiment.lifecycle_stage
                    }
            except:
                pass
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {'error': str(e)}
    
    def cleanup_old_runs(self, keep_last_n: int = 5) -> bool:
        """Clean up old training runs"""
        try:
            runs_dir = Path(self.config['paths']['runs_dir'])
            if not runs_dir.exists():
                return True
            
            # Get all run directories sorted by creation time
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            run_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            
            # Remove old runs
            for run_dir in run_dirs[keep_last_n:]:
                try:
                    import shutil
                    shutil.rmtree(run_dir)
                    logger.info(f"Removed old run: {run_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove {run_dir}: {e}")
            
            logger.info(f"Cleanup completed. Kept {min(len(run_dirs), keep_last_n)} recent runs")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

# Convenience function for external use
def create_training_service(config_path: str = "config/config.yaml") -> TrainingService:
    """Create and return a training service instance"""
    return TrainingService(config_path)
