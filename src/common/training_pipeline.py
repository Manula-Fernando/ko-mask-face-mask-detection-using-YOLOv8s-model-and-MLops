"""
Training Pipeline for Face Mask Detection
"""

import os
import json
import shutil
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

import mlflow
import mlflow.pytorch
from ultralytics import YOLO

from .config import TrainingConfig, ProjectConfig

class TrainingPipeline:
    """Complete MLOps training pipeline with experiment tracking"""
    
    def __init__(self, config: TrainingConfig, project_config: ProjectConfig):
        self.config = config
        self.project_config = project_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.mlflow_run_id = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(project_config.mlflow_tracking_uri)
        mlflow.set_experiment(project_config.experiment_name)
    
    def setup_model(self):
        """Initialize YOLO model with configuration"""
        try:
            self.model = YOLO(self.config.model_name)
            self.logger.info(f"✅ Model {self.config.model_name} loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def log_hyperparameters(self):
        """Log hyperparameters to MLflow"""
        params = {
            "model_name": self.config.model_name,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "image_size": self.config.image_size,
            "learning_rate": self.config.learning_rate,
            "patience": self.config.patience,
            "classes": str(self.config.classes)
        }
        
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        self.logger.info("📊 Hyperparameters logged to MLflow")
    
    def train_model(self, data_yaml_path: str):
        """Train model with comprehensive MLOps tracking"""
        if not self.model:
            if not self.setup_model():
                raise RuntimeError("Failed to setup model")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            self.mlflow_run_id = run.info.run_id
            self.logger.info(f"🚀 Started MLflow run: {self.mlflow_run_id}")
            
            # Log hyperparameters
            self.log_hyperparameters()
            
            # Log dataset info
            mlflow.log_param("dataset_yaml", data_yaml_path)
            
            try:
                # Train the model
                self.logger.info("🎯 Starting model training...")
                results = self.model.train(
                    data=data_yaml_path,
                    epochs=self.config.epochs,
                    batch=self.config.batch_size,
                    imgsz=self.config.image_size,
                    lr0=self.config.learning_rate,
                    patience=self.config.patience,
                    project=f"{self.project_config.models_dir}/training",
                    name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    save=True,
                    verbose=True,
                    plots=True
                )
                
                # Log training metrics
                if hasattr(results, 'results_dict'):
                    for key, value in results.results_dict.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                # Log model artifacts
                model_path = self.model.trainer.best
                mlflow.log_artifact(model_path, "models")
                
                # Save best model to standard location
                best_model_path = Path(self.project_config.models_dir) / "best_model.pt"
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, best_model_path)
                
                self.logger.info(f"✅ Training completed successfully!")
                self.logger.info(f"📁 Best model saved to: {best_model_path}")
                
                return {
                    "success": True,
                    "best_model_path": str(best_model_path),
                    "mlflow_run_id": self.mlflow_run_id,
                    "results": str(results)  # Convert to string for JSON serialization
                }
                
            except Exception as e:
                self.logger.error(f"❌ Training failed: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def validate_model(self, model_path: Optional[str] = None):
        """Validate trained model"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        if not model:
            raise ValueError("No model available for validation")
        
        # Validation metrics
        metrics = model.val()
        
        # Log validation metrics to MLflow
        if hasattr(metrics, 'results_dict'):
            for key, value in metrics.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{key}", value)
        
        self.logger.info("✅ Model validation completed")
        return metrics
    
    def generate_training_report(self, results):
        """Generate comprehensive training report"""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_config": asdict(self.config),
            "mlflow_run_id": self.mlflow_run_id,
            "training_results": results,
            "status": "completed"
        }
        
        # Save report
        report_path = Path(self.project_config.reports_dir) / "training" / f"training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Training report saved: {report_path}")
        return report_path
