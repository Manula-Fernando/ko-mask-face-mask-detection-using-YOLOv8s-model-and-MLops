"""
Common configuration classes for MLOps pipeline
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ProjectConfig:
    """Complete project configuration"""
    # Project paths
    project_root: str = "../"
    data_dir: str = "../data"
    models_dir: str = "../models"
    logs_dir: str = "../logs"
    reports_dir: str = "../reports"
    src_dir: str = "../src"
    
    # Docker and Services
    docker_compose_file: str = "../docker-compose.yml"
    training_service_port: int = 8001
    inference_service_port: int = 8002
    monitoring_service_port: int = 8003
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "face_mask_detection_mlops"

@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    # Data paths
    raw_data_path: str = "../data/raw"
    processed_data_path: str = "../data/processed"
    dataset_yaml: str = "../data/processed/data.yaml"
    
    # Model configuration
    model_name: str = "yolov8s.pt"
    epochs: int = 35
    batch_size: int = 16
    image_size: int = 640
    
    # Training parameters
    learning_rate: float = 0.001
    patience: int = 10
    
    # Classes
    classes: List[str] = field(default_factory=lambda: ["with_mask", "without_mask", "mask_weared_incorrect"])

@dataclass
class InferenceConfig:
    """Inference pipeline configuration"""
    model_path: str = "../models/best_model.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 1000
    device: str = "auto"  # auto, cpu, or cuda

@dataclass
class MonitoringConfig:
    """Monitoring pipeline configuration"""
    metrics_database: str = "../data/monitoring/metrics.db"
    alert_threshold: float = 0.1  # Model performance degradation threshold
    data_drift_threshold: float = 0.05
    monitoring_interval: int = 3600  # seconds

@dataclass
class DriftDetectionConfig:
    """Drift detection configuration"""
    reference_data_path: str = "../data/processed/reference"
    current_data_path: str = "../data/collected/current"
    output_dir: str = "../reports/drift_analysis"
    drift_detection_method: str = "ks_test"

def load_config(config_type: str = "inference") -> object:
    """
    Load and return the specified configuration class.
    Args:
        config_type (str): One of "project", "training", "inference", "monitoring", "drift".
    Returns:
        An instance of the requested configuration dataclass.
    """
    if config_type == "project":
        return ProjectConfig()
    elif config_type == "training":
        return TrainingConfig()
    elif config_type == "inference":
        return InferenceConfig()
    elif config_type == "monitoring":
        return MonitoringConfig()
    elif config_type == "drift":
        return DriftDetectionConfig()
    else:
        raise ValueError(f"Unknown config_type: {config_type}")

