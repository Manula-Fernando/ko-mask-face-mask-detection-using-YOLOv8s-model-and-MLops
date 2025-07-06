# Detailed `src/` Directory Structure and Purpose

## Top-Level

- **common/**: Shared utilities, configuration, and pipeline logic for training, inference, and monitoring.
- **inference/**: All inference logic, including FastAPI API server, prediction services, and web UI templates.
- **monitoring/**: Monitoring and drift detection services, including Streamlit dashboards and metrics collectors.
- **training/**: Model training, data processing, and orchestration scripts.
- **realtime_webcam_app.py**: Standalone real-time webcam detection app with medical-grade UI.
- **README.md**: Explains the purpose and usage of each main component.

## common/
- `config.py`: Loads and manages configuration settings.
- `data_processor.py`: Handles data preprocessing and augmentation.
- `inference_pipeline.py`: Orchestrates the inference workflow.
- `logger.py`: Centralized logging utility.
- `monitoring_pipeline.py`: Connects model outputs to monitoring/drift detection.
- `training_pipeline.py`: Orchestrates the training workflow.
- `utils.py`: General utility functions.

## inference/
- `api.py`: Main FastAPI server for inference.
- `inference_service.py`: Service logic for running inference.
- `predictor.py`: Loads YOLO model and runs predictions.
- `service.py`: Additional service logic (possibly for orchestration).
- `templates/index.html`: Web UI for the API.
- `__init__.py`: Package marker.

## monitoring/
- `dashboard.py`: Streamlit dashboard for monitoring.
- `drift_detector.py`: Detects data/model drift.
- `metrics_collector.py`: Collects and aggregates metrics.
- `monitoring_service.py`: Service logic for monitoring.
- `service.py`: Orchestration for monitoring.
- `__init__.py`: Package marker.

## training/
- `data_processing.py`: Data loading and preprocessing for training.
- `model.py`: Model architecture and loading.
- `service.py`: Orchestration for training.
- `train.py`: Main training script.
- `training_service.py`: Service logic for training.
- `__init__.py`: Package marker.

## Standalone
- `realtime_webcam_app.py`: Main entry point for real-time webcam detection.

---

**All legacy, duplicate, and unused files have been archived or removed. The codebase is now clean, maintainable, and production-ready.**
