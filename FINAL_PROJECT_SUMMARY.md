# FINAL PROJECT SUMMARY: Face Mask Detection MLOps

## Project Overview

This project delivers a **production-grade, end-to-end MLOps pipeline** for real-time face mask detection using deep learning (YOLOv8) and modern MLOps best practices. It is designed for healthcare, workplace safety, and public compliance monitoring, with robust automation, monitoring, and deployment capabilities.

---

## Objectives
- **Automate the full ML lifecycle**: From data ingestion and preprocessing to model training, evaluation, deployment, and ongoing monitoring, every stage is automated for consistency and reliability. This ensures minimal manual intervention, reduces human error, and accelerates iteration cycles.
- **Enable real-time face mask detection** with high accuracy and low latency: The system is optimized for both speed and precision, making it suitable for real-world applications where immediate feedback is critical (e.g., hospital entrances, public transport, workplaces).
- **Ensure reproducibility, traceability, and scalability** using DVC, MLflow, and Docker: All experiments, data, and models are versioned and tracked, making it easy to reproduce results, audit changes, and scale the solution across environments.
- **Provide robust monitoring, drift detection, and alerting** for deployed models: The pipeline includes tools to monitor model performance, detect data or concept drift, and trigger alerts for retraining or investigation, ensuring long-term reliability.
- **Support CI/CD and containerized deployment** for rapid iteration and production readiness: Automated testing, linting, and deployment pipelines ensure that new features and fixes can be safely and quickly delivered to production.

---

## Architecture & Workflow

### 1. Data Pipeline
- **Raw data**: Images are sourced from diverse datasets (medical, public, synthetic) to ensure model generalizability. Data is stored in a DVC-managed structure for version control.
- **Preprocessing**: Automated scripts handle cleaning (removing corrupt/duplicate images), augmentation (rotations, flips, color adjustments), and splitting into train/val/test sets. This ensures robust model training and fair evaluation.
- **Versioning**: DVC tracks all data and pipeline stages, enabling full reproducibility. Any change in data or code triggers a new pipeline run, and previous states can be restored at any time.

### 2. Model Development
- **Model**: YOLOv8 (Ultralytics) is chosen for its state-of-the-art real-time object detection capabilities, balancing speed and accuracy.
- **Training**: Training is fully automated via scripts and DVC pipelines. Hyperparameter tuning and class balancing are supported to maximize performance. Training runs are reproducible and tracked.
- **Evaluation**: Automated evaluation scripts compute metrics (precision, recall, mAP) and generate plots. All results are logged to MLflow for easy comparison and analysis.

### 3. Inference & API
- **FastAPI**: Provides a production-ready REST API for real-time predictions. Designed for scalability and easy integration with other systems.
- **Webcam App**: A user-friendly UI (e.g., Streamlit or OpenCV-based) allows live camera feeds to be analyzed in real time, demonstrating the model's capabilities interactively.
- **Batch & single image prediction**: The API supports both single-image and batch processing, making it flexible for different use cases (e.g., real-time monitoring vs. retrospective analysis).

### 4. Monitoring & Drift Detection
- **Streamlit Dashboard**: Offers real-time visualization of predictions, compliance rates, and system health. Enables stakeholders to monitor model performance and operational metrics at a glance.
- **Drift Detection**: Automated scripts periodically analyze incoming data and predictions to detect data or concept drift. Alerts are triggered if significant drift is detected, prompting retraining or investigation.
- **Logging**: All services produce centralized, structured logs (JSON or text), facilitating debugging, auditing, and compliance.

### 5. Experiment Tracking
- **MLflow**: All experiments, metrics, artifacts, and model versions are tracked in MLflow. This enables easy comparison of different runs, rollback to previous models, and transparent reporting.

### 6. Deployment & Orchestration
- **Docker & Docker Compose**: Every service (API, training, monitoring, MLflow, etc.) is containerized for consistent, reproducible deployment across environments (local, staging, production). Docker Compose orchestrates multi-service deployments.
- **Makefile**: Common tasks (build, test, deploy, clean, lint, etc.) are automated via Makefile targets, reducing manual errors and onboarding time for new contributors.
- **launch_pipeline.py**: A unified script to launch all services locally, simplifying development and testing.

### 7. CI/CD
- **GitHub Actions**: Automated workflows for testing, linting, and deployment. Ensures code quality and enables rapid, safe delivery of new features and fixes.

Here’s where each of these functionalities is implemented in your project, with the relevant files and their roles:


---

### 1. Data Pipeline

- **Raw Data (DVC-managed)**
  - **Files/Dirs:**  
    - raw, processed, data (all DVC-tracked)
    - dvc.yaml, dvc.lock (pipeline and data versioning)
  - **What it does:**  
    - Stores all datasets (medical, public, synthetic).
    - DVC manages data versioning, so every change is tracked and reproducible.

- **Preprocessing**
  - **Files/Dirs:**  
    - `src/data/preprocessing.py` (or similar in `src/data/`)
    - scripts (may contain batch/utility scripts)
    - dvc.yaml (defines preprocessing as a pipeline stage)
  - **What it does:**  
    - Cleans, augments, and splits data into train/val/test.
    - Can be run via DVC or Makefile.

- **Versioning**
  - **Files/Dirs:**  
    - dvc.yaml, dvc.lock, .dvcignore
    - All data folders under DVC control
  - **What it does:**  
    - Tracks every data and pipeline change.
    - Enables full reproducibility and rollback.

---

### 2. Model Development

- **Model (YOLOv8)**
  - **Files/Dirs:**  
    - `src/models/` (YOLOv8 code, config, weights)
    - models (trained model artifacts, managed by DVC/MLflow)
    - requirements.txt (lists YOLOv8/Ultralytics as a dependency)
  - **What it does:**  
    - Implements and stores the YOLOv8 model for detection.

- **Training**
  - **Files/Dirs:**  
    - train.py (or similar)
    - dvc.yaml (defines training as a pipeline stage)
    - Makefile (may have `make train`)
  - **What it does:**  
    - Automates model training, supports hyperparameter tuning and class balancing.
    - Training runs are reproducible and tracked.

- **Evaluation**
  - **Files/Dirs:**  
    - `src/evaluation/evaluate.py` (or similar)
    - dvc.yaml (defines evaluation as a pipeline stage)
    - mlruns (MLflow experiment tracking)
  - **What it does:**  
    - Computes metrics (precision, recall, mAP), generates plots.
    - Logs results to MLflow for comparison and analysis.

---

### 3. Inference & API

- **FastAPI**
  - **Files/Dirs:**  
    - `src/inference/realtime_mask_detector.py`, `src/main.py` (API entrypoint)
    - Dockerfile.inference (containerizes the API)
    - docker-compose.yml (service: `inference-service`)
  - **What it does:**  
    - Provides a REST API for real-time predictions.
    - Scalable and easy to integrate with other systems.

- **Webcam App**
  - **Files/Dirs:**  
    - webcam_app.py or similar (may use Streamlit or OpenCV)
    - Dockerfile.monitoring (if using Streamlit)
  - **What it does:**  
    - UI for live camera feeds, demonstrates real-time detection.

- **Batch & Single Image Prediction**
  - **Files/Dirs:**  
    - API endpoints in `src/inference/realtime_mask_detector.py` or `src/main.py`
    - Test scripts: test_api.py
  - **What it does:**  
    - Supports both single-image and batch prediction via API endpoints.

---

### 4. Monitoring & Drift Detection

- **Streamlit Dashboard**
  - **File(s):**  
    - dashboard.py (or similar in monitoring)
    - Dockerfile.monitoring (for containerization)
  - **What it does:**  
    - Provides a real-time dashboard for visualizing predictions, compliance, and system health.
    - Launched as the `monitoring-service` in docker-compose.yml.

- **Drift Detection**
  - **File(s):**  
    - drift_detection.py (or similar)
    - `deployment/Dockerfile.drift`
    - Service: `drift-detection` in docker-compose.yml
  - **What it does:**  
    - Periodically analyzes new data and predictions for drift.
    - Triggers alerts or retraining if drift is detected.

- **Logging**
  - **File(s):**  
    - All services log to the logs directory.
    - Logging code is typically in src modules (e.g., `src/common/logging.py` or within each service).
    - Logs are mounted in containers via `docker-compose.yml`.

---

### 5. Experiment Tracking

- **MLflow**
  - **File(s):**  
    - docker-compose.yml (service: `mlflow`)
    - src code (training, retraining, and evaluation scripts use `mlflow` Python API)
    - mlruns directory (stores MLflow runs and artifacts)
  - **What it does:**  
    - Tracks all experiments, metrics, and model versions.
    - UI available at http://localhost:5000.

---

### 6. Deployment & Orchestration

- **Docker & Docker Compose**
  - **File(s):**  
    - docker-compose.yml
    - `deployment/Dockerfile.*` (for each service: training, inference, monitoring, drift, etc.)
  - **What it does:**  
    - Containerizes all services for reproducible, scalable deployment.
    - Orchestrates multi-service environments for local, staging, or production.

- **Makefile**
  - **File(s):**  
    - Makefile (at project root)
  - **What it does:**  
    - Automates common tasks: build, test, deploy, clean, lint, etc.
    - Reduces manual errors and speeds up onboarding.

- **launch_pipeline.py**
  - **File(s):**  
    - launch_pipeline.py
  - **What it does:**  
    - Unified script to launch all services locally (for development/testing without Docker Compose).

---

### 7. CI/CD

- **GitHub Actions**
  - **File(s):**  
    - retrain_schedule.yml (and any other workflow YAMLs in workflows)
  - **What it does:**  
    - Automates testing, linting, retraining, and deployment.
    - Can be triggered on schedule, on push, or manually.
    - Ensures code quality and safe, rapid delivery of new features.

---

**Summary Table:**

| Functionality         | Main Files/Dirs Involved                        | Description                                      |
|----------------------|-------------------------------------------------|--------------------------------------------------|
| Data Pipeline        | data, dvc.yaml, `src/data/`                | Data storage, preprocessing, versioning          |
| Model Development    | `src/models/`, training, models, dvc.yaml, mlruns | Model code, training, evaluation, tracking       |
| Inference & API      | inference, `src/main.py`, Dockerfile.inference, `docker-compose.yml` | Real-time/batch API, webcam app                  |
| Monitoring Dashboard         | monitoring, Dockerfile.monitoring   | Real-time dashboard (Streamlit)                  |
| Drift Detection              | drift_detection.py, `deployment/Dockerfile.drift`, `docker-compose.yml` | Automated drift analysis and alerting            |
| Logging                      | logs, logging code in src                         | Centralized, structured logs                     |
| MLflow Experiment Tracking   | docker-compose.yml, mlruns, src      | Tracks experiments, metrics, models              |
| Docker & Docker Compose      | docker-compose.yml, `deployment/Dockerfile.*` | Containerization and orchestration               |
| Makefile                     | Makefile                                              | Automation of build, test, deploy, etc.          |
| launch_pipeline.py           | launch_pipeline.py                                | Local unified service launcher                   |
| CI/CD                        | retrain_schedule.yml                | Automated testing, retraining, deployment        |

---

## Technology Stack
- **Python 3.9+**: Chosen for its rich ML ecosystem and compatibility with all major libraries.
- **YOLOv8 (Ultralytics)**: State-of-the-art object detection, optimized for real-time applications.
- **FastAPI, Uvicorn**: High-performance, modern web framework for serving ML models.
- **Streamlit**: Rapid dashboard development for monitoring and visualization.
- **MLflow**: Industry-standard experiment tracking and model registry.
- **DVC**: Data and pipeline versioning for reproducibility and collaboration.
- **Docker, Docker Compose**: Containerization and orchestration for consistent deployments.
- **PyTorch, torchvision**: Deep learning framework and vision utilities.
- **OpenCV, Pillow, albumentations**: Image processing and augmentation.
- **scikit-learn, pandas, matplotlib, seaborn, plotly**: Data analysis, metrics, and visualization.
- **pytest, flake8, black, isort**: Testing and code quality tools.

---

## Directory Structure (Key Folders)
- `src/` — All production code, organized by function (inference, training, monitoring, utils). Ensures a clean separation of concerns and easy navigation.
- `notebooks/` — Jupyter notebooks for data exploration, model development, and evaluation. Used for experimentation and documentation.
- `models/` — Stores trained models, with clear separation between production, staging, and backup versions. Managed by DVC and MLflow.
- `logs/` — Centralized logs and monitoring reports. Facilitates debugging and compliance.
- `data/` — Raw and processed data, fully managed by DVC for versioning and reproducibility.
- `deployment/` — Dockerfiles, docker-compose, and deployment configs. All infrastructure-as-code for easy deployment.
- `scripts/` — Automation and utility scripts (e.g., data download, batch inference, maintenance).
- `tests/` — Organized into unit, integration, and pipeline tests. Ensures code reliability and coverage.
- `config/` — YAML configuration files for all services and pipelines. Centralizes settings for easy management.

---

## Key Results & Achievements
- **High-accuracy YOLOv8 model** for face mask detection, validated on diverse datasets and real-world scenarios.
- **Automated, reproducible pipeline** with DVC and MLflow, ensuring every result can be traced and reproduced.
- **Production-ready API** with real-time performance, suitable for integration into existing systems.
- **Comprehensive monitoring and drift detection** to maintain model reliability over time.
- **CI/CD and containerization** for rapid, reliable deployment and scaling.
- **Extensive documentation and code quality**: All major components are documented, and code adheres to modern Python standards.

---

## How to Use
1. **Install dependencies**: `make install` — Installs all required Python packages and sets up the environment.
2. **Run the full pipeline**: `make pipeline` or `python src/launch_pipeline.py` — Executes the entire ML pipeline, from data processing to model deployment.
3. **Start all services (Docker)**: `cd deployment && docker-compose up --build` — Launches all containerized services (API, MLflow, monitoring, etc.) for a full production environment.
4. **Access services**:
   - API: http://localhost:8000 — Main REST API for predictions.
   - API Docs: http://localhost:8000/docs — Interactive Swagger UI for API exploration.
   - MLflow: http://localhost:5000 — Experiment tracking and model registry.
   - Monitoring Dashboard: http://localhost:8501 — Real-time monitoring and drift detection dashboard.

---

## Best Practices Followed
- Modular, maintainable codebase with clear separation of concerns.
- All scripts reference the `src/` structure, avoiding hardcoded paths and legacy references.
- No legacy or duplicate files: The codebase has been thoroughly cleaned for clarity and maintainability.
- All logs, models, and data are versioned and organized for traceability and compliance.
- Security: secrets and sensitive files are ignored via `.gitignore`, `.dvcignore`, and `.dockerignore`.
- Documentation for every major component, including architecture, workflows, and usage instructions.

---

## Contributors & Acknowledgements
- Project lead: [Your Name]
- Contributors: [Team Members]
- Based on open-source tools and best practices from the MLOps community

---

**This project is ready for production deployment, research, and further extension.**
