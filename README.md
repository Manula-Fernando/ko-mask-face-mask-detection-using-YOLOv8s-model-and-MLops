# Face Mask Detection MLOps Pipeline

A production-ready, end-to-end MLOps pipeline for real-time face mask detection using YOLOv8, MLflow, DVC, and modern best practices.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
make install

# Train the model
make train

# Run the API server
make webapp

# Run the monitoring dashboard
make notebooks

# Run all tests
make test
```

### Production Deployment (Docker Compose)

```bash
cd deployment
# Build and start all services
docker-compose up -d
```

- **API Docs**: http://localhost:8002/docs
- **MLflow UI**: http://localhost:5000
- **Monitoring Dashboard**: http://localhost:8501

## Automated Weekly Retraining Workflow

To ensure the model remains accurate and adapts to new data, this project supports **automated weekly retraining** using both the original dataset and newly detected images:

- **Detection Images Collection**: All images processed by the API (e.g., from real-time predictions) are saved and curated for potential inclusion in future training cycles.
- **Data Aggregation**: A scheduled job (e.g., via CI/CD, cron, or Makefile target) aggregates new detection images with the existing dataset, applying quality checks and deduplication.
- **Retraining Pipeline**: The retraining process is automated using DVC and Makefile scripts. It:
  1. Updates the training dataset with new images.
  2. Runs preprocessing and augmentation.
  3. Retrains the YOLOv8 model.
  4. Logs results and model versions in MLflow.
  5. Optionally deploys the new model if performance improves.
- **Scheduling**: Weekly retraining can be triggered by a CI/CD workflow (e.g., GitHub Actions) or a scheduled script on your server/cloud.
- **Configuration**: See `Makefile`, `dvc.yaml`, and automation scripts in `scripts/` for how to customize or trigger retraining.

**Best Practices:**
- Review and curate new detection images before adding to the training set to avoid introducing noise or bias.
- Use DVC to version all data and pipeline stages for full reproducibility.
- Monitor model performance after each retraining cycle using the Streamlit dashboard and MLflow.

## Project Structure

```
face-mask-detection-mlops/
├── src/                  # Modular source code (API, training, monitoring, utils)
├── scripts/              # Automation & deployment scripts
├── tests/                # Unit and integration tests
├── notebooks/            # Analysis & reporting
├── deployment/           # Docker Compose & Dockerfiles
├── data/                 # DVC-tracked datasets
├── models/               # Model artifacts
├── logs/                 # Logs
├── mlruns/               # MLflow tracking
├── README.md             # Main documentation
├── ... (other docs)
```

## Features

- Real-time face mask detection API (FastAPI)
- YOLOv8 model with >90% mAP@0.5
- MLflow experiment tracking and model registry
- Automated data versioning with DVC
- Streamlit dashboard for monitoring and drift detection
- Multi-service orchestration with Docker Compose
- Automated retraining and CI/CD with GitHub Actions

## Documentation

- See `PROJECT_DOCUMENTATION.md` and `FINAL_PROJECT_SUMMARY.md` for full details.

---

**This project is clean, production-ready, and follows industry-standard MLOps best practices.**
