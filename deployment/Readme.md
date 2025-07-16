# Face Mask Detection MLOps - Dockerized Pipeline

## Overview

This project uses Docker Compose to orchestrate:
- **MLflow Tracking Server** (`mlflow`)
- **Training Service** (`training-service`)
- **Inference API** (`inference-service`)
- **Monitoring Dashboard** (`monitoring-service`)
- **Drift Detection** (`drift-detection`)

All services share data, models, logs, and experiment tracking via mounted volumes.

---

## Build and Run

1. **Navigate to the deployment directory:**
   ```sh
   cd deployment
   ```

2. **Build and start all services:**
   ```sh
   docker-compose up --build
   ```

   - MLflow: [http://localhost:5000](http://localhost:5000)
   - Inference API: [http://localhost:8000](http://localhost:8000)
   - Monitoring Dashboard: [http://localhost:8003](http://localhost:8003)
   - Drift Detection: [http://localhost:8004](http://localhost:8004) (if API)
   - Training: runs as a job, check logs for output

3. **Stop all services:**
   ```sh
   docker-compose down
   ```

---

## Notes

- **Data Persistence:** All important data (models, logs, reports, mlruns, collected data) is persisted on your host via volumes.
- **GPU Support:** For GPU inference/training, use a CUDA-enabled Python base image and add the following to your service in `docker-compose.yml`:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
  ```
- **Logs:** Check logs with:
  ```sh
  docker-compose logs inference-service
  docker-compose logs monitoring-service
  ```

---

## Troubleshooting

- If you change code, re-run `docker-compose up --build` to rebuild images.
- Make sure all required Python packages are in the correct `requirements-*.txt`.

---

**Enjoy your production-ready, containerized MLOps pipeline!**