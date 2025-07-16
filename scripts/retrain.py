"""
retrain.py: Automated Weekly Retraining Orchestration Script

This script collects new detection images, merges them with the main dataset, triggers the DVC pipeline for retraining, and logs results to MLflow. Intended for use in CI/CD, cron, or manual execution.
"""

import shutil
from pathlib import Path
import subprocess
import sys
import mlflow
import time
from typing import Optional

# CONFIGURABLE PATHS
NEW_IMAGES_DIR = Path("data/curated/")  # Where new detection images are saved
DATASET_DIR = Path("data/raw/")       # Main dataset directory
MODEL_PATH = Path("models/latest_model.pt")  # Example model artifact path

def collect_new_images() -> int:
    """Collect new detection images and copy to dataset directory."""
    try:
        if not NEW_IMAGES_DIR.exists():
            print(f"No curated images found in {NEW_IMAGES_DIR}.")
            return 0
        images = list(NEW_IMAGES_DIR.glob("*.jpg"))
        count = len(images)
        if count < 50:
            print(f"Only {count} images in {NEW_IMAGES_DIR}. Need at least 50 to retrain.")
            return 0
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img_file in images:
            dest = DATASET_DIR / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
                copied += 1
        print(f"Collected {copied} new images from curated/ to data/raw/.")
        return copied
    except Exception as e:
        print(f"[ERROR] Failed to collect new images: {e}")
        return 0

def curate_images() -> None:
    """Placeholder for data curation step (manual/automated)."""
    print("[INFO] Data curation step (manual/automated) can be implemented here.")

def run_dvc_pipeline() -> None:
    """Run DVC pipeline for retraining."""
    print("[INFO] Running DVC pipeline for retraining...")
    try:
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("[ERROR] DVC pipeline failed:")
            print(result.stderr)
            sys.exit(1)
        print("[SUCCESS] DVC pipeline completed.")
    except Exception as e:
        print(f"[ERROR] DVC pipeline execution failed: {e}")
        sys.exit(1)

def post_retrain_actions(mlflow_run: Optional[mlflow.ActiveRun] = None) -> None:
    """Post-retraining actions (evaluation, deployment, logging to MLflow)."""
    print("[INFO] Post-retraining actions (evaluation, deployment) can be added here.")
    if mlflow_run and MODEL_PATH.exists():
        try:
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
            print(f"[MLflow] Model artifact logged: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to log model artifact to MLflow: {e}")

if __name__ == "__main__":
    print("=== Automated Weekly Retraining Script ===")
    mlflow.set_experiment("Weekly_Retraining")
    with mlflow.start_run(run_name=f"retrain_{time.strftime('%Y%m%d_%H%M%S')}") as run:
        n = collect_new_images()
        mlflow.log_param("new_images_collected", n)
        curate_images()  # Optional/manual step
        if n > 0:
            run_dvc_pipeline()
            # Optionally log metrics, artifacts, etc. here
            post_retrain_actions(mlflow_run=run)
            mlflow.log_param("retraining_triggered", True)
        else:
            print("Not enough new images to retrain on (need at least 50). Skipping retraining.")
            mlflow.log_param("retraining_triggered", False)
        print("=== Done ===")

# --- Scheduling Guidance ---
# To automate this script every Sunday night, use a cron job or GitHub Actions schedule:
# Example cron: 0 21 * * 0 (every Sunday at 21:00 UTC)
# Or use Windows Task Scheduler to run weekly.

# --- CI/CD Integration Guidance ---
'''
To integrate this retraining script into your CI/CD workflow:

1. **Scheduled Retraining (Recommended):**
   - Use the provided `.github/workflows/retrain_schedule.yml` to schedule weekly retraining (or adjust the cron as needed).
   - The workflow will:
     - Checkout the repo
     - Install dependencies
     - Run this script
     - Commit/push any updated data/models
     - All retraining runs are tracked in MLflow

2. **Manual Trigger:**
   - You can trigger retraining manually from the GitHub Actions UI ("Run workflow" button) or by dispatching the workflow via API.

3. **On-Demand Triggers:**
   - Add additional triggers to your workflow file, such as:
     - On push to `main` or `release` branches
     - On new data upload
     - On model performance drop (monitoring alert)

Example trigger block for `.github/workflows/retrain_schedule.yml`:

on:
  schedule:
    - cron: '0 3 * * 1'  # Every Monday at 03:00 UTC
  workflow_dispatch:      # Manual trigger from GitHub UI
  push:
    branches:
      - main
      - release/*

See `RETRAINING_WORKFLOW.md` and the workflow YAML for more details and customization.
'''
