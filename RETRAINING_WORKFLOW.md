# Automated Retraining Scripts & Scheduling

This project supports automated weekly retraining using new detection images and the main dataset. Below are the key scripts and workflow files:

## 1. scripts/retrain.py
- Orchestrates the retraining process: collects new detection images, merges with the dataset, triggers DVC pipeline, and logs results to MLflow for full experiment tracking.
- Run manually, via Makefile, or as part of CI/CD.

## 2. scripts/data_curation.py (optional)
- Allows manual or semi-automated review/approval of new detection images before adding to the training set.
- Extend for quality checks or annotation as needed.

## 3. .github/workflows/retrain_schedule.yml
- Example GitHub Actions workflow to schedule retraining every week (Monday 03:00 UTC) as part of your CI/CD pipeline.
- Can be adapted for other CI/CD systems or cron jobs.
- The workflow will:
  - Install dependencies
  - Run the retraining script
  - Commit and push any updated data/models
  - All retraining runs and results are tracked in MLflow

## How to Use
- Place new detection images in `detections/` (automated by API or manually).
- (Optional) Run `python scripts/data_curation.py` to curate images.
- Run `python scripts/retrain.py` to trigger retraining, or let CI/CD handle it on schedule.
- All changes to data/models are versioned with DVC and tracked in MLflow.
- Monitor retraining runs, parameters, and artifacts in the MLflow UI.

**Integrating with CI/CD:**
- The provided workflow file (`.github/workflows/retrain_schedule.yml`) enables fully automated, scheduled retraining as part of your CI/CD pipeline.
- You can also trigger retraining manually via the GitHub Actions UI or adapt the workflow for other CI/CD providers.

See the README and FINAL_PROJECT_SUMMARY.md for more details.
