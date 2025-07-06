# Inference Module Structure

This directory contains all code for running face mask detection inference in both programmatic (Python/CLI) and API (FastAPI) modes.

## Main Components

- **predictor.py**: Contains the single, unified `FaceMaskPredictor` class. This is the only inference class you should use for all scripts, CLI, and API endpoints. It supports model loading, single/batch prediction, and model info.
- **api.py**: FastAPI application for serving inference as a web API. All endpoints use `FaceMaskPredictor` for predictions. This is the only API you need for production.
- **service.py**: (Legacy) Previously used for inference logic. You may keep for reference, but all new code should use `FaceMaskPredictor`.
- **inference_service.py**: (Legacy) Old FastAPI app using a different pipeline/config. Safe to delete after confirming all needed logic is in `api.py`.
- **__init__.py**: Module marker.

## Usage

- **Python/CLI:**
  ```python
  from src.inference.predictor import FaceMaskPredictor
  predictor = FaceMaskPredictor()
  result = predictor.predict(image)
  ```
- **API:**
  Run `api.py` with Uvicorn. All endpoints use the unified predictor class.

## Maintenance
- Only maintain and improve `FaceMaskPredictor` for all inference logic.
- Remove or archive legacy files after migration.
- Keep this README up to date with any changes to the inference workflow.
