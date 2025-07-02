# Trained Models Directory

This directory contains the trained face mask detection models.

## Files:
- `mask_detector.h5` - Main trained model (TensorFlow/Keras format)
- `haarcascade_frontalface_default.xml` - OpenCV face detection cascade

## Usage:
Models are versioned using DVC and tracked in MLflow model registry.

To add new models to version control:
```bash
dvc add models/mask_detector.h5
git add models/mask_detector.h5.dvc
git commit -m "Add new model version"
```
