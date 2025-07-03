# 🚀 Setup Guide - Face Mask Detection MLOps Project

## Quick Setup (5 minutes)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv face_mask_detection_venv

# Activate environment
.\face_mask_detection_venv\Scripts\activate  # Windows
# source face_mask_detection_venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Test model training (quick test)
python -c "from src.model_training import *; print('✅ Training module loaded')"

# Test web app
python -c "from app.main import app; print('✅ Web app loaded')"

# Test prediction
python -c "from src.predict import *; print('✅ Prediction module loaded')"
```

### 3. Train Your Model
```bash
# Start training with MLflow tracking
python scripts/train_model.py

# View results in MLflow UI
python scripts/start_mlflow.py
# Open: http://localhost:5000
```

### 4. Run Web Application
```bash
# Start Flask web app
python scripts/start_webapp.py
# Open: http://localhost:5000
```

## Using Makefile (Recommended)

If you have `make` installed:

```bash
# Install dependencies
make install

# Train model
make train

# Start web app
make webapp

# Start MLflow UI
make mlflow

# Run tests
make test

# Clean project
make clean
```

## Project Structure After Setup

```
📁 face-mask-detection-mlops/
├── 🔧 src/               # Core ML modules
├── 🌐 app/               # Web application  
├── 🛠️ scripts/          # Entry point scripts
├── 📊 data/              # Dataset (your data here)
├── 🤖 models/            # Trained models (generated)
├── 📈 mlruns/            # MLflow tracking (generated)
├── 🧪 tests/             # Unit tests
├── 🐳 deployment/        # Docker configs
├── 📚 docs/              # Documentation
├── 📋 requirements.txt   # Dependencies
└── 🚀 Makefile          # Easy commands
```

## Common Commands

| Task | Command |
|------|---------|
| **Train Model** | `python scripts/train_model.py` |
| **Web App** | `python scripts/start_webapp.py` |
| **MLflow UI** | `python scripts/start_mlflow.py` |
| **Prediction** | `python scripts/run_prediction_analysis.py` |
| **Webcam Demo** | `python scripts/run_simple_webcam.py` |
| **Tests** | `python -m pytest tests/ -v` |

## Troubleshooting

### Issue: Import Errors
```bash
# Make sure you're in the right directory
cd face-mask-detection-mlops

# Activate virtual environment
.\face_mask_detection_venv\Scripts\activate
```

### Issue: MLflow UI Won't Start
```bash
# Try different port
mlflow ui --host 127.0.0.1 --port 5001
```

### Issue: Missing Data
- Place your dataset in `data/raw/images/`
- Run data preprocessing if needed
- Check `data/processed/splits/` for train/val/test files

## Next Steps

1. ✅ **Setup Complete** - You're ready to go!
2. 🎯 **Train Model** - Run your first training
3. 🌐 **Try Web App** - Test the interface
4. 📊 **Explore MLflow** - View experiment tracking
5. 🔧 **Customize** - Adapt for your use case

## Need Help?

- 📖 Check [README.md](README.md) for full documentation
- 🏗️ Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details
- 🧪 Run tests to verify everything works
- 📊 Use MLflow UI to monitor experiments

**Happy coding! 🎉**
