# 🎯 Face Mask Detection - Production MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-red.svg)](https://dvc.org)
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey.svg)](https://flask.palletsprojects.com)

Production-ready MLOps pipeline for face mask detection using MobileNetV2 with comprehensive data versioning, experiment tracking, and deployment capabilities.

## 🚀 Features

- ✅ **Automated Data Pipeline**: Extraction, validation, and stratified splitting
- ✅ **Advanced Model Architecture**: MobileNetV2 with best-practice optimizations
- ✅ **Experiment Tracking**: Complete MLflow integration
- ✅ **Data Versioning**: DVC with Google Drive remote storage
- ✅ **Production API**: Flask deployment with real-time inference
- ✅ **Containerization**: Docker-ready for scalable deployment

## 📊 Dataset

- **Source**: [Andrew Ng Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data)
- **Images**: 853 images with PASCAL VOC annotations
- **Classes**: `with_mask`, `without_mask`, `mask_weared_incorrect`

## 🏗️ Architecture

```
face-mask-detection-mlops/
├── 📓 Complete_MLOps_Setup_Guide.ipynb   # Complete pipeline notebook
├── 🗂️ src/                               # Source code
│   ├── data_preprocessing.py             # Data processing pipeline
│   ├── model_training.py                 # Model training with MLflow
│   └── predict.py                        # Prediction utilities
├── 🌐 app/                               # Flask API
│   ├── main.py                          # API server
│   └── templates/index.html             # Web interface
├── 📁 data/                             # Data storage
│   ├── raw/                             # Original dataset
│   └── processed/                       # Processed splits
├── 🤖 models/                           # Trained models
├── 📊 mlruns/                           # MLflow experiments
├── 🐳 Dockerfile                        # Container configuration
└── 📋 requirements.txt                  # Dependencies
```

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd face-mask-detection-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

```bash
# Download dataset from Kaggle
# Save as: data/raw/images.zip
```

### 3. Training Pipeline

```bash
# Run the complete notebook
jupyter notebook Complete_MLOps_Setup_Guide.ipynb

# Or run individual components
python src/data_preprocessing.py
python src/model_training.py
```

### 4. Experiment Tracking

```bash
# Start MLflow UI
mlflow ui

# View at: http://localhost:5000
```

### 5. API Deployment

```bash
# Start Flask API
cd app && python main.py

# Access at: http://localhost:8000
```

### 6. Docker Deployment

```bash
# Build container
docker build -t facemask-api .

# Run container
docker run -p 8000:8000 facemask-api
```

## 🔧 Model Architecture

```python
MobileNetV2 (ImageNet weights)
├── GlobalAveragePooling2D()
├── BatchNormalization()
├── Dropout(0.5)
├── Dense(256, activation='relu')
├── BatchNormalization()
├── Dropout(0.3)
└── Dense(3, activation='softmax')
```

**Key Improvements:**
- GlobalAveragePooling2D instead of Flatten (reduces overfitting)
- Proper BatchNormalization placement
- Advanced callbacks (EarlyStopping, ReduceLROnPlateau)
- Class weight balancing for imbalanced datasets

## 📊 Performance Metrics

The model tracks comprehensive metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores  
- **Recall**: Per-class recall scores
- **AUC**: Area under the ROC curve
- **F1-Score**: Harmonic mean of precision and recall

## 🔄 MLOps Pipeline

### Data Versioning (DVC)
```bash
# Setup Google Drive remote
dvc remote add -d gdrive gdrive://your-folder-id
dvc push
```

### Experiment Tracking (MLflow)
- Automated parameter logging
- Model versioning and artifacts
- Performance comparison dashboard
- Training history visualization

### Continuous Integration
- Automated testing with pytest
- Model validation pipelines
- Performance regression detection

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Image prediction |
| `/health` | GET | Health check |
| `/api/info` | GET | API information |

### Example Usage

```python
import requests

# Upload image for prediction
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 🐳 Production Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  facemask-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: facemask-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: facemask-detection
  template:
    metadata:
      labels:
        app: facemask-detection
    spec:
      containers:
      - name: api
        image: facemask-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 📈 Monitoring & Observability

- **Health Checks**: Built-in health monitoring endpoints
- **Logging**: Structured logging with configurable levels
- **Metrics**: Custom metrics for model performance tracking
- **Alerts**: Integration-ready for monitoring systems

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_data_preprocessing.py
pytest tests/test_model_training.py
pytest tests/test_predict.py
pytest tests/test_api.py
```

## 📚 Documentation

- **API Documentation**: Interactive docs at `/docs` (when using FastAPI)
- **Model Documentation**: Detailed architecture and training process
- **Deployment Guide**: Step-by-step production deployment
- **Troubleshooting**: Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Andrew Ng Face Mask Detection Dataset
- **Framework**: TensorFlow/Keras for deep learning
- **MLOps**: MLflow for experiment tracking
- **Data Versioning**: DVC for data pipeline management

## 🔧 Troubleshooting

### Common Issues

1. **Model not loading**: Ensure model file exists in `models/` directory
2. **Dependencies missing**: Run `pip install -r requirements.txt`
3. **GPU issues**: Set `CUDA_VISIBLE_DEVICES=""` for CPU-only inference
4. **Port conflicts**: Change port in `app/main.py` if 8000 is occupied

### Performance Optimization

- **Batch Processing**: Use batch prediction for multiple images
- **Model Optimization**: Consider TensorRT or TensorFlow Lite for edge deployment
- **Caching**: Implement Redis caching for frequent predictions
- **Load Balancing**: Use nginx or similar for production traffic

---

**🌟 Ready for Production Deployment! 🌟**

For questions and support, please open an issue or contact the development team.
