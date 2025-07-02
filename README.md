# Face Mask Detection MLOps Pipeline

[![CI/CD Pipeline](https://github.com/username/face-mask-detection-mlops/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/username/face-mask-detection-mlops/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.7+-green.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com/)

A complete MLOps implementation for real-time face mask detection using deep learning, featuring experiment tracking, model versioning, CI/CD pipelines, and production deployment.

## 🎯 Project Overview

This project demonstrates enterprise-grade MLOps practices for deploying a face mask detection system with:

- **High-accuracy model** (>95%) using MobileNetV2 architecture
- **Complete MLOps pipeline** with MLflow and DVC
- **CI/CD automation** with GitHub Actions
- **Production deployment** with Docker and Flask
- **Real-time monitoring** and drift detection
- **Comprehensive documentation** and reproducibility

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Model Layer   │    │   Service Layer │
│                 │    │                 │    │                 │
│ • Raw Images    │───▶│ • Preprocessing │───▶│ • Flask Web App │
│ • DVC Tracking  │    │ • MobileNetV2   │    │ • REST API      │
│ • Versioning    │    │ • MLflow Track  │    │ • Webcam Stream │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Infrastructure  │    │   Monitoring    │    │     CI/CD       │
│                 │    │                 │    │                 │
│ • Docker        │    │ • Model Metrics │    │ • GitHub Actions│
│ • Compose       │    │ • Drift Detection│    │ • Auto Testing  │
│ • Kubernetes    │    │ • Logging       │    │ • Auto Deploy   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
face-mask-detection-mlops/
├── .dvc/                   # DVC metadata
├── .github/                # GitHub-specific files
│   └── workflows/
│       └── main.yml        # CI/CD workflow
├── .dockerignore
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py             # Flask application
│   └── templates/
│       └── index.html
├── config/
│   └── config.yaml         # Configuration file
├── data/
│   ├── processed/          # Processed data
│   └── raw/                # Raw data (add to DVC)
│       ├── with_mask/      # Images with masks
│       └── without_mask/   # Images without masks
├── models/                 # Trained models (add to DVC)
├── notebooks/
│   └── model_development_report.ipynb # Project report
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── model_training.py   # Model training pipeline
│   ├── predict.py          # Prediction functions
│   └── monitoring.py       # Model monitoring
├── tests/
│   ├── __init__.py
│   └── test_app.py         # Unit tests
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── dvc.yaml                # DVC pipeline definition
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- DVC (optional)

### 1. Clone Repository

```bash
git clone https://github.com/username/face-mask-detection-mlops.git
cd face-mask-detection-mlops
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Data

```bash
# Create data directories
mkdir -p data/raw/with_mask data/raw/without_mask

# Add your training images:
# - data/raw/with_mask/ (images of people wearing masks)
# - data/raw/without_mask/ (images of people not wearing masks)
```

### 4. Configure MLflow

```bash
# Start MLflow server
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

### 5. Train Model

```bash
# Preprocess data
python src/data_preprocessing.py

# Train model with MLflow tracking
python src/model_training.py
```

### 6. Run Web Application

```bash
# Start Flask app
python app/main.py
# Open http://localhost:8080 in your browser
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t face-mask-detection .

# Run container
docker run -p 8080:8080 face-mask-detection

# Or use Docker Compose
docker-compose up -d
```

### Access Services

- **Web App:** http://localhost:8080
- **MLflow UI:** http://localhost:5000
- **API Health:** http://localhost:8080/health

## 🔬 MLOps Features

### Experiment Tracking

- **MLflow Integration:** Track experiments, parameters, metrics
- **Model Registry:** Version and stage models
- **Artifact Storage:** Store models, plots, and data
- **Reproducibility:** Consistent environment and results

### Data Management

- **DVC Pipeline:** Version control for data and models
- **Data Validation:** Automated quality checks
- **Preprocessing:** Standardized data preparation
- **Augmentation:** Enhance training data diversity

### CI/CD Pipeline

- **Automated Testing:** Unit tests, integration tests
- **Code Quality:** Linting, formatting, type checking
- **Security Scanning:** Vulnerability detection
- **Deployment:** Automated container builds and deployment

### Model Monitoring

- **Performance Tracking:** Real-time accuracy monitoring
- **Data Drift Detection:** Input distribution changes
- **Alert System:** Automated notifications
- **Logging:** Comprehensive prediction logging

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | >95% |
| Precision | >93% |
| Recall | >94% |
| F1-Score | >93% |
| Inference Time | <50ms |

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ app/ tests/

# Format code
black src/ app/ tests/
```

### Project Commands

```bash
# Data preprocessing
python src/data_preprocessing.py

# Model training
python src/model_training.py --experiment-name "my_experiment"

# Model evaluation
python src/predict.py

# Run monitoring
python src/monitoring.py

# Start web app
python app/main.py
```

## 📈 Monitoring & Metrics

### Health Check

```bash
curl http://localhost:8080/health
```

### Metrics Endpoint

```bash
curl http://localhost:8080/metrics
```

### MLflow UI

Access experiment tracking at http://localhost:5000

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Model parameters
- Training settings
- Data paths
- MLflow configuration
- Deployment settings

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v --cov=src --cov=app
```

### Test Categories

- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end workflow testing
- **API Tests:** REST endpoint validation
- **Model Tests:** Model performance validation

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | GET/POST | Image upload and prediction |
| `/api/predict` | POST | REST API for predictions |
| `/webcam` | GET | Real-time webcam detection |
| `/health` | GET | System health check |
| `/metrics` | GET | Performance metrics |

### Example API Usage

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post(
    "http://localhost:8080/api/predict",
    json={"image": image_data}
)

result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 🚀 Deployment Options

### Local Development

```bash
python app/main.py
```

### Docker

```bash
docker-compose up -d
```

### Cloud Deployment

- **AWS:** ECS, EKS, Lambda
- **Google Cloud:** Cloud Run, GKE
- **Azure:** Container Instances, AKS
- **Heroku:** Container deployment

## 🔒 Security

- Input validation and sanitization
- Rate limiting for API endpoints
- Container security scanning
- Dependency vulnerability checks
- No sensitive data logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Python:** PEP 8 compliance
- **Formatting:** Black code formatter
- **Linting:** Flake8 standards
- **Type Hints:** MyPy type checking
- **Testing:** Pytest with >80% coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **MLflow Community** for experiment tracking tools
- **DVC Team** for data version control
- **Flask Community** for the web framework
- **Open Source Contributors** for various tools and libraries

## 📞 Support

For questions and support:

- **GitHub Issues:** [Report bugs and feature requests](https://github.com/username/face-mask-detection-mlops/issues)
- **Documentation:** Check the [Wiki](https://github.com/username/face-mask-detection-mlops/wiki)
- **Email:** support@example.com

## 📈 Roadmap

- [ ] Multi-class mask type detection
- [ ] Mobile app deployment
- [ ] Real-time video stream processing
- [ ] Cloud-native deployment
- [ ] Advanced drift detection methods
- [ ] Edge device optimization

---

**Built with ❤️ for public health and safety**
