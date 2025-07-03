# Face Mask Detection MLOps Project - Makefile

.PHONY: help install train predict webapp mlflow test clean docker lint format

# Default target
help:
	@echo "🎯 Face Mask Detection MLOps Project"
	@echo "="
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the model"
	@echo "  make predict    - Run prediction analysis"
	@echo "  make webapp     - Start web application"
	@echo "  make mlflow     - Start MLflow UI"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean temporary files"
	@echo "  make docker     - Build Docker image"

# Environment setup
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Training
train:
	@echo "🚀 Starting model training..."
	python scripts/train_model.py

# Prediction analysis
predict:
	@echo "🔍 Running prediction analysis..."
	python scripts/run_prediction_analysis.py

# Web application
webapp:
	@echo "🌐 Starting web application..."
	python scripts/start_webapp.py

# MLflow UI
mlflow:
	@echo "📊 Starting MLflow UI..."
	python scripts/start_mlflow.py

# Testing
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Code quality
lint:
	@echo "🔍 Running linter..."
	flake8 src/ app/ scripts/ tests/

format:
	@echo "✨ Formatting code..."
	black src/ app/ scripts/ tests/
	isort src/ app/ scripts/ tests/

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Docker
docker:
	@echo "🐳 Building Docker image..."
	docker build -t face-mask-detection -f deployment/Dockerfile .

# Full pipeline
pipeline: clean install train predict
	@echo "✅ Full pipeline completed!"
