# Face Mask Detection MLOps Project - Makefile

.PHONY: help install train predict webapp mlflow test clean docker lint format pipeline compose dvc notebooks

# Default target
help:
	@echo "🎯 Face Mask Detection MLOps Project"
	@echo "="
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make train          - Train the model (src/training/train.py)"
	@echo "  make predict        - Run prediction analysis (src/inference/predictor.py)"
	@echo "  make webapp         - Start FastAPI web application (src/inference/api.py)"
	@echo "  make mlflow         - Start MLflow UI"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run code linting (flake8)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make clean          - Clean temporary files"
	@echo "  make docker         - Build Docker image"
	@echo "  make compose        - Run docker-compose stack"
	@echo "  make dvc            - Run DVC pipeline"
	@echo "  make notebooks      - Lint Jupyter notebooks"
	@echo "  make pipeline       - Full pipeline: clean, install, train, predict"

# Environment setup
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Training
train:
	@echo "🚀 Starting model training..."
	python src/training/train.py

# Prediction analysis
predict:
	@echo "🔍 Running prediction analysis..."
	python src/inference/predictor.py --analyze

# Web application (FastAPI)
webapp:
	@echo "🌐 Starting FastAPI web application..."
	python src/inference/api.py

# MLflow UI
mlflow:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

# Testing
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

# Code quality
lint:
	@echo "🔍 Running linter..."
	flake8 src/ tests/

format:
	@echo "✨ Formatting code..."
	black src/ tests/
	isort src/ tests/

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
# Build only the main Docker image (for local dev)
docker:
	@echo "🐳 Building Docker image..."
	docker build -t face-mask-detection -f deployment/Dockerfile.inference .

# Docker Compose stack
compose:
	@echo "🐳 Starting full MLOps stack with docker-compose..."
	cd deployment && docker-compose up --build

# DVC pipeline
# Run the full DVC pipeline
dvc:
	@echo "🔄 Running DVC pipeline..."
	dvc repro

# Lint all Jupyter notebooks
notebooks:
	@echo "🔍 Linting Jupyter notebooks..."
	python -m nbqa flake8 notebooks/

# Full pipeline
pipeline: clean install train predict
	@echo "✅ Full pipeline completed!"
