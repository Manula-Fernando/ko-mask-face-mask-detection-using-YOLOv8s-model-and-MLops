# Docker Containerization Guide for Face Mask Detection Model

## 🐳 Overview

This guide explains how to containerize your Face Mask Detection model using Docker for consistent deployment across different environments. Your project already includes production-ready Docker configurations.

---

## 📁 Docker Files Structure

Your project includes these Docker-related files:
```
├── deployment/
│   └── Dockerfile              # Multi-stage production Dockerfile
├── docker-compose.yml          # Full stack orchestration
└── requirements.txt           # Python dependencies
```

---

## 🏗️ Dockerfile Architecture

### Multi-Stage Build Process

#### **Stage 1: Builder** 
- Uses Python 3.10.11-slim as base
- Installs build dependencies (OpenCV, ML libraries)
- Creates virtual environment
- Installs Python packages

#### **Stage 2: Runtime**
- Minimal runtime environment
- Copies only necessary components
- Creates non-root user for security
- Configures health checks

### Key Features:
- ✅ **Multi-stage build** for smaller final image
- ✅ **Security-focused** with non-root user
- ✅ **Health checks** for monitoring
- ✅ **Production-ready** with Gunicorn
- ✅ **Optimized layers** for faster builds

---

## 🚀 Quick Start - Containerizing Your Model

### Prerequisites
1. **Install Docker Desktop** (if not already installed):
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and restart your computer
   - **Start Docker Desktop** from Start Menu or Desktop icon

2. **Verify Docker is Running**:
```bash
# Check Docker version
docker --version

# Test Docker is working
docker run hello-world
```

### 1. **Build the Docker Image**

```bash
# Navigate to project root
cd c:\Users\wwmsf\Desktop\face-mask-detection-mlops

# Build the image (this will take a few minutes on first build)
docker build -f deployment/Dockerfile -t face-mask-detector:latest .
```

### 2. **Run Single Container**

```bash
# Run the containerized model
docker run -d \
  --name mask-detector \
  -p 5000:5000 \
  -v ${PWD}/logs:/app/logs \
  -v ${PWD}/models:/app/models \
  face-mask-detector:latest
```

### 3. **Using Docker Compose (Recommended)**

```bash
# Start the full stack (Web App + MLflow)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🔧 Docker Commands Reference

### Building Images
```bash
# Build development image
docker build -f deployment/Dockerfile -t face-mask-detector:dev .

# Build with specific tag
docker build -f deployment/Dockerfile -t face-mask-detector:v1.0 .

# Build with no cache (fresh build)
docker build --no-cache -f deployment/Dockerfile -t face-mask-detector:latest .
```

### Running Containers
```bash
# Run in background
docker run -d -p 5000:5000 face-mask-detector:latest

# Run with environment variables
docker run -d \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -e MODEL_PATH=/app/models/best_mask_detector_imbalance_optimized.h5 \
  face-mask-detector:latest

# Run with volume mounts
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  face-mask-detector:latest

# Run interactively (for debugging)
docker run -it \
  -p 5000:5000 \
  face-mask-detector:latest \
  /bin/bash
```

### Container Management
```bash
# List running containers
docker ps

# View container logs
docker logs mask-detector

# Execute commands in running container
docker exec -it mask-detector /bin/bash

# Stop container
docker stop mask-detector

# Remove container
docker rm mask-detector

# View container resource usage
docker stats mask-detector
```

---

## 📦 Docker Compose Services

### Full Stack Deployment
Your `docker-compose.yml` includes:

#### **Web Service** (Face Mask Detection API)
- **Port**: 5000
- **Features**: 
  - Health checks
  - Volume mounts for logs and models
  - Auto-restart policy
  - Production Gunicorn server

#### **MLflow Service** (Experiment Tracking)
- **Port**: 5001
- **Features**:
  - SQLite backend storage
  - Persistent volumes
  - Auto-restart policy
  - Health monitoring

### Docker Compose Commands
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d web

# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f web

# Scale web service
docker-compose up -d --scale web=3

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and start
docker-compose up -d --build
```

---

## 🔧 Configuration Options

### Environment Variables
```bash
# Flask Configuration
FLASK_ENV=production
FLASK_APP=app.main

# Model Configuration
MODEL_PATH=/app/models/best_mask_detector_imbalance_optimized.h5

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# Logging
LOG_LEVEL=INFO
```

### Volume Mounts
```bash
# Logs persistence
-v ./logs:/app/logs

# Model files
-v ./models:/app/models

# MLflow artifacts
-v mlflow_data:/mlflow
```

### Port Mapping
```bash
# Web application
-p 5000:5000

# MLflow tracking
-p 5001:5000

# Custom port
-p 8080:5000
```

---

## 🏥 Health Checks and Monitoring

### Container Health Check
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' mask-detector

# View health check logs
docker inspect mask-detector | grep -A 10 "Health"
```

### Application Health Endpoints
```bash
# Health check endpoint
curl http://localhost:5000/health

# Detailed health status
curl http://localhost:5000/health/detailed

# MLflow health
curl http://localhost:5001/health
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### **0. Docker Desktop Not Running**
```bash
# Error: "error during connect: ... dockerDesktopLinuxEngine: The system cannot find the file specified"

# Solution:
1. Start Docker Desktop from Windows Start Menu
2. Wait for Docker to fully start (whale icon in system tray)
3. Test with: docker run hello-world
4. If still issues, restart Docker Desktop
```

#### **1. Build Failures**
```bash
# Clear Docker cache
docker system prune -a

# Build with verbose output
docker build --progress=plain -f deployment/Dockerfile .

# Check disk space
docker system df
```

#### **2. Container Won't Start**
```bash
# Check container logs
docker logs mask-detector

# Run interactively to debug
docker run -it face-mask-detector:latest /bin/bash

# Check port conflicts
netstat -tulpn | grep :5000
```

#### **3. Model Loading Issues**
```bash
# Verify model file exists in container
docker exec mask-detector ls -la /app/models/

# Check file permissions
docker exec mask-detector ls -la /app/models/best_mask_detector_imbalance_optimized.h5

# Mount model directory
docker run -v $(pwd)/models:/app/models face-mask-detector:latest
```

#### **4. Memory Issues**
```bash
# Monitor container memory usage
docker stats mask-detector

# Increase container memory limit
docker run -m 4g face-mask-detector:latest

# Check available memory
docker system df
```

---

## 🚀 Production Deployment

### Best Practices

#### **1. Image Optimization**
```bash
# Use multi-stage builds (already implemented)
# Minimize layers
# Use .dockerignore to exclude unnecessary files
```

#### **2. Security**
```bash
# Run as non-root user (already implemented)
# Use specific base image versions
# Scan for vulnerabilities
docker scan face-mask-detector:latest
```

#### **3. Resource Management**
```bash
# Set memory limits
docker run -m 2g face-mask-detector:latest

# Set CPU limits
docker run --cpus="1.5" face-mask-detector:latest

# Set restart policy
docker run --restart=unless-stopped face-mask-detector:latest
```

### Cloud Deployment

#### **AWS ECR (Elastic Container Registry)**
```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Tag image
docker tag face-mask-detector:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/face-mask-detector:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/face-mask-detector:latest
```

#### **Google Container Registry**
```bash
# Tag for GCR
docker tag face-mask-detector:latest gcr.io/<project-id>/face-mask-detector:latest

# Push to GCR
docker push gcr.io/<project-id>/face-mask-detector:latest
```

---

## 📊 Monitoring and Logging

### Container Monitoring
```bash
# Real-time stats
docker stats

# Container processes
docker exec mask-detector ps aux

# Container network
docker exec mask-detector netstat -tulpn
```

### Application Logs
```bash
# View application logs
docker logs -f mask-detector

# MLflow logs
docker-compose logs -f mlflow

# Export logs
docker logs mask-detector > app.log 2>&1
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Build and Deploy Docker Image

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: |
        docker build -f deployment/Dockerfile -t face-mask-detector:${{ github.sha }} .
    
    - name: Run tests
      run: |
        docker run --rm face-mask-detector:${{ github.sha }} python -m pytest tests/
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push face-mask-detector:${{ github.sha }}
```

---

## 📈 Performance Optimization

### Image Size Optimization
```bash
# Check image size
docker images face-mask-detector

# Use multi-stage builds (implemented)
# Use Alpine base images for smaller size
# Remove package caches
```

### Runtime Performance
```bash
# Use Gunicorn with multiple workers (implemented)
# Configure appropriate worker count
# Set up load balancing for high traffic
```

---

## 🎯 Next Steps

1. **Test the Container**:
   ```bash
   docker-compose up -d
   curl http://localhost:5000/health
   ```

2. **Make Predictions**:
   ```bash
   curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
   ```

3. **Access MLflow**:
   ```bash
   open http://localhost:5001
   ```

4. **Monitor Logs**:
   ```bash
   docker-compose logs -f web
   ```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

Your Face Mask Detection model is now fully containerized and ready for deployment in any Docker-compatible environment! 🚀
