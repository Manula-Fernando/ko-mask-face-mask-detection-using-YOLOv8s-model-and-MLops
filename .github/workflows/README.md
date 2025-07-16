# GitHub Workflows Documentation

## Overview
This directory contains GitHub Actions workflows for the Face Mask Detection MLOps pipeline. These workflows provide automated CI/CD, testing, deployment, and model management.

## Active Workflows

### 1. `ci.yml` - Continuous Integration Pipeline ✅
**Purpose**: Code quality, linting, type checking, security scanning, unit testing, Docker build/test

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs**:
1. **Quality Tests** - Code linting, formatting, type checking, security scanning
2. **Unit Tests** - Pytest with coverage
3. **Docker Build** - Container image building and testing
4. **Security Scan** - Vulnerability scanning with Trivy

**Key Features**:
- Multi-Python version testing (3.8, 3.9, 3.10)
- Fast feedback for code quality and test failures
- Docker containerization validation
- Security scanning

### 2. `retraining-cd.yml` - Model Retraining & Deployment Pipeline ✅
**Purpose**: Automated model retraining, evaluation, and deployment

**Triggers**:
- Weekly schedule (Sunday 2 AM UTC)
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual dispatch with force retrain option

**Key Features**:
- Collects all data from `data/collected/`, merges and splits into `data/processed/yolo_dataset`
- Model training with MLflow tracking
- Model validation and testing
- Performance comparison with previous models
- Automated model deployment if performance improves
- Drift detection and data quality monitoring

## Workflow Integration with Your Project

### Required for Your MLOps Pipeline:
✅ **Both workflows are essential** for a complete MLOps setup:

1. **`ci.yml`** - Handles code quality, testing, and Docker validation
2. **`retraining-cd.yml`** - Manages model lifecycle, retraining, and deployment

### How They Work Together:
```
Code Push → ci.yml (CI)
     ↓
Quality Tests → Unit Tests → Docker Build/Test → Security Scan
     ↓
Weekly Schedule or Manual/Push → retraining-cd.yml
     ↓
Data Collection → Data Merge/Split → Model Training → Model Evaluation → Model Deployment
```

## Required GitHub Secrets
To use these workflows, configure these secrets in your GitHub repository:

```bash
GITHUB_TOKEN        # Automatically provided by GitHub
MLFLOW_TRACKING_URI # MLflow server URL (if external)
DOCKER_REGISTRY_URL # Container registry URL
DOCKER_USERNAME     # Container registry username
DOCKER_PASSWORD     # Container registry password
SLACK_WEBHOOK_URL   # Slack notifications (optional)
```

## Customization Points

### Environment-Specific Configuration:
- Modify `env` section in workflows for your environment
- Update deployment commands in the `deploy` job
- Configure notification endpoints

### Testing Configuration:
- Adjust Python versions in strategy matrix
- Modify performance thresholds
- Add/remove testing steps as needed

### Deployment Targets:
- Update staging/production deployment commands
- Configure Kubernetes/Docker deployment scripts
- Set up environment-specific configurations

## Benefits for Your Project

### 1. **Automated Quality Assurance**
- Code formatting and linting
- Type checking and security scanning
- Unit and integration testing
- Docker and security validation

### 2. **Continuous Deployment**
- Automated Docker image building
- Environment-specific deployments
- Rollback capabilities

### 3. **Model Lifecycle Management**
- Automated retraining based on data drift
- Model performance tracking
- Model versioning with MLflow

### 4. **Monitoring and Alerting**
- Pipeline status notifications
- Performance degradation alerts
- Security vulnerability reporting

## Usage Instructions

### For Development:
1. **Push code** to `develop` branch - triggers quality tests
2. **Create PR** to `main` - runs full test suite
3. **Merge to main** - deploys to staging automatically (if configured)

### For Production:
1. **Manual dispatch** `retraining-cd.yml` with force retrain if needed
2. **Monitor** automated retraining weekly
3. **Review** model performance in MLflow

### For Emergency Retraining:
1. **Manual dispatch** `retraining-cd.yml` with force retrain
2. **Monitor** training progress in workflow logs
3. **Validate** new model performance before deployment

## Maintenance

### Regular Tasks:
- Review and update Python versions in matrix
- Update dependencies in workflow files
- Monitor workflow execution times
- Adjust performance thresholds as needed

### Security:
- Regularly update action versions (e.g., `actions/checkout@v4`)
- Review and rotate secrets
- Monitor security scan results

These workflows provide a production-ready MLOps pipeline that aligns with industry best practices and supports your face mask detection project's requirements.
