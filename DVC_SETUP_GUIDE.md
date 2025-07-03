# DVC Configuration for Professional Face Mask Detection MLOps Project

## Overview
Data Version Control (DVC) is used to track and version our datasets, models, and large files while keeping Git repositories clean and fast.

## Setup DVC

### Installation
```bash
python -m pip install dvc[all]  # Full installation with all remote storage support
```

### Initialize DVC
```bash
python -m dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Data Versioning Strategy

### 1. Track Data Files
```bash
# Track raw data
python -m dvc add data/raw/

# Track processed data  
python -m dvc add data/processed/

# Track trained models
python -m dvc add models/best_mask_detector_imbalance_optimized.h5

# Commit DVC files to Git
git add data/raw.dvc .dvc/.gitignore data/.gitignore models/best_mask_detector_imbalance_optimized.h5.dvc
git commit -m "Add DVC tracking for data and models

- Track raw dataset with DVC for version control
- Track trained model files with DVC
- Enable data versioning and reproducibility
- Implement professional-grade MLOps data management"
```

### 2. Remote Storage Setup
```bash
# Example with AWS S3
dvc remote add -d myremote s3://my-bucket/dvcstore

# Example with Google Drive
dvc remote add -d myremote gdrive://folder-id

# Example with local remote for testing
dvc remote add -d myremote /tmp/dvcstore

# Configure credentials (if needed)
dvc remote modify myremote access_key_id mykey
dvc remote modify myremote secret_access_key mysecret
```

### 3. Push/Pull Data
```bash
# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

## DVC Pipeline Configuration

The project uses `dvc.yaml` to define reproducible ML pipelines:

```yaml
stages:
  data_preprocessing:
    cmd: python src/data_preprocessing.py
    deps:
    - src/data_preprocessing.py
    - data/raw/
    outs:
    - data/processed/

  train:
    cmd: python src/model_training.py
    deps:
    - src/model_training.py
    - data/processed/
    params:
    - train.batch_size
    - train.epochs
    - train.learning_rate
    outs:
    - models/best_mask_detector_imbalance_optimized.h5
    metrics:
    - metrics.json

  evaluate:
    cmd: python src/evaluate.py
    deps:
    - src/evaluate.py
    - models/best_mask_detector_imbalance_optimized.h5
    - data/processed/test/
    metrics:
    - evaluation.json
```

## Benefits of DVC Integration

1. **Data Versioning**: Track changes to datasets and models
2. **Reproducibility**: Recreate exact experiments with specific data versions
3. **Collaboration**: Share data efficiently across team members
4. **Storage Optimization**: Keep large files out of Git while maintaining version control
5. **Pipeline Management**: Define and reproduce ML pipelines consistently

## Common DVC Commands

```bash
# Check status
dvc status

# Reproduce pipeline
dvc repro

# Show data/model lineage
dvc dag

# Compare experiments
dvc metrics show
dvc metrics diff

# Track experiment parameters
dvc params show
dvc params diff
```

## Integration with Git

DVC integrates seamlessly with Git:
- `.dvc` files are tracked in Git (small metadata files)
- Actual data/models are stored in DVC remote storage
- Git branches can have different data versions
- Data lineage is preserved across Git commits

This setup ensures our Professional Face Mask Detection project has enterprise-grade data and model versioning capabilities.
