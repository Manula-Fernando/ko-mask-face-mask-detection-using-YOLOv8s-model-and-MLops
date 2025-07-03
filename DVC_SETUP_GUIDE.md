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

# Google Drive (YOUR ACTUAL FOLDER)
dvc remote add -d myremote gdrive://1xbvj8QxoaOSgXUf935QvClsR-Zgykc4T

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

## Google Drive Setup (Your Configuration)

### Your Google Drive Folder
- **Folder URL**: https://drive.google.com/drive/u/0/folders/1xbvj8QxoaOSgXUf935QvClsR-Zgykc4T
- **Folder ID**: `1xbvj8QxoaOSgXUf935QvClsR-Zgykc4T`

### Step-by-Step Setup
```bash
# 1. Add your Google Drive as DVC remote
dvc remote add -d myremote gdrive://1xbvj8QxoaOSgXUf935QvClsR-Zgykc4T

# 2. Commit the DVC configuration
git add .dvc/config
git commit -m "Add Google Drive as DVC remote storage"

# 3. Push your data to Google Drive (first time authentication required)
dvc push

# 4. Verify the upload
dvc status -c
```

### First-Time Authentication
When you run `dvc push` for the first time:
1. Your browser will open automatically
2. Sign in to your Google account
3. Grant DVC permission to access Google Drive
4. DVC will store the authentication token locally

### Verification
After setup, check your Google Drive folder:
- You'll see a `files/` directory with hash-named subdirectories
- Your actual data files are stored with hash names
- DVC maps these hashes to your original files

## Troubleshooting Google Drive Authentication

### Issue: "This app is blocked" Error
If you encounter the error "This app is blocked" when trying to authenticate with Google Drive, follow these steps:

#### Option 1: Try a Different Browser (Quick Fix)
Sometimes the default browser blocks DVC authentication. Try these browsers:

1. **Copy the authentication URL** from the terminal output
2. **Open in a different browser**:
   - If using Edge, try Chrome or Firefox
   - If using Chrome, try Edge or Firefox
   - Try an incognito/private window
3. **Paste the URL** and complete authentication
4. **Copy the authorization code** back to your terminal

```bash
# The URL will look like this:
# https://accounts.google.com/o/oauth2/auth?client_id=710796635688-iivsgbgsb6uv1fap6635dhvuei09o66c.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2F&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.appdata&access_type=offline&response_type=code&approval_prompt=force

# After successful authentication, you'll get a code to paste back in terminal
```

#### Option 2: Enable Less Secure App Access (Temporary Solution)
1. Go to your Google Account settings: https://myaccount.google.com/
2. Navigate to "Security" → "Less secure app access"
3. Turn on "Allow less secure apps"
4. Try `dvc push` again

#### Option 2: Use Service Account (Recommended for Production)
```bash
# 1. Create a service account in Google Cloud Console
# 2. Download the JSON key file
# 3. Configure DVC to use service account
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path /path/to/service-account.json
```

#### Option 3: Alternative Storage Solutions
If Google Drive continues to cause issues, consider these alternatives:

```bash
# AWS S3 (if you have AWS account)
dvc remote add -d myremote s3://your-bucket-name/dvcstore

# Local network storage (for testing)
dvc remote add -d myremote /path/to/shared/storage

# GitHub LFS (for smaller files)
git lfs track "*.h5"
git add .gitattributes
```

#### Option 4: Local Storage Solution (RECOMMENDED FOR NOW)
If Google Drive authentication continues to fail, use local storage:

```bash
# Remove the problematic Google Drive remote
dvc remote remove myremote

# Add local storage (works immediately, no authentication needed)
dvc remote add -d myremote C:\dvc-storage

# Create the storage directory
mkdir C:\dvc-storage

# Test the setup
dvc push
```

This creates a local DVC storage that:
- ✅ Works immediately without authentication
- ✅ Allows full DVC functionality for development
- ✅ Can be moved to cloud storage later
- ✅ Perfect for local development and testing

### Verification Steps
After successful authentication:
```bash
# Check remote status
dvc remote list

# Verify connection
dvc status -c

# Test with a small file first
echo "test" > test.txt
dvc add test.txt
dvc push test.txt.dvc
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
