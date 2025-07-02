# Face Mask Detection Dataset Information

## Dataset Structure
- **with_mask/**: Images of people wearing face masks
- **without_mask/**: Images of people not wearing face masks

## Instructions
1. Add your image files to the respective directories
2. Use `dvc add data/raw` to track the data
3. Use `dvc push` to upload to Google Drive remote storage

## Example Commands
```bash
# Track the data directory
dvc add data/raw

# Commit the .dvc file to git
git add data/raw.dvc .gitignore
git commit -m "Add face mask detection dataset"

# Push data to remote storage
dvc push
```
