#!/usr/bin/env python3
"""
Main Training Script - Face Mask Detection
Execute the complete model training pipeline with MLflow tracking.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import and run training
from model_training import train_enhanced_model_with_comprehensive_mlflow

def main():
    """Execute model training with optimal settings."""
    print("🚀 Starting Face Mask Detection Training Pipeline")
    print("="*50)
    
    # Training configuration
    config = {
        "data_dir": "data/processed/splits",
        "model_save_path": "models/best_mask_detector.h5",
        "batch_size": 32,
        "epochs": 50,
        "use_class_weights": True
    }
    
    try:
        model, history = train_enhanced_model_with_comprehensive_mlflow(**config)
        print("✅ Training completed successfully!")
        print("📊 View results in MLflow UI: http://localhost:5000")
        return model, history
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
