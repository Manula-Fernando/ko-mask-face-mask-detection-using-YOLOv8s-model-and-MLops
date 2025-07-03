#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive MLflow integration
across all three core pipeline files: data_preprocessing, model_training, and predict
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_preprocessing_mlflow():
    """Test data preprocessing with MLflow tracking."""
    logger.info("🧪 Testing Data Preprocessing MLflow Integration")
    
    try:
        from data_preprocessing import DataProcessor
        
        # Setup paths
        raw_data_dir = Path("data/raw")
        processed_data_dir = Path("data/processed")
        
        if not raw_data_dir.exists():
            logger.warning("⚠️ Raw data directory not found, skipping data preprocessing test")
            return False
        
        # Initialize processor
        processor = DataProcessor(raw_data_dir, processed_data_dir)
        
        # Create dummy data for testing
        dummy_data = {
            'image_path': [f'test_image_{i}.jpg' for i in range(100)],
            'class_id': np.random.choice([0, 1, 2], size=100),
            'bbox': [(10, 10, 50, 50) for _ in range(100)]
        }
        df = pd.DataFrame(dummy_data)
        
        # Test MLflow logging methods
        processor.log_data_quality_metrics(df)
        logger.info("✅ Data preprocessing MLflow integration tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data preprocessing test failed: {e}")
        return False

def test_model_training_mlflow():
    """Test model training with MLflow tracking."""
    logger.info("🧪 Testing Model Training MLflow Integration")
    
    try:
        from model_training import MaskDetectorTrainer
        
        # Initialize trainer
        trainer = MaskDetectorTrainer()
        
        # Create dummy training data
        dummy_train_data = {
            'image_path': [f'train_image_{i}.jpg' for i in range(50)],
            'class_id': np.random.choice([0, 1, 2], size=50)
        }
        train_df = pd.DataFrame(dummy_train_data)
        
        # Test class weights calculation and logging
        class_weights = trainer.calculate_class_weights(train_df)
        trainer.log_class_weights(class_weights)
        
        # Test model architecture logging (create a simple model)
        model = trainer.build_model()
        trainer.log_model_architecture(model)
        
        logger.info("✅ Model training MLflow integration tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training test failed: {e}")
        return False

def test_prediction_mlflow():
    """Test prediction with MLflow tracking."""
    logger.info("🧪 Testing Prediction MLflow Integration")
    
    try:
        from predict import FaceMaskPredictor
        
        # Check if model exists
        model_path = "models/best_mask_detector_imbalance_optimized.h5"
        if not os.path.exists(model_path):
            logger.warning("⚠️ Model file not found, creating dummy predictor for testing")
            # Create dummy predictor without loading model
            predictor = FaceMaskPredictor.__new__(FaceMaskPredictor)
            predictor.model_path = model_path
            predictor.model = None
            predictor.logger = logger
            predictor.prediction_history = []
            predictor.performance_metrics = {
                'total_predictions': 10,
                'inference_times': [0.1, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.10, 0.14, 0.12],
                'confidence_scores': [0.95, 0.87, 0.92, 0.78, 0.98, 0.85, 0.91, 0.88, 0.93, 0.89],
                'class_predictions': {'with_mask': 6, 'without_mask': 3, 'mask_weared_incorrect': 1}
            }
            predictor.class_confidence_history = {
                'with_mask': [0.95, 0.92, 0.98, 0.91, 0.93, 0.89],
                'without_mask': [0.87, 0.78, 0.85],
                'mask_weared_incorrect': [0.88]
            }
            predictor.setup_mlflow()
        else:
            # Use actual predictor
            predictor = FaceMaskPredictor(model_path)
            
            # Generate some dummy performance data
            predictor.performance_metrics = {
                'total_predictions': 10,
                'inference_times': [0.1, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.10, 0.14, 0.12],
                'confidence_scores': [0.95, 0.87, 0.92, 0.78, 0.98, 0.85, 0.91, 0.88, 0.93, 0.89],
                'class_predictions': {'with_mask': 6, 'without_mask': 3, 'mask_weared_incorrect': 1}
            }
            predictor.class_confidence_history = {
                'with_mask': [0.95, 0.92, 0.98, 0.91, 0.93, 0.89],
                'without_mask': [0.87, 0.78, 0.85],
                'mask_weared_incorrect': [0.88]
            }
        
        # Test prediction session logging
        session_info = {
            'session_start': datetime.now().isoformat(),
            'model_path': model_path,
            'total_images_processed': 10,
            'session_duration_minutes': 5.2
        }
        
        predictor.log_prediction_session(session_info)
        logger.info("✅ Prediction MLflow integration tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all MLflow integration tests."""
    logger.info("🚀 Starting MLflow Integration Tests")
    logger.info("=" * 60)
    
    results = {
        'data_preprocessing': False,
        'model_training': False,
        'prediction': False
    }
    
    # Test each component
    results['data_preprocessing'] = test_data_preprocessing_mlflow()
    results['model_training'] = test_model_training_mlflow()
    results['prediction'] = test_prediction_mlflow()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 MLflow Integration Test Results:")
    logger.info("=" * 60)
    
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{component.replace('_', ' ').title()}: {status}")
    
    overall_success = all(results.values())
    
    if overall_success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("🔍 Check your MLflow UI to see the logged experiments:")
        logger.info("   - Face_Mask_Detection_Data_Preprocessing")
        logger.info("   - Face_Mask_Detection_Model_Training")
        logger.info("   - Face_Mask_Detection_Predictions")
        logger.info("\n🚀 To view MLflow UI, run: mlflow ui")
        logger.info("📊 Then open: http://localhost:5000")
    else:
        failed_tests = [k for k, v in results.items() if not v]
        logger.error(f"\n❌ Some tests failed: {', '.join(failed_tests)}")
        logger.info("🔧 Check the error messages above for troubleshooting")
    
    return overall_success

if __name__ == "__main__":
    print("🧪 MLflow Integration Test Suite")
    print("🎯 Testing comprehensive MLflow tracking across all pipeline components")
    print("=" * 70)
    
    success = main()
    
    if success:
        print("\n✨ MLflow integration is working perfectly!")
        print("🎯 Your pipeline will now provide rich insights in MLflow")
    else:
        print("\n🛠️ Some issues detected. Please review and fix before proceeding.")
    
    sys.exit(0 if success else 1)
