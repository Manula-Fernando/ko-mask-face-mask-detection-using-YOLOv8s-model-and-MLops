#!/usr/bin/env python3
"""
Run Face Mask Detection Prediction Analysis on Real Data for MLflow
This script runs the prediction analysis on actual test images from the dataset.
"""

import os
import sys
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import glob
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')
from predict import FaceMaskPredictor

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('prediction_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def find_test_images():
    """Find real test images from the dataset."""
    logger = logging.getLogger(__name__)
    
    # Look for images in data directory
    image_paths = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    # Search in different possible locations
    search_dirs = [
        'data/raw/images',
        'data/processed',
        'high_confidence_detections'
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ext in image_extensions:
                pattern = os.path.join(search_dir, '**', ext)
                found_images = glob.glob(pattern, recursive=True)
                image_paths.extend(found_images)
                logger.info(f"Found {len(found_images)} {ext} files in {search_dir}")
    
    # Remove duplicates and get absolute paths
    image_paths = list(set([os.path.abspath(p) for p in image_paths]))
    
    logger.info(f"Total unique images found: {len(image_paths)}")
    return image_paths

def run_prediction_analysis():
    """Run comprehensive prediction analysis on real test data."""
    logger = setup_logging()
    
    # Set MLflow experiment
    mlflow.set_experiment("Face_Mask_Detection_Prediction_Analysis_REAL")
    
    # Check if model exists
    model_path = "models/best_mask_detector_imbalance_optimized.h5"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        logger.error("No test images found! Cannot run prediction analysis.")
        return
    
    # Limit to reasonable number for demo (take first 50)
    if len(test_images) > 50:
        test_images = test_images[:50]
        logger.info(f"Limited to {len(test_images)} images for analysis")
    
    with mlflow.start_run(run_name=f"Prediction_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        logger.info("Starting Real Prediction Analysis for MLflow")
        
        # Initialize predictor
        predictor = FaceMaskPredictor(model_path)
        
        # Load model (this will create its own MLflow run)
        predictor.load_model()
        
        if predictor.model is None:
            logger.error("Failed to load model!")
            return
        
        # Log analysis parameters
        mlflow.log_param("total_test_images", len(test_images))
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("analysis_type", "real_data_prediction")
        
        # Process images in batches to avoid memory issues
        batch_size = 10
        all_results = []
        
        logger.info(f"Processing {len(test_images)} images in batches of {batch_size}...")
        
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_images)-1)//batch_size + 1}")
            
            # Process batch
            for img_path in batch_images:
                try:
                    result = predictor.predict(img_path)
                    if 'error' not in result:
                        all_results.append(result)
                        logger.debug(f"Predicted {result['prediction']} with {result['confidence']:.3f} confidence for {os.path.basename(img_path)}")
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
        
        logger.info(f"Successfully processed {len(all_results)} images")
        
        if all_results:
            # Calculate aggregate metrics
            predictions = [r['prediction'] for r in all_results]
            confidences = [r['confidence'] for r in all_results]
            inference_times = [r['inference_time_ms'] for r in all_results]
            
            # Class distribution
            class_counts = pd.Series(predictions).value_counts()
            
            # Log aggregate metrics
            mlflow.log_metric("total_successful_predictions", len(all_results))
            mlflow.log_metric("avg_confidence", np.mean(confidences))
            mlflow.log_metric("min_confidence", np.min(confidences))
            mlflow.log_metric("max_confidence", np.max(confidences))
            mlflow.log_metric("std_confidence", np.std(confidences))
            mlflow.log_metric("avg_inference_time_ms", np.mean(inference_times))
            mlflow.log_metric("total_inference_time_ms", np.sum(inference_times))
            
            # Log class distributions
            for class_name, count in class_counts.items():
                mlflow.log_metric(f"count_{class_name}", count)
                mlflow.log_metric(f"pct_{class_name}", count / len(all_results) * 100)
            
            # High/low confidence analysis
            high_conf_threshold = 0.9
            low_conf_threshold = 0.7
            
            high_conf_count = sum(1 for c in confidences if c >= high_conf_threshold)
            low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)
            
            mlflow.log_metric("high_confidence_predictions", high_conf_count)
            mlflow.log_metric("low_confidence_predictions", low_conf_count)
            mlflow.log_metric("high_confidence_rate", high_conf_count / len(confidences) * 100)
            mlflow.log_metric("low_confidence_rate", low_conf_count / len(confidences) * 100)
            
            # Save detailed results
            results_df = pd.DataFrame(all_results)
            results_csv = "prediction_analysis_results.csv"
            results_df.to_csv(results_csv, index=False, encoding='utf-8')
            mlflow.log_artifact(results_csv)
            
            # Generate and log analytics visualization
            predictor.log_analytics()
            
            logger.info("Prediction analysis completed successfully!")
            logger.info(f"Summary: {len(all_results)} predictions, avg confidence: {np.mean(confidences):.3f}")
            logger.info(f"Class distribution: {dict(class_counts)}")
            
        else:
            logger.error("No successful predictions made!")
            mlflow.log_metric("total_successful_predictions", 0)

if __name__ == "__main__":
    run_prediction_analysis()
