"""
Face Mask Detection Data Drift Detector

This module detects data drift in input images and model performance
to ensure model reliability and trigger retraining when necessary.
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from scipy import stats
from sklearn.metrics import wasserstein_distance
import pickle

from ..common.logger import get_logger
from ..common.utils import load_config

logger = get_logger(__name__)


@dataclass
class DriftResult:
    """Container for drift detection results"""
    timestamp: float
    drift_type: str
    drift_detected: bool
    drift_score: float
    threshold: float
    feature_name: str
    reference_period: str
    current_period: str
    severity: str  # 'low', 'medium', 'high'
    recommendation: str


class ImageFeatureExtractor:
    """Extract statistical features from images for drift detection"""
    
    @staticmethod
    def extract_basic_features(image_path: str) -> Dict[str, float]:
        """Extract basic image features
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            features = {}
            
            # Basic statistics
            features['mean_brightness'] = float(np.mean(hsv[:, :, 2]))
            features['std_brightness'] = float(np.std(hsv[:, :, 2]))
            features['mean_saturation'] = float(np.mean(hsv[:, :, 1]))
            features['std_saturation'] = float(np.std(hsv[:, :, 1]))
            features['mean_hue'] = float(np.mean(hsv[:, :, 0]))
            features['std_hue'] = float(np.std(hsv[:, :, 0]))
            
            # Contrast
            features['contrast'] = float(np.std(gray))
            
            # Blur detection (variance of Laplacian)
            features['blur_score'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # Texture features (Local Binary Pattern approximation)
            features['texture_uniformity'] = ImageFeatureExtractor._calculate_texture_uniformity(gray)
            
            # Color distribution
            b, g, r = cv2.split(img)
            features['red_mean'] = float(np.mean(r))
            features['green_mean'] = float(np.mean(g))
            features['blue_mean'] = float(np.mean(b))
            features['red_std'] = float(np.std(r))
            features['green_std'] = float(np.std(g))
            features['blue_std'] = float(np.std(b))
            
            # Image dimensions
            h, w, c = img.shape
            features['aspect_ratio'] = float(w / h)
            features['image_size'] = float(h * w)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            return {}
    
    @staticmethod
    def _calculate_texture_uniformity(gray_image: np.ndarray) -> float:
        """Calculate texture uniformity using local variance"""
        try:
            # Calculate local variance using a sliding window
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            return float(np.mean(local_variance))
        except Exception:
            return 0.0


class DataDriftDetector:
    """Detect data drift using statistical methods"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize drift detector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.drift_config = self.config.get('monitoring', {}).get('drift_detection', {})
        
        # Drift detection thresholds
        self.ks_threshold = self.drift_config.get('ks_threshold', 0.05)
        self.wasserstein_threshold = self.drift_config.get('wasserstein_threshold', 0.1)
        self.psi_threshold = self.drift_config.get('psi_threshold', 0.2)
        
        # Reference data storage
        self.reference_data_path = self.drift_config.get('reference_data_path', 'logs/reference_data.pkl')
        self.reference_features = None
        
        # Create storage directory
        Path(self.reference_data_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing reference data if available
        self._load_reference_data()
        
        logger.info("DataDriftDetector initialized")
    
    def set_reference_data(self, image_paths: List[str]) -> None:
        """Set reference data for drift detection
        
        Args:
            image_paths: List of paths to reference images
        """
        try:
            logger.info(f"Extracting features from {len(image_paths)} reference images")
            
            reference_features = []
            for img_path in image_paths:
                features = ImageFeatureExtractor.extract_basic_features(img_path)
                if features:
                    reference_features.append(features)
            
            if not reference_features:
                raise ValueError("No valid features extracted from reference data")
            
            # Convert to DataFrame for easier analysis
            self.reference_features = pd.DataFrame(reference_features)
            
            # Save reference data
            self._save_reference_data()
            
            logger.info(f"Reference data set with {len(reference_features)} samples and {len(self.reference_features.columns)} features")
            
        except Exception as e:
            logger.error(f"Failed to set reference data: {e}")
            raise
    
    def detect_drift(self, image_paths: List[str], method: str = 'ks') -> List[DriftResult]:
        """Detect data drift in new images
        
        Args:
            image_paths: List of paths to new images
            method: Drift detection method ('ks', 'wasserstein', 'psi')
            
        Returns:
            List of drift detection results
        """
        if self.reference_features is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        try:
            # Extract features from new images
            current_features = []
            for img_path in image_paths:
                features = ImageFeatureExtractor.extract_basic_features(img_path)
                if features:
                    current_features.append(features)
            
            if not current_features:
                raise ValueError("No valid features extracted from current data")
            
            current_df = pd.DataFrame(current_features)
            
            # Ensure same columns
            common_columns = list(set(self.reference_features.columns) & set(current_df.columns))
            if not common_columns:
                raise ValueError("No common features between reference and current data")
            
            reference_subset = self.reference_features[common_columns]
            current_subset = current_df[common_columns]
            
            # Detect drift for each feature
            drift_results = []
            
            for feature in common_columns:
                if method == 'ks':
                    result = self._kolmogorov_smirnov_test(
                        reference_subset[feature].values,
                        current_subset[feature].values,
                        feature
                    )
                elif method == 'wasserstein':
                    result = self._wasserstein_distance_test(
                        reference_subset[feature].values,
                        current_subset[feature].values,
                        feature
                    )
                elif method == 'psi':
                    result = self._population_stability_index(
                        reference_subset[feature].values,
                        current_subset[feature].values,
                        feature
                    )
                else:
                    raise ValueError(f"Unknown drift detection method: {method}")
                
                drift_results.append(result)
            
            # Log summary
            drift_detected = sum(1 for r in drift_results if r.drift_detected)
            logger.info(f"Drift detection completed: {drift_detected}/{len(drift_results)} features show drift")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to detect drift: {e}")
            raise
    
    def _kolmogorov_smirnov_test(self, 
                                reference: np.ndarray, 
                                current: np.ndarray, 
                                feature_name: str) -> DriftResult:
        """Perform Kolmogorov-Smirnov test for drift detection"""
        try:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference, current)
            
            # Determine if drift is detected
            drift_detected = p_value < self.ks_threshold
            
            # Determine severity
            if p_value > 0.1:
                severity = 'low'
                recommendation = 'No action required'
            elif p_value > 0.05:
                severity = 'medium'
                recommendation = 'Monitor closely'
            else:
                severity = 'high'
                recommendation = 'Consider retraining model'
            
            return DriftResult(
                timestamp=datetime.now().timestamp(),
                drift_type='distribution',
                drift_detected=drift_detected,
                drift_score=ks_statistic,
                threshold=self.ks_threshold,
                feature_name=feature_name,
                reference_period=f"baseline_{len(reference)}_samples",
                current_period=f"current_{len(current)}_samples",
                severity=severity,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"KS test failed for {feature_name}: {e}")
            raise
    
    def _wasserstein_distance_test(self, 
                                  reference: np.ndarray, 
                                  current: np.ndarray, 
                                  feature_name: str) -> DriftResult:
        """Perform Wasserstein distance test for drift detection"""
        try:
            # Calculate Wasserstein distance
            distance = wasserstein_distance(reference, current)
            
            # Normalize by the range of reference data
            ref_range = np.max(reference) - np.min(reference)
            normalized_distance = distance / ref_range if ref_range > 0 else 0
            
            # Determine if drift is detected
            drift_detected = normalized_distance > self.wasserstein_threshold
            
            # Determine severity
            if normalized_distance < self.wasserstein_threshold / 2:
                severity = 'low'
                recommendation = 'No action required'
            elif normalized_distance < self.wasserstein_threshold:
                severity = 'medium'
                recommendation = 'Monitor closely'
            else:
                severity = 'high'
                recommendation = 'Consider retraining model'
            
            return DriftResult(
                timestamp=datetime.now().timestamp(),
                drift_type='distribution',
                drift_detected=drift_detected,
                drift_score=normalized_distance,
                threshold=self.wasserstein_threshold,
                feature_name=feature_name,
                reference_period=f"baseline_{len(reference)}_samples",
                current_period=f"current_{len(current)}_samples",
                severity=severity,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Wasserstein test failed for {feature_name}: {e}")
            raise
    
    def _population_stability_index(self, 
                                   reference: np.ndarray, 
                                   current: np.ndarray, 
                                   feature_name: str) -> DriftResult:
        """Calculate Population Stability Index (PSI) for drift detection"""
        try:
            # Create bins based on reference data quantiles
            n_bins = min(10, len(np.unique(reference)))
            bins = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bins)
            cur_counts, _ = np.histogram(current, bins=bins)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 1e-10, ref_props)
            cur_props = np.where(cur_props == 0, 1e-10, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            # Determine if drift is detected
            drift_detected = psi > self.psi_threshold
            
            # Determine severity
            if psi < 0.1:
                severity = 'low'
                recommendation = 'No action required'
            elif psi < 0.2:
                severity = 'medium'
                recommendation = 'Monitor closely'
            else:
                severity = 'high'
                recommendation = 'Consider retraining model'
            
            return DriftResult(
                timestamp=datetime.now().timestamp(),
                drift_type='distribution',
                drift_detected=drift_detected,
                drift_score=psi,
                threshold=self.psi_threshold,
                feature_name=feature_name,
                reference_period=f"baseline_{len(reference)}_samples",
                current_period=f"current_{len(current)}_samples",
                severity=severity,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"PSI calculation failed for {feature_name}: {e}")
            raise
    
    def detect_model_performance_drift(self, 
                                     recent_metrics: List[Dict[str, Any]], 
                                     historical_metrics: List[Dict[str, Any]]) -> List[DriftResult]:
        """Detect drift in model performance metrics
        
        Args:
            recent_metrics: Recent model performance metrics
            historical_metrics: Historical baseline metrics
            
        Returns:
            List of drift detection results for performance metrics
        """
        try:
            if not recent_metrics or not historical_metrics:
                return []
            
            # Convert to DataFrames
            recent_df = pd.DataFrame(recent_metrics)
            historical_df = pd.DataFrame(historical_metrics)
            
            # Performance metrics to check
            performance_columns = [
                'avg_confidence', 'avg_inference_time', 'accuracy', 
                'precision', 'recall', 'f1_score'
            ]
            
            drift_results = []
            
            for metric in performance_columns:
                if metric not in recent_df.columns or metric not in historical_df.columns:
                    continue
                
                # Remove null values
                recent_values = recent_df[metric].dropna().values
                historical_values = historical_df[metric].dropna().values
                
                if len(recent_values) == 0 or len(historical_values) == 0:
                    continue
                
                # Use KS test for performance drift
                result = self._kolmogorov_smirnov_test(
                    historical_values, recent_values, f"performance_{metric}"
                )
                drift_results.append(result)
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to detect performance drift: {e}")
            return []
    
    def _save_reference_data(self) -> None:
        """Save reference data to disk"""
        try:
            with open(self.reference_data_path, 'wb') as f:
                pickle.dump(self.reference_features, f)
            logger.info(f"Reference data saved to {self.reference_data_path}")
        except Exception as e:
            logger.error(f"Failed to save reference data: {e}")
    
    def _load_reference_data(self) -> None:
        """Load reference data from disk"""
        try:
            if os.path.exists(self.reference_data_path):
                with open(self.reference_data_path, 'rb') as f:
                    self.reference_features = pickle.load(f)
                logger.info(f"Reference data loaded from {self.reference_data_path}")
        except Exception as e:
            logger.warning(f"Failed to load reference data: {e}")
    
    def generate_drift_report(self, drift_results: List[DriftResult]) -> Dict[str, Any]:
        """Generate a comprehensive drift report
        
        Args:
            drift_results: List of drift detection results
            
        Returns:
            Drift report dictionary
        """
        try:
            if not drift_results:
                return {'error': 'No drift results provided'}
            
            # Overall summary
            total_features = len(drift_results)
            features_with_drift = sum(1 for r in drift_results if r.drift_detected)
            drift_percentage = (features_with_drift / total_features) * 100
            
            # Severity breakdown
            severity_counts = {}
            for result in drift_results:
                severity = result.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Feature-wise details
            feature_details = []
            for result in drift_results:
                feature_details.append({
                    'feature': result.feature_name,
                    'drift_detected': result.drift_detected,
                    'drift_score': result.drift_score,
                    'threshold': result.threshold,
                    'severity': result.severity,
                    'recommendation': result.recommendation
                })
            
            # Overall recommendation
            if features_with_drift == 0:
                overall_recommendation = "No drift detected. Model is stable."
            elif drift_percentage < 25:
                overall_recommendation = "Low drift detected. Continue monitoring."
            elif drift_percentage < 50:
                overall_recommendation = "Moderate drift detected. Consider retraining soon."
            else:
                overall_recommendation = "High drift detected. Immediate retraining recommended."
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_features': total_features,
                    'features_with_drift': features_with_drift,
                    'drift_percentage': drift_percentage,
                    'overall_recommendation': overall_recommendation
                },
                'severity_breakdown': severity_counts,
                'feature_details': feature_details,
                'high_priority_features': [
                    r.feature_name for r in drift_results 
                    if r.drift_detected and r.severity == 'high'
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            return {'error': str(e)}


def main():
    """CLI entry point for drift detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mask Detection Drift Detector')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--set-reference', type=str,
                       help='Directory containing reference images')
    parser.add_argument('--detect-drift', type=str,
                       help='Directory containing current images')
    parser.add_argument('--method', type=str, choices=['ks', 'wasserstein', 'psi'],
                       default='ks', help='Drift detection method')
    parser.add_argument('--output', type=str, help='Output file for drift report')
    
    args = parser.parse_args()
    
    detector = DataDriftDetector(args.config)
    
    if args.set_reference:
        # Set reference data
        ref_dir = Path(args.set_reference)
        image_paths = list(ref_dir.glob('*.jpg')) + list(ref_dir.glob('*.png'))
        
        if not image_paths:
            print(f"No images found in {ref_dir}")
            return
        
        detector.set_reference_data([str(p) for p in image_paths])
        print(f"Reference data set with {len(image_paths)} images")
        
    elif args.detect_drift:
        # Detect drift
        current_dir = Path(args.detect_drift)
        image_paths = list(current_dir.glob('*.jpg')) + list(current_dir.glob('*.png'))
        
        if not image_paths:
            print(f"No images found in {current_dir}")
            return
        
        drift_results = detector.detect_drift([str(p) for p in image_paths], args.method)
        
        # Generate report
        report = detector.generate_drift_report(drift_results)
        
        # Save or print report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Drift report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
    
    else:
        print("Please specify --set-reference or --detect-drift")


if __name__ == "__main__":
    main()
