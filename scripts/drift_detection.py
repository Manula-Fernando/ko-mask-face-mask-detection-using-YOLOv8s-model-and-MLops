"""
Data Drift Detection System for Face Mask Detection Model
Monitors input data distribution changes and model performance degradation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mlflow
import sqlite3
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DataDriftDetector:
    """
    Advanced data drift detection system for monitoring input data changes
    """
    
    def __init__(self, 
                 reference_data_path: str = "data/processed/train",
                 drift_threshold: float = 0.05,
                 window_size: int = 1000,
                 output_dir: str = "reports/drift_analysis"):
        
        self.reference_data_path = Path(reference_data_path)
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load reference data
        self.reference_data = self.load_reference_data()
        
        # Initialize drift history
        self.drift_history = []
        
        self.logger.info("✅ Data drift detector initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("drift_detector")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "drift_detection.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def load_reference_data(self) -> Optional[np.ndarray]:
        """Load reference/baseline data for comparison"""
        try:
            # In a real implementation, you would load your training data
            # For demo purposes, we'll create synthetic reference data
            np.random.seed(42)
            reference_data = np.random.normal(0.5, 0.2, (5000, 224, 224, 3))
            reference_data = np.clip(reference_data, 0, 1)
            
            self.logger.info(f"📊 Reference data loaded: {reference_data.shape}")
            return reference_data
            
        except Exception as e:
            self.logger.error(f"Failed to load reference data: {e}")
            return None

    def extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """Extract statistical features from images for drift detection"""
        
        features = []
        
        for img in images:
            # Basic statistical features
            feature_vector = []
            
            # Global statistics
            feature_vector.extend([
                np.mean(img),           # Overall mean
                np.std(img),            # Overall std
                np.min(img),            # Min value
                np.max(img),            # Max value
                np.median(img),         # Median
                stats.skew(img.flatten()),    # Skewness
                stats.kurtosis(img.flatten()) # Kurtosis
            ])
            
            # Per-channel statistics
            for channel in range(img.shape[-1]):
                channel_data = img[:, :, channel]
                feature_vector.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])
            
            # Histogram features
            hist, _ = np.histogram(img.flatten(), bins=10, range=(0, 1))
            hist_normalized = hist / np.sum(hist)
            feature_vector.extend(hist_normalized.tolist())
            
            features.append(feature_vector)
        
        return np.array(features)

    def detect_drift_statistical(self, 
                                current_data: np.ndarray,
                                test_name: str = "ks") -> Dict[str, any]:
        """
        Detect drift using statistical tests
        """
        if self.reference_data is None:
            return {"error": "No reference data available"}
        
        # Extract features from both datasets
        ref_features = self.extract_image_features(self.reference_data[:1000])  # Sample for efficiency
        curr_features = self.extract_image_features(current_data)
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "test_method": test_name,
            "drift_detected": False,
            "feature_drifts": {},
            "overall_drift_score": 0.0,
            "summary": {}
        }
        
        n_features = ref_features.shape[1]
        drift_count = 0
        total_drift_score = 0.0
        
        for i in range(n_features):
            ref_feature = ref_features[:, i]
            curr_feature = curr_features[:, i]
            
            if test_name == "ks":
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(ref_feature, curr_feature)
                is_drift = p_value < self.drift_threshold
                
            elif test_name == "anderson":
                # Anderson-Darling test
                statistic, _, p_value = anderson_ksamp([ref_feature, curr_feature])
                is_drift = p_value < self.drift_threshold
                
            else:
                # Default to KS test
                statistic, p_value = ks_2samp(ref_feature, curr_feature)
                is_drift = p_value < self.drift_threshold
            
            drift_results["feature_drifts"][f"feature_{i}"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_drift": is_drift,
                "drift_magnitude": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
            }
            
            if is_drift:
                drift_count += 1
            
            total_drift_score += statistic
        
        # Overall assessment
        drift_percentage = (drift_count / n_features) * 100
        drift_results["overall_drift_score"] = total_drift_score / n_features
        drift_results["drift_detected"] = drift_percentage > 20  # Alert if >20% features drift
        
        drift_results["summary"] = {
            "total_features": n_features,
            "drifted_features": drift_count,
            "drift_percentage": drift_percentage,
            "severity": "high" if drift_percentage > 50 else "medium" if drift_percentage > 20 else "low"
        }
        
        return drift_results

    def detect_drift_pca(self, current_data: np.ndarray) -> Dict[str, any]:
        """
        Detect drift using PCA-based dimensionality reduction
        """
        if self.reference_data is None:
            return {"error": "No reference data available"}
        
        # Extract features
        ref_features = self.extract_image_features(self.reference_data[:1000])
        curr_features = self.extract_image_features(current_data)
        
        # Apply PCA
        pca = PCA(n_components=min(10, ref_features.shape[1]))
        ref_pca = pca.fit_transform(ref_features)
        curr_pca = pca.transform(curr_features)
        
        # Compare distributions in PCA space
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "method": "pca_based",
            "pca_components": pca.n_components_,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "component_drifts": {}
        }
        
        total_drift = 0
        for i in range(pca.n_components_):
            # Test each PCA component
            statistic, p_value = ks_2samp(ref_pca[:, i], curr_pca[:, i])
            
            drift_results["component_drifts"][f"pc_{i}"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_drift": p_value < self.drift_threshold,
                "explained_variance": float(pca.explained_variance_ratio_[i])
            }
            
            total_drift += statistic * pca.explained_variance_ratio_[i]
        
        drift_results["weighted_drift_score"] = float(total_drift)
        drift_results["drift_detected"] = total_drift > 0.1  # Threshold for PCA-based drift
        
        return drift_results

    def create_drift_visualizations(self, 
                                  current_data: np.ndarray,
                                  drift_results: Dict[str, any]) -> str:
        """
        Create comprehensive drift visualization dashboard
        """
        if self.reference_data is None:
            return None
        
        # Extract features for visualization
        ref_features = self.extract_image_features(self.reference_data[:500])
        curr_features = self.extract_image_features(current_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Feature Distribution Comparison", "PCA Projection",
                "Drift Scores by Feature", "Temporal Drift Trend",
                "Sample Images - Reference", "Sample Images - Current"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "image"}, {"type": "image"}]
            ]
        )
        
        # 1. Feature distribution comparison (first few features)
        for i in range(min(3, ref_features.shape[1])):
            fig.add_trace(
                go.Histogram(
                    x=ref_features[:, i],
                    name=f"Ref Feature {i}",
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(
                    x=curr_features[:, i],
                    name=f"Curr Feature {i}",
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
        
        # 2. PCA projection
        pca = PCA(n_components=2)
        ref_pca = pca.fit_transform(ref_features)
        curr_pca = pca.transform(curr_features)
        
        fig.add_trace(
            go.Scatter(
                x=ref_pca[:, 0], y=ref_pca[:, 1],
                mode='markers',
                name='Reference Data',
                marker=dict(color='blue', size=4, opacity=0.6)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=curr_pca[:, 0], y=curr_pca[:, 1],
                mode='markers',
                name='Current Data',
                marker=dict(color='red', size=4, opacity=0.6)
            ),
            row=1, col=2
        )
        
        # 3. Drift scores by feature
        if "feature_drifts" in drift_results:
            feature_names = list(drift_results["feature_drifts"].keys())
            drift_scores = [drift_results["feature_drifts"][f]["statistic"] 
                          for f in feature_names]
            
            fig.add_trace(
                go.Bar(
                    x=feature_names[:20],  # Show first 20 features
                    y=drift_scores[:20],
                    name='Drift Scores',
                    marker_color=['red' if drift_results["feature_drifts"][f]["is_drift"] 
                                else 'blue' for f in feature_names[:20]]
                ),
                row=2, col=1
            )
        
        # 4. Temporal drift trend (if we have history)
        if self.drift_history:
            timestamps = [entry["timestamp"] for entry in self.drift_history[-50:]]
            drift_scores = [entry.get("overall_drift_score", 0) for entry in self.drift_history[-50:]]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=drift_scores,
                    mode='lines+markers',
                    name='Drift Trend',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Data Drift Analysis Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Save visualization
        output_path = self.output_dir / f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        
        self.logger.info(f"📊 Drift visualization saved: {output_path}")
        return str(output_path)

    def generate_drift_report(self, 
                            current_data: np.ndarray,
                            include_visualizations: bool = True) -> Dict[str, any]:
        """
        Generate comprehensive drift detection report
        """
        
        self.logger.info("🔍 Starting drift detection analysis...")
        
        # Perform different drift detection methods
        ks_results = self.detect_drift_statistical(current_data, "ks")
        pca_results = self.detect_drift_pca(current_data)
        
        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "reference_size": len(self.reference_data) if self.reference_data is not None else 0,
                "current_size": len(current_data),
                "drift_threshold": self.drift_threshold
            },
            "statistical_drift": ks_results,
            "pca_drift": pca_results,
            "overall_assessment": self.assess_overall_drift([ks_results, pca_results]),
            "recommendations": self.generate_recommendations(ks_results, pca_results)
        }
        
        # Add to drift history
        self.drift_history.append(report)
        
        # Create visualizations
        if include_visualizations:
            viz_path = self.create_drift_visualizations(current_data, ks_results)
            report["visualization_path"] = viz_path
        
        # Save report
        report_path = self.output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Drift report saved: {report_path}")
        
        return report

    def assess_overall_drift(self, individual_results: List[Dict]) -> Dict[str, any]:
        """
        Assess overall drift based on multiple detection methods
        """
        
        drift_indicators = []
        confidence_scores = []
        
        for result in individual_results:
            if "drift_detected" in result:
                drift_indicators.append(result["drift_detected"])
                
                # Calculate confidence based on method
                if "overall_drift_score" in result:
                    confidence_scores.append(result["overall_drift_score"])
                elif "weighted_drift_score" in result:
                    confidence_scores.append(result["weighted_drift_score"])
        
        # Overall assessment
        drift_detected = sum(drift_indicators) >= len(drift_indicators) / 2
        confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        severity = "low"
        if confidence > 0.3:
            severity = "high"
        elif confidence > 0.1:
            severity = "medium"
        
        return {
            "drift_detected": drift_detected,
            "confidence_score": float(confidence),
            "severity": severity,
            "methods_agreeing": sum(drift_indicators),
            "total_methods": len(drift_indicators),
            "consensus": "strong" if sum(drift_indicators) == len(drift_indicators) else 
                        "weak" if sum(drift_indicators) == 0 else "mixed"
        }

    def generate_recommendations(self, ks_results: Dict, pca_results: Dict) -> List[str]:
        """
        Generate actionable recommendations based on drift detection results
        """
        
        recommendations = []
        
        # Check statistical drift
        if ks_results.get("drift_detected", False):
            severity = ks_results.get("summary", {}).get("severity", "low")
            
            if severity == "high":
                recommendations.extend([
                    "🚨 CRITICAL: Significant data drift detected. Immediate model retraining recommended.",
                    "📊 Review data collection process for potential changes in input distribution.",
                    "🔄 Consider implementing online learning or model adaptation strategies."
                ])
            elif severity == "medium":
                recommendations.extend([
                    "⚠️ Moderate drift detected. Monitor closely and prepare for model updates.",
                    "📈 Increase monitoring frequency to track drift progression."
                ])
        
        # Check PCA drift
        if pca_results.get("drift_detected", False):
            recommendations.append(
                "🔍 PCA analysis indicates structural changes in data. "
                "Investigate feature engineering pipeline."
            )
        
        # General recommendations
        recommendations.extend([
            "📝 Document current findings for drift tracking history.",
            "🔧 Consider adjusting drift detection thresholds based on business requirements.",
            "📊 Set up automated alerts for future drift detection."
        ])
        
        if not (ks_results.get("drift_detected", False) or pca_results.get("drift_detected", False)):
            recommendations = [
                "✅ No significant drift detected. Continue monitoring.",
                "📊 Current model performance should remain stable.",
                "🔄 Next drift analysis recommended in 24-48 hours."
            ]
        
        return recommendations

def main():
    """
    Main function to run drift detection analysis
    """
    
    # Initialize drift detector
    detector = DataDriftDetector()
    
    # Simulate current production data (in practice, this would come from your API logs)
    print("🔬 Generating simulated production data...")
    np.random.seed(123)  # Different seed to simulate drift
    current_data = np.random.normal(0.6, 0.25, (100, 224, 224, 3))  # Slight distribution shift
    current_data = np.clip(current_data, 0, 1)
    
    # Run drift detection
    print("🔍 Running drift detection analysis...")
    report = detector.generate_drift_report(current_data)
    
    # Display results
    print("\n" + "="*80)
    print("🔍 DATA DRIFT DETECTION REPORT")
    print("="*80)
    
    print(f"\n📊 Dataset Information:")
    print(f"  • Reference dataset size: {report['dataset_info']['reference_size']}")
    print(f"  • Current dataset size: {report['dataset_info']['current_size']}")
    print(f"  • Drift threshold: {report['dataset_info']['drift_threshold']}")
    
    print(f"\n🎯 Overall Assessment:")
    assessment = report['overall_assessment']
    print(f"  • Drift detected: {'🚨 YES' if assessment['drift_detected'] else '✅ NO'}")
    print(f"  • Confidence score: {assessment['confidence_score']:.4f}")
    print(f"  • Severity: {assessment['severity'].upper()}")
    print(f"  • Consensus: {assessment['consensus'].upper()}")
    
    print(f"\n📋 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    if "visualization_path" in report:
        print(f"\n📊 Visualization saved: {report['visualization_path']}")
    
    print("="*80)

if __name__ == "__main__":
    main()
