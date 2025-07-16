"""
Data Drift Detection System for Face Mask Detection Model
Monitors input data distribution changes and model performance degradation
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import argparse
import warnings

from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class DataDriftDetector:
    """
    Advanced data drift detection system for monitoring input data changes
    """

    def __init__(
        self,
        reference_data_path: str = "data/processed/yolo_dataset/train",
        drift_threshold: float = 0.05,
        window_size: int = 1000,
        output_dir: str = "reports/drift_analysis"
    ):
        self.reference_data_path = Path(reference_data_path)
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Load reference data
        self.reference_data = self.load_images(self.reference_data_path)

        # Initialize drift history
        self.drift_history = []

        self.logger.info("Data drift detector initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("drift_detector")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_dir / "drift_detection.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def load_images(self, directory: Path, limit: int = 1000) -> Optional[np.ndarray]:
        """Load images from a directory as numpy arrays"""
        image_files = list(directory.glob("**/*.jpg")) + list(directory.glob("**/*.png"))
        if not image_files:
            self.logger.error(f"No images found in {directory}")
            return None
        images = []
        for img_path in image_files[:limit]:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                images.append(np.asarray(img) / 255.0)
            except Exception as e:
                self.logger.warning(f"Failed to load image {img_path}: {e}")
        if not images:
            self.logger.error(f"No valid images loaded from {directory}")
            return None
        images = np.stack(images)
        self.logger.info(f"Loaded {len(images)} images from {directory}")
        return images

    def extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """Extract statistical features from images for drift detection"""
        features = []
        for img in images:
            feature_vector = []
            feature_vector.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img),
                stats.skew(img.flatten()),
                stats.kurtosis(img.flatten())
            ])
            for channel in range(img.shape[-1]):
                channel_data = img[:, :, channel]
                feature_vector.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])
            hist, _ = np.histogram(img.flatten(), bins=10, range=(0, 1))
            hist_normalized = hist / np.sum(hist)
            feature_vector.extend(hist_normalized.tolist())
            features.append(feature_vector)
        return np.array(features)

    def detect_drift_statistical(self, current_data: np.ndarray, test_name: str = "ks") -> Dict[str, any]:
        """Detect drift using statistical tests"""
        if self.reference_data is None:
            return {"error": "No reference data available"}

        ref_features = self.extract_image_features(self.reference_data[:1000])
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
                statistic, p_value = ks_2samp(ref_feature, curr_feature)
                is_drift = p_value < self.drift_threshold
            elif test_name == "anderson":
                statistic, _, p_value = anderson_ksamp([ref_feature, curr_feature])
                is_drift = p_value < self.drift_threshold
            else:
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

        drift_percentage = (drift_count / n_features) * 100
        drift_results["overall_drift_score"] = total_drift_score / n_features
        drift_results["drift_detected"] = drift_percentage > 20
        drift_results["summary"] = {
            "total_features": n_features,
            "drifted_features": drift_count,
            "drift_percentage": drift_percentage,
            "severity": "high" if drift_percentage > 50 else "medium" if drift_percentage > 20 else "low"
        }
        return drift_results

    def detect_drift_pca(self, current_data: np.ndarray) -> Dict[str, any]:
        """Detect drift using PCA-based dimensionality reduction"""
        if self.reference_data is None:
            return {"error": "No reference data available"}
        ref_features = self.extract_image_features(self.reference_data[:1000])
        curr_features = self.extract_image_features(current_data)
        pca = PCA(n_components=min(10, ref_features.shape[1]))
        ref_pca = pca.fit_transform(ref_features)
        curr_pca = pca.transform(curr_features)
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "method": "pca_based",
            "pca_components": pca.n_components_,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "component_drifts": {}
        }
        total_drift = 0
        for i in range(pca.n_components_):
            statistic, p_value = ks_2samp(ref_pca[:, i], curr_pca[:, i])
            drift_results["component_drifts"][f"pc_{i}"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_drift": p_value < self.drift_threshold,
                "explained_variance": float(pca.explained_variance_ratio_[i])
            }
            total_drift += statistic * pca.explained_variance_ratio_[i]
        drift_results["weighted_drift_score"] = float(total_drift)
        drift_results["drift_detected"] = total_drift > 0.1
        return drift_results

    def create_drift_visualizations(self, current_data: np.ndarray, drift_results: Dict[str, any]) -> str:
        """Create comprehensive drift visualization dashboard"""
        if self.reference_data is None:
            return None
        ref_features = self.extract_image_features(self.reference_data[:500])
        curr_features = self.extract_image_features(current_data)
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
        if "feature_drifts" in drift_results:
            feature_names = list(drift_results["feature_drifts"].keys())
            drift_scores = [drift_results["feature_drifts"][f]["statistic"] for f in feature_names]
            fig.add_trace(
                go.Bar(
                    x=feature_names[:20],
                    y=drift_scores[:20],
                    name='Drift Scores',
                    marker_color=['red' if drift_results["feature_drifts"][f]["is_drift"]
                                  else 'blue' for f in feature_names[:20]]
                ),
                row=2, col=1
            )
        # Use correct timestamp key for drift history
        if self.drift_history:
            timestamps = [
                entry.get("analysis_timestamp", entry.get("timestamp", ""))
                for entry in self.drift_history[-50:]
            ]
            drift_scores = [
                entry.get("statistical_drift", {}).get("overall_drift_score", 0)
                for entry in self.drift_history[-50:]
            ]
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
        fig.update_layout(
            title="Data Drift Analysis Dashboard",
            height=1200,
            showlegend=True
        )
        output_path = self.output_dir / f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        self.logger.info(f"Drift visualization saved: {output_path}")
        return str(output_path)

    def generate_drift_report(self, current_data: np.ndarray, include_visualizations: bool = True) -> Dict[str, any]:
        """Generate comprehensive drift detection report"""
        self.logger.info("Starting drift detection analysis...")
        ks_results = self.detect_drift_statistical(current_data, "ks")
        pca_results = self.detect_drift_pca(current_data)
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
        self.drift_history.append(report)
        if include_visualizations:
            viz_path = self.create_drift_visualizations(current_data, ks_results)
            report["visualization_path"] = viz_path
        report_path = self.output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Drift report saved: {report_path}")
        return report

    def assess_overall_drift(self, individual_results: List[Dict]) -> Dict[str, any]:
        """Assess overall drift based on multiple detection methods"""
        drift_indicators = []
        confidence_scores = []
        for result in individual_results:
            if "drift_detected" in result:
                drift_indicators.append(result["drift_detected"])
                if "overall_drift_score" in result:
                    confidence_scores.append(result["overall_drift_score"])
                elif "weighted_drift_score" in result:
                    confidence_scores.append(result["weighted_drift_score"])
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
        """Generate actionable recommendations based on drift detection results"""
        recommendations = []
        if ks_results.get("drift_detected", False):
            severity = ks_results.get("summary", {}).get("severity", "low")
            if severity == "high":
                recommendations.extend([
                    "CRITICAL: Significant data drift detected. Immediate model retraining recommended.",
                    "Review data collection process for potential changes in input distribution.",
                    "Consider implementing online learning or model adaptation strategies."
                ])
            elif severity == "medium":
                recommendations.extend([
                    "Moderate drift detected. Monitor closely and prepare for model updates.",
                    "Increase monitoring frequency to track drift progression."
                ])
        if pca_results.get("drift_detected", False):
            recommendations.append(
                "PCA analysis indicates structural changes in data. Investigate feature engineering pipeline."
            )
        recommendations.extend([
            "Document current findings for drift tracking history.",
            "Consider adjusting drift detection thresholds based on business requirements.",
            "Set up automated alerts for future drift detection."
        ])
        if not (ks_results.get("drift_detected", False) or pca_results.get("drift_detected", False)):
            recommendations = [
                "No significant drift detected. Continue monitoring.",
                "Current model performance should remain stable.",
                "Next drift analysis recommended in 24-48 hours."
            ]
        return recommendations

def main():
    """
    Main function to run drift detection analysis
    """
    parser = argparse.ArgumentParser(description="Face Mask Detection Data Drift Detector")
    parser.add_argument("--reference-dir", type=str, default="data/processed/yolo_dataset/train",
                        help="Directory containing reference images")
    parser.add_argument("--current-dir", type=str, required=True,
                        help="Directory containing current images")
    parser.add_argument("--output-dir", type=str, default="reports/drift_analysis",
                        help="Directory to save drift reports and visualizations")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization (for headless/server use)")
    args = parser.parse_args()

    detector = DataDriftDetector(
        reference_data_path=args.reference_dir,
        output_dir=args.output_dir
    )

    # Load current data
    current_data = detector.load_images(Path(args.current_dir))
    if current_data is None:
        print(f"No valid images found in {args.current_dir}")
        return

    # Run drift detection
    print("Running drift detection analysis...")
    report = detector.generate_drift_report(current_data, include_visualizations=not args.no_viz)

    # Display results
    print("\n" + "="*80)
    print("DATA DRIFT DETECTION REPORT")
    print("="*80)
    print(f"\nDataset Information:")
    print(f"  • Reference dataset size: {report['dataset_info']['reference_size']}")
    print(f"  • Current dataset size: {report['dataset_info']['current_size']}")
    print(f"  • Drift threshold: {report['dataset_info']['drift_threshold']}")
    print(f"\nOverall Assessment:")
    assessment = report['overall_assessment']
    print(f"  • Drift detected: {'YES' if assessment['drift_detected'] else 'NO'}")
    print(f"  • Confidence score: {assessment['confidence_score']:.4f}")
    print(f"  • Severity: {assessment['severity'].upper()}")
    print(f"  • Consensus: {assessment['consensus'].upper()}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    if "visualization_path" in report:
        print(f"\nVisualization saved: {report['visualization_path']}")
    print("="*80)

if __name__ == "__main__":
    main()