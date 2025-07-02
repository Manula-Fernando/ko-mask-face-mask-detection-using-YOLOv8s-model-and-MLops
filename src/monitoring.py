"""
Monitoring module for face mask detection model.
This module handles model monitoring, drift detection, and performance tracking.
"""

import os
import json
import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.tracking
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_set import DataDriftMetricSet, DataQualityMetricSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Model monitoring and drift detection for face mask detection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model monitor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.monitoring_config = self.config['monitoring']
        self.mlflow_config = self.config['mlflow']
        
        # Create monitoring directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("monitoring", exist_ok=True)
        
        # Initialize tracking
        self.predictions_log = []
        self.performance_log = []
        
        logger.info("ModelMonitor initialized")
    
    def log_prediction(self, input_data: Dict[str, Any], 
                      prediction: Dict[str, Any], 
                      ground_truth: str = None):
        """Log a single prediction for monitoring."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'ground_truth': ground_truth,
            'input_shape': input_data.get('shape'),
            'confidence': prediction.get('confidence'),
            'predicted_class': prediction.get('predicted_class')
        }
        
        self.predictions_log.append(log_entry)
        
        # Save to file periodically
        if len(self.predictions_log) % 100 == 0:
            self._save_predictions_log()
    
    def _save_predictions_log(self):
        """Save predictions log to file."""
        log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.predictions_log, f, indent=2)
    
    def calculate_performance_metrics(self, y_true: List[str], 
                                    y_pred: List[str]) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Convert string labels to numeric
        label_map = {'No Mask': 0, 'Mask': 1}
        y_true_numeric = [label_map[label] for label in y_true]
        y_pred_numeric = [label_map[label] for label in y_pred]
        
        metrics = {
            'accuracy': accuracy_score(y_true_numeric, y_pred_numeric),
            'precision': precision_score(y_true_numeric, y_pred_numeric, average='weighted'),
            'recall': recall_score(y_true_numeric, y_pred_numeric, average='weighted'),
            'f1_score': f1_score(y_true_numeric, y_pred_numeric, average='weighted'),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.performance_log.append(metrics)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            for key, value in metrics.items():
                if key != 'timestamp':
                    mlflow.log_metric(f"monitoring_{key}", value)
        
        # Check for performance degradation
        self._check_performance_degradation(metrics)
        
        logger.info(f"Performance metrics logged: {metrics}")
    
    def _check_performance_degradation(self, current_metrics: Dict[str, float]):
        """Check for model performance degradation."""
        threshold = self.monitoring_config['performance_threshold']
        
        if current_metrics['accuracy'] < threshold:
            self._send_alert(
                f"Model accuracy dropped below threshold: {current_metrics['accuracy']:.3f} < {threshold}"
            )
        
        if current_metrics['f1_score'] < threshold:
            self._send_alert(
                f"Model F1-score dropped below threshold: {current_metrics['f1_score']:.3f} < {threshold}"
            )
    
    def detect_data_drift(self, reference_data: np.ndarray, 
                         current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using statistical tests."""
        try:
            # Flatten images for statistical analysis
            ref_features = reference_data.reshape(reference_data.shape[0], -1)
            curr_features = current_data.reshape(current_data.shape[0], -1)
            
            # Calculate basic statistics
            ref_mean = np.mean(ref_features, axis=0)
            curr_mean = np.mean(curr_features, axis=0)
            
            ref_std = np.std(ref_features, axis=0)
            curr_std = np.std(curr_features, axis=0)
            
            # Calculate drift metrics
            mean_drift = np.mean(np.abs(ref_mean - curr_mean))
            std_drift = np.mean(np.abs(ref_std - curr_std))
            
            drift_threshold = self.monitoring_config['drift_threshold']
            
            drift_result = {
                'mean_drift': float(mean_drift),
                'std_drift': float(std_drift),
                'drift_detected': mean_drift > drift_threshold or std_drift > drift_threshold,
                'threshold': drift_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log drift metrics
            with mlflow.start_run(run_name=f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_metric("mean_drift", drift_result['mean_drift'])
                mlflow.log_metric("std_drift", drift_result['std_drift'])
                mlflow.log_metric("drift_detected", int(drift_result['drift_detected']))
            
            if drift_result['drift_detected']:
                self._send_alert(f"Data drift detected: mean_drift={mean_drift:.3f}, std_drift={std_drift:.3f}")
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            return {'error': str(e), 'drift_detected': False}
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        logger.info(f"Generating monitoring report for last {days} days")
        
        # Load recent predictions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            pred for pred in self.predictions_log
            if datetime.fromisoformat(pred['timestamp']) > cutoff_date
        ]
        
        # Load recent performance metrics
        recent_metrics = [
            metric for metric in self.performance_log
            if datetime.fromisoformat(metric['timestamp']) > cutoff_date
        ]
        
        # Calculate summary statistics
        total_predictions = len(recent_predictions)
        
        if total_predictions > 0:
            avg_confidence = np.mean([pred['confidence'] for pred in recent_predictions])
            
            class_distribution = {}
            for pred in recent_predictions:
                class_name = pred['predicted_class']
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        else:
            avg_confidence = 0
            class_distribution = {}
        
        # Performance trends
        if recent_metrics:
            latest_accuracy = recent_metrics[-1]['accuracy']
            avg_accuracy = np.mean([m['accuracy'] for m in recent_metrics])
        else:
            latest_accuracy = 0
            avg_accuracy = 0
        
        report = {
            'period': f"Last {days} days",
            'total_predictions': total_predictions,
            'average_confidence': float(avg_confidence),
            'class_distribution': class_distribution,
            'latest_accuracy': float(latest_accuracy),
            'average_accuracy': float(avg_accuracy),
            'performance_metrics_count': len(recent_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_file = f"monitoring/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._generate_monitoring_plots(recent_predictions, recent_metrics)
        
        logger.info(f"Monitoring report generated: {report_file}")
        return report
    
    def _generate_monitoring_plots(self, predictions: List[Dict], 
                                 metrics: List[Dict]):
        """Generate monitoring visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Prediction confidence distribution
        if predictions:
            confidences = [pred['confidence'] for pred in predictions]
            axes[0, 0].hist(confidences, bins=20, alpha=0.7)
            axes[0, 0].set_title('Prediction Confidence Distribution')
            axes[0, 0].set_xlabel('Confidence')
            axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Class distribution
        if predictions:
            classes = [pred['predicted_class'] for pred in predictions]
            class_counts = pd.Series(classes).value_counts()
            axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Class Distribution')
        
        # Plot 3: Performance trends
        if metrics:
            timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics]
            accuracies = [m['accuracy'] for m in metrics]
            axes[1, 0].plot(timestamps, accuracies, marker='o')
            axes[1, 0].set_title('Accuracy Trend')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Accuracy')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Predictions over time
        if predictions:
            timestamps = [datetime.fromisoformat(pred['timestamp']) for pred in predictions]
            daily_counts = pd.Series(timestamps).dt.date.value_counts().sort_index()
            axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o')
            axes[1, 1].set_title('Daily Prediction Volume')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Number of Predictions')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_file = f"monitoring/plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        plt.close()
        
        # Log plots to MLflow
        with mlflow.start_run(run_name=f"monitoring_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact(plot_file)
    
    def _send_alert(self, message: str):
        """Send monitoring alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'alert_type': 'model_monitoring'
        }
        
        # Log alert
        logger.warning(f"ALERT: {message}")
        
        # Save to alerts file
        alerts_file = "logs/alerts.json"
        alerts = []
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # In a real system, you would send email/Slack notification here
        # self._send_email_alert(message)
        # self._send_slack_alert(message)

def main():
    """Example usage of the model monitor."""
    monitor = ModelMonitor()
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report(days=7)
    print(f"Monitoring report: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    main()
