"""
Production Model Monitoring System for Face Mask Detection
Tracks model performance, data drift, and system health in real-time
"""

import os
import json
import logging
import time
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3
import hashlib
import threading
import requests
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from numpy.typing import NDArray
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class PredictionLog:
    """Structure for logging predictions"""
    timestamp: datetime
    input_hash: str
    prediction: str
    confidence: float
    processing_time: float
    model_version: str
    user_feedback: Optional[str] = None

@dataclass
class ModelMetrics:
    """Structure for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    prediction_count: int
    timestamp: datetime

class ModelMonitor:
    """
    Comprehensive model monitoring system with drift detection,
    performance tracking, and alerting capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "face_mask_detector",
                 db_path: str = "logs/model_monitoring.db",
                 mlflow_uri: str = "http://localhost:5001",
                 alert_thresholds: Optional[Dict[str, float]] = None):
        
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.mlflow_uri = mlflow_uri
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,  # Alert if accuracy drops by 5%
            'drift_p_value': 0.05,  # Statistical significance for drift
            'confidence_drop': 0.1,  # Alert if avg confidence drops by 10%
            'error_rate': 0.1,      # Alert if error rate exceeds 10%
            'response_time': 2.0    # Alert if response time > 2 seconds
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Load baseline metrics
        self.baseline_metrics = self.load_baseline_metrics()
        
        self.logger.info("Model monitoring system initialized")

    def setup_logging(self):
        """Configure comprehensive logging (prevents duplicate handlers)"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{self.model_name}_monitor")
        self.logger.setLevel(logging.INFO)
        # Prevent duplicate handlers
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_dir / "model_monitoring.log", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def init_database(self):
        """Initialize SQLite database for logging"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    user_feedback TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    avg_confidence REAL,
                    prediction_count INTEGER,
                    period TEXT
                )
            ''')
            
            # Drift detection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    p_value REAL NOT NULL,
                    is_drift BOOLEAN NOT NULL,
                    baseline_period TEXT,
                    current_period TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_timestamp TEXT
                )
            ''')
            
            conn.commit()

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment("Face_Mask_Detection_Monitoring")
            self.logger.info(f"MLflow connected: {self.mlflow_uri}")
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")
            self.logger.info("Continuing without MLflow tracking")

    def log_prediction(self, 
                      input_data: NDArray[Any],
                      prediction: str,
                      confidence: float,
                      processing_time: float,
                      model_version: str = "v1.0") -> None:
        """Log a single prediction with metadata"""
        
        # Create input hash for deduplication and tracking
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()[:16]
        
        prediction_log = PredictionLog(
            timestamp=datetime.now(),
            input_hash=input_hash,
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time,
            model_version=model_version
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, input_hash, prediction, confidence, processing_time, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_log.timestamp.isoformat(),
                prediction_log.input_hash,
                prediction_log.prediction,
                prediction_log.confidence,
                prediction_log.processing_time,
                prediction_log.model_version
            ))
            conn.commit()
        
        # Log to MLflow if available
        if mlflow.active_run() or mlflow.get_tracking_uri():
            try:
                with mlflow.start_run(run_name=f"prediction_{input_hash}", nested=True):
                    mlflow.log_metric("confidence", confidence)
                    mlflow.log_metric("processing_time", processing_time)
                    mlflow.log_param("prediction", prediction)
                    mlflow.log_param("model_version", model_version)
            except Exception as e:
                self.logger.warning(f"MLflow logging failed: {e}")

    def calculate_performance_metrics(self, period_hours: int = 24) -> Optional[ModelMetrics]:
        """Calculate model performance metrics for a given period"""
        
        start_time = datetime.now() - timedelta(hours=period_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT prediction, confidence, user_feedback
                FROM predictions
                WHERE timestamp >= ?
            '''
            df = pd.read_sql_query(query, conn, params=[start_time.isoformat()])
        
        if df.empty:
            return None
        
        avg_confidence = df['confidence'].mean()
        prediction_count = len(df)
        
        # Simulated metrics (replace with real ground truth in production)
        accuracy = 0.85 + np.random.normal(0, 0.05)
        precision = 0.83 + np.random.normal(0, 0.05)
        recall = 0.87 + np.random.normal(0, 0.05)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics = ModelMetrics(
            accuracy=max(0, min(1, accuracy)),
            precision=max(0, min(1, precision)),
            recall=max(0, min(1, recall)),
            f1_score=max(0, min(1, f1)),
            avg_confidence=avg_confidence,
            prediction_count=prediction_count,
            timestamp=datetime.now()
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics
                (timestamp, accuracy, precision, recall, f1_score, avg_confidence, prediction_count, period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.avg_confidence,
                metrics.prediction_count,
                f"{period_hours}h"
            ))
            conn.commit()
        
        return metrics

    def detect_data_drift(self, 
                         current_data: NDArray[Any],
                         baseline_data: Optional[NDArray[Any]] = None,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests
        """
        if baseline_data is None:
            baseline_data = self.load_baseline_data()
        
        if baseline_data is None:
            self.logger.warning("No baseline data available for drift detection")
            return {}
        
        drift_results = {}
        
        if len(current_data.shape) > 1:
            n_features = current_data.shape[1]
            feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
            
            for i, feature_name in enumerate(feature_names):
                current_feature = current_data[:, i].flatten()
                baseline_feature = baseline_data[:, i].flatten()
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(baseline_feature, current_feature)
                
                is_drift = p_value < self.alert_thresholds['drift_p_value']
                
                # Corrected drift severity logic
                if p_value < 0.01:
                    drift_severity = 'high'
                elif p_value < 0.05:
                    drift_severity = 'medium'
                else:
                    drift_severity = 'low'
                
                drift_results[feature_name] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'is_drift': is_drift,
                    'drift_severity': drift_severity
                }
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO drift_detection
                        (timestamp, feature_name, drift_score, p_value, is_drift, baseline_period, current_period)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        feature_name,
                        ks_stat,
                        p_value,
                        is_drift,
                        "baseline",
                        "current"
                    ))
                    conn.commit()
                
                if is_drift:
                    self.trigger_alert(
                        "data_drift",
                        drift_severity,
                        f"Data drift detected in {feature_name} (p-value: {p_value:.4f})"
                    )
        
        return drift_results

    def trigger_alert(self, alert_type: str, severity: str, message: str):
        """Trigger an alert and log it"""
        
        self.logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), alert_type, severity, message))
            conn.commit()
        
        # In production, you could integrate with:
        # - Slack/Teams notifications
        # - Email alerts
        # - PagerDuty
        # - Prometheus/Grafana alerts

    def generate_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get recent metrics
        metrics = self.calculate_performance_metrics(hours)
        
        with sqlite3.connect(self.db_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=[start_time.isoformat()])
            predictions_df = pd.read_sql_query('''
                SELECT * FROM predictions
                WHERE timestamp >= ?
            ''', conn, params=[start_time.isoformat()])
        
        report = {
            'period': f"Last {hours} hours",
            'timestamp': end_time.isoformat(),
            'model_metrics': {
                'accuracy': metrics.accuracy if metrics else None,
                'precision': metrics.precision if metrics else None,
                'recall': metrics.recall if metrics else None,
                'f1_score': metrics.f1_score if metrics else None,
                'avg_confidence': metrics.avg_confidence if metrics else None,
                'prediction_count': metrics.prediction_count if metrics else 0
            },
            'alerts': {
                'total_alerts': len(alerts_df),
                'critical_alerts': len(alerts_df[alerts_df['severity'] == 'critical']) if not alerts_df.empty else 0,
                'warnings': len(alerts_df[alerts_df['severity'] == 'medium']) if not alerts_df.empty else 0,
                'recent_alerts': alerts_df.head(5).to_dict('records') if not alerts_df.empty else []
            },
            'predictions': {
                'total_predictions': len(predictions_df),
                'avg_processing_time': predictions_df['processing_time'].mean() if not predictions_df.empty else None,
                'prediction_distribution': predictions_df['prediction'].value_counts().to_dict() if not predictions_df.empty else {}
            }
        }
        
        return report

    def load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics for comparison"""
        # In production, this would load from a baseline dataset
        return {
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.91,
            'f1_score': 0.89,
            'avg_confidence': 0.85
        }

    def load_baseline_data(self) -> Optional[NDArray[Any]]:
        """Load baseline data for drift detection"""
        # In production, this would load actual baseline data
        # For demo, return None to skip drift detection
        return None

    def start_monitoring(self, check_interval: int = 300):
        """Start continuous monitoring (every 5 minutes by default)"""
        
        def monitoring_loop():
            while True:
                try:
                    # Calculate current metrics
                    metrics = self.calculate_performance_metrics(1)  # Last hour
                    
                    if metrics and self.baseline_metrics:
                        # Check for performance degradation
                        if metrics.accuracy < (self.baseline_metrics['accuracy'] - self.alert_thresholds['accuracy_drop']):
                            self.trigger_alert(
                                "performance_degradation",
                                "high",
                                f"Accuracy dropped to {metrics.accuracy:.3f} (baseline: {self.baseline_metrics['accuracy']:.3f})"
                            )
                        
                        if metrics.avg_confidence < (self.baseline_metrics['avg_confidence'] - self.alert_thresholds['confidence_drop']):
                            self.trigger_alert(
                                "confidence_drop",
                                "medium",
                                f"Average confidence dropped to {metrics.avg_confidence:.3f}"
                            )
                    
                    # Generate and log report
                    report = self.generate_monitoring_report(1)
                    self.logger.info(f"Monitoring report: {report['model_metrics']}")
                    
                    # Wait for next check
                    time.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info(f"Continuous monitoring started (interval: {check_interval}s)")
        return monitoring_thread

def main():
    """Main function to start model monitoring"""
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Start continuous monitoring
    monitor.start_monitoring(check_interval=300)  # Check every 5 minutes
    
    # Generate initial report
    report = monitor.generate_monitoring_report(24)
    print("\n" + "="*80)
    print("FACE MASK DETECTION - MODEL MONITORING REPORT")
    print("="*80)
    print(json.dumps(report, indent=2, default=str))
    print("="*80)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main()