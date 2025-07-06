"""
Monitoring Pipeline for Face Mask Detection
"""

import os
import json
import sqlite3
import logging
import datetime
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.stats import entropy
except ImportError:
    entropy = None

from .config import MonitoringConfig, ProjectConfig

class MonitoringPipeline:
    """Complete monitoring pipeline for MLOps model tracking"""
    
    def __init__(self, config: MonitoringConfig, project_config: ProjectConfig):
        self.config = config
        self.project_config = project_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = config.metrics_database
        self.monitoring_active = False
        self.baseline_metrics = {}
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    model_version TEXT,
                    data_source TEXT
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create data_drift table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    feature_drifts TEXT  -- JSON string
                )
            ''')
            
            conn.commit()
        
        self.logger.info("✅ Monitoring database initialized")
    
    def log_metric(self, metric_name: str, metric_value: float, model_version: str = None, data_source: str = None):
        """Log a metric to the monitoring database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metrics (timestamp, metric_name, metric_value, model_version, data_source)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                metric_name,
                metric_value,
                model_version,
                data_source
            ))
            conn.commit()
    
    def log_alert(self, alert_type: str, severity: str, message: str):
        """Log an alert to the monitoring database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                alert_type,
                severity,
                message
            ))
            conn.commit()
        
        self.logger.warning(f"🚨 ALERT [{severity}] {alert_type}: {message}")
    
    def calculate_data_drift(self, current_data: np.ndarray, reference_data: np.ndarray):
        """Calculate data drift between current and reference data"""
        try:
            if entropy is None:
                self.logger.warning("scipy not available for drift calculation")
                return None
                
            # Simple statistical drift detection using KL divergence
            # Normalize data to create probability distributions
            current_hist, _ = np.histogram(current_data.flatten(), bins=50, density=True)
            reference_hist, _ = np.histogram(reference_data.flatten(), bins=50, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            current_hist += epsilon
            reference_hist += epsilon
            
            # Calculate KL divergence
            drift_score = entropy(current_hist, reference_hist)
            
            # Determine if drift is detected
            drift_detected = drift_score > self.config.data_drift_threshold
            
            # Log to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO data_drift (timestamp, drift_score, drift_detected, feature_drifts)
                    VALUES (?, ?, ?, ?)
                ''', (
                    datetime.datetime.now().isoformat(),
                    float(drift_score),
                    drift_detected,
                    json.dumps({"kl_divergence": float(drift_score)})
                ))
                conn.commit()
            
            if drift_detected:
                self.log_alert(
                    "DATA_DRIFT",
                    "HIGH",
                    f"Data drift detected with score {drift_score:.4f} (threshold: {self.config.data_drift_threshold})"
                )
            
            return {
                "drift_score": drift_score,
                "drift_detected": drift_detected,
                "threshold": self.config.data_drift_threshold
            }
            
        except Exception as e:
            self.logger.error(f"❌ Data drift calculation failed: {e}")
            return None
    
    def monitor_model_performance(self, inference_pipeline):
        """Monitor model performance and detect degradation"""
        try:
            performance_summary = inference_pipeline.get_performance_summary()
            
            if "error" in performance_summary:
                return
            
            # Log key metrics
            metrics_to_log = [
                ("average_confidence", performance_summary.get("average_confidence", 0)),
                ("average_inference_time", performance_summary.get("average_inference_time", 0)),
                ("predictions_per_second", performance_summary.get("predictions_per_second", 0)),
                ("total_predictions", performance_summary.get("total_predictions", 0))
            ]
            
            for metric_name, metric_value in metrics_to_log:
                self.log_metric(metric_name, metric_value)
            
            # Check for performance degradation
            if self.baseline_metrics:
                current_confidence = performance_summary.get("average_confidence", 0)
                baseline_confidence = self.baseline_metrics.get("average_confidence", 0)
                
                if baseline_confidence > 0:
                    confidence_drop = (baseline_confidence - current_confidence) / baseline_confidence
                    
                    if confidence_drop > self.config.alert_threshold:
                        self.log_alert(
                            "PERFORMANCE_DEGRADATION",
                            "MEDIUM",
                            f"Model confidence dropped by {confidence_drop:.2%} from baseline"
                        )
            
            else:
                # Set baseline if not exists
                self.baseline_metrics = performance_summary.copy()
                self.logger.info("📊 Baseline metrics established")
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
    
    def get_monitoring_dashboard_data(self, hours: int = 24):
        """Get data for monitoring dashboard"""
        cutoff_time = (datetime.datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent metrics
            try:
                metrics_df = pd.read_sql_query('''
                    SELECT timestamp, metric_name, metric_value 
                    FROM model_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', conn, params=(cutoff_time,))
            except Exception as e:
                self.logger.warning(f"Could not read metrics: {e}")
                metrics_df = pd.DataFrame()
            
            # Get recent alerts
            try:
                alerts_df = pd.read_sql_query('''
                    SELECT timestamp, alert_type, severity, message, resolved
                    FROM alerts 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', conn, params=(cutoff_time,))
            except Exception as e:
                self.logger.warning(f"Could not read alerts: {e}")
                alerts_df = pd.DataFrame()
            
            # Get drift data
            try:
                drift_df = pd.read_sql_query('''
                    SELECT timestamp, drift_score, drift_detected
                    FROM data_drift 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', conn, params=(cutoff_time,))
            except Exception as e:
                self.logger.warning(f"Could not read drift data: {e}")
                drift_df = pd.DataFrame()
        
        return {
            "metrics": metrics_df.to_dict('records') if not metrics_df.empty else [],
            "alerts": alerts_df.to_dict('records') if not alerts_df.empty else [],
            "drift_data": drift_df.to_dict('records') if not drift_df.empty else []
        }
    
    def create_monitoring_visualizations(self, hours: int = 24):
        """Create monitoring visualizations"""
        dashboard_data = self.get_monitoring_dashboard_data(hours)
        
        if not dashboard_data["metrics"]:
            self.logger.info("No metrics data available for visualization")
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Model Monitoring Dashboard - Last {hours} Hours', fontsize=16)
            
            metrics_df = pd.DataFrame(dashboard_data["metrics"])
            
            if not metrics_df.empty:
                # Convert timestamp to datetime
                metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                
                # Plot confidence over time
                conf_data = metrics_df[metrics_df['metric_name'] == 'average_confidence']
                if not conf_data.empty:
                    axes[0, 0].plot(conf_data['timestamp'], conf_data['metric_value'], 'b-o')
                    axes[0, 0].set_title('Average Confidence Over Time')
                    axes[0, 0].set_ylabel('Confidence')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Plot inference time over time
                time_data = metrics_df[metrics_df['metric_name'] == 'average_inference_time']
                if not time_data.empty:
                    axes[0, 1].plot(time_data['timestamp'], time_data['metric_value'], 'r-o')
                    axes[0, 1].set_title('Average Inference Time')
                    axes[0, 1].set_ylabel('Time (seconds)')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Plot predictions per second
                pps_data = metrics_df[metrics_df['metric_name'] == 'predictions_per_second']
                if not pps_data.empty:
                    axes[1, 0].plot(pps_data['timestamp'], pps_data['metric_value'], 'g-o')
                    axes[1, 0].set_title('Predictions Per Second')
                    axes[1, 0].set_ylabel('Predictions/sec')
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot alerts by severity
            if dashboard_data["alerts"]:
                alerts_df = pd.DataFrame(dashboard_data["alerts"])
                alert_counts = alerts_df['severity'].value_counts()
                axes[1, 1].pie(alert_counts.values, labels=alert_counts.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Alerts by Severity')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.project_config.reports_dir) / "monitoring" / f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"📊 Monitoring dashboard saved: {plot_path}")
            plt.close()  # Close to free memory
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")
            return None
    
    def start_continuous_monitoring(self, inference_pipeline, interval: int = None):
        """Start continuous monitoring in background"""
        interval = interval or self.config.monitoring_interval
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.monitor_model_performance(inference_pipeline)
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"❌ Monitoring loop error: {e}")
                    time.sleep(interval)
        
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info(f"🔄 Continuous monitoring started (interval: {interval}s)")
        return monitoring_thread
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("⏹️ Continuous monitoring stopped")
