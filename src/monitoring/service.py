"""
Face Mask Detection Monitoring Service

This module provides a comprehensive monitoring service that orchestrates
metrics collection, drift detection, and alerting for the face mask detection system.
"""

import os
import time
import json
import schedule
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
from threading import Thread, Event
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import requests

from ..common.logger import get_logger
from ..common.utils import load_config
from .metrics_collector import MetricsCollector, ModelMetrics, DataMetrics, SystemMetrics, BusinessMetrics
from drift.drift_detector import DataDriftDetector

logger = get_logger(__name__)


@dataclass
class Alert:
    """Alert notification container"""
    timestamp: float
    alert_type: str  # 'system', 'performance', 'drift', 'business'
    severity: str    # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_collection_interval: int = 60  # seconds
    drift_detection_interval: int = 3600   # seconds (1 hour)
    alert_check_interval: int = 300        # seconds (5 minutes)
    system_health_check_interval: int = 30 # seconds
    
    # Thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    response_time_threshold: float = 2.0   # seconds
    confidence_threshold: float = 0.5
    drift_threshold: float = 0.05
    
    # Notification settings
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    enable_slack_alerts: bool = False
    
    # Data paths
    metrics_retention_days: int = 30
    drift_reference_update_days: int = 7


class AlertManager:
    """Manages alert generation and notification"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize alert manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alert_config = config.get('monitoring', {}).get('alerts', {})
        self.notification_handlers = []
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_alert_history = 1000
        
        logger.info("AlertManager initialized")
    
    def _setup_notification_handlers(self):
        """Setup notification handlers based on configuration"""
        if self.alert_config.get('enable_email_alerts', False):
            self.notification_handlers.append(self._send_email_alert)
        
        if self.alert_config.get('enable_webhook_alerts', False):
            self.notification_handlers.append(self._send_webhook_alert)
        
        if self.alert_config.get('enable_slack_alerts', False):
            self.notification_handlers.append(self._send_slack_alert)
        
        # Always add console handler
        self.notification_handlers.append(self._send_console_alert)
    
    def send_alert(self, alert: Alert) -> None:
        """Send alert through all configured channels
        
        Args:
            alert: Alert to send
        """
        try:
            # Add to history
            self.alert_history.append(alert)
            
            # Trim history if needed
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
            
            # Send through all handlers
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert through handler {handler.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, alert: Alert) -> None:
        """Send alert via email"""
        try:
            smtp_config = self.alert_config.get('email', {})
            
            if not all(k in smtp_config for k in ['smtp_server', 'smtp_port', 'username', 'password', 'recipients']):
                logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = smtp_config['username']
            msg['To'] = ', '.join(smtp_config['recipients'])
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            
            Type: {alert.alert_type}
            Severity: {alert.severity}
            Source: {alert.source}
            Time: {datetime.fromtimestamp(alert.timestamp)}
            
            Message:
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """Send alert via webhook"""
        try:
            webhook_config = self.alert_config.get('webhook', {})
            
            if 'url' not in webhook_config:
                logger.warning("Webhook URL not configured, skipping webhook alert")
                return
            
            payload = {
                'timestamp': alert.timestamp,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            response.raise_for_status()
            logger.info(f"Webhook alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack_alert(self, alert: Alert) -> None:
        """Send alert via Slack"""
        try:
            slack_config = self.alert_config.get('slack', {})
            
            if 'webhook_url' not in slack_config:
                logger.warning("Slack webhook URL not configured, skipping Slack alert")
                return
            
            # Color coding based on severity
            color_map = {
                'low': '#36a64f',      # Green
                'medium': '#ff9500',   # Orange
                'high': '#ff0000',     # Red
                'critical': '#8b0000'  # Dark Red
            }
            
            payload = {
                'attachments': [
                    {
                        'color': color_map.get(alert.severity, '#36a64f'),
                        'title': alert.title,
                        'text': alert.message,
                        'fields': [
                            {
                                'title': 'Type',
                                'value': alert.alert_type,
                                'short': True
                            },
                            {
                                'title': 'Severity',
                                'value': alert.severity,
                                'short': True
                            },
                            {
                                'title': 'Source',
                                'value': alert.source,
                                'short': True
                            },
                            {
                                'title': 'Time',
                                'value': datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                'short': True
                            }
                        ],
                        'footer': 'Face Mask Detection Monitoring',
                        'ts': int(alert.timestamp)
                    }
                ]
            }
            
            response = requests.post(slack_config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_console_alert(self, alert: Alert) -> None:
        """Send alert to console/logs"""
        severity_symbols = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🔴',
            'critical': '🚨'
        }
        
        symbol = severity_symbols.get(alert.severity, '⚪')
        timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        log_message = f"{symbol} [{alert.severity.upper()}] {alert.title} | {alert.message} | Source: {alert.source} | Time: {timestamp_str}"
        
        if alert.severity in ['high', 'critical']:
            logger.error(log_message)
        elif alert.severity == 'medium':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts within specified time range
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class MonitoringService:
    """Main monitoring service orchestrator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize monitoring service
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.monitoring_config = MonitoringConfig(**self.config.get('monitoring', {}))
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config_path)
        self.drift_detector = DataDriftDetector(config_path)
        self.alert_manager = AlertManager(self.config)
        
        # Service control
        self._stop_event = Event()
        self._monitoring_thread = None
        
        # Tracking variables
        self.last_drift_check = 0
        self.last_reference_update = 0
        
        logger.info("MonitoringService initialized")
    
    def start(self) -> None:
        """Start monitoring service"""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Monitoring service already running")
            return
        
        self._stop_event.clear()
        
        # Start metrics collection
        self.metrics_collector.start_background_collection()
        
        # Setup scheduled tasks
        self._setup_scheduler()
        
        # Start monitoring thread
        self._monitoring_thread = Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        logger.info("Monitoring service started")
    
    def stop(self) -> None:
        """Stop monitoring service"""
        if self._monitoring_thread is None:
            return
        
        self._stop_event.set()
        
        # Stop metrics collection
        self.metrics_collector.stop_background_collection()
        
        # Wait for monitoring thread to finish
        self._monitoring_thread.join(timeout=10)
        
        logger.info("Monitoring service stopped")
    
    def _setup_scheduler(self) -> None:
        """Setup scheduled monitoring tasks"""
        # Schedule drift detection
        schedule.every(self.monitoring_config.drift_detection_interval).seconds.do(
            self._check_data_drift
        )
        
        # Schedule system health checks
        schedule.every(self.monitoring_config.system_health_check_interval).seconds.do(
            self._check_system_health
        )
        
        # Schedule alert checks
        schedule.every(self.monitoring_config.alert_check_interval).seconds.do(
            self._check_alerts
        )
        
        # Schedule reference data updates (weekly)
        schedule.every(self.monitoring_config.drift_reference_update_days).days.do(
            self._update_drift_reference
        )
        
        # Schedule cleanup (daily)
        schedule.every().day.at("02:00").do(self._cleanup_old_data)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short interval
                self._stop_event.wait(5)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(5)
    
    def _check_data_drift(self) -> None:
        """Check for data drift"""
        try:
            logger.info("Checking for data drift...")
            
            # Get recent images for drift detection
            # This would typically come from your data pipeline
            recent_images = self._get_recent_images()
            
            if len(recent_images) < 10:  # Need minimum samples
                logger.warning("Insufficient recent images for drift detection")
                return
            
            # Perform drift detection
            drift_results = self.drift_detector.detect_drift(recent_images)
            
            # Check for significant drift
            high_drift_features = [r for r in drift_results if r.drift_detected and r.severity == 'high']
            medium_drift_features = [r for r in drift_results if r.drift_detected and r.severity == 'medium']
            
            if high_drift_features:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='drift',
                    severity='high',
                    title='High Data Drift Detected',
                    message=f'Significant drift detected in {len(high_drift_features)} features: {[f.feature_name for f in high_drift_features]}',
                    source='drift_detector',
                    metadata={'drift_results': [asdict(r) for r in high_drift_features]}
                )
                self.alert_manager.send_alert(alert)
            
            elif medium_drift_features:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='drift',
                    severity='medium',
                    title='Moderate Data Drift Detected',
                    message=f'Moderate drift detected in {len(medium_drift_features)} features: {[f.feature_name for f in medium_drift_features]}',
                    source='drift_detector',
                    metadata={'drift_results': [asdict(r) for r in medium_drift_features]}
                )
                self.alert_manager.send_alert(alert)
            
            self.last_drift_check = time.time()
            
        except Exception as e:
            logger.error(f"Failed to check data drift: {e}")
    
    def _check_system_health(self) -> None:
        """Check system health and generate alerts if needed"""
        try:
            # Collect current system metrics
            self.metrics_collector.collect_system_metrics()
            
            # Get latest metrics
            recent_metrics = self.metrics_collector.get_metrics('system', limit=1)
            
            if not recent_metrics:
                return
            
            latest = recent_metrics[0]
            
            # Check CPU usage
            if latest.get('cpu_usage', 0) > self.monitoring_config.cpu_threshold:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='system',
                    severity='high',
                    title='High CPU Usage',
                    message=f'CPU usage is {latest["cpu_usage"]:.1f}% (threshold: {self.monitoring_config.cpu_threshold}%)',
                    source='system_monitor',
                    metadata={'cpu_usage': latest['cpu_usage']}
                )
                self.alert_manager.send_alert(alert)
            
            # Check memory usage
            if latest.get('memory_usage', 0) > self.monitoring_config.memory_threshold:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='system',
                    severity='high',
                    title='High Memory Usage',
                    message=f'Memory usage is {latest["memory_usage"]:.1f}% (threshold: {self.monitoring_config.memory_threshold}%)',
                    source='system_monitor',
                    metadata={'memory_usage': latest['memory_usage']}
                )
                self.alert_manager.send_alert(alert)
            
            # Check response time
            if latest.get('response_time', 0) > self.monitoring_config.response_time_threshold:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='performance',
                    severity='medium',
                    title='High Response Time',
                    message=f'Response time is {latest["response_time"]:.3f}s (threshold: {self.monitoring_config.response_time_threshold}s)',
                    source='system_monitor',
                    metadata={'response_time': latest['response_time']}
                )
                self.alert_manager.send_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    def _check_alerts(self) -> None:
        """Check for alert conditions in model performance"""
        try:
            # Get recent model metrics
            recent_metrics = self.metrics_collector.get_metrics('model', limit=10)
            
            if len(recent_metrics) < 5:  # Need minimum samples
                return
            
            # Calculate averages
            avg_confidence = np.mean([m.get('avg_confidence', 0) for m in recent_metrics])
            
            # Check average confidence
            if avg_confidence < self.monitoring_config.confidence_threshold:
                alert = Alert(
                    timestamp=time.time(),
                    alert_type='performance',
                    severity='medium',
                    title='Low Model Confidence',
                    message=f'Average model confidence is {avg_confidence:.3f} (threshold: {self.monitoring_config.confidence_threshold})',
                    source='performance_monitor',
                    metadata={'avg_confidence': avg_confidence}
                )
                self.alert_manager.send_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
    
    def _get_recent_images(self) -> List[str]:
        """Get list of recent images for drift detection
        
        Returns:
            List of image paths
        """
        try:
            # This would typically integrate with your data pipeline
            # For now, we'll look in common directories
            
            image_dirs = [
                Path("data/processed"),
                Path("temp_uploads"),
                Path("detections")
            ]
            
            recent_images = []
            cutoff_time = time.time() - 3600  # Last hour
            
            for img_dir in image_dirs:
                if img_dir.exists():
                    for img_path in img_dir.glob("*.jpg"):
                        if img_path.stat().st_mtime > cutoff_time:
                            recent_images.append(str(img_path))
            
            return recent_images[:100]  # Limit to 100 recent images
            
        except Exception as e:
            logger.error(f"Failed to get recent images: {e}")
            return []
    
    def _update_drift_reference(self) -> None:
        """Update drift detection reference data"""
        try:
            logger.info("Updating drift detection reference data...")
            
            # Get images from the last week
            cutoff_time = time.time() - (7 * 24 * 3600)
            reference_images = []
            
            image_dirs = [Path("data/processed")]
            
            for img_dir in image_dirs:
                if img_dir.exists():
                    for img_path in img_dir.glob("*.jpg"):
                        if img_path.stat().st_mtime > cutoff_time:
                            reference_images.append(str(img_path))
            
            if len(reference_images) >= 50:  # Need minimum samples
                self.drift_detector.set_reference_data(reference_images)
                self.last_reference_update = time.time()
                logger.info(f"Updated drift reference with {len(reference_images)} images")
            else:
                logger.warning("Insufficient images for reference update")
                
        except Exception as e:
            logger.error(f"Failed to update drift reference: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old monitoring data"""
        try:
            logger.info("Cleaning up old monitoring data...")
            
            # This would typically involve database cleanup
            # For now, just log the action
            
            cutoff_time = time.time() - (self.monitoring_config.metrics_retention_days * 24 * 3600)
            
            # Cleanup alert history
            self.alert_manager.alert_history = [
                alert for alert in self.alert_manager.alert_history
                if alert.timestamp > cutoff_time
            ]
            
            logger.info("Old data cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status
        
        Returns:
            Dictionary containing monitoring status
        """
        try:
            recent_alerts = self.alert_manager.get_recent_alerts(24)
            
            status = {
                'service_running': self._monitoring_thread is not None and self._monitoring_thread.is_alive(),
                'last_drift_check': datetime.fromtimestamp(self.last_drift_check).isoformat() if self.last_drift_check else None,
                'last_reference_update': datetime.fromtimestamp(self.last_reference_update).isoformat() if self.last_reference_update else None,
                'recent_alerts_count': len(recent_alerts),
                'alert_summary': {
                    'critical': len([a for a in recent_alerts if a.severity == 'critical']),
                    'high': len([a for a in recent_alerts if a.severity == 'high']),
                    'medium': len([a for a in recent_alerts if a.severity == 'medium']),
                    'low': len([a for a in recent_alerts if a.severity == 'low'])
                },
                'configuration': {
                    'metrics_collection_interval': self.monitoring_config.metrics_collection_interval,
                    'drift_detection_interval': self.monitoring_config.drift_detection_interval,
                    'cpu_threshold': self.monitoring_config.cpu_threshold,
                    'memory_threshold': self.monitoring_config.memory_threshold,
                    'response_time_threshold': self.monitoring_config.response_time_threshold
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return {'error': str(e)}


def main():
    """CLI entry point for monitoring service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mask Detection Monitoring Service')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start', action='store_true',
                       help='Start monitoring service')
    parser.add_argument('--status', action='store_true',
                       help='Show monitoring status')
    
    args = parser.parse_args()
    
    service = MonitoringService(args.config)
    
    if args.start:
        try:
            service.start()
            print("Monitoring service started. Press Ctrl+C to stop.")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring service...")
            service.stop()
            print("Monitoring service stopped.")
    
    elif args.status:
        status = service.get_monitoring_status()
        print(json.dumps(status, indent=2))
    
    else:
        print("Please specify --start or --status")


if __name__ == "__main__":
    main()
