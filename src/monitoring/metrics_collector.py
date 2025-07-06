"""
Face Mask Detection Metrics Collector

This module collects and stores various metrics for monitoring model performance,
data drift, system health, and business KPIs.
"""

import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from threading import Thread, Event
import psutil
import cv2

from ..common.logger import get_logger
from ..common.utils import load_config

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: float
    model_version: str
    total_predictions: int
    with_mask_count: int
    without_mask_count: int
    avg_confidence: float
    avg_inference_time: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class DataMetrics:
    """Data quality and drift metrics"""
    timestamp: float
    image_count: int
    avg_image_size: Tuple[int, int]
    avg_brightness: float
    avg_contrast: float
    blur_score: float
    noise_level: float
    color_distribution: Dict[str, float]


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    inference_queue_size: int
    response_time: float


@dataclass
class BusinessMetrics:
    """Business and operational metrics"""
    timestamp: float
    total_api_calls: int
    successful_predictions: int
    failed_predictions: int
    mask_compliance_rate: float
    peak_usage_hours: List[int]
    geographic_distribution: Dict[str, int]


class MetricsCollector:
    """Central metrics collection and storage system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize metrics collector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.db_path = self.config.get('monitoring', {}).get('metrics_db', 'logs/metrics.db')
        self.collection_interval = self.config.get('monitoring', {}).get('collection_interval', 60)
        
        # Create database directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Collection control
        self._stop_event = Event()
        self._collection_thread = None
        
        # Metrics buffers
        self._model_metrics_buffer = []
        self._data_metrics_buffer = []
        self._system_metrics_buffer = []
        self._business_metrics_buffer = []
        
        logger.info(f"MetricsCollector initialized with DB: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Model metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        total_predictions INTEGER NOT NULL,
                        with_mask_count INTEGER NOT NULL,
                        without_mask_count INTEGER NOT NULL,
                        avg_confidence REAL NOT NULL,
                        avg_inference_time REAL NOT NULL,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL
                    )
                ''')
                
                # Data metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        image_count INTEGER NOT NULL,
                        avg_width INTEGER NOT NULL,
                        avg_height INTEGER NOT NULL,
                        avg_brightness REAL NOT NULL,
                        avg_contrast REAL NOT NULL,
                        blur_score REAL NOT NULL,
                        noise_level REAL NOT NULL,
                        color_distribution TEXT NOT NULL
                    )
                ''')
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        cpu_usage REAL NOT NULL,
                        memory_usage REAL NOT NULL,
                        disk_usage REAL NOT NULL,
                        gpu_usage REAL,
                        gpu_memory REAL,
                        inference_queue_size INTEGER NOT NULL,
                        response_time REAL NOT NULL
                    )
                ''')
                
                # Business metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS business_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        total_api_calls INTEGER NOT NULL,
                        successful_predictions INTEGER NOT NULL,
                        failed_predictions INTEGER NOT NULL,
                        mask_compliance_rate REAL NOT NULL,
                        peak_usage_hours TEXT NOT NULL,
                        geographic_distribution TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def collect_model_metrics(self, 
                            model_version: str,
                            predictions: List[Dict[str, Any]],
                            inference_times: List[float],
                            ground_truth: Optional[List[Dict[str, Any]]] = None) -> None:
        """Collect model performance metrics
        
        Args:
            model_version: Version of the model used
            predictions: List of prediction results
            inference_times: List of inference times
            ground_truth: Optional ground truth for accuracy calculation
        """
        try:
            timestamp = time.time()
            
            # Count predictions
            with_mask = sum(1 for pred in predictions 
                          for detection in pred.get('predictions', []) 
                          if detection.get('class') == 'with_mask')
            without_mask = sum(1 for pred in predictions 
                             for detection in pred.get('predictions', []) 
                             if detection.get('class') == 'without_mask')
            
            # Calculate confidence statistics
            all_confidences = []
            for pred in predictions:
                for detection in pred.get('predictions', []):
                    all_confidences.append(detection.get('confidence', 0.0))
            
            avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
            avg_inference_time = np.mean(inference_times) if inference_times else 0.0
            
            # Calculate accuracy metrics if ground truth is available
            accuracy = precision = recall = f1_score = None
            if ground_truth:
                accuracy, precision, recall, f1_score = self._calculate_performance_metrics(
                    predictions, ground_truth
                )
            
            metrics = ModelMetrics(
                timestamp=timestamp,
                model_version=model_version,
                total_predictions=len(predictions),
                with_mask_count=with_mask,
                without_mask_count=without_mask,
                avg_confidence=avg_confidence,
                avg_inference_time=avg_inference_time,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            self._model_metrics_buffer.append(metrics)
            logger.debug(f"Collected model metrics: {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")
    
    def collect_data_metrics(self, image_paths: List[str]) -> None:
        """Collect data quality metrics
        
        Args:
            image_paths: List of paths to analyzed images
        """
        try:
            timestamp = time.time()
            
            if not image_paths:
                return
            
            # Analyze image properties
            sizes = []
            brightness_values = []
            contrast_values = []
            blur_scores = []
            noise_levels = []
            color_distributions = {'red': [], 'green': [], 'blue': []}
            
            for img_path in image_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Image size
                    h, w, _ = img.shape
                    sizes.append((w, h))
                    
                    # Convert to different color spaces
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    
                    # Brightness (mean of V channel in HSV)
                    brightness = np.mean(hsv[:, :, 2])
                    brightness_values.append(brightness)
                    
                    # Contrast (standard deviation of grayscale)
                    contrast = np.std(gray)
                    contrast_values.append(contrast)
                    
                    # Blur detection (variance of Laplacian)
                    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_scores.append(blur)
                    
                    # Noise level (standard deviation of difference from median filtered)
                    median_filtered = cv2.medianBlur(gray, 5)
                    noise = np.std(gray.astype(float) - median_filtered.astype(float))
                    noise_levels.append(noise)
                    
                    # Color distribution
                    b, g, r = cv2.split(img)
                    color_distributions['red'].append(np.mean(r))
                    color_distributions['green'].append(np.mean(g))
                    color_distributions['blue'].append(np.mean(b))
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze image {img_path}: {e}")
                    continue
            
            if not sizes:
                return
            
            # Calculate aggregated metrics
            avg_size = (
                int(np.mean([s[0] for s in sizes])),
                int(np.mean([s[1] for s in sizes]))
            )
            
            color_dist = {
                'red': float(np.mean(color_distributions['red'])),
                'green': float(np.mean(color_distributions['green'])),
                'blue': float(np.mean(color_distributions['blue']))
            }
            
            metrics = DataMetrics(
                timestamp=timestamp,
                image_count=len(image_paths),
                avg_image_size=avg_size,
                avg_brightness=float(np.mean(brightness_values)),
                avg_contrast=float(np.mean(contrast_values)),
                blur_score=float(np.mean(blur_scores)),
                noise_level=float(np.mean(noise_levels)),
                color_distribution=color_dist
            )
            
            self._data_metrics_buffer.append(metrics)
            logger.debug(f"Collected data metrics for {len(image_paths)} images")
            
        except Exception as e:
            logger.error(f"Failed to collect data metrics: {e}")
    
    def collect_system_metrics(self, 
                             inference_queue_size: int = 0,
                             response_time: float = 0.0) -> None:
        """Collect system resource metrics
        
        Args:
            inference_queue_size: Current size of inference queue
            response_time: Average response time for recent requests
        """
        try:
            timestamp = time.time()
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # GPU metrics (if available)
            gpu_usage = gpu_memory = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = float(gpu_info.gpu)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = (mem_info.used / mem_info.total) * 100
                
            except ImportError:
                logger.debug("pynvml not available, skipping GPU metrics")
            except Exception as e:
                logger.debug(f"Failed to collect GPU metrics: {e}")
            
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                inference_queue_size=inference_queue_size,
                response_time=response_time
            )
            
            self._system_metrics_buffer.append(metrics)
            logger.debug("Collected system metrics")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def collect_business_metrics(self,
                               api_calls: int,
                               successful_predictions: int,
                               failed_predictions: int,
                               predictions: List[Dict[str, Any]],
                               geographic_data: Optional[Dict[str, int]] = None) -> None:
        """Collect business and operational metrics
        
        Args:
            api_calls: Total number of API calls
            successful_predictions: Number of successful predictions
            failed_predictions: Number of failed predictions
            predictions: Recent prediction results
            geographic_data: Geographic distribution of requests
        """
        try:
            timestamp = time.time()
            
            # Calculate mask compliance rate
            total_detections = 0
            with_mask_detections = 0
            
            for pred in predictions:
                for detection in pred.get('predictions', []):
                    total_detections += 1
                    if detection.get('class') == 'with_mask':
                        with_mask_detections += 1
            
            compliance_rate = (with_mask_detections / total_detections * 100 
                             if total_detections > 0 else 0.0)
            
            # Determine peak usage hours (simplified)
            current_hour = datetime.now().hour
            peak_hours = [current_hour] if api_calls > 10 else []
            
            # Geographic distribution
            geo_dist = geographic_data or {'unknown': api_calls}
            
            metrics = BusinessMetrics(
                timestamp=timestamp,
                total_api_calls=api_calls,
                successful_predictions=successful_predictions,
                failed_predictions=failed_predictions,
                mask_compliance_rate=compliance_rate,
                peak_usage_hours=peak_hours,
                geographic_distribution=geo_dist
            )
            
            self._business_metrics_buffer.append(metrics)
            logger.debug("Collected business metrics")
            
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
    
    def _calculate_performance_metrics(self, 
                                     predictions: List[Dict[str, Any]], 
                                     ground_truth: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """Calculate accuracy, precision, recall, and F1 score"""
        try:
            # Simplified implementation - in practice, you'd need proper matching
            # between predictions and ground truth based on IoU or other criteria
            
            true_positives = false_positives = false_negatives = 0
            
            # This is a simplified example - replace with proper evaluation logic
            for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
                pred_classes = [d.get('class') for d in pred.get('predictions', [])]
                gt_classes = [d.get('class') for d in gt.get('annotations', [])]
                
                # Simple counting (not ideal - use proper IoU matching in practice)
                for gt_class in gt_classes:
                    if gt_class in pred_classes:
                        true_positives += 1
                        pred_classes.remove(gt_class)
                    else:
                        false_negatives += 1
                
                false_positives += len(pred_classes)
            
            # Calculate metrics
            accuracy = true_positives / (true_positives + false_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return accuracy, precision, recall, f1_score
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def flush_metrics(self) -> None:
        """Flush all buffered metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Flush model metrics
                for metrics in self._model_metrics_buffer:
                    cursor.execute('''
                        INSERT INTO model_metrics 
                        (timestamp, model_version, total_predictions, with_mask_count, 
                         without_mask_count, avg_confidence, avg_inference_time, 
                         accuracy, precision, recall, f1_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.model_version, metrics.total_predictions,
                        metrics.with_mask_count, metrics.without_mask_count,
                        metrics.avg_confidence, metrics.avg_inference_time,
                        metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score
                    ))
                
                # Flush data metrics
                for metrics in self._data_metrics_buffer:
                    cursor.execute('''
                        INSERT INTO data_metrics 
                        (timestamp, image_count, avg_width, avg_height, avg_brightness,
                         avg_contrast, blur_score, noise_level, color_distribution)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.image_count,
                        metrics.avg_image_size[0], metrics.avg_image_size[1],
                        metrics.avg_brightness, metrics.avg_contrast,
                        metrics.blur_score, metrics.noise_level,
                        json.dumps(metrics.color_distribution)
                    ))
                
                # Flush system metrics
                for metrics in self._system_metrics_buffer:
                    cursor.execute('''
                        INSERT INTO system_metrics 
                        (timestamp, cpu_usage, memory_usage, disk_usage, gpu_usage,
                         gpu_memory, inference_queue_size, response_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.cpu_usage, metrics.memory_usage,
                        metrics.disk_usage, metrics.gpu_usage, metrics.gpu_memory,
                        metrics.inference_queue_size, metrics.response_time
                    ))
                
                # Flush business metrics
                for metrics in self._business_metrics_buffer:
                    cursor.execute('''
                        INSERT INTO business_metrics 
                        (timestamp, total_api_calls, successful_predictions, failed_predictions,
                         mask_compliance_rate, peak_usage_hours, geographic_distribution)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.total_api_calls,
                        metrics.successful_predictions, metrics.failed_predictions,
                        metrics.mask_compliance_rate, json.dumps(metrics.peak_usage_hours),
                        json.dumps(metrics.geographic_distribution)
                    ))
                
                conn.commit()
                
                # Clear buffers
                total_flushed = (len(self._model_metrics_buffer) + 
                               len(self._data_metrics_buffer) + 
                               len(self._system_metrics_buffer) + 
                               len(self._business_metrics_buffer))
                
                self._model_metrics_buffer.clear()
                self._data_metrics_buffer.clear()
                self._system_metrics_buffer.clear()
                self._business_metrics_buffer.clear()
                
                logger.info(f"Flushed {total_flushed} metrics to database")
                
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    def get_metrics(self, 
                   metric_type: str, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve metrics from database
        
        Args:
            metric_type: Type of metrics ('model', 'data', 'system', 'business')
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of metric records
        """
        try:
            table_map = {
                'model': 'model_metrics',
                'data': 'data_metrics',
                'system': 'system_metrics',
                'business': 'business_metrics'
            }
            
            if metric_type not in table_map:
                raise ValueError(f"Invalid metric type: {metric_type}")
            
            table = table_map[metric_type]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = f"SELECT * FROM {table}"
                params = []
                
                if start_time is not None or end_time is not None:
                    query += " WHERE "
                    conditions = []
                    
                    if start_time is not None:
                        conditions.append("timestamp >= ?")
                        params.append(start_time)
                    
                    if end_time is not None:
                        conditions.append("timestamp <= ?")
                        params.append(end_time)
                    
                    query += " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve {metric_type} metrics: {e}")
            return []
    
    def start_background_collection(self) -> None:
        """Start background metrics collection"""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            logger.warning("Background collection already running")
            return
        
        self._stop_event.clear()
        self._collection_thread = Thread(target=self._background_collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        
        logger.info("Started background metrics collection")
    
    def stop_background_collection(self) -> None:
        """Stop background metrics collection"""
        if self._collection_thread is None:
            return
        
        self._stop_event.set()
        self._collection_thread.join(timeout=10)
        
        # Flush any remaining metrics
        self.flush_metrics()
        
        logger.info("Stopped background metrics collection")
    
    def _background_collection_loop(self) -> None:
        """Background collection loop"""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics periodically
                self.collect_system_metrics()
                
                # Flush metrics if buffer is getting full
                total_buffered = (len(self._model_metrics_buffer) + 
                                len(self._data_metrics_buffer) + 
                                len(self._system_metrics_buffer) + 
                                len(self._business_metrics_buffer))
                
                if total_buffered >= 100:  # Flush every 100 metrics
                    self.flush_metrics()
                
                # Wait for next collection interval
                self._stop_event.wait(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in background collection: {e}")
                self._stop_event.wait(5)  # Wait 5 seconds on error


def main():
    """CLI entry point for metrics collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mask Detection Metrics Collector')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-collection', action='store_true',
                       help='Start background metrics collection')
    parser.add_argument('--view-metrics', type=str, choices=['model', 'data', 'system', 'business'],
                       help='View metrics of specified type')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of recent metrics to display')
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.config)
    
    if args.start_collection:
        try:
            collector.start_background_collection()
            print("Background metrics collection started. Press Ctrl+C to stop.")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping metrics collection...")
            collector.stop_background_collection()
            
    elif args.view_metrics:
        metrics = collector.get_metrics(args.view_metrics, limit=args.limit)
        
        print(f"\nRecent {args.view_metrics} metrics:")
        print("-" * 60)
        
        for metric in metrics:
            timestamp = datetime.fromtimestamp(metric['timestamp'])
            print(f"Time: {timestamp}")
            for key, value in metric.items():
                if key not in ['id', 'timestamp']:
                    print(f"  {key}: {value}")
            print()
    
    else:
        print("Please specify --start-collection or --view-metrics")


if __name__ == "__main__":
    main()
