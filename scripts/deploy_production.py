#!/usr/bin/env python3
"""
Production Deployment Script for Face Mask Detection MLOps Project
Starts all necessary services including API (FastAPI), monitoring, and MLflow
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
import mlflow

class MLOpsDeployment:
    """Manages deployment of the complete MLOps pipeline"""
    
    def __init__(self):
        self.processes = []
        self.running = True
        self.model_path = Path("models/yolov8_real_face_mask_detection/weights/best.pt")
        self.api_host = "127.0.0.1"
        self.api_port = 8000
        self.mlflow_port = 5000
    
    def start_mlflow_server(self):
        """Start MLflow tracking server"""
        print("Starting MLflow tracking server...")
        mlflow_dir = Path("mlruns")
        mlflow_dir.mkdir(exist_ok=True)
        mlflow_cmd = [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "0.0.0.0",
            "--port", str(self.mlflow_port)
        ]
        try:
            process = subprocess.Popen(
                mlflow_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=10)
            print("MLflow stdout:", stdout)
            print("MLflow stderr:", stderr)
            print(f"MLflow server started on http://127.0.0.1:{self.mlflow_port}")
            return True
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")
            return False
    
    def start_monitoring_service(self):
        """Start model monitoring service"""
        print("Starting model monitoring service...")
        try:
            monitor_cmd = [
                sys.executable, "scripts/model_monitoring.py"
            ]
            process = subprocess.Popen(
                monitor_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("Monitor", process))
            print("Model monitoring service started")
            return True
        except Exception as e:
            print(f"Failed to start monitoring service: {e}")
            return False
    
    def start_drift_detection(self):
        """Start drift detection service"""
        print("Starting drift detection service...")
        try:
            drift_cmd = [
                    sys.executable, "scripts/drift_detection.py",
                     "--reference-dir", "data/processed/yolo_dataset/train",
                     "--current-dir", "data/collected/webcam_detections/images",
                     "--output-dir", "reports/drift_analysis"
                     ]
            subprocess.run(drift_cmd, check=True, capture_output=True, text=True)
            print("Drift detection baseline established")
            return True
        except Exception as e:
            print(f"Failed to run drift detection: {e}")
            return False
    
    def start_web_application(self):
        """Start the main FastAPI web application using uvicorn"""
        print("Starting Face Mask Detection API (FastAPI)...")
        try:
            app_cmd = [
                sys.executable, "-m", "uvicorn", "src.inference.api:app",
                "--host", self.api_host,
                "--port", str(self.api_port),
                "--reload"
            ]
            process = subprocess.Popen(
                app_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("WebApp", process))
            print(f"Web application started on http://{self.api_host}:{self.api_port}")
            return True
        except Exception as e:
            print(f"Failed to start web application: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor all running processes"""
        def check_processes():
            while self.running:
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"WARNING: {name} process has stopped unexpectedly")
                time.sleep(30)
        monitor_thread = threading.Thread(target=check_processes, daemon=True)
        monitor_thread.start()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down MLOps services...")
        self.running = False
        for name, process in self.processes:
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print("All services stopped")
        sys.exit(0)
    
    def deploy(self):
        """Deploy the complete MLOps pipeline"""
        print("FACE MASK DETECTION - PRODUCTION MLOPS DEPLOYMENT")
        print("=" * 60)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        # Check if YOLOv8 model exists
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}. Please ensure the model is trained and available.")
            return False
        # Create necessary directories
        for directory in ["logs", "reports", "temp_uploads"]:
            Path(directory).mkdir(exist_ok=True)
        services_started = 0
        if self.start_mlflow_server():
            services_started += 1
            time.sleep(5)
        if self.start_monitoring_service():
            services_started += 1
            time.sleep(3)
        if self.start_drift_detection():
            services_started += 1
        if self.start_web_application():
            services_started += 1
            time.sleep(3)
        print("\n" + "=" * 60)
        print("DEPLOYMENT SUMMARY")
        print("=" * 60)
        print(f"Services started: {services_started}/4")
        if services_started >= 3:
            print("Deployment successful!")
            print("\nSERVICE URLS:")
            print(f"  Web Application:    http://127.0.0.1:{self.api_port}")
            print(f"  MLflow Tracking:    http://127.0.0.1:{self.mlflow_port}")
            print(f"  Health Check:       http://127.0.0.1:{self.api_port}/health")
            print(f"  Monitoring:         http://127.0.0.1:{self.api_port}/monitoring/dashboard")
            print(f"  Alerts:             http://127.0.0.1:{self.api_port}/monitoring/alerts")
            print("\nAPI ENDPOINTS:")
            print("  POST /predict          - Single image prediction")
            print("  POST /batch-predict    - Batch image prediction")
            print("  GET  /model-info       - Model information")
            print("  GET  /health           - System health check")
            print("\nMONITORING FEATURES:")
            print("  - Real-time performance tracking")
            print("  - Data drift detection")
            print("  - Automated alerts")
            print("  - Comprehensive logging")
            print("  - MLflow experiment tracking")
            print("\n" + "=" * 60)
            print("Press Ctrl+C to stop all services")
            print("=" * 60)
            self.monitor_processes()
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.signal_handler(signal.SIGINT, None)
        else:
            print("Deployment failed - not enough services started")
            self.signal_handler(signal.SIGTERM, None)
            return False
        return True

def main():
    """Main deployment function"""
    deployment = MLOpsDeployment()
    deployment.deploy()

if __name__ == "__main__":
    main()