#!/usr/bin/env python3
"""
Production Deployment Script for Face Mask Detection MLOps Project
Starts all necessary services including API, monitoring, and MLflow
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class MLOpsDeployment:
    """Manages deployment of the complete MLOps pipeline"""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_mlflow_server(self):
        """Start MLflow tracking server"""
        print("🚀 Starting MLflow tracking server...")
        
        # Create MLflow directory
        mlflow_dir = Path("mlruns")
        mlflow_dir.mkdir(exist_ok=True)
        
        # Start MLflow server
        mlflow_cmd = [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "0.0.0.0",
            "--port", "5001"
        ]
        
        try:
            process = subprocess.Popen(
                mlflow_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("MLflow", process))
            print("✅ MLflow server started on http://localhost:5001")
            return True
        except Exception as e:
            print(f"❌ Failed to start MLflow server: {e}")
            return False
    
    def start_monitoring_service(self):
        """Start model monitoring service"""
        print("📊 Starting model monitoring service...")
        
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
            print("✅ Model monitoring service started")
            return True
        except Exception as e:
            print(f"❌ Failed to start monitoring service: {e}")
            return False
    
    def start_drift_detection(self):
        """Start drift detection service"""
        print("🔍 Starting drift detection service...")
        
        try:
            drift_cmd = [
                sys.executable, "scripts/drift_detection.py"
            ]
            
            # Run drift detection once to establish baseline
            subprocess.run(drift_cmd, check=True, capture_output=True, text=True)
            print("✅ Drift detection baseline established")
            return True
        except Exception as e:
            print(f"❌ Failed to run drift detection: {e}")
            return False
    
    def start_web_application(self):
        """Start the main web application"""
        print("🌐 Starting Face Mask Detection API...")
        
        try:
            app_cmd = [
                sys.executable, "app/main.py"
            ]
            
            process = subprocess.Popen(
                app_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("WebApp", process))
            print("✅ Web application started on http://localhost:5000")
            return True
        except Exception as e:
            print(f"❌ Failed to start web application: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor all running processes"""
        def check_processes():
            while self.running:
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️ {name} process has stopped unexpectedly")
                        # In production, you might want to restart the process
                time.sleep(30)  # Check every 30 seconds
        
        monitor_thread = threading.Thread(target=check_processes, daemon=True)
        monitor_thread.start()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down MLOps services...")
        self.running = False
        
        for name, process in self.processes:
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All services stopped")
        sys.exit(0)
    
    def deploy(self):
        """Deploy the complete MLOps pipeline"""
        print("🚀 FACE MASK DETECTION - PRODUCTION MLOPS DEPLOYMENT")
        print("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check if model exists
        model_path = Path("models/best_mask_detector_imbalance_optimized.h5")
        if not model_path.exists():
            print("❌ Model file not found. Please ensure the model is trained and available.")
            return False
        
        # Create necessary directories
        for directory in ["logs", "reports", "temp_uploads"]:
            Path(directory).mkdir(exist_ok=True)
        
        # Start services in order
        services_started = 0
        
        # 1. Start MLflow server
        if self.start_mlflow_server():
            services_started += 1
            time.sleep(5)  # Give MLflow time to start
        
        # 2. Start monitoring service
        if self.start_monitoring_service():
            services_started += 1
            time.sleep(3)
        
        # 3. Run drift detection baseline
        if self.start_drift_detection():
            services_started += 1
        
        # 4. Start web application
        if self.start_web_application():
            services_started += 1
            time.sleep(3)
        
        print("\n" + "=" * 60)
        print("🎯 DEPLOYMENT SUMMARY")
        print("=" * 60)
        print(f"Services started: {services_started}/4")
        
        if services_started >= 3:  # At minimum need web app and one monitoring service
            print("✅ Deployment successful!")
            print("\n📍 SERVICE URLS:")
            print("  🌐 Web Application:    http://localhost:5000")
            print("  📊 MLflow Tracking:    http://localhost:5001")
            print("  ❤️ Health Check:       http://localhost:5000/health")
            print("  📈 Monitoring:         http://localhost:5000/monitoring/dashboard")
            print("  🔍 Alerts:             http://localhost:5000/monitoring/alerts")
            
            print("\n🎮 API ENDPOINTS:")
            print("  POST /predict          - Single image prediction")
            print("  POST /batch-predict    - Batch image prediction")
            print("  GET  /model-info       - Model information")
            print("  GET  /health           - System health check")
            
            print("\n📊 MONITORING FEATURES:")
            print("  • Real-time performance tracking")
            print("  • Data drift detection")
            print("  • Automated alerts")
            print("  • Comprehensive logging")
            print("  • MLflow experiment tracking")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to stop all services")
            print("=" * 60)
            
            # Start process monitoring
            self.monitor_processes()
            
            # Keep main thread alive
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.signal_handler(signal.SIGINT, None)
                
        else:
            print("❌ Deployment failed - not enough services started")
            self.signal_handler(signal.SIGTERM, None)
            return False
        
        return True

def main():
    """Main deployment function"""
    deployment = MLOpsDeployment()
    deployment.deploy()

if __name__ == "__main__":
    main()
