#!/usr/bin/env python3
"""
Face Mask Detection MLOps Pipeline Launcher

This script integrates and launches the complete MLOps pipeline with your trained model.
It provides a unified entry point to start all services and components.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path
from threading import Thread
import webbrowser
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.common.logger import get_logger
from src.common.utils import load_config

logger = get_logger(__name__)


class MLOpsPipelineLauncher:
    """Complete MLOps pipeline launcher and manager"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline launcher
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        self.processes: List[subprocess.Popen] = []
        self.services = {
            'api': None,
            'monitoring': None,
            'dashboard': None,
            'mlflow': None
        }
        
        # Validate model exists
        model_path = Path(self.config['model']['path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        
        logger.info(f"MLOps Pipeline initialized with model: {model_path}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "logs",
            "temp_uploads", 
            "detections",
            "mlruns",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def start_mlflow_server(self, port: int = 5000):
        """Start MLflow tracking server"""
        try:
            logger.info(f"Starting MLflow server on port {port}...")
            
            # Check if MLflow is already running
            import requests
            try:
                response = requests.get(f"http://localhost:{port}")
                if response.status_code == 200:
                    logger.info("MLflow server already running")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            # Start MLflow server
            mlflow_cmd = [
                sys.executable, "-m", "mlflow", "server",
                "--backend-store-uri", "sqlite:///mlruns/mlflow.db",
                "--default-artifact-root", "./mlruns",
                "--host", "0.0.0.0",
                "--port", str(port)
            ]
            
            process = subprocess.Popen(
                mlflow_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.services['mlflow'] = process
            self.processes.append(process)
            
            # Wait a moment for server to start
            time.sleep(3)
            logger.info(f"✅ MLflow server started: http://localhost:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
    
    def start_api_server(self, port: int = 8000):
        """Start FastAPI inference server"""
        try:
            logger.info(f"Starting API server on port {port}...")
            
            api_cmd = [
                sys.executable, "-m", "uvicorn",
                "src.inference.api:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload"
            ]
            
            process = subprocess.Popen(
                api_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.services['api'] = process
            self.processes.append(process)
            
            # Wait for API to start
            time.sleep(5)
            logger.info(f"✅ API server started: http://localhost:{port}")
            logger.info(f"📚 API docs available at: http://localhost:{port}/docs")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
    
    def start_monitoring_dashboard(self, port: int = 8501):
        """Start Streamlit monitoring dashboard"""
        try:
            logger.info(f"Starting monitoring dashboard on port {port}...")
            
            dashboard_cmd = [
                sys.executable, "-m", "streamlit", "run",
                "src/monitoring/dashboard.py",
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            process = subprocess.Popen(
                dashboard_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.services['dashboard'] = process
            self.processes.append(process)
            
            # Wait for dashboard to start
            time.sleep(8)
            logger.info(f"✅ Monitoring dashboard started: http://localhost:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring dashboard: {e}")
    
    def start_monitoring_service(self):
        """Start background monitoring service"""
        try:
            logger.info("Starting background monitoring service...")
            
            monitor_cmd = [
                sys.executable, "src/monitoring/service.py",
                "--config", self.config_path,
                "--start"
            ]
            
            process = subprocess.Popen(
                monitor_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.services['monitoring'] = process
            self.processes.append(process)
            
            time.sleep(2)
            logger.info("✅ Background monitoring service started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {e}")
    
    def health_check(self):
        """Perform health check on all services"""
        logger.info("Performing health check on all services...")
        
        import requests
        
        services_status = {}
        
        # Check API server
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            services_status['api'] = response.status_code == 200
        except:
            services_status['api'] = False
        
        # Check MLflow server
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            services_status['mlflow'] = response.status_code == 200
        except:
            services_status['mlflow'] = False
        
        # Check Streamlit dashboard
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            services_status['dashboard'] = response.status_code == 200
        except:
            services_status['dashboard'] = False
        
        # Check monitoring service (process-based)
        services_status['monitoring'] = (
            self.services['monitoring'] is not None and 
            self.services['monitoring'].poll() is None
        )
        
        return services_status
    
    def display_status(self, services_status: Dict[str, bool]):
        """Display service status"""
        print("\n" + "="*60)
        print("🚀 FACE MASK DETECTION MLOps PIPELINE STATUS")
        print("="*60)
        
        for service, status in services_status.items():
            status_icon = "✅" if status else "❌"
            service_name = service.replace('_', ' ').title()
            print(f"{status_icon} {service_name:<20} {'Running' if status else 'Failed'}")
        
        print("\n📊 SERVICE URLS:")
        if services_status['api']:
            print("🔗 API Server:           http://localhost:8000")
            print("📚 API Documentation:    http://localhost:8000/docs")
        
        if services_status['mlflow']:
            print("🔗 MLflow Tracking:      http://localhost:5000")
        
        if services_status['dashboard']:
            print("🔗 Monitoring Dashboard: http://localhost:8501")
        
        print("\n🎯 QUICK COMMANDS:")
        print("• Test API:              curl http://localhost:8000/health")
        print("• Real-time Detection:   python src/realtime_webcam_app.py")
        print("• View Logs:             tail -f logs/app.log")
        print("• Stop Pipeline:         Ctrl+C")
        print("="*60)
    
    def run_tests(self):
        """Run basic integration tests"""
        logger.info("Running integration tests...")
        
        try:
            import requests
            
            # Test API health
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200, "API health check failed"
            
            # Test model prediction (if test image available)
            test_image_path = Path("data/test_image.jpg")
            if test_image_path.exists():
                with open(test_image_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post("http://localhost:8000/predict/single", files=files)
                    assert response.status_code == 200, "Prediction test failed"
            
            logger.info("✅ Integration tests passed")
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
    
    def open_browser_tabs(self):
        """Open browser tabs for all services"""
        urls = [
            "http://localhost:8000/docs",  # API docs
            "http://localhost:8501",       # Dashboard
            "http://localhost:5000"        # MLflow
        ]
        
        for url in urls:
            try:
                webbrowser.open(url)
                time.sleep(1)
            except:
                pass
    
    def start_complete_pipeline(self, open_browser: bool = True):
        """Start the complete MLOps pipeline"""
        logger.info("🚀 Starting Face Mask Detection MLOps Pipeline...")
        
        try:
            # Setup
            self.setup_directories()
            
            # Start services in order
            self.start_mlflow_server()
            self.start_monitoring_service()
            self.start_api_server()
            self.start_monitoring_dashboard()
            
            # Health check
            time.sleep(5)
            services_status = self.health_check()
            self.display_status(services_status)
            
            # Run tests
            if services_status['api']:
                self.run_tests()
            
            # Open browser tabs
            if open_browser and any(services_status.values()):
                time.sleep(2)
                self.open_browser_tabs()
            
            logger.info("🎉 Pipeline startup complete!")
            
            return services_status
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Shutdown all services gracefully"""
        logger.info("🛑 Shutting down MLOps pipeline...")
        
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        logger.info("✅ Pipeline shutdown complete")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Face Mask Detection MLOps Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser tabs automatically')
    parser.add_argument('--api-only', action='store_true',
                       help='Start only the API server')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Start only the monitoring dashboard')
    parser.add_argument('--test', action='store_true',
                       help='Run integration tests only')
    
    args = parser.parse_args()
    
    try:
        # Initialize launcher
        launcher = MLOpsPipelineLauncher(args.config)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, launcher.signal_handler)
        signal.signal(signal.SIGTERM, launcher.signal_handler)
        
        if args.test:
            # Run tests only
            launcher.setup_directories()
            launcher.start_api_server()
            time.sleep(10)
            launcher.run_tests()
            launcher.shutdown()
            
        elif args.api_only:
            # Start API only
            launcher.setup_directories()
            launcher.start_api_server()
            print("✅ API server started. Press Ctrl+C to stop.")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.shutdown()
                
        elif args.dashboard_only:
            # Start dashboard only
            launcher.setup_directories()
            launcher.start_monitoring_dashboard()
            print("✅ Monitoring dashboard started. Press Ctrl+C to stop.")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.shutdown()
        
        else:
            # Start complete pipeline
            services_status = launcher.start_complete_pipeline(not args.no_browser)
            
            if any(services_status.values()):
                print("\n🎯 Pipeline is running! Press Ctrl+C to stop.")
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    launcher.shutdown()
            else:
                logger.error("❌ Failed to start any services")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline startup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
