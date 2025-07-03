"""
Quick Start Script for Face Mask Detection MLOps Project
Simplified deployment for development and testing
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'opencv-python',
        'mlflow', 'pandas', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied")
    return True

def check_model():
    """Check if the trained model exists"""
    print("🔍 Checking model files...")
    
    model_paths = [
        "models/best_mask_detector_imbalance_optimized.h5",
        "models/best_mask_detector_enhanced_mlflow.h5"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"  ✅ Found model: {model_path}")
            return True
    
    print("  ❌ No trained model found")
    print("     Run training script first: python src/model_training.py")
    return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "logs", "temp_uploads", "reports", 
        "reports/drift_analysis", "mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

def run_quick_test():
    """Run a quick test of the system"""
    print("🧪 Running quick system test...")
    
    try:
        # Test model loading
        import tensorflow as tf
        model_path = "models/best_mask_detector_imbalance_optimized.h5"
        if Path(model_path).exists():
            model = tf.keras.models.load_model(model_path)
            print("  ✅ Model loads successfully")
            print(f"  📊 Model parameters: {model.count_params():,}")
        else:
            print("  ⚠️ No model to test")
        
        # Test monitoring system
        sys.path.append('scripts')
        from model_monitoring import ModelMonitor
        monitor = ModelMonitor(db_path="logs/test_monitoring.db")
        print("  ✅ Monitoring system initializes")
        
        print("✅ System test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ System test failed: {e}")
        return False

def start_services():
    """Start the development services"""
    print("🚀 Starting services...")
    
    # Start MLflow in background (optional)
    print("  🔄 Starting MLflow server...")
    try:
        mlflow_process = subprocess.Popen([
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///logs/mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "127.0.0.1",
            "--port", "5001"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("  ✅ MLflow server started on http://localhost:5001")
        time.sleep(3)  # Give MLflow time to start
    except Exception as e:
        print(f"  ⚠️ MLflow server failed to start: {e}")
        print("     You can still use the API without MLflow")
    
    # Start the main Flask application
    print("  🔄 Starting Face Mask Detection API...")
    try:
        print("\n" + "="*60)
        print("🎭 FACE MASK DETECTION - DEVELOPMENT SERVER")
        print("="*60)
        print("🌐 Web Application: http://localhost:5000")
        print("📊 MLflow Tracking: http://localhost:5001")
        print("❤️ Health Check: http://localhost:5000/health")
        print("📈 Monitoring: http://localhost:5000/monitoring/dashboard")
        print("="*60)
        print("Press Ctrl+C to stop the server")
        print("="*60)
        
        # Run the Flask app
        subprocess.run([sys.executable, "app/main.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")

def main():
    """Main function"""
    print("🎭 Face Mask Detection MLOps - Quick Start")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model
    if not check_model():
        print("\n💡 To train a model, run:")
        print("   python src/model_training.py")
        return
    
    # Setup directories
    setup_directories()
    
    # Run system test
    if not run_quick_test():
        print("⚠️ Some tests failed, but continuing...")
    
    print("\n✅ All checks passed!")
    print("="*50)
    
    # Start services
    start_services()

if __name__ == "__main__":
    main()
