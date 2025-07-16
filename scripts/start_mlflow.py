#!/usr/bin/env python3
"""
MLflow UI Launcher
Start the MLflow tracking UI to view experiments and models.
"""

import subprocess
import sys
import os

def main() -> None:
    """Launch MLflow UI."""
    print("📊 Starting MLflow Tracking UI")
    print("="*50)
    
    try:
        print("🚀 Starting MLflow server...")
        print("📈 Access MLflow UI at: http://localhost:5000")
        print("🔍 View experiments, metrics, and model registry")
        print("⭐ Press Ctrl+C to stop the server")
        
        # Start MLflow UI (blocking call)
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--host", "127.0.0.1",
            "--port", "5000"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n✅ MLflow UI stopped")
    except FileNotFoundError:
        print("❌ MLflow executable not found.")
        print("💡 Make sure MLflow is installed: pip install mlflow")
    except Exception as e:
        print(f"❌ Failed to start MLflow UI: {e}")
        print("💡 Make sure MLflow is installed and your environment is activated.")
        raise

if __name__ == "__main__":
    main()
