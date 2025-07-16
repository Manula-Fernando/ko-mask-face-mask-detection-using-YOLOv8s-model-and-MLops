#!/usr/bin/env python3
"""
Face Mask Detection System - Quick Start
General launcher for development, demo, or non-medical use.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import socket


PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def print_header():
    print("\n" + "=" * 60)
    print("😷 Face Mask Detection System - Quick Start")
    print("=" * 60)
    print("🔬 General Detection • Real Data Only")
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies exist"""
    print("🔍 Checking system dependencies...")
    predictor_path = PROJECT_ROOT / "src" / "inference" / "predictor.py"
    if not predictor_path.exists() or predictor_path.stat().st_size == 0:
        print("❌ Error: Predictor not found or empty!")
        return False
    print("✅ Predictor module: OK")
    model_paths = [
        PROJECT_ROOT / "models" / "best.pt",
        PROJECT_ROOT / "models" / "yolov8_real_face_mask_detection" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    ]
    model_found = any(p.exists() for p in model_paths)
    if model_found:
        print("✅ YOLO model: OK")
    else:
        print("⚠️  No trained model found, will use pretrained fallback")
    return True

def start_service(cmd, name, port=None):
    """Start a service and return process, with port check."""
    if port and is_port_in_use(port):
        print(f"❌ {name} port {port} already in use. Please free the port and try again.")
        return None
    print(f"🚀 Starting {name}...")
    try:
        process = subprocess.Popen(cmd, shell=True, cwd=PROJECT_ROOT)
        if port:
            time.sleep(3)
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def launch_app(choice):
    """Launch specific application based on user choice."""
    if choice.lower() == 'w':
        print("🎥 Launching webcam detector...")
        try:
            sys.path.append(str(PROJECT_ROOT / "src"))
            from inference.predictor import FaceMaskPredictor
            print("✅ Predictor module loaded successfully")
            webcam_cmd = f'"{sys.executable}" src/realtime_webcam_app.py'
            subprocess.run(webcam_cmd, shell=True, cwd=PROJECT_ROOT)
        except ImportError as e:
            print(f"❌ Could not import FaceMaskPredictor. Details: {e}")
            print("Please ensure src/inference/predictor.py exists and is properly configured.")
    elif choice.lower() == 'd':
        print("📊 Launching monitoring dashboard...")
        dashboard_cmd = f'"{sys.executable}" -m streamlit run src/monitoring/dashboard.py --server.port 8501 --server.headless true'
        start_service(dashboard_cmd, "Dashboard", port=8501)
    elif choice.lower() == 'a':
        print("🌐 Launching API interface...")
        api_cmd = f'"{sys.executable}" -m uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8001'
        start_service(api_cmd, "API", port=8001)
    elif choice.lower() == 'q':
        print("👋 Shutting down system...")
        return False
    else:
        print(f"❓ Unknown command: {choice}. Use: w=webcam, d=dashboard, a=api, q=quit")
    return True

def main():
    print_header()
    if not check_dependencies():
        print("❌ System check failed. Please fix dependencies first.")
        return
    print("\nOptions:")
    print("   w - 🎥 Webcam Detection (Real-time)")
    print("   d - 📊 Monitoring Dashboard")
    print("   a - 🌐 API Interface")
    print("   q - 👋 Quit")
    print()
    while True:
        try:
            choice = input("Enter command (w/d/a/q): ").strip()
            if not launch_app(choice):
                break
            print()
        except KeyboardInterrupt:
            print("\n👋 System shutdown requested")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
