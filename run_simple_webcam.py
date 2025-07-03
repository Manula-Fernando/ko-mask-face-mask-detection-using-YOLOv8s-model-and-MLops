#!/usr/bin/env python3
"""
Advanced launcher for the Face Mask Detection Webcam Application
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from app.simple_webcam import main
    
    print("🎭 Advanced Face Mask Detection System")
    print("=" * 55)
    print("🚀 Starting advanced webcam application...")
    print("📹 This will open a new window showing your webcam feed")
    print("🔍 Face mask detection will run in real-time")
    print("📊 Features: Statistics, High-confidence saving, Performance monitoring")
    print("🎮 Controls: 'q' to quit, 's' for stats, 'c' for confidence bars, 'r' to reset")
    print()
    
    main()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install opencv-python tensorflow albumentations")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check that your webcam is connected and working.")
