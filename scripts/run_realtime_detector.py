#!/usr/bin/env python3
"""
Real-Time Face Mask Detection Launcher
Launch the enhanced real-time face mask detector application.
"""

import sys
import os
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

def main():
    """Launch the real-time mask detector."""
    print("🎭 Real-Time Face Mask Detection Launcher")
    print("="*50)
    
    try:
        # Import and run the detector
        from realtime_mask_detector import main as detector_main
        
        print("🚀 Starting Real-Time Face Mask Detector...")
        print("📹 Make sure your webcam is connected and working")
        print("⭐ Press Ctrl+C to stop")
        print()
        
        detector_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure app/realtime_mask_detector.py exists")
    except Exception as e:
        print(f"❌ Error starting detector: {e}")
        raise

if __name__ == "__main__":
    main()
