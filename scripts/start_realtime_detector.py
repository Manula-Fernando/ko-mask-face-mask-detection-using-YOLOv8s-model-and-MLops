#!/usr/bin/env python3
"""
Real-Time Face Mask Detection Launcher
Launch the enhanced real-time face mask detector application.
"""

import sys
import os
from pathlib import Path

def main() -> None:
    """Launch the real-time mask detector."""
    print("🎭 Real-Time Face Mask Detection Launcher")
    print("="*50)
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.append(str(src_path))
    try:
        # Import and run the detector
        from realtime_webcam_app import main as detector_main
        print("🚀 Starting Real-Time Face Mask Detector...")
        print("📹 Make sure your webcam is connected and working")
        print("⭐ Press Ctrl+C to stop")
        print()
        detector_main()
    except ImportError as e:
        print(f"❌ Failed to import real-time detector: {e}")
        print(f"Please ensure {src_path}/realtime_webcam_app.py exists and is properly configured.")
    except Exception as e:
        print(f"❌ An error occurred while running the detector: {e}")
        print("Check your webcam, dependencies, and configuration.")

if __name__ == "__main__":
    main()
