#!/usr/bin/env python3
"""
Web Application Launcher
Start the FastAPI web application for face mask detection.
"""

import sys
from pathlib import Path

def main() -> None:
    """Launch the FastAPI web application."""
    print("🌐 Starting Face Mask Detection Web Application")
    print("="*50)
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.append(str(src_path))
    try:
        # Import and run FastAPI app
        from main import app
        import uvicorn
        print("🚀 Starting server...")
        print("📱 Access the web interface at: http://localhost:8001")
        print("🔍 Upload images or use the webcam for detection")
        print("⭐ Press Ctrl+C to stop the server")
        uvicorn.run(app, host='0.0.0.0', port=8001, reload=False)
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        print(f"Please ensure {src_path}/main.py or {src_path}/app.py exists and is properly configured.")
    except Exception as e:
        print(f"❌ An error occurred while running the web application: {e}")
        print("Check your FastAPI app, dependencies, and configuration.")

if __name__ == "__main__":
    main()
