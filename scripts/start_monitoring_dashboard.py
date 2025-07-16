#!/usr/bin/env python3
"""
Monitoring Dashboard Launcher
Start the model monitoring dashboard for face mask detection.
"""

import sys
from pathlib import Path

def main() -> None:
    """Launch the monitoring dashboard web application."""
    print("📊 Starting Face Mask Detection Monitoring Dashboard")
    print("="*50)
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.append(str(src_path))
    try:
        # Import and run the monitoring dashboard app
        from monitoring_dashboard import app
        import uvicorn
        print("🚀 Starting monitoring dashboard server...")
        print("📈 Access the dashboard at: http://localhost:8501")
        print("🔍 View model performance, drift, and alerts")
        print("⭐ Press Ctrl+C to stop the server")
        uvicorn.run(app, host='0.0.0.0', port=8501, reload=False)
    except ImportError as e:
        print(f"❌ Failed to import monitoring dashboard app: {e}")
        print(f"Please ensure {src_path}/monitoring_dashboard.py exists and is properly configured.")
    except Exception as e:
        print(f"❌ An error occurred while running the monitoring dashboard: {e}")
        print("Check your dashboard app, dependencies, and configuration.")

if __name__ == "__main__":
    main()
