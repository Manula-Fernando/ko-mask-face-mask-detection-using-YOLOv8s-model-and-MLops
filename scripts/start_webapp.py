#!/usr/bin/env python3
"""
Web Application Launcher
Start the Flask web application for face mask detection.
"""

import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

def main():
    """Launch the Flask web application."""
    print("🌐 Starting Face Mask Detection Web Application")
    print("="*50)
    
    try:
        # Import and run Flask app
        from main import app
        
        print("🚀 Starting server...")
        print("📱 Access the web interface at: http://localhost:5000")
        print("🔍 Upload images or use the webcam for detection")
        print("⭐ Press Ctrl+C to stop the server")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start web application: {e}")
        raise

if __name__ == "__main__":
    main()
