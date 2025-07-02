"""
Simplified Flask app for testing
"""
from flask import Flask, Response, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    """Home page."""
    return '<html><body><h1>Face Mask Detection</h1><p>Home Page</p></body></html>'

@app.route('/webcam')
def webcam():
    """Webcam page for real-time detection."""
    return '<html><body><h1>Webcam Feed</h1><p>Webcam Page</p></body></html>'

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate_test_frames():
        # Simple test frame data
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\ntest_frame_data\r\n'
    
    return Response(generate_test_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint."""
    return json.dumps({
        'status': 'healthy',
        'models_loaded': False,  # For testing purposes
        'camera_available': False
    })

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return "Internal server error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
