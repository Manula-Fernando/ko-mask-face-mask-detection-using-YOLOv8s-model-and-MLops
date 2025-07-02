"""
Flask web application for face mask detection.
This module provides a web interface for uploading images and getting mask predictions.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import yaml
import logging
from datetime import datetime
import base64
import io
from PIL import Image

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import MaskPredictor
from monitoring import ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor and monitor
predictor = None
monitor = ModelMonitor()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_predictor():
    """Initialize the mask predictor."""
    global predictor
    try:
        predictor = MaskPredictor()
        predictor.load_model()
        logger.info("Mask predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        predictor = None

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle image upload and prediction."""
    if request.method == 'GET':
        return render_template('upload.html')
    
    if predictor is None:
        flash('Model not loaded. Please try again later.', 'error')
        return redirect(url_for('predict'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('predict'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('predict'))
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process image
            image = cv2.imread(filepath)
            if image is None:
                flash('Invalid image file', 'error')
                return redirect(url_for('predict'))
            
            # Make prediction
            result = predictor.predict_single(image)
            
            # Log prediction for monitoring
            input_data = {'shape': image.shape}
            monitor.log_prediction(input_data, result)
            
            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return render_template('result.html', 
                                 prediction=result,
                                 image_data=image_base64,
                                 filename=filename)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('predict'))
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.', 'error')
        return redirect(url_for('predict'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Handle base64 encoded image
        if 'image' in request.json:
            image_data = request.json['image']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        elif 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Read image directly from memory
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Make prediction
        result = predictor.predict_single(image_np)
        
        # Log prediction for monitoring
        input_data = {'shape': image_np.shape}
        monitor.log_prediction(input_data, result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/webcam')
def webcam():
    """Webcam detection page."""
    return render_template('webcam.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None
    }
    return jsonify(status)

@app.route('/metrics')
def metrics():
    """Get monitoring metrics."""
    try:
        report = monitor.generate_monitoring_report(days=1)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('predict'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize predictor
    init_predictor()
    
    # Get configuration
    host = config['deployment']['api_host']
    port = config['deployment']['api_port']
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
