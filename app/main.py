"""
Enhanced Flask API for Face Mask Detection
Production-ready deployment with comprehensive monitoring and MLOps features
"""

import os
import io
import cv2
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# Import monitoring components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.model_monitoring import ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = "models/best_mask_detector_imbalance_optimized.h5"
BACKUP_MODEL_PATH = "models/best_mask_detector_enhanced_mlflow.h5"
CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Global model variable and monitor
model = None
monitor = None

def initialize_monitor():
    """Initialize the model monitoring system"""
    global monitor
    try:
        monitor = ModelMonitor(
            model_name="face_mask_detector_api",
            db_path="logs/api_monitoring.db",
            mlflow_uri="http://localhost:5001"
        )
        # Start continuous monitoring
        monitor.start_monitoring(check_interval=300)  # Check every 5 minutes
        logger.info("✅ Model monitoring initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring: {e}")

def load_model():
    """Load the trained model with fallback options."""
    global model
    
    # Try primary model first
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Primary model loaded: {MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load primary model: {e}")
    
    # Try backup model
    if os.path.exists(BACKUP_MODEL_PATH):
        try:
            model = tf.keras.models.load_model(BACKUP_MODEL_PATH)
            logger.info(f"✅ Backup model loaded: {BACKUP_MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load backup model: {e}")
    
    logger.error("❌ No valid model found")
    return False

def preprocess_image(image_data):
    """Preprocess image for model prediction."""
    try:
        # Convert PIL Image to numpy array
        if hasattr(image_data, 'convert'):
            image = np.array(image_data.convert('RGB'))
        else:
            image = np.array(image_data)
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Normalize using ImageNet statistics (same as training)
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed."""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint with monitoring status."""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'monitoring_active': monitor is not None,
        'timestamp': datetime.now().isoformat(),
        'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else BACKUP_MODEL_PATH,
        'classes': CLASSES
    }
    
    # Add monitoring health if available
    if monitor:
        try:
            recent_report = monitor.generate_monitoring_report(1)  # Last hour
            health_status['monitoring'] = {
                'predictions_last_hour': recent_report['predictions']['total_predictions'],
                'avg_confidence': recent_report['model_metrics']['avg_confidence'],
                'alerts_last_hour': recent_report['alerts']['total_alerts']
            }
        except Exception as e:
            health_status['monitoring'] = {'error': str(e)}
    
    return jsonify(health_status)

@app.route('/model-info')
def model_info():
    """Get detailed model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        return jsonify({
            'model_loaded': True,
            'total_parameters': int(model.count_params()),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'classes': CLASSES,
            'num_classes': len(CLASSES),
            'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else BACKUP_MODEL_PATH
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image with comprehensive monitoring."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400
    
    # Start timing for performance monitoring
    start_time = time.time()
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = CLASSES[predicted_class_idx]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # All class probabilities
        all_predictions = {
            CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))
        }
        
        # Confidence level assessment
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        # Log prediction for monitoring (if monitor is available)
        if monitor:
            try:
                monitor.log_prediction(
                    input_data=processed_image,
                    prediction=predicted_class,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_version="v1.0"
                )
            except Exception as e:
                logger.warning(f"Failed to log prediction to monitor: {e}")
        
        response = {
            'prediction': predicted_class,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'all_predictions': all_predictions,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'filename': secure_filename(file.filename),
            'model_version': 'v1.0'
        }
        
        logger.info(f"Prediction made: {predicted_class} ({confidence:.2%}) for {file.filename} in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions on multiple uploaded images."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    successful_predictions = 0
    
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type or empty filename'
            })
            continue
        
        try:
            # Process image
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
            if processed_image is None:
                results.append({
                    'filename': secure_filename(file.filename),
                    'error': 'Failed to process image'
                })
                continue
            
            # Make prediction
            predictions = model.predict(processed_image, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            predicted_class = CLASSES[predicted_class_idx]
            
            # All class probabilities
            all_predictions = {
                CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))
            }
            
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            
            results.append({
                'filename': secure_filename(file.filename),
                'prediction': predicted_class,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'all_predictions': all_predictions
            })
            
            successful_predictions += 1
            
        except Exception as e:
            results.append({
                'filename': secure_filename(file.filename),
                'error': str(e)
            })
    
    response = {
        'total_files': len(files),
        'successful_predictions': successful_predictions,
        'failed_predictions': len(files) - successful_predictions,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Batch prediction completed: {successful_predictions}/{len(files)} successful")
    return jsonify(response)

@app.route('/stats')
def stats():
    """Get API usage statistics."""
    # In a production environment, you would track these in a database
    return jsonify({
        'uptime': 'Available since app start',
        'model_info': {
            'loaded': model is not None,
            'path': MODEL_PATH if os.path.exists(MODEL_PATH) else BACKUP_MODEL_PATH
        },
        'supported_formats': ['PNG', 'JPG', 'JPEG', 'GIF', 'BMP', 'TIFF'],
        'max_file_size': '16MB',
        'classes': CLASSES,
        'endpoints': ['/predict', '/batch-predict', '/health', '/model-info', '/stats']
    })

@app.route('/monitoring/dashboard')
def monitoring_dashboard():
    """Display monitoring dashboard."""
    if monitor is None:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        # Generate comprehensive report
        report = monitor.generate_monitoring_report(24)  # Last 24 hours
        
        return jsonify({
            'status': 'success',
            'monitoring_report': report,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to generate monitoring report: {str(e)}'}), 500

@app.route('/monitoring/metrics')
def monitoring_metrics():
    """Get current model metrics for monitoring."""
    if monitor is None:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        # Calculate recent metrics
        metrics = monitor.calculate_performance_metrics(1)  # Last hour
        
        if metrics:
            return jsonify({
                'status': 'success',
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'avg_confidence': metrics.avg_confidence,
                    'prediction_count': metrics.prediction_count,
                    'timestamp': metrics.timestamp.isoformat()
                }
            })
        else:
            return jsonify({
                'status': 'no_data',
                'message': 'No metrics available for the specified period'
            })
    except Exception as e:
        return jsonify({'error': f'Failed to get metrics: {str(e)}'}), 500

@app.route('/monitoring/alerts')
def monitoring_alerts():
    """Get recent alerts from monitoring system."""
    if monitor is None:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        import sqlite3
        with sqlite3.connect(monitor.db_path) as conn:
            query = '''
                SELECT timestamp, alert_type, severity, message, resolved
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT 50
            '''
            alerts_df = pd.read_sql_query(query, conn)
        
        alerts = alerts_df.to_dict('records') if not alerts_df.empty else []
        
        return jsonify({
            'status': 'success',
            'alerts': alerts,
            'alert_count': len(alerts)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get alerts: {str(e)}'}), 500

# Initialize everything when the app starts
def initialize_app():
    """Initialize the application with model and monitoring."""
    logger.info("🚀 Initializing Face Mask Detection API...")
    
    # Load model
    if not load_model():
        logger.error("❌ Failed to load model")
        return False
    
    # Initialize monitoring
    initialize_monitor()
    
    logger.info("✅ Application initialized successfully")
    return True

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle server errors."""
    return jsonify({'error': 'Internal server error'}), 500

# Load model on startup
if __name__ == '__main__':
    print("🎭 Face Mask Detection API - Production MLOps")
    print("=" * 50)
    
    # Initialize application
    if initialize_app():
        print(f"🤖 Model: ✅ Loaded")
        print(f"📊 Monitoring: {'✅ Active' if monitor else '❌ Inactive'}")
        print(f"🌐 URL: http://localhost:5000")
        print(f"📊 Health: http://localhost:5000/health")
        print(f"� Monitoring: http://localhost:5000/monitoring/dashboard")
        print("=" * 50)
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    else:
        print("❌ Failed to initialize application")
        print("=" * 50)
