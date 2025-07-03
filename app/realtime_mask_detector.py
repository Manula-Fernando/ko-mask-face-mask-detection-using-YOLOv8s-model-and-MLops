#!/usr/bin/env python3
"""
Professional Face Mask Detection - Advanced Real-time Webcam Application
Enterprise-grade real-time face mask detection with sophisticated UI and comprehensive features.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
import sys
import os
from datetime import datetime
import logging
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from predict import FaceMaskPredictor
except ImportError:
    print("❌ Error: Could not import FaceMaskPredictor.")
    print("Please ensure src/predict.py exists and is properly configured.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalWebcamDetector:
    """ real-time face mask detection with advanced UI and analytics."""
    
    def __init__(self, model_path: str):
        """Initialize the enterprise webcam detector."""
        self.model_path = model_path
        self.predictor = None
        self.cap = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        self.detection_count = 0
        self.session_start = datetime.now()
        
        # Initialize face cascade for face detection
        self.face_cascade = None
        self.setup_face_detection()
        
        # Enterprise color scheme (BGR format) - Tech-focused design
        self.colors = {
            'with_mask': (0, 255, 127),        # Bright Green
            'without_mask': (0, 69, 255),      # Bright Red  
            'mask_weared_incorrect': (0, 165, 255),  # Orange
            'unknown': (128, 128, 128),        # Gray
            'ui_primary': (40, 40, 40),        # Dark Gray
            'ui_secondary': (20, 20, 20),      # Very Dark Gray
            'ui_accent': (255, 255, 0),        # Cyan/Yellow
            'ui_text': (255, 255, 255),        # White
            'ui_success': (0, 255, 0),         # Pure Green
            'ui_warning': (0, 255, 255),       # Yellow
            'ui_error': (0, 0, 255),           # Pure Red
            'neon_blue': (255, 150, 0),        # Neon Blue
            'neon_cyan': (255, 255, 0),        # Neon Cyan
            'tech_green': (0, 255, 100)        # Tech Green
        }
        
        # Professional class display names
        self.class_names = {
            'with_mask': 'MASK DETECTED',
            'without_mask': 'NO MASK DETECTED',
            'mask_weared_incorrect': 'IMPROPER MASK USAGE',
            'unknown': 'ANALYSIS PENDING'
        }
        
        # Confidence level indicators
        self.confidence_levels = {
            'high': {'threshold': 0.85, 'label': 'HIGH CONFIDENCE', 'color': (0, 255, 0)},
            'medium': {'threshold': 0.65, 'label': 'MEDIUM CONFIDENCE', 'color': (0, 255, 255)},
            'low': {'threshold': 0.0, 'label': 'LOW CONFIDENCE', 'color': (0, 165, 255)}
        }
        
        # Detection analytics
        self.analytics = {
            'total_detections': 0,
            'with_mask': 0,
            'without_mask': 0,
            'mask_weared_incorrect': 0,
            'high_confidence_detections': 0,
            'avg_confidence': 0.0,
            'detection_history': []
        }
        
        # Enhanced detection saving
        self.detection_dir = Path(__file__).parent.parent / "high_confident_FMD_images"
        self.detection_dir.mkdir(exist_ok=True)
        self.high_confidence_dir = self.detection_dir  # For compatibility
        self.high_confidence_threshold = 0.85
        self.save_detections = True
        
        # Advanced window and display settings
        self.window_width = 1360  # Enhanced resolution (more standard)
        self.window_height = 1160  # Enhanced resolution (more standard)
        self.window_name = "Face Mask Detection System"
        self.ui_panel_width = 320  # Side panel for analytics
        
        # Font configurations for professional look
        self.fonts = {
            'title': {'font': cv2.FONT_HERSHEY_TRIPLEX, 'scale': 1.0, 'thickness': 2},
            'header': {'font': cv2.FONT_HERSHEY_DUPLEX, 'scale': 0.7, 'thickness': 2},
            'body': {'font': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.6, 'thickness': 1},
            'small': {'font': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.5, 'thickness': 1},
            'mono': {'font': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.5, 'thickness': 1}
        }
        
        # Initialize the model
        self.load_model()
        
    def setup_face_detection(self):
        """Initialize face detection cascade."""
        try:
            # Try to load OpenCV's pre-trained face classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Check if cascade loaded properly
            if self.face_cascade.empty():
                print("⚠️  Warning: Could not load face cascade classifier")
                # Try alternative path in models directory
                alt_path = Path(__file__).parent.parent / "models" / "haarcascade_frontalface_default.xml"
                if alt_path.exists():
                    self.face_cascade = cv2.CascadeClassifier(str(alt_path))
                    print(f"✅ Loaded face cascade from: {alt_path}")
                else:
                    self.face_cascade = None
                    print("❌ Face detection will be disabled")
            else:
                print("✅ Face cascade classifier loaded successfully")
        except Exception as e:
            print(f"❌ Error setting up face detection: {e}")
            self.face_cascade = None
        
    def load_model(self) -> bool:
        """Load the trained model with error handling."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Please train the model first using the notebook.")
            return False
            
        try:
            self.predictor = FaceMaskPredictor(self.model_path)
            
            # Explicitly load the model (required for new predictor structure)
            self.predictor.load_model()
            
            if self.predictor.model is not None:
                print("✅ Model loaded successfully")
                return True
            else:
                print("❌ Failed to load model")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        if self.face_cascade is None:
            return []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(80, 80),
                maxSize=(400, 400)
            )
            return faces
        except Exception:
            return []
    
    def predict_face_mask_from_frame(self, face_region):
        """Predict mask status directly from face region."""
        try:
            # Convert BGR to RGB for the predictor
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Use frame-based prediction
            preprocessed = self.predictor.preprocess_frame(face_rgb)
            if preprocessed is None:
                return {'prediction': 'unknown', 'confidence': 0.0}
            
            # Get prediction
            predictions = self.predictor.model.predict(preprocessed, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.predictor.CLASSES[predicted_class_idx]
            
            return {
                'prediction': predicted_class,
                'confidence': confidence
            }
            
        except Exception:
            return {'prediction': 'unknown', 'confidence': 0.0}
    
    def save_high_confidence_detection(self, frame, face_coords, result):
        """Save high-confidence detections for analysis."""
        try:
            if result['confidence'] >= self.high_confidence_threshold:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Save full frame (not just face region)
                filename = f"{timestamp}_{prediction}_{confidence:.3f}.jpg"
                filepath = self.high_confidence_dir / filename
                cv2.imwrite(str(filepath), frame)
                
                print(f"💾 Saved high-confidence detection: {prediction} ({confidence:.1%}) - Full frame saved")
                
        except Exception as e:
            print(f"Error saving detection: {e}")
    
    def draw_analytics_panel(self, frame):
        """Draw advanced analytics panel on the right side."""
        try:
            height, width = frame.shape[:2]
            panel_x = width - self.ui_panel_width
            
            # Semi-transparent dark background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, 60), (width, height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            # Panel border with tech accent
            cv2.rectangle(frame, (panel_x, 60), (width, height), self.colors['neon_cyan'], 2)
            
            # Panel title
            font = self.fonts['header']
            cv2.putText(frame, "ANALYTICS", (panel_x + 15, 90), 
                       font['font'], font['scale'], self.colors['neon_cyan'], font['thickness'])
            
            # Session stats
            y_offset = 130
            line_height = 35
            
            # Session duration
            session_duration = datetime.now() - self.session_start
            duration_str = str(session_duration).split('.')[0]  # Remove microseconds
            cv2.putText(frame, "SESSION:", (panel_x + 15, y_offset), 
                       self.fonts['small']['font'], 0.5, (200, 200, 200), 1)
            cv2.putText(frame, duration_str, (panel_x + 15, y_offset + 20), 
                       self.fonts['small']['font'], 0.45, (255, 255, 255), 1)
            
            y_offset += line_height + 10
            
            # FPS counter
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (panel_x + 15, y_offset), 
                       self.fonts['body']['font'], 0.5, self.colors['tech_green'], 1)
            
            y_offset += line_height
            
            # Detection stats
            cv2.putText(frame, "DETECTIONS:", (panel_x + 15, y_offset), 
                       self.fonts['small']['font'], 0.5, (200, 200, 200), 1)
            y_offset += 25
            
            # Individual counters
            stats_items = [
                ("TOTAL", self.analytics['total_detections'], self.colors['ui_text']),
                ("WITH MASK", self.analytics['with_mask'], self.colors['with_mask']),
                ("NO MASK", self.analytics['without_mask'], self.colors['without_mask']),
                ("INCORRECT", self.analytics['mask_weared_incorrect'], self.colors['mask_weared_incorrect']),
                ("HIGH CONF.", self.analytics['high_confidence_detections'], self.colors['neon_cyan'])
            ]
            
            for label, count, color in stats_items:
                cv2.putText(frame, f"{label}: {count}", (panel_x + 15, y_offset), 
                           self.fonts['small']['font'], 0.45, color, 1)
                y_offset += 25
            
            # Average confidence
            y_offset += 15
            avg_conf = self.analytics.get('avg_confidence', 0.0)
            cv2.putText(frame, f"AVG CONF: {avg_conf:.1%}", (panel_x + 15, y_offset), 
                       self.fonts['small']['font'], 0.5, self.colors['ui_warning'], 1)
            
            # Confidence bar
            bar_width = self.ui_panel_width - 40
            bar_height = 8
            bar_x = panel_x + 15
            bar_y = y_offset + 15
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence fill
            fill_width = int(bar_width * avg_conf)
            if fill_width > 0:
                color = self.colors['tech_green'] if avg_conf > 0.7 else self.colors['ui_warning']
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            
            # Status indicators at bottom
            y_bottom = height - 50
            cv2.putText(frame, "STATUS: ACTIVE", (panel_x + 15, y_bottom), 
                       self.fonts['small']['font'], 0.5, self.colors['tech_green'], 1)
            
            # Animated status dot
            dot_x = panel_x + 160
            dot_y = y_bottom - 8
            pulse = int(time.time() * 3) % 2  # Pulse effect
            radius = 4 if pulse else 6
            cv2.circle(frame, (dot_x, dot_y), radius, self.colors['tech_green'], -1)
            
        except Exception as e:
            print(f"Error drawing analytics panel: {e}")
    
    def update_analytics(self, prediction, confidence):
        """Update analytics with new detection."""
        self.analytics['total_detections'] += 1
        
        if prediction in self.analytics:
            self.analytics[prediction] += 1
        
        if confidence >= self.high_confidence_threshold:
            self.analytics['high_confidence_detections'] += 1
        
        # Update average confidence
        history = self.analytics['detection_history']
        history.append(confidence)
        
        # Keep only last 100 detections for moving average
        if len(history) > 100:
            history.pop(0)
        
        self.analytics['avg_confidence'] = sum(history) / len(history) if history else 0.0
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:  # Update every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def draw_professional_title_bar(self, frame):
        """Draw enhanced professional title bar with system info."""
        try:
            height, width = frame.shape[:2]
            
            # Main title bar background with gradient effect
            cv2.rectangle(frame, (0, 0), (width, 60), (15, 15, 15), -1)
            
            # Gradient overlay effect
            for i in range(60):
                alpha = (60 - i) / 60.0 * 0.3
                color_val = int(15 + alpha * 25)
                cv2.line(frame, (0, i), (width, i), (color_val, color_val, color_val), 1)
            
            # Tech accent lines
            cv2.rectangle(frame, (0, 55), (width, 58), self.colors['neon_cyan'], 2)
            cv2.rectangle(frame, (0, 58), (width, 60), self.colors['tech_green'], 1)
            
            # Main title with icon
            title = " PROFESSIONAL FACE MASK DETECTION APP"
            font = self.fonts['title']
            text_size = cv2.getTextSize(title, font['font'], font['scale'], font['thickness'])[0]
            text_x = 20
            
            cv2.putText(frame, title, (text_x, 35), font['font'], font['scale'], 
                       self.colors['ui_text'], font['thickness'])
            
            # Version info
            version_text = "v2.0"
            cv2.putText(frame, version_text, (text_x + text_size[0] + 20, 35), 
                       self.fonts['small']['font'], 0.5, (150, 150, 150), 1)
            
            # Real-time status with animated indicator
            status_x = width - 200
            cv2.putText(frame, "LIVE DETECTION", (status_x, 25), 
                       self.fonts['body']['font'], 0.6, self.colors['tech_green'], 2)
            
            # Time display
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, current_time, (status_x, 45), 
                       self.fonts['small']['font'], 0.5, (200, 200, 200), 1)
            
            # Animated status indicators
            for i in range(3):
                x_pos = status_x - 40 + i * 8
                y_pos = 30
                pulse_phase = (time.time() * 2 + i * 0.5) % (2 * 3.14159)
                brightness = int(100 + 155 * abs(np.sin(pulse_phase)))
                cv2.circle(frame, (x_pos, y_pos), 2, (0, brightness, 0), -1)
            
        except Exception as e:
            print(f"Error drawing title bar: {e}")
    
    def run(self):
        """Run the enhanced Professional webcam application."""
        if self.predictor is None or self.predictor.model is None:
            print("❌ Model not available. Cannot start webcam application.")
            return
        
        # Initialize webcam with optimal settings
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            print("Please check that your webcam is connected and not being used by another application.")
            return
        
        # Enhanced webcam properties for better quality and color
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Ensure good brightness
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.6)    # Good contrast
        self.cap.set(cv2.CAP_PROP_SATURATION, 0.7)  # Ensure color saturation
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure for color
        
        # Create named window with specific size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        print("🎥 Starting Professional Face Mask Detection System...")
        print("� Enhanced Features:")
        print(f"   ✨ Professional UI with advanced analytics")
        print(f"   📊 Real-time statistics and confidence tracking")
        print(f"   🎯 {self.high_confidence_threshold:.0%} confidence threshold for saving detections")
        print(f"   💾 High-confidence detections saved to: {self.detection_dir}")
        print(f"   🎨 Full-color video with enhanced visual indicators")
        print("📋 Controls:")
        print("   - Press 'q' or ESC to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'r' to reset analytics")
        print()
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Ensure frame is in color (BGR format)
                if len(frame.shape) != 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Resize frame to fit window
                frame = cv2.resize(frame, (self.window_width, self.window_height))
                
                # Update FPS counter
                self.update_fps()
                
                # Draw enhanced UI components
                self.draw_professional_title_bar(frame)
                self.draw_analytics_panel(frame)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each detected face
                for face_coords in faces:
                    x, y, w, h = face_coords
                    
                    # Skip faces that are too close to the title bar or in analytics panel
                    if y < 70 or x > (self.window_width - self.ui_panel_width):
                        continue
                    
                    # Extract face region
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Predict mask status
                    result = self.predict_face_mask_from_frame(face_region)
                    prediction = result.get('prediction', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    
                    # Update analytics
                    self.update_analytics(prediction, confidence)
                    
                    # Save high-confidence detections
                    self.save_high_confidence_detection(frame, face_coords, result)
                    
                    # Get color for prediction
                    color = self.colors.get(prediction, (255, 255, 255))
                    
                    # Draw enhanced bounding box with tech styling
                    thickness = 3
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Draw tech-style corner accents
                    corner_length = 25
                    corner_thickness = 5
                    
                    # Top-left corner
                    cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
                    cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)
                    
                    # Top-right corner
                    cv2.line(frame, (x+w, y), (x+w - corner_length, y), color, corner_thickness)
                    cv2.line(frame, (x+w, y), (x+w, y + corner_length), color, corner_thickness)
                    
                    # Bottom-left corner
                    cv2.line(frame, (x, y+h), (x + corner_length, y+h), color, corner_thickness)
                    cv2.line(frame, (x, y+h), (x, y+h - corner_length), color, corner_thickness)
                    
                    # Bottom-right corner
                    cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), color, corner_thickness)
                    cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), color, corner_thickness)
                    
                    # Draw enhanced professional label
                    self.draw_enhanced_label(frame, x, y, w, h, prediction, confidence)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"F-M-D-{timestamp}.jpg"
                    filepath = self.detection_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"📸 Frame saved as: {filename}")
                elif key == ord('r'):  # Reset analytics
                    self.analytics = {
                        'total_detections': 0,
                        'with_mask': 0,
                        'without_mask': 0,
                        'mask_weared_incorrect': 0,
                        'high_confidence_detections': 0,
                        'avg_confidence': 0.0,
                        'detection_history': []
                    }
                    print("📊 Analytics reset")
                    
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        
        finally:
            # Clean up
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Professional webcam application closed")
            print(f"📊 Final session stats:")
            print(f"   Total detections: {self.analytics['total_detections']}")
            print(f"   High-confidence: {self.analytics['high_confidence_detections']}")
            print(f"   Average confidence: {self.analytics.get('avg_confidence', 0):.1%}")
            
    def draw_enhanced_label(self, frame, x, y, w, h, prediction, confidence):
        """Draw enhanced professional labels with tech styling."""
        try:
            # Get display name and color
            display_name = self.class_names.get(prediction, prediction.upper())
            color = self.colors.get(prediction, (255, 255, 255))
            
            # Create label text with confidence indicator
            confidence_text = f"{confidence:.1%}"
            
            # Determine confidence level
            conf_level = 'low'
            for level, data in self.confidence_levels.items():
                if confidence >= data['threshold']:
                    conf_level = level
                    break
            
            # Calculate text sizes
            font = self.fonts['header']
            name_size = cv2.getTextSize(display_name, font['font'], font['scale'], font['thickness'])[0]
            conf_size = cv2.getTextSize(confidence_text, font['font'], font['scale'] - 0.2, font['thickness'])[0]
            
            # Enhanced label dimensions
            label_height = 70
            label_width = max(name_size[0], conf_size[0]) + 30
            
            # Position for label (above face)
            label_x = x
            label_y = y - label_height - 5
            
            # Adjust if label goes outside frame
            if label_y < 65:  # Account for title bar
                label_y = y + h + 10
            
            # Draw modern label background with gradient effect
            overlay = frame.copy()
            cv2.rectangle(overlay, (label_x, label_y), 
                         (label_x + label_width, label_y + label_height), 
                         (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw tech border
            cv2.rectangle(frame, (label_x, label_y), 
                         (label_x + label_width, label_y + label_height), 
                         color, 2)
            
            # Add accent lines
            cv2.line(frame, (label_x, label_y + 25), (label_x + label_width, label_y + 25), 
                    color, 1)
            
            # Draw texts with enhanced styling
            cv2.putText(frame, display_name, 
                       (label_x + 15, label_y + 30), 
                       font['font'], font['scale'], color, font['thickness'])
            
            cv2.putText(frame, confidence_text, 
                       (label_x + 15, label_y + 55), 
                       font['font'], font['scale'] - 0.1, (255, 255, 255), font['thickness'])
            
            # High confidence indicator with animation
            if confidence >= self.high_confidence_threshold:
                pulse = int(time.time() * 4) % 2
                star_color = self.colors['neon_cyan'] if pulse else (255, 255, 255)
                cv2.circle(frame, (label_x + label_width - 20, label_y + 20), 10, star_color, -1)
                cv2.putText(frame, "★", (label_x + label_width - 26, label_y + 26), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                          
        except Exception as e:
            print(f"Error drawing enhanced label: {e}")

def main():
    """Main function to run the professional webcam application."""
    # Model path - try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "models" / "best_mask_detector_imbalance_optimized.h5",
        Path(__file__).parent.parent / "models" / "best_mask_detector.h5",
        Path(__file__).parent.parent / "models" / "face_mask_detector.h5"
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = str(path)
            break
    
    if model_path is None:
        print("❌ No trained model found. Please check the following locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nPlease train the model first using the Complete_MLOps_Setup_Guide.ipynb notebook.")
        return
    
    print("🎭 Professional Face Mask Detection System")
    print("=" * 55)
    print(f"📦 Using model: {Path(model_path).name}")
    print("🚀 Initializing professional application...")
    print()
    
    # Create and run the application
    app = ProfessionalWebcamDetector(model_path)
    app.run()

if __name__ == "__main__":
    main()
