#!/usr/bin/env python3
"""
Working Medical Face Mask Detection - Real-time Webcam Application
Based on the working app structure with proper imports and error handling
"""

import cv2
import numpy as np
from pathlib import Path
import time
import sys
import os
from datetime import datetime
import logging
import json
from typing import Optional, List, Dict, Tuple
import mlflow

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.inference.predictor import FaceMaskPredictor
    print("✅ Medical predictor module loaded successfully")
except ImportError as e:
    print(f"❌ Error: Could not import FaceMaskPredictor: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalWebcamDetector:
    """Medical-grade real-time face mask detection with professional UI."""
    
    def __init__(self):
        """Initialize the medical webcam detector."""
        self.predictor = None
        self.cap = None
        
        # UI Configuration  
        self.window_name = "🏥 Medical Face Mask Compliance Monitor"
        self.window_width = 1310
        self.window_height = 1010
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        # Detection analytics
        self.analytics = {
            'total_detections': 0,
            'with_mask': 0,
            'without_mask': 0,
            'mask_weared_incorrect': 0,
            'session_start': time.time(),
            'detection_history': []
        }
        
        # Medical color scheme (BGR format for OpenCV)
        self.colors = {
            'with_mask': (0, 255, 0),          # Green
            'without_mask': (0, 0, 255),       # Red
            'mask_weared_incorrect': (0, 165, 255),  # Orange
            'background': (20, 20, 20),        # Dark
            'text': (255, 255, 255),           # White
            'panel': (40, 40, 40),             # Panel
            'accent': (0, 255, 255),           # Cyan
            'border': (0, 255, 255),           # Border
            'success': (0, 255, 0),            # Green
            'error': (0, 0, 255),              # Red
            'warning': (0, 165, 255)           # Orange
        }
        
        # Medical status mapping
        self.medical_status = {
            'with_mask': '✅ COMPLIANT',
            'without_mask': '❌ NON-COMPLIANT',
            'mask_weared_incorrect': '⚠️ IMPROPER USAGE'
        }
        
        # Font configurations
        self.fonts = {
            'title': cv2.FONT_HERSHEY_SIMPLEX,
            'label': cv2.FONT_HERSHEY_SIMPLEX,
            'stats': cv2.FONT_HERSHEY_SIMPLEX
        }
        
        # Detection storage directories for YOLO format
        self.images_dir = Path("data/collected/webcam_detections/images")
        self.labels_dir = Path("data/collected/webcam_detections/labels")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.detection_dir = self.images_dir  # For backward compatibility
        
        # Initialize predictor
        self.init_predictor()
    
    def detect_faces_and_masks(self, frame) -> List[Dict]:
        """Detect faces and masks in frame using the real predictor."""
        detections = []
        
        try:
            if self.predictor:
                result = self.predictor.predict(frame)
                if result and 'prediction' in result and result['prediction'] != 'no_detection':
                    # Convert predictor result to detection format
                    detection = {
                        'class': result['prediction'],
                        'confidence': result.get('confidence', 0),
                        'bbox': result.get('bbox', [0, 0, 100, 100])
                    }
                    detections.append(detection)
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def resize_and_square(self, img, size=620):
        """Resize and pad/crop image to a square of given size."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        # Create a black square
        squared = np.zeros((size, size, 3), dtype=img.dtype)
        # Center the resized image
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        squared[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return squared

    def save_detection(self, frame, detection: Dict):
        """Save high-confidence detections as 620x620 squared images and YOLO label files, without boundary box and no JSON. Logs each detection as an MLflow run."""
        try:
            if detection['confidence'] >= 0.8:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                base_filename = f"{timestamp}_{detection['class']}_{detection['confidence']:.3f}"
                img_filename = f"{base_filename}.jpg"
                label_filename = f"{base_filename}.txt"
                img_path = self.images_dir / img_filename
                label_path = self.labels_dir / label_filename
                # Save a clean copy of the frame (no box)
                squared_frame = self.resize_and_square(frame.copy(), 620)
                cv2.imwrite(str(img_path), squared_frame)
                # Save YOLO label (class_id x_center y_center width height, normalized)
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                class_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
                class_id = class_map.get(detection['class'], 0)
                # Convert bbox to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / 620
                y_center = ((y1 + y2) / 2) / 620
                width = (x2 - x1) / 620
                height = (y2 - y1) / 620
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                # MLflow logging for this detection
                with mlflow.start_run(run_name=f"webcam_{base_filename}"):
                    mlflow.log_param("timestamp", timestamp)
                    mlflow.log_param("class", detection['class'])
                    mlflow.log_metric("confidence", detection['confidence'])
                    mlflow.log_metric("x_center", x_center)
                    mlflow.log_metric("y_center", y_center)
                    mlflow.log_metric("width", width)
                    mlflow.log_metric("height", height)
                    mlflow.log_artifact(str(img_path))
                    mlflow.log_artifact(str(label_path))
        except Exception as e:
            logger.error(f"Save detection error: {e}")
        
    def init_predictor(self):
        """Initialize the face mask predictor."""
        try:
            print("🔬 Initializing Medical Face Mask Predictor...")
            self.predictor = FaceMaskPredictor()
            print("✅ Medical AI System Ready")
        except Exception as e:
            print(f"❌ Failed to initialize predictor: {e}")
            self.predictor = None
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        self.fps_counter += 1
        if time.time() - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = time.time()
    
    def update_analytics(self, detection_class: str, confidence: float):
        """Update detection analytics with proper class mapping."""
        self.analytics['total_detections'] += 1
        
        # Update class counts
        if detection_class == 'with_mask':
            self.analytics['with_mask'] += 1
        elif detection_class == 'without_mask':
            self.analytics['without_mask'] += 1
        elif detection_class == 'mask_weared_incorrect':
            self.analytics['mask_weared_incorrect'] += 1
        
        # Add to history (keep last 100)
        self.analytics['detection_history'].append({
            'timestamp': time.time(),
            'class': detection_class,
            'confidence': confidence
        })
        
        if len(self.analytics['detection_history']) > 100:
            self.analytics['detection_history'] = self.analytics['detection_history'][-100:]
    
    def draw_ui_panels(self, frame: np.ndarray):
        """Draw comprehensive medical UI panels matching app directory style."""
        try:
            h, w = frame.shape[:2]
            
            # Header panel
            header_height = 80
            cv2.rectangle(frame, (0, 0), (w, header_height), self.colors['panel'], -1)
            cv2.rectangle(frame, (0, 0), (w, header_height), self.colors['accent'], 2)
            
            # Medical title
            title_text = "🏥 MEDICAL FACE MASK COMPLIANCE MONITOR"
            cv2.putText(frame, title_text, (20, 30), 
                       self.fonts['title'], 1.2, self.colors['accent'], 3)
            
            # System status
            status_text = f"FPS: {self.current_fps} | Frame: {self.frame_count} | STATUS: ACTIVE"
            cv2.putText(frame, status_text, (20, 60), 
                       self.fonts['label'], 0.6, self.colors['text'], 2)
            
            # Real-time scanner line effect
            scanner_y = int((time.time() * 100) % h)
            cv2.line(frame, (0, scanner_y), (w, scanner_y), 
                    self.colors['accent'], 2)
            
            # Side status panel
            panel_width = 280
            cv2.rectangle(frame, (w - panel_width, header_height), (w, h - 120), 
                         self.colors['panel'], -1)
            cv2.rectangle(frame, (w - panel_width, header_height), (w, h - 120), 
                         self.colors['accent'], 2)
            
            # Detection status
            y_pos = header_height + 30
            cv2.putText(frame, "DETECTION STATUS", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.7, self.colors['accent'], 2)
            
            # Analytics display
            total = self.analytics['total_detections']
            compliant = self.analytics.get('with_mask', 0)
            non_compliant = self.analytics.get('without_mask', 0)
            improper = self.analytics.get('mask_weared_incorrect', 0)
            
            compliance_rate = (compliant / total * 100) if total > 0 else 0
            
            y_pos += 40
            cv2.putText(frame, f"Total Scans: {total}", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.5, self.colors['text'], 1)
            y_pos += 25
            cv2.putText(frame, f"Compliant: {compliant}", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.5, self.colors['success'], 1)
            y_pos += 25
            cv2.putText(frame, f"Non-Compliant: {non_compliant}", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.5, self.colors['error'], 1)
            y_pos += 25
            cv2.putText(frame, f"Improper: {improper}", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.5, self.colors['warning'], 1)
            y_pos += 35
            cv2.putText(frame, f"Compliance: {compliance_rate:.1f}%", (w - panel_width + 10, y_pos), 
                       self.fonts['label'], 0.6, self.colors['accent'], 2)
            
            # Bottom analytics panel
            panel_height = 120
            cv2.rectangle(frame, (0, h - panel_height), (w, h), self.colors['panel'], -1)
            cv2.rectangle(frame, (0, h - panel_height), (w, h), 
                         self.colors['success'], 2)
            
            # Analytics text
            y_start = h - 100
            cv2.putText(frame, "DETECTION STATISTICS", (20, y_start), 
                       self.fonts['stats'], 0.6, self.colors['accent'], 2)
            cv2.putText(frame, f"Total Detections: {total}", (20, y_start + 25), 
                       self.fonts['stats'], 0.5, self.colors['text'], 1)
            cv2.putText(frame, f"Compliance Rate: {compliance_rate:.1f}%", (20, y_start + 45), 
                       self.fonts['stats'], 0.5, self.colors['success'], 1)
            
            # Column 2: Classification Breakdown
            col2_x = 300
            cv2.putText(frame, "CLASSIFICATION BREAKDOWN", (col2_x, y_start), 
                       self.fonts['stats'], 0.6, self.colors['accent'], 2)
            cv2.putText(frame, f"Compliant: {compliant}", (col2_x, y_start + 25), 
                       self.fonts['stats'], 0.5, self.colors['success'], 1)
            cv2.putText(frame, f"Non-Compliant: {non_compliant}", (col2_x, y_start + 45), 
                       self.fonts['stats'], 0.5, self.colors['error'], 1)
            cv2.putText(frame, f"Improper: {improper}", (col2_x, y_start + 65), 
                       self.fonts['stats'], 0.5, self.colors['warning'], 1)
            
            # Column 3: Session Info
            col3_x = 600
            session_duration = time.time() - self.analytics['session_start']
            duration_str = f"{int(session_duration//3600):02d}:{int((session_duration%3600)//60):02d}:{int(session_duration%60):02d}"
            
            cv2.putText(frame, "SESSION INFORMATION", (col3_x, y_start), 
                       self.fonts['stats'], 0.6, self.colors['accent'], 2)
            cv2.putText(frame, f"Duration: {duration_str}", (col3_x, y_start + 25), 
                       self.fonts['stats'], 0.5, self.colors['text'], 1)
            cv2.putText(frame, f"Frame: {self.frame_count}", (col3_x, y_start + 45), 
                       self.fonts['stats'], 0.5, self.colors['text'], 1)
            
            # Controls info
            controls_x = w - 300
            cv2.putText(frame, "CONTROLS", (controls_x, y_start), 
                       self.fonts['stats'], 0.6, self.colors['accent'], 2)
            cv2.putText(frame, "Q/ESC: Quit", (controls_x, y_start + 25), 
                       self.fonts['stats'], 0.4, self.colors['text'], 1)
            cv2.putText(frame, "S: Save Frame", (controls_x, y_start + 40), 
                       self.fonts['stats'], 0.4, self.colors['text'], 1)
            cv2.putText(frame, "R: Reset Stats", (controls_x, y_start + 55), 
                       self.fonts['stats'], 0.4, self.colors['text'], 1)
            
        except Exception as e:
            print(f"❌ Error drawing UI panels: {e}")
    
    def draw_detection_box(self, frame, detection: Dict):
        """Draw detection box and label based on app directory style."""
        try:
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class']
                confidence = detection.get('confidence', 0)
                
                # Choose color based on prediction
                color = self.colors.get(class_name, self.colors['text'])
                
                # Draw bounding box with thicker lines for medical appearance
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Medical status label
                medical_label = self.medical_status.get(class_name, class_name.upper())
                label = f"{medical_label} ({confidence:.1%})"
                
                # Calculate label size
                label_size = cv2.getTextSize(label, self.fonts['label'], 0.7, 2)[0]
                
                # Draw label background with padding
                label_bg_y1 = int(y1) - label_size[1] - 15
                label_bg_y2 = int(y1) - 5
                label_bg_x1 = int(x1)
                label_bg_x2 = int(x1) + label_size[0] + 10
                
                cv2.rectangle(frame, (label_bg_x1, label_bg_y1), 
                             (label_bg_x2, label_bg_y2), color, -1)
                
                # Draw white border around label
                cv2.rectangle(frame, (label_bg_x1, label_bg_y1), 
                             (label_bg_x2, label_bg_y2), self.colors['text'], 1)
                
                # Draw label text
                cv2.putText(frame, label, (int(x1) + 5, int(y1) - 10), 
                           self.fonts['label'], 0.7, self.colors['text'], 2)
                
                # Draw confidence indicator
                conf_bar_width = int((x2 - x1) * confidence)
                cv2.rectangle(frame, (int(x1), int(y2) + 5), 
                             (int(x1) + conf_bar_width, int(y2) + 15), color, -1)
        
        except Exception as e:
            print(f"❌ Error drawing detection box: {e}")
    
    def reset_analytics(self):
        """Reset detection analytics."""
        self.analytics = {
            'total_detections': 0,
            'with_mask': 0,
            'without_mask': 0,
            'mask_weared_incorrect': 0,
            'session_start': time.time(),
            'detection_history': []
        }
        print("📊 Analytics reset")
    
    def run(self):
        """Run the medical webcam detector - main application entry point."""
        if self.predictor is None or self.predictor.model is None:
            print("❌ Cannot start - medical model not loaded")
            return
        
        # Aggressively destroy any existing OpenCV windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process window destruction
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)  # Try webcam 0 first
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)  # Try webcam 1
            if not self.cap.isOpened():
                print("❌ Cannot access webcam")
                return
        
        # Set webcam properties for higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create single window and set size immediately
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        # Move window to center of screen
        cv2.moveWindow(self.window_name, 100, 50)
        
        print("🎥 Starting Medical Face Mask Detection System...")
        print("📋 Controls:")
        print("   - Press 'q' or ESC to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'r' to reset analytics")
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Cannot read from webcam")
                    break
                
                # Flip frame horizontally to prevent mirror effect (so it's not inverted)
                frame = cv2.flip(frame, 1)
                
                # Resize frame to match window size for better display
                frame = cv2.resize(frame, (self.window_width, self.window_height))
                
                self.frame_count += 1
                self.update_fps()
                
                # Detect faces and masks
                detections = self.detect_faces_and_masks(frame)

                # Make a clean copy before any drawing
                clean_frame = frame.copy()

                # Process each detection
                for detection in detections:
                    # Update analytics
                    self.update_analytics(detection['class'], detection['confidence'])

                    # Save high-confidence detections (use clean frame)
                    if detection['confidence'] >= 0.8:
                        self.save_detection(clean_frame, detection)

                    # Draw detection box
                    self.draw_detection_box(frame, detection)
                
                # Draw UI panels
                self.draw_ui_panels(frame)
                
                # Display frame in the single window only
                cv2.imshow(self.window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"manual_save_{timestamp}"
                    img_filename = f"{base_filename}.jpg"
                    label_filename = f"{base_filename}.txt"
                    squared_frame = self.resize_and_square(frame.copy(), 620)
                    img_path = self.images_dir / img_filename
                    label_path = self.labels_dir / label_filename
                    cv2.imwrite(str(img_path), squared_frame)
                    # Save empty label file for manual saves (no detection)
                    with open(label_path, 'w') as f:
                        f.write("")
                    print(f"� Frame saved: {img_filename}")
                elif key == ord('r'):  # Reset analytics
                    self.reset_analytics()
                    
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            print("\n✅ Medical webcam application closed")
            print(f"📊 Final session statistics:")
            print(f"   Total detections: {self.analytics['total_detections']}")
            print(f"   Compliant: {self.analytics.get('with_mask', 0)}")
            print(f"   Non-compliant: {self.analytics.get('without_mask', 0)}")
            print(f"   Improper usage: {self.analytics.get('mask_weared_incorrect', 0)}")
            
            if self.analytics['total_detections'] > 0:
                compliance_rate = (self.analytics.get('with_mask', 0) / 
                                 self.analytics['total_detections'] * 100)
                print(f"   Compliance rate: {compliance_rate:.1f}%")

def main():
    """Main function to run the medical webcam detector."""
    # Aggressive cleanup to prevent multiple windows
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    print("🔬 Medical Face Mask Compliance Monitor")
    print("=" * 55)
    print("🏥 Medical AI Model: YOLO Medical Detection")
    print("🚀 Initializing medical compliance system...")
    print()
    
    # Create and run the application
    app = MedicalWebcamDetector()
    app.run()

if __name__ == "__main__":
    main()
