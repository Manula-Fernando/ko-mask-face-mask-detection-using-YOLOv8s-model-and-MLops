# Face Mask Detection - Advanced Real-time Webcam Application
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
import sys
import os
import json
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from predict import FaceMaskPredictor
except ImportError:
    print("Warning: Could not import FaceMaskPredictor. Please ensure src/predict.py exists.")
    sys.exit(1)

class AdvancedWebcamApp:
    """Advanced real-time face mask detection with comprehensive features."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = None
        self.cap = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Face cascade for face detection
        cascade_path = Path(__file__).parent.parent / "models" / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        else:
            # Use default cascade if available
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                self.logger.warning("Could not load face cascade classifier")
                self.face_cascade = None
        
        # Colors for different predictions (BGR format)
        self.colors = {
            'with_mask': (0, 255, 0),        # Green
            'without_mask': (0, 0, 255),     # Red
            'mask_weared_incorrect': (0, 165, 255)  # Orange
        }
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'predictions': defaultdict(int),
            'high_confidence_detections': 0,
            'session_start': datetime.now()
        }
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.prediction_times = deque(maxlen=100)
        
        # High-confidence detection saving
        self.high_confidence_dir = Path(__file__).parent.parent / "high_confidence_detections"
        self.high_confidence_dir.mkdir(exist_ok=True)
        self.high_confidence_threshold = 0.95
        
        # UI settings
        self.show_statistics = True
        self.show_confidence_bars = True
        self.ui_scale = 1.0
        
        # Initialize predictor
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the trained model with comprehensive error handling."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found: {self.model_path}")
            print(f"❌ Model not found: {self.model_path}")
            print("Please train the model first using the notebook.")
            return False
            
        try:
            self.predictor = FaceMaskPredictor(self.model_path)
            if self.predictor.model is not None:
                self.logger.info("Model loaded successfully")
                print("✅ Model loaded successfully")
                return True
            else:
                self.logger.error("Failed to load model")
                print("❌ Failed to load model")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame with enhanced error handling."""
        if self.face_cascade is None:
            return []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60),
                maxSize=(300, 300)
            )
            return faces
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []
    
    def predict_face_mask_from_frame(self, face_region: np.ndarray) -> Dict:
        """Predict mask status directly from face region (faster than file-based)."""
        try:
            # Convert BGR to RGB for the predictor
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Use frame-based prediction (faster)
            preprocessed = self.predictor.preprocess_frame(face_rgb)
            if preprocessed is None:
                return {'prediction': 'unknown', 'confidence': 0.0, 'all_probabilities': {}}
            
            # Time the prediction
            start_time = time.time()
            predictions = self.predictor.model.predict(preprocessed, verbose=0)
            prediction_time = time.time() - start_time
            
            self.prediction_times.append(prediction_time)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.predictor.CLASSES[predicted_class_idx]
            
            # All probabilities
            all_probs = {
                class_name: float(prob) for class_name, prob in 
                zip(self.predictor.CLASSES, predictions[0])
            }
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'prediction_time': prediction_time
            }
            
        except Exception as e:
            self.logger.error(f"Frame prediction error: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0, 'all_probabilities': {}}
    
    def save_high_confidence_detection(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int], 
                                     result: Dict) -> None:
        """Save high-confidence detections for analysis."""
        try:
            if result['confidence'] >= self.high_confidence_threshold:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Extract face region
                x, y, w, h = face_coords
                face_region = frame[y:y+h, x:x+w]
                
                # Save face image
                filename = f"{timestamp}_{prediction}_{confidence:.3f}.jpg"
                filepath = self.high_confidence_dir / filename
                cv2.imwrite(str(filepath), face_region)
                
                # Save metadata
                metadata = {
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': confidence,
                    'all_probabilities': result['all_probabilities'],
                    'face_coordinates': face_coords,
                    'filename': filename
                }
                
                metadata_file = self.high_confidence_dir / f"{timestamp}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.stats['high_confidence_detections'] += 1
                self.logger.info(f"Saved high-confidence detection: {filename}")
                
        except Exception as e:
            self.logger.error(f"Error saving high-confidence detection: {e}")
    
    def draw_confidence_bar(self, frame: np.ndarray, x: int, y: int, w: int, 
                          result: Dict) -> None:
        """Draw confidence bars for all classes."""
        if not self.show_confidence_bars:
            return
            
        try:
            bar_height = 20
            bar_width = w
            start_y = y + 50
            
            for i, (class_name, prob) in enumerate(result['all_probabilities'].items()):
                bar_y = start_y + i * (bar_height + 5)
                
                # Background bar
                cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), 
                            (50, 50, 50), -1)
                
                # Confidence bar
                filled_width = int(bar_width * prob)
                color = self.colors.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x, bar_y), (x + filled_width, bar_y + bar_height), 
                            color, -1)
                
                # Text label
                label = f"{class_name.replace('_', ' ')}: {prob*100:.1f}%"
                cv2.putText(frame, label, (x, bar_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                          
        except Exception as e:
            self.logger.error(f"Error drawing confidence bars: {e}")
    
    def draw_statistics(self, frame: np.ndarray) -> None:
        """Draw comprehensive statistics on the frame."""
        if not self.show_statistics:
            return
            
        try:
            # Calculate session duration
            session_duration = datetime.now() - self.stats['session_start']
            duration_str = str(session_duration).split('.')[0]  # Remove microseconds
            
            # Calculate average FPS
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            
            # Calculate average prediction time
            avg_pred_time = np.mean(self.prediction_times) if self.prediction_times else 0
            
            # Statistics text
            stats_text = [
                f"Session: {duration_str}",
                f"Frames: {self.stats['total_frames']:,}",
                f"Faces: {self.stats['faces_detected']:,}",
                f"Avg FPS: {avg_fps:.1f}",
                f"Pred Time: {avg_pred_time*1000:.1f}ms",
                f"High Conf: {self.stats['high_confidence_detections']:,}",
                "",
                "Predictions:",
            ]
            
            # Add prediction counts
            for pred_class, count in self.stats['predictions'].items():
                stats_text.append(f"  {pred_class.replace('_', ' ')}: {count:,}")
            
            # Draw background
            text_height = len(stats_text) * 25
            cv2.rectangle(frame, (10, 10), (300, text_height + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, text_height + 20), (255, 255, 255), 2)
            
            # Draw text
            for i, text in enumerate(stats_text):
                y_pos = 35 + i * 25
                color = (0, 255, 255) if text.startswith("  ") else (255, 255, 255)
                cv2.putText(frame, text, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                          
        except Exception as e:
            self.logger.error(f"Error drawing statistics: {e}")
    
    def update_statistics(self, faces_count: int, predictions: List[str]) -> None:
        """Update session statistics."""
        self.stats['total_frames'] += 1
        self.stats['faces_detected'] += faces_count
        
        for prediction in predictions:
            self.stats['predictions'][prediction] += 1
    
    def run(self) -> None:
        """Run the advanced webcam application with comprehensive features."""
        if self.predictor is None or self.predictor.model is None:
            self.logger.error("Model not available. Cannot start webcam application.")
            print("❌ Model not available. Cannot start webcam application.")
            return
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.logger.error("Could not open webcam")
            print("❌ Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Starting Advanced Face Mask Detection System...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to toggle statistics")
        print("   - Press 'c' to toggle confidence bars")
        print("   - Press 'r' to reset statistics")
        print("   - Press ESC to exit")
        print("🎯 Features:")
        print(f"   - High-confidence detection saving (>{self.high_confidence_threshold*100:.1f}%)")
        print(f"   - Saves to: {self.high_confidence_dir}")
        print("   - Real-time performance monitoring")
        print("   - Comprehensive statistics tracking")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Could not read frame from webcam")
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                frame_predictions = []
                
                # Process each detected face
                for face_coords in faces:
                    x, y, w, h = face_coords
                    
                    # Extract face region
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Predict mask status using frame-based prediction
                    result = self.predict_face_mask_from_frame(face_region)
                    
                    prediction = result.get('prediction', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    frame_predictions.append(prediction)
                    
                    # Save high-confidence detections
                    self.save_high_confidence_detection(frame, face_coords, result)
                    
                    # Get color for prediction
                    color = self.colors.get(prediction, (255, 255, 255))  # White for unknown
                    
                    # Draw bounding box with thickness based on confidence
                    thickness = max(2, int(confidence * 5))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Prepare main label text
                    label = f"{prediction.replace('_', ' ')}: {confidence*100:.1f}%"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x + 5, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw confidence bars
                    self.draw_confidence_bar(frame, x, y, w, result)
                    
                    # High confidence indicator
                    if confidence >= self.high_confidence_threshold:
                        cv2.circle(frame, (x + w - 20, y + 20), 10, (0, 255, 255), -1)
                        cv2.putText(frame, "HIGH", (x + w - 50, y + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Update statistics
                self.update_statistics(len(faces), frame_predictions)
                
                # Calculate and update FPS
                fps_counter += 1
                if fps_counter >= 10:  # Update FPS every 10 frames
                    fps_end_time = time.time()
                    current_fps = fps_counter / (fps_end_time - fps_start_time)
                    self.fps_history.append(current_fps)
                    fps_counter = 0
                    fps_start_time = fps_end_time
                
                # Draw current FPS
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                          (frame.shape[1] - 150, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw face count
                cv2.putText(frame, f"Faces: {len(faces)}", 
                          (frame.shape[1] - 150, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw statistics panel
                self.draw_statistics(frame)
                
                # Draw instructions at bottom
                instructions = [
                    "Press 'q' to quit | 's' for stats | 'c' for confidence bars | 'r' to reset"
                ]
                for i, instruction in enumerate(instructions):
                    cv2.putText(frame, instruction, (10, frame.shape[0] - 20 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Advanced Face Mask Detection System', frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # Toggle statistics
                    self.show_statistics = not self.show_statistics
                    print(f"Statistics display: {'ON' if self.show_statistics else 'OFF'}")
                elif key == ord('c'):  # Toggle confidence bars
                    self.show_confidence_bars = not self.show_confidence_bars
                    print(f"Confidence bars: {'ON' if self.show_confidence_bars else 'OFF'}")
                elif key == ord('r'):  # Reset statistics
                    self.reset_statistics()
                    print("📊 Statistics reset")
                    
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        
        finally:
            # Print final statistics
            self.print_final_statistics()
            
            # Clean up
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Advanced webcam application closed")
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'predictions': defaultdict(int),
            'high_confidence_detections': 0,
            'session_start': datetime.now()
        }
        self.fps_history.clear()
        self.prediction_times.clear()
    
    def print_final_statistics(self) -> None:
        """Print comprehensive final statistics."""
        print("\n" + "="*60)
        print("📊 FINAL SESSION STATISTICS")
        print("="*60)
        
        session_duration = datetime.now() - self.stats['session_start']
        duration_str = str(session_duration).split('.')[0]
        
        print(f"Session Duration: {duration_str}")
        print(f"Total Frames Processed: {self.stats['total_frames']:,}")
        print(f"Total Faces Detected: {self.stats['faces_detected']:,}")
        print(f"High-Confidence Detections: {self.stats['high_confidence_detections']:,}")
        
        if self.fps_history:
            print(f"Average FPS: {np.mean(self.fps_history):.2f}")
            print(f"Max FPS: {np.max(self.fps_history):.2f}")
            print(f"Min FPS: {np.min(self.fps_history):.2f}")
        
        if self.prediction_times:
            avg_pred_time = np.mean(self.prediction_times) * 1000
            print(f"Average Prediction Time: {avg_pred_time:.2f}ms")
        
        print("\nPrediction Distribution:")
        total_predictions = sum(self.stats['predictions'].values())
        for pred_class, count in self.stats['predictions'].items():
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            print(f"  {pred_class.replace('_', ' ')}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nHigh-confidence detections saved to: {self.high_confidence_dir}")
        print("="*60)

def main():
    """Main function to run the advanced webcam application."""
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
    
    print("🎭 Advanced Face Mask Detection System")
    print("=" * 50)
    print(f"📦 Using model: {Path(model_path).name}")
    print("🚀 Initializing application...")
    
    # Create and run the application
    app = AdvancedWebcamApp(model_path)
    app.run()

if __name__ == "__main__":
    main()
