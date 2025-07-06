"""
Inference Pipeline for Face Mask Detection
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from dataclasses import asdict

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .config import InferenceConfig, ProjectConfig

class InferencePipeline:
    """Complete inference pipeline with performance monitoring"""
    
    def __init__(self, config: InferenceConfig, project_config: ProjectConfig):
        self.config = config
        self.project_config = project_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.performance_metrics = {
            "total_predictions": 0,
            "average_confidence": 0.0,
            "inference_times": [],
            "class_distribution": {}
        }
        
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model for inference"""
        if YOLO is None:
            self.logger.error("❌ YOLO not available. Install ultralytics package.")
            return False
            
        model_path = model_path or self.config.model_path
        
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
                
            self.model = YOLO(model_path)
            self.logger.info(f"✅ Model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict_single_image(self, image_path: str, save_results: bool = True):
        """Predict on single image with comprehensive logging"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = datetime.datetime.now()
        
        try:
            # Run prediction
            results = self.model.predict(
                source=image_path,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.config.device,
                verbose=False
            )
            
            end_time = datetime.datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            # Process results
            processed_results = []
            total_confidence = 0
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "class_id": int(box.cls[0]) if hasattr(box, 'cls') else 0,
                            "class_name": self.model.names[int(box.cls[0])] if hasattr(box, 'cls') and hasattr(self.model, 'names') else "unknown",
                            "confidence": float(box.conf[0]) if hasattr(box, 'conf') else 0.0,
                            "bbox": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else [],
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        processed_results.append(detection)
                        total_confidence += detection["confidence"]
            
            # Update performance metrics
            self.performance_metrics["total_predictions"] += 1
            self.performance_metrics["inference_times"].append(inference_time)
            
            if processed_results:
                avg_conf = total_confidence / len(processed_results)
                self.performance_metrics["average_confidence"] = (
                    (self.performance_metrics["average_confidence"] * (self.performance_metrics["total_predictions"] - 1) + avg_conf) /
                    self.performance_metrics["total_predictions"]
                )
                
                # Update class distribution
                for detection in processed_results:
                    class_name = detection["class_name"]
                    self.performance_metrics["class_distribution"][class_name] = (
                        self.performance_metrics["class_distribution"].get(class_name, 0) + 1
                    )
            
            # Save results if requested
            if save_results and processed_results and len(results) > 0:
                self._save_detection_results(image_path, processed_results, results[0])
            
            return {
                "success": True,
                "predictions": processed_results,
                "inference_time": inference_time,
                "image_path": image_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed for {image_path}: {e}")
            return {"success": False, "error": str(e), "image_path": image_path}
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32):
        """Process batch of images efficiently"""
        self.logger.info(f"🔄 Processing batch of {len(image_paths)} images...")
        
        results = []
        failed_predictions = []
        
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch processing"):
            batch = image_paths[i:i + batch_size]
            
            for image_path in batch:
                result = self.predict_single_image(image_path, save_results=False)
                
                if result["success"]:
                    results.append(result)
                else:
                    failed_predictions.append(result)
        
        self.logger.info(f"✅ Batch processing complete: {len(results)} successful, {len(failed_predictions)} failed")
        
        return {
            "successful_predictions": results,
            "failed_predictions": failed_predictions,
            "total_processed": len(image_paths),
            "success_rate": len(results) / len(image_paths) if image_paths else 0
        }
    
    def _save_detection_results(self, image_path: str, detections: List[Dict], yolo_result):
        """Save detection results and annotated images"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = Path(image_path).stem
        
        # Save detection data
        for i, detection in enumerate(detections):
            detection_filename = f"{timestamp}_{base_name}_{detection['class_name']}_{detection['confidence']:.3f}"
            
            # Save JSON metadata
            json_path = Path("../detections") / f"{detection_filename}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(detection, f, indent=2)
            
            # Save annotated image if possible
            try:
                if hasattr(yolo_result, 'plot'):
                    annotated_img = yolo_result.plot()
                    img_path = Path("../detections") / f"{detection_filename}.jpg"
                    cv2.imwrite(str(img_path), annotated_img)
            except Exception as e:
                self.logger.warning(f"Could not save annotated image: {e}")
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.performance_metrics["inference_times"]:
            return {"error": "No predictions made yet"}
        
        inference_times = self.performance_metrics["inference_times"]
        
        summary = {
            "total_predictions": self.performance_metrics["total_predictions"],
            "average_confidence": round(self.performance_metrics["average_confidence"], 3),
            "average_inference_time": round(np.mean(inference_times), 4),
            "min_inference_time": round(min(inference_times), 4),
            "max_inference_time": round(max(inference_times), 4),
            "class_distribution": self.performance_metrics["class_distribution"],
            "predictions_per_second": round(1 / np.mean(inference_times), 2) if inference_times else 0
        }
        
        return summary
    
    def create_performance_report(self):
        """Generate detailed performance report"""
        summary = self.get_performance_summary()
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_path": self.config.model_path,
            "configuration": asdict(self.config),
            "performance_summary": summary,
            "detailed_metrics": self.performance_metrics
        }
        
        # Save report
        report_path = Path(self.project_config.reports_dir) / "inference" / f"inference_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Performance report saved: {report_path}")
        return report_path
