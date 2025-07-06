"""
Common utilities for Face Mask Detection MLOps Pipeline
"""

import os
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import hashlib
import shutil
from PIL import Image
import base64
import io

from .logger import get_logger

logger = get_logger("utils")

class FileUtils:
    """File operations utilities"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if not"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """Get MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def copy_with_backup(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy file with backup if destination exists"""
        try:
            src, dst = Path(src), Path(dst)
            if dst.exists():
                backup_path = dst.with_suffix(f"{dst.suffix}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copy2(dst, backup_path)
                logger.info(f"Backup created: {backup_path}")
            
            shutil.copy2(src, dst)
            logger.info(f"File copied: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load image and optionally resize"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if target_size:
                image = cv2.resize(image, target_size)
            
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: Union[str, Path]) -> bool:
        """Save image to path"""
        try:
            FileUtils.ensure_dir(Path(save_path).parent)
            success = cv2.imwrite(str(save_path), image)
            if success:
                logger.info(f"Image saved: {save_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to save image {save_path}: {e}")
            return False
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        """Encode image to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            raise
    
    @staticmethod
    def decode_base64_to_image(base64_string: str) -> np.ndarray:
        """Decode base64 string to image"""
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 to image: {e}")
            raise

class ConfigUtils:
    """Configuration management utilities"""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
        """Save configuration to YAML file"""
        try:
            FileUtils.ensure_dir(Path(config_path).parent)
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config {config_path}: {e}")
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations with override taking precedence"""
        merged = base_config.copy()
        merged.update(override_config)
        return merged

class MetricsUtils:
    """Metrics calculation utilities"""
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        try:
            # Convert to [x1, y1, x2, y2] format
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            inter_x1 = max(x1_min, x2_min)
            inter_y1 = max(y1_min, y2_min)
            inter_x2 = min(x1_max, x2_max)
            inter_y2 = min(y1_max, y2_max)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            
            # Calculate union
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Failed to calculate IoU: {e}")
            return 0.0
    
    @staticmethod
    def calculate_map(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
        """Calculate mean Average Precision (mAP)"""
        # Simplified mAP calculation for demonstration
        # In production, use more sophisticated implementation
        try:
            if not predictions or not ground_truths:
                return 0.0
            
            total_precision = 0.0
            total_count = 0
            
            for pred in predictions:
                best_iou = 0.0
                for gt in ground_truths:
                    if pred.get('class') == gt.get('class'):
                        iou = MetricsUtils.calculate_iou(pred['bbox'], gt['bbox'])
                        best_iou = max(best_iou, iou)
                
                if best_iou >= iou_threshold:
                    total_precision += pred.get('confidence', 0.0)
                total_count += 1
            
            return total_precision / total_count if total_count > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Failed to calculate mAP: {e}")
            return 0.0

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """Normalize bounding box coordinates to [0, 1] range"""
        try:
            x1, y1, x2, y2 = bbox
            return [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height
            ]
        except Exception as e:
            logger.error(f"Failed to normalize bbox: {e}")
            return bbox
    
    @staticmethod
    def denormalize_bbox(normalized_bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """Denormalize bounding box coordinates from [0, 1] range"""
        try:
            x1, y1, x2, y2 = normalized_bbox
            return [
                x1 * image_width,
                y1 * image_height,
                x2 * image_width,
                y2 * image_height
            ]
        except Exception as e:
            logger.error(f"Failed to denormalize bbox: {e}")
            return normalized_bbox
    
    @staticmethod
    def filter_predictions_by_confidence(predictions: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """Filter predictions by minimum confidence threshold"""
        return [pred for pred in predictions if pred.get('confidence', 0.0) >= min_confidence]

class TimeUtils:
    """Time and date utilities"""
    
    @staticmethod
    def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime(format_str)
    
    @staticmethod
    def get_utc_timestamp() -> str:
        """Get UTC timestamp in ISO format"""
        return datetime.utcnow().isoformat()

# Convenience functions
def ensure_dir(path: Union[str, Path]) -> Path:
    """Shorthand for FileUtils.ensure_dir"""
    return FileUtils.ensure_dir(path)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Shorthand for ConfigUtils.load_config"""
    return ConfigUtils.load_config(config_path)

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Shorthand for TimeUtils.get_timestamp"""
    return TimeUtils.get_timestamp(format_str)
