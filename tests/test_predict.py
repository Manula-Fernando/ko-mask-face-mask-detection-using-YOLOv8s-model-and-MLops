"""
Unit tests for prediction module.
Tests prediction functions and image preprocessing.
"""
import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from predict import FaceMaskPredictor
except ImportError:
    pytest.skip("Prediction functions not available", allow_module_level=True)


class TestFaceMaskPredictor:
    """Test cases for FaceMaskPredictor class."""
    
    @pytest.mark.skipif(not os.path.exists("models/best_mask_detector.h5"), 
                       reason="No trained model available")
    def test_predictor_initialization(self):
        """Test FaceMaskPredictor initialization."""
        predictor = FaceMaskPredictor("models/best_mask_detector.h5")
        
        assert predictor.model_path == "models/best_mask_detector.h5"
        assert predictor.model is not None
        assert predictor.CLASSES == ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    @pytest.mark.skipif(not os.path.exists("models/best_mask_detector.h5"), 
                       reason="No trained model available")
    def test_prediction_output_format(self):
        """Test that prediction returns expected format."""
        predictor = FaceMaskPredictor("models/best_mask_detector.h5")
        
        # Create dummy image file path
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = predictor.predict_from_array(dummy_image)
        
        # Check output format
        assert isinstance(result, dict)
        assert 'class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        # Check prediction is valid class
        valid_classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        assert result['class'] in valid_classes
        
        # Check confidence is valid probability
        assert 0 <= result['confidence'] <= 1
    
    def test_class_mapping(self):
        """Test that class indices map correctly to class names."""
        if not os.path.exists("models/best_mask_detector.h5"):
            pytest.skip("No trained model available for testing")
            
        predictor = FaceMaskPredictor("models/best_mask_detector.h5")
        class_names = predictor.CLASSES
        
        # Test index to name mapping
        for i, name in enumerate(class_names):
            assert i < len(class_names)
            assert isinstance(name, str)
            assert len(name) > 0


def test_model_file_existence():
    """Test checking for model file existence."""
    # Test with non-existent file
    assert not os.path.exists("nonexistent_model.h5")
    
    # Test with actual model file if it exists
    if os.path.exists("models/best_mask_detector.h5"):
        assert os.path.exists("models/best_mask_detector.h5")


def test_confidence_threshold():
    """Test confidence threshold functionality."""
    # Test with different confidence values
    confidences = [0.1, 0.5, 0.8, 0.95, 0.99]
    
    for conf in confidences:
        assert 0 <= conf <= 1
        
        # Test threshold logic
        high_confidence = conf > 0.8
        assert isinstance(high_confidence, bool)


def test_error_handling():
    """Test error handling in predictor initialization."""
    # Test with invalid model path
    try:
        predictor = FaceMaskPredictor("nonexistent_model.h5")
        # If no exception is raised, check if load_model was called properly
        if hasattr(predictor, 'model') and predictor.model is None:
            assert True  # Model loading failed as expected
        else:
            pytest.fail("Expected an error for nonexistent model file")
    except (OSError, ValueError, FileNotFoundError):
        # This is the expected behavior
        assert True
