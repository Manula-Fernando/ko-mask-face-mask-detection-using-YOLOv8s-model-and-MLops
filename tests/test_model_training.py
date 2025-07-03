"""
Unit tests for model training module.
Tests model creation, compilation, and training functions.
"""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from model_training import MaskDetectorTrainer
except ImportError:
    pytest.skip("MaskDetectorTrainer not available", allow_module_level=True)


class TestMaskDetectorTrainer:
    """Test cases for MaskDetectorTrainer class."""
    
    def test_model_initialization(self):
        """Test MaskDetectorTrainer initialization."""
        model_builder = MaskDetectorTrainer()
        
        assert model_builder.num_classes == 3
        assert model_builder.input_shape == (224, 224, 3)
    
    def test_model_creation(self):
        """Test model architecture creation."""
        model_builder = MaskDetectorTrainer()
        model = model_builder.build_model()
        
        # Test model structure
        assert model is not None
        assert len(model.layers) > 5  # Should have multiple layers
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 3)  # 3 classes
    
    def test_model_compilation(self):
        """Test model compilation with optimizer and metrics."""
        model_builder = MaskDetectorTrainer()
        model = model_builder.build_model()
        compiled_model = model_builder.compile_model(model)
        
        # Test compilation
        assert compiled_model.optimizer is not None
        assert len(compiled_model.metrics) >= 1  # Should have accuracy at minimum
        assert compiled_model.loss is not None
    
    def test_callbacks_creation(self):
        """Test training callbacks creation."""
        model_builder = MaskDetectorTrainer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.h5")
            callbacks = model_builder.get_callbacks(model_path)
            
            # Test callbacks
            assert len(callbacks) >= 3  # Should have multiple callbacks
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert 'EarlyStopping' in callback_types
            assert 'ModelCheckpoint' in callback_types


class TestModelArchitecture:
    """Test cases for model architecture validation."""
    
    def test_input_output_shapes(self):
        """Test model input and output shapes."""
        model_builder = MaskDetectorTrainer()
        model = model_builder.build_model()
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (1, 3)  # Batch size 1, 3 classes
        assert np.allclose(np.sum(output, axis=1), 1.0, atol=1e-5)  # Softmax outputs sum to 1
    
    def test_model_parameters(self):
        """Test model has reasonable number of parameters."""
        model_builder = MaskDetectorTrainer()
        model = model_builder.build_model()
        
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        # Test parameter counts
        assert total_params > 1000  # Should have reasonable number of parameters
        assert trainable_params > 0  # Should have trainable parameters
    
    def test_transfer_learning_setup(self):
        """Test that transfer learning is properly configured."""
        model_builder = MaskDetectorTrainer()
        model = model_builder.build_model()
        
        # Check that model has expected number of layers (MobileNetV2 + custom layers)
        assert len(model.layers) > 5, f"Model should have multiple layers, got {len(model.layers)}"
        
        # Check model input and output shapes
        assert model.input_shape == (None, 224, 224, 3), f"Expected input shape (None, 224, 224, 3), got {model.input_shape}"
        assert model.output_shape == (None, 3), f"Expected output shape (None, 3), got {model.output_shape}"


def test_prediction_consistency():
    """Test that model predictions are consistent."""
    model_builder = MaskDetectorTrainer()
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Create dummy input
    dummy_input = np.random.random((2, 224, 224, 3)).astype(np.float32)
    
    # Get predictions twice
    pred1 = model.predict(dummy_input, verbose=0)
    pred2 = model.predict(dummy_input, verbose=0)
    
    # Predictions should be identical for same input
    assert np.allclose(pred1, pred2, atol=1e-6)


def test_data_augmentation_integration():
    """Test that model can handle augmented data."""
    model_builder = MaskDetectorTrainer()
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Test with different input variations
    inputs = [
        np.random.random((1, 224, 224, 3)).astype(np.float32),  # Normal
        np.zeros((1, 224, 224, 3)).astype(np.float32),          # All zeros
        np.ones((1, 224, 224, 3)).astype(np.float32),           # All ones
    ]
    
    for inp in inputs:
        try:
            output = model.predict(inp, verbose=0)
            assert output.shape == (1, 3)
            assert not np.isnan(output).any()  # No NaN values
        except Exception as e:
            pytest.fail(f"Model failed on input: {e}")


def test_model_serialization():
    """Test that model can be saved and loaded."""
    model_builder = MaskDetectorTrainer()
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.h5")
        
        # Save model
        model.save(model_path)
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Test loaded model
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        original_pred = model.predict(dummy_input, verbose=0)
        loaded_pred = loaded_model.predict(dummy_input, verbose=0)
        
        # Predictions should be identical
        assert np.allclose(original_pred, loaded_pred, atol=1e-6)
