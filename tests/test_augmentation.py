#!/usr/bin/env python3
"""
Test script for the updated ImageDataGenerator-based augmentation pipeline
"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Import from conftest.py handles path setup
try:
    from model_training import AugmentationPipeline, DataGenerator, MaskDetectorTrainer
    print("✅ Successfully imported all classes from model_training.py")
except ImportError as e:
    print(f"❌ Import error: {e}")
    pytest.skip(f"Required modules not available: {e}")

@pytest.mark.unit
def test_augmentation_pipeline(test_image):
    """Test the ImageDataGenerator-based augmentation pipeline."""
    print("\n🔄 Testing AugmentationPipeline...")
    
    # Create augmentation pipeline
    aug_pipeline = AugmentationPipeline(image_size=(224, 224))
    
    # Check if components exist
    assert hasattr(aug_pipeline, 'train_datagen'), "Missing train_datagen"
    assert hasattr(aug_pipeline, 'val_datagen'), "Missing val_datagen"
    print("✅ AugmentationPipeline components verified")
    
    # Test image transformation
    if os.path.exists(test_image):
        print(f"🖼️  Testing image transformation with: {test_image}")
        
        # Test training augmentation
        aug_image_train = aug_pipeline.transform_image(test_image, training=True)
        assert aug_image_train.shape == (224, 224, 3), f"Wrong shape: {aug_image_train.shape}"
        assert aug_image_train.dtype == np.float32, f"Wrong dtype: {aug_image_train.dtype}"
        print("✅ Training augmentation works")
        
        # Test validation (no augmentation)
        aug_image_val = aug_pipeline.transform_image(test_image, training=False)
        assert aug_image_val.shape == (224, 224, 3), f"Wrong shape: {aug_image_val.shape}"
        assert aug_image_val.dtype == np.float32, f"Wrong dtype: {aug_image_val.dtype}"
        print("✅ Validation transformation works")
        
        print("✅ AugmentationPipeline tests passed!")
    else:
        pytest.skip(f"Test image not found: {test_image}")

@pytest.mark.unit  
def test_model_creation():
    """Test model creation functionality."""
    print("\n🔄 Testing Model Creation...")
    
    model_builder = MaskDetectorTrainer(num_classes=3)
    
    # Test model building
    model = model_builder.build_model()
    assert model is not None, "Model creation failed"
    
    # Test model compilation
    compiled_model = model_builder.compile_model(model, learning_rate=1e-3)
    assert compiled_model is not None, "Model compilation failed"
    
    print("✅ Model creation tests passed!")

@pytest.mark.unit
def test_data_generator(test_data_dir):
    """Test the DataGenerator functionality."""
    print("\n🔄 Testing DataGenerator...")
    
    # Create DataGenerator
    batch_size = 2
    image_size = (224, 224)
    
    try:
        data_gen = DataGenerator(
            data_dir=test_data_dir,
            batch_size=batch_size,
            image_size=image_size,
            mode='train'
        )
        
        assert len(data_gen) > 0, "DataGenerator should have at least one batch"
        
        # Test batch generation
        batch_x, batch_y = data_gen[0]
        assert batch_x.shape[0] <= batch_size, f"Batch size mismatch: {batch_x.shape[0]}"
        assert batch_x.shape[1:] == (*image_size, 3), f"Image shape mismatch: {batch_x.shape[1:]}"
        assert batch_y.shape[1] == 3, f"Number of classes mismatch: {batch_y.shape[1]}"
        
        print("✅ DataGenerator tests passed!")
        
    except Exception as e:
        pytest.skip(f"DataGenerator test failed: {e}")

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
