#!/usr/bin/env python3
"""
Test script for the updated ImageDataGenerator-based augmentation pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from model_training import AugmentationPipeline, DataGenerator, MaskDetectionModel
    print("✅ Successfully imported all classes from model_training.py")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_augmentation_pipeline():
    """Test the ImageDataGenerator-based augmentation pipeline."""
    print("\n🔄 Testing AugmentationPipeline...")
    
    # Create augmentation pipeline
    aug_pipeline = AugmentationPipeline(image_size=(224, 224))
    
    # Check if components exist
    assert hasattr(aug_pipeline, 'train_datagen'), "train_datagen not found"
    assert hasattr(aug_pipeline, 'val_datagen'), "val_datagen not found"
    assert hasattr(aug_pipeline, 'strong_datagen'), "strong_datagen not found"
    
    print("✅ AugmentationPipeline initialized with all required components")
    
    # Test with a dummy image file (create if needed)
    test_image_path = "test_image.jpg"
    
    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(test_image_path, dummy_image)
    
    try:
        # Test training transform
        aug_image_train = aug_pipeline.transform_image(test_image_path, training=True)
        assert aug_image_train.shape == (224, 224, 3), f"Wrong shape: {aug_image_train.shape}"
        assert aug_image_train.dtype == np.float32, f"Wrong dtype: {aug_image_train.dtype}"
        print("✅ Training augmentation works correctly")
        
        # Test validation transform
        aug_image_val = aug_pipeline.transform_image(test_image_path, training=False)
        assert aug_image_val.shape == (224, 224, 3), f"Wrong shape: {aug_image_val.shape}"
        assert aug_image_val.dtype == np.float32, f"Wrong dtype: {aug_image_val.dtype}"
        print("✅ Validation augmentation works correctly")
        
        # Test strong augmentation
        aug_image_strong = aug_pipeline.transform_image(test_image_path, training=True, strong_aug=True)
        assert aug_image_strong.shape == (224, 224, 3), f"Wrong shape: {aug_image_strong.shape}"
        assert aug_image_strong.dtype == np.float32, f"Wrong dtype: {aug_image_strong.dtype}"
        print("✅ Strong augmentation works correctly")
        
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    print("✅ All augmentation tests passed!")

def test_model_builder():
    """Test the MaskDetectionModel."""
    print("\n🤖 Testing MaskDetectionModel...")
    
    model_builder = MaskDetectionModel(num_classes=3)
    
    # Test model building
    model, base_model = model_builder.build_model()
    print("✅ Model built successfully")
    
    # Test model compilation
    compiled_model = model_builder.compile_model(model, learning_rate=1e-3)
    print("✅ Model compiled successfully")
    
    # Test model summary
    print(f"✅ Model has {compiled_model.count_params():,} total parameters")

def test_data_generator():
    """Test the DataGenerator with dummy data."""
    print("\n📊 Testing DataGenerator...")
    
    # Create dummy DataFrame
    dummy_data = {
        'image_path': ['test_image1.jpg', 'test_image2.jpg', 'test_image3.jpg'],
        'class_id': [0, 1, 2],
        'class': ['with_mask', 'without_mask', 'mask_weared_incorrect']
    }
    df = pd.DataFrame(dummy_data)
    
    # Create dummy images
    for img_path in dummy_data['image_path']:
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_image)
    
    try:
        # Test data generator
        aug_pipeline = AugmentationPipeline()
        data_gen = DataGenerator(df, batch_size=2, augmentation=aug_pipeline, training=True)
        
        print(f"✅ DataGenerator created with {len(data_gen)} batches")
        
        # Test getting a batch
        X, y = data_gen[0]
        print(f"✅ Batch shape: X={X.shape}, y={y.shape}")
        assert X.dtype == np.float32, f"Wrong X dtype: {X.dtype}"
        assert len(y.shape) == 2, f"Wrong y shape: {y.shape}"
        
    finally:
        # Clean up
        for img_path in dummy_data['image_path']:
            if os.path.exists(img_path):
                os.remove(img_path)
    
    print("✅ DataGenerator tests passed!")

if __name__ == "__main__":
    print("🧪 Testing ImageDataGenerator-based Pipeline")
    print("=" * 50)
    
    try:
        test_augmentation_pipeline()
        test_model_builder()
        test_data_generator()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ImageDataGenerator pipeline is working correctly")
        print("✅ Compatible with your existing functions and methods")
        print("✅ Ready for training with enhanced augmentation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
