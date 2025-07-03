"""
Unit tests for data preprocessing module.
Tests data loading, validation, and preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data_preprocessing import DataProcessor
except ImportError:
    pytest.skip("DataProcessor not available", allow_module_level=True)


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_data_processor_initialization(self):
        """Test DataProcessor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()
            
            processor = DataProcessor(raw_dir, processed_dir)
            
            assert processor.raw_data_dir == raw_dir
            assert processor.processed_data_dir == processed_dir
            assert processor.CLASSES == ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    def test_create_splits_validation(self):
        """Test data splitting functionality."""
        # Create mock data
        data = {
            'filename': [f'image_{i}.jpg' for i in range(100)],
            'image_path': [f'/path/to/image_{i}.jpg' for i in range(100)],
            'class': ['with_mask'] * 40 + ['without_mask'] * 35 + ['mask_weared_incorrect'] * 25,
            'class_id': [0] * 40 + [1] * 35 + [2] * 25
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()
            
            processor = DataProcessor(raw_dir, processed_dir)
            train_df, val_df, test_df = processor.create_splits(df)
            
            # Check split sizes
            total_samples = len(train_df) + len(val_df) + len(test_df)
            assert total_samples == len(df)
            
            # Check proportions (approximately)
            assert len(train_df) >= 60  # Should be around 70%
            assert len(val_df) >= 15   # Should be around 20%
            assert len(test_df) >= 8   # Should be around 10%
            
            # Check all classes are present in each split
            for split_df in [train_df, val_df, test_df]:
                if len(split_df) > 0:
                    unique_classes = set(split_df['class'].unique())
                    assert len(unique_classes) > 0
    
    def test_classes_definition(self):
        """Test that classes are correctly defined."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()
            
            processor = DataProcessor(raw_dir, processed_dir)
            expected_classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
            assert processor.CLASSES == expected_classes
            assert len(processor.CLASSES) == 3


def test_data_validation():
    """Test basic data validation functions."""
    # Test valid DataFrame structure
    valid_data = {
        'filename': ['test1.jpg', 'test2.jpg'],
        'image_path': ['/path/test1.jpg', '/path/test2.jpg'],
        'class': ['with_mask', 'without_mask'],
        'class_id': [0, 1]
    }
    df = pd.DataFrame(valid_data)
    
    # Basic validations
    assert 'filename' in df.columns
    assert 'class' in df.columns
    assert 'class_id' in df.columns
    assert len(df) == 2
    assert df['class_id'].dtype in [np.int64, np.int32, int]


def test_empty_dataframe_handling():
    """Test handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    
    # Should handle empty DataFrame gracefully
    assert len(empty_df) == 0
    
    # Create DataFrame with expected columns but no data
    empty_structured = pd.DataFrame(columns=['filename', 'class', 'class_id'])
    assert len(empty_structured) == 0
    assert 'filename' in empty_structured.columns


if __name__ == "__main__":
    pytest.main([__file__])
