"""
Simple tests for the MLOps pipeline components.
"""

import pytest
import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataPreprocessing:
    """Test data preprocessing pipeline."""
    
    def test_data_preprocessing_module_imports(self):
        """Test that data preprocessing module can be imported."""
        try:
            from data_preprocessing import DataPreprocessor
            assert DataPreprocessor is not None
        except ImportError:
            pytest.skip("Data preprocessing module not available")
    
    def test_config_loading(self):
        """Test configuration loading."""
        config_data = {
            'data': {
                'image_size': [224, 224],
                'batch_size': 32,
                'classes': ['with_mask', 'without_mask', 'mask_weared_incorrect']
            }
        }
        
        with patch('yaml.safe_load', return_value=config_data), \
             patch('builtins.open'):
            
            from data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            
            assert preprocessor.image_size == (224, 224)
            assert preprocessor.batch_size == 32
            assert len(preprocessor.classes) == 3


class TestModelTraining:
    """Test model training pipeline."""
    
    def test_model_training_module_imports(self):
        """Test that model training module can be imported."""
        try:
            from model_training import FaceMaskDetector
            assert FaceMaskDetector is not None
        except ImportError:
            pytest.skip("Model training module not available")
    
    @patch('model_training.yaml.safe_load')
    def test_model_initialization(self, mock_yaml):
        """Test model detector initialization."""
        mock_yaml.return_value = {
            'model': {
                'input_shape': [224, 224, 3],
                'num_classes': 3,
                'dropout_rate': 0.5
            },
            'training': {
                'epochs': 30,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'patience': 10
            }
        }
        
        with patch('builtins.open'):
            from model_training import FaceMaskDetector
            detector = FaceMaskDetector()
            
            assert detector.model is None
            assert detector.history is None


class TestPrediction:
    """Test prediction pipeline."""
    
    def test_prediction_module_imports(self):
        """Test that prediction module can be imported."""
        try:
            from predict import MaskPredictor
            assert MaskPredictor is not None
        except ImportError:
            pytest.skip("Prediction module not available")
    
    @patch('predict.yaml.safe_load')
    def test_predictor_initialization(self, mock_yaml):
        """Test predictor initialization."""
        mock_yaml.return_value = {
            'model': {'input_shape': [224, 224, 3]},
            'data': {'classes': ['with_mask', 'without_mask', 'mask_weared_incorrect']},
            'paths': {'test_dir': 'test'},
            'mlflow': {'experiment_name': 'test'}
        }
        
        with patch('builtins.open'):
            from predict import MaskPredictor
            predictor = MaskPredictor()
            
            assert predictor.model is None
            assert len(predictor.class_names) == 3


class TestDVCPipeline:
    """Test DVC pipeline configuration."""
    
    def test_dvc_yaml_exists(self):
        """Test that DVC pipeline configuration exists."""
        dvc_path = os.path.join(os.path.dirname(__file__), '..', 'dvc.yaml')
        assert os.path.exists(dvc_path)
    
    def test_config_yaml_exists(self):
        """Test that configuration file exists."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        assert os.path.exists(config_path)
    
    def test_dvc_pipeline_structure(self):
        """Test DVC pipeline has required stages."""
        dvc_path = os.path.join(os.path.dirname(__file__), '..', 'dvc.yaml')
        
        if os.path.exists(dvc_path):
            import yaml
            with open(dvc_path, 'r') as f:
                dvc_config = yaml.safe_load(f)
            
            assert 'stages' in dvc_config
            stages = dvc_config['stages']
            
            # Check required stages exist
            required_stages = ['data_preprocessing', 'model_training', 'model_evaluation']
            for stage in required_stages:
                assert stage in stages


class TestModelsDirectory:
    """Test models directory and outputs."""
    
    def test_models_directory_exists(self):
        """Test that models directory exists."""
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        assert os.path.exists(models_path)
    
    def test_trained_model_exists(self):
        """Test that trained model file exists."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mask_detector.h5')
        if os.path.exists(model_path):
            # If model exists, check it's not empty
            assert os.path.getsize(model_path) > 0
        else:
            pytest.skip("Trained model not found - run pipeline first")
    
    def test_evaluation_metrics_exist(self):
        """Test that evaluation metrics exist."""
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'evaluation_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            assert 'accuracy' in metrics
            assert 'test_samples' in metrics
            assert isinstance(metrics['accuracy'], (int, float))
        else:
            pytest.skip("Evaluation metrics not found - run pipeline first")


class TestProjectStructure:
    """Test overall project structure."""
    
    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        assert os.path.exists(src_path)
    
    def test_data_directory_structure(self):
        """Test data directory structure."""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        if os.path.exists(data_path):
            # Check for processed data if it exists
            processed_path = os.path.join(data_path, 'processed')
            if os.path.exists(processed_path):
                # Check for train/val/test splits
                for split in ['train', 'val', 'test']:
                    split_path = os.path.join(processed_path, split)
                    if os.path.exists(split_path):
                        assert os.path.isdir(split_path)


if __name__ == '__main__':
    pytest.main([__file__])
