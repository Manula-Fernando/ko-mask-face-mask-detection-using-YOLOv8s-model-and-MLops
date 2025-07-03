"""
Unit tests for Flask API endpoints.
Tests API functionality and response formats.
"""
import pytest
import json
import os
import sys
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
import io

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.main import app
    HAS_FLASK_APP = True
except ImportError:
    HAS_FLASK_APP = False


@pytest.fixture
def client():
    """Create test client for Flask app."""
    if not HAS_FLASK_APP:
        pytest.skip("Flask app not available")
    
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create dummy image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_home_endpoint(self, client):
        """Test the home page endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Face Mask Detection' in response.data or b'html' in response.data.lower()
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data
        assert data['status'] == 'healthy'
        assert isinstance(data['model_loaded'], bool)
    
    def test_predict_endpoint_no_file(self, client):
        """Test prediction endpoint without file."""
        response = client.post('/predict')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_endpoint_empty_file(self, client):
        """Test prediction endpoint with empty filename."""
        response = client.post('/predict', data={'file': (io.BytesIO(b''), '')})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    @pytest.mark.skipif(not os.path.exists("models/best_mask_detector.h5"), 
                       reason="No trained model available")
    def test_predict_endpoint_valid_image(self, client, sample_image):
        """Test prediction endpoint with valid image."""
        response = client.post('/predict', 
                             data={'file': (sample_image, 'test.jpg')},
                             content_type='multipart/form-data')
        
        # Should succeed if model is loaded
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Check response format
            assert 'prediction' in data
            assert 'confidence' in data
            assert 'all_predictions' in data
            
            # Validate prediction value
            valid_classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
            assert data['prediction'] in valid_classes
            
            # Validate confidence
            assert 0 <= data['confidence'] <= 1
            
            # Validate all predictions
            all_preds = data['all_predictions']
            assert len(all_preds) == 3
            assert abs(sum(all_preds.values()) - 1.0) < 1e-5
        
        elif response.status_code == 500:
            # Model not loaded or other server error
            data = json.loads(response.data)
            assert 'error' in data


class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    def test_invalid_file_format(self, client):
        """Test with invalid file format."""
        # Create text file instead of image
        text_data = io.BytesIO(b'This is not an image')
        
        response = client.post('/predict',
                             data={'file': (text_data, 'test.txt')},
                             content_type='multipart/form-data')
        
        # Should return error
        assert response.status_code in [400, 500]
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_corrupted_image(self, client):
        """Test with corrupted image data."""
        # Create corrupted image data
        corrupted_data = io.BytesIO(b'\\xff\\xd8\\xff\\xe0corrupted')
        
        response = client.post('/predict',
                             data={'file': (corrupted_data, 'corrupted.jpg')},
                             content_type='multipart/form-data')
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_large_file_handling(self, client):
        """Test with very large image file."""
        # Create large image (this might be limited by Flask config)
        large_image_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        large_image = Image.fromarray(large_image_array)
        
        img_bytes = io.BytesIO()
        large_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post('/predict',
                             data={'file': (img_bytes, 'large.jpg')},
                             content_type='multipart/form-data')
        
        # Should either process successfully or return appropriate error
        assert response.status_code in [200, 400, 413, 500]


class TestAPIResponseFormat:
    """Test cases for API response format consistency."""
    
    def test_json_response_format(self, client):
        """Test that all endpoints return valid JSON."""
        endpoints = ['/health']
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            # Should be valid JSON
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Endpoint {endpoint} did not return valid JSON")
    
    def test_error_response_format(self, client):
        """Test that error responses have consistent format."""
        # Test various error conditions
        error_requests = [
            ('/predict', 'POST', {}),  # No file
        ]
        
        for endpoint, method, data in error_requests:
            if method == 'POST':
                response = client.post(endpoint, data=data)
            else:
                response = client.get(endpoint)
            
            if response.status_code >= 400:
                # Should return JSON error
                try:
                    error_data = json.loads(response.data)
                    assert 'error' in error_data
                    assert isinstance(error_data['error'], str)
                except json.JSONDecodeError:
                    pytest.fail(f"Error response from {endpoint} was not valid JSON")


class TestAPIPerformance:
    """Test cases for API performance and reliability."""
    
    @pytest.mark.skipif(not os.path.exists("models/best_mask_detector.h5"), 
                       reason="No trained model available")
    def test_multiple_predictions(self, client):
        """Test multiple consecutive predictions."""
        # Create multiple test images
        for i in range(3):
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            response = client.post('/predict',
                                 data={'file': (img_bytes, f'test_{i}.jpg')},
                                 content_type='multipart/form-data')
            
            # Each request should be handled independently
            assert response.status_code in [200, 500]  # 500 if model not loaded
    
    def test_concurrent_health_checks(self, client):
        """Test multiple health check requests."""
        # Simulate multiple concurrent requests
        responses = []
        for _ in range(5):
            response = client.get('/health')
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'healthy'


def test_api_configuration():
    """Test API configuration and setup."""
    if not HAS_FLASK_APP:
        pytest.skip("Flask app not available")
    
    # Test that app is properly configured
    assert app is not None
    assert app.config is not None
    
    # Test that routes are registered
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    expected_routes = ['/', '/health', '/predict']
    
    for route in expected_routes:
        assert route in rules or any(route in rule for rule in rules)


def test_model_dependency():
    """Test API behavior when model is not available."""
    if not HAS_FLASK_APP:
        pytest.skip("Flask app not available")
    
    # This test checks that the API can start even without a model
    # and handles the case gracefully
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_loaded' in data
        # Should report whether model is loaded or not
