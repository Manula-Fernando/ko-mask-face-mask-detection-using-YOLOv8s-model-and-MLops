"""
Test suite for the Flask web application
"""
import pytest
import json
import io
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app, allowed_file


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io


class TestRoutes:
    """Test Flask application routes"""
    
    def test_home_route(self, client):
        """Test the home page route"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Face Mask Detection' in response.data
    
    def test_upload_route(self, client):
        """Test the upload page route"""
        response = client.get('/upload')
        assert response.status_code == 200
        assert b'Upload Image' in response.data
    
    def test_webcam_route(self, client):
        """Test the webcam page route"""
        response = client.get('/webcam')
        assert response.status_code == 200
        assert b'Real-time Detection' in response.data
    
    def test_result_route(self, client):
        """Test the result page route"""
        response = client.get('/result')
        assert response.status_code == 200
        assert b'Analysis Results' in response.data
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'model_loaded' in data
    
    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint"""
        response = client.get('/metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'total_predictions' in data
        assert 'model_version' in data
        assert 'uptime' in data


class TestFileUpload:
    """Test file upload functionality"""
    
    def test_allowed_file_valid_extensions(self):
        """Test allowed file function with valid extensions"""
        assert allowed_file('test.jpg')
        assert allowed_file('test.jpeg')
        assert allowed_file('test.png')
        assert allowed_file('test.gif')
        assert allowed_file('TEST.JPG')  # Case insensitive
    
    def test_allowed_file_invalid_extensions(self):
        """Test allowed file function with invalid extensions"""
        assert not allowed_file('test.txt')
        assert not allowed_file('test.pdf')
        assert not allowed_file('test')
        assert not allowed_file('')
    
    @patch('main.predictor.predict')
    def test_predict_api_success(self, mock_predict, client, sample_image):
        """Test successful prediction API call"""
        # Mock the prediction result
        mock_predict.return_value = {
            'prediction': 'Mask',
            'confidence': 0.95,
            'faces_detected': 1,
            'processing_time': 0.5,
            'model_version': 'v1.0'
        }
        
        data = {
            'file': (sample_image, 'test.jpg')
        }
        
        response = client.post('/api/predict', 
                              data=data, 
                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['prediction'] == 'Mask'
        assert result['confidence'] == 0.95
        assert mock_predict.called
    
    def test_predict_api_no_file(self, client):
        """Test prediction API with no file"""
        response = client.post('/api/predict')
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
    
    def test_predict_api_invalid_file(self, client):
        """Test prediction API with invalid file"""
        data = {
            'file': (io.StringIO('not an image'), 'test.txt')
        }
        
        response = client.post('/api/predict', 
                              data=data, 
                              content_type='multipart/form-data')
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
    
    @patch('main.predictor.predict')
    def test_predict_api_prediction_error(self, mock_predict, client, sample_image):
        """Test prediction API when prediction fails"""
        # Mock prediction to raise an exception
        mock_predict.side_effect = Exception("Model error")
        
        data = {
            'file': (sample_image, 'test.jpg')
        }
        
        response = client.post('/api/predict', 
                              data=data, 
                              content_type='multipart/form-data')
        
        assert response.status_code == 500
        result = json.loads(response.data)
        assert 'error' in result


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent-route')
        assert response.status_code == 404
    
    def test_500_error_handling(self, client):
        """Test 500 error handling"""
        # This would need to be triggered by an actual server error
        # For now, just test that the error template exists
        pass


class TestSecurity:
    """Test security features"""
    
    def test_file_size_limit(self, client):
        """Test file size limit"""
        # Create a large file (this is a simplified test)
        large_data = b'x' * (16 * 1024 * 1024 + 1)  # > 16MB
        data = {
            'file': (io.BytesIO(large_data), 'large.jpg')
        }
        
        response = client.post('/api/predict', 
                              data=data, 
                              content_type='multipart/form-data')
        
        # Flask should reject this before it reaches our handler
        assert response.status_code in [413, 400]
    
    def test_cors_headers(self, client):
        """Test CORS headers are not exposed in production"""
        response = client.get('/')
        # In production, CORS should be restricted
        assert 'Access-Control-Allow-Origin' not in response.headers


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Mock environment variables
    with patch.dict(os.environ, {
        'FLASK_ENV': 'testing',
        'MODEL_PATH': 'test_model.h5'
    }):
        yield


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
