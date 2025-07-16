"""
Pytest configuration and fixtures for the Professional Face Mask Detection project.
"""
import pytest
import sys
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

# Add src and app to Python path for imports in tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "app"))

@pytest.fixture(scope="session")
def test_image():
    """Create a test image file and return its path. Deletes after test session."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory with subfolders and test images for each mask class."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for class_name in ['with_mask', 'without_mask', 'mask_weared_incorrect']:
            class_dir = Path(temp_dir) / class_name
            class_dir.mkdir(exist_ok=True)
            for i in range(3):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(class_dir / f"test_image_{i}.jpg")
        yield temp_dir

@pytest.fixture
def sample_predictions():
    """Return a sample prediction dictionary for testing."""
    return {
        'prediction': 'with_mask',
        'confidence': 0.95,
        'processing_time': 0.045
    }

@pytest.fixture
def dummy_image_array():
    """Return a dummy numpy image array for direct model input tests."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def api_test_client():
    """
    Provide a FastAPI test client if your API is built with FastAPI.
    Usage: pass as argument to your test function.
    """
    try:
        from fastapi.testclient import TestClient
        from src.inference.api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI or app not available for API testing.")

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for saving test results or files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir