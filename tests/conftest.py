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

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "app"))

@pytest.fixture(scope="session")
def test_image():
    """Create a test image for testing purposes."""
    # Create a dummy RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories
        for class_name in ['with_mask', 'without_mask', 'mask_weared_incorrect']:
            class_dir = Path(temp_dir) / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Create a few test images in each directory
            for i in range(3):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(class_dir / f"test_image_{i}.jpg")
        
        yield temp_dir

@pytest.fixture
def sample_predictions():
    """Sample prediction data for testing."""
    return {
        'prediction': 'with_mask',
        'confidence': 0.95,
        'processing_time': 0.045
    }
