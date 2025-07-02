"""
Legacy test file - imports from test_flask_app.py for compatibility
Phase 3 MLOps Implementation - All tests use real Flask app, no mocks
"""

import pytest
from tests.test_flask_app import *

if __name__ == '__main__':
    pytest.main([__file__])
