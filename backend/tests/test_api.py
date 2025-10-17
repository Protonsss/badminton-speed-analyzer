"""
Tests for API Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import io

from app.main import app
from app.api.routes import set_processor
from app.core.processor import VideoProcessor


@pytest.fixture
def client():
    """Test client fixture"""
    # Initialize processor
    processor = VideoProcessor(use_yolo=False)  # Disable YOLO for tests
    set_processor(processor)
    
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "cuda_available" in data
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        files = {"file": ("test.txt", io.BytesIO(b"test"), "text/plain")}
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 400
    
    def test_calibrate_success(self, client):
        """Test successful calibration"""
        data = {
            "points": [[100, 100], [700, 100], [700, 500], [100, 500]],
            "court_type": "singles"
        }
        response = client.post("/api/v1/calibrate", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["court_type"] == "singles"
    
    def test_calibrate_invalid_points(self, client):
        """Test calibration with invalid points"""
        data = {
            "points": [[100, 100], [700, 100]],  # Only 2 points
            "court_type": "singles"
        }
        response = client.post("/api/v1/calibrate", json=data)
        assert response.status_code == 422  # Validation error
    
    def test_calibrate_collinear_points(self, client):
        """Test calibration with collinear points"""
        data = {
            "points": [[100, 100], [200, 100], [300, 100], [400, 100]],
            "court_type": "singles"
        }
        response = client.post("/api/v1/calibrate", json=data)
        assert response.status_code == 400
    
    def test_analyze_without_file(self, client):
        """Test analyze without uploaded file"""
        data = {
            "file_id": "/nonexistent/file.mp4",
            "use_yolo": False,
            "start_frame": 0
        }
        response = client.post("/api/v1/analyze", data=data)
        assert response.status_code == 404

