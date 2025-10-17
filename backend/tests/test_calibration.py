"""
Tests for Court Calibration
"""
import pytest
import numpy as np
from app.core.calibration import CourtCalibration, COURT_DIMENSIONS
from app.core.exceptions import ValidationError, CalibrationError


class TestCourtCalibration:
    """Test court calibration functionality"""
    
    def test_init_singles(self):
        """Test initialization with singles court"""
        calib = CourtCalibration(court_type='singles')
        assert calib.court_type == 'singles'
        assert calib.dimensions == COURT_DIMENSIONS['singles']
        assert not calib.is_calibrated()
    
    def test_init_doubles(self):
        """Test initialization with doubles court"""
        calib = CourtCalibration(court_type='doubles')
        assert calib.court_type == 'doubles'
        assert calib.dimensions == COURT_DIMENSIONS['doubles']
    
    def test_init_invalid_court_type(self):
        """Test initialization with invalid court type"""
        with pytest.raises(ValidationError):
            CourtCalibration(court_type='invalid')
    
    def test_calibrate_success(self):
        """Test successful calibration"""
        calib = CourtCalibration()
        points = [(100, 100), (700, 100), (700, 500), (100, 500)]
        calib.calibrate(points)
        assert calib.is_calibrated()
        assert calib.H is not None
        assert calib.H_inv is not None
    
    def test_calibrate_wrong_number_of_points(self):
        """Test calibration with wrong number of points"""
        calib = CourtCalibration()
        with pytest.raises(ValidationError):
            calib.calibrate([(100, 100), (700, 100)])
    
    def test_calibrate_collinear_points(self):
        """Test calibration with collinear points"""
        calib = CourtCalibration()
        points = [(100, 100), (200, 100), (300, 100), (400, 100)]
        with pytest.raises(ValidationError):
            calib.calibrate(points)
    
    def test_pixel_to_meter(self):
        """Test pixel to meter conversion"""
        calib = CourtCalibration()
        points = [(0, 0), (640, 0), (640, 480), (0, 480)]
        calib.calibrate(points)
        
        x_m, y_m = calib.pixel_to_meter(320, 240)
        assert isinstance(x_m, float)
        assert isinstance(y_m, float)
    
    def test_pixel_to_meter_not_calibrated(self):
        """Test pixel to meter without calibration"""
        calib = CourtCalibration()
        with pytest.raises(CalibrationError):
            calib.pixel_to_meter(100, 100)
    
    def test_meter_to_pixel(self):
        """Test meter to pixel conversion"""
        calib = CourtCalibration()
        points = [(0, 0), (640, 0), (640, 480), (0, 480)]
        calib.calibrate(points)
        
        x_px, y_px = calib.meter_to_pixel(2.5, 6.7)
        assert isinstance(x_px, float)
        assert isinstance(y_px, float)
    
    def test_get_calibration_points(self):
        """Test retrieving calibration points"""
        calib = CourtCalibration()
        points = [(100, 100), (700, 100), (700, 500), (100, 500)]
        calib.calibrate(points)
        
        retrieved_points = calib.get_calibration_points()
        assert len(retrieved_points) == 4
        assert all(isinstance(p, tuple) and len(p) == 2 for p in retrieved_points)

