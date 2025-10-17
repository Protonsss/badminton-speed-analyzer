"""
Custom Exception Classes
"""
from typing import Any, Optional


class BadmintonAnalyzerException(Exception):
    """Base exception for all application errors"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BadmintonAnalyzerException):
    """Raised when input validation fails"""
    pass


class CalibrationError(BadmintonAnalyzerException):
    """Raised when court calibration fails"""
    pass


class DetectionError(BadmintonAnalyzerException):
    """Raised when shuttle detection fails"""
    pass


class TrackingError(BadmintonAnalyzerException):
    """Raised when tracking fails"""
    pass


class VideoProcessingError(BadmintonAnalyzerException):
    """Raised when video processing fails"""
    pass


class ModelLoadError(BadmintonAnalyzerException):
    """Raised when ML model loading fails"""
    pass


class FileUploadError(BadmintonAnalyzerException):
    """Raised when file upload fails"""
    pass


class InsufficientDataError(BadmintonAnalyzerException):
    """Raised when there's not enough tracking data"""
    pass


class ConfigurationError(BadmintonAnalyzerException):
    """Raised when configuration is invalid"""
    pass

