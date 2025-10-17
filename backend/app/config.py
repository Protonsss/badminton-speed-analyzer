"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    APP_NAME: str = "Badminton Smash Speed Analyzer"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080", "*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_VIDEO_FORMATS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    UPLOAD_DIR: str = "/tmp/badminton_analyzer/uploads"
    RESULTS_DIR: str = "/tmp/badminton_analyzer/results"
    CLEANUP_INTERVAL: int = 3600  # Clean up old files every hour
    FILE_RETENTION: int = 7200  # Keep files for 2 hours
    
    # Model Settings
    YOLO_MODEL_PATH: Optional[str] = None  # If None, uses yolov8n.pt
    YOLO_CONF_THRESHOLD: float = 0.3
    YOLO_IOU_THRESHOLD: float = 0.4
    YOLO_DEVICE: str = "cuda"  # "cuda" or "cpu", auto-detected if cuda unavailable
    
    # Processing Settings
    DEFAULT_FPS: float = 60.0
    RELOCK_INTERVAL: int = 8  # Re-run YOLO every N frames
    MIN_TRACK_LENGTH: int = 5  # Minimum tracked points for valid analysis
    SMOOTH_WINDOW: int = 5  # Savitzky-Golay window size
    
    # Court Settings
    DEFAULT_COURT_TYPE: str = "singles"
    
    # Kalman Filter Settings
    KALMAN_MEASUREMENT_NOISE: float = 5.0
    KALMAN_PROCESS_NOISE: float = 0.1
    
    # Motion Detector Settings
    MOTION_MIN_SIZE: int = 6
    MOTION_MAX_SIZE: int = 40
    MOTION_THRESHOLD: int = 35
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None  # If None, logs to console only
    
    # Database (for future use)
    DATABASE_URL: Optional[str] = None
    
    # Redis (for future use - caching/job queue)
    REDIS_URL: Optional[str] = None
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Initialize directories
def init_directories():
    """Create necessary directories if they don't exist"""
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)

