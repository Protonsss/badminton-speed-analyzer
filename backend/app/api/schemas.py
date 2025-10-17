"""
API Request/Response Schemas
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple
from enum import Enum


class CourtType(str, Enum):
    """Court type enum"""
    singles = "singles"
    doubles = "doubles"


class UploadResponse(BaseModel):
    """Response for video upload"""
    success: bool
    file_id: str
    filename: str
    size_bytes: int


class CalibrationRequest(BaseModel):
    """Request for court calibration"""
    points: List[List[float]] = Field(..., description="4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    court_type: CourtType = Field(default=CourtType.singles, description="Court type")
    
    @validator('points')
    def validate_points(cls, v):
        if len(v) != 4:
            raise ValueError("Must provide exactly 4 calibration points")
        for point in v:
            if len(point) != 2:
                raise ValueError("Each point must have exactly 2 coordinates [x, y]")
        return v


class CalibrationResponse(BaseModel):
    """Response for calibration"""
    success: bool
    message: str
    court_type: str
    dimensions: dict


class AnalysisRequest(BaseModel):
    """Request for video analysis"""
    file_id: str = Field(..., description="File ID from upload")
    use_yolo: bool = Field(default=True, description="Use YOLO detector")
    start_frame: int = Field(default=0, ge=0, description="Start frame index")
    end_frame: Optional[int] = Field(default=None, ge=0, description="End frame index (None = end of video)")


class SpeedData(BaseModel):
    """Speed in multiple units"""
    mps: float = Field(..., description="Speed in meters per second")
    kmh: float = Field(..., description="Speed in kilometers per hour")
    mph: float = Field(..., description="Speed in miles per hour")


class InitialSpeedFit(BaseModel):
    """Physics-based initial speed fit"""
    v0: float = Field(..., description="Initial speed (m/s)")
    k: float = Field(..., description="Drag coefficient")
    fit_error: float = Field(..., description="Root mean square error")
    r_squared: float = Field(..., description="R² coefficient of determination")


class AnalysisResponse(BaseModel):
    """Response for video analysis"""
    success: bool
    fps: float
    frames_processed: int
    processing_time: float
    trajectory_pixel: List[Tuple[float, float, float]]
    trajectory_meter: List[Tuple[float, float, float]]
    peak_speed: SpeedData
    avg_speed: SpeedData
    initial_speed: Optional[SpeedData]
    initial_speed_fit: Optional[InitialSpeedFit]
    speeds: List[float]
    times: List[float]


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    cuda_available: bool

