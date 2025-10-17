"""
API Route Handlers
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
import shutil
import torch

from app.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import (
    BadmintonAnalyzerException, ValidationError, CalibrationError,
    VideoProcessingError, InsufficientDataError
)
from app.core.processor import VideoProcessor
from app.api.schemas import (
    UploadResponse, CalibrationRequest, CalibrationResponse,
    AnalysisRequest, AnalysisResponse, ErrorResponse, HealthResponse
)
from app import __version__

logger = get_logger(__name__)
router = APIRouter()

# Global processor instance (will be initialized in lifespan)
_processor: Optional[VideoProcessor] = None


def get_processor() -> VideoProcessor:
    """Dependency to get processor instance"""
    if _processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    return _processor


def set_processor(processor: VideoProcessor):
    """Set global processor instance"""
    global _processor
    _processor = processor


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        cuda_available=torch.cuda.is_available()
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video file
    
    Returns temporary file path for processing
    """
    settings = get_settings()
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {settings.ALLOWED_VIDEO_FORMATS}"
        )
    
    try:
        # Create upload directory
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / f"upload_{file.filename}"
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            file_path.unlink()
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB"
            )
        
        logger.info(f"Uploaded file: {file.filename} ({file_size / (1024*1024):.1f} MB)")
        
        return UploadResponse(
            success=True,
            file_id=str(file_path),
            filename=file.filename,
            size_bytes=file_size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_court(
    request: CalibrationRequest,
    processor: VideoProcessor = Depends(get_processor)
):
    """
    Calibrate court with 4 corner points
    """
    try:
        points = [(p[0], p[1]) for p in request.points]
        
        # Reinitialize processor with correct court type
        settings = get_settings()
        new_processor = VideoProcessor(
            court_type=request.court_type.value,
            use_yolo=True
        )
        new_processor.calibrate(points)
        set_processor(new_processor)
        
        logger.info(f"Calibration successful: {request.court_type.value}")
        
        return CalibrationResponse(
            success=True,
            message=f"Calibration complete for {request.court_type.value} court",
            court_type=request.court_type.value,
            dimensions=new_processor.calibration.dimensions
        )
    
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except CalibrationError as e:
        logger.warning(f"Calibration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file_id: str = Form(...),
    use_yolo: bool = Form(True),
    start_frame: int = Form(0),
    end_frame: Optional[int] = Form(None),
    processor: VideoProcessor = Depends(get_processor)
):
    """
    Analyze video and compute shuttle speed
    """
    video_path = Path(file_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        logger.info(f"Starting analysis: {video_path}")
        
        result = processor.process_video(
            str(video_path),
            start_frame=start_frame,
            end_frame=end_frame
        )
        
        logger.info(
            f"Analysis complete: peak={result['peak_speed']['kmh']:.1f} km/h, "
            f"{result['frames_processed']} frames"
        )
        
        return AnalysisResponse(**result)
    
    except InsufficientDataError as e:
        logger.warning(f"Insufficient data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except VideoProcessingError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

