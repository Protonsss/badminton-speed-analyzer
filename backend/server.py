"""
FastAPI Server for Badminton Speed Analysis
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import shutil
from pathlib import Path
import logging

from processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Badminton Smash Speed Analyzer API")

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor: Optional[VideoProcessor] = None


class CalibrationRequest(BaseModel):
    points: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    court_type: str = 'singles'


class AnalysisRequest(BaseModel):
    use_yolo: bool = True
    start_frame: int = 0
    end_frame: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup"""
    global processor
    logger.info("Initializing VideoProcessor...")
    processor = VideoProcessor(use_yolo=True)
    logger.info("Server ready")


@app.get("/")
async def root():
    return {"message": "Badminton Smash Speed Analyzer API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video file
    
    Returns temporary file path for processing
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Save to temporary file
    try:
        temp_dir = Path(tempfile.gettempdir()) / "badminton_analyzer"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / f"upload_{file.filename}"
        
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded video saved to {temp_file}")
        
        return {
            "success": True,
            "file_id": str(temp_file),
            "filename": file.filename
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibrate")
async def calibrate(request: CalibrationRequest):
    """
    Calibrate court with 4 corner points
    """
    global processor
    
    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    try:
        points = [(p[0], p[1]) for p in request.points]
        
        if len(points) != 4:
            raise HTTPException(status_code=400, detail="Need exactly 4 calibration points")
        
        # Reinitialize processor with correct court type
        processor = VideoProcessor(court_type=request.court_type, use_yolo=True)
        processor.calibrate(points)
        
        return {
            "success": True,
            "message": f"Calibration complete for {request.court_type} court"
        }
    
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def analyze_video(
    file_id: str = Form(...),
    use_yolo: bool = Form(True),
    start_frame: int = Form(0),
    end_frame: Optional[int] = Form(None)
):
    """
    Analyze video and compute shuttle speed
    """
    global processor
    
    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    video_path = Path(file_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        logger.info(f"Processing video: {video_path}")
        
        result = processor.process_video(
            str(video_path),
            start_frame=start_frame,
            end_frame=end_frame
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

