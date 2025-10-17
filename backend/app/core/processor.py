"""
Video Processing Pipeline
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time

from app.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import VideoProcessingError, InsufficientDataError
from app.core.detector import ShuttleDetector, MotionBasedDetector
from app.core.tracker import HybridTracker
from app.core.calibration import CourtCalibration
from app.core.speed_calculator import SpeedCalculator, format_speed

logger = get_logger(__name__)


class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(
        self, 
        court_type: str = 'singles',
        use_yolo: bool = True,
        yolo_model_path: Optional[str] = None
    ):
        """
        Initialize video processor
        
        Args:
            court_type: 'singles' or 'doubles'
            use_yolo: Whether to use YOLO detector
            yolo_model_path: Path to custom YOLO model
        """
        settings = get_settings()
        self.use_yolo = use_yolo
        self.settings = settings
        
        # Initialize components
        if use_yolo:
            logger.info("Initializing YOLODetector...")
            self.yolo_detector = ShuttleDetector(yolo_model_path)
        else:
            self.yolo_detector = None
            
        self.motion_detector = MotionBasedDetector()
        self.calibration = CourtCalibration(court_type)
        self.tracker = None  # Initialize after FPS known
        self.speed_calc = None
        
        logger.info(f"VideoProcessor initialized (court_type={court_type}, use_yolo={use_yolo})")
    
    def calibrate(self, calibration_points: List[Tuple[float, float]]):
        """
        Calibrate court
        
        Args:
            calibration_points: 4 corner points in pixel coordinates
        """
        self.calibration.calibrate(calibration_points)
        logger.info(f"Calibration complete for {self.calibration.court_type} court")
    
    def process_video(
        self, 
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Process video and extract shuttle trajectory + speed
        
        Args:
            video_path: Path to video file
            start_frame: Frame to start processing
            end_frame: Frame to end processing (None = until end)
            progress_callback: Optional callback(frame_idx, total_frames)
            
        Returns:
            Dict with trajectory, speeds, and analysis results
            
        Raises:
            VideoProcessingError: If processing fails
            InsufficientDataError: If not enough tracking data
        """
        if not self.calibration.is_calibrated():
            raise VideoProcessingError("Court calibration required before processing")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(
                f"Processing video: {fps:.1f} fps, {total_frames} frames, "
                f"{width}×{height}, path={video_path}"
            )
            
            # Initialize tracker with video FPS
            self.tracker = HybridTracker(fps=fps)
            self.speed_calc = SpeedCalculator(fps=fps)
            
            # Seek to start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            if end_frame is None:
                end_frame = total_frames
            
            # Process frames
            positions_pixel = []  # [(t, x, y)] in pixels
            positions_meter = []  # [(t, x, y)] in meters
            frame_idx = start_frame
            relock_interval = self.settings.RELOCK_INTERVAL
            
            start_time = time.time()
            
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Detect
                detection = None
                force_relock = (frame_idx - start_frame) % relock_interval == 0
                
                if self.use_yolo and self.yolo_detector and force_relock:
                    try:
                        detections = self.yolo_detector.detect(frame)
                        if detections:
                            # Use highest confidence detection
                            best_det = max(detections, key=lambda d: d[4])
                            detection = self.yolo_detector.get_center(best_det)
                            logger.debug(f"Frame {frame_idx}: YOLO detection at {detection}")
                    except Exception as e:
                        logger.warning(f"YOLO detection failed: {e}")
                
                # If YOLO failed or not used, try motion-based
                if detection is None and not self.use_yolo:
                    try:
                        detection = self.motion_detector.detect(frame)
                    except Exception as e:
                        logger.warning(f"Motion detection failed: {e}")
                
                # Track
                try:
                    tracked_pos = self.tracker.track(frame, detection, force_relock=force_relock)
                except Exception as e:
                    logger.warning(f"Tracking failed at frame {frame_idx}: {e}")
                    tracked_pos = None
                
                if tracked_pos is not None:
                    x_px, y_px = tracked_pos
                    positions_pixel.append((timestamp, x_px, y_px))
                    
                    # Convert to meters
                    try:
                        x_m, y_m = self.calibration.pixel_to_meter(x_px, y_px)
                        positions_meter.append((timestamp, x_m, y_m))
                    except Exception as e:
                        logger.warning(f"Coordinate conversion failed: {e}")
                
                frame_idx += 1
                
                # Progress callback
                if progress_callback and frame_idx % 30 == 0:
                    progress_callback(frame_idx - start_frame, end_frame - start_frame)
                
                # Log progress
                if frame_idx % 60 == 0:
                    logger.info(
                        f"Processed {frame_idx - start_frame}/{end_frame - start_frame} frames "
                        f"({len(positions_meter)} tracked points)"
                    )
            
            processing_time = time.time() - start_time
            logger.info(
                f"Processing complete: {len(positions_meter)} positions in "
                f"{processing_time:.1f}s ({(end_frame - start_frame) / processing_time:.1f} fps)"
            )
            
            # Analyze speeds
            if len(positions_meter) < self.settings.MIN_TRACK_LENGTH:
                raise InsufficientDataError(
                    f"Insufficient tracking data: {len(positions_meter)} points "
                    f"(minimum {self.settings.MIN_TRACK_LENGTH})"
                )
            
            analysis = self.speed_calc.analyze(positions_meter)
            
            return {
                'success': True,
                'fps': fps,
                'frames_processed': frame_idx - start_frame,
                'processing_time': processing_time,
                'trajectory_pixel': positions_pixel,
                'trajectory_meter': positions_meter,
                'peak_speed': format_speed(analysis['peak_speed']),
                'avg_speed': format_speed(analysis['avg_speed']),
                'initial_speed': (
                    format_speed(analysis['initial_speed']['v0']) 
                    if analysis['initial_speed'] else None
                ),
                'initial_speed_fit': analysis['initial_speed'],
                'speeds': analysis['speeds'],
                'times': analysis['times']
            }
            
        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error(f"Video processing error: {e}", exc_info=True)
            raise VideoProcessingError(f"Failed to process video: {e}")
        finally:
            cap.release()
    
    def __repr__(self) -> str:
        return (
            f"VideoProcessor(court={self.calibration.court_type}, "
            f"yolo={self.use_yolo}, calibrated={self.calibration.is_calibrated()})"
        )

