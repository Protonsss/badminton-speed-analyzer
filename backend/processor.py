"""
Video Processing Pipeline
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from detector import ShuttleDetector, MotionBasedDetector
from tracker import HybridTracker
from calibration import CourtCalibration
from speed_calculator import SpeedCalculator, format_speed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, 
                 court_type: str = 'singles',
                 use_yolo: bool = True,
                 yolo_model_path: Optional[str] = None):
        self.use_yolo = use_yolo
        
        # Initialize components
        if use_yolo:
            self.yolo_detector = ShuttleDetector(yolo_model_path)
        self.motion_detector = MotionBasedDetector()
        self.calibration = CourtCalibration(court_type)
        self.tracker = None  # Initialize after FPS known
        self.speed_calc = None
    
    def calibrate(self, calibration_points: List[Tuple[float, float]]):
        """
        Calibrate court
        
        Args:
            calibration_points: 4 corner points in pixel coordinates
        """
        self.calibration.calibrate(calibration_points)
        logger.info(f"Calibration complete for {self.calibration.court_type} court")
    
    def process_video(self, 
                      video_path: str,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None) -> Dict:
        """
        Process video and extract shuttle trajectory + speed
        
        Args:
            video_path: Path to video file
            start_frame: Frame to start processing
            end_frame: Frame to end processing (None = until end)
            
        Returns:
            Dict with trajectory, speeds, and analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {fps:.1f} fps, {total_frames} frames")
        
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
        relock_interval = 8  # Re-lock with YOLO every N frames
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Detect
            detection = None
            force_relock = (frame_idx - start_frame) % relock_interval == 0
            
            if self.use_yolo and force_relock:
                detections = self.yolo_detector.detect(frame)
                if detections:
                    # Use highest confidence detection
                    best_det = max(detections, key=lambda d: d[4])
                    detection = self.yolo_detector.get_center(best_det)
            
            # If YOLO failed or not used, try motion-based
            if detection is None and not self.use_yolo:
                detection = self.motion_detector.detect(frame)
            
            # Track
            tracked_pos = self.tracker.track(frame, detection, force_relock=force_relock)
            
            if tracked_pos is not None:
                x_px, y_px = tracked_pos
                positions_pixel.append((timestamp, x_px, y_px))
                
                # Convert to meters
                try:
                    x_m, y_m = self.calibration.pixel_to_meter(x_px, y_px)
                    positions_meter.append((timestamp, x_m, y_m))
                except RuntimeError:
                    logger.warning("Calibration not performed")
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx - start_frame}/{end_frame - start_frame} frames")
        
        cap.release()
        
        logger.info(f"Tracking complete: {len(positions_meter)} positions")
        
        # Analyze speeds
        if len(positions_meter) < 2:
            return {
                'success': False,
                'error': 'Insufficient tracking data',
                'trajectory_pixel': positions_pixel,
                'trajectory_meter': positions_meter
            }
        
        analysis = self.speed_calc.analyze(positions_meter)
        
        return {
            'success': True,
            'fps': fps,
            'trajectory_pixel': positions_pixel,
            'trajectory_meter': positions_meter,
            'peak_speed': format_speed(analysis['peak_speed']),
            'avg_speed': format_speed(analysis['avg_speed']),
            'initial_speed': format_speed(analysis['initial_speed']['v0']) if analysis['initial_speed'] else None,
            'speeds': analysis['speeds'],
            'times': analysis['times']
        }

