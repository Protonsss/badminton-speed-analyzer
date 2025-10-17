import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import DetectionError, ModelLoadError

logger = get_logger(__name__)


class ShuttleDetector:
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        settings = get_settings()
        
        # Set device
        if device is None:
            device = settings.YOLO_DEVICE if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Set confidence threshold
        self.conf_threshold = conf_threshold or settings.YOLO_CONF_THRESHOLD
        
        logger.info(f"Initializing ShuttleDetector on device: {self.device}")
        
        # Load YOLOv8 model
        try:
            model_path = model_path or settings.YOLO_MODEL_PATH
            if model_path and Path(model_path).exists():
                logger.info(f"Loading custom YOLO model from: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info("Loading pretrained YOLOv8n model")
                self.model = YOLO('yolov8n.pt')
            
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect shuttlecock in frame
        
        Args:
            frame: BGR image array
            
        Returns:
            List of (x1, y1, x2, y2, confidence) bounding boxes
            
        Raises:
            DetectionError: If detection fails
        """
        if frame is None or frame.size == 0:
            raise DetectionError("Invalid frame provided")
        
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append((
                            int(x1), int(y1), int(x2), int(y2), float(conf)
                        ))
            
            logger.debug(f"Detected {len(detections)} shuttlecocks")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise DetectionError(f"Detection failed: {e}")
    
    def get_center(self, bbox: Tuple[int, int, int, int, float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2, _ = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def __repr__(self) -> str:
        return f"ShuttleDetector(device={self.device}, conf_threshold={self.conf_threshold})"


class MotionBasedDetector:
    """Fallback detector using motion-based approach"""
    
    def __init__(
        self, 
        min_size: Optional[int] = None, 
        max_size: Optional[int] = None,
        threshold: Optional[int] = None
    ):
        """
        Initialize motion detector
        
        Args:
            min_size: Minimum shuttle size in pixels
            max_size: Maximum shuttle size in pixels
            threshold: Motion detection threshold
        """
        settings = get_settings()
        
        self.min_size = min_size or settings.MOTION_MIN_SIZE
        self.max_size = max_size or settings.MOTION_MAX_SIZE
        self.threshold = threshold or settings.MOTION_THRESHOLD
        
        self.prev_frame = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        logger.info(
            f"Initialized MotionBasedDetector "
            f"(size: {self.min_size}-{self.max_size}, threshold: {self.threshold})"
        )
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect shuttle using background subtraction
        
        Args:
            frame: BGR image
            
        Returns:
            (x, y) center coordinates or None
            
        Raises:
            DetectionError: If frame processing fails
        """
        if frame is None or frame.size == 0:
            raise DetectionError("Invalid frame provided")
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return None
            
            # Background subtraction
            diff = cv2.absdiff(gray, self.prev_frame)
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            best_candidate = None
            best_score = -1
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Size filtering
                if area < self.min_size ** 2 or area > self.max_size ** 2:
                    continue
                
                # Score by intensity + area (shuttle is bright)
                cx, cy = x + w // 2, y + h // 2
                intensity = gray[cy, cx]
                score = intensity + min(4000, area)
                
                if score > best_score:
                    best_score = score
                    best_candidate = (float(cx), float(cy))
            
            self.prev_frame = gray
            return best_candidate
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            raise DetectionError(f"Motion detection failed: {e}")
    
    def reset(self):
        """Reset detector state"""
        self.prev_frame = None
        logger.debug("Motion detector reset")
    
    def __repr__(self) -> str:
        return f"MotionBasedDetector(size: {self.min_size}-{self.max_size})"

