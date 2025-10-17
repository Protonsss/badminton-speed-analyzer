"""
YOLOv8-based Shuttle Detector with PyTorch
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class ShuttleDetector:
    """Detect shuttlecock using YOLOv8"""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.3):
        """
        Initialize detector
        
        Args:
            model_path: Path to custom YOLOv8 model (if None, uses pretrained)
            conf_threshold: Confidence threshold for detections
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 model (use yolov8n for speed, yolov8x for accuracy)
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Start with nano model for fast inference
            self.model = YOLO('yolov8n.pt')
        
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect shuttlecock in frame
        
        Args:
            frame: BGR image array
            
        Returns:
            List of (x1, y1, x2, y2, confidence) bounding boxes
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        return detections
    
    def get_center(self, bbox: Tuple[int, int, int, int, float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2, _ = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class MotionBasedDetector:
    """Fallback detector using motion-based approach"""
    
    def __init__(self, min_size: int = 6, max_size: int = 40):
        self.min_size = min_size
        self.max_size = max_size
        self.prev_frame = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect shuttle using background subtraction
        
        Args:
            frame: BGR image
            
        Returns:
            (x, y) center coordinates or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
        
        # Background subtraction
        diff = cv2.absdiff(gray, self.prev_frame)
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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

