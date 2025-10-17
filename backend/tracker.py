"""
Optical Flow + Kalman Filter Tracker
"""
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, Tuple, List


class OpticalFlowTracker:
    """Lucas-Kanade optical flow tracker"""
    
    def __init__(self):
        # LK parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_point = None
    
    def track(self, frame: np.ndarray, initial_point: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
        """
        Track point using optical flow
        
        Args:
            frame: Current BGR frame
            initial_point: (x, y) to start tracking, or None to continue previous
            
        Returns:
            (x, y) tracked point or None if tracking failed
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if initial_point is not None:
            self.prev_point = np.array([[initial_point]], dtype=np.float32)
            self.prev_gray = gray
            return initial_point
        
        if self.prev_gray is None or self.prev_point is None:
            return None
        
        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_point, None, **self.lk_params
        )
        
        # Update
        self.prev_gray = gray
        
        if status[0][0] == 1 and err[0][0] < 25:
            self.prev_point = next_pts
            return (float(next_pts[0][0][0]), float(next_pts[0][0][1]))
        else:
            # Tracking failed
            self.prev_point = None
            return None


class KalmanTracker:
    """
    2D Kalman filter with constant velocity model
    State: [x, y, vx, vy]
    """
    
    def __init__(self, dt: float = 1/60.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = dt
        self.initialized = False
        
        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R = np.eye(2) * 5.0
        
        # Process noise
        q = 0.1
        self.kf.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*10, 0],
            [0, 0, 0, q*10]
        ])
        
        # Initial covariance
        self.kf.P *= 10
    
    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Initialize filter with position and velocity"""
        self.kf.x = np.array([x, y, vx, vy])
        self.initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state"""
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized")
        
        self.kf.predict()
        return (float(self.kf.x[0]), float(self.kf.x[1]))
    
    def update(self, measurement: Tuple[float, float]):
        """Update filter with measurement"""
        self.kf.update(np.array(measurement))
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current state [x, y, vx, vy]"""
        return (float(self.kf.x[0]), float(self.kf.x[1]), 
                float(self.kf.x[2]), float(self.kf.x[3]))


class HybridTracker:
    """
    Hybrid tracker combining:
    - YOLO detection for re-lock
    - Optical flow for frame-to-frame tracking
    - Kalman filter for smoothing
    """
    
    def __init__(self, fps: float = 60.0):
        self.optical_flow = OpticalFlowTracker()
        self.kalman = KalmanTracker(dt=1.0/fps)
        self.track_initialized = False
    
    def track(self, 
              frame: np.ndarray, 
              detection: Optional[Tuple[float, float]] = None,
              force_relock: bool = False) -> Optional[Tuple[float, float]]:
        """
        Track shuttle in frame
        
        Args:
            frame: Current BGR frame
            detection: YOLO detection center (x, y) if available
            force_relock: Force re-initialization with detection
            
        Returns:
            Filtered (x, y) position or None
        """
        # If we have a detection and need to (re)initialize
        if detection is not None and (force_relock or not self.track_initialized):
            self.optical_flow.track(frame, initial_point=detection)
            if not self.track_initialized:
                self.kalman.init(detection[0], detection[1])
                self.track_initialized = True
            else:
                # Update Kalman with detection
                self.kalman.update(detection)
            return detection
        
        # Try optical flow tracking
        flow_result = self.optical_flow.track(frame)
        
        if flow_result is not None:
            # Predict + update Kalman
            self.kalman.predict()
            self.kalman.update(flow_result)
            x, y, _, _ = self.kalman.get_state()
            return (x, y)
        elif detection is not None:
            # Flow failed but we have detection - use it
            self.optical_flow.track(frame, initial_point=detection)
            self.kalman.update(detection)
            x, y, _, _ = self.kalman.get_state()
            return (x, y)
        else:
            # No tracking possible
            return None

