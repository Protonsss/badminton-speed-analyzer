"""
Court Calibration and Homography
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict

from app.core.logging import get_logger
from app.core.exceptions import CalibrationError, ValidationError

logger = get_logger(__name__)


# Official badminton court dimensions (meters)
COURT_DIMENSIONS = {
    'singles': {'length': 13.40, 'width': 5.18},
    'doubles': {'length': 13.40, 'width': 6.10}
}


class CourtCalibration:
    """Handle court calibration and pixel-to-meter conversion"""
    
    def __init__(self, court_type: str = 'singles'):
        """
        Initialize court calibration
        
        Args:
            court_type: 'singles' or 'doubles'
            
        Raises:
            ValidationError: If court_type is invalid
        """
        if court_type not in COURT_DIMENSIONS:
            raise ValidationError(
                f"Invalid court type: {court_type}. Must be 'singles' or 'doubles'."
            )
        
        self.court_type = court_type
        self.dimensions = COURT_DIMENSIONS[court_type]
        self.H = None  # Homography matrix
        self.H_inv = None  # Inverse homography
        self._src_points = None  # Store original calibration points
        
        logger.info(
            f"Initialized CourtCalibration "
            f"({court_type}: {self.dimensions['length']}m × {self.dimensions['width']}m)"
        )
    
    def calibrate(self, src_points: List[Tuple[float, float]]):
        """
        Calibrate using 4 corner points
        
        Args:
            src_points: List of 4 (x, y) points in image coordinates (pixels)
                       Should be [top-left, top-right, bottom-right, bottom-left]
                       
        Raises:
            ValidationError: If points are invalid
            CalibrationError: If calibration computation fails
        """
        if len(src_points) != 4:
            raise ValidationError(f"Need exactly 4 calibration points, got {len(src_points)}")
        
        try:
            # Convert to numpy array
            src_array = np.array(src_points, dtype=np.float32)
            
            # Validate and order points
            src_ordered = self._order_points(src_array)
            
            # Check for collinearity
            if not self._validate_quad(src_ordered):
                raise ValidationError(
                    "Calibration points are nearly collinear or form a degenerate quadrilateral"
                )
            
            # Destination points in meters (court coordinate system)
            w, l = self.dimensions['width'], self.dimensions['length']
            dst_points = np.array([
                [0, 0],
                [w, 0],
                [w, l],
                [0, l]
            ], dtype=np.float32)
            
            # Compute homography
            self.H, status = cv2.findHomography(src_ordered, dst_points, method=0)
            self.H_inv, _ = cv2.findHomography(dst_points, src_ordered, method=0)
            
            if self.H is None or self.H_inv is None:
                raise CalibrationError("Failed to compute homography matrix")
            
            self._src_points = src_ordered
            
            logger.info(
                f"Calibration successful for {self.court_type} court. "
                f"Points: {src_ordered.tolist()}"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise CalibrationError(f"Calibration computation failed: {e}")
    
    def pixel_to_meter(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to meters
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            
        Returns:
            (x, y) in meters
            
        Raises:
            CalibrationError: If calibration not performed
        """
        if self.H is None:
            raise CalibrationError("Calibration not performed. Call calibrate() first.")
        
        try:
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.H)
            return (float(transformed[0][0][0]), float(transformed[0][0][1]))
        except Exception as e:
            logger.error(f"Pixel to meter conversion failed: {e}")
            raise CalibrationError(f"Coordinate transformation failed: {e}")
    
    def meter_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert meter coordinates to pixels
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            (x, y) in pixels
            
        Raises:
            CalibrationError: If calibration not performed
        """
        if self.H_inv is None:
            raise CalibrationError("Calibration not performed. Call calibrate() first.")
        
        try:
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.H_inv)
            return (float(transformed[0][0][0]), float(transformed[0][0][1]))
        except Exception as e:
            logger.error(f"Meter to pixel conversion failed: {e}")
            raise CalibrationError(f"Coordinate transformation failed: {e}")
    
    def is_calibrated(self) -> bool:
        """Check if calibration has been performed"""
        return self.H is not None
    
    def get_calibration_points(self) -> List[Tuple[float, float]]:
        """Get the calibration points used"""
        if self._src_points is None:
            return []
        return [(float(p[0]), float(p[1])) for p in self._src_points]
    
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order points as [top-left, top-right, bottom-right, bottom-left]
        
        Args:
            pts: 4x2 array of points
            
        Returns:
            Ordered 4x2 array
        """
        # Sort by angle from centroid
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_pts = pts[np.argsort(angles)]
        
        # Find top-left (smallest y, then smallest x)
        top_left_idx = np.argmin(sorted_pts[:, 1] + sorted_pts[:, 0] * 0.1)
        
        # Rotate array to start at top-left
        ordered = np.roll(sorted_pts, -top_left_idx, axis=0)
        
        logger.debug(f"Ordered calibration points: {ordered.tolist()}")
        return ordered
    
    @staticmethod
    def _validate_quad(pts: np.ndarray) -> bool:
        """
        Check if 4 points form a valid quadrilateral
        
        Args:
            pts: 4x2 array of points
            
        Returns:
            True if valid, False otherwise
        """
        # Check area
        area = 0.5 * abs(
            (pts[0][0] * pts[1][1] - pts[1][0] * pts[0][1]) +
            (pts[1][0] * pts[2][1] - pts[2][0] * pts[1][1]) +
            (pts[2][0] * pts[3][1] - pts[3][0] * pts[2][1]) +
            (pts[3][0] * pts[0][1] - pts[0][0] * pts[3][1])
        )
        
        if area < 1000:
            logger.warning(f"Quadrilateral area too small: {area}")
            return False
        
        # Check for near-collinearity via cross products
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            c = pts[(i + 2) % 4]
            v1 = b - a
            v2 = c - b
            cross = abs(np.cross(v1, v2))
            if cross < 0.01:
                logger.warning(f"Points are nearly collinear at index {i}")
                return False
        
        logger.debug(f"Quadrilateral validated (area={area:.1f})")
        return True
    
    def __repr__(self) -> str:
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        return f"CourtCalibration(type={self.court_type}, status={status})"

