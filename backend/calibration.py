"""
Court Calibration and Homography
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict


# Official badminton court dimensions (meters)
COURT_DIMENSIONS = {
    'singles': {'length': 13.40, 'width': 5.18},
    'doubles': {'length': 13.40, 'width': 6.10}
}


class CourtCalibration:
    """Handle court calibration and pixel-to-meter conversion"""
    
    def __init__(self, court_type: str = 'singles'):
        self.court_type = court_type
        self.dimensions = COURT_DIMENSIONS[court_type]
        self.H = None  # Homography matrix
        self.H_inv = None  # Inverse homography
    
    def calibrate(self, src_points: List[Tuple[float, float]]):
        """
        Calibrate using 4 corner points
        
        Args:
            src_points: List of 4 (x, y) points in image coordinates (pixels)
                       Should be [top-left, top-right, bottom-right, bottom-left]
        """
        if len(src_points) != 4:
            raise ValueError("Need exactly 4 calibration points")
        
        # Validate and order points
        src_ordered = self._order_points(np.array(src_points, dtype=np.float32))
        
        # Check for collinearity
        if not self._validate_quad(src_ordered):
            raise ValueError("Calibration points are nearly collinear or degenerate")
        
        # Destination points in meters (court coordinate system)
        w, l = self.dimensions['width'], self.dimensions['length']
        dst_points = np.array([
            [0, 0],
            [w, 0],
            [w, l],
            [0, l]
        ], dtype=np.float32)
        
        # Compute homography
        self.H, _ = cv2.findHomography(src_ordered, dst_points, method=0)
        self.H_inv, _ = cv2.findHomography(dst_points, src_ordered, method=0)
    
    def pixel_to_meter(self, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to meters"""
        if self.H is None:
            raise RuntimeError("Calibration not performed")
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    def meter_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert meter coordinates to pixels"""
        if self.H_inv is None:
            raise RuntimeError("Calibration not performed")
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H_inv)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order points as [top-left, top-right, bottom-right, bottom-left]
        """
        # Sort by angle from centroid
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_pts = pts[np.argsort(angles)]
        
        # Find top-left (smallest y, then smallest x)
        top_left_idx = np.argmin(sorted_pts[:, 1] + sorted_pts[:, 0] * 0.1)
        
        # Rotate array to start at top-left
        ordered = np.roll(sorted_pts, -top_left_idx, axis=0)
        return ordered
    
    @staticmethod
    def _validate_quad(pts: np.ndarray) -> bool:
        """Check if 4 points form a valid quadrilateral"""
        # Check area
        area = 0.5 * abs(
            (pts[0][0] * pts[1][1] - pts[1][0] * pts[0][1]) +
            (pts[1][0] * pts[2][1] - pts[2][0] * pts[1][1]) +
            (pts[2][0] * pts[3][1] - pts[3][0] * pts[2][1]) +
            (pts[3][0] * pts[0][1] - pts[0][0] * pts[3][1])
        )
        
        if area < 1000:
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
                return False
        
        return True

