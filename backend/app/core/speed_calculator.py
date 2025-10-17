"""
Speed Calculation with Physics-Based Fitting
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional

from app.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import InsufficientDataError

logger = get_logger(__name__)


class SpeedCalculator:
    """Calculate shuttle speed from tracked positions"""
    
    def __init__(self, fps: float = 60.0):
        """
        Initialize speed calculator
        
        Args:
            fps: Video frame rate
        """
        self.fps = fps
        self.dt = 1.0 / fps
        logger.info(f"Initialized SpeedCalculator (fps={fps})")
    
    def compute_speeds(self, positions: List[Tuple[float, float, float]]) -> List[float]:
        """
        Compute instantaneous speeds from positions
        
        Args:
            positions: List of (t, x, y) in meters and seconds
            
        Returns:
            List of speeds in m/s
            
        Raises:
            InsufficientDataError: If not enough positions
        """
        if len(positions) < 2:
            raise InsufficientDataError("Need at least 2 positions to compute speed")
        
        speeds = []
        for i in range(1, len(positions)):
            t0, x0, y0 = positions[i-1]
            t1, x1, y1 = positions[i]
            dt = t1 - t0
            
            if dt <= 0:
                speeds.append(0.0)
                continue
            
            dx = x1 - x0
            dy = y1 - y0
            dist = np.hypot(dx, dy)
            speed = dist / dt
            speeds.append(speed)
        
        logger.debug(f"Computed {len(speeds)} speed values")
        return speeds
    
    def smooth_speeds(
        self, 
        speeds: List[float], 
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Smooth speeds using Savitzky-Golay filter
        
        Args:
            speeds: Raw speed values
            window: Window size (must be odd, uses config default if None)
            
        Returns:
            Smoothed speeds
        """
        if window is None:
            settings = get_settings()
            window = settings.SMOOTH_WINDOW
        
        if len(speeds) < window:
            logger.warning(f"Not enough data points for smoothing (need {window}, got {len(speeds)})")
            return np.array(speeds)
        
        # Ensure odd window
        if window % 2 == 0:
            window += 1
        
        # Clamp window to data length
        window = min(window, len(speeds) if len(speeds) % 2 == 1 else len(speeds) - 1)
        
        if window < 3:
            return np.array(speeds)
        
        # Savitzky-Golay with polynomial order 3
        try:
            smoothed = savgol_filter(speeds, window, polyorder=min(3, window-1))
            logger.debug(f"Smoothed speeds with window={window}")
            return smoothed
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning raw speeds")
            return np.array(speeds)
    
    def fit_initial_speed(
        self, 
        positions: List[Tuple[float, float, float]]
    ) -> Optional[Dict]:
        """
        Fit quadratic drag model to estimate initial speed
        
        Model: dv/dt = -k*v^2  =>  v(t) = v0 / (1 + k*v0*t)
        
        Args:
            positions: List of (t, x, y) in meters
            
        Returns:
            Dict with v0 (initial speed), k (drag coefficient), fit_error, r_squared
        """
        if len(positions) < 5:
            logger.warning("Not enough positions for physics fit (need at least 5)")
            return None
        
        try:
            # Compute segment speeds
            speeds = self.compute_speeds(positions)
            if len(speeds) < 4:
                return None
            
            times = np.array([p[0] - positions[0][0] for p in positions[1:]])
            speeds_arr = np.array(speeds)
            
            # Initial guess
            v0_guess = np.max(speeds_arr)
            k_guess = 0.02
            
            def residual(params):
                v0, k = params
                predicted = v0 / (1 + k * v0 * times)
                return predicted - speeds_arr
            
            # Bounds: v0 > 0, k > 0
            result = least_squares(
                residual,
                [v0_guess, k_guess],
                bounds=([0.1, 0.001], [200, 0.2]),
                method='trf'
            )
            
            if result.success:
                v0, k = result.x
                fit_error = np.sqrt(np.mean(result.fun ** 2))
                
                # Calculate R²
                predicted = v0 / (1 + k * v0 * times)
                ss_res = np.sum((speeds_arr - predicted) ** 2)
                ss_tot = np.sum((speeds_arr - np.mean(speeds_arr)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                logger.info(
                    f"Physics fit: v0={v0:.2f} m/s, k={k:.4f}, "
                    f"RMSE={fit_error:.2f}, R²={r_squared:.3f}"
                )
                
                return {
                    'v0': float(v0),
                    'k': float(k),
                    'fit_error': float(fit_error),
                    'r_squared': float(r_squared)
                }
            else:
                logger.warning(f"Physics fit failed: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"Physics fit error: {e}")
            return None
    
    def analyze(
        self, 
        positions: List[Tuple[float, float, float]], 
        smooth_window: Optional[int] = None
    ) -> Dict:
        """
        Full analysis of trajectory
        
        Args:
            positions: List of (t, x, y) in meters
            smooth_window: Savitzky-Golay window size
            
        Returns:
            Dict with peak_speed, avg_speed, initial_speed_fit, speeds, times
            
        Raises:
            InsufficientDataError: If not enough data
        """
        if len(positions) < 2:
            raise InsufficientDataError(
                f"Need at least 2 positions for analysis, got {len(positions)}"
            )
        
        speeds = self.compute_speeds(positions)
        
        if len(speeds) == 0:
            return {
                'peak_speed': 0.0,
                'avg_speed': 0.0,
                'initial_speed': None,
                'speeds': [],
                'times': []
            }
        
        # Smooth
        smoothed = self.smooth_speeds(speeds, smooth_window)
        
        # Physics fit
        fit_result = self.fit_initial_speed(positions)
        
        peak_speed = float(np.max(smoothed))
        avg_speed = float(np.mean(smoothed))
        
        logger.info(
            f"Analysis complete: peak={peak_speed:.2f} m/s, "
            f"avg={avg_speed:.2f} m/s, {len(smoothed)} points"
        )
        
        return {
            'peak_speed': peak_speed,
            'avg_speed': avg_speed,
            'initial_speed': fit_result,
            'speeds': smoothed.tolist(),
            'times': [p[0] for p in positions[1:]]
        }


def format_speed(speed_mps: float) -> Dict[str, float]:
    """
    Convert m/s to multiple units
    
    Args:
        speed_mps: Speed in meters per second
        
    Returns:
        Dict with mps, kmh, mph
    """
    return {
        'mps': round(speed_mps, 2),
        'kmh': round(speed_mps * 3.6, 2),
        'mph': round(speed_mps * 2.23694, 2)
    }

