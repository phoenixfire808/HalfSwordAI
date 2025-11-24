"""
Time Utilities
Provides timing functions, rate limiting, and duration formatting
"""
import time
import logging
from typing import Callable, Optional
from collections import deque

logger = logging.getLogger(__name__)

def get_timestamp() -> float:
    """
    Get current timestamp
    
    Returns:
        Current time as float (seconds since epoch)
    """
    return time.time()

def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        precision: Decimal precision for seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45.67s")
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs:.{precision}f}s")
    
    return " ".join(parts)

def rate_limiter(min_interval: float):
    """
    Create a rate limiter decorator
    
    Args:
        min_interval: Minimum time between calls (seconds)
    
    Returns:
        Decorator function
    """
    last_call = [0.0]
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_call[0]
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            last_call[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

class Timer:
    """
    Context manager for timing code blocks
    """
    
    def __init__(self, name: str = "Operation", logger_instance: Optional[logging.Logger] = None):
        """
        Initialize timer
        
        Args:
            name: Name of the operation being timed
            logger_instance: Optional logger to log timing (defaults to module logger)
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.logger = logger_instance or logger
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.elapsed()
        
        if self.logger:
            self.logger.debug(f"{self.name} took {format_duration(elapsed)}")
        
        return False
    
    def elapsed(self) -> float:
        """
        Get elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def reset(self):
        """Reset timer"""
        self.start_time = time.time()
        self.end_time = None

class IntervalChecker:
    """
    Check if enough time has passed since last check
    Useful for periodic operations
    """
    
    def __init__(self, interval: float):
        """
        Initialize interval checker
        
        Args:
            interval: Minimum interval between checks (seconds)
        """
        self.interval = interval
        self.last_check = 0.0
    
    def should_run(self) -> bool:
        """
        Check if interval has elapsed
        
        Returns:
            True if interval has passed, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_check >= self.interval:
            self.last_check = current_time
            return True
        return False
    
    def reset(self):
        """Reset last check time"""
        self.last_check = time.time()
    
    def time_until_next(self) -> float:
        """
        Get time remaining until next check
        
        Returns:
            Seconds until next check
        """
        elapsed = time.time() - self.last_check
        remaining = self.interval - elapsed
        return max(0.0, remaining)

class FPSCounter:
    """
    Calculate FPS from frame timestamps
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
    
    def tick(self) -> Optional[float]:
        """
        Record a frame and return current FPS
        
        Returns:
            Current FPS, or None if not enough frames
        """
        self.frame_times.append(time.time())
        
        if len(self.frame_times) < 2:
            return None
        
        # Calculate FPS from time difference
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            return (len(self.frame_times) - 1) / time_span
        
        return None
    
    def get_fps(self) -> Optional[float]:
        """
        Get current FPS without recording a frame
        
        Returns:
            Current FPS, or None if not enough frames
        """
        if len(self.frame_times) < 2:
            return None
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            return (len(self.frame_times) - 1) / time_span
        
        return None
    
    def reset(self):
        """Reset FPS counter"""
        self.frame_times.clear()

