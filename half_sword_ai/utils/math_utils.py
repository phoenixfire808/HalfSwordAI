"""
Math and Statistics Utilities
Provides mathematical operations and statistical calculations
"""
import numpy as np
from typing import List, Deque, Optional, Dict, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)

def moving_average(values: Union[List, Deque, np.ndarray], window: int = None) -> float:
    """
    Calculate moving average
    
    Args:
        values: Sequence of values
        window: Window size (defaults to all values)
    
    Returns:
        Moving average
    """
    if len(values) == 0:
        return 0.0
    
    if window is None:
        window = len(values)
    
    window = min(window, len(values))
    recent = list(values)[-window:]
    
    return np.mean(recent)

def exponential_moving_average(
    current_ema: float,
    new_value: float,
    alpha: float = 0.1
) -> float:
    """
    Calculate exponential moving average
    
    Args:
        current_ema: Current EMA value
        new_value: New value to incorporate
        alpha: Smoothing factor (0-1), higher = more responsive
    
    Returns:
        Updated EMA
    """
    return alpha * new_value + (1 - alpha) * current_ema

def calculate_stats(values: Union[List, Deque, np.ndarray]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a sequence
    
    Args:
        values: Sequence of numeric values
    
    Returns:
        Dictionary with statistics: mean, std, min, max, median, q25, q75
    """
    if len(values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'count': 0
        }
    
    arr = np.array(list(values))
    
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'count': len(arr)
    }

def normalize(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 0.0,
    target_max: float = 1.0
) -> float:
    """
    Normalize value to target range
    
    Args:
        value: Value to normalize
        min_val: Minimum value in source range
        max_val: Maximum value in source range
        target_min: Target minimum (default 0.0)
        target_max: Target maximum (default 1.0)
    
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return target_min
    
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)

def clip_value(
    value: float,
    min_val: float,
    max_val: float
) -> float:
    """
    Clip value to range
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))

def smooth_step(t: float) -> float:
    """
    Smooth step function (S-curve interpolation)
    
    Args:
        t: Input value (0-1)
    
    Returns:
        Smooth interpolated value
    """
    t = clip_value(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0-1)
    
    Returns:
        Interpolated value
    """
    return a + (b - a) * clip_value(t, 0.0, 1.0)

def calculate_percentile(
    values: Union[List, Deque, np.ndarray],
    percentile: float
) -> float:
    """
    Calculate percentile value
    
    Args:
        values: Sequence of values
        percentile: Percentile (0-100)
    
    Returns:
        Percentile value
    """
    if len(values) == 0:
        return 0.0
    
    arr = np.array(list(values))
    return float(np.percentile(arr, percentile))

def calculate_variance(values: Union[List, Deque, np.ndarray]) -> float:
    """
    Calculate variance
    
    Args:
        values: Sequence of values
    
    Returns:
        Variance
    """
    if len(values) == 0:
        return 0.0
    
    arr = np.array(list(values))
    return float(np.var(arr))

def calculate_std(values: Union[List, Deque, np.ndarray]) -> float:
    """
    Calculate standard deviation
    
    Args:
        values: Sequence of values
    
    Returns:
        Standard deviation
    """
    if len(values) == 0:
        return 0.0
    
    arr = np.array(list(values))
    return float(np.std(arr))

def calculate_median(values: Union[List, Deque, np.ndarray]) -> float:
    """
    Calculate median
    
    Args:
        values: Sequence of values
    
    Returns:
        Median value
    """
    if len(values) == 0:
        return 0.0
    
    arr = np.array(list(values))
    return float(np.median(arr))

def calculate_mode(values: Union[List, Deque, np.ndarray]) -> Optional[float]:
    """
    Calculate mode (most frequent value)
    
    Args:
        values: Sequence of values
    
    Returns:
        Mode value, or None if no unique mode
    """
    if len(values) == 0:
        return None
    
    arr = np.array(list(values))
    unique, counts = np.unique(arr, return_counts=True)
    max_count_idx = np.argmax(counts)
    
    # Check if mode is unique
    if np.sum(counts == counts[max_count_idx]) > 1:
        return None
    
    return float(unique[max_count_idx])

def calculate_correlation(x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
    """
    Calculate Pearson correlation coefficient
    
    Args:
        x: First sequence
        y: Second sequence
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    x_arr = np.array(list(x))
    y_arr = np.array(list(y))
    
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    
    return float(np.corrcoef(x_arr, y_arr)[0, 1])

def calculate_gradient(values: Union[List, Deque, np.ndarray]) -> float:
    """
    Calculate gradient (rate of change) of values
    
    Args:
        values: Sequence of values
    
    Returns:
        Average gradient
    """
    if len(values) < 2:
        return 0.0
    
    arr = np.array(list(values))
    diffs = np.diff(arr)
    return float(np.mean(diffs))




