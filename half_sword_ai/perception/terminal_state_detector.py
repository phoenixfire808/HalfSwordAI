"""
Terminal State Detector - ScrimBrain Integration
Detects game over / death screen using visual analysis
Based on ScrimBrain's approach for episode termination
"""
import cv2
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TerminalStateDetector:
    """
    Detects terminal states (game over, death) from visual cues
    Uses color histogram analysis for fast detection
    """
    
    def __init__(self):
        self.red_threshold = 0.3  # Average red channel intensity threshold
        self.black_threshold = 10  # Average intensity for black screen
        self.death_overlay_threshold = 0.4  # Red overlay threshold
        
        # Detection history for stability
        self.detection_buffer = []
        self.buffer_size = 5
        self.confidence_threshold = 0.6  # Need 60% of frames to agree
    
    def detect_death_screen(self, frame: np.ndarray) -> Dict:
        """
        Detect death screen using color histogram analysis (ScrimBrain guide method)
        
        Args:
            frame: Current frame (grayscale or color)
            
        Returns:
            Dictionary with detection results:
            {
                'is_terminal': bool,
                'confidence': float,
                'reason': str,
                'red_intensity': float,
                'black_intensity': float
            }
        """
        if frame is None or frame.size == 0:
            return {
                'is_terminal': False,
                'confidence': 0.0,
                'reason': 'invalid_frame'
            }
        
        # Convert to color if grayscale (for histogram analysis)
        if len(frame.shape) == 2:
            # Grayscale - check for black screen (guide: np.mean(screen_region) < 10)
            mean_intensity = np.mean(frame)
            is_black = mean_intensity < self.black_threshold
            
            result = {
                'is_terminal': is_black,
                'confidence': 1.0 if is_black else 0.0,
                'reason': 'black_screen' if is_black else 'normal',
                'red_intensity': 0.0,
                'black_intensity': float(mean_intensity)
            }
        else:
            # Color image - use color histogram analysis (guide recommendation)
            # Convert to RGB if needed
            if frame.shape[2] == 4:  # BGRA
                import cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3:  # BGR
                import cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Extract red channel for histogram analysis
            red_channel = frame_rgb[:, :, 0]
            
            # Calculate color histogram statistics (guide method)
            mean_red = np.mean(red_channel)
            mean_intensity = np.mean(frame_rgb)
            
            # Check for black screen (guide: if np.mean(screen_region) < 10: done = True)
            is_black = mean_intensity < self.black_threshold
            
            # Check for red overlay (bloody screen effect)
            # Guide: Check if average "Red" channel intensity exceeds threshold
            is_red_overlay = mean_red > (255 * self.death_overlay_threshold)
            
            is_terminal = is_black or is_red_overlay
            confidence = 1.0 if (is_black or is_red_overlay) else 0.0
            reason = 'black_screen' if is_black else ('red_overlay' if is_red_overlay else 'normal')
            
            result = {
                'is_terminal': is_terminal,
                'confidence': confidence,
                'reason': reason,
                'red_intensity': float(mean_red),
                'black_intensity': float(mean_intensity)
            }
        
        # Add to detection buffer for stability
        self.detection_buffer.append(result['is_terminal'])
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        # Calculate confidence from buffer
        if len(self.detection_buffer) >= self.buffer_size:
            terminal_ratio = sum(self.detection_buffer) / len(self.detection_buffer)
            result['confidence'] = terminal_ratio
            result['is_terminal'] = terminal_ratio >= self.confidence_threshold
        
        return result
    
    def reset(self):
        """Reset detection buffer"""
        self.detection_buffer = []
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            'buffer_size': len(self.detection_buffer),
            'terminal_detections': sum(self.detection_buffer) if self.detection_buffer else 0,
            'confidence': sum(self.detection_buffer) / len(self.detection_buffer) if self.detection_buffer else 0.0
        }

