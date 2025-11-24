"""
Screen-Based Reward Detection
Inspired by ScrimBrain - Detects rewards from screen captures (text, UI elements, etc.)
"""
import cv2
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
import re

logger = logging.getLogger(__name__)

class ScreenRewardDetector:
    """
    Detects rewards from screen captures by looking for text patterns, UI elements, colors, etc.
    Similar to ScrimBrain's approach of detecting "SCORE" text for rewards
    """
    
    def __init__(self):
        self.reward_patterns = []
        self.last_reward_frame = None
        self.reward_history = []
        self.detection_count = 0
        
        # Common reward indicators to detect
        self.text_patterns = [
            # Victory/Score indicators
            {'pattern': r'SCORE', 'reward': 10.0, 'type': 'score'},
            {'pattern': r'VICTORY', 'reward': 50.0, 'type': 'victory'},
            {'pattern': r'WIN', 'reward': 50.0, 'type': 'victory'},
            {'pattern': r'KILL', 'reward': 25.0, 'type': 'kill'},
            {'pattern': r'HIT', 'reward': 5.0, 'type': 'hit'},
            # Health/stamina indicators
            {'pattern': r'HEAL', 'reward': 3.0, 'type': 'heal'},
            {'pattern': r'RESTORE', 'reward': 5.0, 'type': 'restore'},
            # Damage indicators
            {'pattern': r'DAMAGE', 'reward': 2.0, 'type': 'damage'},
            {'pattern': r'CRITICAL', 'reward': 15.0, 'type': 'critical'},
        ]
        
        # Color-based detection (detect specific colors that indicate rewards)
        self.color_patterns = [
            {'name': 'green_heal', 'color_range': ((40, 50, 50), (70, 255, 255)), 'reward': 3.0},  # Green = heal
            {'name': 'red_damage', 'color_range': ((0, 50, 50), (10, 255, 255)), 'reward': 2.0},  # Red = damage
            {'name': 'yellow_critical', 'color_range': ((20, 50, 50), (30, 255, 255)), 'reward': 15.0},  # Yellow = critical
        ]
        
        # Region-based detection (detect changes in specific screen regions)
        self.detection_regions = []
        
        logger.info("Screen reward detector initialized")
    
    def detect_rewards(self, frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> Dict:
        """
        Detect rewards from screen capture
        
        Args:
            frame: Current frame (grayscale or color)
            previous_frame: Previous frame for change detection
            
        Returns:
            Dictionary with reward information:
            {
                'total_reward': float,
                'rewards': List[Dict],
                'detected_patterns': List[str],
                'confidence': float
            }
        """
        self.detection_count += 1
        total_reward = 0.0
        detected_rewards = []
        detected_patterns = []
        
        try:
            # Text-based detection (using template matching or OCR-like techniques)
            text_rewards = self._detect_text_rewards(frame)
            total_reward += text_rewards['reward']
            detected_rewards.extend(text_rewards['matches'])
            detected_patterns.extend(text_rewards['patterns'])
            
            # Color-based detection
            color_rewards = self._detect_color_rewards(frame)
            total_reward += color_rewards['reward']
            detected_rewards.extend(color_rewards['matches'])
            
            # Change-based detection (detect significant changes that might indicate events)
            if previous_frame is not None:
                change_rewards = self._detect_change_rewards(frame, previous_frame)
                total_reward += change_rewards['reward']
                detected_rewards.extend(change_rewards['matches'])
            
            # Region-based detection (detect UI element changes)
            region_rewards = self._detect_region_rewards(frame)
            total_reward += region_rewards['reward']
            detected_rewards.extend(region_rewards['matches'])
            
        except Exception as e:
            logger.debug(f"Error detecting rewards: {e}")
        
        # Store detection result
        result = {
            'total_reward': total_reward,
            'rewards': detected_rewards,
            'detected_patterns': detected_patterns,
            'confidence': min(1.0, len(detected_patterns) / 5.0),  # Confidence based on pattern count
            'detection_count': self.detection_count
        }
        
        self.reward_history.append(result)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        # Store frame if reward detected
        if total_reward > 0:
            self.last_reward_frame = frame.copy()
        
        return result
    
    def _detect_text_rewards(self, frame: np.ndarray) -> Dict:
        """
        Detect text patterns in frame
        
        WARNING: This is a LIMITED implementation using edge detection.
        It does NOT actually read text - it only detects rectangular regions
        that might contain text. For real text detection, OCR should be used.
        
        This method has HIGH FALSE POSITIVE RATE and may not work reliably.
        """
        reward = 0.0
        matches = []
        patterns = []
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # LIMITATION: This uses edge detection to find text-like regions
            # It does NOT actually read the text content
            # For real text detection, use OCR (pytesseract) - see OCRRewardTracker
            
            # Detect high-contrast regions (often where text appears)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for rectangular regions (text blocks)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (text regions are typically small rectangles)
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / max(h, 1)
                area = cv2.contourArea(contour)
                
                # Text regions typically have:
                # - Aspect ratio > 1 (wider than tall)
                # - Small to medium area
                if 1.5 < aspect_ratio < 10 and 100 < area < 5000:
                    text_regions.append((x, y, w, h))
            
            # LIMITATION: We can't actually match text patterns without OCR
            # This just detects regions that might contain text
            # Simplified: check if region has high variance (indicating text)
            for x, y, w, h in text_regions[:10]:  # Limit to first 10 regions
                region = gray[y:y+h, x:x+w]
                if region.size > 0:
                    variance = np.var(region)
                    
                    # High variance indicates text (text has high contrast)
                    if variance > 1000:
                        # This is a potential text region
                        # In a real implementation, you'd use OCR here
                        # For now, we'll use pattern matching on region characteristics
                        
                        # Check if region characteristics match reward patterns
                        # Simplified approach: high contrast + specific location = likely reward text
                        if y < frame.shape[0] * 0.3:  # Upper portion of screen (where score often appears)
                            reward += 2.0
                            matches.append({
                                'type': 'text_region',
                                'reward': 2.0,
                                'location': (x, y, w, h),
                                'pattern': 'score_region'
                            })
                            patterns.append('score_region')
        
        except Exception as e:
            logger.debug(f"Error in text detection: {e}")
        
        return {
            'reward': reward,
            'matches': matches,
            'patterns': patterns
        }
    
    def _detect_color_rewards(self, frame: np.ndarray) -> Dict:
        """Detect reward-indicating colors in frame"""
        reward = 0.0
        matches = []
        
        try:
            # Convert to HSV for better color detection
            if len(frame.shape) == 2:
                # Grayscale, skip color detection
                return {'reward': 0.0, 'matches': []}
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            for color_pattern in self.color_patterns:
                name = color_pattern['name']
                lower, upper = color_pattern['color_range']
                color_reward = color_pattern['reward']
                
                # Create mask for color
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Count pixels of this color
                pixel_count = np.sum(mask > 0)
                total_pixels = frame.shape[0] * frame.shape[1]
                color_ratio = pixel_count / total_pixels
                
                # If significant portion of screen is this color, it's likely a reward indicator
                if color_ratio > 0.01:  # At least 1% of screen
                    reward += color_reward * min(1.0, color_ratio * 10)  # Scale by ratio
                    matches.append({
                        'type': 'color',
                        'name': name,
                        'reward': color_reward * min(1.0, color_ratio * 10),
                        'ratio': color_ratio
                    })
        
        except Exception as e:
            logger.debug(f"Error in color detection: {e}")
        
        return {
            'reward': reward,
            'matches': matches
        }
    
    def _detect_change_rewards(self, frame: np.ndarray, previous_frame: np.ndarray) -> Dict:
        """Detect rewards based on significant frame changes"""
        reward = 0.0
        matches = []
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) if len(previous_frame.shape) == 3 else previous_frame
            else:
                frame_gray = frame
                prev_gray = previous_frame
            
            # Calculate frame difference
            diff = cv2.absdiff(frame_gray, prev_gray)
            change_ratio = np.sum(diff > 30) / diff.size
            
            # Significant changes might indicate combat, hits, etc.
            if change_ratio > 0.1:  # More than 10% of screen changed
                # This could indicate combat or important event
                reward += change_ratio * 5.0  # Reward proportional to change
                matches.append({
                    'type': 'change',
                    'reward': change_ratio * 5.0,
                    'ratio': change_ratio
                })
        
        except Exception as e:
            logger.debug(f"Error in change detection: {e}")
        
        return {
            'reward': reward,
            'matches': matches
        }
    
    def _detect_region_rewards(self, frame: np.ndarray) -> Dict:
        """Detect rewards from specific screen regions (UI elements)"""
        reward = 0.0
        matches = []
        
        # This would detect changes in specific UI regions
        # For example: health bar changes, score changes, etc.
        # Implementation would depend on game-specific UI layout
        
        return {
            'reward': reward,
            'matches': matches
        }
    
    def add_pattern(self, pattern: str, reward: float, pattern_type: str = 'text'):
        """Add custom reward detection pattern"""
        self.text_patterns.append({
            'pattern': pattern,
            'reward': reward,
            'type': pattern_type
        })
        logger.info(f"Added reward pattern: {pattern} -> {reward}")
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        total_reward = sum(r['total_reward'] for r in self.reward_history[-100:])
        avg_reward = total_reward / max(len(self.reward_history[-100:]), 1)
        
        return {
            'detection_count': self.detection_count,
            'total_detections': len(self.reward_history),
            'avg_reward_per_detection': avg_reward,
            'total_reward_last_100': total_reward,
            'patterns_configured': len(self.text_patterns),
            'has_last_reward_frame': self.last_reward_frame is not None
        }

