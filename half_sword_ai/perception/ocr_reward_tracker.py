"""
OCR Reward Tracker - ScrimBrain Integration
Tracks score from Abyss mode using Optical Character Recognition
Based on ScrimBrain's approach for reward shaping
"""
import cv2
import numpy as np
import logging
import time
from typing import Dict, Optional, Tuple
from collections import deque
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - OCR disabled")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class OCRRewardTracker:
    """
    Tracks score from screen using OCR
    Optimized for Abyss mode score counter
    """
    
    def __init__(self, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize OCR reward tracker
        
        Args:
            roi: Region of Interest (top, left, width, height) for score counter
                 If None, will use default position (upper right)
        """
        self.roi = roi or (50, 1700, 100, 50)  # Default: upper right area
        self.last_score = 0
        self.current_score = 0
        self.score_history = deque(maxlen=100)
        self.ocr_interval = config.OCR_INTERVAL if hasattr(config, 'OCR_INTERVAL') else 60  # Run OCR every N frames (optimized)
        self.frame_count = 0
        self.last_ocr_time = 0
        self.ocr_min_interval = 0.5  # Minimum seconds between OCR calls
        
        # OCR engine selection
        self.use_easyocr = EASYOCR_AVAILABLE
        self.easyocr_reader = None
        
        if TESSERACT_AVAILABLE:
            logger.info("OCR initialized with Tesseract")
        elif EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("OCR initialized with EasyOCR")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.use_easyocr = False
        else:
            logger.warning("No OCR library available - score tracking disabled")
    
    def extract_score(self, frame: np.ndarray) -> Optional[int]:
        """
        Extract score from frame using OCR
        
        Args:
            frame: Full screen frame (grayscale or color)
            
        Returns:
            Score value or None if extraction failed
        """
        if not TESSERACT_AVAILABLE and not self.use_easyocr:
            return None
        
        # Extract ROI
        top, left, width, height = self.roi
        
        # Validate ROI bounds
        if top < 0 or left < 0 or top + height > frame.shape[0] or left + width > frame.shape[1]:
            logger.debug(f"OCR ROI out of bounds: roi={self.roi}, frame_shape={frame.shape}")
            return None
        
        if len(frame.shape) == 3:
            # Color image
            roi = frame[top:top+height, left:left+width]
            if roi.size == 0:
                logger.debug("OCR ROI is empty (color)")
                return None
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            # Grayscale
            roi = frame[top:top+height, left:left+width]
            if roi.size == 0:
                logger.debug("OCR ROI is empty (grayscale)")
                return None
            gray = roi
        
        # Check if gray is empty
        if gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
            logger.debug("OCR gray image is empty")
            return None
        
        # Preprocess for OCR
        # Threshold to binary (black/white) for better OCR accuracy
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Check if binary is empty before dilation
        if binary.size == 0 or binary.shape[0] == 0 or binary.shape[1] == 0:
            logger.debug("OCR binary image is empty")
            return None
        
        # Dilate to thicken text (helps with stylized fonts)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Try OCR
        try:
            if self.use_easyocr and self.easyocr_reader:
                # EasyOCR
                results = self.easyocr_reader.readtext(binary)
                if results:
                    # Extract first result (usually the score)
                    text = results[0][1]  # Text content
                    # Extract numbers only
                    numbers = ''.join(filter(str.isdigit, text))
                    if numbers:
                        return int(numbers)
            elif TESSERACT_AVAILABLE:
                # Tesseract OCR
                # Configure for single line, numbers only
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(binary, config=custom_config)
                # Extract numbers
                numbers = ''.join(filter(str.isdigit, text))
                if numbers:
                    return int(numbers)
        except Exception as e:
            logger.debug(f"OCR extraction error: {e}")
        
        return None
    
    def update(self, frame: np.ndarray) -> Dict:
        """
        Update score tracking
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary with reward information:
            {
                'score': int,
                'score_delta': int,
                'reward': float,
                'success': bool
            }
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Only run OCR at specified interval
        if (self.frame_count % self.ocr_interval == 0 and 
            current_time - self.last_ocr_time >= self.ocr_min_interval):
            
            score = self.extract_score(frame)
            self.last_ocr_time = current_time
            
            if score is not None:
                self.last_score = self.current_score
                self.current_score = score
                self.score_history.append(score)
                
                # Calculate reward (score increase)
                score_delta = self.current_score - self.last_score
                reward = max(0, score_delta)  # Only positive rewards
                
                return {
                    'score': self.current_score,
                    'score_delta': score_delta,
                    'reward': float(reward),
                    'success': True
                }
        
        # Return cached values if OCR not run
        return {
            'score': self.current_score,
            'score_delta': 0,
            'reward': 0.0,
            'success': False
        }
    
    def get_current_score(self) -> int:
        """Get current tracked score"""
        return self.current_score
    
    def get_score_delta(self) -> int:
        """Get score change since last update"""
        return self.current_score - self.last_score
    
    def set_roi(self, roi: Tuple[int, int, int, int]):
        """Set Region of Interest for score counter"""
        self.roi = roi
        logger.info(f"OCR ROI updated: {roi}")
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        return {
            'current_score': self.current_score,
            'last_score': self.last_score,
            'score_delta': self.current_score - self.last_score,
            'ocr_enabled': TESSERACT_AVAILABLE or self.use_easyocr,
            'roi': self.roi,
            'frame_count': self.frame_count,
            'history_size': len(self.score_history)
        }

