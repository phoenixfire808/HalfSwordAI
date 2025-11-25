"""
YOLO Detection Overlay - Live visualization of YOLO detections
Displays bounding boxes and labels in real-time on a transparent overlay window
"""
import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, Optional
from collections import deque
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

class YOLOOverlay:
    """
    Live overlay window displaying YOLO detection boxes
    Shows bounding boxes, labels, and confidence scores in real-time
    """
    
    def __init__(self, window_name: str = "YOLO Detections", scale_factor: float = 2.0):
        """
        Initialize YOLO overlay window
        
        Args:
            window_name: Name of the overlay window
            scale_factor: Scale factor for display (2.0 = 2x size for visibility)
        """
        self.window_name = window_name
        self.scale_factor = scale_factor
        self.is_running = False
        self.current_frame = None
        self.current_detections = {}
        self.lock = threading.Lock()
        
        # Class colors matching Half Sword v5 dataset
        self.class_colors = {
            'Blood': (0, 0, 255),      # Red
            'Enemy': (255, 0, 0),      # Blue (BGR format)
            'Player': (0, 255, 0),     # Green
            'You Won': (0, 255, 255),  # Yellow
        }
        
        # Default colors for unknown classes
        self.default_colors = [
            (255, 165, 0),  # Orange
            (255, 0, 255),  # Magenta
            (128, 0, 128),  # Purple
        ]
        
        self.overlay_thread = None
        
    def start(self):
        """Start the overlay window in a separate thread"""
        if self.is_running:
            logger.warning("Overlay already running")
            return
        
        self.is_running = True
        self.overlay_thread = threading.Thread(target=self._overlay_loop, daemon=True)
        self.overlay_thread.start()
        # Give the thread a moment to create the window
        time.sleep(0.2)
        logger.info(f"YOLO overlay started: {self.window_name}")
    
    def stop(self):
        """Stop the overlay window"""
        self.is_running = False
        if self.overlay_thread:
            self.overlay_thread.join(timeout=2.0)
        cv2.destroyWindow(self.window_name)
        logger.info("YOLO overlay stopped")
    
    def update(self, frame: np.ndarray, detections: Dict):
        """
        Update overlay with new frame and detections
        
        Args:
            frame: Current frame (can be grayscale or RGB)
            detections: Detection results from YOLODetector
        """
        with self.lock:
            self.current_frame = frame.copy() if frame is not None else None
            self.current_detections = detections.copy() if detections else {}
    
    def _overlay_loop(self):
        """Main overlay loop running in separate thread"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)  # Always on top
            logger.info(f"Overlay window '{self.window_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create overlay window: {e}", exc_info=True)
            self.is_running = False
            return
        
        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        
        while self.is_running:
            try:
                with self.lock:
                    frame = self.current_frame.copy() if self.current_frame is not None else None
                    detections = self.current_detections
                
                if frame is None:
                    # Create placeholder frame so window appears immediately
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for frames...", (10, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(self.window_name, placeholder)
                    cv2.waitKey(1)
                    time.sleep(0.1)  # Small sleep when waiting
                    continue
                
                # Log frame info occasionally (reduced frequency for high FPS)
                if frame_count % 180 == 0:  # Every 3 seconds at 60 FPS
                    logger.debug(f"Overlay: Frame shape={frame.shape}, dtype={frame.dtype}")
                
                # Convert frame to BGR for OpenCV display (cv2.imshow expects BGR)
                if len(frame.shape) == 2:
                    # Grayscale - convert to BGR
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 1:
                        # Single channel - convert to BGR
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 3:
                        # Assume it's RGB (from get_game_window_frame which converts BGRA->RGB)
                        # OpenCV imshow expects BGR, so convert RGB->BGR for display
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif frame.shape[2] == 4:
                        # BGRA - convert to BGR for OpenCV display
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        display_frame = frame.copy()
                else:
                    display_frame = frame.copy()
                
                # Skip uniform frame check for performance (only log occasionally)
                if frame_count % 180 == 0 and display_frame.max() == display_frame.min():
                    logger.warning(f"Overlay: Frame appears to be uniform (all {display_frame.max()})")
                
                # Scale up for better visibility (use LINEAR for better quality at high FPS)
                if self.scale_factor != 1.0:
                    height, width = display_frame.shape[:2]
                    new_width = int(width * self.scale_factor)
                    new_height = int(height * self.scale_factor)
                    # Use INTER_LINEAR for better quality (still fast)
                    display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    scale_x = self.scale_factor
                    scale_y = self.scale_factor
                else:
                    scale_x = 1.0
                    scale_y = 1.0
                
                # Draw detections
                display_frame = self._draw_detections(display_frame, detections, scale_x, scale_y)
                
                # Draw FPS
                frame_count += 1
                if time.time() - last_fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_fps_time = time.time()
                
                cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw detection count
                obj_count = len(detections.get('objects', []))
                cv2.putText(display_frame, f"Detections: {obj_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show window
                cv2.imshow(self.window_name, display_frame)
                # Process OpenCV events (required for window to respond)
                cv2.waitKey(1)
                
                # Handle window close (non-blocking check)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.is_running = False
                    break
                
                # No sleep - run as fast as possible to match game FPS
                # The frame updates are throttled by the actor process anyway
                
            except Exception as e:
                logger.error(f"Overlay error: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict, scale_x: float, scale_y: float) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Frame to draw on
            detections: Detection results
            scale_x: X scale factor
            scale_y: Y scale factor
            
        Returns:
            Frame with detections drawn
        """
        if not detections or 'objects' not in detections:
            return frame
        
        for obj in detections['objects']:
            # Get bounding box (scale if needed)
            bbox = obj.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Get class and confidence
            class_name = obj.get('class_name', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            
            # Get color for this class
            color = self.class_colors.get(class_name)
            if color is None:
                # Use default color based on hash
                color_idx = hash(class_name) % len(self.default_colors)
                color = self.default_colors[color_idx]
            
            # Draw bounding box
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Label background rectangle
            label_y = max(y1 - 5, label_height + 5)
            cv2.rectangle(frame,
                         (x1, label_y - label_height - 5),
                         (x1 + label_width + 5, label_y + baseline),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 2, label_y - 2),
                       font, font_scale, (255, 255, 255), font_thickness)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame

