"""Direct test of overlay with game window"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from half_sword_ai.perception.vision import ScreenCapture
from half_sword_ai.perception.yolo_detector import YOLODetector
from half_sword_ai.monitoring.yolo_overlay import YOLOOverlay
from half_sword_ai.config import config

print("=" * 80)
print("Testing YOLO Overlay with Game Window")
print("=" * 80)

# Initialize components
print("\n1. Initializing screen capture...")
screen_capture = ScreenCapture()
print(f"   Game window found: {screen_capture.game_window_info is not None}")
if screen_capture.game_window_info:
    print(f"   Window: {screen_capture.game_window_info['title']} ({screen_capture.game_window_info['width']}x{screen_capture.game_window_info['height']})")

print("\n2. Initializing YOLO detector...")
detector = YOLODetector(
    model_path=config.YOLO_MODEL_PATH if config.YOLO_USE_CUSTOM_MODEL else None,
    confidence_threshold=0.15  # Lower threshold for testing
)
print(f"   Model loaded: {detector.model is not None}")

print("\n3. Initializing overlay...")
overlay = YOLOOverlay(window_name="Half Sword - YOLO Test", scale_factor=0.5)  # Smaller scale for testing
overlay.start()
print("   Overlay started")

print("\n4. Capturing frames and running detection...")
print("   Press 'q' in overlay window to stop")
print("=" * 80)

frame_count = 0
last_detection_time = 0

try:
    while True:
        # Get game window frame
        frame = screen_capture.get_game_window_frame()
        
        if frame is None:
            print("   No frame captured")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Run detection every 0.5 seconds
        current_time = time.time()
        if current_time - last_detection_time >= 0.5:
            print(f"\n   Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
            detections = detector.detect(frame, full_screen_frame=frame)
            
            obj_count = detections.get('count', 0)
            print(f"   Detections: {obj_count} objects")
            
            if obj_count > 0:
                print("   Found:")
                for obj in detections.get('objects', [])[:5]:  # Show first 5
                    print(f"     - {obj.get('class_name', 'Unknown')}: {obj.get('confidence', 0):.3f}")
            
            # Update overlay
            overlay.update(frame, detections)
            
            last_detection_time = current_time
        
        # Check if overlay window is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.033)  # ~30 FPS

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    overlay.stop()
    print("\nOverlay stopped")
    print("=" * 80)

