"""
Test YOLO Detection - Debug script to see what's happening
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import logging
from half_sword_ai.perception.yolo_detector import YOLODetector
from half_sword_ai.config import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_detection():
    """Test YOLO detection with a sample frame"""
    print("=" * 80)
    print("YOLO Detection Test")
    print("=" * 80)
    
    # Initialize detector
    detector = YOLODetector(
        model_path=config.YOLO_MODEL_PATH if config.YOLO_USE_CUSTOM_MODEL else None,
        confidence_threshold=0.25  # Lower threshold for testing
    )
    
    if detector.model is None:
        print("ERROR: Model not loaded!")
        return
    
    print(f"Model loaded: {detector.model is not None}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"Classes: {detector.class_names}")
    print()
    
    # Try to capture a frame from screen
    print("Attempting to capture frame...")
    try:
        from half_sword_ai.perception.vision import ScreenCapture
        screen_capture = ScreenCapture()
        frame = screen_capture.get_latest_frame()
        
        if frame is None:
            print("ERROR: Could not capture frame!")
            print("Make sure the game is running and visible on screen")
            return
        
        print(f"Frame captured: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame min/max: {frame.min()}/{frame.max()}")
        print()
        
        # Get full screen frame for better detection
        print("Getting full screen frame for YOLO detection...")
        full_screen_frame = screen_capture.get_full_screen_frame()
        if full_screen_frame is not None:
            print(f"Full screen frame: {full_screen_frame.shape}")
        else:
            print("WARNING: Could not get full screen frame, using small frame")
        
        # Run detection
        print("Running YOLO detection...")
        detections = detector.detect(frame, full_screen_frame=full_screen_frame)
        
        print(f"Detection results:")
        print(f"  Objects found: {detections.get('count', 0)}")
        print(f"  Inference time: {detections.get('inference_time', 0)*1000:.2f}ms")
        print()
        
        if detections.get('objects'):
            print("Detections:")
            for i, obj in enumerate(detections['objects']):
                print(f"  {i+1}. {obj.get('class_name', 'Unknown')}: "
                      f"confidence={obj.get('confidence', 0):.3f}, "
                      f"bbox={obj.get('bbox', [])}")
        else:
            print("No detections found!")
            print()
            print("Possible issues:")
            print("  1. Confidence threshold too high (current: {})".format(detector.confidence_threshold))
            print("  2. Frame size mismatch (captured: {}, model trained on: 512x512)".format(frame.shape))
            print("  3. No objects matching classes in frame")
            print("  4. Model needs more training data")
            print()
            print("Trying with lower confidence threshold...")
            detector.confidence_threshold = 0.1
            detections = detector.detect(frame)
            print(f"  With 0.1 threshold: {detections.get('count', 0)} objects")
            
            detector.confidence_threshold = 0.05
            detections = detector.detect(frame)
            print(f"  With 0.05 threshold: {detections.get('count', 0)} objects")
        
        # Visualize
        if detections.get('objects'):
            vis_frame = detector.visualize_detections(frame, detections)
            cv2.imshow("YOLO Test - Detections", vis_frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection()

