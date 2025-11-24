"""
YOLOv8 Object Detection Module
Real-time object detection for game elements (enemies, weapons, player, etc.)
"""
import os
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not available - install with: pip install ultralytics")

class YOLODetector:
    """
    YOLOv8-based object detector for game elements
    Detects enemies, weapons, player, health indicators, etc.
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize enhanced YOLO detector with temporal tracking
        
        Args:
            model_path: Path to custom YOLOv8 model (or None for pretrained)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.detection_history = deque(maxlen=100)  # Track last 100 detections
        self.last_detection_time = 0
        self.detection_count = 0
        self.total_inference_time = 0.0
        
        # Temporal tracking
        self.tracked_objects = {}  # Track objects across frames
        self.track_id_counter = 0
        self.max_track_age = 5  # Frames before track is lost
        
        # Multi-scale detection
        self.multi_scale_enabled = True
        self.scale_factors = [0.8, 1.0, 1.2]  # Different scales to try
        
        # Game-specific classes (can be customized)
        self.class_names = {
            0: 'enemy',
            1: 'weapon',
            2: 'player',
            3: 'health_indicator',
            4: 'stamina_indicator',
            5: 'attack_indicator',
            6: 'defense_indicator'
        }
        
        if YOLO_AVAILABLE:
            self._init_model()
        else:
            logger.warning("YOLO detector disabled - ultralytics not installed")
    
    def _init_model(self):
        """Initialize YOLO model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading custom YOLO model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained YOLOv8n (nano - fastest) or YOLOv8s (small - balanced)
                logger.info("Loading pretrained YOLOv8n model (will need custom training for game-specific objects)")
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                logger.warning("Using generic YOLOv8 model - train custom model for game-specific detection")
            
            # Set device
            device = 'cuda' if config.DEVICE == 'cuda' else 'cpu'
            self.model.to(device)
            logger.info(f"YOLO model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray, use_tracking: bool = True) -> Dict[str, any]:
        """
        Enhanced detection with multi-scale and temporal tracking
        
        Args:
            frame: Input frame (grayscale or RGB)
            use_tracking: Whether to use temporal tracking
            
        Returns:
            Dictionary with detections, bounding boxes, and metadata
        """
        if self.model is None:
            return self._get_empty_detection()
        
        try:
            # Convert grayscale to RGB if needed
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame.copy()
            
            # Multi-scale detection
            all_detections = []
            inference_start = time.time()
            
            if self.multi_scale_enabled and len(self.scale_factors) > 1:
                # Try multiple scales and combine results
                for scale in self.scale_factors:
                    h, w = frame_rgb.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_frame = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    results = self.model(scaled_frame, conf=self.confidence_threshold, verbose=False)
                    detections = self._parse_results(results[0], scaled_frame.shape)
                    
                    # Scale bounding boxes back to original size
                    for obj in detections.get('objects', []):
                        bbox = obj['bbox']
                        obj['bbox'] = [
                            bbox[0] / scale,
                            bbox[1] / scale,
                            bbox[2] / scale,
                            bbox[3] / scale
                        ]
                        obj['center'] = [obj['center'][0] / scale, obj['center'][1] / scale]
                    
                    all_detections.extend(detections.get('objects', []))
            else:
                # Single scale detection
                results = self.model(frame_rgb, conf=self.confidence_threshold, verbose=False)
                detections = self._parse_results(results[0], frame.shape)
                all_detections = detections.get('objects', [])
            
            inference_time = time.time() - inference_start
            
            # Remove duplicates (same object detected at multiple scales)
            all_detections = self._remove_duplicate_detections(all_detections)
            
            # Reconstruct detections dict
            detections = {
                'objects': all_detections,
                'enemies': [obj for obj in all_detections if obj.get('class_name') in ['person', 'enemy'] or obj.get('class_id') == 0],
                'weapons': [obj for obj in all_detections if 'weapon' in obj.get('class_name', '').lower()],
                'player': next((obj for obj in all_detections if 'player' in obj.get('class_name', '').lower()), None),
                'count': len(all_detections),
                'frame_shape': frame.shape,
                'inference_time': inference_time,
                'timestamp': time.time()
            }
            
            # Temporal tracking
            if use_tracking:
                detections = self._apply_temporal_tracking(detections)
            
            self.total_inference_time += inference_time
            self.detection_count += 1
            
            self.detection_history.append(detections)
            self.last_detection_time = time.time()
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}", exc_info=True)
            return self._get_empty_detection()
    
    def _remove_duplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections using IoU"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        for det in sorted_detections:
            is_duplicate = False
            for existing in filtered:
                iou = self._calculate_iou(det['bbox'], existing['bbox'])
                same_class = det['class_id'] == existing['class_id']
                
                if iou > iou_threshold and same_class:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_temporal_tracking(self, detections: Dict) -> Dict:
        """Apply temporal tracking to detections"""
        current_time = time.time()
        current_objects = detections.get('objects', [])
        
        # Update tracked objects
        for obj in current_objects:
            center = obj['center']
            best_match_id = None
            best_match_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.tracked_objects.items():
                if track['class_id'] != obj['class_id']:
                    continue
                
                last_center = track['last_center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < best_match_distance and distance < 50:  # Max distance threshold
                    best_match_distance = distance
                    best_match_id = track_id
            
            # Update or create track
            if best_match_id is not None:
                track = self.tracked_objects[best_match_id]
                track['last_center'] = center
                track['last_seen'] = current_time
                track['age'] = 0
                obj['track_id'] = best_match_id
                obj['track_age'] = track['age']
            else:
                # Create new track
                track_id = self.track_id_counter
                self.track_id_counter += 1
                self.tracked_objects[track_id] = {
                    'class_id': obj['class_id'],
                    'last_center': center,
                    'last_seen': current_time,
                    'age': 0
                }
                obj['track_id'] = track_id
                obj['track_age'] = 0
        
        # Age and remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracked_objects.items():
            track['age'] += 1
            if track['age'] > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        detections['tracked_objects'] = len(self.tracked_objects)
        return detections
    
    def _parse_results(self, result, frame_shape: Tuple[int, ...]) -> Dict[str, any]:
        """Parse YOLO results into structured format"""
        detections = {
            'objects': [],
            'enemies': [],
            'weapons': [],
            'player': None,
            'count': 0,
            'frame_shape': frame_shape
        }
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = result.names.get(class_id, f'class_{class_id}')
            
            # Calculate center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float(center_x), float(center_y)],
                'size': [float(width), float(height)],
                'area': float(width * height)
            }
            
            detections['objects'].append(detection)
            detections['count'] += 1
            
            # Categorize detections
            if 'person' in class_name.lower() or class_id in [0]:  # COCO person class or custom enemy
                detections['enemies'].append(detection)
            elif 'weapon' in class_name.lower() or 'sword' in class_name.lower():
                detections['weapons'].append(detection)
            elif 'player' in class_name.lower():
                detections['player'] = detection
        
        return detections
    
    def _get_empty_detection(self) -> Dict[str, any]:
        """Return empty detection structure"""
        return {
            'objects': [],
            'enemies': [],
            'weapons': [],
            'player': None,
            'count': 0,
            'inference_time': 0.0,
            'timestamp': time.time()
        }
    
    def get_nearest_enemy(self, detections: Dict, frame_center: Tuple[int, int] = None) -> Optional[Dict]:
        """
        Get nearest enemy to frame center or specified point
        
        Args:
            detections: Detection results
            frame_center: Center point (x, y) - if None, uses frame center
            
        Returns:
            Nearest enemy detection or None
        """
        if not detections.get('enemies'):
            return None
        
        if frame_center is None:
            frame_shape = detections.get('frame_shape', (96, 96))
            frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
        
        nearest = None
        min_distance = float('inf')
        
        for enemy in detections['enemies']:
            center = enemy['center']
            distance = np.sqrt((center[0] - frame_center[0])**2 + (center[1] - frame_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest = enemy
        
        if nearest:
            nearest['distance_to_center'] = min_distance
        
        return nearest
    
    def get_enemy_direction(self, enemy: Dict, frame_center: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate direction vector from frame center to enemy
        
        Args:
            enemy: Enemy detection
            frame_center: Frame center point
            
        Returns:
            Normalized direction vector (dx, dy)
        """
        if not enemy:
            return (0.0, 0.0)
        
        center = enemy['center']
        dx = center[0] - frame_center[0]
        dy = center[1] - frame_center[1]
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return (dx, dy)
    
    def get_threat_level(self, detections: Dict) -> str:
        """
        Assess threat level based on detections
        
        Returns:
            'high', 'medium', or 'low'
        """
        enemy_count = len(detections.get('enemies', []))
        weapon_count = len(detections.get('weapons', []))
        
        if enemy_count > 2:
            return 'high'
        elif enemy_count > 0 or weapon_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def visualize_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw detections on frame for visualization
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            Frame with bounding boxes drawn
        """
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            vis_frame = frame.copy()
        
        for obj in detections.get('objects', []):
            x1, y1, x2, y2 = map(int, obj['bbox'])
            confidence = obj['confidence']
            class_name = obj['class_name']
            
            # Color based on class
            if obj in detections.get('enemies', []):
                color = (0, 0, 255)  # Red for enemies
            elif obj in detections.get('weapons', []):
                color = (255, 165, 0)  # Orange for weapons
            else:
                color = (0, 255, 0)  # Green for others
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        avg_inference_time = (self.total_inference_time / self.detection_count 
                              if self.detection_count > 0 else 0.0)
        
        return {
            'detection_count': self.detection_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'last_detection_time': self.last_detection_time,
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold
        }
    
    def train_custom_model(self, dataset_path: str, epochs: int = 100):
        """
        Train custom YOLOv8 model on game-specific data
        
        Args:
            dataset_path: Path to YOLO format dataset
            epochs: Number of training epochs
        """
        if not YOLO_AVAILABLE:
            logger.error("Cannot train - ultralytics not installed")
            return
        
        try:
            logger.info(f"Starting YOLOv8 training on {dataset_path}")
            model = YOLO('yolov8n.pt')  # Start from pretrained
            
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=config.DEVICE,
                project='yolo_training',
                name='half_sword_detector'
            )
            
            logger.info(f"Training complete! Model saved to {results.save_dir}")
            return results.save_dir
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return None

