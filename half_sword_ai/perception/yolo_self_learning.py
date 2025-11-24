"""
YOLO Self-Learning System
Self-labeling, action-response learning, and reward-based improvement
"""
import numpy as np
import cv2
import time
import json
import os
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

class YOLOSelfLearner:
    """
    Self-learning YOLO system that:
    1. Self-labels detections based on action outcomes
    2. Learns from action-response pairs
    3. Uses rewards to improve detection confidence
    """
    
    def __init__(self, yolo_detector):
        self.yolo_detector = yolo_detector
        self.action_detection_pairs = deque(maxlen=5000)  # Track detection → action → reward
        self.self_labeled_data = deque(maxlen=10000)  # Self-labeled training data
        self.detection_rewards = {}  # Track rewards per detection type
        self.confidence_adjustments = {}  # Dynamic confidence adjustments
        
        # Self-labeling parameters
        self.min_reward_for_labeling = config.YOLO_MIN_REWARD_FOR_LABELING
        self.confidence_boost_per_reward = 0.01  # Boost confidence per positive reward
        self.confidence_penalty_per_negative = 0.02  # Penalty for negative rewards
        
        # Active learning
        self.active_learning_enabled = True
        self.uncertainty_threshold = 0.3  # Label uncertain detections
        self.uncertainty_samples = deque(maxlen=1000)  # Track uncertain samples
        
        # Reward integration
        self.reward_history = deque(maxlen=1000)  # Track reward trends
        self.reward_weight_decay = 0.95  # Decay old rewards
        self.adaptive_labeling = True  # Adjust labeling threshold based on performance
        
        # Training data storage
        self.training_data_path = os.path.join(config.DATA_SAVE_PATH, "yolo_self_training")
        os.makedirs(self.training_data_path, exist_ok=True)
        
        # Statistics
        self.total_labels_created = 0
        self.positive_labels = 0
        self.negative_labels = 0
        self.uncertainty_labels = 0
        self.training_iterations = 0
        
    def record_detection_action_pair(self, frame: np.ndarray, detections: Dict,
                                    action: np.ndarray, reward: float,
                                    game_state: Dict, timestamp: float = None):
        """
        Enhanced recording with active learning and adaptive labeling
        
        Args:
            frame: Game frame
            detections: YOLO detection results
            action: Action taken based on detections
            reward: Reward received after action
            game_state: Game state at time of action
            timestamp: Action timestamp
        """
        timestamp = timestamp or time.time()
        
        pair = {
            'timestamp': timestamp,
            'frame': frame.copy(),
            'detections': detections.copy(),
            'action': action.copy() if isinstance(action, np.ndarray) else action,
            'reward': reward,
            'game_state': game_state.copy(),
            'labeled': False,
            'uncertainty': self._calculate_uncertainty(detections)
        }
        
        self.action_detection_pairs.append(pair)
        self.reward_history.append(reward)
        
        # Update detection rewards with decay
        self._update_detection_rewards(detections, reward)
        
        # Active learning: label uncertain detections
        if self.active_learning_enabled and pair['uncertainty'] > self.uncertainty_threshold:
            self.uncertainty_samples.append(pair)
            self._create_uncertainty_label(pair)
        
        # Adaptive labeling threshold
        if self.adaptive_labeling:
            self._update_labeling_threshold()
        
        # Self-label if reward is significant
        effective_threshold = self.min_reward_for_labeling
        if abs(reward) > effective_threshold:
            self._create_self_label(pair)
    
    def _calculate_uncertainty(self, detections: Dict) -> float:
        """Calculate uncertainty score for detections (0-1)"""
        if not detections.get('objects'):
            return 1.0  # High uncertainty if no detections
        
        confidences = [obj['confidence'] for obj in detections.get('objects', [])]
        
        if not confidences:
            return 1.0
        
        # Uncertainty is inverse of average confidence
        avg_confidence = np.mean(confidences)
        uncertainty = 1.0 - avg_confidence
        
        # Also consider detection count (few detections = more uncertain)
        detection_count = len(confidences)
        if detection_count < 2:
            uncertainty = min(1.0, uncertainty + 0.2)
        
        return uncertainty
    
    def _create_uncertainty_label(self, pair: Dict):
        """Create label for uncertain detection (active learning)"""
        frame = pair['frame']
        detections = pair['detections']
        uncertainty = pair['uncertainty']
        
        # Label uncertain detections for manual review or automatic correction
        labels = []
        for obj in detections.get('objects', []):
            confidence = obj['confidence']
            
            # Low confidence detections are uncertain
            if confidence < 0.6:
                label = {
                    'class_id': self._get_class_id(obj['class_name']),
                    'class_name': obj['class_name'],
                    'bbox_normalized': self._normalize_bbox(obj['bbox'], frame.shape),
                    'bbox_pixels': obj['bbox'],
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'label_type': 'uncertain',
                    'timestamp': pair['timestamp']
                }
                labels.append(label)
        
        if labels:
            labeled_data = {
                'frame': frame,
                'labels': labels,
                'reward': pair['reward'],
                'game_state': pair['game_state'],
                'action': pair['action'],
                'uncertainty': uncertainty
            }
            
            self.self_labeled_data.append(labeled_data)
            self.uncertainty_labels += len(labels)
            self.total_labels_created += len(labels)
    
    def _normalize_bbox(self, bbox: List[float], frame_shape: Tuple) -> List[float]:
        """Normalize bounding box to YOLO format"""
        frame_h, frame_w = frame_shape[:2]
        x_center = (bbox[0] + bbox[2]) / 2.0 / frame_w
        y_center = (bbox[1] + bbox[3]) / 2.0 / frame_h
        width = (bbox[2] - bbox[0]) / frame_w
        height = (bbox[3] - bbox[1]) / frame_h
        return [x_center, y_center, width, height]
    
    def _update_labeling_threshold(self):
        """Adaptively update labeling threshold based on performance"""
        if len(self.reward_history) < 100:
            return
        
        # Calculate recent reward trend
        recent_rewards = list(self.reward_history)[-100:]
        avg_reward = np.mean(recent_rewards)
        
        # If rewards are improving, be more selective
        # If rewards are degrading, be less selective
        if avg_reward > 0.5:
            # Good performance - only label high-impact samples
            self.min_reward_for_labeling = max(0.3, config.YOLO_MIN_REWARD_FOR_LABELING * 1.2)
        elif avg_reward < -0.5:
            # Poor performance - label more samples
            self.min_reward_for_labeling = max(0.1, config.YOLO_MIN_REWARD_FOR_LABELING * 0.8)
        else:
            # Normal performance
            self.min_reward_for_labeling = config.YOLO_MIN_REWARD_FOR_LABELING
    
    def _update_detection_rewards(self, detections: Dict, reward: float):
        """Update reward statistics with decay for each detection type"""
        # Apply decay to existing rewards
        for class_name in self.detection_rewards:
            stats = self.detection_rewards[class_name]
            stats['total_reward'] *= self.reward_weight_decay
        
        for obj in detections.get('objects', []):
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            if class_name not in self.detection_rewards:
                self.detection_rewards[class_name] = {
                    'total_reward': 0.0,
                    'count': 0,
                    'avg_reward': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'weighted_reward': 0.0  # Decay-weighted reward
                }
            
            stats = self.detection_rewards[class_name]
            
            # Weight reward by confidence (higher confidence = more weight)
            weighted_reward = reward * confidence
            stats['total_reward'] += weighted_reward
            stats['count'] += 1
            stats['avg_reward'] = stats['total_reward'] / stats['count']
            stats['weighted_reward'] = stats['weighted_reward'] * self.reward_weight_decay + weighted_reward
            
            if reward > 0:
                stats['positive_count'] += 1
            elif reward < 0:
                stats['negative_count'] += 1
            
            # Adjust confidence threshold based on rewards (with momentum)
            if class_name not in self.confidence_adjustments:
                self.confidence_adjustments[class_name] = 0.0
            
            # Boost confidence if detection led to positive reward
            adjustment = 0.0
            if reward > 0:
                adjustment = self.confidence_boost_per_reward * confidence
            elif reward < -1.0:  # Significant negative reward
                adjustment = -self.confidence_penalty_per_negative * (1.0 - confidence)
            
            # Apply with momentum (smooth changes)
            momentum = 0.7
            self.confidence_adjustments[class_name] = (
                momentum * self.confidence_adjustments[class_name] + 
                (1 - momentum) * adjustment
            )
            
            # Clamp adjustments
            self.confidence_adjustments[class_name] = max(-0.3, min(0.3, 
                self.confidence_adjustments[class_name]))
    
    def _create_self_label(self, pair: Dict):
        """Create self-label from detection-action-reward pair"""
        frame = pair['frame']
        detections = pair['detections']
        reward = pair['reward']
        
        # Only label if reward is significant
        if abs(reward) < self.min_reward_for_labeling:
            return
        
        # Create labels for each detection
        labels = []
        for obj in detections.get('objects', []):
            class_name = obj['class_name']
            bbox = obj['bbox']
            confidence = obj['confidence']
            
            # Adjust label confidence based on reward
            # Positive reward = good detection, increase confidence
            # Negative reward = bad detection, decrease confidence
            adjusted_confidence = confidence
            
            if reward > 0:
                # Positive reward - this was a good detection
                adjusted_confidence = min(1.0, confidence + abs(reward) * 0.1)
                label_type = 'positive'
            else:
                # Negative reward - this might be a false positive
                adjusted_confidence = max(0.1, confidence - abs(reward) * 0.1)
                label_type = 'negative'
            
            # Create YOLO format label (normalized coordinates)
            frame_h, frame_w = frame.shape[:2]
            x_center = (bbox[0] + bbox[2]) / 2.0 / frame_w
            y_center = (bbox[1] + bbox[3]) / 2.0 / frame_h
            width = (bbox[2] - bbox[0]) / frame_w
            height = (bbox[3] - bbox[1]) / frame_h
            
            label = {
                'class_id': self._get_class_id(class_name),
                'class_name': class_name,
                'bbox_normalized': [x_center, y_center, width, height],
                'bbox_pixels': bbox,
                'confidence': adjusted_confidence,
                'original_confidence': confidence,
                'reward': reward,
                'label_type': label_type,
                'timestamp': pair['timestamp']
            }
            
            labels.append(label)
        
        if labels:
            # Store self-labeled data
            labeled_data = {
                'frame': frame,
                'labels': labels,
                'reward': reward,
                'game_state': pair['game_state'],
                'action': pair['action']
            }
            
            self.self_labeled_data.append(labeled_data)
            self.total_labels_created += len(labels)
            
            if reward > 0:
                self.positive_labels += len(labels)
            else:
                self.negative_labels += len(labels)
            
            pair['labeled'] = True
    
    def _get_class_id(self, class_name: str) -> int:
        """Get class ID for YOLO format (map class names to IDs)"""
        class_map = {
            'person': 0,  # Enemy
            'enemy': 0,
            'weapon': 1,
            'sword': 1,
            'player': 2,
            'health_indicator': 3,
            'stamina_indicator': 4
        }
        
        class_name_lower = class_name.lower()
        for key, class_id in class_map.items():
            if key in class_name_lower:
                return class_id
        
        return 0  # Default to enemy
    
    def adjust_detection_confidence(self, detections: Dict) -> Dict:
        """
        Adjust detection confidence based on learned rewards
        
        Args:
            detections: Original detections
            
        Returns:
            Detections with adjusted confidence
        """
        adjusted_detections = detections.copy()
        
        for obj in adjusted_detections.get('objects', []):
            class_name = obj['class_name']
            
            # Apply confidence adjustment based on reward history
            if class_name in self.confidence_adjustments:
                adjustment = self.confidence_adjustments[class_name]
                original_confidence = obj['confidence']
                
                # Adjust confidence
                new_confidence = original_confidence + adjustment
                new_confidence = max(0.0, min(1.0, new_confidence))
                
                obj['confidence'] = new_confidence
                obj['original_confidence'] = original_confidence
                obj['confidence_adjustment'] = adjustment
        
        return adjusted_detections
    
    def get_action_guidance_from_detections(self, detections: Dict, 
                                          previous_rewards: List[float] = None) -> Dict:
        """
        Get action guidance based on detections and reward history
        
        Args:
            detections: Current detections
            previous_rewards: Recent reward history
            
        Returns:
            Dictionary with action guidance
        """
        guidance = {
            'target_direction': (0.0, 0.0),
            'action_priority': 'neutral',
            'confidence': 0.5
        }
        
        # Find best target based on reward history
        enemies = detections.get('enemies', [])
        if not enemies:
            return guidance
        
        # Score each enemy based on detection quality and reward history
        best_enemy = None
        best_score = -float('inf')
        
        for enemy in enemies:
            class_name = enemy.get('class_name', 'enemy')
            confidence = enemy['confidence']
            
            # Get reward history for this class
            reward_score = 0.0
            if class_name in self.detection_rewards:
                stats = self.detection_rewards[class_name]
                reward_score = stats['avg_reward']
            
            # Combined score: confidence + reward history
            score = confidence * 0.6 + reward_score * 0.4
            
            if score > best_score:
                best_score = score
                best_enemy = enemy
        
        if best_enemy:
            # Calculate direction to best target
            frame_center = (detections.get('frame_shape', (96, 96))[1] // 2,
                           detections.get('frame_shape', (96, 96))[0] // 2)
            center = best_enemy['center']
            
            dx = center[0] - frame_center[0]
            dy = center[1] - frame_center[1]
            
            # Normalize
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > 0:
                dx /= magnitude
                dy /= magnitude
            
            guidance['target_direction'] = (dx, dy)
            guidance['action_priority'] = 'attack' if best_score > 0.5 else 'defend'
            guidance['confidence'] = best_score
            guidance['target_enemy'] = best_enemy
        
        return guidance
    
    def save_training_data(self, min_reward_threshold: float = 0.3):
        """
        Save self-labeled training data for YOLO fine-tuning
        
        Args:
            min_reward_threshold: Minimum reward to include in training set
        """
        if len(self.self_labeled_data) == 0:
            logger.warning("No self-labeled data to save")
            return
        
        # Filter by reward threshold
        training_data = [d for d in self.self_labeled_data 
                        if abs(d['reward']) >= min_reward_threshold]
        
        if len(training_data) == 0:
            logger.warning("No data above reward threshold")
            return
        
        # Save in YOLO format
        images_dir = os.path.join(self.training_data_path, "images")
        labels_dir = os.path.join(self.training_data_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        saved_count = 0
        for i, data in enumerate(training_data):
            try:
                # Save image
                image_path = os.path.join(images_dir, f"self_labeled_{i:06d}.jpg")
                cv2.imwrite(image_path, data['frame'])
                
                # Save labels (YOLO format: class_id x_center y_center width height)
                label_path = os.path.join(labels_dir, f"self_labeled_{i:06d}.txt")
                with open(label_path, 'w') as f:
                    for label in data['labels']:
                        bbox = label['bbox_normalized']
                        f.write(f"{label['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} "
                               f"{bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save training data {i}: {e}")
        
        logger.info(f"💾 Saved {saved_count} self-labeled training samples")
        return saved_count
    
    def train_on_self_labels(self, epochs: int = 10):
        """
        Fine-tune YOLO model on self-labeled data
        
        Args:
            epochs: Number of training epochs
        """
        if not YOLO_AVAILABLE or self.yolo_detector.model is None:
            logger.error("Cannot train - YOLO not available")
            return
        
        if len(self.self_labeled_data) < 10:
            logger.warning(f"Not enough self-labeled data ({len(self.self_labeled_data)}). Need at least 10 samples.")
            return
        
        # Save training data first
        saved_count = self.save_training_data()
        if saved_count < 10:
            logger.warning("Not enough training data saved")
            return
        
        try:
            logger.info(f"🔄 Starting YOLO self-training on {saved_count} self-labeled samples...")
            
            # Create dataset config
            dataset_config = {
                'path': self.training_data_path,
                'train': 'images',
                'val': 'images',  # Use same for validation (small dataset)
                'nc': 6,  # Number of classes
                'names': ['enemy', 'weapon', 'player', 'health_indicator', 'stamina_indicator', 'attack_indicator']
            }
            
            dataset_yaml = os.path.join(self.training_data_path, "dataset.yaml")
            try:
                import yaml
            except ImportError:
                # Fallback to manual YAML writing
                with open(dataset_yaml, 'w') as f:
                    f.write(f"path: {self.training_data_path}\n")
                    f.write("train: images\n")
                    f.write("val: images\n")
                    f.write("nc: 6\n")
                    f.write("names: ['enemy', 'weapon', 'player', 'health_indicator', 'stamina_indicator', 'attack_indicator']\n")
            else:
                with open(dataset_yaml, 'w') as f:
                    yaml.dump(dataset_config, f)
            
            # Fine-tune model
            model = self.yolo_detector.model
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=640,
                batch=min(8, saved_count // 2),  # Small batch for limited data
                device=config.DEVICE,
                project='yolo_self_training',
                name='reward_learned_detector',
                resume=False
            )
            
            self.training_iterations += 1
            logger.info(f"✅ YOLO self-training complete! Model improved from rewards. (Iteration {self.training_iterations})")
            return results.save_dir
            
        except Exception as e:
            logger.error(f"Self-training error: {e}", exc_info=True)
            return None
    
    def get_stats(self) -> Dict:
        """Get enhanced self-learning statistics"""
        stats = {
            'total_labels_created': self.total_labels_created,
            'positive_labels': self.positive_labels,
            'negative_labels': self.negative_labels,
            'uncertainty_labels': self.uncertainty_labels,
            'action_detection_pairs': len(self.action_detection_pairs),
            'self_labeled_data': len(self.self_labeled_data),
            'detection_rewards': {k: v['avg_reward'] for k, v in self.detection_rewards.items()},
            'confidence_adjustments': self.confidence_adjustments.copy(),
            'training_iterations': self.training_iterations,
            'active_learning_enabled': self.active_learning_enabled,
            'uncertainty_samples': len(self.uncertainty_samples)
        }
        
        # Add reward statistics
        if len(self.reward_history) > 0:
            stats['reward_stats'] = {
                'avg_reward': float(np.mean(self.reward_history)),
                'std_reward': float(np.std(self.reward_history)),
                'min_reward': float(np.min(self.reward_history)),
                'max_reward': float(np.max(self.reward_history)),
                'recent_avg': float(np.mean(list(self.reward_history)[-100:])) if len(self.reward_history) >= 100 else float(np.mean(self.reward_history))
            }
        
        return stats

