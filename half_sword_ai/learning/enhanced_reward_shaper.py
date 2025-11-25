"""
Enhanced Reward Shaper - High-Frequency, Granular Rewards
Provides frame-by-frame feedback for better learning signal

Key Improvements:
- Frame-by-frame rewards (not throttled)
- Granular movement quality rewards
- Action smoothness rewards
- Proximity and engagement rewards
- Momentum-based rewards
- Better reward scaling and normalization
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from collections import deque
from half_sword_ai.config import config
from half_sword_ai.learning.reward_shaper import RewardShaper, CurriculumPhase

logger = logging.getLogger(__name__)


class EnhancedRewardShaper(RewardShaper):
    """
    Enhanced reward shaper with high-frequency granular rewards
    Extends base RewardShaper with frame-by-frame feedback
    """
    
    def __init__(self, 
                 curriculum_phase: CurriculumPhase = CurriculumPhase.MASTER,
                 gamma: float = 0.99,
                 enable_pbrs: bool = True):
        super().__init__(curriculum_phase, gamma, enable_pbrs)
        
        # Enhanced reward tracking
        self.action_history = deque(maxlen=10)  # Track recent actions for smoothness
        self.movement_history = deque(maxlen=10)  # Track movement patterns
        self.reward_history = deque(maxlen=100)  # Track reward distribution
        
        # Frame-by-frame reward weights (always active)
        self.frame_weights = {
            'survival': 0.01,  # Small reward for staying alive
            'engagement': 0.02,  # Reward for being near enemies
            'movement_quality': 0.05,  # Reward for smooth movements
            'action_smoothness': 0.03,  # Reward for smooth actions
            'momentum': 0.02,  # Reward for building momentum
            'proximity': 0.01,  # Reward for optimal spacing
            'activity': 0.01,  # Reward for active gameplay
        }
        
        # Reward normalization
        self.reward_scale = 1.0
        self.reward_clip_min = -10.0
        self.reward_clip_max = 10.0
        
        # Adaptive reward scaling
        self.adaptive_scaling = True
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        logger.info("EnhancedRewardShaper initialized with frame-by-frame rewards")
    
    def calculate_reward(self, 
                        game_state: Dict,
                        action: Optional[np.ndarray] = None,
                        previous_state: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate enhanced reward with frame-by-frame feedback
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        reward = 0.0
        breakdown = {}
        
        # 1. Base comprehensive rewards (from parent class)
        base_reward, base_breakdown = super().calculate_reward(
            game_state, action, previous_state
        )
        reward += base_reward
        breakdown.update(base_breakdown)
        
        # 2. Frame-by-frame granular rewards (always calculated)
        frame_rewards = self._calculate_frame_rewards(game_state, action, previous_state)
        reward += frame_rewards['total']
        breakdown['frame_rewards'] = frame_rewards
        
        # 3. Action smoothness reward
        if action is not None:
            smoothness_reward = self._calculate_action_smoothness(action)
            reward += smoothness_reward * self.frame_weights['action_smoothness']
            breakdown['action_smoothness'] = smoothness_reward
        
        # 4. Movement quality reward
        movement_reward = self._calculate_movement_quality(game_state, previous_state)
        reward += movement_reward * self.frame_weights['movement_quality']
        breakdown['movement_quality'] = movement_reward
        
        # 5. Momentum reward
        momentum_reward = self._calculate_momentum_reward(game_state, previous_state)
        reward += momentum_reward * self.frame_weights['momentum']
        breakdown['momentum'] = momentum_reward
        
        # 6. Normalize and clip reward
        if self.adaptive_scaling:
            reward = self._normalize_reward(reward)
        
        reward = np.clip(reward, self.reward_clip_min, self.reward_clip_max)
        
        # Track reward distribution
        self.reward_history.append(reward)
        if len(self.reward_history) >= 100:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history)
        
        breakdown['total'] = float(reward)
        return float(reward), breakdown
    
    def _calculate_frame_rewards(self, 
                                 game_state: Dict,
                                 action: Optional[np.ndarray],
                                 previous_state: Optional[Dict]) -> Dict:
        """Calculate frame-by-frame granular rewards"""
        rewards = {}
        total = 0.0
        
        # Survival reward (small positive for staying alive)
        if not game_state.get('is_dead', False):
            survival_reward = self.frame_weights['survival']
            rewards['survival'] = survival_reward
            total += survival_reward
        
        # Engagement reward (reward for being near enemies)
        enemy_distance = game_state.get('enemy_distance')
        if enemy_distance is not None:
            # Reward for being in engagement range (not too far, not too close)
            optimal_range = 1.5  # meters
            distance_error = abs(enemy_distance - optimal_range)
            engagement_reward = max(0, 1.0 - distance_error / optimal_range)
            engagement_reward *= self.frame_weights['engagement']
            rewards['engagement'] = engagement_reward
            total += engagement_reward
        
        # Proximity reward (reward for optimal spacing)
        if enemy_distance is not None:
            weapon_reach = game_state.get('weapon_reach', 1.0)
            optimal_proximity = weapon_reach * 0.8  # Just inside reach
            proximity_error = abs(enemy_distance - optimal_proximity)
            proximity_reward = np.exp(-proximity_error)
            proximity_reward *= self.frame_weights['proximity']
            rewards['proximity'] = proximity_reward
            total += proximity_reward
        
        # Activity reward (reward for active gameplay)
        is_attacking = game_state.get('is_attacking', False)
        has_movement = action is not None and np.linalg.norm(action[:2]) > 0.01 if action is not None else False
        
        if is_attacking or has_movement:
            activity_reward = self.frame_weights['activity']
            rewards['activity'] = activity_reward
            total += activity_reward
        
        rewards['total'] = total
        return rewards
    
    def _calculate_action_smoothness(self, action: np.ndarray) -> float:
        """
        Reward smooth actions (low jerk)
        Penalizes sudden direction changes
        """
        if len(self.action_history) < 2:
            self.action_history.append(action.copy() if action is not None else np.zeros(2))
            return 0.0
        
        # Get last action
        last_action = self.action_history[-1]
        if len(last_action) < 2 or action is None or len(action) < 2:
            self.action_history.append(action.copy() if action is not None else np.zeros(2))
            return 0.0
        
        # Calculate action change (jerk)
        action_delta = action[:2] - last_action[:2]
        jerk_magnitude = np.linalg.norm(action_delta)
        
        # Smoothness reward (inverse of jerk)
        # Lower jerk = higher reward
        smoothness = 1.0 / (1.0 + jerk_magnitude * 10.0)
        
        self.action_history.append(action.copy())
        return float(smoothness)
    
    def _calculate_movement_quality(self, 
                                   game_state: Dict,
                                   previous_state: Optional[Dict]) -> float:
        """
        Reward high-quality movements
        - Smooth velocity changes
        - Directed movements (not random)
        - Appropriate magnitude
        """
        if previous_state is None:
            return 0.0
        
        # Get movement pattern
        movement_pattern = game_state.get('movement_pattern', [])
        if len(movement_pattern) < 2:
            return 0.0
        
        # Calculate movement quality metrics
        recent_movements = movement_pattern[-2:]
        
        # 1. Direction consistency (reward consistent direction)
        if len(recent_movements) >= 2:
            dir1 = np.array([recent_movements[0].get('dx', 0), recent_movements[0].get('dy', 0)])
            dir2 = np.array([recent_movements[1].get('dx', 0), recent_movements[1].get('dy', 0)])
            
            # Normalize
            norm1 = np.linalg.norm(dir1)
            norm2 = np.linalg.norm(dir2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                dir1_norm = dir1 / norm1
                dir2_norm = dir2 / norm2
                
                # Cosine similarity (consistency)
                consistency = np.dot(dir1_norm, dir2_norm)
                
                # 2. Appropriate magnitude (not too small, not too large)
                avg_magnitude = (norm1 + norm2) / 2.0
                optimal_magnitude = 0.5  # Normalized
                magnitude_error = abs(avg_magnitude - optimal_magnitude)
                magnitude_score = 1.0 - min(1.0, magnitude_error / optimal_magnitude)
                
                # Combined quality score
                quality = (consistency + magnitude_score) / 2.0
                return float(max(0.0, quality))
        
        return 0.0
    
    def _calculate_momentum_reward(self,
                                  game_state: Dict,
                                  previous_state: Optional[Dict]) -> float:
        """
        Reward building momentum (for physics-based combat)
        Momentum is important for effective strikes
        """
        if previous_state is None:
            return 0.0
        
        # Estimate momentum from movement pattern
        movement_pattern = game_state.get('movement_pattern', [])
        if len(movement_pattern) < 3:
            return 0.0
        
        # Calculate velocity trend (increasing velocity = building momentum)
        velocities = []
        for move in movement_pattern[-3:]:
            dx = move.get('dx', 0)
            dy = move.get('dy', 0)
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        if len(velocities) >= 2:
            # Check if velocity is increasing (momentum building)
            velocity_trend = velocities[-1] - velocities[0]
            
            # Reward positive trend (building momentum)
            if velocity_trend > 0:
                momentum_reward = min(1.0, velocity_trend * 2.0)
                return float(momentum_reward)
        
        return 0.0
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running statistics
        Helps stabilize learning
        """
        if self.reward_std > 1e-6:
            normalized = (reward - self.reward_mean) / self.reward_std
            return float(normalized)
        return reward
    
    def update_frame_weights(self, weights: Dict[str, float]):
        """Update frame-by-frame reward weights"""
        self.frame_weights.update(weights)
        logger.info(f"Frame reward weights updated: {weights}")
    
    def get_reward_stats(self) -> Dict:
        """Get reward statistics"""
        if len(self.reward_history) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        rewards_array = np.array(self.reward_history)
        return {
            'mean': float(np.mean(rewards_array)),
            'std': float(np.std(rewards_array)),
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'count': len(self.reward_history)
        }

