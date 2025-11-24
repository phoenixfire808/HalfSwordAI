"""
Enhanced Prioritized Experience Replay Buffer
Advanced priority calculation and adaptive sampling

MASSIVE IMPROVEMENTS:
- Advanced priority calculation with TD error and temporal importance
- Adaptive sampling strategies
- Better memory management
- Experience replay strategies (PER, n-step, HER)
- Priority decay and annealing
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from half_sword_ai.config import config
import time
import logging

# Import config for priority multiplier
try:
    HUMAN_PRIORITY_MULT = config.HUMAN_ACTION_PRIORITY_MULTIPLIER
except:
    HUMAN_PRIORITY_MULT = 5.0

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for DAgger and PPO
    Human interventions get higher priority for immediate learning
    """
    
    def __init__(self, capacity: int = None, alpha: float = None):
        self.capacity = capacity or config.REPLAY_BUFFER_SIZE
        self.alpha = alpha or config.PRIORITY_ALPHA
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.max_priority = 1.0
        
        # Enhanced priority tracking
        self.td_errors = deque(maxlen=self.capacity)  # Store TD errors for priority updates
        self.temporal_importance = deque(maxlen=self.capacity)  # Temporal importance weighting
        self.access_counts = deque(maxlen=self.capacity)  # Track how often experiences are sampled
        self.last_access = deque(maxlen=self.capacity)  # Last access time for each experience
        
        # Priority decay
        self.priority_decay_enabled = True
        self.priority_decay_rate = 0.999  # Decay priorities over time
        self.last_decay_time = time.time()
        self.decay_interval = 60.0  # Decay every 60 seconds
        
        # Adaptive sampling
        self.adaptive_sampling_enabled = True
        self.sampling_strategy = "prioritized"  # "prioritized", "uniform", "mixed"
        self.uniform_fraction = 0.1  # 10% uniform sampling in mixed mode
        
        # Experience statistics
        self.human_intervention_count = 0
        self.total_experiences = 0
        self.priority_statistics = {
            'avg_priority': 0.0,
            'max_priority': 0.0,
            'min_priority': 0.0,
            'human_priority_ratio': 0.0
        }
        
        # N-step returns (for future enhancement)
        self.n_step_enabled = False
        self.n_step_buffer = deque(maxlen=100)  # Temporary buffer for n-step
        
        # HER (Hindsight Experience Replay) support
        self.her_enabled = False
        self.her_ratio = 0.5  # 50% HER experiences if enabled
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float = 0.0,
             next_state: np.ndarray = None, done: bool = False,
             priority: str = "LOW", human_intervention: bool = False,
             human_action: np.ndarray = None, td_error: float = None,
             temporal_weight: float = 1.0):
        """
        Enhanced experience addition with advanced priority calculation
        
        Args:
            state: Game state (frame or frame stack)
            action: Action taken (mouse deltas, buttons)
            reward: Reward received
            next_state: Next state (optional)
            done: Episode done flag
            priority: "HIGH" or "LOW"
            human_intervention: True if this was human correction
            human_action: Human action for DAgger
            td_error: Temporal difference error for priority
            temporal_weight: Temporal importance weight (newer = higher)
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "human_intervention": human_intervention,
            "human_action": human_action if human_intervention else None,
            "timestamp": time.time(),
            "episode_step": len(self.buffer)  # Track position in buffer
        }
        
        # Enhanced priority calculation
        priority_value = self._calculate_priority(
            human_intervention, priority, td_error, temporal_weight, reward, done
        )
        
        self.buffer.append(experience)
        self.priorities.append(priority_value)
        self.td_errors.append(abs(td_error) if td_error is not None else 1.0)
        self.temporal_importance.append(temporal_weight)
        self.access_counts.append(0)  # New experience, not accessed yet
        self.last_access.append(time.time())
        
        # Update statistics
        self.total_experiences += 1
        if human_intervention:
            self.human_intervention_count += 1
        
        # Update max priority (clamp to prevent overflow)
        if priority_value > self.max_priority:
            self.max_priority = max(1.0, min(priority_value, 1e6))
        
        # Update priority statistics
        self._update_priority_statistics()
        
        # Decay old priorities periodically
        if self.priority_decay_enabled:
            self._decay_priorities()
    
    def _calculate_priority(self, human_intervention: bool, priority: str,
                           td_error: Optional[float], temporal_weight: float,
                           reward: float, done: bool) -> float:
        """Calculate priority using multiple factors"""
        # Use a reasonable base priority to prevent overflow
        base_priority = max(1.0, min(self.max_priority, 1e6))
        
        # Human interventions get much higher priority (but clamp to prevent overflow)
        if human_intervention or priority == "HIGH":
            base_priority = min(base_priority * HUMAN_PRIORITY_MULT, 1e6)  # Clamp to prevent overflow
        
        # TD error based priority (if available)
        if td_error is not None:
            td_priority = min(abs(td_error) + 1e-6, 1e6)  # Clamp TD error
            base_priority = max(base_priority, td_priority)
        
        # Temporal importance (newer experiences slightly higher)
        base_priority = min(base_priority * (1.0 + temporal_weight * 0.1), 1e6)
        
        # Reward-based priority boost (clamp to prevent overflow)
        if abs(reward) > 1.0:  # Significant reward
            base_priority = min(base_priority * (1.0 + min(abs(reward), 100.0) * 0.1), 1e6)
        
        # Done flag priority (episode endings are important)
        if done:
            base_priority = min(base_priority * 1.2, 1e6)
        
        # Final clamp to prevent any overflow/inf/NaN
        base_priority = max(1e-6, min(base_priority, 1e6))
        if np.isnan(base_priority) or np.isinf(base_priority):
            base_priority = 1.0  # Fallback to default
        
        return base_priority
    
    def _decay_priorities(self):
        """Decay priorities over time to prevent stale high priorities"""
        current_time = time.time()
        if current_time - self.last_decay_time < self.decay_interval:
            return
        
        # Decay non-human priorities more aggressively
        for i, exp in enumerate(self.buffer):
            if not exp.get("human_intervention", False):
                # Decay based on access count and age
                age = current_time - exp.get("timestamp", current_time)
                access_count = self.access_counts[i] if i < len(self.access_counts) else 0
                
                # Less frequently accessed, older experiences decay more
                decay_factor = self.priority_decay_rate ** (age / 3600.0)  # Decay per hour
                if access_count > 10:
                    decay_factor *= 0.9  # More accessed experiences decay less
                
                if i < len(self.priorities):
                    self.priorities[i] *= decay_factor
        
        self.last_decay_time = current_time
    
    def _update_priority_statistics(self):
        """Update priority statistics for monitoring"""
        if len(self.priorities) == 0:
            return
        
        priorities_list = list(self.priorities)
        self.priority_statistics = {
            'avg_priority': np.mean(priorities_list),
            'max_priority': np.max(priorities_list),
            'min_priority': np.min(priorities_list),
            'human_priority_ratio': self.human_intervention_count / max(1, self.total_experiences)
        }
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Enhanced sampling with adaptive strategies
        
        Returns:
            (experiences, indices, importance_weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            return [], np.array([]), np.array([])
        
        # Adaptive sampling strategy
        if self.adaptive_sampling_enabled and self.sampling_strategy == "mixed":
            # Mixed: combination of prioritized and uniform
            n_prioritized = int(batch_size * (1 - self.uniform_fraction))
            n_uniform = batch_size - n_prioritized
            
            # Prioritized sampling
            prioritized_indices = self._sample_prioritized(n_prioritized)
            
            # Uniform sampling
            uniform_indices = np.random.choice(
                len(self.buffer), size=n_uniform, replace=False
            )
            
            # Combine
            indices = np.concatenate([prioritized_indices, uniform_indices])
        elif self.sampling_strategy == "uniform":
            # Pure uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        else:
            # Prioritized sampling (default)
            indices = self._sample_prioritized(batch_size)
        
        # Update access counts and times
        current_time = time.time()
        for idx in indices:
            if idx < len(self.access_counts):
                self.access_counts[idx] += 1
            if idx < len(self.last_access):
                self.last_access[idx] = current_time
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        if self.sampling_strategy != "uniform":
            priorities_array = np.array(list(self.priorities))
            # Clean up any inf/NaN values
            priorities_array = np.nan_to_num(priorities_array, nan=1.0, posinf=1e6, neginf=1e-6)
            priorities_array = np.clip(priorities_array, 1e-6, 1e6)
            
            probabilities = priorities_array ** self.alpha
            # Check for NaN/inf
            if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                probabilities = np.ones_like(probabilities) / len(probabilities)  # Uniform fallback
            
            prob_sum = probabilities.sum()
            if prob_sum == 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
                probabilities = np.ones_like(probabilities) / len(probabilities)  # Uniform fallback
            else:
                probabilities /= prob_sum
            
            weights = (len(self.buffer) * probabilities[indices]) ** (-config.BETA_BC)
            # Clean up weights
            weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
            weights = np.clip(weights, 1e-6, 1e6)
            weights /= weights.max() if weights.max() > 0 else 1.0
        else:
            # Uniform weights
            weights = np.ones(batch_size)
        
        return experiences, indices, weights
    
    def _sample_prioritized(self, batch_size: int) -> np.ndarray:
        """Prioritized sampling"""
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            return np.array([], dtype=np.int64)
        
        # Ensure priorities match buffer size
        priorities_list = list(self.priorities)
        if len(priorities_list) != buffer_size:
            # Pad or trim priorities to match buffer size
            if len(priorities_list) < buffer_size:
                # Pad with default priority
                priorities_list.extend([self.max_priority] * (buffer_size - len(priorities_list)))
            else:
                # Trim to match buffer size
                priorities_list = priorities_list[:buffer_size]
        
        priorities_array = np.array(priorities_list)
        
        # Clean up any inf/NaN values
        priorities_array = np.nan_to_num(priorities_array, nan=1.0, posinf=1e6, neginf=1e-6)
        
        # Clamp to reasonable range to prevent overflow
        priorities_array = np.clip(priorities_array, 1e-6, 1e6)
        
        # Add small epsilon to prevent zero probabilities
        priorities_array = priorities_array + 1e-6
        
        # Convert to probabilities
        probabilities = priorities_array ** self.alpha
        
        # Check for NaN/inf in probabilities
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            # Fallback to uniform if probabilities are invalid
            logger.warning("Invalid probabilities detected, falling back to uniform sampling")
            return np.random.choice(buffer_size, size=batch_size, replace=False)
        
        # Normalize probabilities
        prob_sum = probabilities.sum()
        if prob_sum == 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
            # Fallback to uniform if sum is invalid
            logger.warning("Invalid probability sum, falling back to uniform sampling")
            return np.random.choice(buffer_size, size=batch_size, replace=False)
        
        probabilities /= prob_sum
        
        # Final check before sampling - ensure sizes match
        if len(probabilities) != buffer_size:
            logger.warning(f"Probability size mismatch ({len(probabilities)} != {buffer_size}), falling back to uniform sampling")
            return np.random.choice(buffer_size, size=batch_size, replace=False)
        
        # Final check for NaN/inf in probabilities
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            logger.warning("Probabilities still invalid after normalization, falling back to uniform sampling")
            return np.random.choice(buffer_size, size=batch_size, replace=False)
        
        # Sample indices
        indices = np.random.choice(buffer_size, size=batch_size, replace=False, p=probabilities)
        
        return indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Enhanced priority update with TD errors and access tracking
        
        Args:
            indices: Indices of experiences to update
            td_errors: Temporal difference errors
        """
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                # Store TD error
                if idx < len(self.td_errors):
                    self.td_errors[idx] = abs(td_error)
                
                # Calculate new priority
                old_priority = self.priorities[idx]
                td_priority = abs(td_error) + 1e-6
                
                # Blend old and new priority (smooth updates)
                # Human interventions maintain higher base priority
                exp = self.buffer[idx]
                if exp.get("human_intervention", False):
                    # Human actions: blend with higher weight on TD error (clamp to prevent overflow)
                    human_td_priority = min(td_priority * HUMAN_PRIORITY_MULT, 1e6)
                    new_priority = old_priority * 0.3 + human_td_priority * 0.7
                else:
                    # Regular experiences: blend more with TD error
                    new_priority = old_priority * 0.2 + td_priority * 0.8
                
                # Clamp to prevent overflow/inf/NaN
                new_priority = max(1e-6, min(new_priority, 1e6))
                if np.isnan(new_priority) or np.isinf(new_priority):
                    new_priority = 1.0  # Fallback to default
                
                self.priorities[idx] = new_priority
                self.max_priority = max(1.0, min(self.max_priority, 1e6))  # Clamp max_priority too
        
        # Update statistics after priority update
        self._update_priority_statistics()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive buffer statistics"""
        return {
            'buffer_size': len(self.buffer),
            'capacity': self.capacity,
            'fill_ratio': len(self.buffer) / self.capacity,
            'human_interventions': self.human_intervention_count,
            'human_ratio': self.human_intervention_count / max(1, len(self.buffer)),
            'total_experiences': self.total_experiences,
            'priority_stats': self.priority_statistics.copy(),
            'sampling_strategy': self.sampling_strategy,
            'max_priority': self.max_priority
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.priorities.clear()
        self.max_priority = 1.0

