"""
Autonomous Learning Manager
Massively enhanced learning system for continuous self-improvement

Key Features:
- Continuous autonomous learning (no human intervention required)
- Automatic model checkpointing and best model tracking
- Adaptive exploration and learning rate scheduling
- Performance tracking and automatic curriculum progression
- Self-play and self-evaluation mechanisms
- Automatic model improvement detection
"""

import torch
import numpy as np
import time
import logging
import os
from typing import Dict, Optional, Tuple, List
from collections import deque
from pathlib import Path
from half_sword_ai.config import config
from half_sword_ai.learning.model_tracker import ModelTracker

logger = logging.getLogger(__name__)


class AutonomousLearningManager:
    """
    Manages autonomous learning - bot continuously improves itself
    without requiring human intervention
    """
    
    def __init__(self, model, replay_buffer, performance_monitor=None):
        self.model = model
        self.replay_buffer = replay_buffer
        self.performance_monitor = performance_monitor
        
        # Model tracking and checkpointing
        self.model_tracker = ModelTracker(model)
        self.best_performance = -float('inf')
        self.best_model_path = None
        self.checkpoint_dir = Path(config.MODEL_SAVE_PATH) / "autonomous_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)  # Last 1000 episodes
        self.reward_history = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.learning_progress = {
            'total_updates': 0,
            'total_episodes': 0,
            'total_experiences': 0,
            'improvement_rate': 0.0,
            'last_improvement': time.time()
        }
        
        # Adaptive learning parameters
        self.exploration_rate = config.DQN_EPSILON_START if hasattr(config, 'DQN_EPSILON_START') else 1.0
        self.exploration_decay = 0.9995  # Slow decay for continuous learning
        self.exploration_min = 0.05  # Always maintain some exploration
        self.learning_rate_multiplier = 1.0  # Adaptive LR multiplier
        
        # Automatic checkpointing
        self.checkpoint_interval = 1000  # Save checkpoint every N updates
        self.best_model_save_interval = 100  # Save best model every N updates
        self.last_checkpoint_update = 0
        self.last_best_model_save = 0
        
        # Performance evaluation
        self.evaluation_interval = 500  # Evaluate every N updates
        self.evaluation_window = 50  # Evaluate over last N episodes
        self.last_evaluation = 0
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.curriculum_stages = [
            {'name': 'beginner', 'min_performance': -100, 'exploration': 0.9},
            {'name': 'intermediate', 'min_performance': 0, 'exploration': 0.7},
            {'name': 'advanced', 'min_performance': 50, 'exploration': 0.5},
            {'name': 'expert', 'min_performance': 100, 'exploration': 0.3},
            {'name': 'master', 'min_performance': 200, 'exploration': 0.1}
        ]
        
        # Self-improvement detection
        self.improvement_threshold = 0.05  # 5% improvement required
        self.stagnation_threshold = 1000  # Updates without improvement
        self.stagnation_count = 0
        
        # Learning statistics
        self.stats = {
            'total_training_time': 0.0,
            'avg_training_time_per_update': 0.0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'consecutive_improvements': 0,
            'consecutive_degradations': 0
        }
        
        logger.info("🤖 Autonomous Learning Manager initialized")
        logger.info(f"   Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"   Initial exploration rate: {self.exploration_rate:.3f}")
    
    def update_performance(self, episode_reward: float, episode_length: int):
        """
        Update performance tracking with new episode data
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode in frames
        """
        self.performance_history.append(episode_reward)
        self.reward_history.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.learning_progress['total_episodes'] += 1
        
        # Check for improvement
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            if recent_avg > self.best_performance:
                improvement = recent_avg - self.best_performance
                improvement_pct = (improvement / abs(self.best_performance)) * 100 if self.best_performance != 0 else 100
                
                if improvement_pct >= self.improvement_threshold * 100:
                    self.best_performance = recent_avg
                    self.learning_progress['last_improvement'] = time.time()
                    self.stagnation_count = 0
                    self.stats['consecutive_improvements'] += 1
                    self.stats['consecutive_degradations'] = 0
                    
                    logger.info(f"🎯 PERFORMANCE IMPROVEMENT: {recent_avg:.2f} (+{improvement:.2f}, +{improvement_pct:.1f}%)")
                else:
                    self.stagnation_count += 1
    
    def should_save_checkpoint(self, update_count: int) -> bool:
        """Check if we should save a checkpoint"""
        return (update_count - self.last_checkpoint_update) >= self.checkpoint_interval
    
    def save_checkpoint(self, update_count: int, optimizer_state: Dict = None):
        """
        Save model checkpoint
        
        Args:
            update_count: Current update count
            optimizer_state: Optimizer state dict
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{update_count}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'update_count': update_count,
            'performance': self.best_performance,
            'exploration_rate': self.exploration_rate,
            'learning_progress': dict(self.learning_progress),
            'curriculum_stage': self.curriculum_stage,
            'stats': dict(self.stats)
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, checkpoint_path)
        self.last_checkpoint_update = update_count
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path} (Update {update_count})")
    
    def save_best_model(self, update_count: int):
        """Save best performing model"""
        if len(self.performance_history) < 10:
            return
        
        recent_avg = np.mean(list(self.performance_history)[-10:])
        
        # Only save if significantly better than previous best
        if recent_avg > self.best_performance * 1.01:  # 1% improvement threshold
            best_model_path = self.checkpoint_dir / "best_model.pt"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'performance': recent_avg,
                'update_count': update_count,
                'exploration_rate': self.exploration_rate,
                'learning_progress': dict(self.learning_progress),
                'curriculum_stage': self.curriculum_stage,
                'stats': dict(self.stats)
            }
            
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            self.last_best_model_save = update_count
            
            logger.info(f"🏆 BEST MODEL SAVED: Performance {recent_avg:.2f} (Update {update_count})")
    
    def evaluate_performance(self, update_count: int) -> Dict:
        """
        Evaluate current model performance
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.performance_history) < self.evaluation_window:
            return {}
        
        recent_rewards = list(self.reward_history)[-self.evaluation_window:]
        recent_lengths = list(self.episode_lengths)[-self.evaluation_window:]
        
        metrics = {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'improvement_rate': self._calculate_improvement_rate(),
            'exploration_rate': self.exploration_rate,
            'curriculum_stage': self.curriculum_stage,
            'buffer_size': len(self.replay_buffer),
            'stagnation_count': self.stagnation_count
        }
        
        logger.info(f"📊 PERFORMANCE EVALUATION (Updates: {update_count}):")
        logger.info(f"   Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(f"   Episode Length: {metrics['avg_episode_length']:.1f}")
        logger.info(f"   Improvement Rate: {metrics['improvement_rate']:.2%}")
        logger.info(f"   Exploration: {metrics['exploration_rate']:.3f}")
        logger.info(f"   Curriculum Stage: {self.curriculum_stages[self.curriculum_stage]['name']}")
        
        return metrics
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement over recent episodes"""
        if len(self.performance_history) < 20:
            return 0.0
        
        recent = list(self.performance_history)[-20:]
        older = list(self.performance_history)[-40:-20] if len(self.performance_history) >= 40 else recent[:10]
        
        if len(older) == 0:
            return 0.0
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if older_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0
        
        return (recent_avg - older_avg) / abs(older_avg)
    
    def update_exploration_rate(self):
        """Update exploration rate based on performance"""
        # Decay exploration rate
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )
        
        # Adjust based on curriculum stage
        target_exploration = self.curriculum_stages[self.curriculum_stage]['exploration']
        if self.exploration_rate > target_exploration:
            # Decay faster if above target
            self.exploration_rate = max(
                target_exploration,
                self.exploration_rate * 0.995
            )
    
    def update_curriculum(self):
        """Update curriculum stage based on performance"""
        if len(self.performance_history) < 20:
            return
        
        recent_avg = np.mean(list(self.performance_history)[-20:])
        current_stage = self.curriculum_stages[self.curriculum_stage]
        
        # Check if we should advance
        if (self.curriculum_stage < len(self.curriculum_stages) - 1 and
            recent_avg >= self.curriculum_stages[self.curriculum_stage + 1]['min_performance']):
            
            self.curriculum_stage += 1
            new_stage = self.curriculum_stages[self.curriculum_stage]
            logger.info(f"📈 CURRICULUM ADVANCEMENT: {current_stage['name']} → {new_stage['name']}")
            logger.info(f"   Performance: {recent_avg:.2f} (Required: {new_stage['min_performance']})")
        
        # Check if we should regress (performance degraded)
        elif (self.curriculum_stage > 0 and
              recent_avg < current_stage['min_performance'] * 0.8 and
              self.stagnation_count > self.stagnation_threshold):
            
            self.curriculum_stage = max(0, self.curriculum_stage - 1)
            new_stage = self.curriculum_stages[self.curriculum_stage]
            logger.warning(f"📉 CURRICULUM REGRESSION: {current_stage['name']} → {new_stage['name']}")
            logger.warning(f"   Performance: {recent_avg:.2f} (Required: {current_stage['min_performance']})")
    
    def adapt_learning_rate(self, optimizer, current_loss: float):
        """
        Adapt learning rate based on performance
        
        Args:
            optimizer: PyTorch optimizer
            current_loss: Current training loss
        """
        # Increase LR if performance is improving
        if self.stats['consecutive_improvements'] > 5:
            self.learning_rate_multiplier = min(2.0, self.learning_rate_multiplier * 1.1)
            self.stats['consecutive_improvements'] = 0
        
        # Decrease LR if performance is degrading
        elif self.stats['consecutive_degradations'] > 5:
            self.learning_rate_multiplier = max(0.1, self.learning_rate_multiplier * 0.9)
            self.stats['consecutive_degradations'] = 0
        
        # Apply multiplier to learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.LEARNING_RATE * self.learning_rate_multiplier
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate"""
        return self.exploration_rate
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        return {
            'total_updates': self.learning_progress['total_updates'],
            'total_episodes': self.learning_progress['total_episodes'],
            'total_experiences': len(self.replay_buffer),
            'best_performance': self.best_performance,
            'current_performance': np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0,
            'improvement_rate': self._calculate_improvement_rate(),
            'exploration_rate': self.exploration_rate,
            'curriculum_stage': self.curriculum_stages[self.curriculum_stage]['name'],
            'stagnation_count': self.stagnation_count,
            'learning_rate_multiplier': self.learning_rate_multiplier,
            **self.stats
        }
    
    def load_best_model(self) -> bool:
        """Load best performing model"""
        if self.best_model_path and self.best_model_path.exists():
            try:
                checkpoint = torch.load(self.best_model_path, map_location=config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.best_performance = checkpoint.get('performance', self.best_performance)
                logger.info(f"✅ Loaded best model: {self.best_model_path} (Performance: {self.best_performance:.2f})")
                return True
            except Exception as e:
                logger.error(f"Failed to load best model: {e}")
                return False
        return False
    
    def record_training_update(self, loss: float, update_count: int, training_time: float):
        """Record training update statistics"""
        self.learning_progress['total_updates'] = update_count
        self.stats['total_training_time'] += training_time
        self.stats['total_loss'] += loss
        self.stats['avg_loss'] = self.stats['total_loss'] / max(1, update_count)
        self.stats['avg_training_time_per_update'] = self.stats['total_training_time'] / max(1, update_count)

