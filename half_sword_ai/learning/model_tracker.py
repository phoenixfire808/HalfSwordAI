"""
Model Tracker: Advanced monitoring with learning rate adaptation and analytics
Tracks how the model is improving as it learns from human actions

MASSIVE IMPROVEMENTS:
- Advanced metrics and analytics
- Learning rate adaptation
- Performance trend analysis
- Early stopping detection
- Model comparison tracking
"""
import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from collections import deque
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

class ModelTracker:
    """
    Tracks model training progress and improvements
    Monitors how well the model is learning from human demonstrations
    """
    
    def __init__(self, model):
        self.model = model
        self.training_history = deque(maxlen=5000)  # Extended history
        
        # Enhanced improvement metrics
        self.improvement_metrics = {
            'loss_trend': deque(maxlen=500),
            'bc_loss_trend': deque(maxlen=500),
            'ppo_loss_trend': deque(maxlen=500),
            'human_action_accuracy': deque(maxlen=500),
            'action_similarity': deque(maxlen=500),
            'gradient_norm': deque(maxlen=500),
            'learning_rate': deque(maxlen=500),
            'value_loss': deque(maxlen=500),
            'policy_loss': deque(maxlen=500),
            'entropy': deque(maxlen=500)
        }
        
        # Performance tracking
        self.performance_windows = {
            'short': deque(maxlen=50),   # Last 50 steps
            'medium': deque(maxlen=200),  # Last 200 steps
            'long': deque(maxlen=1000)    # Last 1000 steps
        }
        
        # Learning rate adaptation
        self.learning_rate_history = deque(maxlen=500)
        self.optimal_learning_rate = config.LEARNING_RATE
        self.lr_adaptation_enabled = True
        self.patience_counter = 0
        self.patience_threshold = 100  # Steps without improvement
        self.best_loss = float('inf')
        
        # Early stopping
        self.early_stopping_enabled = False
        self.early_stopping_patience = 500
        self.early_stopping_counter = 0
        
        # Model comparison
        self.model_snapshots = []  # Store model checkpoints for comparison
        self.snapshot_interval = 500
        
        # Advanced analytics
        self.convergence_detected = False
        self.overfitting_detected = False
        self.performance_plateau = False
        
        self.last_checkpoint = None
        self.checkpoint_interval = 100
        self.update_count = 0
        self.start_time = time.time()
        
    def record_training_step(self, loss: float, ppo_loss: float, bc_loss: float,
                            human_actions_in_batch: int, total_batch_size: int,
                            value_loss: float = None, policy_loss: float = None,
                            entropy: float = None, gradient_norm: float = None,
                            learning_rate: float = None):
        """Record an enhanced training step with advanced metrics"""
        self.update_count += 1
        current_time = time.time()
        
        # Enhanced record
        record = {
            'update_count': self.update_count,
            'timestamp': current_time,
            'elapsed_time': current_time - self.start_time,
            'total_loss': loss,
            'ppo_loss': ppo_loss,
            'bc_loss': bc_loss,
            'value_loss': value_loss if value_loss is not None else 0,
            'policy_loss': policy_loss if policy_loss is not None else 0,
            'entropy': entropy if entropy is not None else 0,
            'gradient_norm': gradient_norm if gradient_norm is not None else 0,
            'learning_rate': learning_rate if learning_rate is not None else self.optimal_learning_rate,
            'human_action_ratio': human_actions_in_batch / max(1, total_batch_size)
        }
        
        self.training_history.append(record)
        
        # Update all metrics
        self.improvement_metrics['loss_trend'].append(loss)
        self.improvement_metrics['bc_loss_trend'].append(bc_loss)
        self.improvement_metrics['ppo_loss_trend'].append(ppo_loss)
        if value_loss is not None:
            self.improvement_metrics['value_loss'].append(value_loss)
        if policy_loss is not None:
            self.improvement_metrics['policy_loss'].append(policy_loss)
        if entropy is not None:
            self.improvement_metrics['entropy'].append(entropy)
        if gradient_norm is not None:
            self.improvement_metrics['gradient_norm'].append(gradient_norm)
        if learning_rate is not None:
            self.improvement_metrics['learning_rate'].append(learning_rate)
            self.learning_rate_history.append(learning_rate)
        
        # Update performance windows
        self.performance_windows['short'].append(loss)
        self.performance_windows['medium'].append(loss)
        self.performance_windows['long'].append(loss)
        
        # Check for improvement with enhanced analysis
        if len(self.improvement_metrics['loss_trend']) > 10:
            self._analyze_improvement()
        
        # Learning rate adaptation
        if self.lr_adaptation_enabled and len(self.improvement_metrics['loss_trend']) > 50:
            self._adapt_learning_rate(loss)
        
        # Early stopping check
        if self.early_stopping_enabled:
            self._check_early_stopping(loss)
        
        # Convergence detection
        if len(self.improvement_metrics['loss_trend']) > 100:
            self._detect_convergence()
        
        # Overfitting detection
        if len(self.training_history) > 200:
            self._detect_overfitting()
    
    def _analyze_improvement(self):
        """Enhanced improvement analysis"""
        if len(self.improvement_metrics['loss_trend']) < 20:
            return
        
        recent_avg = np.mean(list(self.improvement_metrics['loss_trend'])[-10:])
        older_avg = np.mean(list(self.improvement_metrics['loss_trend'])[-20:-10])
        
        improvement = (older_avg - recent_avg) / max(abs(older_avg), 1e-6)
        
        if improvement > 0.05:  # 5% improvement
            logger.info(f"Model improving! Loss decreased by {improvement*100:.1f}%")
            self.patience_counter = 0
            if recent_avg < self.best_loss:
                self.best_loss = recent_avg
        elif improvement < -0.02:  # 2% degradation
            self.patience_counter += 1
        else:
            # Minimal change
            self.patience_counter += 1
    
    def _adapt_learning_rate(self, current_loss: float):
        """Adapt learning rate based on performance"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return
        
        # Increase patience if no improvement
        self.patience_counter += 1
        
        if self.patience_counter >= self.patience_threshold:
            # Reduce learning rate
            self.optimal_learning_rate *= 0.9
            self.patience_counter = 0
            logger.info(f"📉 Reducing learning rate to {self.optimal_learning_rate:.6f}")
    
    def _check_early_stopping(self, current_loss: float):
        """Check for early stopping conditions"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        if self.early_stopping_counter >= self.early_stopping_patience:
            logger.warning("⏸️  Early stopping triggered - no improvement detected")
    
    def _detect_convergence(self):
        """Detect if model has converged"""
        if len(self.improvement_metrics['loss_trend']) < 100:
            return
        
        recent_losses = list(self.improvement_metrics['loss_trend'])[-50:]
        variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)
        
        # Low variance relative to mean suggests convergence
        if variance < mean_loss * 0.01:  # Less than 1% variance
            if not self.convergence_detected:
                logger.info("✅ Model convergence detected!")
                self.convergence_detected = True
    
    def _detect_overfitting(self):
        """Detect potential overfitting"""
        if len(self.improvement_metrics['loss_trend']) < 200:
            return
        
        # Check if training loss is decreasing but validation would be increasing
        # Simple heuristic: if loss decreases very slowly but variance increases
        recent = list(self.improvement_metrics['loss_trend'])[-50:]
        older = list(self.improvement_metrics['loss_trend'])[-100:-50]
        
        recent_var = np.var(recent)
        older_var = np.var(older)
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        # Variance increasing while loss plateaus suggests overfitting
        if recent_var > older_var * 1.5 and abs(recent_mean - older_mean) < older_mean * 0.05:
            if not self.overfitting_detected:
                logger.warning("⚠️  Potential overfitting detected!")
                self.overfitting_detected = True
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        if len(self.training_history) == 0:
            return {
                'update_count': 0,
                'avg_loss': 0.0,
                'trend': 'no_data',
                'elapsed_time': 0
            }
        
        recent_losses = list(self.improvement_metrics['loss_trend'])
        recent_bc = list(self.improvement_metrics['bc_loss_trend'])
        recent_ppo = list(self.improvement_metrics['ppo_loss_trend'])
        
        # Enhanced trend calculation
        trend = 'initializing'
        improvement_rate = 0.0
        
        if len(recent_losses) > 10:
            recent_avg = np.mean(recent_losses[-10:])
            older_avg = np.mean(recent_losses[-20:-10]) if len(recent_losses) > 20 else recent_avg
            
            if recent_avg < older_avg * 0.95:
                trend = 'improving'
                improvement_rate = (older_avg - recent_avg) / older_avg
            elif recent_avg > older_avg * 1.1:
                trend = 'degrading'
                improvement_rate = -(recent_avg - older_avg) / older_avg
            else:
                trend = 'stable'
        
        # Calculate metrics for different windows
        window_stats = {}
        for window_name, window_data in self.performance_windows.items():
            if len(window_data) > 0:
                window_stats[f'{window_name}_avg_loss'] = np.mean(window_data)
                window_stats[f'{window_name}_min_loss'] = np.min(window_data)
                window_stats[f'{window_name}_max_loss'] = np.max(window_data)
        
        # Gradient statistics
        gradient_stats = {}
        if len(self.improvement_metrics['gradient_norm']) > 0:
            grad_norms = list(self.improvement_metrics['gradient_norm'])
            gradient_stats = {
                'avg_gradient_norm': np.mean(grad_norms),
                'max_gradient_norm': np.max(grad_norms),
                'min_gradient_norm': np.min(grad_norms)
            }
        
        base_stats = {
            'update_count': self.update_count,
            'elapsed_time': time.time() - self.start_time,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0.0,
            'min_loss': np.min(recent_losses) if recent_losses else 0.0,
            'max_loss': np.max(recent_losses) if recent_losses else 0.0,
            'avg_bc_loss': np.mean(recent_bc) if recent_bc else 0.0,
            'avg_ppo_loss': np.mean(recent_ppo) if recent_ppo else 0.0,
            'trend': trend,
            'improvement_rate': improvement_rate,
            'optimal_learning_rate': self.optimal_learning_rate,
            'current_learning_rate': list(self.improvement_metrics['learning_rate'])[-1] if self.improvement_metrics['learning_rate'] else self.optimal_learning_rate,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'convergence_detected': self.convergence_detected,
            'overfitting_detected': self.overfitting_detected,
            'human_action_ratio': np.mean([r['human_action_ratio'] for r in list(self.training_history)[-10:]]) if len(self.training_history) > 0 else 0.0
        }
        
        # Merge all statistics
        base_stats.update(window_stats)
        base_stats.update(gradient_stats)
        
        return base_stats
    
    def save_checkpoint(self, path: str, include_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None):
        """Save enhanced model checkpoint with full training state"""
        try:
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'update_count': self.update_count,
                'training_stats': self.get_training_stats(),
                'improvement_metrics': {k: list(v) for k, v in self.improvement_metrics.items()},
                'best_loss': self.best_loss,
                'optimal_learning_rate': self.optimal_learning_rate,
                'timestamp': time.time(),
                'elapsed_time': time.time() - self.start_time
            }
            
            if include_optimizer and optimizer is not None:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint_data, path)
            logger.info(f"💾 Enhanced model checkpoint saved: {path}")
            
            # Store snapshot for comparison
            if self.update_count % self.snapshot_interval == 0:
                self.model_snapshots.append({
                    'path': path,
                    'stats': self.get_training_stats(),
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_learning_rate_recommendation(self) -> float:
        """Get recommended learning rate based on performance"""
        if not self.lr_adaptation_enabled:
            return config.LEARNING_RATE
        
        return self.optimal_learning_rate
    
    def should_early_stop(self) -> bool:
        """Check if training should stop early"""
        return self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience

