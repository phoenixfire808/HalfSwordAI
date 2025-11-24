"""
Enhanced Learner Process: Advanced DAgger, curriculum learning, and better loss functions
Runs continuously in background, updating policy from replay buffer

MASSIVE IMPROVEMENTS:
- Advanced DAgger with uncertainty estimation
- Curriculum learning for progressive difficulty
- Enhanced loss functions (Focal loss, Huber loss)
- Adaptive learning rates
- Better gradient handling
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from collections import deque
from half_sword_ai.config import config
from half_sword_ai.core.model import HalfSwordPolicyNetwork
from half_sword_ai.learning.replay_buffer import PrioritizedReplayBuffer
from half_sword_ai.learning.model_tracker import ModelTracker
from half_sword_ai.monitoring.performance_monitor import PerformanceMonitor

# Import DQN model for type checking
try:
    from half_sword_ai.core.dqn_model import DQNNetwork
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

class LearnerProcess:
    """
    Background training process
    Implements hybrid DAgger + PPO loss for continuous learning
    """
    
    def __init__(self, shared_model, replay_buffer: PrioritizedReplayBuffer, 
                 performance_monitor: PerformanceMonitor = None):
        self.model = shared_model
        self.replay_buffer = replay_buffer
        self.performance_monitor = performance_monitor
        
        # Detect model type (DQN vs PPO)
        self.is_dqn = DQN_AVAILABLE and isinstance(self.model, DQNNetwork)
        if self.is_dqn:
            logger.info("Learner initialized for DQN (ScrimBrain-style)")
        else:
            logger.info("Learner initialized for PPO (Continuous actions)")
        
        # Enhanced optimizer with adaptive learning rate
        self.base_lr = config.LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=50
        )
        
        # DQN-specific parameters (ScrimBrain guide)
        if self.is_dqn:
            self.epsilon = config.DQN_EPSILON_START
            self.epsilon_end = config.DQN_EPSILON_END
            self.epsilon_decay = config.DQN_EPSILON_DECAY
            self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ
            
            # Create target network for DQN
            try:
                from half_sword_ai.core.dqn_model import create_dqn_model
                self.target_model = create_dqn_model(device=config.DEVICE)
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()
                logger.info("DQN target network created")
            except Exception as e:
                logger.warning(f"Failed to create target network: {e}")
                self.target_model = None
        
        self.running = False
        self.beta_bc = config.BETA_BC  # Behavioral cloning weight (anneals)
        self.beta_bc_min = 0.1  # Minimum BC weight
        self.beta_bc_decay = 0.998  # Decay rate
        
        # Advanced DAgger parameters
        self.dagger_uncertainty_threshold = 0.5  # Threshold for uncertainty-based weighting
        self.uncertainty_estimation_enabled = True
        self.uncertainty_history = deque(maxlen=500)
        
        # Curriculum learning
        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_difficulty = 0.0  # 0 = easy, 1 = hard
        self.curriculum_progression_threshold = 0.85  # Performance threshold to advance
        self.curriculum_stages = [
            {'name': 'easy', 'human_ratio': 1.0, 'exploration': 0.3},
            {'name': 'medium', 'human_ratio': 0.7, 'exploration': 0.5},
            {'name': 'hard', 'human_ratio': 0.5, 'exploration': 0.7},
            {'name': 'expert', 'human_ratio': 0.3, 'exploration': 0.9}
        ]
        
        # Enhanced loss functions
        self.use_focal_loss = True  # Focal loss for better handling of hard examples
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.use_huber_loss = False  # Huber loss for robustness
        self.huber_delta = 1.0
        
        # Gradient tracking
        self.gradient_norm_history = deque(maxlen=500)
        self.clip_grad_norm = 0.5
        
        # Training statistics
        self.update_count = 0
        self.model_tracker = ModelTracker(shared_model)
        self.last_checkpoint_save = 0
        self.last_performance_check = time.time()
        self.performance_check_interval = 30.0  # Check every 30 seconds
        
        # Experience replay improvements
        self.td_error_history = deque(maxlen=500)
        self.value_error_history = deque(maxlen=500)
        
    def start(self):
        """Start learner process - continuous online learning"""
        self.running = True
        logger.info("Learner process started - Continuous online learning active")
        logger.info("   Model will train in real-time as you play!")
        
        last_training_time = time.time()
        last_status_log = time.time()
        status_log_interval = 5.0  # Log status every 5 seconds
        
        while self.running:
            try:
                current_time = time.time()
                
                # Train at specified frequency (but train immediately if we have data)
                buffer_size = len(self.replay_buffer)
                min_batch_size = max(1, config.MIN_BATCH_SIZE_FOR_TRAINING)
                
                # Train if we have enough data (even if time hasn't elapsed)
                if buffer_size >= min_batch_size:
                    # Check if we have human actions - train more frequently if so
                    recent_experiences = list(self.replay_buffer.buffer)[-50:] if len(self.replay_buffer) > 0 else []
                    has_human_actions = any(exp.get("human_intervention", False) for exp in recent_experiences)
                    
                    # Train more frequently when human actions are present (every 0.02s = 50 Hz)
                    training_freq = config.TRAINING_FREQUENCY * 0.4 if has_human_actions else config.TRAINING_FREQUENCY
                    
                    if current_time - last_training_time >= training_freq:
                        self._training_step()
                        last_training_time = current_time
                    else:
                        # Small sleep to avoid busy waiting
                        time.sleep(0.01)
                else:
                    # Not enough data - log status periodically
                    if current_time - last_status_log >= status_log_interval:
                        logger.info(f"[LEARNER STATUS] Waiting for data: {buffer_size}/{min_batch_size} | "
                                   f"Training paused | "
                                   f"Updates so far: {self.update_count}")
                        last_status_log = current_time
                    time.sleep(0.1)  # Longer sleep when waiting for data
                
                # Periodic status summary (every 30 seconds)
                if current_time - last_status_log >= 30.0:
                    stats = self.model_tracker.get_training_stats() if hasattr(self.model_tracker, 'get_training_stats') else {}
                    avg_loss = stats.get('avg_loss', 0) if stats else 0
                    logger.info(f"[LEARNER SUMMARY] Updates: {self.update_count} | "
                               f"Avg Loss: {avg_loss:.4f} | "
                               f"Buffer: {buffer_size} | "
                               f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                    last_status_log = current_time
                    
            except Exception as e:
                logger.error(f"Learner error: {e}", exc_info=True)
                
                # Check if we should stop due to errors
                if hasattr(self, 'error_handler') and self.error_handler:
                    should_stop = self.error_handler.record_error(e, context="learner_loop", component="learner")
                    if should_stop:
                        logger.critical("Error handler requested stop - exiting learner loop")
                        self.running = False
                        break
                
                time.sleep(0.1)
    
    def stop(self):
        """Stop learner process"""
        self.running = False
        logger.info("Learner process stopped")
    
    def _training_step(self):
        """Perform one training step - continuous online learning with REAL-TIME updates"""
        # Start training as soon as we have ANY data, even if batch is small
        min_batch_size = max(1, config.MIN_BATCH_SIZE_FOR_TRAINING)  # Train with smaller batches if needed
        
        # Check if we have data in buffer
        buffer_size = len(self.replay_buffer)
        if buffer_size < min_batch_size:
            # Not enough data yet - log frequently for visibility
            if self.update_count % 10 == 0:  # Log every 10 checks (more frequent)
                logger.warning(f"[DATA WAIT] Buffer: {buffer_size}/{min_batch_size} | "
                              f"Training paused | Need {min_batch_size - buffer_size} more experiences")
            return  # Not enough data yet
        
        # Check if we have human actions in recent buffer (calculate first)
        recent_experiences = list(self.replay_buffer.buffer)[-100:] if len(self.replay_buffer) > 0 else []
        human_action_count = sum(1 for exp in recent_experiences 
                                if exp.get("human_intervention", False))
        
        # Log that we're starting training (less verbose)
        if self.update_count % 10 == 0:  # Log every 10 steps
            logger.info(f"TRAINING | Step {self.update_count + 1} | Buffer: {buffer_size} | Human: {human_action_count}")
        
        # Sample batch - use smaller batch if we have human actions to train faster
        batch_size = config.BATCH_SIZE
        if human_action_count > 0:
            # Use smaller batches for faster learning from human actions
            batch_size = max(min_batch_size, min(config.BATCH_SIZE, len(self.replay_buffer)))
        
        # Sample batch with priority on human actions
        experiences, indices, importance_weights = self.replay_buffer.sample(batch_size)
        
        if len(experiences) == 0:
            return
        
        # Prepare batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        human_actions = []
        human_interventions = []
        
        for exp in experiences:
            states.append(exp["state"])
            actions.append(exp["action"])
            rewards.append(exp["reward"])
            next_states.append(exp.get("next_state"))
            dones.append(exp["done"])
            human_interventions.append(exp.get("human_intervention", False))
            # For DAgger, store human action if available
            if exp.get("human_intervention"):
                # Use stored human_action if available, otherwise use action
                human_action = exp.get("human_action")
                if human_action is None:
                    human_action = exp["action"]
                human_actions.append(human_action)
            else:
                human_actions.append(None)
        
        # Convert to tensors
        states_tensor = self._prepare_states(states)
        rewards_tensor = torch.FloatTensor(rewards).to(config.DEVICE)
        weights_tensor = torch.FloatTensor(importance_weights).to(config.DEVICE)
        
        # Branch based on model type (DQN vs PPO)
        if self.is_dqn:
            # DQN: Convert continuous actions to discrete action indices
            # Actions are stored as [dx, dy, left, right, space, alt] arrays
            # Need to convert to discrete action IDs (0-8)
            from half_sword_ai.input.action_discretizer import ActionDiscretizer
            # Use cached discretizer to avoid repeated initialization
            from half_sword_ai.core.dqn_model import _get_cached_discretizer
            discretizer = _get_cached_discretizer()
            discrete_actions = []
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    action_array = np.array(action)
                    if len(action_array) >= 6:
                        # Extract dx, dy, buttons
                        dx = action_array[0]
                        dy = action_array[1]
                        buttons = {
                            'left': bool(action_array[2] > 0.5),
                            'right': bool(action_array[3] > 0.5),
                            'space': bool(action_array[4] > 0.5),
                            'alt': bool(action_array[5] > 0.5)
                        }
                        # Map to discrete action
                        action_id = discretizer.map_continuous_to_discrete(dx, dy, buttons)
                        discrete_actions.append(action_id)
                    else:
                        # Fallback: use neutral action
                        discrete_actions.append(0)
                else:
                    # Already discrete?
                    discrete_actions.append(int(action) if isinstance(action, (int, float)) else 0)
            
            actions_tensor = torch.LongTensor(discrete_actions).to(config.DEVICE)
            
            # Validate batch sizes match
            if states_tensor.shape[0] != actions_tensor.shape[0]:
                logger.error(f"Batch size mismatch: states={states_tensor.shape[0]}, actions={actions_tensor.shape[0]}, "
                           f"num_experiences={len(experiences)}")
                # Truncate to match
                min_batch = min(states_tensor.shape[0], actions_tensor.shape[0])
                states_tensor = states_tensor[:min_batch]
                actions_tensor = actions_tensor[:min_batch]
                rewards_tensor = rewards_tensor[:min_batch]
                weights_tensor = weights_tensor[:min_batch]
                next_states = next_states[:min_batch]
                dones = dones[:min_batch]
            
            # DQN training (ScrimBrain-style) with BC loss for human actions
            total_loss, dqn_loss, value_loss_tensor, bc_loss = self._calculate_dqn_loss(
                states_tensor, actions_tensor, rewards_tensor, 
                next_states, dones, weights_tensor, experiences
            )
            ppo_loss = torch.tensor(0.0).to(config.DEVICE)
            
            policy_loss = dqn_loss.item() if isinstance(dqn_loss, torch.Tensor) else dqn_loss
            value_loss = value_loss_tensor.item() if isinstance(value_loss_tensor, torch.Tensor) else value_loss_tensor
            entropy = None
            values = None  # Not used in DQN path
        else:
            # PPO training (continuous actions)
            actions_tensor = torch.FloatTensor(np.array(actions)).to(config.DEVICE)
            state_features_tensor = self._prepare_state_features(experiences)
            action_dist, discrete_probs, values = self.model(states_tensor, state_features_tensor)
            
            ppo_loss = self._calculate_ppo_loss(
                action_dist, discrete_probs, values,
                actions_tensor, rewards_tensor, weights_tensor, experiences
            )
            
            bc_loss = self._calculate_bc_loss(
                action_dist, discrete_probs,
                actions_tensor, human_actions, human_interventions
            )
            
            # Enhanced loss combination with adaptive weighting
            total_loss = ppo_loss + self.beta_bc * bc_loss
            
            # Calculate value and policy losses separately
            try:
                value_loss = F.mse_loss(values.squeeze(), rewards_tensor).item()
                estimated_policy_loss = ppo_loss.item() - (0.5 * value_loss)
                policy_loss = max(0.0, estimated_policy_loss)
            except:
                value_loss = None
                policy_loss = None
            
            # Calculate entropy if available
            try:
                if hasattr(action_dist, 'entropy'):
                    entropy = action_dist.entropy().mean().item()
                else:
                    entropy = None
            except:
                entropy = None
        
        # Add regularization if needed
        if self.update_count % 100 == 0:
            # L2 regularization on weights (small)
            l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters()) * 1e-5
            total_loss = total_loss + l2_reg
        
        # Backward pass with gradient tracking
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Track gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.gradient_norm_history.append(grad_norm.item())
        
        # Adaptive gradient clipping based on history
        if len(self.gradient_norm_history) > 10:
            avg_grad_norm = np.mean(list(self.gradient_norm_history)[-10:])
            if avg_grad_norm > 1.0:
                # Increase clipping if gradients are large
                self.clip_grad_norm = min(1.0, self.clip_grad_norm * 1.1)
            elif avg_grad_norm < 0.1:
                # Decrease clipping if gradients are small
                self.clip_grad_norm = max(0.1, self.clip_grad_norm * 0.9)
        
        self.optimizer.step()
        
        # Update learning rate based on performance
        self.scheduler.step(total_loss.item())
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update model tracker with enhanced metrics
        self.update_count += 1
        
        # Track training progress with enhanced metrics (calculate early for use below)
        human_in_batch = sum(1 for exp in experiences if exp.get("human_intervention", False))
        # human_action_count was calculated earlier from recent_experiences (line 206-207)
        # It's already in scope, so we can use it directly
        
        # Update curriculum if enabled
        if self.curriculum_enabled:
            self._update_curriculum(human_in_batch, total_loss.item())
        
        # Model is updated immediately - actor will use new weights on next inference
        # Since we're using threading (not multiprocessing), model updates are immediately visible
        
        # Anneal BC weight over time (but keep it higher if we have human actions)
        if self.update_count % 100 == 0:
            # Keep BC weight higher if we're actively learning from human
            if human_action_count > 0:
                # Maintain much higher BC weight when learning from human (critical for emulation)
                self.beta_bc = max(1.5, self.beta_bc * 0.999)  # Very slow decay - prioritize human actions
            else:
                self.beta_bc = max(0.1, self.beta_bc * 0.995)  # Normal decay
        
        # Update priorities based on TD errors
        if self.update_count % config.UPDATE_FREQUENCY == 0:
            if self.is_dqn:
                # For DQN, calculate TD errors from Q-values
                with torch.no_grad():
                    q_values = self.model(states_tensor)
                    td_errors = torch.abs(rewards_tensor.unsqueeze(1) - q_values.gather(1, actions_tensor.long().unsqueeze(1))).squeeze()
            else:
                # For PPO, need values tensor (only available in PPO path)
                if 'values' in locals():
                    td_errors = self._calculate_td_errors(values, rewards_tensor, dones)
                else:
                    # Fallback: use reward as TD error estimate
                    td_errors = torch.abs(rewards_tensor)
            
            # Ensure td_errors is at least 1D (handle scalar case)
            td_errors_np = td_errors.cpu().numpy()
            if td_errors_np.ndim == 0:
                # Scalar (0-d array) - convert to 1D array
                td_errors_np = np.array([td_errors_np])
            
            # Only update if we have valid errors and indices
            if td_errors_np.ndim > 0 and len(td_errors_np) > 0 and len(indices) > 0:
                self.replay_buffer.update_priorities(indices, td_errors_np)
        
        # Update target network for DQN (ScrimBrain guide: every N steps)
        if self.is_dqn and self.target_model and self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logger.debug(f"DQN target network updated at step {self.update_count}")
        
        # Decay epsilon for DQN exploration (ScrimBrain guide)
        if self.is_dqn:
            self.epsilon = max(self.epsilon_end, self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay)
        
        # Convert value_loss to scalar if needed
        value_loss_scalar = value_loss if isinstance(value_loss, (int, float)) else (value_loss.item() if isinstance(value_loss, torch.Tensor) else None)
        policy_loss_scalar = policy_loss if isinstance(policy_loss, (int, float)) else (policy_loss.item() if isinstance(policy_loss, torch.Tensor) else None)
        
        self.model_tracker.record_training_step(
            total_loss.item(),
            ppo_loss.item(),
            bc_loss.item(),
            human_in_batch,
            len(experiences),
            value_loss=value_loss_scalar,
            policy_loss=policy_loss_scalar,
            entropy=entropy,
            gradient_norm=grad_norm.item(),
            learning_rate=current_lr
        )
        
        # Record enhanced training loss in performance monitor
        if self.performance_monitor:
            self.performance_monitor.record_training_loss(
                total_loss.item(),
                ppo_loss.item(),
                bc_loss.item(),
                value_loss=value_loss_scalar,
                policy_loss=policy_loss_scalar,
                entropy=entropy,
                gradient_norm=grad_norm.item(),
                learning_rate=current_lr
            )
        
        # REAL-TIME training progress logging - every step!
        # Always log training metrics for real-time visibility
        stats = self.model_tracker.get_training_stats() if hasattr(self.model_tracker, 'get_training_stats') else {}
        trend = stats.get('trend', 'unknown') if stats else 'unknown'
        
        # Log every step for real-time updates
        if self.is_dqn:
            # Format value_loss_display properly (can't use ternary in format specifier)
            value_loss_display = value_loss if isinstance(value_loss, (int, float)) else (value_loss.item() if isinstance(value_loss, torch.Tensor) else 'N/A')
            value_str = f"{value_loss_display:.4f}" if isinstance(value_loss_display, (int, float)) else str(value_loss_display)
            bc_loss_display = bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss
            bc_str = f"{bc_loss_display:.4f}" if isinstance(bc_loss_display, (int, float)) else str(bc_loss_display)
            logger.info(f"TRAINING | Step {self.update_count} | "
                       f"Loss: {total_loss.item():.3f} | DQN: {dqn_loss.item():.3f} | BC: {bc_str} | "
                       f"Human: {human_in_batch}/{len(experiences)} | Buffer: {len(self.replay_buffer)} | {trend}")
        else:
            # Format values properly (can't use ternary in format specifier)
            value_str = f"{value_loss:.4f}" if value_loss is not None else "N/A"
            policy_str = f"{policy_loss:.4f}" if policy_loss is not None else "N/A"
            entropy_str = f"{entropy:.4f}" if entropy is not None else "N/A"
            logger.info(f"TRAINING | Step {self.update_count} | "
                       f"Loss: {total_loss.item():.3f} | PPO: {ppo_loss.item():.3f} | BC: {bc_loss.item():.3f} | "
                       f"Human: {human_in_batch}/{len(experiences)} | Buffer: {len(self.replay_buffer)} | {trend}")
        
        # Detailed metrics for debugging (less verbose)
        if self.update_count % 10 == 0:
            logger.debug(f"Training Metrics: "
                        f"Buffer size: {len(self.replay_buffer)} | "
                        f"Batch size: {len(experiences)} | "
                        f"Human actions in batch: {human_in_batch} | "
                        f"Curriculum stage: {self.curriculum_stage if self.curriculum_enabled else 'N/A'}")
        
        # Save checkpoint periodically
        if self.update_count % config.SAVE_INTERVAL == 0:
            checkpoint_path = f"{config.MODEL_SAVE_PATH}/checkpoint_step_{self.update_count}.pt"
            self.model_tracker.save_checkpoint(checkpoint_path)
            self.last_checkpoint_save = time.time()
    
    def _prepare_states(self, states: list) -> torch.Tensor:
        """Prepare state tensors - expects (T, H, W) frame stacks"""
        # Handle frame stacks
        if isinstance(states[0], np.ndarray):
            if len(states[0].shape) == 3:  # (T, H, W) - frame stack
                # Convert list of (T, H, W) arrays to batch tensor
                # Result: (batch, T, H, W) where T = frame_stack_size (channels)
                batch = torch.FloatTensor(np.array(states)).to(config.DEVICE)
                return batch
            elif len(states[0].shape) == 2:  # (H, W) - single frame
                # Single frame, repeat to create frame stack
                batch_list = []
                for state in states:
                    # Repeat single frame to match frame_stack_size
                    stacked = np.stack([state] * config.FRAME_STACK_SIZE, axis=0)
                    batch_list.append(stacked)
                batch = torch.FloatTensor(np.array(batch_list)).to(config.DEVICE)
                return batch
        
        # Fallback: create dummy tensor (batch, T, H, W)
        batch_size = len(states)
        return torch.zeros(batch_size, config.FRAME_STACK_SIZE, config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH).to(config.DEVICE)
    
    def _prepare_state_features(self, experiences: list) -> torch.Tensor:
        """Extract state features from experiences"""
        features = []
        for exp in experiences:
            # Extract from state if available, otherwise use defaults
            state = exp.get("state_features", {})
            feat = [
                state.get("health", 100.0) / 100.0,
                state.get("stamina", 100.0) / 100.0,
                state.get("enemy_health", 100.0) / 100.0,
                1.0 if state.get("is_dead", False) else 0.0,
                1.0 if state.get("enemy_dead", False) else 0.0,
                state.get("position", {}).get("x", 0.0) / 1000.0,  # Normalize
                state.get("position", {}).get("y", 0.0) / 1000.0
            ]
            features.append(feat)
        
        return torch.FloatTensor(features).to(config.DEVICE)
    
    def _calculate_ppo_loss(self, action_dist, discrete_probs, values,
                            actions, rewards, weights, experiences) -> torch.Tensor:
        """Calculate PPO loss"""
        # Sample actions from current policy
        sampled_actions = action_dist.sample()
        
        # Calculate log probabilities
        log_probs = action_dist.log_prob(sampled_actions).sum(dim=1)
        old_log_probs = action_dist.log_prob(actions[:, :2]).sum(dim=1)  # Only continuous part
        
        # Calculate advantages (simplified - real PPO needs GAE)
        advantages = rewards - values.squeeze()
        
        # PPO clip
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - config.PPO_CLIP, 1 + config.PPO_CLIP)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        # Entropy bonus (encourage exploration)
        entropy = action_dist.entropy().mean()
        
        ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return ppo_loss
    
    def _calculate_bc_loss(self, action_dist, discrete_probs, actions,
                          human_actions, human_interventions) -> torch.Tensor:
        """Enhanced Behavioral Cloning loss with uncertainty estimation and focal loss"""
        if not any(human_interventions):
            return torch.tensor(0.0).to(config.DEVICE)
        
        bc_losses = []
        uncertainties = []
        
        for i, (human_action, is_human) in enumerate(zip(human_actions, human_interventions)):
            if is_human and human_action is not None:
                # Convert to tensor
                if isinstance(human_action, (list, np.ndarray)):
                    human_action = torch.FloatTensor(human_action).to(config.DEVICE)
                
                # Calculate uncertainty (variance in action distribution)
                if hasattr(action_dist, 'variance'):
                    uncertainty = action_dist.variance[i].mean()
                    uncertainties.append(uncertainty.item())
                else:
                    # Estimate uncertainty from standard deviation
                    if hasattr(action_dist, 'stddev'):
                        uncertainty = action_dist.stddev[i].mean()
                    else:
                        uncertainty = torch.tensor(0.5)  # Default uncertainty
                
                # Continuous action loss with uncertainty weighting
                pred_mean = action_dist.mean[i]
                target = human_action[:2]
                
                # Use Huber loss if enabled for robustness
                if self.use_huber_loss:
                    continuous_loss = F.smooth_l1_loss(pred_mean, target, delta=self.huber_delta)
                else:
                    continuous_loss = F.mse_loss(pred_mean, target)
                
                # Weight by uncertainty (higher uncertainty = more important to learn)
                uncertainty_weight = 1.0 + uncertainty * 0.5
                continuous_loss = continuous_loss * uncertainty_weight
                
                # Discrete action loss with focal loss if enabled
                if len(human_action) > 2:
                    discrete_target = human_action[2:].unsqueeze(0)
                    discrete_probs_clamped = torch.clamp(discrete_probs[i].unsqueeze(0), 1e-7, 1.0 - 1e-7)
                    discrete_target_clamped = torch.clamp(discrete_target, 0.0, 1.0)
                    
                    if self.use_focal_loss:
                        # Focal loss for better handling of hard examples
                        ce_loss = F.binary_cross_entropy(discrete_probs_clamped, discrete_target_clamped, reduction='none')
                        p_t = discrete_probs_clamped * discrete_target_clamped + (1 - discrete_probs_clamped) * (1 - discrete_target_clamped)
                        focal_weight = (1 - p_t) ** self.focal_gamma
                        discrete_loss = (self.focal_alpha * focal_weight * ce_loss).mean()
                    else:
                        discrete_loss = F.binary_cross_entropy(discrete_probs_clamped, discrete_target_clamped)
                else:
                    discrete_loss = torch.tensor(0.0).to(config.DEVICE)
                
                bc_losses.append(continuous_loss + discrete_loss)
        
        if len(bc_losses) == 0:
            return torch.tensor(0.0).to(config.DEVICE)
        
        # Track uncertainty for curriculum learning
        if uncertainties:
            self.uncertainty_history.extend(uncertainties)
        
        return torch.stack(bc_losses).mean()
    
    def _update_curriculum(self, human_in_batch: int, current_loss: float):
        """Update curriculum learning stage based on performance"""
        if self.curriculum_stage >= len(self.curriculum_stages) - 1:
            return  # Already at max stage
        
        current_stage = self.curriculum_stages[self.curriculum_stage]
        
        # Check performance periodically
        current_time = time.time()
        if current_time - self.last_performance_check < self.performance_check_interval:
            return
        
        self.last_performance_check = current_time
        
        # Check if we should advance curriculum
        # Criteria: low loss, low uncertainty, high human action ratio decreasing
        stats = self.model_tracker.get_training_stats()
        
        # Calculate performance score (0-1)
        if len(self.model_tracker.improvement_metrics['loss_trend']) >= 50:
            recent_loss = np.mean(list(self.model_tracker.improvement_metrics['loss_trend'])[-50:])
            performance_score = 1.0 / (1.0 + recent_loss)  # Higher loss = lower score
            
            # Check uncertainty
            avg_uncertainty = np.mean(list(self.uncertainty_history)[-100:]) if len(self.uncertainty_history) >= 100 else 0.5
            uncertainty_score = 1.0 - min(1.0, avg_uncertainty)  # Lower uncertainty = higher score
            
            # Combined performance
            combined_score = (performance_score + uncertainty_score) / 2.0
            
            if combined_score >= self.curriculum_progression_threshold:
                # Advance to next stage
                self.curriculum_stage += 1
                self.curriculum_difficulty = self.curriculum_stage / len(self.curriculum_stages)
                logger.info(f"Curriculum advanced to stage {self.curriculum_stage}: {self.curriculum_stages[self.curriculum_stage]['name']}")
    
    def get_curriculum_params(self) -> Dict:
        """Get current curriculum parameters"""
        if not self.curriculum_enabled:
            return {}
        
        current_stage = self.curriculum_stages[min(self.curriculum_stage, len(self.curriculum_stages) - 1)]
        return {
            'stage': self.curriculum_stage,
            'stage_name': current_stage['name'],
            'difficulty': self.curriculum_difficulty,
            'human_ratio': current_stage['human_ratio'],
            'exploration': current_stage['exploration']
        }
    
    def _calculate_dqn_loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                           next_states: list, dones: list, weights: torch.Tensor, experiences: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate DQN loss (ScrimBrain-style) with Behavioral Cloning for human actions
        
        Args:
            states: Current states (batch, T, H, W)
            actions: Action indices (batch,)
            rewards: Rewards (batch,)
            next_states: Next states (list)
            dones: Done flags (list)
            weights: Importance weights (batch,)
            experiences: Experience tuples
            
        Returns:
            (total_loss, dqn_loss, value_loss, bc_loss)
        """
        # Get Q-values for current states
        q_values = self.model(states)  # (batch, num_actions)
        
        # Get Q-values for actions taken
        actions_long = actions.long()  # Ensure integer actions
        # Handle both scalar and 1D tensor cases
        if actions_long.dim() == 0:
            actions_long = actions_long.unsqueeze(0)
        if actions_long.dim() == 1 and actions_long.shape[0] == 1:
            # Single action, ensure proper shape for gather
            actions_long = actions_long.unsqueeze(1)
        elif actions_long.dim() == 1:
            # Multiple actions, add dimension for gather
            actions_long = actions_long.unsqueeze(1)
        
        q_values_selected = q_values.gather(1, actions_long).squeeze(1)  # (batch,)
        
        # Calculate target Q-values using target network
        with torch.no_grad():
            if self.target_model and len(next_states) > 0 and next_states[0] is not None:
                # Prepare next states
                next_states_tensor = self._prepare_states(next_states)
                next_q_values = self.target_model(next_states_tensor)  # (batch, num_actions)
                next_q_max = next_q_values.max(dim=1)[0]  # (batch,)
            else:
                next_q_max = torch.zeros_like(rewards)
            
            # Calculate target: r + gamma * max(Q(s', a')) * (1 - done)
            dones_tensor = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(config.DEVICE)
            target_q = rewards + config.GAMMA * next_q_max * (1.0 - dones_tensor)
        
        # DQN loss (Huber loss for robustness)
        td_errors = target_q - q_values_selected
        dqn_loss = F.smooth_l1_loss(q_values_selected, target_q, reduction='none')
        
        # Apply importance weights
        dqn_loss = (dqn_loss * weights).mean()
        
        # BEHAVIORAL CLONING LOSS for human actions - CRITICAL for learning from human movements
        bc_loss = torch.tensor(0.0).to(config.DEVICE)
        human_action_count = 0
        
        # Extract human actions from experiences
        human_interventions = [exp.get("human_intervention", False) for exp in experiences]
        human_actions = []
        for i, exp in enumerate(experiences):
            if exp.get("human_intervention", False):
                # Get the human's discrete action (already converted during batch prep)
                human_action = actions_long[i].item() if i < len(actions_long) else None
                if human_action is not None:
                    human_actions.append((i, human_action))
                    human_action_count += 1
        
        # Calculate BC loss: model should predict high Q-value for human's chosen action
        if human_action_count > 0:
            bc_losses = []
            for i, human_action_id in human_actions:
                # Get Q-value for human's action
                human_q = q_values[i, human_action_id]
                # Get max Q-value (what model currently thinks is best)
                max_q = q_values[i].max()
                # BC loss: encourage model to assign high Q-value to human's action
                # Use cross-entropy style loss: -log(softmax(Q)[human_action])
                log_probs = F.log_softmax(q_values[i], dim=0)
                bc_loss_i = -log_probs[human_action_id]  # Negative log probability of human action
                bc_losses.append(bc_loss_i)
            
            if bc_losses:
                # Weight BC loss more heavily for human actions (5x multiplier)
                bc_loss = torch.stack(bc_losses).mean() * config.HUMAN_ACTION_PRIORITY_MULTIPLIER
                # Also increase weight for human actions in DQN loss
                human_mask = torch.tensor([exp.get("human_intervention", False) for exp in experiences]).to(config.DEVICE)
                human_weight_multiplier = torch.where(human_mask, torch.tensor(5.0).to(config.DEVICE), torch.tensor(1.0).to(config.DEVICE))
                dqn_loss = (dqn_loss.unsqueeze(0) * human_weight_multiplier).mean() if dqn_loss.dim() == 0 else (dqn_loss * human_weight_multiplier).mean()
        
        # Value loss (for tracking)
        value_loss = F.mse_loss(q_values_selected, target_q)
        
        # Total loss combines DQN loss with BC loss
        # Use adaptive BC weight that's higher when we have human actions
        # Increase BC weight significantly when we have human actions (for better learning)
        bc_weight = self.beta_bc * 2.0 if human_action_count > 0 else 0.0  # Double BC weight for human actions
        total_loss = dqn_loss + bc_weight * bc_loss
        
        # Log BC loss when human actions are present
        if human_action_count > 0 and self.update_count % 10 == 0:
            logger.info(f"LEARNING | Human: {human_action_count}/{len(experiences)} | "
                       f"BC Loss: {bc_loss.item():.4f} | Weight: {bc_weight:.2f} | "
                       f"Total: {total_loss.item():.4f}")
        
        return total_loss, dqn_loss, value_loss, bc_loss
    
    def _calculate_td_errors(self, values: torch.Tensor, rewards: torch.Tensor, dones: list) -> torch.Tensor:
        """Calculate TD errors for priority updates"""
        # Simplified TD error calculation
        td_errors = torch.abs(rewards - values.squeeze())
        # Ensure at least 1D (handle scalar case)
        if td_errors.dim() == 0:
            td_errors = td_errors.unsqueeze(0)
        return td_errors

