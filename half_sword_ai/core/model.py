"""
Neural Network Architecture for Half Sword Agent
CNN for visual processing + MLP for action prediction
Supports both DQN (ScrimBrain-style) and PPO models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Union
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

# Import DQN model
try:
    from half_sword_ai.core.dqn_model import DQNNetwork, create_dqn_model
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    logger.warning("DQN model not available")

class HalfSwordPolicyNetwork(nn.Module):
    """
    Policy network for Half Sword combat
    Processes visual frames and outputs continuous mouse control + discrete buttons
    """
    
    def __init__(self, frame_stack_size: int = None, action_dim: int = 6):
        super().__init__()
        self.frame_stack_size = frame_stack_size or config.FRAME_STACK_SIZE
        self.action_dim = action_dim  # [mouse_x, mouse_y, left_click, right_click, space, alt]
        
        # Visual encoder (CNN) - Enhanced with deeper architecture inspired by ScrimBrain
        # ScrimBrain uses deeper CNNs for better feature extraction from screen captures
        self.conv1 = nn.Conv2d(self.frame_stack_size, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # Additional conv layer for deeper feature extraction (ScrimBrain-style)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate conv output size (for 84x84 input - ScrimBrain standard)
        # Support both 84x84 and 96x96 for backward compatibility
        conv_out_size = self._get_conv_out_size(84, 84)
        
        # State encoder (for memory-based features)
        self.state_encoder = nn.Sequential(
            nn.Linear(7, 32),  # health, stamina, enemy_health, is_dead, enemy_dead, pos_x, pos_y
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Combined feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_out_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Policy head (outputs mean and std for continuous actions)
        self.policy_mean = nn.Linear(256, 2)  # mouse_x, mouse_y
        self.policy_std = nn.Linear(256, 2)
        
        # Discrete action head (button presses)
        self.discrete_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # left, right, space, alt
            nn.Sigmoid()  # Probabilities
        )
        
        # Value head (for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_out_size(self, h: int, w: int) -> int:
        """Calculate convolutional output size"""
        x = torch.zeros(1, self.frame_stack_size, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # Include additional conv layer
        return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, frames: torch.Tensor, state_features: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            frames: (batch, T, H, W) or (batch, H, W) frame stack
            state_features: (batch, 7) state features from memory
        
        Returns:
            (action_distribution, discrete_actions, value)
        """
        # Handle frame input
        # Expected input: (batch, T, H, W) where T = frame_stack_size (channels)
        # If we get (batch, H, W), we need to add T dimension
        if len(frames.shape) == 3:
            # (batch, H, W) -> (batch, 1, H, W) -> repeat to (batch, T, H, W)
            frames = frames.unsqueeze(1).repeat(1, self.frame_stack_size, 1, 1)
        elif len(frames.shape) == 4 and frames.shape[1] != self.frame_stack_size:
            # If channels don't match, adjust
            if frames.shape[1] == 1:
                # Single channel, repeat to match frame_stack_size
                frames = frames.repeat(1, self.frame_stack_size, 1, 1)
        
        # Visual encoding - frames should now be (batch, T, H, W) where T = frame_stack_size
        # Enhanced CNN with deeper layers (ScrimBrain-style)
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # Additional layer for better feature extraction
        x = x.view(x.size(0), -1)  # Flatten
        
        # State encoding
        state_encoded = self.state_encoder(state_features)
        
        # Safety check: ensure no NaN in intermediate values
        if torch.isnan(x).any():
            logger.warning("NaN in visual encoding, using zeros")
            x = torch.zeros_like(x)
        if torch.isnan(state_encoded).any():
            logger.warning("NaN in state encoding, using zeros")
            state_encoded = torch.zeros_like(state_encoded)
        
        # Combine features
        combined = torch.cat([x, state_encoded], dim=1)
        features = self.feature_extractor(combined)
        
        # Final safety check
        if torch.isnan(features).any():
            logger.warning("NaN in features, using zeros")
            features = torch.zeros_like(features)
        
        # Policy outputs
        mean = self.policy_mean(features)
        std = F.softplus(self.policy_std(features)) + 1e-5  # Ensure positive
        
        # Safety check: prevent NaN values
        mean = torch.clamp(mean, -10.0, 10.0)  # Clamp to reasonable range
        std = torch.clamp(std, 1e-5, 5.0)  # Ensure std is positive and bounded
        
        # Check for NaN and replace with zeros if found
        if torch.isnan(mean).any():
            logger.warning("NaN detected in policy mean, replacing with zeros")
            mean = torch.zeros_like(mean)
        if torch.isnan(std).any():
            logger.warning("NaN detected in policy std, replacing with defaults")
            std = torch.ones_like(std) * 0.1
        
        action_dist = torch.distributions.Normal(mean, std)
        
        # Discrete actions
        discrete_probs = self.discrete_head(features)
        
        # Value estimate
        value = self.value_head(features)
        
        return action_dist, discrete_probs, value
    
    def get_action(self, frames: torch.Tensor, state_features: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy
        
        Args:
            frames: Input frames
            state_features: State features
            deterministic: If True, use mean instead of sampling
        
        Returns:
            (continuous_action, discrete_action)
        """
        with torch.no_grad():
            action_dist, discrete_probs, _ = self.forward(frames, state_features)
            
            if deterministic:
                continuous_action = action_dist.mean
            else:
                continuous_action = action_dist.sample()
            
            # Sample discrete actions
            discrete_action = (discrete_probs > 0.5).float()
            
            return continuous_action, discrete_action

# Helper function to create model - supports both DQN and PPO
def create_model(device: str = None, use_dqn: bool = None) -> Union[HalfSwordPolicyNetwork, 'DQNNetwork']:
    """
    Create and initialize model
    
    Args:
        device: Device to use (cuda/cpu)
        use_dqn: If True, use DQN; if False, use PPO; if None, use config.USE_DISCRETE_ACTIONS
        
    Returns:
        Model instance (DQNNetwork or HalfSwordPolicyNetwork)
    """
    device = device or config.DEVICE
    
    # Determine which model to use
    if use_dqn is None:
        use_dqn = config.USE_DISCRETE_ACTIONS
    
    # Use DQN if discrete actions enabled and DQN is available
    if use_dqn and DQN_AVAILABLE:
        logger.info("Creating DQN model (ScrimBrain-style) for discrete actions")
        model = create_dqn_model(device=device)
        return model
    else:
        # Use PPO for continuous actions
        if use_dqn and not DQN_AVAILABLE:
            logger.warning("DQN requested but not available, falling back to PPO")
        logger.info("Creating PPO model for continuous actions")
        model = HalfSwordPolicyNetwork()
        model = model.to(device)
        return model

