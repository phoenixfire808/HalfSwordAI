"""
DQN Model - ScrimBrain Integration
Discrete action space model for Half Sword combat
Based on ScrimBrain's DQN architecture adapted for physics-based combat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple
from half_sword_ai.config import config
from half_sword_ai.input.action_discretizer import ActionDiscretizer

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action space
    Based on ScrimBrain architecture with Dueling DQN head
    """
    
    def __init__(self, frame_stack_size: int = None, num_actions: int = 9):
        super().__init__()
        self.frame_stack_size = frame_stack_size or config.FRAME_STACK_SIZE
        self.num_actions = num_actions
        
        # Visual encoder (Nature CNN - from ScrimBrain)
        self.conv1 = nn.Conv2d(self.frame_stack_size, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate conv output size (for 84x84 input - ScrimBrain standard)
        conv_out_size = self._get_conv_out_size(84, 84)
        
        # Dueling DQN architecture (Value + Advantage streams)
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value estimate
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)  # Advantage per action
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_out_size(self, h: int, w: int) -> int:
        """Calculate convolutional output size"""
        x = torch.zeros(1, self.frame_stack_size, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        """Initialize network weights (orthogonal initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - outputs Q-values for each discrete action
        
        Args:
            frames: (batch, T, H, W) frame stack
            
        Returns:
            Q-values: (batch, num_actions)
        """
        # Handle frame input
        if len(frames.shape) == 3:
            frames = frames.unsqueeze(1).repeat(1, self.frame_stack_size, 1, 1)
        elif len(frames.shape) == 4 and frames.shape[1] != self.frame_stack_size:
            if frames.shape[1] == 1:
                frames = frames.repeat(1, self.frame_stack_size, 1, 1)
        
        # Visual encoding
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, frames: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            frames: Input frames
            epsilon: Exploration rate (0 = greedy, 1 = random)
            
        Returns:
            Action index (0 to num_actions-1)
        """
        with torch.no_grad():
            if np.random.random() < epsilon:
                # Random action
                return np.random.randint(0, self.num_actions)
            else:
                # Greedy action
                q_values = self.forward(frames)
                return q_values.argmax(dim=1).item()

# Cache discretizer to avoid repeated initialization
_discretizer_cache = None

def _get_cached_discretizer():
    """Get cached ActionDiscretizer instance"""
    global _discretizer_cache
    if _discretizer_cache is None:
        _discretizer_cache = ActionDiscretizer()
    return _discretizer_cache

# Helper function to create DQN model
def create_dqn_model(device: str = None, num_actions: int = None) -> DQNNetwork:
    """Create and initialize DQN model"""
    device = device or config.DEVICE
    
    # Get number of actions from discretizer (use cached instance)
    if num_actions is None:
        discretizer = _get_cached_discretizer()
        num_actions = discretizer.get_num_actions()
    
    model = DQNNetwork(num_actions=num_actions)
    model = model.to(device)
    return model

