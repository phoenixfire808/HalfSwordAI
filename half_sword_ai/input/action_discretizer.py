"""
Action Discretization - ScrimBrain Integration
Maps continuous control space to discrete macro-actions for DQN compatibility
Based on Table 1 from the ScrimBrain Half Sword Integration Guide
"""
import logging
import numpy as np
from typing import Dict, List, Tuple
from enum import IntEnum
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

class MacroAction(IntEnum):
    """Discrete macro-actions for Half Sword combat"""
    NEUTRAL_RESET = 0
    HIGH_GUARD_LEFT = 1
    HIGH_GUARD_RIGHT = 2
    LOW_GUARD_LEFT = 3
    LOW_GUARD_RIGHT = 4
    OVERHEAD_STRIKE = 5
    HORIZONTAL_SLASH_L = 6
    HORIZONTAL_SLASH_R = 7
    THRUST = 8

class ActionDiscretizer:
    """
    Discretizes continuous actions into macro-actions
    Implements Table 1 from ScrimBrain integration guide
    """
    
    def __init__(self):
        # Action mapping table (from guide Table 1)
        self.action_table = {
            MacroAction.NEUTRAL_RESET: {
                'name': 'Neutral / Reset',
                'dx': 0,
                'dy': 0,
                'duration_ms': 100,
                'keys': {}
            },
            MacroAction.HIGH_GUARD_LEFT: {
                'name': 'High Guard Left',
                'dx': -50,
                'dy': -50,
                'duration_ms': 200,
                'keys': {}
            },
            MacroAction.HIGH_GUARD_RIGHT: {
                'name': 'High Guard Right',
                'dx': 50,
                'dy': -50,
                'duration_ms': 200,
                'keys': {}
            },
            MacroAction.LOW_GUARD_LEFT: {
                'name': 'Low Guard Left',
                'dx': -50,
                'dy': 50,
                'duration_ms': 200,
                'keys': {}
            },
            MacroAction.LOW_GUARD_RIGHT: {
                'name': 'Low Guard Right',
                'dx': 50,
                'dy': 50,
                'duration_ms': 200,
                'keys': {}
            },
            MacroAction.OVERHEAD_STRIKE: {
                'name': 'Overhead Strike',
                'dx': 0,
                'dy': 400,
                'duration_ms': 50,
                'keys': {'left': True}  # Hold left mouse button for attack
            },
            MacroAction.HORIZONTAL_SLASH_L: {
                'name': 'Horizontal Slash L',
                'dx': 400,
                'dy': 0,
                'duration_ms': 50,
                'keys': {'left': True}  # Hold left mouse button for attack
            },
            MacroAction.HORIZONTAL_SLASH_R: {
                'name': 'Horizontal Slash R',
                'dx': -400,
                'dy': 0,
                'duration_ms': 50,
                'keys': {'left': True}  # Hold left mouse button for attack
            },
            MacroAction.THRUST: {
                'name': 'Thrust',
                'dx': 0,
                'dy': -100,
                'duration_ms': 150,
                'keys': {'alt': True}  # Hold ALT for thrust
            }
        }
        
        self.num_actions = len(self.action_table)
        logger.info(f"Action discretizer initialized with {self.num_actions} macro-actions")
    
    def get_action_config(self, action_id: int) -> Dict:
        """
        Get action configuration for given action ID
        
        Args:
            action_id: Action index (0-8)
            
        Returns:
            Action configuration dictionary
        """
        if action_id < 0 or action_id >= self.num_actions:
            logger.warning(f"Invalid action_id: {action_id}, using NEUTRAL_RESET")
            action_id = MacroAction.NEUTRAL_RESET
        
        return self.action_table[action_id].copy()
    
    def get_action_name(self, action_id: int) -> str:
        """Get human-readable action name"""
        if action_id < 0 or action_id >= self.num_actions:
            return "INVALID"
        return self.action_table[action_id]['name']
    
    def get_num_actions(self) -> int:
        """Get total number of discrete actions"""
        return self.num_actions
    
    def map_continuous_to_discrete(self, dx: float, dy: float, buttons: Dict[str, bool] = None) -> int:
        """
        Map continuous action to nearest discrete action
        Enhanced mapping that preserves more human movement patterns
        Now includes movement keys (WASD) and grab mechanics (Q/E)
        
        Args:
            dx: Normalized X movement (-1 to 1) OR pixel movement
            dy: Normalized Y movement (-1 to 1) OR pixel movement
            buttons: Button states (includes WASD, Q/E, mouse buttons, etc.)
            
        Returns:
            Discrete action ID
        """
        if buttons is None:
            buttons = {}
        
        # Check for thrust (ALT key) - highest priority
        if buttons.get('alt', False):
            return MacroAction.THRUST
        
        # Note: Movement keys (WASD) are handled separately in the action space
        # The discrete actions focus on sword movements, while WASD is continuous
        # This allows the agent to combine movement with sword techniques
        
        # Normalize if needed (handle both normalized and pixel values)
        # If values are > 1, assume they're pixels and normalize
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            # Pixel values - normalize by sensitivity
            dx_norm = dx / config.MOUSE_SENSITIVITY if hasattr(config, 'MOUSE_SENSITIVITY') else dx / 100.0
            dy_norm = dy / config.MOUSE_SENSITIVITY if hasattr(config, 'MOUSE_SENSITIVITY') else dy / 100.0
            dx_norm = max(-1.0, min(1.0, dx_norm))
            dy_norm = max(-1.0, min(1.0, dy_norm))
        else:
            dx_norm = dx
            dy_norm = dy
        
        abs_dx = abs(dx_norm)
        abs_dy = abs(dy_norm)
        
        # Small movements -> neutral/reset
        if abs_dx < 0.15 and abs_dy < 0.15:
            return MacroAction.NEUTRAL_RESET
        
        # Calculate magnitude to determine if it's a strike or guard
        magnitude = np.sqrt(dx_norm**2 + dy_norm**2)
        
        # Large magnitude movements (>0.7) are strikes, smaller are guards
        is_strike = magnitude > 0.7
        
        if is_strike:
            # Strike actions - prioritize direction
            if abs_dy > abs_dx * 1.5:
                # Primarily vertical
                if dy_norm > 0:
                    return MacroAction.OVERHEAD_STRIKE
                else:
                    # Upward strike - map to guard based on horizontal component
                    if dx_norm < -0.2:
                        return MacroAction.HIGH_GUARD_LEFT
                    elif dx_norm > 0.2:
                        return MacroAction.HIGH_GUARD_RIGHT
                    else:
                        return MacroAction.HIGH_GUARD_LEFT  # Default
            else:
                # Primarily horizontal
                if dx_norm > 0:
                    return MacroAction.HORIZONTAL_SLASH_L
                else:
                    return MacroAction.HORIZONTAL_SLASH_R
        else:
            # Guard actions - slower, more controlled movements
            if abs_dy > abs_dx:
                # Vertical guard
                if dy_norm < 0:  # Upward
                    if dx_norm < -0.1:
                        return MacroAction.HIGH_GUARD_LEFT
                    elif dx_norm > 0.1:
                        return MacroAction.HIGH_GUARD_RIGHT
                    else:
                        return MacroAction.HIGH_GUARD_LEFT  # Default to left
                else:  # Downward
                    if dx_norm < -0.1:
                        return MacroAction.LOW_GUARD_LEFT
                    elif dx_norm > 0.1:
                        return MacroAction.LOW_GUARD_RIGHT
                    else:
                        return MacroAction.LOW_GUARD_LEFT  # Default to left
            else:
                # Horizontal guard
                if dx_norm > 0:
                    return MacroAction.LOW_GUARD_RIGHT
                else:
                    return MacroAction.LOW_GUARD_LEFT

