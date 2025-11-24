"""
Gesture Engine - ScrimBrain Integration
Converts discrete macro-actions into smooth physics-compatible mouse gestures
Implements micro-step interpolation for realistic weapon movement
"""
import time
import logging
from typing import Tuple, List, Dict, Optional
from half_sword_ai.input.direct_input import DirectInput

logger = logging.getLogger(__name__)

class GestureEngine:
    """
    Converts high-level actions into smooth mouse gestures
    Breaks down macro-actions into micro-steps for physics engine compatibility
    """
    
    def __init__(self, direct_input: DirectInput):
        from half_sword_ai.config import config
        self.direct_input = direct_input
        self.micro_step_duration = config.GESTURE_MICRO_STEP_DURATION  # From config (guide: 10ms)
        self.min_step_size = 1  # Minimum pixels per step
        
    def perform_smooth_gesture(self, total_dx: int, total_dy: int, duration_ms: int) -> bool:
        """
        Perform smooth gesture by breaking into micro-steps
        
        Args:
            total_dx: Total X movement in pixels
            total_dy: Total Y movement in pixels
            duration_ms: Total duration in milliseconds
            
        Returns:
            True if successful
        """
        if total_dx == 0 and total_dy == 0:
            return True
        
        # Calculate number of steps
        num_steps = max(1, int(duration_ms / (self.micro_step_duration * 1000)))
        num_steps = max(1, min(num_steps, abs(total_dx) + abs(total_dy)))  # At least 1 step per pixel
        
        # Calculate step size
        step_dx = total_dx / num_steps
        step_dy = total_dy / num_steps
        
        # Perform micro-steps
        for i in range(num_steps):
            # Round to integer pixels
            dx = int(round(step_dx))
            dy = int(round(step_dy))
            
            # Only move if step is significant enough
            if abs(dx) >= self.min_step_size or abs(dy) >= self.min_step_size:
                self.direct_input.move_mouse_relative(dx, dy)
            
            # Sleep for micro-step duration
            time.sleep(self.micro_step_duration)
        
        return True
    
    def perform_macro_action(self, action_id: int, action_config: Dict, buttons: Dict[str, bool] = None) -> bool:
        """
        Perform a macro-action from the action discretization table
        For attack swings, holds mouse button during movement
        
        Args:
            action_id: Action index (0-8)
            action_config: Action configuration with dx, dy, duration, keys
            buttons: Button states (for holding LMB/RMB during swings)
            
        Returns:
            True if successful
        """
        from half_sword_ai.input.action_discretizer import MacroAction
        
        dx = action_config.get('dx', 0)
        dy = action_config.get('dy', 0)
        duration_ms = action_config.get('duration_ms', 50)
        keys = action_config.get('keys', {})
        
        # Determine if this is an attack swing (strikes need button held)
        is_attack = action_id in [
            MacroAction.OVERHEAD_STRIKE,
            MacroAction.HORIZONTAL_SLASH_L,
            MacroAction.HORIZONTAL_SLASH_R,
            MacroAction.THRUST
        ]
        
        # Determine which mouse button to hold for attack
        mouse_button_to_hold = None
        if buttons:
            if buttons.get('left', False):
                mouse_button_to_hold = 'left'
            elif buttons.get('right', False):
                mouse_button_to_hold = 'right'
        # Default: use left button for attacks if no button specified
        elif is_attack:
            mouse_button_to_hold = 'left'
        
        # Hold mouse button BEFORE starting swing (critical for Half Sword)
        if is_attack and mouse_button_to_hold:
            self.direct_input.press_mouse_button(mouse_button_to_hold)
        
        # Handle keyboard key presses
        for key, pressed in keys.items():
            self.direct_input.set_key_state(key, pressed)
        
        # Perform mouse gesture WHILE button is held
        if dx != 0 or dy != 0:
            self.perform_smooth_gesture(dx, dy, duration_ms)
        
        # Release mouse button AFTER swing completes
        if is_attack and mouse_button_to_hold:
            self.direct_input.release_mouse_button(mouse_button_to_hold)
        
        # Release keyboard keys if needed
        for key, pressed in keys.items():
            if not pressed:
                self.direct_input.set_key_state(key, False)
        
        return True

