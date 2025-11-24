"""
Half Sword Environment - ScrimBrain Integration
Implements gym.Env interface for ScrimBrain-style RL training
Based on ScrimBrain architecture adapted for Half Sword physics combat
"""
import numpy as np
import gym
from gym import spaces
import logging
import time
from typing import Dict, Optional, Tuple, Any
from collections import deque

from half_sword_ai.config import config
from half_sword_ai.perception.vision import ScreenCapture, MemoryReader, VisionProcessor
from half_sword_ai.perception.ocr_reward_tracker import OCRRewardTracker
from half_sword_ai.perception.terminal_state_detector import TerminalStateDetector
from half_sword_ai.input.direct_input import DirectInput
from half_sword_ai.input.gesture_engine import GestureEngine
from half_sword_ai.input.action_discretizer import ActionDiscretizer, MacroAction

logger = logging.getLogger(__name__)

class HalfSwordEnv(gym.Env):
    """
    ScrimBrain-style environment wrapper for Half Sword
    Implements standard RL environment interface
    
    Observation Space: Box(0, 255, (4, 84, 84)) - Stacked grayscale frames
    Action Space: Discrete(9) - Macro-actions from Table 1
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.screen_capture = ScreenCapture(width=84, height=84, fps=60)
        self.memory_reader = MemoryReader()
        self.vision_processor = VisionProcessor(self.screen_capture)
        
        # Initialize reward tracking (ScrimBrain-style)
        if config.OCR_ENABLED:
            self.ocr_tracker = OCRRewardTracker()
        else:
            self.ocr_tracker = None
        
        # Initialize terminal state detection
        if config.TERMINAL_STATE_DETECTION:
            self.terminal_detector = TerminalStateDetector()
        else:
            self.terminal_detector = None
        
        # Initialize input system
        self.direct_input = DirectInput()
        self.gesture_engine = GestureEngine(self.direct_input)
        self.action_discretizer = ActionDiscretizer()
        
        # Frame stacking for temporal information
        self.frame_stack = deque(maxlen=config.FRAME_STACK_SIZE)
        self.frame_skip = config.FRAME_SKIP if hasattr(config, 'FRAME_SKIP') else 4
        
        # Define spaces per ScrimBrain guide
        # Observation: 4 stacked frames of 84x84 grayscale
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(config.FRAME_STACK_SIZE, 84, 84),
            dtype=np.uint8
        )
        
        # Action: Discrete macro-actions (0-8)
        self.action_space = spaces.Discrete(self.action_discretizer.get_num_actions())
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_score = 0
        self.current_score = 0
        
        # State tracking
        self.current_observation = None
        self.done = False
        self.info = {}
        
        logger.info("HalfSwordEnv initialized - ScrimBrain architecture")
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Frame skip: {self.frame_skip}")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial observation
        
        Returns:
            Initial observation (4, 84, 84) frame stack
        """
        logger.debug("Resetting environment...")
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.done = False
        self.last_score = 0
        self.current_score = 0
        
        # Clear frame stack
        self.frame_stack.clear()
        
        # Reset detectors
        if self.terminal_detector:
            self.terminal_detector.reset()
        
        # Capture initial frames to fill stack
        for _ in range(config.FRAME_STACK_SIZE):
            frame = self._capture_frame()
            if frame is not None:
                self.frame_stack.append(frame)
            else:
                # If capture fails, use zeros
                self.frame_stack.append(np.zeros((84, 84), dtype=np.uint8))
        
        # Create observation from stack
        self.current_observation = np.array(list(self.frame_stack), dtype=np.uint8)
        
        logger.debug(f"Environment reset - observation shape: {self.current_observation.shape}")
        
        return self.current_observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next observation, reward, done, info
        
        Args:
            action: Discrete action ID (0-8)
            
        Returns:
            observation: Next frame stack
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        if self.done:
            logger.warning("Episode already done, call reset() first")
            return self.current_observation, 0.0, True, self.info
        
        # Get action configuration
        action_config = self.action_discretizer.get_action_config(action)
        
        # Execute action with frame skip (for physics stability)
        for _ in range(self.frame_skip):
            # Perform gesture
            self.gesture_engine.perform_macro_action(action, action_config)
            
            # Capture frame
            frame = self._capture_frame()
            if frame is not None:
                self.frame_stack.append(frame)
        
        # Get current observation
        if len(self.frame_stack) >= config.FRAME_STACK_SIZE:
            self.current_observation = np.array(list(self.frame_stack), dtype=np.uint8)
        else:
            # Pad with last frame if needed
            while len(self.frame_stack) < config.FRAME_STACK_SIZE:
                self.frame_stack.append(self.frame_stack[-1] if self.frame_stack else np.zeros((84, 84), dtype=np.uint8))
            self.current_observation = np.array(list(self.frame_stack), dtype=np.uint8)
        
        # Calculate reward (ScrimBrain-style: score delta + survival)
        reward = self._calculate_reward()
        
        # Check terminal state
        self.done = self._check_terminal_state()
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1
        
        # Build info dictionary
        self.info = {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'score': self.current_score,
            'score_delta': self.current_score - self.last_score,
            'action_name': self.action_discretizer.get_action_name(action)
        }
        
        return self.current_observation, reward, self.done, self.info
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture and preprocess frame
        
        Returns:
            Preprocessed grayscale frame (84, 84) or None
        """
        frame = self.screen_capture.get_latest_frame()
        
        if frame is None:
            return None
        
        # Resize to 84x84 (ScrimBrain standard)
        if frame.shape[:2] != (84, 84):
            import cv2
            frame = cv2.resize(frame, (84, 84))
        
        # Ensure grayscale
        if len(frame.shape) == 3:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        return frame
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward using ScrimBrain approach:
        - Score delta from OCR
        - Survival bonus
        - Terminal state penalty
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Score-based reward (primary signal)
        if self.ocr_tracker:
            # Get latest frame for OCR
            frame = self.screen_capture.get_latest_frame()
            if frame is not None:
                ocr_result = self.ocr_tracker.update(frame)
                if ocr_result.get('success', False):
                    self.current_score = ocr_result.get('score', 0)
                    score_delta = ocr_result.get('score_delta', 0)
                    reward += float(score_delta)  # Score increase = reward
                    self.last_score = self.current_score
        
        # Survival bonus (small positive reward for staying alive)
        if not self.done:
            reward += 0.01
        
        # Terminal state penalty
        if self.done:
            reward -= 10.0  # Large penalty for death
        
        return reward
    
    def _check_terminal_state(self) -> bool:
        """
        Check if episode is done using terminal state detection
        
        Returns:
            True if episode is done
        """
        # Check terminal detector
        if self.terminal_detector:
            frame = self.screen_capture.get_latest_frame()
            if frame is not None:
                detection = self.terminal_detector.detect_death_screen(frame)
                if detection.get('is_terminal', False):
                    logger.info(f"Terminal state detected: {detection.get('reason', 'unknown')}")
                    return True
        
        # Check memory reader for death state
        if self.memory_reader:
            state = self.memory_reader.get_state()
            if state.get('is_dead', False):
                logger.info("Death detected from memory reader")
                return True
        
        return False
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render environment (for debugging)
        
        Args:
            mode: 'human' or 'rgb_array'
            
        Returns:
            Rendered frame or None
        """
        if mode == 'rgb_array':
            if self.current_observation is not None:
                # Return last frame from stack
                return self.current_observation[-1]
        elif mode == 'human':
            # Could display frame in window (not implemented)
            pass
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self.screen_capture:
            self.screen_capture.stop()
        logger.info("Environment closed")




