"""
Configuration module for Half Sword AI Agent
Centralized configuration for all system components
"""
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class Config:
    # LLM configuration removed
    
    # Screen Capture Configuration - ScrimBrain Standard (84x84)
    CAPTURE_FPS: int = 60  # Target 60 FPS for smooth gameplay
    MIN_FPS_TARGET: int = 60  # Minimum acceptable FPS
    MAX_FPS_TARGET: int = 120  # Maximum FPS cap
    CAPTURE_WIDTH: int = 84  # ScrimBrain standard (was 96)
    CAPTURE_HEIGHT: int = 84  # ScrimBrain standard (was 96)
    CAPTURE_ROI: Tuple[int, int, int, int] = None  # Will be set to center of screen
    FRAME_STACK_SIZE: int = 4  # ScrimBrain standard - 4 frames stacked as channels
    
    # YOLO Detection Configuration - Optimized for high FPS
    YOLO_ENABLED: bool = True
    YOLO_MODEL_PATH: Optional[str] = "yolo_training/half_sword_detector2/weights/best.pt"  # Trained Half Sword v5 model
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5  # Higher confidence threshold to reduce false positives
    YOLO_DETECTION_INTERVAL: float = 0.1  # Run detection every 0.1s (10 FPS) - optimized for 60+ FPS
    YOLO_USE_CUSTOM_MODEL: bool = True  # Using trained Half Sword v5 model (15 images, proper train/val/test split)
    YOLO_OVERLAY_ENABLED: bool = False  # Disabled - YOLO display is now integrated into GUI dashboard (single window)
    
    # YOLO Self-Learning Configuration
    YOLO_SELF_LEARNING_ENABLED: bool = True  # Enable self-labeling and reward learning
    YOLO_MIN_REWARD_FOR_LABELING: float = 0.5  # Minimum reward to create self-label
    YOLO_SELF_TRAINING_INTERVAL: int = 1000  # Train on self-labels every N detections
    YOLO_CONFIDENCE_ADJUSTMENT_ENABLED: bool = True  # Adjust confidence based on rewards
    
    # Memory Reading Configuration
    GAME_PROCESS_NAME: str = "HalfSwordUE5-Win64-Shipping.exe"  # Actual process name when game is running
    GAME_EXECUTABLE_PATH: str = r"D:\Steam\steamapps\common\Half Sword Demo\HalfSwordUE5.exe"
    AUTO_LAUNCH_GAME: bool = True  # Automatically launch game if not detected
    GAME_LAUNCH_TIMEOUT: float = 30.0  # Seconds to wait for game to start
    MEMORY_SCAN_INTERVAL: float = 0.1  # seconds
    POINTER_UPDATE_INTERVAL: float = 5.0  # seconds
    
    # Memory Pattern Scanning Configuration
    # These patterns need to be found using Cheat Engine or UE4SS
    # Format: Array of Bytes (AOB) pattern with wildcards (?? = any byte)
    # Example: "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ?? 48 8B 48 08"
    MEMORY_PATTERNS: Dict[str, any] = field(default_factory=lambda: {
        # Player health pattern (needs to be found via Cheat Engine)
        "player_health": None,  # Set to AOB pattern once found
        # Player stamina pattern
        "player_stamina": None,
        # Player position pattern (X, Y, Z floats)
        "player_position": None,
        # Enemy health pattern
        "enemy_health": None,
        # Pointer chain offsets (offsets from base address)
        "pointer_chain": None,  # List of offsets: [0x1234, 0x5678, 0x9ABC]
    })
    
    # Memory scanning settings
    MEMORY_SCAN_REGION_START: int = 0x400000  # Start of scan region
    MEMORY_SCAN_REGION_SIZE: int = 0x10000000  # Size of scan region (256MB)
    MEMORY_PATTERN_MATCH_THRESHOLD: int = 1  # Number of matches required to validate pattern
    
    # Input Configuration
    MOUSE_SENSITIVITY: float = 100.0  # pixels per normalized action
    NOISE_THRESHOLD: float = 0.5  # pixels - movement below this is considered noise
    HUMAN_TIMEOUT: float = 0.5  # seconds of no movement before returning to AUTO mode
    KILL_BUTTON: str = "f8"  # Hotkey for emergency kill switch
    
    # Physics-Based Mouse Control Configuration
    USE_PHYSICS_CONTROLLER: bool = True  # Enable PID controller and Bezier smoothing
    PID_KP: float = 0.5  # Proportional gain
    PID_KI: float = 0.0  # Integral gain (usually 0 for mouse control)
    PID_KD: float = 0.2  # Derivative gain
    PID_MAX_OUTPUT: float = 1.0  # Maximum output magnitude
    USE_BEZIER_SMOOTHING: bool = True  # Enable Bezier curve smoothing for momentum
    
    # UE4SS Integration Configuration
    UE4SS_ENABLED: bool = False  # Enable UE4SS internal automation
    UE4SS_GAME_PATH: Optional[str] = None  # Auto-detected if None
    UE4SS_MODS_DIRECTORY: Optional[str] = None  # Auto-detected if None
    UE4SS_AUTO_INSTALL: bool = False  # Automatically install UE4SS if not found
    UE4SS_STATE_BRIDGE_ENABLED: bool = False  # Enable Lua → Python state bridge
    
    # ScrimBrain Integration Configuration - Per Guide Specifications
    USE_DIRECTINPUT: bool = True  # Use ctypes SendInput instead of PyAutoGUI (REQUIRED)
    USE_DISCRETE_ACTIONS: bool = True  # Use discrete macro-actions (DQN style) - ScrimBrain default
    GESTURE_MICRO_STEP_DURATION: float = 0.01  # 10ms per micro-step (guide recommendation)
    OCR_ENABLED: bool = True  # Enable OCR for score tracking (Abyss mode)
    OCR_INTERVAL: int = 30  # Run OCR every 30-60 frames (guide: 0.5-1s intervals) - optimized
    TERMINAL_STATE_DETECTION: bool = True  # Enable death screen detection
    TERMINAL_DETECTION_INTERVAL: int = 10  # Check terminal state every 10 frames (optimized)
    
    # DQN Configuration (ScrimBrain-style)
    DQN_EPSILON_START: float = 1.0  # Initial exploration rate
    DQN_EPSILON_END: float = 0.01  # Final exploration rate
    DQN_EPSILON_DECAY: int = 1000000  # Decay over 1M frames (guide recommendation)
    DQN_TARGET_UPDATE_FREQ: int = 1000  # Update target network every N steps
    
    # Learning Configuration
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    GAMMA: float = 0.99  # Discount factor
    PPO_CLIP: float = 0.2
    BETA_BC: float = 2.0  # Behavioral cloning weight (increased for better human action learning)
    REPLAY_BUFFER_SIZE: int = 10000
    PRIORITY_ALPHA: float = 0.6  # Prioritized replay exponent
    
    # Comprehensive Reward Shaping Configuration
    ENABLE_COMPREHENSIVE_REWARDS: bool = True  # Enable multi-layered reward architecture
    USE_ENHANCED_REWARDS: bool = True  # Use enhanced frame-by-frame rewards
    CURRICULUM_PHASE: str = "master"  # "toddler", "swordsman", "duelist", "master"
    ENABLE_PBRS: bool = True  # Enable Potential-Based Reward Shaping
    REWARD_ALIGNMENT_POWER: float = 3.0  # Power for edge alignment filter (harsh filtering)
    BALANCE_REWARD_K: float = 0.1  # Exponential decay constant for balance reward
    
    # Enhanced Reward Configuration (frame-by-frame)
    REWARD_SURVIVAL: float = 0.01  # Reward per frame for staying alive
    REWARD_ENGAGEMENT: float = 0.02  # Reward for being near enemies
    REWARD_MOVEMENT_QUALITY: float = 0.05  # Reward for smooth movements
    REWARD_ACTION_SMOOTHNESS: float = 0.03  # Reward for smooth actions
    REWARD_MOMENTUM: float = 0.02  # Reward for building momentum
    REWARD_PROXIMITY: float = 0.01  # Reward for optimal spacing
    REWARD_ACTIVITY: float = 0.01  # Reward for active gameplay
    
    # Reward Normalization
    REWARD_NORMALIZATION: bool = True  # Enable adaptive reward normalization
    REWARD_CLIP_MIN: float = -10.0  # Minimum reward value
    REWARD_CLIP_MAX: float = 10.0  # Maximum reward value
    
    # Online Learning Configuration
    MIN_BATCH_SIZE_FOR_TRAINING: int = 1  # Start training with ANY data (very aggressive)
    TRAINING_FREQUENCY: float = 0.05  # Train every 0.05 seconds (20 Hz) - very frequent
    HUMAN_ACTION_PRIORITY_MULTIPLIER: float = 5.0  # Human actions sampled 5x more often
    
    # Training Configuration
    UPDATE_FREQUENCY: int = 4  # Update every N batches
    MAX_EPISODES: int = 10000
    SAVE_INTERVAL: int = 100  # Save model every N episodes
    
    # Human-in-the-Loop Configuration
    ALWAYS_RECORD_HUMAN: bool = True  # Always record when human is playing
    RECORDING_MODE: str = "continuous"  # "continuous" or "intervention_only"
    MIN_HUMAN_ACTIONS_FOR_TRAINING: int = 100  # Minimum actions before training starts
    
    # Enhanced Dataset Collection Configuration
    ENABLE_DATASET_COLLECTION: bool = False  # Enable comprehensive dataset collection
    DATASET_COLLECTION_MODE: str = "continuous"  # "continuous" or "human_only" or "episode_only"
    DATASET_SAVE_INTERVAL: int = 10000  # Save dataset every N frames (auto-save)
    DATASET_NAME_PREFIX: str = "half_sword_training"  # Prefix for dataset names
    
    # Watchdog Configuration
    DEATH_WAIT_TIME: float = 2.0  # seconds to wait after death
    BLACK_SCREEN_THRESHOLD: float = 0.95  # 95% black pixels = bug
    FALLING_Z_THRESHOLD: float = -500.0
    MEMORY_LEAK_THRESHOLD: int = 10 * 1024 * 1024 * 1024  # 10GB
    
    # Paths
    MODEL_SAVE_PATH: str = "models"
    DATA_SAVE_PATH: str = "data"
    LOG_PATH: str = "logs"
    PERFORMANCE_REPORT_INTERVAL: int = 300  # Generate report every N seconds (5 minutes)
    DETAILED_LOGGING: bool = True  # Enable detailed logging for debugging
    MIN_FPS_TARGET: int = 60  # Minimum FPS target
    MAX_FPS_TARGET: int = 120  # Maximum FPS target (will cap if needed)
    FRAME_SKIP: int = 4  # Frame skip for physics stability (guide: 4 frames per action decision)
    
    # Device Configuration
    DEVICE: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    NUM_WORKERS: int = 2
    
    def __post_init__(self):
        """Initialize derived configurations"""
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.DATA_SAVE_PATH, exist_ok=True)
        os.makedirs(self.LOG_PATH, exist_ok=True)

# Global configuration instance
config = Config()

__all__ = ['Config', 'config']
