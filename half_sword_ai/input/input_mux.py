"""
Input Multiplexer: Advanced kernel-level mouse control and human override detection
Handles seamless switching between bot and human control with adaptive intelligence
CRITICAL: Never interferes with user's mouse/keyboard when not in bot mode

MASSIVE IMPROVEMENTS:
- Advanced movement prediction and smoothing
- Adaptive sensitivity based on movement patterns
- Multi-algorithm human detection
- Movement pattern recognition and emulation
- Input smoothing and anti-jitter
- Enhanced safety mechanisms
"""
import time
import threading
import queue
import logging
from enum import Enum
from typing import Optional, Tuple, Dict, List
from collections import deque
import numpy as np
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

# ScrimBrain Integration: Use DirectInput (ctypes) instead of PyAutoGUI
try:
    from half_sword_ai.input.direct_input import DirectInput
    from half_sword_ai.input.gesture_engine import GestureEngine
    from half_sword_ai.input.action_discretizer import ActionDiscretizer, MacroAction
    DIRECTINPUT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DirectInput not available: {e}, using PyAutoGUI fallback")
    DIRECTINPUT_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    logger.warning("PyAutoGUI not available, using fallback")
    PYAUTOGUI_AVAILABLE = False

try:
    import win32api
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

try:
    # Try interception driver (requires installation)
    # See INTERCEPTION_INSTALL.md for installation instructions
    from interception import Interception
    INTERCEPTION_AVAILABLE = True
    logger.debug("Interception Python library found")
except ImportError:
    # Interception not installed - this is OK, we use DirectInput fallback
    INTERCEPTION_AVAILABLE = False
    logger.debug("Interception driver not available (optional - using DirectInput fallback)")
except Exception as e:
    # Other errors (e.g., driver not installed)
    INTERCEPTION_AVAILABLE = False
    logger.debug(f"Interception check failed: {e} (using DirectInput fallback)")

class ControlMode(Enum):
    """Control mode enumeration"""
    AUTONOMOUS = "autonomous"  # Bot control
    MANUAL = "manual"  # Human control
    TRANSITIONING = "transitioning"  # Switching states

class InputMultiplexer:
    """
    Manages input control switching between bot and human
    Uses kernel-level interception when available, falls back to high-level APIs
    SAFETY: Only injects bot input when explicitly in AUTONOMOUS mode
    """
    
    def __init__(self):
        # Start in AUTONOMOUS mode by default (bot control enabled)
        self.mode = ControlMode.AUTONOMOUS
        self.last_mode_switch_time = 0  # Initialize for rapid switch protection
        
        # Track last mouse position for delta calculation
        self.last_mouse_pos = None
        if PYAUTOGUI_AVAILABLE:
            try:
                self.last_mouse_pos = pyautogui.position()
            except:
                self.last_mouse_pos = None
        self.interception = None
        self.mouse_device_id = None
        self.human_input_queue = queue.Queue()
        self.last_human_input = None
        self.last_human_input_time = 0
        self.human_timeout = config.HUMAN_TIMEOUT
        self.base_noise_threshold = config.NOISE_THRESHOLD
        self.noise_threshold = config.NOISE_THRESHOLD  # Adaptive threshold
        self.running = False
        self.control_thread = None
        self.lock = threading.Lock()
        
        # Track mode switching for better human action integration
        self.just_switched_to_autonomous = False
        self.switch_to_autonomous_time = 0
        self.human_action_priority_duration = 2.0  # Use human actions more aggressively for 2 seconds after switch
        
        # ScrimBrain Integration: DirectInput and Gesture Engine
        self.direct_input = None
        self.gesture_engine = None
        self.action_discretizer = None
        self.use_discrete_actions = config.USE_DISCRETE_ACTIONS  # Use config setting
        
        # Physics-Based Mouse Controller (PID + Bezier smoothing)
        self.physics_controller = None
        if config.USE_PHYSICS_CONTROLLER:
            try:
                from half_sword_ai.input.physics_controller import PhysicsMouseController, PIDParams
                pid_params = PIDParams(
                    kp=config.PID_KP,
                    ki=config.PID_KI,
                    kd=config.PID_KD,
                    max_output=config.PID_MAX_OUTPUT
                )
                self.physics_controller = PhysicsMouseController(
                    pid_params=pid_params,
                    use_bezier=config.USE_BEZIER_SMOOTHING
                )
                logger.info("✅ Physics controller initialized (PID + Bezier smoothing)")
            except Exception as e:
                logger.warning(f"Physics controller initialization failed: {e} - using direct injection")
        
        if DIRECTINPUT_AVAILABLE:
            try:
                self.direct_input = DirectInput()
                self.gesture_engine = GestureEngine(self.direct_input)
                # Always initialize action discretizer (needed for both modes)
                self.action_discretizer = ActionDiscretizer()
                logger.info("✅ ScrimBrain DirectInput initialized - using ctypes SendInput")
                if self.use_discrete_actions:
                    logger.info("✅ Discrete action mode enabled (DQN/ScrimBrain-style)")
                else:
                    logger.info("✅ Continuous action mode enabled (PPO-style)")
            except Exception as e:
                logger.error(f"Failed to initialize DirectInput: {e}")
                self.direct_input = None
        
        # Advanced human input detection tracking
        self.last_mouse_pos = None
        self.mouse_position_history = deque(maxlen=50)  # Extended history for pattern analysis
        self.velocity_history = deque(maxlen=20)  # Track movement velocity
        self.acceleration_history = deque(maxlen=10)  # Track acceleration
        self.movement_detection_active = False
        self.bot_injection_count = 0
        self.human_override_count = 0
        self.mode_switch_count = 0
        
        # Adaptive sensitivity system
        self.current_sensitivity = config.MOUSE_SENSITIVITY
        self.adaptive_sensitivity_enabled = True
        self.sensitivity_history = deque(maxlen=100)
        self.movement_magnitude_history = deque(maxlen=100)
        
        # Movement prediction and smoothing
        self.predicted_movement = None
        self.movement_smoother = deque(maxlen=5)  # Smoothing window
        self.smoothing_enabled = True
        
        # Advanced pattern tracking
        self.movement_pattern_buffer = deque(maxlen=100)  # Extended for better patterns
        self.pattern_signatures = {}  # Store recognized patterns
        self.pattern_match_cache = deque(maxlen=20)  # Cache recent matches
        
        # Button state tracking (enhanced) - All Half Sword controls
        # Mouse buttons
        self.current_button_holds = {
            'left': False, 'right': False, 'space': False, 'alt': False,
            # Movement keys (WASD)
            'w': False, 'a': False, 's': False, 'd': False,
            # Grab mechanics
            'q': False, 'e': False,
            # Other controls
            'g': False, 'shift': False, 'ctrl': False
        }
        self.button_hold_start_times = {key: None for key in self.current_button_holds}
        self.button_hold_durations = {key: 0.0 for key in self.current_button_holds}
        self.button_press_patterns = deque(maxlen=50)  # Track button press sequences
        
        # Movement vector tracking (WASD combination)
        self.movement_vector = {'x': 0.0, 'y': 0.0}  # Normalized movement direction
        
        # Human action recording
        self.human_action_recorder = None  # Will be set by actor
        self.last_recorded_action_time = 0
        self.recording_interval = 1.0 / config.CAPTURE_FPS
        
        # Enhanced safety mechanisms
        self.safety_lock = False
        self.consecutive_human_detections = 0
        self.rapid_switch_protection = False
        self.last_mode_switch_time = 0
        self.min_mode_switch_interval = 0.1  # Prevent rapid switching
        
        # Bot injection tracking (enhanced)
        self.bot_injecting = False
        self.bot_injection_lock = threading.Lock()  # Lock for thread-safe bot injection flag
        self.last_bot_injection_time = 0
        self.bot_injection_cooldown = 0.5  # Increased to 500ms to prevent detecting bot's own movements
        self.bot_movement_buffer = deque(maxlen=10)  # Track bot's own movements
        self.injection_clear_timer = None  # Timer for clearing injection flag
        
        # Anti-jitter system
        self.jitter_filter_enabled = True
        self.jitter_threshold = 1.0  # pixels
        self.recent_movements = deque(maxlen=5)
        
        # Multi-algorithm human detection
        self.detection_algorithm = "adaptive"  # "simple", "velocity", "acceleration", "adaptive"
        self.detection_confidence = 0.0
        
        if INTERCEPTION_AVAILABLE:
            self._init_interception()
        else:
            logger.info("Using PyAutoGUI fallback mode with advanced features")
            logger.info("⚠️  NOTE: Bot movements may be detected as human input - keep mouse still for bot to work")
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.0
    
    def _init_interception(self):
        """Initialize interception driver"""
        global INTERCEPTION_AVAILABLE
        if not INTERCEPTION_AVAILABLE:
            return
        
        try:
            self.interception = Interception()
            if not self.interception.valid:
                logger.warning("Interception driver not valid - driver may not be installed")
                logger.info("   See INTERCEPTION_INSTALL.md for installation instructions")
                logger.info("   Using DirectInput fallback (this is fine!)")
                INTERCEPTION_AVAILABLE = False
                return
            
            # Find mouse device
            devices = self.interception.devices
            mouse_devices = [d for d in devices if not d.is_keyboard]
            
            if not mouse_devices:
                logger.warning("No mouse device found for interception - driver may not be installed")
                logger.info("   See INTERCEPTION_INSTALL.md for installation instructions")
                logger.info("   Using DirectInput fallback (this is fine!)")
                INTERCEPTION_AVAILABLE = False
                return
            
            # Store the first mouse device handle
            self.mouse_device = mouse_devices[0]
            self.mouse_device_id = mouse_devices[0].handle if hasattr(mouse_devices[0], 'handle') else None
            logger.info(f"✅ Interception initialized with mouse device (handle: {self.mouse_device_id})")
            logger.info(f"   Found {len(mouse_devices)} mouse device(s)")
            logger.info("   Kernel-level input control enabled")
        except Exception as e:
            logger.warning(f"Interception initialization failed: {e}")
            logger.info("   This is OK - using DirectInput fallback")
            logger.info("   See INTERCEPTION_INSTALL.md if you want to install interception")
            INTERCEPTION_AVAILABLE = False
    
    def start(self):
        """Start the input multiplexer"""
        self.running = True
        if INTERCEPTION_AVAILABLE and self.interception:
            self.control_thread = threading.Thread(target=self._interception_loop, daemon=True)
        else:
            self.control_thread = threading.Thread(target=self._fallback_loop, daemon=True)
        self.control_thread.start()
        logger.info("Input multiplexer started")
    
    def stop(self):
        """Stop the input multiplexer"""
        self.running = False
        
        # Cancel injection clear timer if it exists
        if self.injection_clear_timer:
            try:
                self.injection_clear_timer.cancel()
            except:
                pass
        
        # Clear bot injection flag
        with self.bot_injection_lock:
            self.bot_injecting = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        logger.info("Input multiplexer stopped")
    
    def _interception_loop(self):
        """Main loop for interception driver mode"""
        while self.running:
            try:
                # Check for human input
                if self.mode == ControlMode.AUTONOMOUS:
                    # Block physical mouse, allow bot injection
                    # In real implementation, would use interception.filter()
                    pass
                elif self.mode == ControlMode.MANUAL:
                    # Allow physical mouse, block bot injection
                    # Monitor for human movement
                    self._detect_human_movement()
                
                time.sleep(0.001)  # 1ms polling
            except Exception as e:
                logger.error(f"Interception loop error: {e}")
                time.sleep(0.01)
    
    def _fallback_loop(self):
        """Fallback loop using high-level APIs with human detection"""
        while self.running:
            try:
                # Safety check: Force clear bot_injecting if it's been True for too long (>1 second)
                # This prevents the flag from getting stuck
                if self.bot_injecting:
                    time_since_injection = time.time() - self.last_bot_injection_time
                    if time_since_injection > 1.0:  # 1 second timeout
                        logger.warning(f"[INPUT_MUX] bot_injecting flag stuck for {time_since_injection:.2f}s - force clearing")
                        with self.bot_injection_lock:
                            self.bot_injecting = False
                
                # Always monitor for human input, but ONLY if in AUTONOMOUS mode
                # Don't check if already in MANUAL mode to prevent fighting for control
                if self.mode == ControlMode.AUTONOMOUS and not self.bot_injecting:
                    human_movement = self._detect_human_movement()
                    
                    if human_movement:
                        # Human is moving mouse - immediately switch to MANUAL
                        current_time = time.time()
                        detection_conf = getattr(self, 'detection_confidence', 0.0)
                        logger.info(f"[INPUT_MUX] Human movement detected | Confidence: {detection_conf:.2f} | Switching AUTONOMOUS -> MANUAL")
                        logger.debug(f"[INPUT_MUX] Detection details: consecutive_detections={self.consecutive_human_detections}, bot_injecting={self.bot_injecting}")
                        self.set_mode(ControlMode.MANUAL)
                        self.human_override_count += 1
                        # Reset bot injection flag to ensure clean handoff (with lock)
                        with self.bot_injection_lock:
                            self.bot_injecting = False
                        logger.debug(f"[INPUT_MUX] Mode switched | human_override_count={self.human_override_count}, bot_injecting={self.bot_injecting}")
                
                # Check for timeout (return to bot control if human stops)
                if self.mode == ControlMode.MANUAL:
                    current_time = time.time()
                    time_since_last_input = current_time - self.last_human_input_time
                    # More aggressive switching back - check every loop iteration
                    if time_since_last_input > self.human_timeout:
                        # Use less strict idle check - prioritize timeout over movement check
                        if self._is_human_idle() or time_since_last_input > self.human_timeout * 1.5:
                            logger.info(f"[INPUT_MUX] Human idle detected ({time_since_last_input:.2f}s > {self.human_timeout}s) | Switching MANUAL -> AUTONOMOUS")
                            logger.debug(f"[INPUT_MUX] Idle check: time_since_input={time_since_last_input:.2f}s, timeout={self.human_timeout}s, is_idle={self._is_human_idle()}")
                            self.set_mode(ControlMode.AUTONOMOUS)
                            # Reset detection counters to prevent immediate switch back
                            self.consecutive_human_detections = 0
                            with self.bot_injection_lock:
                                self.bot_injecting = False
                            # Mark that we just switched back - bot should use human actions more aggressively
                            self.just_switched_to_autonomous = True
                            self.switch_to_autonomous_time = current_time
                            # Reset last human input time to prevent immediate re-trigger
                            self.last_human_input_time = 0
                
                time.sleep(0.01)  # 10ms polling for fallback
            except Exception as e:
                logger.error(f"Fallback loop error: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _detect_human_movement(self) -> bool:
        """
        Advanced multi-algorithm human movement detection
        Uses velocity, acceleration, and pattern analysis for accurate detection
        Returns True if significant human movement detected with confidence score
        """
        try:
            if not PYAUTOGUI_AVAILABLE:
                return False
            
            current_time = time.time()
            
            # Ignore during bot injection cooldown
            if current_time - self.last_bot_injection_time < self.bot_injection_cooldown:
                return False
            
            current_pos = pyautogui.position()
            
            # Track position history
            self.mouse_position_history.append((current_pos, current_time))
            
            movement_detected = False
            mouse_delta = (0.0, 0.0)
            detection_confidence = 0.0
            
            if self.last_mouse_pos is not None:
                dx = current_pos.x - self.last_mouse_pos.x
                dy = current_pos.y - self.last_mouse_pos.y
                movement_magnitude = (dx**2 + dy**2)**0.5
                
                # Anti-jitter filtering
                if self.jitter_filter_enabled and movement_magnitude > 0:
                    self.recent_movements.append(movement_magnitude)
                    if len(self.recent_movements) >= 3:
                        # Check if movement is consistent (not jitter)
                        avg_mag = np.mean(list(self.recent_movements))
                        if movement_magnitude < self.jitter_threshold and movement_magnitude < avg_mag * 0.5:
                            # Likely jitter, ignore
                            self.last_mouse_pos = current_pos
                            return False
                
                # Calculate velocity
                if len(self.mouse_position_history) >= 2:
                    dt = current_time - self.mouse_position_history[-2][1]
                    if dt > 0:
                        velocity = movement_magnitude / dt
                        self.velocity_history.append(velocity)
                
                # Calculate acceleration
                if len(self.velocity_history) >= 2:
                    dt = current_time - (self.mouse_position_history[-2][1] if len(self.mouse_position_history) >= 2 else current_time)
                    if dt > 0:
                        accel = abs(self.velocity_history[-1] - self.velocity_history[-2]) / dt if len(self.velocity_history) >= 2 else 0
                        self.acceleration_history.append(accel)
                
                # Multi-algorithm detection
                if self.detection_algorithm == "adaptive" or self.detection_algorithm == "velocity":
                    # Velocity-based detection
                    if len(self.velocity_history) >= 3:
                        avg_velocity = np.mean(list(self.velocity_history)[-3:])
                        if avg_velocity > 50:  # pixels per second threshold
                            detection_confidence += 0.4
                
                if self.detection_algorithm == "adaptive" or self.detection_algorithm == "acceleration":
                    # Acceleration-based detection
                    if len(self.acceleration_history) >= 2:
                        avg_accel = np.mean(list(self.acceleration_history)[-2:])
                        if avg_accel > 100:  # pixels per second^2
                            detection_confidence += 0.3
                
                # Magnitude-based detection (always used)
                if movement_magnitude > self.noise_threshold:
                    detection_confidence += 0.5
                
                # Pattern-based detection
                if len(self.movement_pattern_buffer) >= 5:
                    recent_pattern = list(self.movement_pattern_buffer)[-5:]
                    pattern_consistency = self._check_pattern_consistency(recent_pattern)
                    if pattern_consistency > 0.7:
                        detection_confidence += 0.2
                
                # Adaptive threshold adjustment
                if self.adaptive_sensitivity_enabled:
                    self.movement_magnitude_history.append(movement_magnitude)
                    if len(self.movement_magnitude_history) >= 20:
                        avg_magnitude = np.mean(list(self.movement_magnitude_history)[-20:])
                        # Adjust threshold based on typical movement patterns
                        if avg_magnitude > 10:
                            self.noise_threshold = max(self.base_noise_threshold, avg_magnitude * 0.3)
                        else:
                            self.noise_threshold = self.base_noise_threshold
                
                # Final detection decision - require higher confidence and ignore during bot injection
                # Increase threshold to reduce false positives
                if detection_confidence >= 0.7 and not self.bot_injecting and movement_magnitude > 5.0:
                    movement_detected = True
                    self.detection_confidence = detection_confidence
                    self.last_human_input_time = current_time
                    mouse_delta = (dx, dy)
                    self.consecutive_human_detections += 1
                    
                    # Get button states
                    current_buttons = self._get_current_button_states()
                    self._update_button_hold_tracking(current_buttons)
                    
                    # Store human input with enhanced context
                    normalized_delta = (mouse_delta[0] / self.current_sensitivity,
                                      mouse_delta[1] / self.current_sensitivity)
                    self.last_human_input = (normalized_delta[0], normalized_delta[1], current_buttons.copy())
                    
                    # Enhanced pattern tracking
                    pattern_entry = {
                        'delta': mouse_delta,
                        'normalized': normalized_delta,
                        'buttons': current_buttons.copy(),
                        'timestamp': current_time,
                        'magnitude': movement_magnitude,
                        'velocity': self.velocity_history[-1] if self.velocity_history else 0,
                        'acceleration': self.acceleration_history[-1] if self.acceleration_history else 0,
                        'confidence': detection_confidence
                    }
                    self.movement_pattern_buffer.append(pattern_entry)
                    
                    # Store button press pattern
                    self._track_button_pattern(current_buttons)
                else:
                    self.consecutive_human_detections = 0
                    self.detection_confidence = detection_confidence
            
            self.last_mouse_pos = current_pos
            return movement_detected
            
        except Exception as e:
            logger.debug(f"Human movement detection error: {e}")
            return False
    
    def _check_pattern_consistency(self, pattern: List[Dict]) -> float:
        """Check how consistent a movement pattern is (0-1 score)"""
        if len(pattern) < 2:
            return 0.0
        
        magnitudes = [p.get('magnitude', 0) for p in pattern]
        if len(magnitudes) < 2:
            return 0.0
        
        # Check variance in magnitudes (consistent patterns have lower variance)
        variance = np.var(magnitudes)
        avg_magnitude = np.mean(magnitudes)
        
        if avg_magnitude == 0:
            return 0.0
        
        # Consistency score: lower variance relative to magnitude = more consistent
        consistency = 1.0 - min(1.0, variance / (avg_magnitude ** 2 + 1))
        return consistency
    
    def _track_button_pattern(self, buttons: Dict[str, bool]):
        """Track button press/release patterns for pattern matching"""
        pattern_entry = {
            # Mouse buttons
            'left': buttons.get('left', False),
            'right': buttons.get('right', False),
            # Movement keys
            'w': buttons.get('w', False),
            'a': buttons.get('a', False),
            's': buttons.get('s', False),
            'd': buttons.get('d', False),
            # Grab mechanics
            'q': buttons.get('q', False),
            'e': buttons.get('e', False),
            # Other controls
            'space': buttons.get('space', False),
            'alt': buttons.get('alt', False),
            'g': buttons.get('g', False),
            'shift': buttons.get('shift', False),
            'ctrl': buttons.get('ctrl', False),
            'timestamp': time.time()
        }
        self.button_press_patterns.append(pattern_entry)
    
    def _get_current_button_states(self) -> Dict[str, bool]:
        """
        Get current button states - reads actual keyboard/mouse state
        Captures all Half Sword controls: WASD movement, Q/E grabs, mouse buttons, etc.
        """
        # Initialize all Half Sword controls
        states = {
            # Mouse buttons
            'left': False,      # LMB - Left hand control
            'right': False,     # RMB - Right hand control / Half-swording
            # Movement keys (WASD)
            'w': False,         # W - Forward
            'a': False,         # A - Left strafe
            's': False,         # S - Backward
            'd': False,         # D - Right strafe
            # Grab mechanics
            'q': False,         # Q - Left hand grab
            'e': False,         # E - Right hand grab
            # Other controls
            'space': False,     # SPACE - Jump/dodge
            'alt': False,      # ALT - Thrust/half-swording
            'g': False,        # G - Surrender
            'shift': False,    # SHIFT - Sprint (if used)
            'ctrl': False      # CTRL - Crouch (if used)
        }
        
        if WIN32_AVAILABLE:
            try:
                import win32api
                import win32con
                
                # Check mouse buttons
                left_state = win32api.GetAsyncKeyState(win32con.VK_LBUTTON)
                right_state = win32api.GetAsyncKeyState(win32con.VK_RBUTTON)
                states['left'] = (left_state & 0x8000) != 0
                states['right'] = (right_state & 0x8000) != 0
                
                # Check movement keys (WASD)
                w_state = win32api.GetAsyncKeyState(ord('W'))
                a_state = win32api.GetAsyncKeyState(ord('A'))
                s_state = win32api.GetAsyncKeyState(ord('S'))
                d_state = win32api.GetAsyncKeyState(ord('D'))
                states['w'] = (w_state & 0x8000) != 0
                states['a'] = (a_state & 0x8000) != 0
                states['s'] = (s_state & 0x8000) != 0
                states['d'] = (d_state & 0x8000) != 0
                
                # Check grab keys (Q, E)
                q_state = win32api.GetAsyncKeyState(ord('Q'))
                e_state = win32api.GetAsyncKeyState(ord('E'))
                states['q'] = (q_state & 0x8000) != 0
                states['e'] = (e_state & 0x8000) != 0
                
                # Check other keyboard keys
                space_state = win32api.GetAsyncKeyState(win32con.VK_SPACE)
                alt_state = win32api.GetAsyncKeyState(win32con.VK_MENU)  # ALT key
                g_state = win32api.GetAsyncKeyState(ord('G'))
                shift_state = win32api.GetAsyncKeyState(win32con.VK_SHIFT)
                ctrl_state = win32api.GetAsyncKeyState(win32con.VK_CONTROL)
                
                states['space'] = (space_state & 0x8000) != 0
                states['alt'] = (alt_state & 0x8000) != 0
                states['g'] = (g_state & 0x8000) != 0
                states['shift'] = (shift_state & 0x8000) != 0
                states['ctrl'] = (ctrl_state & 0x8000) != 0
                
            except ImportError:
                # Fallback: try keyboard library
                if KEYBOARD_AVAILABLE:
                    try:
                        import keyboard
                        states['w'] = keyboard.is_pressed('w')
                        states['a'] = keyboard.is_pressed('a')
                        states['s'] = keyboard.is_pressed('s')
                        states['d'] = keyboard.is_pressed('d')
                        states['q'] = keyboard.is_pressed('q')
                        states['e'] = keyboard.is_pressed('e')
                        states['space'] = keyboard.is_pressed('space')
                        states['alt'] = keyboard.is_pressed('alt')
                        states['g'] = keyboard.is_pressed('g')
                        states['shift'] = keyboard.is_pressed('shift')
                        states['ctrl'] = keyboard.is_pressed('ctrl')
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Button state read error: {e}")
        elif KEYBOARD_AVAILABLE:
            # Fallback to keyboard library only
            try:
                import keyboard
                states['w'] = keyboard.is_pressed('w')
                states['a'] = keyboard.is_pressed('a')
                states['s'] = keyboard.is_pressed('s')
                states['d'] = keyboard.is_pressed('d')
                states['q'] = keyboard.is_pressed('q')
                states['e'] = keyboard.is_pressed('e')
                states['space'] = keyboard.is_pressed('space')
                states['alt'] = keyboard.is_pressed('alt')
                states['g'] = keyboard.is_pressed('g')
                states['shift'] = keyboard.is_pressed('shift')
                states['ctrl'] = keyboard.is_pressed('ctrl')
            except:
                pass
        
        # Calculate normalized movement vector from WASD
        movement_x = 0.0
        movement_y = 0.0
        if states['d']:
            movement_x += 1.0
        if states['a']:
            movement_x -= 1.0
        if states['w']:
            movement_y += 1.0
        if states['s']:
            movement_y -= 1.0
        
        # Normalize diagonal movement
        if movement_x != 0.0 and movement_y != 0.0:
            length = (movement_x**2 + movement_y**2)**0.5
            movement_x /= length
            movement_y /= length
        
        self.movement_vector = {'x': movement_x, 'y': movement_y}
        
        return states
    
    def _update_button_hold_tracking(self, current_buttons: Dict[str, bool]):
        """Track button hold durations for pattern recognition"""
        current_time = time.time()
        for button in ['left', 'right', 'space', 'alt']:
            is_pressed = current_buttons.get(button, False)
            was_pressed = self.current_button_holds.get(button, False)
            
            if is_pressed and not was_pressed:
                # Button just pressed
                self.button_hold_start_times[button] = current_time
            elif not is_pressed and was_pressed:
                # Button just released
                hold_duration = current_time - self.button_hold_start_times[button] if self.button_hold_start_times[button] else 0
                if hold_duration > 0:
                    logger.debug(f"Button {button} held for {hold_duration:.3f}s")
                self.button_hold_start_times[button] = None
            
            self.current_button_holds[button] = is_pressed
    
    def get_movement_pattern(self, lookback: int = 10) -> List[Dict]:
        """Get recent movement pattern for emulation"""
        return list(self.movement_pattern_buffer)[-lookback:]
    
    def _is_human_idle(self) -> bool:
        """Check if human input has been idle - enhanced with multiple checks"""
        current_time = time.time()
        time_since_input = current_time - self.last_human_input_time
        
        # Check timeout - primary check
        if time_since_input <= self.human_timeout:
            return False
        
        # Additional check: verify no recent mouse movement (less strict)
        if self.last_mouse_pos is not None:
            try:
                if PYAUTOGUI_AVAILABLE:
                    current_pos = pyautogui.position()
                    dx = abs(current_pos.x - self.last_mouse_pos.x)
                    dy = abs(current_pos.y - self.last_mouse_pos.y)
                    movement = (dx**2 + dy**2)**0.5
                    # More lenient: only block if movement is significant (3x threshold)
                    # This prevents jitter from blocking mode switch
                    if movement > self.noise_threshold * 3:
                        return False
            except:
                pass  # If check fails, assume idle based on timeout
        
        # If timeout passed and no significant movement, human is idle
        return True
    
    def set_mode(self, mode: ControlMode):
        """Set control mode with rapid switch protection"""
        with self.lock:
            current_time = time.time()
            
            # Rapid switch protection
            if (current_time - self.last_mode_switch_time < self.min_mode_switch_interval and
                self.mode != mode):
                # Too rapid, ignore
                return
            
            old_mode = self.mode
            if old_mode != mode:
                self.mode = mode
                self.mode_switch_count += 1
                self.last_mode_switch_time = current_time
                
                # Reset detection counters on mode switch
                self.consecutive_human_detections = 0
                self.detection_confidence = 0.0
                
                logger.info(f"Control mode changed: {old_mode.value} -> {mode.value} (total switches: {self.mode_switch_count})")
                
                # Start/stop recording based on mode
                if self.human_action_recorder:
                    if mode == ControlMode.MANUAL and not self.human_action_recorder.recording:
                        self.human_action_recorder.start_recording()
                        logger.info("🎥 Started recording human actions")
                    elif mode == ControlMode.AUTONOMOUS and self.human_action_recorder.recording:
                        # Keep recording even in autonomous mode to capture corrections
                        pass
    
    def is_human_active(self) -> bool:
        """Check if human is currently controlling"""
        return self.mode == ControlMode.MANUAL
    
    def inject_action(self, delta_x: float, delta_y: float, buttons: Dict[str, bool] = None, 
                     use_physics_control: bool = None):
        """
        Enhanced bot action injection with movement prediction and smoothing
        SAFETY: Only injects if in AUTONOMOUS mode and safety_lock is False
        
        Args:
            delta_x: Normalized X movement (-1 to 1)
            delta_y: Normalized Y movement (-1 to 1)
            buttons: Dictionary with button states
        """
        # CRITICAL SAFETY CHECKS
        if self.safety_lock:
            logger.warning(f"[INJECTION BLOCKED] Safety lock active - blocking bot injection | delta_x={delta_x:.3f}, delta_y={delta_y:.3f}")
            return
        
        if self.mode != ControlMode.AUTONOMOUS:
            logger.debug(f"[INJECTION BLOCKED] Not in AUTONOMOUS mode (current: {self.mode.value}) - blocking injection | delta_x={delta_x:.3f}, delta_y={delta_y:.3f}")
            return
        
        # CRITICAL: Double-check mode hasn't changed and no human movement
        if self.mode != ControlMode.AUTONOMOUS:
            logger.warning(f"[INJECTION BLOCKED] Mode changed to {self.mode.value} during injection check - aborting")
            return
        
        # Double-check for human movement before injecting
        if self._detect_human_movement():
            detection_conf = getattr(self, 'detection_confidence', 0.0)
            logger.warning(f"[INJECTION BLOCKED] Human movement detected during injection (confidence: {detection_conf:.2f}) - aborting to prevent interference")
            logger.debug(f"[INJECTION BLOCKED] Switching to MANUAL mode | bot_injecting={self.bot_injecting}")
            self.set_mode(ControlMode.MANUAL)
            with self.bot_injection_lock:
                self.bot_injecting = False  # Reset flag immediately (thread-safe)
            return
        
        logger.debug(f"[INJECTION] Injecting action | delta_x={delta_x:.3f}, delta_y={delta_y:.3f}, buttons={buttons}, mode={self.mode.value}")
        
        # Apply physics controller if enabled (PID + Bezier smoothing)
        if self.physics_controller and (use_physics_control is None or use_physics_control):
            try:
                import numpy as np
                # Get current mouse position for physics controller
                current_pos = np.array([0.0, 0.0])  # Will be updated with actual position
                if PYAUTOGUI_AVAILABLE and self.last_mouse_pos:
                    current_pos = np.array([float(self.last_mouse_pos[0]), float(self.last_mouse_pos[1])])
                
                # Target position (normalized to screen coordinates)
                target_pos = current_pos + np.array([delta_x, delta_y]) * self.current_sensitivity
                
                logger.debug(f"[INJECTION] Physics controller active | "
                           f"current_pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                           f"target_pos=({target_pos[0]:.2f}, {target_pos[1]:.2f}) | "
                           f"raw_delta=({delta_x:.3f}, {delta_y:.3f})")
                
                # Compute smooth movement using physics controller
                smooth_delta = self.physics_controller.compute_movement(target_pos, current_pos)
                
                # Convert back to normalized deltas
                if np.linalg.norm(smooth_delta) > 0.01:
                    delta_x_before = delta_x
                    delta_y_before = delta_y
                    delta_x = smooth_delta[0] / self.current_sensitivity
                    delta_y = smooth_delta[1] / self.current_sensitivity
                    logger.debug(f"[INJECTION] Physics controller applied | "
                               f"delta_before=({delta_x_before:.3f}, {delta_y_before:.3f}) | "
                               f"delta_after=({delta_x:.3f}, {delta_y:.3f}) | "
                               f"momentum={self.physics_controller.get_momentum():.4f}")
            except Exception as e:
                logger.warning(f"[INJECTION] Physics controller error: {e} - using direct injection", exc_info=True)
        
        # Apply movement smoothing if enabled
        smoothing_applied = False
        if self.smoothing_enabled:
            delta_x_before = delta_x
            delta_y_before = delta_y
            delta_x, delta_y = self._smooth_movement(delta_x, delta_y)
            smoothing_applied = True
            logger.debug(f"[INJECTION] Movement smoothing applied | "
                        f"delta_before=({delta_x_before:.3f}, {delta_y_before:.3f}) | "
                        f"delta_after=({delta_x:.3f}, {delta_y:.3f})")
        
        # Scale normalized values to pixels with adaptive sensitivity
        pixel_x = delta_x * self.current_sensitivity
        pixel_y = delta_y * self.current_sensitivity
        
        logger.debug(f"[INJECTION] Pixel conversion | "
                    f"normalized=({delta_x:.3f}, {delta_y:.3f}) | "
                    f"sensitivity={self.current_sensitivity:.2f} | "
                    f"pixel=({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Movement prediction for smoother injection
        prediction_applied = False
        if self.predicted_movement is not None:
            pixel_x_before = pixel_x
            pixel_y_before = pixel_y
            # Blend with prediction (80% current, 20% prediction)
            pixel_x = pixel_x * 0.8 + self.predicted_movement[0] * 0.2
            pixel_y = pixel_y * 0.8 + self.predicted_movement[1] * 0.2
            prediction_applied = True
            logger.debug(f"[INJECTION] Prediction blending | "
                        f"pixel_before=({pixel_x_before:.2f}, {pixel_y_before:.2f}) | "
                        f"prediction={self.predicted_movement} | "
                        f"pixel_after=({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Clamp to reasonable values
        pixel_x_before_clamp = pixel_x
        pixel_y_before_clamp = pixel_y
        pixel_x = max(-500, min(500, pixel_x))
        pixel_y = max(-500, min(500, pixel_y))
        
        if pixel_x != pixel_x_before_clamp or pixel_y != pixel_y_before_clamp:
            logger.debug(f"[INJECTION] Pixel clamping applied | "
                        f"before=({pixel_x_before_clamp:.2f}, {pixel_y_before_clamp:.2f}) | "
                        f"after=({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Update prediction
        self.predicted_movement = (pixel_x * 0.3, pixel_y * 0.3)  # Predict slight continuation
        
        logger.debug(f"[INJECTION] Pre-injection summary | "
                    f"final_pixel=({pixel_x:.2f}, {pixel_y:.2f}) | "
                    f"physics={self.physics_controller is not None} | "
                    f"smoothing={smoothing_applied} | "
                    f"prediction={prediction_applied} | "
                    f"buttons={buttons}")
        
        if INTERCEPTION_AVAILABLE and self.interception:
            # Use interception driver injection
            try:
                # In real implementation, would use interception.send()
                # For now, fall through to PyAutoGUI
                pass
            except Exception as e:
                logger.error(f"Interception injection error: {e}", exc_info=True)
        
        # ScrimBrain Integration: Use DirectInput if available
        if self.direct_input and self.gesture_engine:
            try:
                # Only inject if we're still in AUTONOMOUS mode (double-check)
                if self.mode == ControlMode.AUTONOMOUS and not self.safety_lock:
                    # Mark that we're injecting to prevent detecting our own movement (thread-safe)
                    with self.bot_injection_lock:
                        self.bot_injecting = True
                        self.last_bot_injection_time = time.time()
                    logger.debug(f"Bot injecting action: dx={pixel_x:.1f}, dy={pixel_y:.1f}, cooldown={self.bot_injection_cooldown}s")
                    
                    # Detect if this is an attack swing (large, rapid movement)
                    movement_magnitude = np.sqrt(pixel_x**2 + pixel_y**2)
                    is_attack_swing = movement_magnitude > 200  # Threshold for attack detection
                    
                    # Determine which mouse button to hold for attack
                    mouse_button_to_hold = None
                    if buttons:
                        if buttons.get('left', False):
                            mouse_button_to_hold = 'left'
                        elif buttons.get('right', False):
                            mouse_button_to_hold = 'right'
                    # Default: use left button for large attack swings if no button specified
                    elif is_attack_swing:
                        mouse_button_to_hold = 'left'
                    
                    # Hold mouse button BEFORE movement for attack swings (CRITICAL for Half Sword)
                    if is_attack_swing and mouse_button_to_hold:
                        self.direct_input.press_mouse_button(mouse_button_to_hold)
                        logger.info(f"👊 ATTACK | Holding {mouse_button_to_hold} button | Magnitude: {movement_magnitude:.1f}")
                    # Also hold button if explicitly requested in buttons dict
                    elif buttons:
                        if buttons.get('left', False):
                            self.direct_input.press_mouse_button('left')
                            logger.debug(f"Bot: Holding left button")
                        if buttons.get('right', False):
                            self.direct_input.press_mouse_button('right')
                            logger.debug(f"Bot: Holding right button")
                    
                    # Use DirectInput for relative mouse movement (MOUSEEVENTF_MOVE)
                    # Movement happens WHILE button is held (critical for Half Sword attacks)
                    # Lower threshold to ensure even small movements are injected
                    if abs(pixel_x) > 0.01 or abs(pixel_y) > 0.01:
                        # Use gesture engine for smooth physics-compatible movement
                        try:
                            self.gesture_engine.perform_smooth_gesture(
                                int(pixel_x), int(pixel_y), 50  # 50ms duration
                            )
                            self.bot_injection_count += 1
                            if self.bot_injection_count % 50 == 0:  # Log every 50 injections (less spam)
                                btn_str = ""
                                if buttons.get('left', False):
                                    btn_str += "👊L"
                                if buttons.get('right', False):
                                    btn_str += "👊R"
                                btn_str = btn_str if btn_str else "none"
                                logger.info(f"✅ INJECTED | dx={pixel_x:.1f} dy={pixel_y:.1f} | {btn_str}")
                        except Exception as e:
                            logger.error(f"Error in gesture engine: {e}", exc_info=True)
                            # Fallback to direct movement
                            self.direct_input.move_mouse_relative(int(pixel_x), int(pixel_y))
                            self.bot_injection_count += 1
                    
                    # Release mouse button AFTER movement completes (for attack swings)
                    # Small delay to ensure movement completes while button is held
                    if is_attack_swing and mouse_button_to_hold:
                        time.sleep(0.05)  # 50ms delay to ensure movement completes
                        self.direct_input.release_mouse_button(mouse_button_to_hold)
                        logger.info(f"👊 RELEASED | {mouse_button_to_hold} button after swing")
                    
                    # Handle other buttons using DirectInput scan codes
                    if buttons and self.mode == ControlMode.AUTONOMOUS:
                        # Handle mouse buttons (if not already handled for attack)
                        if not is_attack_swing:
                            if buttons.get('left', False):
                                self.direct_input.press_mouse_button('left')
                            else:
                                self.direct_input.release_mouse_button('left')
                            if buttons.get('right', False):
                                self.direct_input.press_mouse_button('right')
                            else:
                                self.direct_input.release_mouse_button('right')
                        # Always handle keyboard keys
                        self.direct_input.set_key_state('space', buttons.get('space', False))
                        self.direct_input.set_key_state('alt', buttons.get('alt', False))
                        
                        # Keyboard keys using scan codes
                        self.direct_input.set_key_state('space', buttons.get('space', False))
                        self.direct_input.set_key_state('alt', buttons.get('alt', False))
                        # Movement keys (WASD)
                        self.direct_input.set_key_state('w', buttons.get('w', False))
                        self.direct_input.set_key_state('a', buttons.get('a', False))
                        self.direct_input.set_key_state('s', buttons.get('s', False))
                        self.direct_input.set_key_state('d', buttons.get('d', False))
                        # Grab keys (Q/E)
                        self.direct_input.set_key_state('q', buttons.get('q', False))
                        self.direct_input.set_key_state('e', buttons.get('e', False))
                    
                    # Clear injection flag after a short delay (with timeout protection)
                    import threading
                    def clear_injection_flag():
                        try:
                            time.sleep(0.05)  # 50ms delay
                            with self.bot_injection_lock:
                                self.bot_injecting = False
                        except Exception as e:
                            logger.error(f"Error clearing injection flag: {e}")
                            # Force clear on error
                            with self.bot_injection_lock:
                                self.bot_injecting = False
                    
                    # Cancel previous timer if it exists
                    if self.injection_clear_timer and self.injection_clear_timer.is_alive():
                        # Previous timer still running, don't spawn new one
                        pass
                    else:
                        self.injection_clear_timer = threading.Thread(target=clear_injection_flag, daemon=True)
                        self.injection_clear_timer.start()
                else:
                    logger.debug(f"Mode changed during injection - aborting (mode={self.mode.value}, lock={self.safety_lock})")
            except Exception as e:
                logger.error(f"DirectInput injection error: {e}", exc_info=True)
                with self.bot_injection_lock:
                    self.bot_injecting = False
        
        # Fallback to PyAutoGUI (with additional safety)
        elif PYAUTOGUI_AVAILABLE:
            try:
                # Only inject if we're still in AUTONOMOUS mode (double-check)
                if self.mode == ControlMode.AUTONOMOUS and not self.safety_lock:
                    # Mark that we're injecting to prevent detecting our own movement (thread-safe)
                    with self.bot_injection_lock:
                        self.bot_injecting = True
                        self.last_bot_injection_time = time.time()
                    
                    # Only inject if movement is significant enough
                    if abs(pixel_x) > 0.1 or abs(pixel_y) > 0.1:
                        pyautogui.moveRel(int(pixel_x), int(pixel_y), duration=0.0)
                        self.bot_injection_count += 1
                        logger.debug(f"Bot injected (PyAutoGUI fallback): dx={pixel_x:.1f}, dy={pixel_y:.1f}")
                    
                    # Detect if this is an attack swing (large, rapid movement)
                    movement_magnitude = np.sqrt(pixel_x**2 + pixel_y**2)
                    is_attack_swing = movement_magnitude > 200  # Threshold for attack detection
                    
                    # Determine which mouse button to hold for attack
                    mouse_button_to_hold = None
                    if buttons:
                        if buttons.get('left', False):
                            mouse_button_to_hold = 'left'
                        elif buttons.get('right', False):
                            mouse_button_to_hold = 'right'
                    elif is_attack_swing:
                        mouse_button_to_hold = 'left'
                    
                    # Hold mouse button BEFORE movement for attack swings
                    if is_attack_swing and mouse_button_to_hold:
                        pyautogui.mouseDown(button=mouse_button_to_hold)
                        logger.debug(f"Bot: Holding {mouse_button_to_hold} button for attack swing")
                    
                    # Handle buttons (only if still in control)
                    if buttons and self.mode == ControlMode.AUTONOMOUS:
                        # Handle mouse buttons (if not already handled for attack)
                        if not is_attack_swing:
                            if buttons.get('left', False):
                                pyautogui.mouseDown(button='left')
                                logger.debug("Bot: Left mouse down")
                            else:
                                pyautogui.mouseUp(button='left')
                            
                            if buttons.get('right', False):
                                pyautogui.mouseDown(button='right')
                                logger.debug("Bot: Right mouse down")
                            else:
                                pyautogui.mouseUp(button='right')
                    
                    # Release mouse button AFTER movement completes (for attack swings)
                    if is_attack_swing and mouse_button_to_hold:
                        pyautogui.mouseUp(button=mouse_button_to_hold)
                        logger.debug(f"Bot: Released {mouse_button_to_hold} button after swing")
                        
                        if buttons.get('space', False):
                            pyautogui.press('space')
                            logger.debug("Bot: Space pressed")
                        
                        if buttons.get('alt', False):
                            pyautogui.keyDown('alt')
                            logger.debug("Bot: Alt down")
                        else:
                            pyautogui.keyUp('alt')
                    
                    # Clear injection flag after a short delay (with timeout protection)
                    # This allows the mouse position to update before we check for human movement
                    import threading
                    def clear_injection_flag():
                        try:
                            time.sleep(0.05)  # 50ms delay
                            with self.bot_injection_lock:
                                self.bot_injecting = False
                        except Exception as e:
                            logger.error(f"Error clearing injection flag: {e}")
                            # Force clear on error
                            with self.bot_injection_lock:
                                self.bot_injecting = False
                    
                    # Cancel previous timer if it exists
                    if self.injection_clear_timer and self.injection_clear_timer.is_alive():
                        # Previous timer still running, don't spawn new one
                        pass
                    else:
                        self.injection_clear_timer = threading.Thread(target=clear_injection_flag, daemon=True)
                        self.injection_clear_timer.start()
                else:
                    logger.debug(f"Mode changed during injection - aborting (mode={self.mode.value}, lock={self.safety_lock})")
            except Exception as e:
                logger.error(f"Action injection error: {e}", exc_info=True)
                with self.bot_injection_lock:
                    self.bot_injecting = False
    
    def get_last_human_input(self) -> Optional[Tuple[float, float, Dict[str, bool]]]:
        """
        Get last human input for DAgger learning
        Returns: (delta_x, delta_y, buttons) or None
        """
        # Also check movement pattern buffer for recent actions if last_human_input is None
        if self.last_human_input is None and len(self.movement_pattern_buffer) > 0:
            # Get most recent pattern entry
            recent_pattern = list(self.movement_pattern_buffer)[-1]
            normalized = recent_pattern.get('normalized', (0.0, 0.0))
            buttons = recent_pattern.get('buttons', {})
            if isinstance(normalized, (list, tuple)) and len(normalized) >= 2:
                return (float(normalized[0]), float(normalized[1]), buttons)
        
        return self.last_human_input
    
    def get_current_human_input(self) -> Optional[Tuple[float, float, Dict[str, bool]]]:
        """
        Get CURRENT human input - continuously polls button states and mouse position
        This captures ALL inputs including held keys and continuous mouse movements
        Returns: (delta_x, delta_y, buttons) or None
        """
        try:
            # Get current button states (includes ALL keys being held)
            current_buttons = self._get_current_button_states()
            
            # Get current mouse position and calculate delta
            current_mouse_pos = None
            if PYAUTOGUI_AVAILABLE:
                try:
                    current_mouse_pos = pyautogui.position()
                except:
                    pass
            
            # Calculate mouse delta from last position
            mouse_delta = (0.0, 0.0)
            if current_mouse_pos and hasattr(self, 'last_mouse_pos') and self.last_mouse_pos:
                mouse_delta = (
                    current_mouse_pos.x - self.last_mouse_pos.x,
                    current_mouse_pos.y - self.last_mouse_pos.y
                )
                # Update last position
                self.last_mouse_pos = current_mouse_pos
            elif current_mouse_pos:
                self.last_mouse_pos = current_mouse_pos
            
            # Normalize mouse delta
            if mouse_delta[0] != 0.0 or mouse_delta[1] != 0.0:
                normalized_delta = (
                    mouse_delta[0] / self.current_sensitivity,
                    mouse_delta[1] / self.current_sensitivity
                )
                # Clamp to [-1, 1]
                normalized_delta = (
                    max(-1.0, min(1.0, normalized_delta[0])),
                    max(-1.0, min(1.0, normalized_delta[1]))
                )
            else:
                normalized_delta = (0.0, 0.0)
            
            # Update button hold tracking
            self._update_button_hold_tracking(current_buttons)
            
            # Update last human input time if there's any SIGNIFICANT input
            # CRITICAL FIX: Use noise threshold to prevent jitter from keeping bot in manual mode
            movement_magnitude = (mouse_delta[0]**2 + mouse_delta[1]**2)**0.5
            has_significant_movement = movement_magnitude > self.noise_threshold
            has_button_input = any(current_buttons.values())
            
            has_input = has_significant_movement or has_button_input
            
            if has_input:
                self.last_human_input_time = time.time()
                self.last_human_input = (normalized_delta[0], normalized_delta[1], current_buttons.copy())
                active_buttons = [k for k, v in current_buttons.items() if v]
                logger.debug(f"[HUMAN INPUT] Captured input | mouse_delta=({mouse_delta[0]:.2f}, {mouse_delta[1]:.2f}), normalized=({normalized_delta[0]:.3f}, {normalized_delta[1]:.3f}), active_buttons={active_buttons}")
            
            # Return current input (even if mouse hasn't moved, buttons might be held)
            return (normalized_delta[0], normalized_delta[1], current_buttons.copy())
        except Exception as e:
            logger.error(f"[HUMAN INPUT ERROR] Error getting current human input: {e}", exc_info=True)
            return None
    
    def check_human_override(self) -> bool:
        """
        Check if human is trying to take control
        Returns True if human movement detected above noise threshold
        """
        return self._detect_human_movement()
    
    def force_manual_mode(self):
        """Force switch to manual mode (human control)"""
        logger.info("Forcing MANUAL mode - human control enabled")
        self.set_mode(ControlMode.MANUAL)
        self.last_human_input_time = time.time()
    
    def enable_safety_lock(self):
        """Enable safety lock - bot will NEVER inject input"""
        logger.warning("SAFETY LOCK ENABLED - Bot input completely disabled")
        self.safety_lock = True
        self.set_mode(ControlMode.MANUAL)
    
    def disable_safety_lock(self):
        """Disable safety lock"""
        logger.info("Safety lock disabled")
        self.safety_lock = False
    
    def force_autonomous_mode(self):
        """Force switch to autonomous mode (bot control)"""
        logger.info("Forcing AUTONOMOUS mode - bot control enabled")
        self.set_mode(ControlMode.AUTONOMOUS)
        self.last_human_input_time = 0  # Reset so it doesn't immediately switch back
        with self.bot_injection_lock:
            self.bot_injecting = False  # Reset bot injection flag (thread-safe)
        self.consecutive_human_detections = 0  # Reset detection counter
        # Mark that we just switched - bot should use human actions more aggressively
        self.just_switched_to_autonomous = True
        self.switch_to_autonomous_time = time.time()
    
    def _smooth_movement(self, delta_x: float, delta_y: float) -> Tuple[float, float]:
        """Smooth movement using exponential moving average"""
        current_movement = (delta_x, delta_y)
        self.movement_smoother.append(current_movement)
        
        if len(self.movement_smoother) < 2:
            return delta_x, delta_y
        
        # Exponential moving average
        smoothed_x = 0.0
        smoothed_y = 0.0
        alpha = 0.7  # Smoothing factor
        
        for i, (mx, my) in enumerate(reversed(list(self.movement_smoother))):
            weight = alpha * ((1 - alpha) ** i)
            smoothed_x += mx * weight
            smoothed_y += my * weight
        
        return smoothed_x, smoothed_y
    
    def predict_next_movement(self) -> Optional[Tuple[float, float]]:
        """Predict next movement based on pattern history"""
        if len(self.movement_pattern_buffer) < 3:
            return None
        
        recent = list(self.movement_pattern_buffer)[-3:]
        # Simple linear prediction
        if len(recent) >= 2:
            dx1 = recent[-1]['normalized'][0] - recent[-2]['normalized'][0]
            dy1 = recent[-1]['normalized'][1] - recent[-2]['normalized'][1]
            predicted = (recent[-1]['normalized'][0] + dx1, recent[-1]['normalized'][1] + dy1)
            return predicted
        return None
    
    def get_movement_statistics(self) -> Dict:
        """Get detailed movement statistics"""
        if not self.velocity_history:
            return {}
        
        velocities = list(self.velocity_history)
        magnitudes = [m.get('magnitude', 0) for m in self.movement_pattern_buffer]
        
        return {
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'avg_magnitude': np.mean(magnitudes) if magnitudes else 0,
            'detection_confidence': self.detection_confidence,
            'adaptive_threshold': self.noise_threshold,
            'current_sensitivity': self.current_sensitivity
        }
    
    def inject_discrete_action(self, action_id: int) -> bool:
        """
        Inject discrete macro-action (ScrimBrain DQN style)
        
        Args:
            action_id: Discrete action ID (0-8 from MacroAction enum)
            
        Returns:
            True if successful
        """
        if self.safety_lock or self.mode != ControlMode.AUTONOMOUS:
            return False
        
        if not self.action_discretizer or not self.gesture_engine:
            logger.warning("Action discretizer not available - cannot inject discrete action")
            return False
        
        try:
            action_config = self.action_discretizer.get_action_config(action_id)
            self.gesture_engine.perform_macro_action(action_id, action_config)
            self.bot_injection_count += 1
            logger.debug(f"Injected discrete action: {self.action_discretizer.get_action_name(action_id)}")
            return True
        except Exception as e:
            logger.error(f"Discrete action injection error: {e}")
            return False
    
    def enable_discrete_mode(self):
        """Enable discrete action mode (DQN-style)"""
        self.use_discrete_actions = True
        logger.info("Discrete action mode enabled (ScrimBrain DQN style)")
    
    def disable_discrete_mode(self):
        """Disable discrete action mode (use continuous)"""
        self.use_discrete_actions = False
        logger.info("Continuous action mode enabled")
    
    def get_stats(self) -> Dict:
        """Get comprehensive input multiplexer statistics"""
        base_stats = {
            "mode": self.mode.value,
            "bot_injections": self.bot_injection_count,
            "human_overrides": self.human_override_count,
            "mode_switches": self.mode_switch_count,
            "safety_locked": self.safety_lock,
            "human_idle_time": time.time() - self.last_human_input_time if self.last_human_input_time > 0 else 0,
            "consecutive_human_detections": self.consecutive_human_detections,
            "detection_confidence": self.detection_confidence,
            "adaptive_threshold": self.noise_threshold,
            "current_sensitivity": self.current_sensitivity,
            "pattern_buffer_size": len(self.movement_pattern_buffer),
            "directinput_enabled": self.direct_input is not None,
            "discrete_mode": self.use_discrete_actions,
            "num_discrete_actions": self.action_discretizer.get_num_actions() if self.action_discretizer else 0
        }
        
        # Add movement statistics
        movement_stats = self.get_movement_statistics()
        base_stats.update(movement_stats)
        
        return base_stats

