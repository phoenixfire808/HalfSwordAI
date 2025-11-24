"""
Actor Process: Real-time inference loop
Runs at game frame rate, captures frames, runs inference, injects actions
"""
import torch
import numpy as np
import time
import json
import logging
from typing import Dict, Optional, List
from collections import deque
from half_sword_ai.config import config
from half_sword_ai.perception.vision import ScreenCapture, MemoryReader, VisionProcessor
from half_sword_ai.input.input_mux import InputMultiplexer, ControlMode
from half_sword_ai.core.model import HalfSwordPolicyNetwork
from half_sword_ai.learning.replay_buffer import PrioritizedReplayBuffer
from half_sword_ai.monitoring.watchdog import Watchdog
# LLM integration removed
from half_sword_ai.monitoring.performance_monitor import PerformanceMonitor

# Import DQN for type checking
try:
    from half_sword_ai.core.dqn_model import DQNNetwork
    from half_sword_ai.input.action_discretizer import ActionDiscretizer
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
from half_sword_ai.learning.human_recorder import HumanActionRecorder
from half_sword_ai.perception.yolo_self_learning import YOLOSelfLearner
from half_sword_ai.perception.screen_reward_detector import ScreenRewardDetector
from half_sword_ai.perception.ocr_reward_tracker import OCRRewardTracker
from half_sword_ai.perception.terminal_state_detector import TerminalStateDetector
from half_sword_ai.learning.reward_shaper import RewardShaper, CurriculumPhase

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

class ActorProcess:
    """
    Main inference loop running at game frame rate
    Integrates perception and action injection
    """
    
    def __init__(self, model: HalfSwordPolicyNetwork, replay_buffer: PrioritizedReplayBuffer,
                 input_mux: InputMultiplexer, memory_reader: MemoryReader,
                 screen_capture: ScreenCapture, watchdog: Watchdog,
                 performance_monitor: PerformanceMonitor = None):
        self.model = model
        self.replay_buffer = replay_buffer
        self.input_mux = input_mux
        self.memory_reader = memory_reader
        self.screen_capture = screen_capture
        self.watchdog = watchdog
        # LLM removed
        self.perf_monitor = performance_monitor
        
        # Detect if using DQN (ScrimBrain-style)
        self.is_dqn = DQN_AVAILABLE and isinstance(self.model, DQNNetwork)
        if self.is_dqn:
            self.action_discretizer = ActionDiscretizer()
            logger.info("Actor initialized for DQN (ScrimBrain-style discrete actions)")
        else:
            self.action_discretizer = None
            logger.info("Actor initialized for PPO (continuous actions)")
        
        # Initialize vision processor with YOLO
        self.vision_processor = VisionProcessor(screen_capture)
        
        # Initialize YOLO self-learning system
        if config.YOLO_SELF_LEARNING_ENABLED and self.vision_processor.yolo_detector:
            self.yolo_self_learner = YOLOSelfLearner(self.vision_processor.yolo_detector)
            logger.info("🧠 YOLO self-learning enabled - Building model from rewards!")
        else:
            self.yolo_self_learner = None
        
        # Initialize screen-based reward detector (ScrimBrain-style)
        self.screen_reward_detector = ScreenRewardDetector()
        self.last_frame_for_reward = None  # Store previous frame for change detection
        logger.info("Screen-based reward detection enabled (ScrimBrain-style)")
        
        # Initialize OCR reward tracker for real score tracking (ScrimBrain integration)
        if config.OCR_ENABLED:
            try:
                self.ocr_reward_tracker = OCRRewardTracker()
                logger.info("📊 OCR reward tracker initialized - Real score tracking enabled")
            except Exception as e:
                logger.warning(f"OCR reward tracker initialization failed: {e}")
                self.ocr_reward_tracker = None
        else:
            self.ocr_reward_tracker = None
        
        # Initialize terminal state detector for real death detection (ScrimBrain integration)
        if config.TERMINAL_STATE_DETECTION:
            try:
                self.terminal_state_detector = TerminalStateDetector()
                logger.info("💀 Terminal state detector initialized - Real death detection enabled")
            except Exception as e:
                logger.warning(f"Terminal state detector initialization failed: {e}")
                self.terminal_state_detector = None
        else:
            self.terminal_state_detector = None
        
        # Initialize comprehensive reward shaper (multi-layered reward architecture)
        if config.ENABLE_COMPREHENSIVE_REWARDS:
            try:
                # Map config string to CurriculumPhase enum
                phase_map = {
                    'toddler': CurriculumPhase.TODDLER,
                    'swordsman': CurriculumPhase.SWORDSMAN,
                    'duelist': CurriculumPhase.DUELIST,
                    'master': CurriculumPhase.MASTER
                }
                curriculum_phase = phase_map.get(config.CURRICULUM_PHASE.lower(), CurriculumPhase.MASTER)
                
                self.reward_shaper = RewardShaper(
                    curriculum_phase=curriculum_phase,
                    gamma=config.GAMMA,
                    enable_pbrs=config.ENABLE_PBRS
                )
                
                # Set reward shaper parameters from config
                self.reward_shaper.alignment_power = config.REWARD_ALIGNMENT_POWER
                self.reward_shaper.balance_k = config.BALANCE_REWARD_K
                
                logger.info(f"Comprehensive reward shaper initialized - Phase: {curriculum_phase.value}, PBRS: {config.ENABLE_PBRS}")
            except Exception as e:
                logger.warning(f"Reward shaper initialization failed: {e}")
                self.reward_shaper = None
        else:
            self.reward_shaper = None
            logger.info("Comprehensive reward shaping disabled - using basic rewards only")
        
        # Previous state for PBRS (Potential-Based Reward Shaping)
        self.prev_state_for_reward: Optional[Dict] = None
        
        # Initialize human action recorder
        self.human_recorder = HumanActionRecorder()
        self.input_mux.human_action_recorder = self.human_recorder
        
        # Always start recording (will record whenever human is in control)
        self.human_recorder.start_recording()
        logger.info("Human action recorder initialized - All gameplay will be recorded!")
        
        # Track for self-learning
        self.last_detection_before_action = None
        self.detection_action_pairs = []
        
        self.running = False
        self.frame_count = 0
        self.episode_count = 0
        self.last_valid_frame = None  # Store last valid frame for fallback
        self.last_perf_report = time.time()
        self.latest_detections = {}  # Store latest YOLO detections for dashboard
        self.last_recorded_action_time = 0
        self.yolo_self_training_count = 0
        self.last_yolo_training = 0
        
        # Ensure we start in AUTONOMOUS mode to generate actions
        if self.input_mux.mode != ControlMode.AUTONOMOUS:
            logger.info("Setting initial mode to AUTONOMOUS for bot control")
            self.input_mux.force_autonomous_mode()  # Force it and reset timers
        
        # LLM removed
        
        # Pattern storage for movement emulation
        self.movement_patterns = []  # Store learned patterns
        self.pattern_match_threshold = 0.7  # Similarity threshold for pattern matching
        
        # Exploration strategy
        self.exploration_enabled = True
        self.epsilon = 0.3  # Start with 30% random exploration
        self.epsilon_min = 0.05  # Minimum 5% exploration
        self.epsilon_decay = 0.9995  # Decay epsilon over time
        self.action_noise_scale = 0.15  # Scale of action noise for exploration
        self.action_diversity_tracker = deque(maxlen=1000)  # Track action diversity
        
        # State tracking for proper next_state assignment
        self.last_recorded_state = None
        self.last_recorded_action = None
        self.pending_transitions = []  # Store transitions waiting for next_state
        
        # Enhanced dataset builder (optional)
        self.dataset_builder = None
        
    def start(self):
        """Start actor process"""
        self.running = True
        self.input_mux.start()
        logger.info("Actor process started")
        
        # Main inference loop - Optimized for 60 FPS
        min_fps = getattr(config, 'MIN_FPS_TARGET', 60)
        max_fps = getattr(config, 'MAX_FPS_TARGET', 120)
        target_fps = max(min_fps, min(config.CAPTURE_FPS, max_fps))
        frame_time = 1.0 / target_fps  # ~16.67ms per frame for 60 FPS
        logger.info(f"Target FPS: {target_fps} (frame_time: {frame_time*1000:.2f}ms)")
        
        while self.running:
            loop_start = time.time()
            current_time = time.time()  # Define at start of loop to ensure it's always available
            
            try:
                # Performance monitoring - optimized for FPS
                # Record frame timing (OPTIMIZED for 60 FPS - batch updates every 30 frames)
                if self.perf_monitor:
                    # Record frame timing (batch updates to reduce lock contention)
                    current_time_for_fps = time.time()
                    if not hasattr(self, '_last_frame_time_for_fps'):
                        self._last_frame_time_for_fps = current_time_for_fps
                    frame_time = current_time_for_fps - self._last_frame_time_for_fps
                    self._last_frame_time_for_fps = current_time_for_fps
                    
                    # Batch updates every 30 frames to reduce lock overhead (critical for 60 FPS)
                    if not hasattr(self, '_fps_batch_count'):
                        self._fps_batch_count = 0
                        self._fps_batch = []
                    
                    self._fps_batch.append(frame_time)
                    self._fps_batch_count += 1
                    
                    # Batch update every 30 frames (reduced frequency for 60 FPS)
                    if self._fps_batch_count >= 30:
                        with self.perf_monitor.lock:
                            self.perf_monitor.frame_times.extend(self._fps_batch)
                            self.perf_monitor.frame_count += len(self._fps_batch)
                            # Calculate and store FPS (use last 60 frames for speed)
                            if len(self.perf_monitor.frame_times) > 1:
                                frame_times_list = list(self.perf_monitor.frame_times)
                                avg_frame_time = np.mean(frame_times_list[-60:])  # Reduced from 100
                                fps = 1.0 / max(0.001, avg_frame_time)
                                self.perf_monitor.metrics['fps'].append(fps)
                        self._fps_batch = []
                        self._fps_batch_count = 0
                    
                    # Update expensive system metrics only every 180 frames (~3 seconds at 60 FPS)
                    if not hasattr(self, '_perf_monitor_frame_count'):
                        self._perf_monitor_frame_count = 0
                    self._perf_monitor_frame_count += 1
                    if self._perf_monitor_frame_count >= 180:
                        self.perf_monitor.update_system_metrics()
                        self._perf_monitor_frame_count = 0
                
                # 1. Perception (optimized - capture frame)
                capture_start = time.time()
                frame = self.screen_capture.get_latest_frame()
                capture_latency = time.time() - capture_start
                
                # Record capture latency (optimized - sample every 60 frames for 60 FPS)
                if self.perf_monitor and self.frame_count % 60 == 0:
                    self.perf_monitor.record_capture_latency(capture_latency)
                
                # Store valid frame for fallback (avoid copy if possible - use reference)
                if frame is not None:
                    self.last_valid_frame = frame  # Use reference, not copy for performance
                
                game_state = self.memory_reader.get_state()
                
                if frame is None:
                    # Log occasionally to debug frame capture issues
                    if not hasattr(self, '_last_frame_none_log') or time.time() - self._last_frame_none_log > 5.0:
                        logger.warning(f"[FRAME CAPTURE] Frame {self.frame_count} | Frame capture returned None - using fallback")
                        self._last_frame_none_log = time.time()
                    # Use last valid frame if available
                    if hasattr(self, 'last_valid_frame') and self.last_valid_frame is not None:
                        frame = self.last_valid_frame
                        logger.debug("Using last valid frame for action generation")
                    else:
                        # No frame available, skip to next iteration quickly
                        elapsed = time.time() - loop_start
                        sleep_time = max(0, frame_time - elapsed)
                        if sleep_time > 0:
                            time.sleep(min(sleep_time, 0.01))  # Cap sleep at 10ms
                        continue
                
                # YOLO object detection (ULTRA THROTTLED for 60 FPS - every 120 frames = ~2 seconds)
                detections = {}
                current_time_for_yolo = time.time()
                if not hasattr(self, 'last_yolo_detection_time'):
                    self.last_yolo_detection_time = 0
                    self.cached_detections = {}
                    self._yolo_frame_count = 0
                
                self._yolo_frame_count = getattr(self, '_yolo_frame_count', 0) + 1
                
                # Skip YOLO if frame is None or if we're running too slow or if not time yet
                if self._yolo_frame_count < 120:  # Run YOLO every 2 seconds at 60 FPS
                    # Use cached detections (don't run YOLO)
                    detections = self.cached_detections
                    yolo_latency = 0
                elif frame is not None and current_time_for_yolo - self.last_yolo_detection_time >= config.YOLO_DETECTION_INTERVAL:
                    yolo_start = time.time()
                    try:
                        detections = self.vision_processor.process_frame(frame)
                        yolo_latency = time.time() - yolo_start
                        self.last_yolo_detection_time = current_time_for_yolo
                        self.cached_detections = detections.copy() if detections else {}
                        self._yolo_frame_count = 0  # Reset counter after detection
                        # Only log YOLO runs occasionally
                        if not hasattr(self, '_yolo_log_count'):
                            self._yolo_log_count = 0
                        self._yolo_log_count += 1
                        if self._yolo_log_count % 10 == 0:  # Log every 10th YOLO run
                            logger.debug(f"YOLO detection: {len(detections)} objects (latency: {yolo_latency*1000:.1f}ms)")
                    except Exception as e:
                        logger.debug(f"YOLO detection failed: {e}")
                        detections = self.cached_detections
                        yolo_latency = 0
                        self._yolo_frame_count = 0  # Reset even on error
                else:
                    # Use cached detections
                    detections = self.cached_detections
                    yolo_latency = 0
                
                # Apply reward-based confidence adjustments (optimized - only when YOLO runs)
                if self.yolo_self_learner and config.YOLO_CONFIDENCE_ADJUSTMENT_ENABLED and yolo_latency > 0:
                    detections = self.yolo_self_learner.adjust_detection_confidence(detections)
                
                # Store latest detections for dashboard
                self.latest_detections = detections
                
                # Get current time for timestamp
                current_time = time.time()
                
                # Store detection for action-response learning (only if new detection, optimized - use reference)
                if yolo_latency > 0:  # Only update if we actually ran detection
                    self.last_detection_before_action = {
                        'detections': detections.copy() if detections else {},
                        'frame': frame,  # Use reference, not copy for performance
                        'timestamp': current_time
                    }
                
                if self.perf_monitor and yolo_latency > 0:
                    # Track YOLO latency separately
                    if not hasattr(self.perf_monitor, 'yolo_latencies'):
                        self.perf_monitor.yolo_latencies = deque(maxlen=100)
                    self.perf_monitor.yolo_latencies.append(yolo_latency)
                
                # Add detections to game state (enriched with all visual data)
                game_state['detections'] = detections
                game_state['threat_level'] = detections.get('threat_level', 'unknown')
                if detections.get('nearest_enemy'):
                    game_state['nearest_enemy'] = detections['nearest_enemy']
                    game_state['enemy_direction'] = detections.get('enemy_direction', (0, 0))
                
                # Add visual analysis data (motion, quality) to game_state
                if 'motion' in detections:
                    game_state['motion'] = detections['motion']
                    game_state['motion_magnitude'] = detections['motion'].get('motion_magnitude', 0.0)
                    game_state['has_motion'] = detections['motion'].get('has_motion', False)
                    game_state['motion_direction'] = detections['motion'].get('motion_direction', (0.0, 0.0))
                
                if 'quality' in detections:
                    game_state['frame_quality'] = detections['quality']
                
                # Ensure all expected keys exist (even if None)
                game_state.setdefault('health', None)
                game_state.setdefault('stamina', None)
                game_state.setdefault('enemy_health', None)
                game_state.setdefault('is_dead', None)
                game_state.setdefault('enemy_dead', None)
                game_state.setdefault('position', {'x': None, 'y': None, 'z': None})
                game_state.setdefault('combat_state', 'unknown')
                game_state.setdefault('score', None)
                game_state.setdefault('score_delta', 0)
                game_state.setdefault('ocr_reward', 0.0)
                game_state.setdefault('data_source', game_state.get('data_source', 'unknown'))
                game_state.setdefault('yolo_guidance', None)
                game_state.setdefault('terminal_detection', None)
                
                # Use terminal state detector for REAL death detection (optimized - not every frame)
                if self.terminal_state_detector and frame is not None:
                    if not hasattr(self, '_last_terminal_check'):
                        self._last_terminal_check = 0
                        self._terminal_check_frame_count = 0
                    
                    # Check terminal state every N frames (ULTRA THROTTLED for 60 FPS - every 180 frames = ~3 seconds)
                    self._terminal_check_frame_count += 1
                    if self._terminal_check_frame_count >= 180:  # Run every 3 seconds at 60 FPS
                        terminal_result = self.terminal_state_detector.detect_death_screen(frame)
                        self._cached_terminal_result = terminal_result  # Cache for reward calculation
                        if terminal_result['is_terminal']:
                            # Override game_state with real terminal detection
                            game_state['is_dead'] = True
                            game_state['combat_state'] = 'ended'
                            game_state['terminal_detection'] = terminal_result
                            logger.info(f"💀 Terminal state detected: {terminal_result['reason']} (confidence: {terminal_result['confidence']:.2f})")
                        else:
                            # Update with real detection result (even if not terminal)
                            game_state['terminal_detection'] = terminal_result
                        self._terminal_check_frame_count = 0
                else:
                    # Cache default result if terminal detector not available
                    self._cached_terminal_result = {'is_terminal': False}
                
                # Use OCR reward tracker for REAL score tracking (ULTRA THROTTLED for 60 FPS)
                # OCR is VERY expensive - only run every 300 frames (~5 seconds at 60 FPS)
                if not hasattr(self, '_ocr_frame_count'):
                    self._ocr_frame_count = 0
                self._ocr_frame_count += 1
                
                if self.ocr_reward_tracker and frame is not None and self._ocr_frame_count >= 300:
                    ocr_result = self.ocr_reward_tracker.update(frame)
                    self._cached_ocr_result = ocr_result  # Cache for reward calculation
                    if ocr_result['success']:
                        # Add real score data to game state
                        game_state['score'] = ocr_result['score']
                        game_state['score_delta'] = ocr_result['score_delta']
                        game_state['ocr_reward'] = ocr_result['reward']
                        game_state['data_source'] = 'ocr'
                        if ocr_result['score_delta'] > 0:
                            logger.debug(f"📊 Score increased: {ocr_result['score_delta']} (total: {ocr_result['score']})")
                    else:
                        # Ensure OCR keys exist even if OCR failed
                        game_state.setdefault('score', None)
                        game_state.setdefault('score_delta', 0)
                        game_state.setdefault('ocr_reward', 0.0)
                    self._ocr_frame_count = 0
                else:
                    # Use cached OCR result (don't recalculate every frame)
                    if not hasattr(self, '_cached_ocr_result'):
                        self._cached_ocr_result = {'success': False, 'reward': 0.0}
                    game_state.setdefault('score', None)
                    game_state.setdefault('score_delta', 0)
                    game_state.setdefault('ocr_reward', self._cached_ocr_result.get('reward', 0.0))
                
                # Get action guidance from YOLO self-learner
                if self.yolo_self_learner:
                    guidance = self.yolo_self_learner.get_action_guidance_from_detections(detections)
                    game_state['yolo_guidance'] = guidance
                else:
                    # Ensure key exists even if YOLO self-learner not available
                    game_state.setdefault('yolo_guidance', None)
                
                # 2. Watchdog check
                watchdog_result = self.watchdog.check_game_state()
                if watchdog_result["action"] == "restart":
                    logger.info(f"Watchdog triggered restart: {watchdog_result['status']}")
                    self._handle_restart()
                    continue
                
                # 3. Check for human override ONLY if we're in AUTONOMOUS mode
                # Don't check if already in MANUAL mode to prevent fighting for control
                if self.input_mux.mode.value == 'autonomous' and self.input_mux.check_human_override():
                    # Human detected - switch to manual mode immediately
                    logger.info(f"[MODE SWITCH] Human input detected - switching AUTONOMOUS -> MANUAL | Frame: {self.frame_count}")
                    logger.debug(f"[MODE SWITCH] Detection confidence: {getattr(self.input_mux, 'detection_confidence', 0.0):.2f} | Bot injecting: {getattr(self.input_mux, 'bot_injecting', False)}")
                    self.input_mux.force_manual_mode()
                    if self.perf_monitor:
                        self.perf_monitor.record_warning("Human override detected", "input_mux")
                
                # 4. Model inference or human control
                # Check mode FIRST - if MANUAL, skip bot action generation entirely
                # This prevents bot from fighting for control when human is active
                current_mode = self.input_mux.mode.value
                human_active = (current_mode == 'manual')
                
                # Log mode state for debugging
                if self.frame_count % 60 == 0:  # Log every 60 frames (1 second at 60 FPS)
                    logger.debug(f"[MODE CHECK] Frame {self.frame_count} | Mode: {current_mode} | Human active: {human_active} | Bot injecting: {getattr(self.input_mux, 'bot_injecting', False)}")
                
                # Initialize variables
                human_action = None
                current_time = time.time()
                bot_action = None
                
                # If human is active, skip bot action generation completely
                if human_active:
                    logger.debug(f"[HUMAN ACTIVE] Frame {self.frame_count} | Skipping bot action generation - human has control")
                    # Human is controlling - only record human actions, don't generate bot actions
                    human_input = self.input_mux.get_last_human_input()
                    if human_input is not None:
                        human_action = human_input
                        logger.debug(f"[HUMAN ACTIVE] Got last human input: dx={human_input[0]:.3f}, dy={human_input[1]:.3f}, buttons={human_input[2] if len(human_input) > 2 else {}}")
                    # Skip bot action generation - human has control
                    inference_latency = 0.0
                else:
                    # Human not active - generate bot action
                    logger.debug(f"[BOT ACTIVE] Frame {self.frame_count} | Generating bot action")
                    inference_start = time.time()
                    bot_action = self._get_bot_action(frame, game_state)
                    inference_latency = time.time() - inference_start
                    if bot_action is not None:
                        logger.debug(f"[BOT ACTIVE] Generated action: dx={bot_action[0]:.3f}, dy={bot_action[1]:.3f} | Inference latency: {inference_latency*1000:.2f}ms")
                    else:
                        logger.warning(f"[BOT ACTIVE] Bot action is None - cannot inject")
                
                # Record inference latency (optimized - sample every 60 frames for 60 FPS)
                if self.perf_monitor and self.frame_count % 60 == 0:
                    self.perf_monitor.record_inference_latency(inference_latency)
                
                # Record frame to enhanced dataset builder (if enabled)
                if self.dataset_builder and config.ENABLE_DATASET_COLLECTION:
                    # Check collection mode
                    should_record = False
                    if config.DATASET_COLLECTION_MODE == "continuous":
                        should_record = True
                    elif config.DATASET_COLLECTION_MODE == "human_only" and human_active:
                        should_record = True
                    elif config.DATASET_COLLECTION_MODE == "episode_only" and self.frame_count == 0:
                        # Start recording at episode start
                        if not self.dataset_builder.recording:
                            self.dataset_builder.start_recording()
                        should_record = True
                    
                    if should_record:
                        try:
                            self.dataset_builder.record_frame()
                            
                            # Auto-save periodically
                            if len(self.dataset_builder.entries) > 0 and \
                               len(self.dataset_builder.entries) % config.DATASET_SAVE_INTERVAL == 0:
                                logger.info(f"[DATASET] Auto-saving dataset: {len(self.dataset_builder.entries)} entries")
                                self.dataset_builder._save_dataset()
                                # Start new dataset
                                dataset_name = f"{config.DATASET_NAME_PREFIX}_{int(time.time())}"
                                self.dataset_builder.dataset_name = dataset_name
                                self.dataset_builder.entries = []
                        except Exception as e:
                            logger.debug(f"Dataset recording error: {e}")
                
                if human_active:
                    # Log occasionally to show why bot isn't controlling
                    if not hasattr(self, '_last_human_active_log') or time.time() - self._last_human_active_log > 5.0:
                        logger.info(f"[HUMAN CONTROL] Frame {self.frame_count} | Mode: {self.input_mux.mode.value} | Recording actions, bot paused")
                        self._last_human_active_log = time.time()
                    # Human is controlling - record EVERYTHING for learning
                    # Record at maximum frequency to capture exact patterns
                    current_time = time.time()
                    
                    # Get CURRENT human input (continuously polls button states and mouse)
                    # This captures ALL inputs including held keys and continuous mouse movements
                    human_action = self.input_mux.get_current_human_input()
                    if human_action:
                        logger.debug(f"[HUMAN INPUT] Frame {self.frame_count} | dx={human_action[0]:.3f}, dy={human_action[1]:.3f}, buttons={human_action[2] if len(human_action) > 2 else {}}")
                    else:
                        logger.debug(f"[HUMAN INPUT] Frame {self.frame_count} | No human input detected")
                    
                    # Debug logging DISABLED for 60 FPS performance
                    # #region debug log (DISABLED)
                    if False:  # Disabled for 60 FPS - file I/O is expensive
                        if not hasattr(self, '_check_human_input_log_frame_count'):
                            self._check_human_input_log_frame_count = 0
                        self._check_human_input_log_frame_count += 1
                        if self._check_human_input_log_frame_count >= 50:
                            try:
                                with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                                    log_entry = {
                                        'timestamp': time.time(),
                                        'location': 'actor.py:main_loop:check_human_input',
                                        'message': 'Checking for human input',
                                        'data': {
                                            'has_human_action': human_action is not None,
                                            'mode': self.input_mux.mode.value,
                                            'expert_buffer_size': len(self.human_recorder.expert_buffer) if self.human_recorder else 0,
                                            'human_action_type': type(human_action).__name__ if human_action else None
                                        },
                                        'sessionId': 'debug-session',
                                        'runId': 'run1',
                                        'hypothesisId': 'G'
                                    }
                                    f.write(json.dumps(log_entry) + '\n')
                            except:
                                pass
                        self._check_human_input_log_frame_count = 0
                    # #endregion
                    
                    # Always record when human is active, even if no movement (keys might be held)
                    if human_action is not None:
                        # Get button states - includes ALL keys being held down
                        buttons = human_action[2] if len(human_action) > 2 else {}
                        
                        # Ensure we have button states even if human_action is incomplete
                        if not buttons:
                            buttons = self.input_mux._get_current_button_states() if hasattr(self.input_mux, '_get_current_button_states') else {}
                        
                        # Log button states for debugging
                        active_buttons = [k for k, v in buttons.items() if v]
                        if active_buttons:
                            logger.debug(f"[HUMAN BUTTONS] Frame {self.frame_count} | Active buttons: {active_buttons}")
                        
                        # Only get movement pattern every 10 frames for 60 FPS
                        if not hasattr(self, '_human_pattern_frame_count'):
                            self._human_pattern_frame_count = 0
                        self._human_pattern_frame_count += 1
                        
                        movement_pattern = []
                        pattern_context = {}
                        if self._human_pattern_frame_count >= 10:  # Reduced frequency for 60 FPS
                            try:
                                movement_pattern = self.input_mux.get_movement_pattern(lookback=5)  # Reduced from 10 for performance
                            except Exception as e:
                                movement_pattern = []
                            
                            # Ensure movement_pattern is a list
                            if not isinstance(movement_pattern, list):
                                movement_pattern = []
                            
                            # Enhanced pattern context (simplified for performance)
                            pattern_context = {
                                'pattern': movement_pattern[-2:] if len(movement_pattern) >= 2 else movement_pattern,  # Reduced from 3
                                'pattern_type': self._classify_pattern(movement_pattern[-2:]) if len(movement_pattern) >= 2 else 'unknown',
                                'detection_confidence': self.input_mux.detection_confidence if hasattr(self.input_mux, 'detection_confidence') else 0.0
                            }
                            self._human_pattern_frame_count = 0
                        
                        # Get enhanced movement data from input mux (optimized - only when needed)
                        if self.frame_count % 30 == 0:  # Reduced frequency for 60 FPS
                            movement_stats = self.input_mux.get_movement_statistics() if hasattr(self.input_mux, 'get_movement_statistics') else {}
                            velocity = movement_stats.get('avg_velocity', 0) if movement_stats else None
                            acceleration = movement_stats.get('avg_velocity', 0) if movement_stats else None
                        else:
                            velocity = None
                            acceleration = None
                        
                        # Extract mouse delta from human_action for proper recording
                        mouse_delta_for_recording = None
                        if isinstance(human_action, tuple) and len(human_action) >= 2:
                            # human_action contains normalized deltas, convert back to pixel deltas
                            delta_x_norm = human_action[0] if len(human_action) > 0 else 0.0
                            delta_y_norm = human_action[1] if len(human_action) > 1 else 0.0
                            # Convert normalized back to pixels using sensitivity
                            sensitivity = getattr(self.input_mux, 'current_sensitivity', 400.0)
                            mouse_delta_for_recording = (delta_x_norm * sensitivity, delta_y_norm * sensitivity)
                            
                            # Debug logging DISABLED for 60 FPS performance
                            # #region debug log (DISABLED)
                            if False:  # Disabled for 60 FPS
                                if not hasattr(self, '_human_action_log_frame_count'):
                                    self._human_action_log_frame_count = 0
                                self._human_action_log_frame_count += 1
                                if self._human_action_log_frame_count >= 50:
                                    try:
                                        with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                                            log_entry = {
                                                'timestamp': time.time(),
                                                'location': 'actor.py:record_human_action',
                                                'message': 'Recording human action',
                                                'data': {
                                                    'delta_x_norm': float(delta_x_norm),
                                                    'delta_y_norm': float(delta_y_norm),
                                                    'mouse_delta_pixels': (float(mouse_delta_for_recording[0]), float(mouse_delta_for_recording[1])) if mouse_delta_for_recording else None,
                                                    'buttons': buttons,
                                                    'expert_buffer_size': len(self.human_recorder.expert_buffer) if self.human_recorder else 0
                                                },
                                                'sessionId': 'debug-session',
                                                'runId': 'run1',
                                                'hypothesisId': 'E'
                                            }
                                            f.write(json.dumps(log_entry) + '\n')
                                    except:
                                        pass
                                self._human_action_log_frame_count = 0
                            # #endregion
                        
                        # Record with optimized context (use frame reference, not copy)
                        logger.debug(f"[HUMAN RECORD] Frame {self.frame_count} | Recording action: mouse_delta={mouse_delta_for_recording}, buttons={buttons}, reward={reward:.4f}")
                        self.human_recorder.record_action(
                            frame=frame,  # Use reference, not copy
                            game_state=game_state,
                            mouse_delta=mouse_delta_for_recording,  # Pass actual mouse delta
                            buttons=buttons,
                            timestamp=current_time,
                            velocity=velocity,
                            acceleration=acceleration,
                            pattern_context=pattern_context
                        )
                        logger.debug(f"[HUMAN RECORD] Frame {self.frame_count} | Action recorded | Expert buffer size: {len(self.human_recorder.expert_buffer) if self.human_recorder else 0}")
                        
                        # Also record in replay buffer for immediate learning with enhanced priority
                        # Calculate TD error estimate for better priority (simple reward-based)
                        # Pass frame for screen-based reward detection and action for energy efficiency
                        # Extract delta_x and delta_y from human_action for reward calculation
                        if isinstance(human_action, tuple) and len(human_action) >= 2:
                            delta_x = human_action[0] if len(human_action) > 0 else 0.0
                            delta_y = human_action[1] if len(human_action) > 1 else 0.0
                            human_action_array = np.array([
                                delta_x,
                                delta_y,
                                1.0 if buttons.get('left', False) else 0.0,
                                1.0 if buttons.get('right', False) else 0.0,
                                1.0 if buttons.get('space', False) else 0.0,
                                1.0 if buttons.get('alt', False) else 0.0
                            ])
                        else:
                            human_action_array = None
                        reward = self._calculate_reward(game_state, frame=frame, action=human_action_array)
                        td_error_estimate = abs(reward)  # Simple estimate based on reward magnitude
                        
                        self._record_human_action(frame, game_state, human_action, td_error=td_error_estimate)
                        
                        # Store pattern for pattern-based learning (throttled - only every 30 frames for 60 FPS)
                        if len(movement_pattern) >= 5 and self.frame_count % 30 == 0:
                            self._store_movement_pattern(movement_pattern, game_state, frame)
                        
                        self.last_recorded_action_time = current_time
                        
                        # Track human action count
                        if not hasattr(self, '_human_action_count'):
                            self._human_action_count = 0
                        self._human_action_count += 1
                        
                        if self.perf_monitor:
                            self.perf_monitor.current_episode["human_interventions"] = \
                                self.perf_monitor.current_episode.get("human_interventions", 0) + 1
                    
                    # Skip bot action recording when human is active - human has full control
                    # Bot action is None when human is active, so nothing to record
                    # Continue to next frame - don't generate or inject bot actions
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_time - elapsed)
                    if elapsed < frame_time:
                        time.sleep(min(sleep_time, 0.001))
                    continue  # Skip bot action generation completely
                
                # ALSO record human actions even when bot is controlling (for continuous learning)
                # This allows the bot to learn from human corrections even during autonomous mode
                if human_action and self.input_mux.mode.value == 'autonomous':
                    # Human provided input while bot was controlling - record for learning
                    buttons = human_action[2] if len(human_action) > 2 else {}
                    
                    # Extract mouse delta for recording
                    mouse_delta_for_recording = None
                    if isinstance(human_action, tuple) and len(human_action) >= 2:
                        delta_x_norm = human_action[0] if len(human_action) > 0 else 0.0
                        delta_y_norm = human_action[1] if len(human_action) > 1 else 0.0
                        sensitivity = getattr(self.input_mux, 'current_sensitivity', 400.0)
                        mouse_delta_for_recording = (delta_x_norm * sensitivity, delta_y_norm * sensitivity)
                    
                    # Record human action even in autonomous mode (for learning from corrections)
                    if self.human_recorder:
                        self.human_recorder.record_action(
                            frame=frame,
                            game_state=game_state,
                            mouse_delta=mouse_delta_for_recording,
                            buttons=buttons,
                            timestamp=current_time
                        )
                
                if self.input_mux.mode.value == 'autonomous':
                    # Bot is controlling - inject actions
                    if not hasattr(self, '_last_bot_control_log') or time.time() - self._last_bot_control_log > 5.0:
                        logger.info(f"Bot is controlling (mode: {self.input_mux.mode.value}) - generating actions")
                        self._last_bot_control_log = time.time()
                    
                    # IMMEDIATE HUMAN ACTION REPLAY - replay last N human actions when switching to autonomous
                    if not hasattr(self, '_human_action_replay_queue'):
                        self._human_action_replay_queue = []
                        self._last_mode = None
                    
                    # Check if we just switched to autonomous mode
                    if self._last_mode != 'autonomous' and self.input_mux.mode.value == 'autonomous':
                        # Just switched to autonomous - replay last human actions immediately
                        logger.info(f"[MODE SWITCH] Frame {self.frame_count} | Just switched to AUTONOMOUS - replaying human actions for immediate learning")
                        if self.human_recorder and len(self.human_recorder.expert_buffer) > 0:
                            # Get more recent human actions (last 15) for better replay
                            recent_human_actions = self.human_recorder.get_expert_actions(count=15)
                            if recent_human_actions:
                                # Convert to action queue for immediate replay - replay more actions
                                replay_count = min(10, len(recent_human_actions))  # Replay up to 10 actions
                                for action_record in recent_human_actions[-replay_count:]:
                                    buttons = action_record.get('buttons', {})
                                    normalized_delta = action_record.get('normalized_delta', (0, 0))
                                    
                                    # Convert to action format
                                    replay_action = np.array([
                                        normalized_delta[0],
                                        normalized_delta[1],
                                        1.0 if buttons.get('left', False) else 0.0,
                                        1.0 if buttons.get('right', False) else 0.0,
                                        1.0 if buttons.get('space', False) else 0.0,
                                        1.0 if buttons.get('alt', False) else 0.0
                                    ])
                                    self._human_action_replay_queue.append(replay_action)
                                logger.info(f"Replaying {len(self._human_action_replay_queue)} human actions immediately")
                    
                    self._last_mode = self.input_mux.mode.value
                    
                    # Replay human actions first if queue has items
                    if len(self._human_action_replay_queue) > 0:
                        replay_action = self._human_action_replay_queue.pop(0)
                        # Ensure replay_action is numpy array
                        if not isinstance(replay_action, np.ndarray):
                            replay_action = np.array(replay_action)
                        # Ensure correct shape
                        if len(replay_action) < 6:
                            replay_action = np.pad(replay_action, (0, max(0, 6 - len(replay_action))), 'constant')
                        injection_start = time.time()
                        self._inject_action(replay_action)
                        injection_latency = time.time() - injection_start
                        btn_str = ""
                        if replay_action[2] > 0.5:
                            btn_str += "👊L"
                        if replay_action[3] > 0.5:
                            btn_str += "👊R"
                        btn_str = btn_str if btn_str else "none"
                        logger.info(f"REPLAY | dx={replay_action[0]:.3f} dy={replay_action[1]:.3f} | {btn_str}")
                        self._record_bot_action(frame, game_state, replay_action, was_injected=True)
                        # Skip bot action generation this frame - use replayed human action
                        elapsed = time.time() - loop_start
                        sleep_time = max(0, frame_time - elapsed)
                        if elapsed < frame_time:
                            time.sleep(min(sleep_time, 0.001))
                        continue
                    
                    if bot_action is not None:
                        # Log action generation (occasionally to reduce spam)
                        if self.frame_count % 100 == 0:
                            logger.debug(f"Bot action: dx={bot_action[0]:.3f}, dy={bot_action[1]:.3f}, "
                                       f"buttons=[L:{bot_action[2]:.1f} R:{bot_action[3]:.1f} S:{bot_action[4]:.1f} A:{bot_action[5]:.1f}]")
                        
                        # Enhance action with YOLO detection guidance and self-learning
                        if detections.get('nearest_enemy') and detections.get('enemy_direction'):
                            # Adjust action based on enemy position
                            direction = detections['enemy_direction']
                            # Slight bias toward enemy direction
                            bot_action[0] = bot_action[0] * 0.7 + direction[0] * 0.3  # X component
                            bot_action[1] = bot_action[1] * 0.7 + direction[1] * 0.3  # Y component
                        
                        # Use YOLO self-learning guidance if available
                        if game_state.get('yolo_guidance'):
                            guidance = game_state['yolo_guidance']
                            if guidance.get('target_direction'):
                                target_dir = guidance['target_direction']
                                # Blend with guidance based on confidence
                                guidance_weight = guidance.get('confidence', 0.5) * 0.2
                                bot_action[0] = bot_action[0] * (1 - guidance_weight) + target_dir[0] * guidance_weight
                                bot_action[1] = bot_action[1] * (1 - guidance_weight) + target_dir[1] * guidance_weight
                        
                        # CRITICAL: Check mode again before injection - human might have taken over
                        if self.input_mux.mode.value != 'autonomous':
                            # Mode changed to manual - skip injection completely
                            logger.debug(f"[INJECTION BLOCKED] Frame {self.frame_count} | Mode changed to {self.input_mux.mode.value} - skipping injection")
                            continue  # Human has control - skip to next frame
                        
                        # Final mode check - ensure we're still in autonomous before injection
                        if self.input_mux.mode.value == 'autonomous' and bot_action is not None:
                            logger.debug(f"[INJECTION] Frame {self.frame_count} | Injecting bot action: dx={bot_action[0]:.3f}, dy={bot_action[1]:.3f}")
                            # Ensure bot_action is numpy array
                            if hasattr(bot_action, 'cpu'):
                                bot_action = bot_action.cpu().numpy()
                            elif not isinstance(bot_action, np.ndarray):
                                bot_action = np.array(bot_action)
                            
                            injection_start = time.time()
                            self._inject_action(bot_action)
                            injection_latency = time.time() - injection_start
                            logger.debug(f"[INJECTION] Frame {self.frame_count} | Injection completed | Latency: {injection_latency*1000:.2f}ms")
                            
                            # Record injection latency (optimized - sample every 60 frames for 60 FPS)
                            if self.perf_monitor and self.frame_count % 60 == 0:
                                self.perf_monitor.record_injection_latency(injection_latency)
                            
                            if self.frame_count % 50 == 0:  # More frequent logging
                                logger.info(f"[INJECTION] Frame {self.frame_count} | Bot action injected")
                                btn_str = ""
                                if bot_action[2] > 0.5:
                                    btn_str += "L"
                                if bot_action[3] > 0.5:
                                    btn_str += "R"
                                if bot_action[4] > 0.5:
                                    btn_str += "SPACE"
                                if bot_action[5] > 0.5:
                                    btn_str += "ALT"
                                btn_str = btn_str if btn_str else "none"
                                logger.info(f"INJECTED | dx={bot_action[0]:.3f} dy={bot_action[1]:.3f} | {btn_str} | latency: {injection_latency*1000:.1f}ms")
                                
                                # Record for self-learning (after successful injection)
                                self._record_bot_action(frame, game_state, bot_action, was_injected=True)
                            else:
                                logger.warning("Bot action is None - cannot inject")
                            
                            # Record detection-action-reward pair for YOLO self-learning
                            if self.yolo_self_learner and self.last_detection_before_action:
                                # Calculate reward for this action (with screen-based detection)
                                reward = self._calculate_reward(game_state, frame=frame, action=bot_action)
                                
                                # Record pair for self-learning
                                self.yolo_self_learner.record_detection_action_pair(
                                    frame=self.last_detection_before_action['frame'],
                                    detections=self.last_detection_before_action['detections'],
                                    action=bot_action,
                                    reward=reward,
                                    game_state=game_state,
                                    timestamp=self.last_detection_before_action['timestamp']
                                )
                    else:
                        # Bot action generation returned None - use fallback
                        logger.warning("Bot action generation returned None - generating fallback action")
                        fallback_action = self._get_fallback_action(game_state)
                        # Convert to numpy
                        if isinstance(fallback_action, torch.Tensor):
                            bot_action_np = fallback_action.cpu().numpy()
                        elif isinstance(fallback_action, np.ndarray):
                            bot_action_np = fallback_action
                        else:
                            bot_action_np = np.array(fallback_action)
                        
                        if self.input_mux.mode.value == 'autonomous':
                            injection_start = time.time()
                            self._inject_action(bot_action_np)
                            injection_latency = time.time() - injection_start
                            if self.perf_monitor:
                                self.perf_monitor.record_injection_latency(injection_latency)
                            logger.info(f"🔄 Fallback action injected: dx={bot_action_np[0]:.3f}, dy={bot_action_np[1]:.3f} (latency: {injection_latency*1000:.2f}ms)")
                            self._record_bot_action(frame, game_state, bot_action_np, was_injected=True)
                        else:
                            # Record fallback action for learning even if not injected
                            self._record_bot_action(frame, game_state, bot_action_np, was_injected=False)
                
                # Increment frame count (always increment to track loop iterations)
                self.frame_count += 1
                
                # Frame count recorded for monitoring
                if self.perf_monitor:
                    self.perf_monitor.current_episode["frame_count"] = self.frame_count
                
                # Periodic status logging (every 10 seconds or 300 frames, whichever comes first)
                if not hasattr(self, '_last_status_log'):
                    self._last_status_log = 0
                    self._last_status_frame_count = 0
                
                elapsed_since_log = current_time - self._last_status_log
                frames_since_log = self.frame_count - self._last_status_frame_count
                
                # Reduced logging frequency for 60 FPS (every 60s or 3600 frames)
                if elapsed_since_log >= 60.0 or frames_since_log >= 3600:
                    # Calculate FPS
                    if elapsed_since_log > 0:
                        actual_fps = frames_since_log / elapsed_since_log
                    else:
                        actual_fps = 0
                    
                    # Get buffer size
                    buffer_size = len(self.replay_buffer)
                    
                    # Get stats
                    inference_count = self.perf_monitor.current_episode.get("inference_count", 0) if self.perf_monitor else 0
                    injection_count = self.perf_monitor.current_episode.get("injection_count", 0) if self.perf_monitor else 0
                    
                    logger.info(f"📊 Status: Frames={self.frame_count} ({actual_fps:.1f} fps), Buffer={buffer_size}, "
                              f"Inference={inference_count}, Injection={injection_count}, "
                              f"Mode={self.input_mux.mode.value}")
                    
                    self._last_status_log = current_time
                    self._last_status_frame_count = self.frame_count
                
                # YOLO self-training (periodic)
                if (self.yolo_self_learner and 
                    self.yolo_self_learner.total_labels_created >= config.YOLO_SELF_TRAINING_INTERVAL and
                    current_time - self.last_yolo_training > 300):  # Train every 5 minutes max
                    
                    logger.info(f"🔄 YOLO self-training triggered ({self.yolo_self_learner.total_labels_created} labels created)")
                    self.yolo_self_learner.train_on_self_labels(epochs=5)
                    self.last_yolo_training = current_time
                    self.yolo_self_training_count += 1
                
                # Generate performance report periodically
                if self.perf_monitor and current_time - self.last_perf_report > config.PERFORMANCE_REPORT_INTERVAL:
                    report = self.perf_monitor.generate_report(
                        f"{config.LOG_PATH}/performance_report_{int(current_time)}.txt"
                    )
                    logger.info(f"\n{report}")
                    self.last_perf_report = current_time
                
                # Maintain target FPS - optimized timing (no sleep if behind schedule)
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                
                # Target 60 FPS - adjust sleep time accordingly (~16.67ms per frame)
                # If we're taking too long, skip sleep entirely to maximize throughput
                if elapsed < frame_time:  # Only sleep if we're ahead of schedule
                    if sleep_time > 0.0005:  # Only sleep if > 0.5ms (more aggressive for 60 FPS)
                        time.sleep(min(sleep_time, 0.0005))  # Cap at 0.5ms for 60 FPS
                elif elapsed > frame_time * 1.2:  # Warn if significantly behind (reduced threshold)
                    if not hasattr(self, '_last_slow_frame_warn') or time.time() - self._last_slow_frame_warn > 10.0:
                        actual_fps = 1.0 / elapsed if elapsed > 0 else 0
                        logger.warning(f"Frame processing slow: {elapsed*1000:.2f}ms (target: {frame_time*1000:.2f}ms, FPS: {actual_fps:.1f})")
                        self._last_slow_frame_warn = time.time()
                
            except KeyboardInterrupt:
                logger.info(f"[ACTOR LOOP] Interrupted by user | Frame: {self.frame_count}")
                self.running = False
                break
            except SystemExit:
                logger.info(f"[ACTOR LOOP] Exiting | Frame: {self.frame_count}")
                self.running = False
                break
            except Exception as e:
                logger.error(f"[ACTOR LOOP ERROR] Frame {self.frame_count} | Error: {e}", exc_info=True)
                logger.debug(f"[ACTOR LOOP ERROR] Error type: {type(e).__name__} | Error args: {e.args}")
                if self.perf_monitor:
                    try:
                        self.perf_monitor.record_error(e, "actor_loop")
                    except Exception as record_error:
                        logger.error(f"[ACTOR LOOP ERROR] Failed to record error: {record_error}")
                
                # Check if we should stop due to errors
                error_type = type(e).__name__
                critical_errors = ['MemoryError', 'SystemError']
                
                if error_type in critical_errors:
                    logger.critical(f"[ACTOR LOOP ERROR] Critical error {error_type} - stopping actor loop | Frame: {self.frame_count}")
                    self.running = False
                    break
                else:
                    # Non-critical error - log and continue
                    logger.warning(f"[ACTOR LOOP ERROR] Non-critical error in actor loop: {e} - continuing | Frame: {self.frame_count}")
                
                time.sleep(0.1)  # Brief pause before retrying
    
    def stop(self):
        """Stop actor process"""
        self.running = False
        
        # Stop recording
        if self.human_recorder:
            self.human_recorder.stop_recording()
        
        self.input_mux.stop()
        self.screen_capture.stop()
        logger.info("Actor process stopped")
    
    # LLM methods removed
    
    def _get_bot_action(self, frame: np.ndarray, game_state: Dict) -> Optional[torch.Tensor]:
        """Get action from neural network - enhanced with pattern matching"""
        try:
            # Ensure frame is valid
            if frame is None:
                logger.warning("Cannot generate action - frame is None")
                return self._get_fallback_action(game_state)
            # First, try to match learned movement patterns (throttled for performance)
            # Match more frequently when just switched back from manual mode
            if not hasattr(self, '_pattern_match_frame_count'):
                self._pattern_match_frame_count = 0
            self._pattern_match_frame_count += 1
            
            # Check if we should match patterns more frequently after switching back
            match_interval = 30  # Default: every 30 frames (~0.5 seconds)
            if hasattr(self.input_mux, 'just_switched_to_autonomous') and self.input_mux.just_switched_to_autonomous:
                time_since_switch = time.time() - self.input_mux.switch_to_autonomous_time
                if time_since_switch < self.input_mux.human_action_priority_duration:
                    match_interval = 5  # Match every 5 frames (~83ms) when just switched back
            
            matched_pattern = None
            if self._pattern_match_frame_count >= match_interval:
                # Try to match human movement patterns for imitation
                matched_pattern = self._match_human_pattern(game_state)
                
                # Debug logging DISABLED for 60 FPS
                # #region debug log (DISABLED)
                if False:  # Disabled for 60 FPS
                    if not hasattr(self, '_pattern_log_frame_count'):
                        self._pattern_log_frame_count = 0
                    self._pattern_log_frame_count += 1
                    if self._pattern_log_frame_count >= 50:
                        try:
                            with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                                log_entry = {
                                    'timestamp': time.time(),
                                    'location': 'actor.py:_get_bot_action:pattern_match',
                                    'message': 'Pattern matching result',
                                    'data': {
                                        'matched_pattern_found': matched_pattern is not None,
                                        'expert_buffer_size': len(self.human_recorder.expert_buffer) if hasattr(self, 'human_recorder') and self.human_recorder else 0,
                                        'pattern_dx': float(matched_pattern['normalized'][0]) if matched_pattern else None,
                                        'pattern_dy': float(matched_pattern['normalized'][1]) if matched_pattern else None
                                    },
                                    'sessionId': 'debug-session',
                                    'runId': 'run1',
                                    'hypothesisId': 'H'
                                }
                                f.write(json.dumps(log_entry) + '\n')
                        except:
                            pass
                    self._pattern_log_frame_count = 0
                # #endregion
                
                self._pattern_match_frame_count = 0
            
            # Build frame stack (optimized - avoid unnecessary stacking)
            frame_stack = self.screen_capture.get_frame_stack()
            if frame_stack is None:
                # Not enough frames yet, use current frame repeated (optimized - reuse frame)
                if not hasattr(self, '_cached_frame_stack') or self._cached_frame_stack is None:
                    self._cached_frame_stack = np.stack([frame] * config.FRAME_STACK_SIZE, axis=0)
                frame_stack = self._cached_frame_stack
            else:
                # Cache valid frame stack
                self._cached_frame_stack = frame_stack
            
            # Ensure frame_stack is (T, H, W) and add channel dimension if needed
            if len(frame_stack.shape) == 2:
                # Single frame, stack it (use cached if available)
                if hasattr(self, '_cached_frame_stack') and self._cached_frame_stack is not None:
                    frame_stack = self._cached_frame_stack
                else:
                    frame_stack = np.stack([frame_stack] * config.FRAME_STACK_SIZE, axis=0)
                    self._cached_frame_stack = frame_stack
            
            # Convert to tensor: (T, H, W) -> (1, T, H, W) for batch
            # Model expects (batch, channels=T, H, W) where channels = frame_stack_size
            frame_tensor = torch.FloatTensor(frame_stack).unsqueeze(0).to(config.DEVICE)
            
            # Prepare state features
            state_features = self._prepare_state_features(game_state)
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(config.DEVICE)
            
            # Model inference - handle DQN vs PPO differently
            with torch.no_grad():
                if self.is_dqn:
                    # DQN: get discrete action index (0-8)
                    epsilon = getattr(self, 'epsilon', 0.0) if hasattr(self, 'epsilon') else 0.0
                    action_id = self.model.get_action(frame_tensor, epsilon=epsilon)
                    
                    # Convert discrete action to continuous format for injection
                    if self.action_discretizer:
                        action_config = self.action_discretizer.get_action_config(action_id)
                        # Create action array: [dx_normalized, dy_normalized, left, right, space, alt]
                        dx = action_config.get('dx', 0) / 400.0  # Normalize to [-1, 1]
                        dy = action_config.get('dy', 0) / 400.0
                        keys = action_config.get('keys', {})
                        
                        # Ensure strikes have mouse button pressed
                        action_name = action_config.get('name', '')
                        is_strike = any(word in action_name.lower() for word in ['strike', 'slash', 'thrust'])
                        if is_strike and not keys.get('left', False) and not keys.get('right', False):
                            # Default to left button for strikes if no button specified
                            keys['left'] = True
                        
                        action = torch.tensor([
                            dx, dy,
                            1.0 if keys.get('left', False) else 0.0,
                            1.0 if keys.get('right', False) else 0.0,
                            1.0 if keys.get('space', False) else 0.0,
                            1.0 if keys.get('alt', False) else 0.0
                        ], dtype=torch.float32)
                        
                        # Log action selection occasionally
                        if self.frame_count % 100 == 0:
                            btn_str = ""
                            if keys.get('left', False):
                                btn_str += "L"
                            if keys.get('right', False):
                                btn_str += "R"
                            if keys.get('space', False):
                                btn_str += "SPACE"
                            if keys.get('alt', False):
                                btn_str += "🗡️"
                            btn_str = btn_str if btn_str else "none"
                            logger.info(f"ACTION | {action_name} | dx={dx:.3f} dy={dy:.3f} | {btn_str}")
                    else:
                        # Fallback: neutral action (shouldn't happen if action_discretizer is set)
                        logger.warning("Action discretizer not available - using fallback action")
                        action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    # PPO: get continuous and discrete actions
                    continuous_action, discrete_action = self.model.get_action(
                        frame_tensor, state_tensor, deterministic=False
                    )
                    
                    # Combine actions
                    action = torch.cat([
                        continuous_action.squeeze(),
                        discrete_action.squeeze()
                    ])
            
            # EXPLORATION: Epsilon-greedy strategy
            import random
            if self.exploration_enabled and random.random() < self.epsilon:
                # Random exploration action
                exploration_action = self._get_exploration_action(game_state)
                # Convert to same type as action
                if isinstance(action, torch.Tensor) and isinstance(exploration_action, torch.Tensor):
                    action = exploration_action
                elif isinstance(action, torch.Tensor):
                    action = torch.tensor(exploration_action, dtype=action.dtype, device=action.device)
                else:
                    action = exploration_action
                if self.perf_monitor:
                    self.perf_monitor.current_episode["exploration_actions"] = \
                        self.perf_monitor.current_episode.get("exploration_actions", 0) + 1
            else:
                # Add action noise for exploration even when using model
                if self.exploration_enabled and isinstance(action, torch.Tensor):
                    noise = torch.randn_like(action[:2]) * self.action_noise_scale
                    action[:2] = action[:2] + noise  # Add noise to movement actions only
                elif self.exploration_enabled:
                    # For numpy arrays
                    noise = np.random.randn(2) * self.action_noise_scale
                    action[:2] = action[:2] + noise
            
            # Blend with matched pattern if found (emulate human movement)
            # Use higher weight when just switched back from manual mode
            if matched_pattern:
                # Check if we just switched to autonomous mode - use human actions more aggressively
                if hasattr(self.input_mux, 'just_switched_to_autonomous') and self.input_mux.just_switched_to_autonomous:
                    time_since_switch = time.time() - self.input_mux.switch_to_autonomous_time
                    if time_since_switch < self.input_mux.human_action_priority_duration:
                        # Very high weight (95%) for first 2 seconds after switching back - almost direct copy
                        pattern_weight = 0.95 - (time_since_switch / self.input_mux.human_action_priority_duration) * 0.25
                        pattern_weight = max(0.7, pattern_weight)  # Don't go below 70%
                        if self.frame_count % 20 == 0:  # More frequent logging
                            logger.info(f"COPYING human actions ({pattern_weight*100:.0f}% weight) | {time_since_switch:.1f}s since switch")
                    else:
                        # Reset flag after priority duration
                        self.input_mux.just_switched_to_autonomous = False
                        pattern_weight = 0.7
                else:
                    pattern_weight = 0.7  # Normal 70% pattern, 30% model
                
                # Blend movement - handle both tensor and numpy
                if isinstance(action, torch.Tensor):
                    action[0] = action[0] * (1 - pattern_weight) + matched_pattern['normalized'][0] * pattern_weight
                    action[1] = action[1] * (1 - pattern_weight) + matched_pattern['normalized'][1] * pattern_weight
                else:
                    action[0] = action[0] * (1 - pattern_weight) + matched_pattern['normalized'][0] * pattern_weight
                    action[1] = action[1] * (1 - pattern_weight) + matched_pattern['normalized'][1] * pattern_weight
                
                # Use pattern buttons more directly - copy button states more aggressively
                if matched_pattern.get('buttons'):
                    for i, button in enumerate(['left', 'right', 'space', 'alt']):
                        if len(action) > 2 + i:
                            if matched_pattern['buttons'].get(button, False):
                                # When just switched back, use almost direct copy (0.95), otherwise 0.9
                                copy_weight = 0.95 if (hasattr(self.input_mux, 'just_switched_to_autonomous') and 
                                                      self.input_mux.just_switched_to_autonomous) else 0.9
                                if isinstance(action, torch.Tensor):
                                    action[2 + i] = copy_weight
                                else:
                                    action[2 + i] = copy_weight  # Strong button copying
                            else:
                                # Reduce button probability if human didn't press it
                                if isinstance(action, torch.Tensor):
                                    action[2 + i] = action[2 + i] * 0.2
                                else:
                                    action[2 + i] = action[2 + i] * 0.2  # More aggressive reduction
                
                # Debug logging throttled to every 50 frames
                # #region debug log
                if False:  # Debug logging DISABLED for 60 FPS
                    if not hasattr(self, '_blend_log_frame_count'):
                        self._blend_log_frame_count = 0
                    self._blend_log_frame_count += 1
                    if self._blend_log_frame_count >= 50:
                        try:
                            with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                                log_entry = {
                                    'timestamp': time.time(),
                                    'location': 'actor.py:_get_bot_action:pattern_blend',
                                    'message': 'Blending human pattern into bot action',
                                    'data': {
                                        'pattern_dx': float(matched_pattern['normalized'][0]),
                                        'pattern_dy': float(matched_pattern['normalized'][1]),
                                        'final_dx': float(action[0].item() if hasattr(action[0], 'item') else action[0]),
                                        'final_dy': float(action[1].item() if hasattr(action[1], 'item') else action[1]),
                                        'pattern_weight': pattern_weight,
                                        'buttons': matched_pattern.get('buttons', {})
                                    },
                                    'sessionId': 'debug-session',
                                    'runId': 'run1',
                                    'hypothesisId': 'F'
                                }
                                f.write(json.dumps(log_entry) + '\n')
                        except:
                            pass
                    self._blend_log_frame_count = 0
                # #endregion
            
            # LLM strategy modulation removed
            
            # Track action diversity (throttled - only every 10 frames for better FPS)
            # Convert to numpy array safely
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().detach().numpy()
            elif isinstance(action, np.ndarray):
                action_np = action.copy()
            else:
                action_np = np.array(action)
            
            # Ensure action has correct shape (6 elements: dx, dy, left, right, space, alt)
            if len(action_np) < 6:
                action_np = np.pad(action_np, (0, max(0, 6 - len(action_np))), 'constant')
            elif len(action_np) > 6:
                action_np = action_np[:6]
            
            # Ensure actions are not all zeros - add small random movement if needed
            if abs(action_np[0]) < 0.01 and abs(action_np[1]) < 0.01 and not any(action_np[2:] > 0.5):
                # Action is too small - add small exploration movement
                action_np[0] = np.random.uniform(-0.1, 0.1)
                action_np[1] = np.random.uniform(-0.1, 0.1)
            
            if not hasattr(self, '_action_diversity_frame_count'):
                self._action_diversity_frame_count = 0
            self._action_diversity_frame_count += 1
            if self._action_diversity_frame_count >= 10:
                self.action_diversity_tracker.append(action_np.copy())
                self._action_diversity_frame_count = 0
            
            return action_np
            
        except Exception as e:
            logger.error(f"Action inference error: {e}", exc_info=True)
            # Return fallback action instead of None - ensure we always return valid array
            try:
                fallback = self._get_fallback_action(game_state)
                if isinstance(fallback, torch.Tensor):
                    return fallback.cpu().detach().numpy()
                elif isinstance(fallback, np.ndarray):
                    return fallback
                else:
                    return np.array(fallback)
            except Exception as fallback_error:
                logger.error(f"Fallback action generation also failed: {fallback_error}")
                # Last resort: return a minimal action to prevent None
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def _get_exploration_action(self, game_state: Dict) -> torch.Tensor:
        """Generate random exploration action"""
        import random
        if self.is_dqn:
            # For DQN: random discrete action
            if self.action_discretizer:
                random_action_id = random.randint(0, self.action_discretizer.get_num_actions() - 1)
                action_config = self.action_discretizer.get_action_config(random_action_id)
                dx = action_config.get('dx', 0) / 400.0
                dy = action_config.get('dy', 0) / 400.0
                keys = action_config.get('keys', {})
                return torch.tensor([
                    dx, dy,
                    1.0 if keys.get('left', False) else 0.0,
                    1.0 if keys.get('right', False) else 0.0,
                    1.0 if keys.get('space', False) else 0.0,
                    1.0 if keys.get('alt', False) else 0.0
                ], dtype=torch.float32)
        # Fallback: random continuous action
        dx = random.uniform(-0.5, 0.5)
        dy = random.uniform(-0.5, 0.5)
        return torch.tensor([dx, dy, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    def _store_movement_pattern(self, pattern: List[Dict], game_state: Dict, frame: np.ndarray):
        """Store a movement pattern for later emulation"""
        if len(pattern) < 5:
            return
        
        # Extract key features from pattern
        pattern_summary = {
            'movements': pattern,
            'game_state_context': {
                'health': game_state.get('health', 100),
                'enemy_health': game_state.get('enemy_health', 100),
                'threat_level': game_state.get('threat_level', 'unknown')
            },
            'pattern_type': self._classify_pattern(pattern),
            'timestamp': time.time()
        }
        
        # Store pattern (keep last 100 patterns)
        self.movement_patterns.append(pattern_summary)
        if len(self.movement_patterns) > 100:
            self.movement_patterns.pop(0)
    
    def _classify_pattern(self, pattern: List[Dict]) -> str:
        """Classify movement pattern type (swing, block, dodge, etc.)"""
        if not pattern:
            return "unknown"
        
        # Analyze pattern characteristics
        total_magnitude = sum(m.get('magnitude', 0) for m in pattern)
        avg_magnitude = total_magnitude / len(pattern) if pattern else 0
        
        # Check button usage
        left_held = any(m.get('buttons', {}).get('left', False) for m in pattern)
        right_held = any(m.get('buttons', {}).get('right', False) for m in pattern)
        
        # Classify based on characteristics
        if left_held and avg_magnitude > 50:
            return "swing_left"
        elif right_held and avg_magnitude > 50:
            return "swing_right"
        elif avg_magnitude < 10:
            return "block"
        elif avg_magnitude > 100:
            return "dodge"
        else:
            return "movement"
    
    def _get_fallback_action(self, game_state: Dict) -> torch.Tensor:
        """Generate a simple fallback action when model inference fails"""
        try:
            # Simple exploration action - small random movement
            import random
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            
            # No button presses by default
            buttons = torch.zeros(4)  # left, right, space, alt
            
            # Combine into action tensor
            action = torch.tensor([dx, dy, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            
            logger.debug(f"Generated fallback action: dx={dx:.3f}, dy={dy:.3f}")
            return action
        except Exception as e:
            logger.error(f"Fallback action generation failed: {e}")
            # Return zero action as last resort
            return torch.zeros(6, dtype=torch.float32)
    
    def _prepare_state_features(self, game_state: Dict) -> np.ndarray:
        """Prepare state features for model (handles None values safely)"""
        position = game_state.get("position", {})
        if not isinstance(position, dict):
            position = {}
        
        # Handle None values safely - use 0.5 as default (neutral) for unknown values
        health = game_state.get("health")
        if health is None:
            health = 50.0  # Neutral value for unknown
        health = max(0.0, min(100.0, health)) / 100.0
        
        stamina = game_state.get("stamina")
        if stamina is None:
            stamina = 50.0  # Neutral value for unknown
        stamina = max(0.0, min(100.0, stamina)) / 100.0
        
        enemy_health = game_state.get("enemy_health")
        if enemy_health is None:
            enemy_health = 50.0  # Neutral value for unknown
        enemy_health = max(0.0, min(100.0, enemy_health)) / 100.0
        
        is_dead = game_state.get("is_dead")
        is_dead_val = 1.0 if (is_dead is True) else 0.0
        
        enemy_dead = game_state.get("enemy_dead")
        enemy_dead_val = 1.0 if (enemy_dead is True) else 0.0
        
        pos_x = position.get("x")
        if pos_x is None:
            pos_x = 0.0
        pos_x = pos_x / 1000.0
        
        pos_y = position.get("y")
        if pos_y is None:
            pos_y = 0.0
        pos_y = pos_y / 1000.0
        
        return np.array([
            health,
            stamina,
            enemy_health,
            is_dead_val,
            enemy_dead_val,
            pos_x,
            pos_y
        ])
    
    # LLM strategy modulation removed
    
    def _inject_action(self, action):
        """Inject action into game - handles both numpy arrays and torch tensors"""
        try:
            # Convert to numpy if it's a torch tensor
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            elif not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # Ensure action has correct shape
            if len(action) < 6:
                logger.warning(f"Action has wrong shape: {action.shape}, padding with zeros")
                action = np.pad(action, (0, max(0, 6 - len(action))), 'constant')
            
            delta_x = float(np.clip(action[0], -1.0, 1.0))
            delta_y = float(np.clip(action[1], -1.0, 1.0))
            
            buttons = {
                "left": bool(action[2] > 0.5) if len(action) > 2 else False,
                "right": bool(action[3] > 0.5) if len(action) > 3 else False,
                "space": bool(action[4] > 0.5) if len(action) > 4 else False,
                "alt": bool(action[5] > 0.5) if len(action) > 5 else False
            }
            
            # Only inject if there's actual movement or button press
            if abs(delta_x) > 0.01 or abs(delta_y) > 0.01 or any(buttons.values()):
                self.input_mux.inject_action(delta_x, delta_y, buttons)
            else:
                logger.debug(f"Skipping injection - action too small: dx={delta_x:.4f}, dy={delta_y:.4f}")
        except Exception as e:
            logger.error(f"Error injecting action: {e} | Action type: {type(action)} | Action: {action}", exc_info=True)
    
    def _record_bot_action(self, frame: np.ndarray, game_state: Dict, action: np.ndarray, was_injected: bool = True):
        """Record bot action for training - always record to ensure data collection"""
        try:
            # Update next_state for previous transition if it exists
            # This is handled by storing the current frame as next_state for the previous experience
            if hasattr(self, 'last_recorded_state') and self.last_recorded_state is not None:
                # Find the last experience in buffer and update its next_state
                # This is a simplified approach - in practice, we update next_state when recording
                pass  # next_state is updated when recording the current experience
            
            # Calculate reward (will be used for YOLO self-learning too)
            # Pass frame for screen-based reward detection and action for energy efficiency
            reward = self._calculate_reward(game_state, frame=frame, action=action)
            
            # Enhanced reward shaping is already handled in _calculate_reward via reward_shaper
            # No additional shaping needed here
            
            # Calculate TD error estimate for priority
            td_error = abs(reward) if was_injected else 0.5 * abs(reward)  # Lower priority if not injected
            
            # Store current state/action for next_state update next frame
            # Store references for next_state (optimized - avoid unnecessary copies)
            self.last_recorded_state = frame if frame is not None else None  # Reference, not copy
            self.last_recorded_action = action.copy()  # Action array needs copy
            self.last_recorded_game_state = game_state  # Dict reference is fine
            
            # Store in replay buffer with enhanced metadata
            # next_state will be set to None initially, then updated next frame
            self.replay_buffer.push(
                state=frame,
                action=action,
                reward=reward,
                next_state=None,  # Will be filled next frame via _update_next_state
                done=game_state.get("is_dead", False) or game_state.get("enemy_dead", False),
                priority="HIGH" if was_injected else "MEDIUM",  # Higher priority for injected actions
                human_intervention=False,
                td_error=td_error,
                temporal_weight=1.0
            )
            
            # Track recording stats
            if not hasattr(self, '_bot_action_count'):
                self._bot_action_count = 0
            self._bot_action_count += 1
            
            # Track in performance monitor
            if self.perf_monitor:
                self.perf_monitor.current_episode["inference_count"] = \
                    self.perf_monitor.current_episode.get("inference_count", 0) + 1
                if was_injected:
                    self.perf_monitor.current_episode["injection_count"] = \
                        self.perf_monitor.current_episode.get("injection_count", 0) + 1
                
                # Track buffer size in real-time
                buffer_size = len(self.replay_buffer)
                self.perf_monitor.current_episode["replay_buffer_size"] = buffer_size
                
                # Log data collection progress every 50 actions
                if not hasattr(self, '_last_data_log') or self._bot_action_count % 50 == 0:
                    logger.info(f"[DATA COLLECTION] Buffer: {buffer_size} | "
                              f"Actions recorded: {self._bot_action_count} | "
                              f"Reward: {reward:.4f} | "
                              f"Injected: {was_injected}")
                    self._last_data_log = time.time()
        except Exception as e:
            logger.debug(f"Error recording bot action: {e}")
        
        # Note: YOLO self-learning is handled separately in the main loop
        # to ensure we have the detection-action-reward triplet
    
    def _record_human_action(self, frame: np.ndarray, game_state: Dict, human_action: tuple, td_error: float = None):
        """Record human action for DAgger learning - enhanced with full context and better priorities"""
        # Convert human action to numpy array
        if isinstance(human_action, tuple):
            # Extract normalized deltas and buttons
            delta_x = human_action[0] if len(human_action) > 0 else 0.0
            delta_y = human_action[1] if len(human_action) > 1 else 0.0
            buttons = human_action[2] if len(human_action) > 2 else {}
            
            # Convert buttons to array
            action_array = np.array([
                delta_x,
                delta_y,
                1.0 if buttons.get('left', False) else 0.0,
                1.0 if buttons.get('right', False) else 0.0,
                1.0 if buttons.get('space', False) else 0.0,
                1.0 if buttons.get('alt', False) else 0.0
            ])
        else:
            action_array = np.array(human_action)
        
        # Calculate reward (action not available for human actions in this context)
        reward = self._calculate_reward(game_state, action=None)
        
        # Calculate temporal weight (newer experiences slightly more important)
        temporal_weight = 1.0 + (0.1 * (1.0 - min(1.0, len(self.replay_buffer) / self.replay_buffer.capacity)))
        
        # Store with HIGH priority and enhanced metadata
        self.replay_buffer.push(
            state=frame,
            action=action_array,
            reward=reward,
            next_state=None,
            done=game_state.get("is_dead", False),
            priority="HIGH",
            human_intervention=True,
            human_action=action_array,  # Store for BC loss
            td_error=td_error if td_error is not None else abs(reward),
            temporal_weight=temporal_weight
        )
    
    def _calculate_reward(self, game_state: Dict, frame: np.ndarray = None, action: Optional[np.ndarray] = None) -> float:
        """
        Calculate reward signal using comprehensive multi-layered reward architecture.
        Combines ScrimBrain-style OCR/terminal detection with physics-based reward shaping.
        OPTIMIZED: Uses cached OCR/terminal results to avoid duplicate expensive operations
        
        Args:
            game_state: Game state from memory/visual detection
            frame: Current frame for screen-based reward detection (optional)
            action: Current action for energy efficiency calculation (optional)
        """
        # Debug logging throttled to every 100 frames for performance
        # #region debug log
        if False:  # Debug logging DISABLED for 60 FPS
            if not hasattr(self, '_reward_log_frame_count'):
                self._reward_log_frame_count = 0
            self._reward_log_frame_count += 1
            if self._reward_log_frame_count >= 100:
                try:
                    with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                        log_entry = {
                            'timestamp': time.time(),
                            'location': 'actor.py:_calculate_reward',
                            'message': 'Reward calculation started',
                            'data': {
                                'frame_count': self.frame_count,
                                'has_frame': frame is not None,
                                'has_action': action is not None,
                                'has_reward_shaper': self.reward_shaper is not None,
                                'game_state_keys': list(game_state.keys())[:10] if game_state else []
                            },
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A'
                        }
                        f.write(json.dumps(log_entry) + '\n')
                except Exception as e:
                    pass  # Don't fail on logging errors
            self._reward_log_frame_count = 0
        # #endregion
        
        reward = 0.0
        reward_breakdown = {}
        
        # Base rewards: OCR and terminal detection (ScrimBrain-style)
        base_reward = 0.0
        
        # 1. OCR-based score reward (use cached result from main loop to avoid duplicate OCR calls)
        if hasattr(self, '_cached_ocr_result'):
            ocr_result = self._cached_ocr_result
            if ocr_result.get('success') and ocr_result.get('reward', 0) > 0:
                base_reward += ocr_result['reward']
                reward_breakdown['ocr'] = ocr_result['reward']
        elif hasattr(self, 'ocr_reward_tracker') and self.ocr_reward_tracker and frame is not None:
            # Fallback: call OCR if cache not available (shouldn't happen, but safe)
            ocr_result = self.ocr_reward_tracker.update(frame)
            if ocr_result['success'] and ocr_result['reward'] > 0:
                base_reward += ocr_result['reward']
                reward_breakdown['ocr'] = ocr_result['reward']
        
        # 2. Screen-based reward detection (throttled - only every N frames)
        if frame is not None and hasattr(self, 'screen_reward_detector'):
            if not hasattr(self, '_screen_reward_frame_count'):
                self._screen_reward_frame_count = 0
            self._screen_reward_frame_count += 1
            
            # Only run expensive screen detection every 10 frames
            if self._screen_reward_frame_count >= 10:
                try:
                    screen_rewards = self.screen_reward_detector.detect_rewards(
                        frame, 
                        previous_frame=self.last_frame_for_reward
                    )
                    base_reward += screen_rewards['total_reward']
                    reward_breakdown['screen'] = screen_rewards['total_reward']
                    
                    # Track screen-based rewards
                    if self.perf_monitor:
                        self.perf_monitor.current_episode["screen_rewards"] = \
                            self.perf_monitor.current_episode.get("screen_rewards", 0) + screen_rewards['total_reward']
                        self.perf_monitor.current_episode["screen_detections"] = \
                            self.perf_monitor.current_episode.get("screen_detections", 0) + len(screen_rewards['detected_patterns'])
                    
                    # Store frame for next detection
                    self.last_frame_for_reward = frame  # Use reference, not copy
                except Exception as e:
                    logger.debug(f"Error in screen reward detection: {e}")
                self._screen_reward_frame_count = 0
        
        # 3. Terminal state detection (use cached result from main loop to avoid duplicate calls)
        terminal_penalty = 0.0
        if hasattr(self, 'terminal_state_detector') and self.terminal_state_detector:
            # Use cached terminal detection result from main loop if available
            if hasattr(self, '_cached_terminal_result'):
                terminal_result = self._cached_terminal_result
            elif frame is not None:
                # Fallback: call terminal detection if cache not available
                terminal_result = self.terminal_state_detector.detect_death_screen(frame)
            else:
                terminal_result = {'is_terminal': False}
            
            if terminal_result.get('is_terminal'):
                # Real death detected - apply penalty
                terminal_penalty = -20.0
                base_reward += terminal_penalty
                reward_breakdown['terminal'] = terminal_penalty
                if self.perf_monitor:
                    self.perf_monitor.current_episode["deaths"] = \
                        self.perf_monitor.current_episode.get("deaths", 0) + 1
        
        # 4. Comprehensive reward shaping (multi-layered architecture)
        # Throttle reward shaping to every 3 frames for better FPS
        if self.reward_shaper:
            if not hasattr(self, '_reward_shaping_frame_count'):
                self._reward_shaping_frame_count = 0
            self._reward_shaping_frame_count += 1
            
            if self._reward_shaping_frame_count >= 3:  # Only calculate every 3 frames
                try:
                    # Enhance game_state with movement pattern for reward shaper
                    enhanced_state = game_state.copy()
                    
                    # Add movement pattern if available (throttled - only every 10 frames)
                    if hasattr(self.input_mux, 'get_movement_pattern') and self.frame_count % 60 == 0:  # Every 60 frames for 60 FPS
                        try:
                            movement_pattern = self.input_mux.get_movement_pattern(lookback=5)
                            if movement_pattern:
                                enhanced_state['movement_pattern'] = movement_pattern
                        except Exception as e:
                            pass  # Skip on error for performance
                
                    # Add weapon/combat context
                    enhanced_state['weapon_mass'] = game_state.get('weapon_mass', 1.0)
                    enhanced_state['weapon_reach'] = game_state.get('weapon_reach', 1.0)
                    enhanced_state['enemy_weapon_reach'] = game_state.get('enemy_weapon_reach', 1.0)
                    enhanced_state['enemy_distance'] = game_state.get('enemy_distance')
                    enhanced_state['is_attacking'] = game_state.get('is_attacking', False)
                    
                    # Add human reference pattern for imitation learning (throttled - only every 10 frames)
                    if hasattr(self, 'human_recorder') and self.human_recorder and self.frame_count % 60 == 0:  # Every 60 frames for 60 FPS
                        try:
                            # Get recent human actions (last 5-10 actions)
                            recent_human_actions = self.human_recorder.get_expert_actions(count=10)
                            if recent_human_actions:
                                # Convert to format expected by imitation reward
                                reference_pattern = []
                                for action_record in recent_human_actions[-5:]:  # Last 5 actions
                                    normalized = action_record.get('normalized_delta', (0.0, 0.0))
                                    if isinstance(normalized, (list, tuple)) and len(normalized) >= 2:
                                        reference_pattern.append({
                                            'dx': float(normalized[0]),
                                            'dy': float(normalized[1]),
                                            'magnitude': action_record.get('mouse_magnitude', 0.0),
                                            'buttons': action_record.get('buttons', {})
                                        })
                                
                                if reference_pattern:
                                    enhanced_state['reference_pattern'] = reference_pattern
                        except Exception as e:
                            pass  # Skip on error for performance
                    
                    # Calculate comprehensive reward
                    shaped_reward, shaped_breakdown = self.reward_shaper.calculate_reward(
                        enhanced_state,
                        action=action,
                        previous_state=self.prev_state_for_reward
                    )
                    
                    reward += shaped_reward
                    reward_breakdown.update(shaped_breakdown)
                    
                    # Update previous state for PBRS
                    self.prev_state_for_reward = enhanced_state.copy()
                    
                    # Debug logging throttled to every 100 frames
                    # #region debug log
                    if False:  # Debug logging DISABLED for 60 FPS
                        if not hasattr(self, '_reward_shaping_log_frame_count'):
                            self._reward_shaping_log_frame_count = 0
                        self._reward_shaping_log_frame_count += 1
                        if self._reward_shaping_log_frame_count >= 100:
                            try:
                                with open(r'd:\AI Projects\ai butler 2\.cursor\debug.log', 'a') as f:
                                    log_entry = {
                                        'timestamp': time.time(),
                                        'location': 'actor.py:_calculate_reward:reward_shaping',
                                        'message': 'Reward shaping completed',
                                        'data': {
                                            'shaped_reward': float(shaped_reward),
                                            'breakdown_keys': list(shaped_breakdown.keys()),
                                            'total_reward': float(reward)
                                        },
                                        'sessionId': 'debug-session',
                                        'runId': 'run1',
                                        'hypothesisId': 'C'
                                    }
                                    f.write(json.dumps(log_entry) + '\n')
                            except:
                                pass
                        self._reward_shaping_log_frame_count = 0
                    # #endregion
                    
                    self._reward_shaping_frame_count = 0
                except Exception as e:
                    logger.debug(f"Error in reward shaping: {e}")
                    self._reward_shaping_frame_count = 0
        
        # 5. Memory-based rewards (ONLY if real data, not mock/placeholder)
        # These are now handled by reward shaper's health potential, but keep for backward compatibility
        if game_state.get('data_source') == 'memory' and not self.reward_shaper:
            # Only use memory data if it's real (not mock/placeholder)
            health = game_state.get('health')
            if health is not None and health != 100.0:  # Avoid using default/mock values
                # Survival reward (small positive for staying alive)
                if health > 0:
                    reward += health / 100.0 * 0.1
                else:
                    # Death penalty
                    reward -= 10.0
                
                # Enemy damage reward (only if we have real enemy health)
                enemy_health = game_state.get('enemy_health')
                if enemy_health is not None and enemy_health != 100.0:
                    damage_dealt = 100.0 - enemy_health
                    reward += damage_dealt * 0.5
                    
                    # Victory bonus (only if real data)
                    if game_state.get('enemy_dead', False):
                        reward += 20.0
                        if self.perf_monitor:
                            self.perf_monitor.current_episode["victories"] = \
                                self.perf_monitor.current_episode.get("victories", 0) + 1
        
        # Add base reward
        reward += base_reward
        
        # Log reward breakdown occasionally for debugging
        if self.frame_count % 300 == 0 and reward_breakdown:  # Reduced frequency for 60 FPS
            logger.debug(f"Reward breakdown: {reward_breakdown}")
        
        if self.perf_monitor:
            self.perf_monitor.record_reward(reward)
        
        return reward
    
    def _handle_restart(self):
        """Handle game restart"""
        logger.info("Handling game restart...")
        if self.perf_monitor:
            self.perf_monitor.end_episode("restart")
            self.perf_monitor.start_episode()
        self.episode_count += 1
        time.sleep(1.0)  # Wait for game to reset
    
    def _match_human_pattern(self, game_state: Dict) -> Optional[Dict]:
        """Match current game state to recent human movement patterns for imitation"""
        # When just switched back, prioritize more recent human actions
        action_count = 10  # Default
        if hasattr(self.input_mux, 'just_switched_to_autonomous') and self.input_mux.just_switched_to_autonomous:
            time_since_switch = time.time() - self.input_mux.switch_to_autonomous_time
            if time_since_switch < self.input_mux.human_action_priority_duration:
                action_count = 20  # Get more actions when just switched back
        
        # First try to get from human_recorder expert buffer
        if hasattr(self, 'human_recorder') and self.human_recorder:
            try:
                # Get recent human actions (get more for better pattern matching)
                recent_human_actions = self.human_recorder.get_expert_actions(count=action_count)
                if recent_human_actions and len(recent_human_actions) >= 1:
                    # When just switched back, use the most recent action more directly
                    if hasattr(self.input_mux, 'just_switched_to_autonomous') and self.input_mux.just_switched_to_autonomous:
                        # Use the most recent human action
                        latest_human = recent_human_actions[-1]
                    else:
                        # Normal mode: use latest but consider context
                        latest_human = recent_human_actions[-1]
                    
                    normalized = latest_human.get('normalized_delta', (0.0, 0.0))
                    
                    # Also check if we have mouse_delta and can calculate normalized from it
                    if normalized == (0.0, 0.0) or (normalized[0] == 0.0 and normalized[1] == 0.0):
                        mouse_delta = latest_human.get('mouse_delta', (0.0, 0.0))
                        if isinstance(mouse_delta, (list, tuple)) and len(mouse_delta) >= 2:
                            # Calculate normalized delta from pixel delta
                            sensitivity = getattr(self.input_mux, 'current_sensitivity', 400.0)
                            normalized = (
                                mouse_delta[0] / sensitivity if sensitivity > 0 else 0.0,
                                mouse_delta[1] / sensitivity if sensitivity > 0 else 0.0
                            )
                            # Clamp to [-1, 1]
                            normalized = (
                                max(-1.0, min(1.0, normalized[0])),
                                max(-1.0, min(1.0, normalized[1]))
                            )
                    
                    if isinstance(normalized, (list, tuple)) and len(normalized) >= 2:
                        buttons = latest_human.get('buttons', {})
                        if not isinstance(buttons, dict):
                            buttons = {}
                        
                        return {
                            'normalized': (float(normalized[0]), float(normalized[1])),
                            'magnitude': latest_human.get('mouse_magnitude', 0.0),
                            'buttons': buttons,
                            'confidence': 0.9  # High confidence for direct imitation
                        }
            except Exception as e:
                logger.debug(f"Error getting expert actions: {e}")
        
        # Fallback: Try to get from input_mux movement pattern buffer
        if hasattr(self, 'input_mux') and self.input_mux:
            try:
                movement_pattern = self.input_mux.get_movement_pattern(lookback=5)
                if movement_pattern and len(movement_pattern) > 0:
                    latest_pattern = movement_pattern[-1]
                    normalized = latest_pattern.get('normalized', (0.0, 0.0))
                    buttons = latest_pattern.get('buttons', {})
                    
                    if isinstance(normalized, (list, tuple)) and len(normalized) >= 2:
                        return {
                            'normalized': (float(normalized[0]), float(normalized[1])),
                            'magnitude': latest_pattern.get('magnitude', 0.0),
                            'buttons': buttons if isinstance(buttons, dict) else {},
                            'confidence': 0.7  # Medium confidence for pattern buffer
                        }
            except Exception as e:
                logger.debug(f"Error getting movement pattern: {e}")
        
        return None
    
    def _classify_pattern(self, pattern: List[Dict]) -> str:
        """Classify movement pattern type (swing, block, dodge, etc.)"""
        if not pattern:
            return "unknown"
        
        # Analyze pattern characteristics
        total_magnitude = sum(m.get('magnitude', 0) for m in pattern)
        avg_magnitude = total_magnitude / len(pattern) if pattern else 0
        
        # Check button usage
        left_held = any(m.get('buttons', {}).get('left', False) for m in pattern)
        right_held = any(m.get('buttons', {}).get('right', False) for m in pattern)
        
        # Classify based on characteristics
        if left_held and avg_magnitude > 50:
            return "swing_left"
        elif right_held and avg_magnitude > 50:
            return "swing_right"
        elif avg_magnitude < 10:
            return "block"
        elif avg_magnitude > 100:
            return "dodge"
        else:
            return "movement"
    
    def _get_fallback_action(self, game_state: Dict) -> torch.Tensor:
        """Generate a simple fallback action when model inference fails"""
        try:
            # Simple exploration action - small random movement
            import random
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            
            # No button presses by default
            buttons = torch.zeros(4)  # left, right, space, alt
            
            # Combine into action tensor
            action = torch.tensor([dx, dy, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            
            logger.debug(f"Generated fallback action: dx={dx:.3f}, dy={dy:.3f}")
            return action
        except Exception as e:
            logger.error(f"Fallback action generation failed: {e}")
            # Return zero action as last resort
            return torch.zeros(6, dtype=torch.float32)

