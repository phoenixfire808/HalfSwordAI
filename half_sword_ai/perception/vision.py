"""
Perception Layer: Screen Capture and Memory Reading
High-speed visual capture with DXCam and direct memory access with Pymem
"""
import numpy as np
import cv2
import time
import logging
from typing import Dict, Optional, Tuple
from collections import deque
from half_sword_ai.config import config

# Import YOLO detector
try:
    from half_sword_ai.perception.yolo_detector import YOLODetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError:
    YOLO_DETECTOR_AVAILABLE = False

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    logger.warning("DXCam not available, falling back to mss")
    DXCAM_AVAILABLE = False
    try:
        import mss
    except ImportError:
        logger.error("Neither DXCam nor mss available!")

try:
    import pymem
    import pymem.process
    PYMEM_AVAILABLE = True
except ImportError:
    logger.warning("Pymem not available - memory reading disabled")
    PYMEM_AVAILABLE = False

class ScreenCapture:
    """High-performance screen capture using DXCam or fallback"""
    
    def __init__(self, width: int = None, height: int = None, fps: int = None):
        self.width = width or config.CAPTURE_WIDTH
        self.height = height or config.CAPTURE_HEIGHT
        self.fps = fps or config.CAPTURE_FPS
        self.camera = None
        self.sct = None
        self.frame_stack = deque(maxlen=config.FRAME_STACK_SIZE)
        self.last_frame_time = 0
        self.game_window_info = None  # Store game window position/size
        
        if DXCAM_AVAILABLE:
            self._init_dxcam()
        else:
            self._init_mss()
        
        # Find game window for YOLO detection
        self._find_game_window()
    
    def _init_dxcam(self):
        """Initialize DXCam for high-speed capture"""
        try:
            self.camera = dxcam.create(device_idx=0, output_color="GRAY")
            self.camera.start(target_fps=self.fps)
            logger.info("DXCam initialized successfully")
        except Exception as e:
            logger.error(f"DXCam initialization failed: {e}")
            self._init_mss()
    
    def _init_mss(self):
        """Fallback to mss for screen capture"""
        try:
            self.sct = mss.mss()
            # Calculate center region
            monitor = self.sct.monitors[1]  # Primary monitor
            center_x = monitor["width"] // 2
            center_y = monitor["height"] // 2
            self.region = {
                "top": center_y - self.height // 2,
                "left": center_x - self.width // 2,
                "width": self.width,
                "height": self.height
            }
            logger.info("MSS fallback initialized")
        except Exception as e:
            logger.error(f"MSS initialization failed: {e}")
    
    def _find_game_window(self):
        """Find Half Sword game window for YOLO detection"""
        try:
            from half_sword_ai.utils.window_finder import WindowFinder
            
            # Try multiple possible window titles (order matters - most specific first)
            # Exclude browser/dashboard windows
            possible_titles = [
                "HalfSwordUE5",  # Most specific - actual game executable name (exact match)
                "Half Sword Demo",  # Steam version
                "HalfSword",  # Generic
            ]
            
            for title in possible_titles:
                window_info = WindowFinder.find_game_window(title)
                if window_info:
                    self.game_window_info = window_info
                    logger.info(f"Game window found: {window_info['title']} "
                              f"({window_info['width']}x{window_info['height']})")
                    return
            
            logger.warning("Game window not found - will use full screen for YOLO detection")
        except Exception as e:
            logger.warning(f"Could not find game window: {e}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame, non-blocking"""
        try:
            if self.camera:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    # Resize if needed
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    self.frame_stack.append(frame)
                    self.last_frame_time = time.time()
                    return frame
            elif self.sct:
                screenshot = self.sct.grab(self.region)
                frame = np.array(screenshot)
                # Convert BGRA to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                self.frame_stack.append(frame)
                self.last_frame_time = time.time()
                return frame
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
        
        return None
    
    def get_game_window_frame(self) -> Optional[np.ndarray]:
        """Get game window frame for YOLO detection (only captures Half Sword window)"""
        try:
            # Always use MSS for game window capture (DXCam doesn't support window-specific capture)
            # Initialize MSS if not already available (DXCam might be used for small region capture)
            if not self.sct:
                try:
                    import mss
                    self.sct = mss.mss()
                    logger.debug("Initialized MSS for game window capture")
                except Exception as e:
                    logger.error(f"Could not initialize MSS for game window capture: {e}")
                    return None
            
            # Use game window if found, otherwise fall back to full screen
            if self.game_window_info:
                from half_sword_ai.utils.window_finder import WindowFinder
                region = WindowFinder.get_window_region(self.game_window_info)
                screenshot = self.sct.grab(region)
                frame = np.array(screenshot)
                logger.debug(f"Captured game window: {region['width']}x{region['height']}, raw shape: {frame.shape}")
            else:
                # Fallback: try to find window again
                self._find_game_window()
                if self.game_window_info:
                    from half_sword_ai.utils.window_finder import WindowFinder
                    region = WindowFinder.get_window_region(self.game_window_info)
                    screenshot = self.sct.grab(region)
                    frame = np.array(screenshot)
                else:
                    # Last resort: full screen
                    logger.warning("Game window not found, using full screen")
                    monitor = self.sct.monitors[1]  # Primary monitor
                    full_region = {
                        "top": monitor["top"],
                        "left": monitor["left"],
                        "width": monitor["width"],
                        "height": monitor["height"]
                    }
                    screenshot = self.sct.grab(full_region)
                    frame = np.array(screenshot)
            
            # Convert BGRA to RGB (model expects RGB)
            # MSS returns BGRA format (4 channels: Blue, Green, Red, Alpha)
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:  # BGRA from MSS
                    # Remove alpha channel and convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                elif frame.shape[2] == 3:  # Already RGB or BGR
                    # Assume BGR and convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.shape[2] == 1:  # Grayscale (unlikely from MSS)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            logger.debug(f"Game window frame final shape: {frame.shape}")
            return frame
        except Exception as e:
            logger.error(f"Game window capture error: {e}", exc_info=True)
        
        return None
    
    def get_full_screen_frame(self) -> Optional[np.ndarray]:
        """Get full screen frame (deprecated - use get_game_window_frame instead)"""
        return self.get_game_window_frame()
    
    def get_frame_stack(self) -> Optional[np.ndarray]:
        """Get stacked frames for temporal information"""
        if len(self.frame_stack) < config.FRAME_STACK_SIZE:
            return None
        
        # Stack frames: (T, H, W) - optimized to avoid list conversion
        if len(self.frame_stack) == config.FRAME_STACK_SIZE:
            # Use numpy array directly if possible (faster)
            return np.array(list(self.frame_stack))
        return None
    
    def get_fps(self) -> float:
        """Calculate current capture FPS"""
        if self.last_frame_time == 0:
            return 0.0
        return 1.0 / max(0.001, time.time() - self.last_frame_time)
    
    def stop(self):
        """Stop capture"""
        if self.camera:
            self.camera.stop()
        if self.sct:
            self.sct.close()

class VisionProcessor:
    """
    Enhanced vision processing with motion detection, quality assessment, and YOLO
    Processes frames with temporal analysis and adaptive settings
    MASSIVE IMPROVEMENTS:
    - Better YOLO integration with pattern recognition
    - Enhanced motion detection with direction tracking
    - Visual pattern recognition for combat states
    - Temporal analysis for enemy behavior prediction
    """
    
    def __init__(self, screen_capture: 'ScreenCapture'):
        self.screen_capture = screen_capture
        self.yolo_detector = None
        self.last_detection_time = 0
        self.detection_interval = config.YOLO_DETECTION_INTERVAL
        
        # Motion detection (enhanced)
        self.previous_frame = None
        self.motion_history = deque(maxlen=100)  # Extended history
        self.motion_threshold = 5.0  # Pixels
        self.motion_directions = deque(maxlen=50)  # Track movement directions
        
        # Frame quality assessment
        self.quality_history = deque(maxlen=200)  # Extended
        self.min_quality_threshold = 0.3  # Minimum quality score (0-1)
        
        # Adaptive settings
        self.adaptive_detection_interval = config.YOLO_DETECTION_INTERVAL
        self.frame_skip_count = 0
        self.max_frame_skip = 2  # Skip frames if quality is too low
        
        # Visual pattern recognition
        self.visual_patterns = deque(maxlen=200)
        self.combat_state_history = deque(maxlen=100)
        self.enemy_behavior_patterns = {}
        
        # Enhanced detection tracking
        self.detection_patterns = deque(maxlen=100)
        self.threat_trend = deque(maxlen=50)
        
        # Lazy YOLO initialization - load on first use to speed up startup
        self._yolo_enabled = config.YOLO_ENABLED and YOLO_DETECTOR_AVAILABLE
        self._yolo_model_path = config.YOLO_MODEL_PATH if config.YOLO_USE_CUSTOM_MODEL else None
        self._yolo_confidence = config.YOLO_CONFIDENCE_THRESHOLD
        self.yolo_detector = None  # Will be initialized on first use
    
    def _ensure_yolo_initialized(self):
        """Lazy initialization of YOLO detector"""
        if self.yolo_detector is None and self._yolo_enabled:
            try:
                self.yolo_detector = YOLODetector(
                    model_path=self._yolo_model_path,
                    confidence_threshold=self._yolo_confidence
                )
                logger.info("YOLO detector initialized (lazy load)")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO detector: {e}")
                self.yolo_detector = None
                logger.info("YOLO detector initialized with enhanced pattern recognition")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO detector: {e}")
                self.yolo_detector = None
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess frame quality (0-1 scale)
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score between 0 and 1
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Check for black/empty frame
        mean_intensity = np.mean(frame)
        if mean_intensity < 10:
            return 0.0
        
        # Check variance (low variance = static/bad frame)
        variance = np.var(frame)
        variance_score = min(1.0, variance / 1000.0)  # Normalize
        
        # Check contrast (using standard deviation)
        std_dev = np.std(frame)
        contrast_score = min(1.0, std_dev / 50.0)  # Normalize
        
        # Check for blur (using Laplacian variance)
        try:
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            blur_score = min(1.0, np.var(laplacian) / 100.0)  # Normalize
        except:
            blur_score = 0.5  # Default if calculation fails
        
        # Combined quality score
        quality = (variance_score * 0.3 + contrast_score * 0.3 + blur_score * 0.4)
        
        self.quality_history.append(quality)
        return quality
    
    def detect_motion(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Enhanced motion detection with pattern recognition
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary with motion information and patterns
        """
        motion_info = {
            'has_motion': False,
            'motion_magnitude': 0.0,
            'motion_center': (0, 0),
            'motion_direction': (0.0, 0.0),
            'motion_velocity': 0.0,
            'motion_acceleration': 0.0,
            'motion_pattern': None
        }
        
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return motion_info
        
        # Calculate frame difference
        diff = cv2.absdiff(frame, self.previous_frame)
        
        # Enhanced thresholding with adaptive threshold
        mean_diff = np.mean(diff)
        adaptive_threshold = max(20, mean_diff * 1.5)
        _, thresh = cv2.threshold(diff, adaptive_threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate motion magnitude
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_ratio = motion_pixels / total_pixels
        motion_magnitude = motion_ratio * 100.0  # Percentage
        
        motion_info['motion_magnitude'] = motion_magnitude
        motion_info['has_motion'] = motion_magnitude > self.motion_threshold
        
        if motion_info['has_motion']:
            # Find motion center
            moments = cv2.moments(thresh)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                motion_info['motion_center'] = (cx, cy)
                
                # Calculate direction from frame center
                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                dx = cx - frame_center[0]
                dy = cy - frame_center[1]
                
                # Normalize
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    motion_info['motion_direction'] = (dx / magnitude, dy / magnitude)
                
                # Calculate velocity and acceleration from history
                if len(self.motion_history) > 0:
                    last_motion = self.motion_history[-1]
                    last_mag = last_motion.get('motion_magnitude', 0)
                    motion_info['motion_velocity'] = motion_magnitude - last_mag
                    
                    if len(self.motion_history) > 1:
                        prev_motion = self.motion_history[-2]
                        prev_mag = prev_motion.get('motion_magnitude', 0)
                        prev_velocity = last_mag - prev_mag
                        motion_info['motion_acceleration'] = motion_info['motion_velocity'] - prev_velocity
                
                # Detect motion pattern
                motion_info['motion_pattern'] = self._detect_motion_pattern()
        
        self.motion_history.append(motion_info)
        self.motion_directions.append(motion_info['motion_direction'])
        self.previous_frame = frame.copy()
        
        return motion_info
    
    def _detect_motion_pattern(self) -> Optional[str]:
        """Detect patterns in motion history"""
        if len(self.motion_history) < 5:
            return None
        
        # Analyze recent motion
        recent_motions = list(self.motion_history)[-10:]
        magnitudes = [m.get('motion_magnitude', 0) for m in recent_motions]
        avg_magnitude = np.mean(magnitudes)
        
        # Pattern: Increasing motion (enemy approaching/attacking)
        if len(magnitudes) >= 3:
            trend = np.polyfit(range(len(magnitudes)), magnitudes, 1)[0]
            if trend > 2.0:
                return 'increasing_motion'
            elif trend < -2.0:
                return 'decreasing_motion'
        
        # Pattern: High combat activity
        if avg_magnitude > 15:
            return 'high_combat'
        elif avg_magnitude < 3:
            return 'calm'
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Enhanced frame processing with quality assessment and motion detection
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with detections, motion, quality, and metadata
        """
        current_time = time.time()
        
        # Assess frame quality
        quality = self.assess_frame_quality(frame)
        
        # Skip low-quality frames
        if quality < self.min_quality_threshold:
            self.frame_skip_count += 1
            if self.frame_skip_count < self.max_frame_skip:
                return {
                    'objects': [],
                    'enemies': [],
                    'threat_level': 'unknown',
                    'timestamp': current_time,
                    'quality': quality,
                    'skipped': True
                }
        else:
            self.frame_skip_count = 0
        
        # Detect motion
        motion = self.detect_motion(frame)
        
        # Run detection at specified interval (adaptive)
        detection_result = {
            'objects': [],
            'enemies': [],
            'threat_level': 'unknown',
            'timestamp': current_time,
            'quality': quality,
            'motion': motion,
            'skipped': False
        }
        
        if (self.yolo_detector and 
            current_time - self.last_detection_time >= self.adaptive_detection_interval):
            
            # Get game window frame for YOLO detection (only captures Half Sword window)
            game_window_frame = self.screen_capture.get_game_window_frame()
            self._ensure_yolo_initialized()
            if self.yolo_detector:
                detections = self.yolo_detector.detect(frame, full_screen_frame=game_window_frame)
            else:
                detections = {}
            self.last_detection_time = current_time
            detection_result.update(detections)
            
            # Enhanced enemy processing with pattern recognition
            if detections.get('enemies'):
                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                self._ensure_yolo_initialized()
                if self.yolo_detector:
                    nearest_enemy = self.yolo_detector.get_nearest_enemy(detections, frame_center)
                else:
                    nearest_enemy = None
                if nearest_enemy:
                    self._ensure_yolo_initialized()
                    if self.yolo_detector:
                        direction = self.yolo_detector.get_enemy_direction(nearest_enemy, frame_center)
                    else:
                        direction = 'unknown'
                    detection_result['nearest_enemy'] = nearest_enemy
                    detection_result['enemy_direction'] = direction
                    
                    # Track enemy behavior patterns
                    enemy_pattern = self._analyze_enemy_behavior(nearest_enemy, detections)
                    if enemy_pattern:
                        detection_result['enemy_pattern'] = enemy_pattern
            
            self._ensure_yolo_initialized()
            detection_result['threat_level'] = self.yolo_detector.get_threat_level(detections) if self.yolo_detector else 'unknown'
            
            # Track threat trends
            threat_value = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(detection_result['threat_level'], 0)
            self.threat_trend.append(threat_value)
            
            # Store detection pattern
            self.detection_patterns.append({
                'enemy_count': len(detections.get('enemies', [])),
                'threat_level': detection_result['threat_level'],
                'timestamp': current_time
            })
            
            # Adaptive detection interval based on motion and threat
            if motion['has_motion'] or detection_result['threat_level'] in ['high', 'critical']:
                # More motion/threat = more frequent detection
                self.adaptive_detection_interval = max(0.05, config.YOLO_DETECTION_INTERVAL * 0.5)
            else:
                # Less motion/threat = less frequent detection
                self.adaptive_detection_interval = min(1.0, config.YOLO_DETECTION_INTERVAL * 1.5)
        
        return detection_result
    
    def _analyze_enemy_behavior(self, enemy: Dict, detections: Dict) -> Optional[str]:
        """Analyze enemy behavior patterns"""
        if len(self.detection_patterns) < 5:
            return None
        
        # Analyze enemy movement patterns
        recent_patterns = list(self.detection_patterns)[-10:]
        enemy_positions = []
        
        for pattern in recent_patterns:
            if pattern.get('enemy_count', 0) > 0:
                # Estimate position from detection (would need actual position data)
                enemy_positions.append(pattern)
        
        if len(enemy_positions) < 3:
            return None
        
        # Detect if enemy is approaching (increasing threat)
        threat_values = [p.get('threat_level', 'low') for p in recent_patterns]
        threat_nums = [{'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(t, 0) for t in threat_values]
        
        if len(threat_nums) >= 3:
            trend = np.polyfit(range(len(threat_nums)), threat_nums, 1)[0]
            if trend > 0.1:
                return 'enemy_approaching'
            elif trend < -0.1:
                return 'enemy_retreating'
        
        return None
    
    def get_visual_patterns(self) -> Dict:
        """Get recognized visual patterns"""
        return {
            'motion_patterns': [m.get('motion_pattern') for m in list(self.motion_history)[-20:] if m.get('motion_pattern')],
            'threat_trend': list(self.threat_trend),
            'recent_detections': list(self.detection_patterns)[-10:]
        }
    
    def get_detector_stats(self) -> Dict:
        """Get YOLO detector statistics with vision processor stats"""
        stats = {
            'model_loaded': False,
            'avg_quality': 0.0,
            'avg_motion': 0.0,
            'frames_skipped': self.frame_skip_count
        }
        
        if self.yolo_detector:
            self._ensure_yolo_initialized()
            detector_stats = self.yolo_detector.get_stats() if self.yolo_detector else {}
            stats.update(detector_stats)
            stats['model_loaded'] = True
        
        if len(self.quality_history) > 0:
            stats['avg_quality'] = float(np.mean(self.quality_history))
        
        if len(self.motion_history) > 0:
            motion_magnitudes = [m['motion_magnitude'] for m in self.motion_history]
            stats['avg_motion'] = float(np.mean(motion_magnitudes))
        
        return stats

class MemoryReader:
    """Direct memory access for game state reading"""
    
    def __init__(self, process_name: str = None):
        self.process_name = process_name or config.GAME_PROCESS_NAME
        # Try multiple possible process names (order matters - most specific first)
        self.possible_names = [
            config.GAME_PROCESS_NAME,
            "HalfSwordUE5-Win64-Shipping.exe",
            "HalfSwordUE5.exe",
            "HalfSword-Win64-Shipping.exe",
            "HalfSword",
            "HalfSwordUE5"
        ]
        self.process = None
        self.base_address = None
        self.health_offset = None
        self.stamina_offset = None
        self.enemy_health_offset = None
        self.position_offset = None  # Added for position reading
        self.last_scan_time = 0
        self.scan_interval = config.MEMORY_SCAN_INTERVAL
        
        if PYMEM_AVAILABLE:
            self._attach_to_process()
        else:
            logger.warning("Memory reading disabled - Pymem not available")
    
    def _attach_to_process(self):
        """Attach to game process - tries multiple possible process names"""
        # First, find which process name actually exists
        actual_process_name = self._find_running_process()
        if not actual_process_name:
            # Only log once per minute to reduce spam
            current_time = time.time()
            if not hasattr(self, '_last_process_warning') or current_time - self._last_process_warning > 60:
                logger.debug(f"Game process not found. Tried: {', '.join(self.possible_names)}")
                logger.debug("Listing running processes with 'Half' or 'Sword' in name...")
                self._list_similar_processes()
                self._last_process_warning = current_time
            self.process = None
            return
        
        # Update process name if we found a different one
        if actual_process_name != self.process_name:
            logger.info(f"Found game process with different name: {actual_process_name} (was looking for {self.process_name})")
            self.process_name = actual_process_name
        
        try:
            self.process = pymem.Pymem(self.process_name)
            logger.info(f"Successfully attached to game process: {self.process_name}")
            self._scan_for_pointers()
        except pymem.exception.ProcessNotFound:
            logger.warning(f"Process {self.process_name} not found by pymem (may need admin rights)")
            self.process = None
        except Exception as e:
            logger.error(f"Failed to attach to process: {e}")
            self.process = None
    
    def _find_running_process(self) -> Optional[str]:
        """Find which of the possible process names is actually running"""
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name']
                    if proc_name:
                        # Check against all possible names
                        for possible_name in self.possible_names:
                            if possible_name.lower() in proc_name.lower():
                                logger.info(f"Found matching process: {proc_name} (matches {possible_name})")
                                return proc_name
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None
        except Exception as e:
            logger.debug(f"Error finding process: {e}")
            return None
    
    def _list_similar_processes(self):
        """List processes with similar names for debugging"""
        try:
            import psutil
            similar = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'] or ''
                    if any(keyword.lower() in proc_name.lower() for keyword in ['half', 'sword', 'ue5']):
                        similar.append(f"  - {proc_name} (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if similar:
                logger.debug("Similar processes found:")
                for proc_info in similar[:10]:  # Limit to 10
                    logger.debug(proc_info)
            else:
                logger.debug("No similar processes found")
        except Exception as e:
            logger.debug(f"Error listing processes: {e}")
    
    def _scan_for_pointers(self):
        """
        Scan for memory pointers using AOB (Array of Bytes) pattern scanning
        
        This method implements:
        1. AOB pattern scanning
        2. Pointer chain traversal
        3. Dynamic address resolution
        
        Patterns must be configured in config.MEMORY_PATTERNS after finding them
        via Cheat Engine or UE4SS.
        """
        if not self.process:
            self.health_offset = None
            self.stamina_offset = None
            self.enemy_health_offset = None
            self.position_offset = None
            return
        
        try:
            self.base_address = self.process.base_address
            logger.debug(f"Base address: {hex(self.base_address)}")
        except Exception as e:
            logger.debug(f"Could not determine base address: {e}")
            self.base_address = None
        
        # Initialize offsets
        self.health_offset = None
        self.stamina_offset = None
        self.enemy_health_offset = None
        self.position_offset = None
        
        # Scan for patterns if configured
        patterns = config.MEMORY_PATTERNS
        
        # Scan for player health
        if patterns.get("player_health"):
            health_addr = self._scan_pattern(patterns["player_health"])
            if health_addr:
                self.health_offset = self._resolve_pointer_chain(health_addr, patterns.get("pointer_chain"))
                logger.info(f"[OK] Found health offset: {hex(self.health_offset) if self.health_offset else 'None'}")
        
        # Scan for player stamina
        if patterns.get("player_stamina"):
            stamina_addr = self._scan_pattern(patterns["player_stamina"])
            if stamina_addr:
                self.stamina_offset = self._resolve_pointer_chain(stamina_addr, patterns.get("pointer_chain"))
                logger.info(f"[OK] Found stamina offset: {hex(self.stamina_offset) if self.stamina_offset else 'None'}")
        
        # Scan for player position
        if patterns.get("player_position"):
            pos_addr = self._scan_pattern(patterns["player_position"])
            if pos_addr:
                self.position_offset = self._resolve_pointer_chain(pos_addr, patterns.get("pointer_chain"))
                logger.info(f"[OK] Found position offset: {hex(self.position_offset) if self.position_offset else 'None'}")
        
        # Scan for enemy health
        if patterns.get("enemy_health"):
            enemy_addr = self._scan_pattern(patterns["enemy_health"])
            if enemy_addr:
                self.enemy_health_offset = self._resolve_pointer_chain(enemy_addr, patterns.get("pointer_chain"))
                logger.info(f"[OK] Found enemy health offset: {hex(self.enemy_health_offset) if self.enemy_health_offset else 'None'}")
        
        # If no patterns configured, try generic UE5 scanning
        if not any(patterns.values()):
            logger.debug("No memory patterns configured - attempting generic UE5 scanning")
            self._scan_generic_ue5_patterns()
        
        # Log results
        if self.health_offset or self.stamina_offset or self.position_offset or self.enemy_health_offset:
            logger.info("[OK] Memory scanning completed - some offsets found")
        else:
            logger.info("Memory scanning framework ready - no patterns configured yet. "
                       "Configure MEMORY_PATTERNS in config after finding patterns via Cheat Engine/UE4SS. "
                       "Use half_sword_ai/tools/find_memory_patterns.py to help find patterns.")
    
    def _scan_pattern(self, pattern: str) -> Optional[int]:
        """
        Scan memory for AOB (Array of Bytes) pattern
        
        Args:
            pattern: AOB pattern string like "48 8B 05 ?? ?? ?? ?? 48 85 C0"
                    Use ?? for wildcard bytes
        
        Returns:
            First matching address or None
        """
        if not self.process or not pattern:
            return None
        
        try:
            # Parse pattern
            pattern_bytes, mask = self._parse_aob_pattern(pattern)
            if not pattern_bytes:
                return None
            
            # Get memory regions
            regions = self._get_memory_regions()
            
            # Scan each region
            for region_start, region_size in regions:
                try:
                    # Read memory region
                    memory_data = pymem.memory.read_bytes(
                        self.process.process_handle,
                        region_start,
                        min(region_size, config.MEMORY_SCAN_REGION_SIZE)
                    )
                    
                    # Search for pattern
                    for i in range(len(memory_data) - len(pattern_bytes) + 1):
                        if self._match_pattern(memory_data[i:i+len(pattern_bytes)], pattern_bytes, mask):
                            address = region_start + i
                            logger.debug(f"Pattern match found at: {hex(address)}")
                            return address
                            
                except Exception as e:
                    logger.debug(f"Error scanning region {hex(region_start)}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Pattern scan error: {e}")
            return None
    
    def _parse_aob_pattern(self, pattern: str) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Parse AOB pattern string into bytes and mask
        
        Args:
            pattern: Pattern like "48 8B 05 ?? ?? ?? ?? 48"
        
        Returns:
            Tuple of (pattern_bytes, mask_bytes) or (None, None) on error
        """
        try:
            parts = pattern.strip().split()
            pattern_bytes = []
            mask_bytes = []
            
            for part in parts:
                if part == "??":
                    pattern_bytes.append(0x00)  # Dummy value
                    mask_bytes.append(0x00)  # 0 = wildcard
                else:
                    byte_val = int(part, 16)
                    pattern_bytes.append(byte_val)
                    mask_bytes.append(0xFF)  # 0xFF = must match
            
            return bytes(pattern_bytes), bytes(mask_bytes)
        except Exception as e:
            logger.debug(f"Pattern parse error: {e}")
            return None, None
    
    def _match_pattern(self, data: bytes, pattern: bytes, mask: bytes) -> bool:
        """Check if data matches pattern with mask"""
        if len(data) != len(pattern) or len(pattern) != len(mask):
            return False
        
        for i in range(len(pattern)):
            if mask[i] == 0xFF:  # Must match
                if data[i] != pattern[i]:
                    return False
            # If mask[i] == 0x00, it's a wildcard - always matches
        
        return True
    
    def _get_memory_regions(self) -> list:
        """Get memory regions to scan"""
        if not self.process:
            return []
        
        try:
            regions = []
            # Get main module regions
            modules = list(self.process.list_modules())
            
            for module in modules:
                try:
                    module_info = pymem.process.module_from_name(self.process.process_handle, module.name)
                    if module_info:
                        regions.append((module_info.lpBaseOfDll, module_info.SizeOfImage))
                except:
                    continue
            
            # Add custom scan region if configured
            if config.MEMORY_SCAN_REGION_START:
                regions.append((
                    config.MEMORY_SCAN_REGION_START,
                    config.MEMORY_SCAN_REGION_SIZE
                ))
            
            return regions if regions else [(self.base_address or 0x400000, 0x10000000)]
            
        except Exception as e:
            logger.debug(f"Error getting memory regions: {e}")
            # Fallback to base address region
            return [(self.base_address or 0x400000, 0x10000000)]
    
    def _resolve_pointer_chain(self, base_address: int, chain: Optional[list]) -> Optional[int]:
        """
        Resolve pointer chain to get final address
        
        Args:
            base_address: Starting address
            chain: List of offsets [0x1234, 0x5678, ...]
        
        Returns:
            Final address or None on error
        """
        if not chain or not self.process:
            return base_address
        
        try:
            current_addr = base_address
            
            for offset in chain:
                # Read pointer at current address + offset
                try:
                    # Read 8 bytes (64-bit pointer) or 4 bytes (32-bit pointer)
                    # Try 64-bit first (most modern games)
                    try:
                        pointer_bytes = pymem.memory.read_bytes(self.process.process_handle, current_addr + offset, 8)
                        pointer = int.from_bytes(pointer_bytes, byteorder='little', signed=False)
                    except:
                        # Fallback to 32-bit
                        pointer_bytes = pymem.memory.read_bytes(self.process.process_handle, current_addr + offset, 4)
                        pointer = int.from_bytes(pointer_bytes, byteorder='little', signed=False)
                    
                    if pointer == 0:
                        return None
                    current_addr = pointer
                except Exception as e:
                    logger.debug(f"Pointer chain resolution error at {hex(current_addr + offset)}: {e}")
                    return None
            
            return current_addr
            
        except Exception as e:
            logger.debug(f"Pointer chain error: {e}")
            return None
    
    def _scan_generic_ue5_patterns(self):
        """
        Attempt to find common UE5 patterns without specific AOB patterns
        
        This is a fallback that tries to find common UE5 structures
        """
        if not self.process:
            return
        
        try:
            # Common UE5 patterns (these are generic and may not work)
            # These would need to be replaced with actual patterns found via Cheat Engine
            
            # Try to find GWorld or GEngine patterns (common UE5 globals)
            # This is a placeholder - real patterns need to be found
            
            logger.debug("Generic UE5 scanning attempted (patterns need to be configured)")
            
        except Exception as e:
            logger.debug(f"Generic UE5 scan error: {e}")
    
    def get_state(self) -> Dict[str, any]:
        """
        Read current game state using real data sources
        Uses visual detection when memory reading unavailable
        NO MOCK DATA - all values are real or explicitly unknown
        """
        # Re-scan for pointers periodically
        current_time = time.time()
        if current_time - self.last_scan_time > config.POINTER_UPDATE_INTERVAL:
            self._scan_for_pointers()
            self.last_scan_time = current_time
        
        # Try to read from memory if available
        # Memory scanning is now implemented - patterns need to be configured in config.MEMORY_PATTERNS
        if self.process and self.health_offset:
            try:
                health = self._read_float(self.health_offset)
                stamina = self._read_float(self.stamina_offset) if self.stamina_offset else None
                enemy_health = self._read_float(self.enemy_health_offset) if self.enemy_health_offset else None
                
                # Only return memory data if we actually got valid values (not None)
                if health is not None:
                    return {
                        "health": max(0.0, min(100.0, health)),
                        "stamina": max(0.0, min(100.0, stamina)) if stamina is not None else None,
                        "enemy_health": max(0.0, min(100.0, enemy_health)) if enemy_health is not None else None,
                        "is_dead": health <= 0.0,
                        "enemy_dead": enemy_health <= 0.0 if enemy_health is not None else None,
                        "position": self._read_position(),
                        "combat_state": "fighting" if health > 0 and (enemy_health is None or enemy_health > 0) else "ended",
                        "data_source": "memory",
                        # Ensure all expected keys exist
                        "detections": {},
                        "threat_level": "unknown",
                        "motion": None,
                        "motion_magnitude": 0.0,
                        "has_motion": False,
                        "motion_direction": (0.0, 0.0),
                        "frame_quality": None,
                        "score": None,
                        "score_delta": 0,
                        "ocr_reward": 0.0,
                        "yolo_guidance": None,
                        "terminal_detection": None,
                        "nearest_enemy": None,
                        "enemy_direction": (0, 0)
                    }
            except Exception as e:
                logger.debug(f"Memory read error: {e}, falling back to visual detection")
        
        # Fall back to visual detection if memory scanning not configured or failed
        # NO MOCK DATA - uses real visual analysis or returns None/unknown
        return self._get_visual_state()
    
    def _read_float(self, address: int) -> Optional[float]:
        """
        Read float from memory address
        Returns None on error (not 0.0) to indicate failure
        """
        if not self.process or address is None:
            return None
        try:
            return pymem.memory.read_float(self.process.process_handle, address)
        except Exception as e:
            logger.debug(f"Memory read failed at {hex(address) if address else 'None'}: {e}")
            return None  # Return None, not 0.0, to indicate failure
    
    def _read_position(self) -> Dict[str, Optional[float]]:
        """
        Read player position from memory if available
        
        Returns:
            Dict with x, y, z coordinates or None if not available
        """
        if not self.process or not hasattr(self, 'position_offset') or self.position_offset is None:
            return {"x": None, "y": None, "z": None}
        
        try:
            # Read 3 floats (X, Y, Z) from position offset
            x = pymem.memory.read_float(self.process.process_handle, self.position_offset)
            y = pymem.memory.read_float(self.process.process_handle, self.position_offset + 4)
            z = pymem.memory.read_float(self.process.process_handle, self.position_offset + 8)
            
            return {"x": float(x), "y": float(y), "z": float(z)}
            
        except Exception as e:
            logger.debug(f"Position read error: {e}")
            return {"x": None, "y": None, "z": None}
    
    def _get_visual_state(self) -> Dict[str, any]:
        """
        Get game state using visual detection methods
        NO MOCK DATA - uses real visual analysis or returns None/unknown
        Enriched with all expected keys for consistency
        """
        # Use terminal state detector if available
        is_dead = None
        combat_state = "unknown"
        
        # Check if we can determine state from visual cues
        # This would integrate with terminal_state_detector
        # For now, return unknown states rather than fake data
        
        # Return comprehensive state with all expected keys
        return {
            "health": None,  # Unknown - not fake 100
            "stamina": None,  # Unknown - not fake 100
            "enemy_health": None,  # Unknown - not fake 100
            "is_dead": is_dead,  # None = unknown, not False
            "enemy_dead": None,  # Unknown
            "position": {"x": None, "y": None, "z": None},  # Unknown, not fake zeros
            "combat_state": combat_state,  # "unknown" not "fighting"
            "data_source": "visual",
            "note": "Using visual detection - some values may be unknown",
            # Additional visual state keys
            "detections": {},
            "threat_level": "unknown",
            "motion": None,
            "motion_magnitude": 0.0,
            "has_motion": False,
            "motion_direction": (0.0, 0.0),
            "frame_quality": None,
            "score": None,
            "score_delta": 0,
            "ocr_reward": 0.0,
            "yolo_guidance": None,
            "terminal_detection": None,
            "nearest_enemy": None,
            "enemy_direction": (0, 0)
        }
    
    def is_process_running(self) -> bool:
        """Check if game process is still running"""
        if not self.process:
            # Try to reattach
            self._attach_to_process()
            if not self.process and config.AUTO_LAUNCH_GAME:
                # Process not found, try to launch game
                self._launch_game_if_needed()
                # Try attaching again after launch attempt
                time.sleep(2.0)  # Give game time to start
                self._attach_to_process()
            return self.process is not None
        try:
            # Try to read a byte to check if process is alive
            pymem.memory.read_bytes(self.process.process_handle, self.process.base_address, 1)
            return True
        except:
            # Process might have closed, try to reattach
            self._attach_to_process()
            if not self.process and config.AUTO_LAUNCH_GAME:
                self._launch_game_if_needed()
                time.sleep(2.0)
                self._attach_to_process()
            return self.process is not None
    
    def _launch_game_if_needed(self):
        """Launch game if not running and auto-launch is enabled"""
        if not config.AUTO_LAUNCH_GAME:
            return
        
        # Check if process is already running
        if self._check_process_exists():
            return
        
        # Check if game executable exists
        import os
        if not os.path.exists(config.GAME_EXECUTABLE_PATH):
            logger.warning(f"Game executable not found at: {config.GAME_EXECUTABLE_PATH}")
            logger.warning("Please update GAME_EXECUTABLE_PATH in config")
            return
        
        logger.info(f"Launching game: {config.GAME_EXECUTABLE_PATH}")
        try:
            import subprocess
            import sys
            
            # Launch game in background
            if sys.platform == 'win32':
                # Use CREATE_NEW_CONSOLE to launch in separate window
                subprocess.Popen(
                    [config.GAME_EXECUTABLE_PATH],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=os.path.dirname(config.GAME_EXECUTABLE_PATH)
                )
            else:
                subprocess.Popen(
                    [config.GAME_EXECUTABLE_PATH],
                    cwd=os.path.dirname(config.GAME_EXECUTABLE_PATH)
                )
            
            logger.info(f"Game launch initiated. Waiting for process to start...")
            
            # Wait for process to appear
            start_time = time.time()
            while time.time() - start_time < config.GAME_LAUNCH_TIMEOUT:
                if self._check_process_exists():
                    logger.info(f"Game process detected: {config.GAME_PROCESS_NAME}")
                    return
                time.sleep(0.5)
            
            logger.warning(f"Game launch timeout - process not detected after {config.GAME_LAUNCH_TIMEOUT}s")
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
    
    def _check_process_exists(self) -> bool:
        """Check if game process exists in process list - checks all possible names"""
        actual_name = self._find_running_process()
        if actual_name:
            # Update our process name if we found it
            if actual_name != self.process_name:
                self.process_name = actual_name
            return True
        return False
    
    def get_process_status(self) -> Dict[str, any]:
        """Get detailed process status"""
        is_running = self.is_process_running()
        return {
            "process_name": self.process_name,
            "is_running": is_running,
            "is_attached": self.process is not None,
            "has_memory_access": PYMEM_AVAILABLE and self.process is not None
        }

