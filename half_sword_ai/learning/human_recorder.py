"""
Human Action Recorder: Advanced recording with pattern extraction and compression
Records every mouse movement, click, and keyboard press for imitation learning

MASSIVE IMPROVEMENTS:
- Pattern extraction and recognition
- Action compression for efficient storage
- Real-time analysis and statistics
- Enhanced context capture
- Adaptive recording quality
"""
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os
import zlib
import pickle
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

class HumanActionRecorder:
    """
    Records all human actions during gameplay
    Captures mouse movements, clicks, keyboard presses with full context
    """
    
    def __init__(self, save_path: str = None):
        self.save_path = save_path or config.DATA_SAVE_PATH
        os.makedirs(self.save_path, exist_ok=True)
        
        # Recording state
        self.recording = False
        self.current_session = []
        self.session_start_time = None
        
        # Enhanced action tracking
        self.last_mouse_pos = None
        self.last_mouse_time = None
        self.mouse_history = deque(maxlen=500)  # Extended history
        self.velocity_history = deque(maxlen=100)
        self.acceleration_history = deque(maxlen=50)
        
        # Button state tracking (enhanced) - All Half Sword controls
        # Mouse buttons
        self.button_states = {
            'left': False,      # LMB - Left hand control
            'right': False,     # RMB - Right hand control / Half-swording mode
            'space': False,     # SPACE - Jump/dodge
            'alt': False,       # ALT - Thrust/half-swording
            # Movement keys (WASD)
            'w': False,         # W - Forward movement
            'a': False,         # A - Left strafe
            's': False,         # S - Backward movement
            'd': False,         # D - Right strafe
            # Grab mechanics
            'q': False,         # Q - Left hand grab
            'e': False,         # E - Right hand grab
            # Other controls
            'g': False,         # G - Surrender
            'shift': False,     # SHIFT - Sprint (if used)
            'ctrl': False       # CTRL - Crouch (if used)
        }
        self.last_button_states = self.button_states.copy()
        self.button_hold_start_times = {key: None for key in self.button_states}
        self.button_sequences = deque(maxlen=100)  # Track button press sequences
        
        # Movement vector tracking (WASD combination)
        self.movement_vector = {'x': 0.0, 'y': 0.0}  # Normalized movement direction
        
        # Pattern recognition
        self.patterns_detected = []
        self.pattern_buffer = deque(maxlen=1000)
        self.pattern_signatures = {}  # Store recognized patterns
        
        # Statistics (enhanced)
        self.total_actions_recorded = 0
        self.total_sessions = 0
        self.recording_start_time = None
        self.recording_quality = "high"  # "high", "medium", "low"
        
        # Action compression
        self.compression_enabled = True
        self.compression_threshold = 1000  # Compress if session > 1000 actions
        
        # Expert demonstration buffer (extended)
        self.expert_buffer = deque(maxlen=50000)  # Last 50k actions
        
        # Real-time analysis
        self.analysis_enabled = True
        self.action_frequency = deque(maxlen=100)
        self.movement_trends = deque(maxlen=50)
        
        # Quality metrics
        self.quality_metrics = {
            'avg_action_interval': 0.0,
            'movement_efficiency': 0.0,
            'button_timing_accuracy': 0.0,
            'pattern_consistency': 0.0
        }
        
    def start_recording(self):
        """Start recording human actions"""
        if self.recording:
            logger.warning("Already recording")
            return
        
        self.recording = True
        self.current_session = []
        self.session_start_time = time.time()
        self.recording_start_time = time.time()
        self.last_mouse_pos = None
        self.last_mouse_time = None
        
        logger.info("🎥 Human action recording started - All your actions will be recorded!")
    
    def stop_recording(self):
        """Stop recording and save session"""
        if not self.recording:
            return
        
        self.recording = False
        session_duration = time.time() - self.session_start_time
        
        if len(self.current_session) > 0:
            self._save_session()
            logger.info(f"Recording stopped. Recorded {len(self.current_session)} actions over {session_duration:.1f}s")
        else:
            logger.info("Recording stopped (no actions recorded)")
        
        self.current_session = []
    
    def record_action(self, frame: np.ndarray, game_state: Dict, 
                     mouse_delta: Tuple[float, float] = None,
                     buttons: Dict[str, bool] = None,
                     timestamp: float = None,
                     velocity: float = None,
                     acceleration: float = None,
                     pattern_context: Dict = None) -> bool:
        """
        Record a human action with full context
        
        Args:
            frame: Current game frame
            game_state: Current game state
            mouse_delta: Mouse movement (dx, dy) in pixels
            buttons: Button states
            timestamp: Action timestamp (or current time if None)
            
        Returns:
            True if action was recorded
        """
        if not self.recording:
            return False
        
        timestamp = timestamp or time.time()
        relative_time = timestamp - self.session_start_time
        
        # Get current mouse position if available
        current_mouse_pos = None
        if PYAUTOGUI_AVAILABLE:
            try:
                current_mouse_pos = pyautogui.position()
            except:
                pass
        
        # Calculate mouse delta if not provided
        if mouse_delta is None:
            if current_mouse_pos and self.last_mouse_pos:
                mouse_delta = (
                    current_mouse_pos.x - self.last_mouse_pos.x,
                    current_mouse_pos.y - self.last_mouse_pos.y
                )
            else:
                mouse_delta = (0.0, 0.0)
        
        # Normalize mouse delta
        mouse_magnitude = np.sqrt(mouse_delta[0]**2 + mouse_delta[1]**2)
        if mouse_magnitude > 0:
            normalized_delta = (
                mouse_delta[0] / config.MOUSE_SENSITIVITY,
                mouse_delta[1] / config.MOUSE_SENSITIVITY
            )
            # Clamp to [-1, 1]
            normalized_delta = (
                max(-1.0, min(1.0, normalized_delta[0])),
                max(-1.0, min(1.0, normalized_delta[1]))
            )
        else:
            normalized_delta = (0.0, 0.0)
        
        # Ensure buttons dict includes all Half Sword controls
        if buttons is None:
            buttons = self.button_states.copy()
        else:
            # Merge with default states to ensure all keys exist
            full_buttons = self.button_states.copy()
            full_buttons.update(buttons)
            buttons = full_buttons
        
        # Calculate movement vector from WASD
        movement_x = 0.0
        movement_y = 0.0
        if buttons.get('d', False):
            movement_x += 1.0
        if buttons.get('a', False):
            movement_x -= 1.0
        if buttons.get('w', False):
            movement_y += 1.0
        if buttons.get('s', False):
            movement_y -= 1.0
        
        # Normalize diagonal movement
        if movement_x != 0.0 and movement_y != 0.0:
            length = (movement_x**2 + movement_y**2)**0.5
            movement_x /= length
            movement_y /= length
        
        self.movement_vector = {'x': movement_x, 'y': movement_y}
        
        if mouse_magnitude == 0:
            normalized_delta = (0.0, 0.0)
        
        # Detect button changes
        if buttons is None:
            buttons = self._get_current_button_states()
        
        button_changes = self._detect_button_changes(buttons)
        
        # Calculate button hold durations for all controls
        button_hold_durations = {}
        if hasattr(self, 'button_hold_start_times'):
            current_time = timestamp
            for button in self.button_states.keys():
                if buttons.get(button, False) and self.button_hold_start_times.get(button):
                    hold_start = self.button_hold_start_times[button]
                    button_hold_durations[button] = current_time - hold_start
                else:
                    button_hold_durations[button] = 0.0
        
        # Enhanced action record with advanced context - includes all Half Sword controls
        action_record = {
            'timestamp': timestamp,
            'relative_time': relative_time,
            # Mouse movement
            'mouse_delta': mouse_delta,
            'normalized_delta': normalized_delta,
            'mouse_magnitude': mouse_magnitude,
            # Movement keys (WASD)
            'movement_vector': self.movement_vector.copy(),
            'movement_x': movement_x,
            'movement_y': movement_y,
            # All button states
            'buttons': buttons.copy(),
            'button_changes': button_changes,
            'button_hold_durations': button_hold_durations.copy(),
            # Motion tracking
            'velocity': velocity if velocity is not None else (self.velocity_history[-1] if self.velocity_history else 0),
            'acceleration': acceleration if acceleration is not None else (self.acceleration_history[-1] if self.acceleration_history else 0),
            # Game state (convert numpy types to Python native types for JSON)
            'game_state': self._make_json_serializable(game_state.copy()),
            'frame_shape': frame.shape if frame is not None else None,
            'frame_hash': hash(frame.tobytes()) if frame is not None else None,
            'pattern_context': pattern_context if pattern_context else {},
            'action_id': self.total_actions_recorded
        }
        
        # Track velocity and acceleration
        if velocity is not None:
            self.velocity_history.append(velocity)
        if acceleration is not None:
            self.acceleration_history.append(acceleration)
        
        # Pattern analysis in real-time
        if self.analysis_enabled and len(self.current_session) >= 5:
            recent_actions = self.current_session[-5:]
            pattern_info = self._analyze_pattern(recent_actions)
            action_record['pattern_info'] = pattern_info
        
        # Update quality metrics
        self._update_quality_metrics(action_record)
        
        # Store in session
        self.current_session.append(action_record)
        
        # Store in expert buffer for immediate learning
        self.expert_buffer.append(action_record)
        
        # Update tracking
        self.last_mouse_pos = current_mouse_pos
        self.last_mouse_time = timestamp
        self.last_button_states = buttons.copy()
        self.total_actions_recorded += 1
        
        # Store in mouse history
        if mouse_magnitude > config.NOISE_THRESHOLD:
            self.mouse_history.append({
                'delta': mouse_delta,
                'normalized': normalized_delta,
                'magnitude': mouse_magnitude,
                'timestamp': timestamp
            })
        
        return True
    
    def _get_current_button_states(self) -> Dict[str, bool]:
        """Get current button states"""
        # In real implementation, would read from input system
        # For now, return last known states
        return self.button_states.copy()
    
    def _detect_button_changes(self, current_buttons: Dict[str, bool]) -> Dict[str, str]:
        """Detect button press/release events for all Half Sword controls"""
        changes = {}
        # Check all controls
        for key in self.button_states.keys():
            current = current_buttons.get(key, False)
            last = self.last_button_states.get(key, False)
            
            if current != last:
                changes[key] = 'press' if current else 'release'
                # Update hold start time
                if current:
                    self.button_hold_start_times[key] = time.time()
                else:
                    self.button_hold_start_times[key] = None
        
        return changes
    
    def _analyze_pattern(self, recent_actions: List[Dict]) -> Dict:
        """Analyze pattern in recent actions"""
        if len(recent_actions) < 3:
            return {}
        
        # Extract movement magnitudes
        magnitudes = [a.get('mouse_magnitude', 0) for a in recent_actions]
        
        # Check for consistent patterns
        pattern_type = "random"
        if np.std(magnitudes) < np.mean(magnitudes) * 0.3:
            pattern_type = "consistent"
        elif magnitudes[-1] > np.mean(magnitudes) * 2:
            pattern_type = "acceleration"
        elif magnitudes[-1] < np.mean(magnitudes) * 0.5:
            pattern_type = "deceleration"
        
        # Button pattern
        button_pattern = "none"
        for action in recent_actions:
            changes = action.get('button_changes', {})
            if changes:
                button_pattern = "active"
                break
        
        return {
            'pattern_type': pattern_type,
            'button_pattern': button_pattern,
            'avg_magnitude': np.mean(magnitudes),
            'consistency': 1.0 - (np.std(magnitudes) / (np.mean(magnitudes) + 1e-6))
        }
    
    def _update_quality_metrics(self, action_record: Dict):
        """Update quality metrics in real-time"""
        if len(self.current_session) < 2:
            return
        
        # Calculate average action interval
        if len(self.current_session) >= 2:
            intervals = []
            for i in range(1, len(self.current_session)):
                dt = self.current_session[i]['timestamp'] - self.current_session[i-1]['timestamp']
                intervals.append(dt)
            self.quality_metrics['avg_action_interval'] = np.mean(intervals) if intervals else 0
        
        # Movement efficiency (ratio of actual movement to potential)
        if action_record.get('mouse_magnitude', 0) > 0:
            efficiency = min(1.0, action_record['mouse_magnitude'] / 100.0)
            self.movement_trends.append(efficiency)
            if len(self.movement_trends) >= 10:
                self.quality_metrics['movement_efficiency'] = np.mean(list(self.movement_trends)[-10:])
        
        # Pattern consistency
        if len(self.current_session) >= 10:
            recent_mags = [a.get('mouse_magnitude', 0) for a in self.current_session[-10:]]
            if np.mean(recent_mags) > 0:
                consistency = 1.0 - (np.std(recent_mags) / (np.mean(recent_mags) + 1e-6))
                self.quality_metrics['pattern_consistency'] = consistency
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-JSON types to JSON-serializable types"""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Try to convert to string as last resort
            return str(obj)
    
    def _save_session(self):
        """Save current recording session with compression if needed"""
        if len(self.current_session) == 0:
            return
        
        self.total_sessions += 1
        session_filename = f"human_session_{int(self.session_start_time)}_{self.total_sessions}"
        
        # Prepare session data (ensure all data is JSON-serializable)
        session_data = {
            'session_id': self.total_sessions,
            'start_time': self.session_start_time,
            'duration': time.time() - self.session_start_time,
            'total_actions': len(self.current_session),
            'quality_metrics': self._make_json_serializable(self.quality_metrics.copy()),
            'recording_quality': self.recording_quality,
            'actions': self._make_json_serializable(self.current_session),
            'statistics': self._make_json_serializable(self.get_action_statistics())
        }
        
        # Compress if session is large
        if self.compression_enabled and len(self.current_session) > self.compression_threshold:
            session_filename += ".pkl.gz"
            session_path = os.path.join(self.save_path, session_filename)
            try:
                compressed = zlib.compress(pickle.dumps(session_data))
                with open(session_path, 'wb') as f:
                    f.write(compressed)
                logger.info(f"Session saved (compressed) to {session_path} ({len(compressed)/1024:.1f}KB)")
            except Exception as e:
                logger.error(f"Failed to save compressed session: {e}")
        else:
            session_filename += ".json"
            session_path = os.path.join(self.save_path, session_filename)
            try:
                with open(session_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Session saved to {session_path}")
            except Exception as e:
                logger.error(f"Failed to save session: {e}", exc_info=True)
    
    def get_expert_actions(self, count: int = None) -> List[Dict]:
        """Get recent expert actions for training"""
        if count is None:
            return list(self.expert_buffer)
        return list(self.expert_buffer)[-count:]
    
    def get_action_statistics(self) -> Dict:
        """Get comprehensive statistics about recorded actions"""
        base_stats = {
            'recording': self.recording,
            'total_actions': self.total_actions_recorded,
            'total_sessions': self.total_sessions,
            'current_session_actions': len(self.current_session),
            'expert_buffer_size': len(self.expert_buffer),
            'recording_quality': self.recording_quality
        }
        
        if len(self.current_session) == 0:
            base_stats['current_session_actions'] = 0
            base_stats['current_session_duration'] = 0
            return base_stats
        
        # Enhanced analysis
        mouse_movements = [a['mouse_magnitude'] for a in self.current_session if a['mouse_magnitude'] > 0]
        button_presses = sum(len(a['button_changes']) for a in self.current_session)
        velocities = [a.get('velocity', 0) for a in self.current_session if a.get('velocity', 0) > 0]
        
        # Pattern statistics
        patterns_detected = [a.get('pattern_info', {}) for a in self.current_session if a.get('pattern_info')]
        
        enhanced_stats = {
            'current_session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'mouse_movements': len(mouse_movements),
            'avg_mouse_magnitude': np.mean(mouse_movements) if mouse_movements else 0,
            'max_mouse_magnitude': np.max(mouse_movements) if mouse_movements else 0,
            'button_presses': button_presses,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'patterns_detected': len(patterns_detected),
            'quality_metrics': self.quality_metrics.copy(),
            'action_rate': len(self.current_session) / max(1, time.time() - self.session_start_time) if self.session_start_time else 0
        }
        
        base_stats.update(enhanced_stats)
        return base_stats
    
    def extract_patterns(self, min_pattern_length: int = 5) -> List[Dict]:
        """Extract recurring patterns from recorded actions"""
        if len(self.current_session) < min_pattern_length * 2:
            return []
        
        patterns = []
        # Simple pattern extraction: find repeated sequences
        for i in range(len(self.current_session) - min_pattern_length * 2):
            candidate = self.current_session[i:i+min_pattern_length]
            
            # Check if this pattern repeats
            for j in range(i + min_pattern_length, len(self.current_session) - min_pattern_length):
                match = self.current_session[j:j+min_pattern_length]
                if self._patterns_match(candidate, match):
                    patterns.append({
                        'pattern': candidate,
                        'occurrences': [i, j],
                        'similarity': self._calculate_pattern_similarity(candidate, match)
                    })
        
        return patterns
    
    def _patterns_match(self, pattern1: List[Dict], pattern2: List[Dict], threshold: float = 0.8) -> bool:
        """Check if two patterns match"""
        if len(pattern1) != len(pattern2):
            return False
        
        similarities = []
        for a1, a2 in zip(pattern1, pattern2):
            # Compare movement magnitudes
            mag1 = a1.get('mouse_magnitude', 0)
            mag2 = a2.get('mouse_magnitude', 0)
            if max(mag1, mag2) > 0:
                mag_sim = 1.0 - abs(mag1 - mag2) / max(mag1, mag2)
                similarities.append(mag_sim)
        
        if not similarities:
            return False
        
        avg_similarity = np.mean(similarities)
        return avg_similarity >= threshold
    
    def _calculate_pattern_similarity(self, pattern1: List[Dict], pattern2: List[Dict]) -> float:
        """Calculate similarity score between two patterns (0-1)"""
        if not self._patterns_match(pattern1, pattern2, threshold=0.0):
            return 0.0
        
        similarities = []
        for a1, a2 in zip(pattern1, pattern2):
            mag1 = a1.get('mouse_magnitude', 0)
            mag2 = a2.get('mouse_magnitude', 0)
            if max(mag1, mag2) > 0:
                sim = 1.0 - abs(mag1 - mag2) / max(mag1, mag2)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def clear_expert_buffer(self):
        """Clear expert buffer (use with caution)"""
        self.expert_buffer.clear()
        logger.warning("Expert buffer cleared")

