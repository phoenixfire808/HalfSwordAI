"""
Watchdog: Advanced game state monitoring with computer vision and recovery
Handles menu navigation, death detection, game restart, and health monitoring
"""
import time
import subprocess
import psutil
import numpy as np
import cv2
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from half_sword_ai.config import config
from half_sword_ai.perception.vision import MemoryReader, ScreenCapture

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

try:
    import pydirectinput
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    logger.warning("PyDirectInput not available")
    PYDIRECTINPUT_AVAILABLE = False

class Watchdog:
    """
    Monitors game state and automates menu navigation
    Handles death, victory, crashes, and restarts
    """
    
    def __init__(self, memory_reader: MemoryReader, screen_capture: ScreenCapture):
        self.memory_reader = memory_reader
        self.screen_capture = screen_capture
        self.game_process = None
        self.last_state_check = 0
        self.check_interval = 0.5  # Check every 500ms
        
        # Enhanced state tracking
        self.state_history = deque(maxlen=100)
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.last_successful_state = None
        
        # Computer vision for menu detection
        self.menu_templates = {}  # Can be populated with menu template images
        self.last_frame_hash = None
        self.stuck_frame_count = 0
        self.max_stuck_frames = 30  # ~1 second at 30fps
        
        # Health monitoring
        self.agent_health_checks = {
            'last_frame_time': time.time(),
            'frame_timeout': 5.0,  # seconds
            'consecutive_failures': 0,
            'max_failures': 10
        }
        
        # Recovery strategies
        self.recovery_strategies = [
            self._restart_match,
            self._restart_game_process,
            self._full_system_recovery
        ]
        
    def check_game_state(self) -> Dict[str, any]:
        """
        Enhanced game state checking with multiple detection methods
        
        Returns:
            Dictionary with state information and actions taken
        """
        current_time = time.time()
        if current_time - self.last_state_check < self.check_interval:
            return {"status": "ok", "action": "none"}
        
        self.last_state_check = current_time
        
        # Check agent health first
        agent_health = self._check_agent_health()
        if not agent_health['healthy']:
            logger.error(f"Agent health check failed: {agent_health['reason']}")
            return {"status": "agent_unhealthy", "action": "recovery", "reason": agent_health['reason']}
        
        # Get game state from memory
        game_state = self.memory_reader.get_state()
        
        # Store state history
        state_entry = {
            'timestamp': current_time,
            'state': game_state,
            'frame_hash': self._get_frame_hash()
        }
        self.state_history.append(state_entry)
        
        # Check for stuck state (same frame for too long)
        if self._check_stuck_state():
            logger.warning("Game appears stuck - same frame detected")
            self._handle_stuck_state()
            return {"status": "stuck", "action": "restart"}
        
        # Check for death
        if game_state.get("is_dead"):
            logger.info("Player death detected")
            time.sleep(config.DEATH_WAIT_TIME)
            self._handle_death()
            return {"status": "dead", "action": "restart"}
        
        # Check for victory
        if game_state.get("enemy_dead"):
            logger.info("Victory detected")
            self._handle_victory()
            return {"status": "victory", "action": "restart"}
        
        # Check for falling (handle None position)
        position = game_state.get("position")
        if position is not None and isinstance(position, dict):
            z_value = position.get("z")
            if z_value is not None and z_value < config.FALLING_Z_THRESHOLD:
                logger.warning("Player falling detected")
                self._handle_falling()
                return {"status": "falling", "action": "restart"}
        
        # Check for black screen bug
        if self._check_black_screen():
            logger.error("Black screen bug detected")
            self._handle_black_screen()
            return {"status": "bug", "action": "restart"}
        
        # Check for memory leak
        if self._check_memory_leak():
            logger.warning("Memory leak detected, restarting game")
            self._restart_game_process()
            return {"status": "memory_leak", "action": "restart"}
        
        # Check for menu state (using computer vision)
        menu_state = self._detect_menu_state()
        if menu_state['in_menu']:
            logger.info(f"Menu detected: {menu_state['menu_type']}")
            if menu_state['needs_navigation']:
                self._navigate_menu()
                return {"status": "menu", "action": "navigate", "menu_type": menu_state['menu_type']}
        
        # State is OK
        self.last_successful_state = state_entry
        self.recovery_attempts = 0
        return {"status": "ok", "action": "none"}
    
    def _check_agent_health(self) -> Dict:
        """Check if the agent itself is healthy"""
        current_time = time.time()
        
        # Check if frames are being captured
        if hasattr(self.screen_capture, 'last_frame_time'):
            time_since_frame = current_time - self.screen_capture.last_frame_time
            if time_since_frame > self.agent_health_checks['frame_timeout']:
                self.agent_health_checks['consecutive_failures'] += 1
                if self.agent_health_checks['consecutive_failures'] > self.agent_health_checks['max_failures']:
                    return {'healthy': False, 'reason': 'frame_capture_timeout'}
            else:
                self.agent_health_checks['consecutive_failures'] = 0
        
        # Check if memory reader is working
        if not self.memory_reader.is_process_running():
            return {'healthy': False, 'reason': 'game_process_not_running'}
        
        return {'healthy': True, 'reason': 'ok'}
    
    def _get_frame_hash(self) -> Optional[int]:
        """Get hash of current frame for stuck detection"""
        frame = self.screen_capture.get_latest_frame()
        if frame is None:
            return None
        return hash(frame.tobytes())
    
    def _check_stuck_state(self) -> bool:
        """Check if game is stuck (same frame for too long)"""
        current_hash = self._get_frame_hash()
        
        if current_hash is None:
            return False
        
        if current_hash == self.last_frame_hash:
            self.stuck_frame_count += 1
        else:
            self.stuck_frame_count = 0
            self.last_frame_hash = current_hash
        
        return self.stuck_frame_count > self.max_stuck_frames
    
    def _detect_menu_state(self) -> Dict:
        """Detect if game is in menu using computer vision"""
        frame = self.screen_capture.get_latest_frame()
        if frame is None:
            return {'in_menu': False, 'menu_type': None, 'needs_navigation': False}
        
        # Simple menu detection based on screen characteristics
        # In menus, screens are typically more static and have UI elements
        frame_variance = np.var(frame)
        
        # Low variance suggests menu (static screen)
        # High variance suggests gameplay (dynamic)
        in_menu = frame_variance < 100  # Threshold may need tuning
        
        menu_type = None
        if in_menu:
            # NOT IMPLEMENTED: Menu type detection would require:
            # - Template matching for different menu screens
            # - OCR to read menu text
            # - State machine to track menu navigation
            # For now, we can only detect that we're in a menu, not which menu
            menu_type = "unknown"
        
        return {
            'in_menu': in_menu,
            'menu_type': menu_type,
            'needs_navigation': in_menu and self.recovery_attempts < self.max_recovery_attempts
        }
    
    def _check_black_screen(self) -> bool:
        """Enhanced black screen detection with multiple checks"""
        frame = self.screen_capture.get_latest_frame()
        if frame is None:
            return False
        
        # Calculate percentage of black pixels
        black_pixels = np.sum(frame < 10)  # Threshold for "black"
        total_pixels = frame.size
        black_ratio = black_pixels / total_pixels
        
        # Also check if we're supposed to be in a match
        game_state = self.memory_reader.get_state()
        in_match = game_state.get("combat_state") == "fighting"
        
        # Additional check: very low variance also indicates black screen
        frame_variance = np.var(frame)
        low_variance = frame_variance < 5
        
        return (black_ratio > config.BLACK_SCREEN_THRESHOLD or low_variance) and in_match
    
    def _check_memory_leak(self) -> bool:
        """Check if game process is using too much memory"""
        if not self.memory_reader.is_process_running():
            return False
        
        try:
            process = psutil.Process(self.memory_reader.process.process_id)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb > (config.MEMORY_LEAK_THRESHOLD / (1024 * 1024))
        except:
            return False
    
    def _handle_death(self):
        """Handle player death - wait and restart"""
        logger.info("Handling death...")
        time.sleep(config.DEATH_WAIT_TIME)
        self._restart_match()
    
    def _handle_victory(self):
        """Handle victory - log and restart"""
        logger.info("Victory! Restarting match...")
        self._restart_match()
    
    def _handle_falling(self):
        """Handle falling into abyss - immediate restart"""
        logger.warning("Falling detected, immediate restart")
        self._restart_match()
    
    def _handle_stuck_state(self):
        """Handle stuck state - try recovery strategies"""
        logger.warning(f"Stuck state detected, attempting recovery (attempt {self.recovery_attempts + 1})")
        self.recovery_attempts += 1
        
        if self.recovery_attempts <= len(self.recovery_strategies):
            strategy = self.recovery_strategies[self.recovery_attempts - 1]
            try:
                strategy()
            except Exception as e:
                logger.error(f"Recovery strategy {self.recovery_attempts} failed: {e}")
                if self.recovery_attempts >= len(self.recovery_strategies):
                    logger.critical("All recovery strategies exhausted")
        else:
            logger.critical("Max recovery attempts reached, performing full restart")
            self._full_system_recovery()
    
    def _full_system_recovery(self):
        """Full system recovery - last resort"""
        logger.critical("Performing full system recovery")
        self._restart_game_process()
        time.sleep(5.0)  # Give system time to recover
        self.recovery_attempts = 0
        self.stuck_frame_count = 0
    
    def _handle_black_screen(self):
        """Handle black screen bug - force restart"""
        logger.error("Black screen bug, forcing game restart")
        self._restart_game_process()
    
    def _restart_match(self):
        """Restart the current match"""
        # Try memory injection first (if mod available)
        if self._try_memory_reset():
            return
        
        # Fallback to menu navigation
        self._navigate_menu()
    
    def _try_memory_reset(self) -> bool:
        """
        Try to reset via memory injection (HalfSwordTrainerMod)
        Returns True if successful
        """
        # In real implementation, would call mod's reset function
        # For now, return False to use menu navigation
        return False
    
    def _navigate_menu(self):
        """Navigate menu to restart match"""
        if not PYDIRECTINPUT_AVAILABLE:
            logger.warning("Cannot navigate menu - PyDirectInput not available")
            return
        
        try:
            logger.info("Navigating menu...")
            
            # Press ESC to open menu
            pydirectinput.press('esc')
            time.sleep(0.5)
            
            # Click "Menu" button (coordinates would need to be calibrated)
            # For now, use keyboard navigation
            pydirectinput.press('m')  # Assuming 'M' goes to menu
            time.sleep(1.0)
            
            # Navigate to Arena
            pydirectinput.press('a')  # Assuming 'A' selects Arena
            time.sleep(1.0)
            
            # Start match
            pydirectinput.press('enter')  # Start match
            time.sleep(2.0)
            
            logger.info("Menu navigation complete")
        except Exception as e:
            logger.error(f"Menu navigation error: {e}")
            # Fallback to process restart
            self._restart_game_process()
    
    def _restart_game_process(self):
        """Kill and restart game process"""
        logger.warning("Restarting game process...")
        
        try:
            # Find and kill process
            killed = False
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if config.GAME_PROCESS_NAME.lower() in proc.info['name'].lower():
                        proc.kill()
                        logger.info(f"Killed process {proc.info['name']}")
                        killed = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed:
                time.sleep(2.0)
            
            # Launch game if auto-launch is enabled
            if config.AUTO_LAUNCH_GAME:
                self._launch_game()
            else:
                logger.info("Game process killed. Manual restart required.")
            
        except Exception as e:
            logger.error(f"Process restart error: {e}")
    
    def _launch_game(self):
        """Launch game executable"""
        import os
        import subprocess
        import sys
        
        if not os.path.exists(config.GAME_EXECUTABLE_PATH):
            logger.warning(f"Game executable not found at: {config.GAME_EXECUTABLE_PATH}")
            logger.warning("Please update GAME_EXECUTABLE_PATH in config")
            return
        
        logger.info(f"Launching game: {config.GAME_EXECUTABLE_PATH}")
        try:
            if sys.platform == 'win32':
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
            
            logger.info("Game launch initiated. Waiting for process to start...")
            
            # Wait for process to appear
            start_time = time.time()
            while time.time() - start_time < config.GAME_LAUNCH_TIMEOUT:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if config.GAME_PROCESS_NAME.lower() in proc.info['name'].lower():
                            logger.info(f"Game process detected: {config.GAME_PROCESS_NAME}")
                            return
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                time.sleep(0.5)
            
            logger.warning(f"Game launch timeout - process not detected after {config.GAME_LAUNCH_TIMEOUT}s")
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")

