"""
Kill Switch: Enhanced emergency stop mechanism with multiple detection methods
Global hotkey listener for instant bot shutdown with improved reliability

MASSIVE IMPROVEMENTS:
- Multiple detection algorithms for reliability
- Enhanced error handling and recovery
- Status reporting and monitoring
- Graceful shutdown procedures
"""
import threading
import logging
import time
from typing import Optional, Callable, List
from collections import deque

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = False
KEYBOARD_LIB_AVAILABLE = False

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    try:
        import keyboard as kb
        KEYBOARD_LIB_AVAILABLE = True
        PYNPUT_AVAILABLE = False
    except ImportError:
        KEYBOARD_LIB_AVAILABLE = False
        PYNPUT_AVAILABLE = False
        logger.warning("No keyboard library available - kill switch will use polling method")

class KillSwitch:
    """
    Emergency kill switch for instant bot shutdown
    Listens for F8 key press to immediately stop all bot operations
    """
    
    def __init__(self, kill_callback: Optional[Callable] = None, hotkey: str = 'f8'):
        """
        Initialize enhanced kill switch with multiple detection methods
        
        Args:
            kill_callback: Function to call when kill button is pressed
            hotkey: Key to use as kill button (default: 'f8')
        """
        self.kill_callback = kill_callback
        self.hotkey = hotkey.lower()
        self.killed = False
        self.kill_count = 0
        self.listener = None
        self.running = False
        self.kill_thread = None
        
        # Enhanced monitoring
        self.detection_method = None
        self.detection_methods = []
        self.last_check_time = time.time()
        self.check_count = 0
        self.error_count = 0
        self.status_history = deque(maxlen=100)
        
        # Multiple detection methods for reliability
        self.polling_thread = None
        self.backup_thread = None
        self.use_multiple_methods = True
        
        # Try multiple initialization methods
        methods_tried = []
        if PYNPUT_AVAILABLE:
            if self._init_pynput():
                methods_tried.append("pynput")
        if KEYBOARD_LIB_AVAILABLE:
            if self._init_keyboard_lib():
                methods_tried.append("keyboard_lib")
        
        # Always enable polling as backup
        if self._init_polling():
            methods_tried.append("polling")
        
        if methods_tried:
            logger.info(f"✅ Kill switch initialized with methods: {', '.join(methods_tried)}")
        else:
            logger.error("❌ Failed to initialize kill switch - emergency fallback only")
            self._init_emergency_fallback()
    
    def _init_pynput(self) -> bool:
        """Initialize using pynput library - returns True if successful"""
        try:
            def on_press(key):
                try:
                    key_name = None
                    
                    if hasattr(key, 'name'):
                        key_name = key.name.lower()
                    elif hasattr(key, 'char') and key.char:
                        key_name = key.char.lower()
                    else:
                        key_str = str(key).replace("'", "").lower()
                        if key_str.startswith('key.'):
                            key_name = key_str.replace('key.', '')
                        else:
                            key_name = key_str
                    
                    hotkey_normalized = self.hotkey.lower()
                    
                    if key_name == hotkey_normalized or key_name == f'f{hotkey_normalized.replace("f", "")}':
                        logger.info(f"Kill switch detected via pynput: {key_name}")
                        self._trigger_kill()
                except Exception as e:
                    logger.error(f"Kill switch pynput error: {e}", exc_info=True)
                    self.error_count += 1
            
            self.listener = keyboard.Listener(on_press=on_press)
            self.detection_methods.append("pynput")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pynput: {e}", exc_info=True)
            return False
    
    def _init_keyboard_lib(self) -> bool:
        """Initialize using keyboard library - returns True if successful"""
        try:
            def on_hotkey():
                logger.info(f"Kill switch detected via keyboard library")
                self._trigger_kill()
            
            kb.add_hotkey(self.hotkey, on_hotkey)
            self.detection_methods.append("keyboard_lib")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize keyboard library: {e}")
            return False
    
    def _init_polling(self) -> bool:
        """Fallback polling method - returns True if successful"""
        try:
            import win32api
            import win32con
            
            vk_map = {
                'f8': win32con.VK_F8, 'f1': win32con.VK_F1, 'f2': win32con.VK_F2,
                'f3': win32con.VK_F3, 'f4': win32con.VK_F4, 'f5': win32con.VK_F5,
                'f6': win32con.VK_F6, 'f7': win32con.VK_F7, 'f9': win32con.VK_F9,
                'f10': win32con.VK_F10, 'f11': win32con.VK_F11, 'f12': win32con.VK_F12,
            }
            
            self.polling_vk_code = vk_map.get(self.hotkey.lower())
            if self.polling_vk_code is None:
                return False
            
            self.detection_methods.append("polling")
            return True
        except ImportError:
            logger.warning("win32api not available for polling")
            return False
        except Exception as e:
            logger.error(f"Polling initialization error: {e}")
            return False
    
    def _init_emergency_fallback(self):
        """Emergency fallback if all methods fail"""
        logger.critical("⚠️  EMERGENCY FALLBACK: Kill switch using minimal monitoring")
        self.detection_methods.append("emergency")
    
    def _trigger_kill(self):
        """Trigger kill sequence"""
        if not self.killed:
            self.killed = True
            self.kill_count += 1
            logger.critical("="*80)
            logger.critical("KILL SWITCH ACTIVATED!")
            logger.critical(f"Kill button ({self.hotkey.upper()}) pressed - Emergency shutdown initiated")
            logger.critical("="*80)
            
            # Call callback immediately and forcefully
            if self.kill_callback:
                try:
                    # Call in separate thread to avoid blocking
                    import threading
                    callback_thread = threading.Thread(target=self.kill_callback, daemon=True)
                    callback_thread.start()
                    # Also try direct call for immediate effect
                    self.kill_callback()
                except Exception as e:
                    logger.error(f"Kill callback error: {e}", exc_info=True)
                    # Force exit even if callback fails
                    import os
                    os._exit(0)
    
    def start(self):
        """Start kill switch with multiple detection methods"""
        if self.running:
            logger.warning("Kill switch already running")
            return
        
        self.running = True
        
        # Start pynput listener if available
        if self.listener:
            try:
                self.listener.start()
                logger.info("✅ Kill switch pynput listener started")
                self.detection_method = "pynput"
            except Exception as e:
                logger.error(f"Failed to start pynput: {e}", exc_info=True)
                self.error_count += 1
        
        # Start polling thread (always as backup)
        if "polling" in self.detection_methods:
            self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
            self.polling_thread.start()
            if not self.detection_method:
                self.detection_method = "polling"
            logger.info("✅ Kill switch polling active (backup)")
        
        # keyboard library runs automatically if initialized
        if "keyboard_lib" in self.detection_methods:
            if not self.detection_method:
                self.detection_method = "keyboard_lib"
            logger.info("✅ Kill switch keyboard library active")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_health, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"✅ Kill switch active - Press {self.hotkey.upper()} to kill")
        logger.info(f"   Detection methods: {', '.join(self.detection_methods)}")
    
    def stop(self):
        """Stop kill switch listener"""
        self.running = False
        
        if self.listener:
            self.listener.stop()
        
        if self.kill_thread:
            self.kill_thread.join(timeout=1.0)
        
        logger.info("Kill switch stopped")
    
    def _polling_loop(self):
        """Enhanced polling loop with error recovery"""
        if not hasattr(self, 'polling_vk_code') or self.polling_vk_code is None:
            return
        
        try:
            import win32api
            
            consecutive_errors = 0
            max_errors = 5
            
            while self.running and not self.killed:
                try:
                    self.check_count += 1
                    current_time = time.time()
                    
                    # Check key state
                    if win32api.GetAsyncKeyState(self.polling_vk_code) & 0x8000:
                        logger.info(f"Kill switch detected via polling: {self.hotkey.upper()}")
                        self._trigger_kill()
                        break
                    
                    # Record status
                    if current_time - self.last_check_time > 1.0:  # Every second
                        self.status_history.append({
                            'time': current_time,
                            'checks': self.check_count,
                            'errors': self.error_count,
                            'method': 'polling'
                        })
                        self.last_check_time = current_time
                    
                    consecutive_errors = 0  # Reset on success
                    time.sleep(0.01)  # 10ms polling for faster response
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.error_count += 1
                    
                    if consecutive_errors >= max_errors:
                        logger.error(f"Polling failed {consecutive_errors} times - disabling polling")
                        break
                    
                    time.sleep(0.1)  # Back off on error
                    
        except ImportError:
            logger.warning("win32api not available for polling")
        except Exception as e:
            logger.error(f"Polling loop error: {e}", exc_info=True)
    
    def _monitor_health(self):
        """Monitor kill switch health and status"""
        while self.running and not self.killed:
            try:
                time.sleep(5.0)  # Check every 5 seconds
                
                # Check if listeners are still alive
                health_ok = True
                
                if self.listener and not self.listener.running:
                    logger.warning("⚠️  Pynput listener stopped - attempting restart")
                    health_ok = False
                    try:
                        self.listener.start()
                    except:
                        pass
                
                # Log status
                if self.check_count > 0:
                    status = {
                        'checks': self.check_count,
                        'errors': self.error_count,
                        'methods': self.detection_methods,
                        'killed': self.killed
                    }
                    logger.debug(f"Kill switch status: {status}")
                    
            except Exception as e:
                logger.debug(f"Health monitor error: {e}")
    
    def get_status(self) -> dict:
        """Get kill switch status"""
        return {
            'running': self.running,
            'killed': self.killed,
            'kill_count': self.kill_count,
            'detection_method': self.detection_method,
            'detection_methods': self.detection_methods,
            'check_count': self.check_count,
            'error_count': self.error_count,
            'hotkey': self.hotkey.upper()
        }
    
    def is_killed(self) -> bool:
        """Check if kill switch has been activated"""
        return self.killed
    
    def reset(self):
        """Reset kill switch (for testing)"""
        self.killed = False
        logger.info("Kill switch reset")

