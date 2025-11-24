"""
DirectInput Interface using ctypes - ScrimBrain Integration
Implements low-level Windows API input simulation for DirectInput compatibility
Based on ScrimBrain architecture for Half Sword physics-based combat

CRITICAL: Uses MOUSEEVENTF_MOVE (relative) not absolute coordinates
"""
import ctypes
import ctypes.wintypes
import time
import logging
from typing import Dict, Tuple, Optional
from enum import IntEnum

logger = logging.getLogger(__name__)

# Windows API Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000  # DO NOT USE for Half Sword - must be relative

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# DirectInput Scan Codes (from guide)
SCANCODE_W = 0x11
SCANCODE_A = 0x1E
SCANCODE_S = 0x1F
SCANCODE_D = 0x20
SCANCODE_Q = 0x10
SCANCODE_E = 0x12
SCANCODE_SPACE = 0x39
SCANCODE_ALT = 0x38
SCANCODE_CTRL = 0x1D
SCANCODE_SHIFT = 0x2A

# Windows API Structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD)
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("ii", INPUT_UNION)
    ]

class DirectInput:
    """
    Low-level DirectInput interface using ctypes
    Provides relative mouse movement and scan code keyboard input
    Compatible with Half Sword's physics engine requirements
    """
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.last_mouse_time = 0
        self.mouse_cooldown = 0.001  # 1ms minimum between mouse events
        
        # Track key states (including mouse buttons)
        self.key_states = {
            'left': False, 'right': False, 'middle': False,  # Mouse buttons
            'w': False, 'a': False, 's': False, 'd': False,
            'q': False, 'e': False, 'space': False, 'alt': False,
            'ctrl': False, 'shift': False
        }
        
        logger.info("DirectInput initialized - using ctypes SendInput")
    
    def move_mouse_relative(self, dx: int, dy: int) -> bool:
        """
        Move mouse relative to current position
        CRITICAL: Uses MOUSEEVENTF_MOVE (relative) not absolute
        
        Args:
            dx: Relative X movement in pixels
            dy: Relative Y movement in pixels
            
        Returns:
            True if successful
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_mouse_time < self.mouse_cooldown:
            time.sleep(self.mouse_cooldown - (current_time - self.last_mouse_time))
        
        # Create mouse input structure
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(
            dx=dx,
            dy=dy,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE,  # Relative movement - CRITICAL
            time=0,
            dwExtraInfo=ctypes.pointer(extra)
        )
        
        # Create input structure
        x = INPUT(
            type=INPUT_MOUSE,
            ii=ii_
        )
        
        # Send input
        result = self.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self.last_mouse_time = time.time()
        
        if result != 1:
            logger.warning(f"SendInput failed for mouse movement: {result}")
            return False
        
        return True
    
    def press_key(self, scancode: int) -> bool:
        """
        Press a key using scan code
        
        Args:
            scancode: DirectInput scan code
            
        Returns:
            True if successful
        """
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.ki = KEYBDINPUT(
            wVk=0,
            wScan=scancode,
            dwFlags=KEYEVENTF_SCANCODE,
            time=0,
            dwExtraInfo=ctypes.pointer(extra)
        )
        
        x = INPUT(
            type=INPUT_KEYBOARD,
            ii=ii_
        )
        
        result = self.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        return result == 1
    
    def release_key(self, scancode: int) -> bool:
        """
        Release a key using scan code
        
        Args:
            scancode: DirectInput scan code
            
        Returns:
            True if successful
        """
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.ki = KEYBDINPUT(
            wVk=0,
            wScan=scancode,
            dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
            time=0,
            dwExtraInfo=ctypes.pointer(extra)
        )
        
        x = INPUT(
            type=INPUT_KEYBOARD,
            ii=ii_
        )
        
        result = self.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        return result == 1
    
    def press_mouse_button(self, button: str = 'left') -> bool:
        """
        Press (hold down) mouse button - for attack swings
        
        Args:
            button: 'left', 'right', or 'middle'
            
        Returns:
            True if successful
        """
        if button == 'left':
            down_flag = MOUSEEVENTF_LEFTDOWN
        elif button == 'right':
            down_flag = MOUSEEVENTF_RIGHTDOWN
        elif button == 'middle':
            down_flag = MOUSEEVENTF_MIDDLEDOWN
        else:
            return False
        
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(
            dx=0, dy=0, mouseData=0,
            dwFlags=down_flag,
            time=0,
            dwExtraInfo=ctypes.pointer(extra)
        )
        x = INPUT(type=INPUT_MOUSE, ii=ii_)
        result = self.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        
        # Track button state
        self.key_states[button] = True
        
        return result == 1
    
    def release_mouse_button(self, button: str = 'left') -> bool:
        """
        Release mouse button - after attack swing completes
        
        Args:
            button: 'left', 'right', or 'middle'
            
        Returns:
            True if successful
        """
        if button == 'left':
            up_flag = MOUSEEVENTF_LEFTUP
        elif button == 'right':
            up_flag = MOUSEEVENTF_RIGHTUP
        elif button == 'middle':
            up_flag = MOUSEEVENTF_MIDDLEUP
        else:
            return False
        
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(
            dx=0, dy=0, mouseData=0,
            dwFlags=up_flag,
            time=0,
            dwExtraInfo=ctypes.pointer(extra)
        )
        x = INPUT(type=INPUT_MOUSE, ii=ii_)
        result = self.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        
        # Track button state
        self.key_states[button] = False
        
        return result == 1
    
    def click_mouse_button(self, button: str = 'left') -> bool:
        """
        Click mouse button (press and release quickly)
        For single clicks, not attack swings
        
        Args:
            button: 'left', 'right', or 'middle'
            
        Returns:
            True if successful
        """
        if not self.press_mouse_button(button):
            return False
        
        # Small delay
        time.sleep(0.01)
        
        # Release
        return self.release_mouse_button(button)
    
    def set_key_state(self, key: str, pressed: bool) -> bool:
        """
        Set key state (press or release)
        
        Args:
            key: Key name ('w', 'a', 's', 'd', 'space', 'alt', etc.)
            pressed: True to press, False to release
            
        Returns:
            True if successful
        """
        scancode_map = {
            'w': SCANCODE_W,
            'a': SCANCODE_A,
            's': SCANCODE_S,
            'd': SCANCODE_D,
            'q': SCANCODE_Q,
            'e': SCANCODE_E,
            'space': SCANCODE_SPACE,
            'alt': SCANCODE_ALT,
            'ctrl': SCANCODE_CTRL,
            'shift': SCANCODE_SHIFT
        }
        
        if key not in scancode_map:
            return False
        
        scancode = scancode_map[key]
        current_state = self.key_states.get(key, False)
        
        # Only send event if state changed
        if current_state != pressed:
            if pressed:
                result = self.press_key(scancode)
            else:
                result = self.release_key(scancode)
            
            if result:
                self.key_states[key] = pressed
            
            return result
        
        return True
    
    def get_key_state(self, key: str) -> bool:
        """Get current key state"""
        return self.key_states.get(key, False)

