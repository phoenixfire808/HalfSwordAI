"""
Window Finder - Find and capture specific application windows
Finds the Half Sword game window and returns its position/size
"""
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

try:
    import win32gui
    import win32con
    WIN32GUI_AVAILABLE = True
except ImportError:
    WIN32GUI_AVAILABLE = False
    logger.warning("win32gui not available - install pywin32")

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    logger.warning("pygetwindow not available - install pygetwindow")

class WindowFinder:
    """Find and get information about application windows"""
    
    @staticmethod
    def find_game_window(window_title: str = "Half Sword") -> Optional[Dict]:
        """
        Find Half Sword game window
        
        Args:
            window_title: Partial window title to search for
            
        Returns:
            Dict with window info: {'hwnd': int, 'left': int, 'top': int, 'width': int, 'height': int, 'title': str}
            or None if not found
        """
        if WIN32GUI_AVAILABLE:
            return WindowFinder._find_with_win32gui(window_title)
        elif PYGETWINDOW_AVAILABLE:
            return WindowFinder._find_with_pygetwindow(window_title)
        else:
            logger.warning("No window finding library available")
            return None
    
    @staticmethod
    def _find_with_win32gui(window_title: str) -> Optional[Dict]:
        """Find window using win32gui"""
        try:
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    # Exclude browser windows and dashboard windows
                    exclude_keywords = ['dashboard', 'opera', 'chrome', 'firefox', 'edge', 'browser', 'agent']
                    if any(keyword in title.lower() for keyword in exclude_keywords):
                        return
                    
                    # Look for game window specifically
                    # Prefer exact matches or windows that start with the title
                    title_lower = title.lower()
                    search_lower = window_title.lower()
                    
                    # Exact match gets highest priority
                    is_exact_match = title_lower == search_lower
                    is_contains_match = search_lower in title_lower
                    
                    if is_exact_match or is_contains_match:
                        rect = win32gui.GetWindowRect(hwnd)
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        # Game windows are typically reasonably sized (exclude tiny windows)
                        if width > 800 and height > 600:
                            windows.append({
                                'hwnd': hwnd,
                                'left': rect[0],
                                'top': rect[1],
                                'width': width,
                                'height': height,
                                'title': title,
                                'exact_match': is_exact_match,
                                'size_score': width * height  # Prefer larger windows
                            })
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                # Sort by exact match first, then by size (largest first)
                windows.sort(key=lambda x: (not x.get('exact_match', False), -x.get('size_score', 0)))
                window = windows[0]
                # Remove internal fields before returning
                window.pop('size_score', None)
                window.pop('exact_match', None)
                logger.info(f"Found game window: {window['title']} ({window['width']}x{window['height']})")
                return window
            else:
                logger.warning(f"Game window not found (searching for: {window_title})")
                return None
        except Exception as e:
            logger.error(f"Error finding window with win32gui: {e}")
            return None
    
    @staticmethod
    def _find_with_pygetwindow(window_title: str) -> Optional[Dict]:
        """Find window using pygetwindow"""
        try:
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                window = windows[0]
                if window.visible:
                    result = {
                        'hwnd': None,  # pygetwindow doesn't provide hwnd
                        'left': window.left,
                        'top': window.top,
                        'width': window.width,
                        'height': window.height,
                        'title': window.title
                    }
                    logger.info(f"Found game window: {result['title']} ({result['width']}x{result['height']})")
                    return result
            logger.warning(f"Game window not found (searching for: {window_title})")
            return None
        except Exception as e:
            logger.error(f"Error finding window with pygetwindow: {e}")
            return None
    
    @staticmethod
    def get_window_region(window_info: Dict) -> Dict:
        """
        Convert window info to mss region format
        
        Args:
            window_info: Window info dict from find_game_window
            
        Returns:
            Dict with 'top', 'left', 'width', 'height' for mss.grab()
        """
        return {
            'top': window_info['top'],
            'left': window_info['left'],
            'width': window_info['width'],
            'height': window_info['height']
        }

