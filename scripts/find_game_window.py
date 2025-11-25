"""Find Half Sword game window"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import win32gui
    import win32con
    
    windows = []
    def callback(hwnd, w):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            w.append((title, width, height, rect))
    
    win32gui.EnumWindows(callback, windows)
    
    print("Windows containing 'Half' or 'Sword':")
    print("=" * 80)
    for title, width, height, rect in windows:
        if 'half' in title.lower() or 'sword' in title.lower():
            exclude = ['dashboard', 'opera', 'chrome', 'firefox', 'edge', 'browser']
            if not any(x in title.lower() for x in exclude):
                print(f"Title: {title}")
                print(f"  Size: {width}x{height}")
                print(f"  Position: ({rect[0]}, {rect[1]})")
                print()
except ImportError:
    print("win32gui not available - install pywin32")

