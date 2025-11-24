"""
Pretty Terminal Logger - Beautiful, readable terminal output
Uses colors, emojis, and clean formatting for better readability
"""
import logging
import sys
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }
    
    # Emojis for different log types
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
    }
    
    # Component icons
    COMPONENT_ICONS = {
        'actor': '🤖',
        'learner': '🧠',
        'input_mux': '🎮',
        'vision': '👁️',
        'perception': '👁️',
        'monitoring': '📊',
        'dashboard': '🌐',
        'error_handler': '🛡️',
        'kill_switch': '🛑',
        'agent': '⚔️',
    }
    
    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_emojis = use_emojis
    
    def format(self, record: logging.LogRecord) -> str:
        # Get component name from logger name
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        component_icon = self.COMPONENT_ICONS.get(component, '•')
        
        # Get emoji for log level
        emoji = self.EMOJIS.get(record.levelname, '') if self.use_emojis else ''
        
        # Get color for log level
        color = self.COLORS.get(record.levelname, '') if self.use_colors else ''
        reset = self.COLORS['RESET'] if self.use_colors else ''
        
        # Format message based on level
        if record.levelname == 'INFO':
            # Clean INFO messages - remove module name clutter
            if 'half_sword_ai' in record.name:
                name_parts = record.name.split('.')
                if len(name_parts) > 2:
                    component = name_parts[-1]
                else:
                    component = name_parts[-1]
            
            # Format: [Component] Message
            formatted = f"{color}{component_icon} {component.upper()}:{reset} {record.getMessage()}"
        elif record.levelname == 'WARNING':
            formatted = f"{color}{emoji} {component.upper()}:{reset} {record.getMessage()}"
        elif record.levelname == 'ERROR':
            formatted = f"{color}{emoji} ERROR [{component}]:{reset} {record.getMessage()}"
        elif record.levelname == 'CRITICAL':
            formatted = f"{color}{self.COLORS['BOLD']}{emoji} CRITICAL [{component}]:{reset} {record.getMessage()}"
        else:
            # DEBUG messages - minimal format
            formatted = f"{self.COLORS['DIM'] if self.use_colors else ''}{emoji} [{component}]:{reset} {record.getMessage()}"
        
        return formatted

class StatusLogger:
    """Status logger for periodic updates"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_status_time = {}
    
    def status(self, message: str, component: str = "STATUS", interval: float = 1.0):
        """Log status message with throttling"""
        import time
        current_time = time.time()
        key = f"{component}:{message[:50]}"
        
        if key not in self.last_status_time or current_time - self.last_status_time[key] >= interval:
            self.logger.info(f"📊 {component}: {message}")
            self.last_status_time[key] = current_time

def setup_pretty_logging(use_colors: bool = True, use_emojis: bool = True):
    """Setup pretty logging for terminal output"""
    from half_sword_ai.utils.safe_logger import SafeStreamHandler
    
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler with safe Unicode handling
    console_handler = SafeStreamHandler(sys.stdout, strip_emojis=not use_emojis)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(use_colors=use_colors, use_emojis=use_emojis)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Keep file handler with standard format
    from half_sword_ai.config import config
    import time
    import os
    os.makedirs(config.LOG_PATH, exist_ok=True)
    file_handler = logging.FileHandler(f'{config.LOG_PATH}/agent_{int(time.time())}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    root_logger.setLevel(logging.INFO)
    
    return root_logger

