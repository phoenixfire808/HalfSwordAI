"""
Main Entry Point: Half Sword AI Agent
ScrimBrain Integration - Modular architecture entry point
"""
import sys
import io

# Wrap sys.stdout to handle Unicode encoding errors on Windows
if sys.platform == 'win32':
    try:
        encoding = sys.stdout.encoding or 'utf-8'
        if encoding.lower() in ('cp1252', 'ascii', 'latin-1', 'cp437'):
            # Wrap stdout to replace Unicode errors instead of raising exceptions
            class SafeStdout:
                def __init__(self, original_stdout):
                    self.original_stdout = original_stdout
                    self.encoding = original_stdout.encoding
                
                def write(self, text):
                    try:
                        self.original_stdout.write(text)
                    except UnicodeEncodeError:
                        # Replace Unicode characters that can't be encoded
                        safe_text = text.encode(self.encoding or 'cp1252', errors='replace').decode(self.encoding or 'cp1252')
                        self.original_stdout.write(safe_text)
                
                def flush(self):
                    self.original_stdout.flush()
                
                def __getattr__(self, name):
                    return getattr(self.original_stdout, name)
            
            sys.stdout = SafeStdout(sys.stdout)
            sys.stderr = SafeStdout(sys.stderr)
    except:
        pass

from half_sword_ai.core.agent import HalfSwordAgent
from half_sword_ai.config import config
from half_sword_ai.utils.pretty_logger import setup_pretty_logging
from half_sword_ai.utils.safe_logger import setup_safe_logging
import logging

def main():
    """Main entry point"""
    # Detect Windows console encoding issues FIRST, before any logging
    is_windows = sys.platform == 'win32'
    try:
        encoding = sys.stdout.encoding or 'utf-8'
        is_windows_console = is_windows and encoding.lower() in ('cp1252', 'ascii', 'latin-1', 'cp437')
    except:
        is_windows_console = is_windows
    
    # Setup safe logging FIRST (handles Windows Unicode issues)
    # This must happen before any other imports that might create loggers
    from half_sword_ai.utils.safe_logger import setup_safe_logging, EmojiFilter
    setup_safe_logging(strip_emojis=is_windows_console)
    
    # Apply emoji filter to root logger and all existing loggers
    if is_windows_console:
        root_logger = logging.getLogger()
        emoji_filter = EmojiFilter()
        root_logger.addFilter(emoji_filter)
        # Apply to all existing loggers
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.addFilter(emoji_filter)
    
    # Then setup pretty logging (will use safe handlers and strip emojis if needed)
    setup_pretty_logging(use_colors=True, use_emojis=not is_windows_console)
    logger = logging.getLogger(__name__)
    
    model_type = "DQN (ScrimBrain-style)" if config.USE_DISCRETE_ACTIONS else "PPO (Continuous)"
    
    # Beautiful startup banner (check Windows console encoding)
    print("\n" + "="*80)
    if is_windows_console:
        print(" " * 20 + "HALF SWORD AI AGENT")
        print(" " * 15 + "Autonomous Learning Agent")
        print(" " * 18 + "ScrimBrain Integration")
    else:
        print(" " * 20 + "⚔️  HALF SWORD AI AGENT  ⚔️")
        print(" " * 15 + "Autonomous Learning Agent")
        print(" " * 18 + "ScrimBrain Integration")
    print("="*80 + "\n")
    
    # Always use plain text on Windows console to avoid Unicode errors
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Frame Size: {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT} (ScrimBrain standard)")
    logger.info(f"Frame Stack: {config.FRAME_STACK_SIZE} frames")
    logger.info(f"Frame Skip: {config.FRAME_SKIP} (physics stability)")
    print("")
    
    agent = HalfSwordAgent()
    agent.initialize()
    agent.start()

if __name__ == "__main__":
    main()
