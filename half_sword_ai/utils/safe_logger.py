"""
Safe Logger - Windows-compatible logging with emoji handling
Handles Unicode encoding errors gracefully on Windows console
"""
import logging
import sys
import io


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode encoding errors"""
    
    def __init__(self, stream=None, strip_emojis=False):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        self.strip_emojis = strip_emojis
    
    def emit(self, record):
        """Emit a record, handling Unicode encoding errors"""
        try:
            msg = self.format(record)
            
            # Strip emojis if requested or if encoding fails
            if self.strip_emojis:
                msg = self._strip_emojis(msg)
            
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # If encoding fails, strip emojis and try again
            try:
                msg = self._strip_emojis(msg)
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                # Last resort: use ASCII-safe message
                safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
                stream.write(safe_msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)
    
    def _strip_emojis(self, text: str) -> str:
        """Remove emoji characters from text"""
        import re
        # Remove emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)


def setup_safe_logging(strip_emojis: bool = None):
    """
    Setup safe logging that handles Windows console encoding issues
    
    Args:
        strip_emojis: If True, strip emojis. If None, auto-detect Windows console
    """
    import sys
    
    # Auto-detect if we should strip emojis
    if strip_emojis is None:
        # Check if running on Windows and console doesn't support UTF-8
        is_windows = sys.platform == 'win32'
        try:
            # Try to detect console encoding
            encoding = sys.stdout.encoding or 'utf-8'
            strip_emojis = is_windows and encoding.lower() in ('cp1252', 'ascii', 'latin-1')
        except:
            strip_emojis = is_windows
    
    root_logger = logging.getLogger()
    
    # Replace existing StreamHandlers with SafeStreamHandler
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)
            safe_handler = SafeStreamHandler(handler.stream, strip_emojis=strip_emojis)
            safe_handler.setLevel(handler.level)
            safe_handler.setFormatter(handler.formatter)
            root_logger.addHandler(safe_handler)
    
    return root_logger

