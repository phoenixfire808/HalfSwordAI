"""
Safe Logger - Windows-compatible logging with emoji handling
Handles Unicode encoding errors gracefully on Windows console
"""
import logging
import sys
import io
import re


class EmojiFilter(logging.Filter):
    """Filter that strips emojis from log records"""
    
    def __init__(self):
        super().__init__()
        # Emoji pattern matching all common emoji ranges
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "\u2705"  # check mark
            "\u23ed"  # next track button
            "\ufe0f"  # variation selector
            "]+", flags=re.UNICODE)
    
    def filter(self, record):
        """Strip emojis from the log message"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.emoji_pattern.sub('', record.msg).strip()
        if hasattr(record, 'getMessage'):
            try:
                msg = record.getMessage()
                record.msg = self.emoji_pattern.sub('', msg).strip()
            except:
                pass
        return True


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode encoding errors"""
    
    def __init__(self, stream=None, strip_emojis=False):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        self.strip_emojis = strip_emojis
    
    def format(self, record):
        """Format record and strip emojis if needed"""
        msg = super().format(record)
        if self.strip_emojis:
            msg = self._strip_emojis(msg)
        return msg
    
    def emit(self, record):
        """Emit a record, handling Unicode encoding errors"""
        try:
            # Format the record (emojis already stripped if strip_emojis=True)
            msg = self.format(record)
            
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # If encoding still fails, strip emojis and try again
            try:
                msg = self._strip_emojis(msg)
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                # Last resort: use ASCII-safe message
                try:
                    safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception:
                    # Silently ignore if we can't even do that
                    pass
        except Exception:
            # Don't call handleError which might print traceback - just silently skip
            pass
    
    def handleError(self, record):
        """Override to prevent error messages from being printed"""
        # Silently ignore errors to prevent "--- Logging error ---" messages
        pass
    
    def _strip_emojis(self, text: str) -> str:
        """Remove emoji characters from text"""
        import re
        # Remove emoji ranges - expanded to catch more emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
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
            strip_emojis = is_windows and encoding.lower() in ('cp1252', 'ascii', 'latin-1', 'cp437')
        except:
            strip_emojis = is_windows
    
    root_logger = logging.getLogger()
    
    # Add emoji filter to root logger if we need to strip emojis
    if strip_emojis:
        emoji_filter = EmojiFilter()
        root_logger.addFilter(emoji_filter)
        # Also add to all existing handlers
        for handler in root_logger.handlers[:]:
            handler.addFilter(emoji_filter)
    
    # Replace existing StreamHandlers with SafeStreamHandler
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)
            safe_handler = SafeStreamHandler(handler.stream, strip_emojis=strip_emojis)
            safe_handler.setLevel(handler.level)
            safe_handler.setFormatter(handler.formatter)
            if strip_emojis:
                safe_handler.addFilter(EmojiFilter())
            root_logger.addHandler(safe_handler)
    
    return root_logger

