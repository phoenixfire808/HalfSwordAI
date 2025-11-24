# Windows Unicode and JSON Serialization Fixes - November 24, 2025

## Issues Found in Logs

### 1. UnicodeEncodeError: Windows Console Can't Encode Emojis
**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4f9' in position 94`
**Location**: Multiple logging statements with emojis
**Root Cause**: Windows console (cp1252 encoding) doesn't support Unicode emoji characters.

**Fix**: Created `SafeStreamHandler` that automatically strips emojis on Windows console or handles encoding errors gracefully.

```python
# Created half_sword_ai/utils/safe_logger.py
class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode encoding errors"""
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.strip_emojis:
                msg = self._strip_emojis(msg)
            stream.write(msg + self.terminator)
        except UnicodeEncodeError:
            # Strip emojis and retry
            msg = self._strip_emojis(msg)
            stream.write(msg + self.terminator)
```

### 2. JSON Serialization Error: numpy.bool_ Not Serializable
**Error**: `Object of type bool is not JSON serializable`
**Location**: `half_sword_ai/learning/human_recorder.py:_save_session`
**Root Cause**: `game_state` dictionary contains numpy bool types (`np.bool_`) which aren't JSON serializable.

**Fix**: Added `_make_json_serializable` method to convert numpy types to Python native types.

```python
def _make_json_serializable(self, obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: self._make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [self._make_json_serializable(item) for item in obj]
    return obj
```

### 3. Removed Emojis from Log Messages
**Fix**: Removed emojis from critical log messages to prevent Unicode errors:
- `📹` removed from human_recorder
- `📈` removed from performance_monitor
- `🔄` removed from learner
- `💾` removed from session saving

## Changes Made

1. **Created `half_sword_ai/utils/safe_logger.py`**:
   - `SafeStreamHandler` class for Windows-compatible logging
   - Automatic emoji stripping on Windows console
   - Graceful Unicode error handling

2. **Updated `half_sword_ai/learning/human_recorder.py`**:
   - Added `_make_json_serializable` method
   - Convert `game_state` before saving
   - Convert all session data before JSON serialization
   - Removed emojis from log messages

3. **Updated `half_sword_ai/monitoring/performance_monitor.py`**:
   - Added `_make_json_serializable` method
   - Convert all metrics data before JSON serialization
   - Removed emojis from log messages

4. **Updated `half_sword_ai/core/agent.py`**:
   - Use `SafeStreamHandler` for console logging
   - Auto-detect Windows console encoding

5. **Updated `half_sword_ai/core/learner.py`**:
   - Removed emojis from log messages

6. **Updated `main.py`**:
   - Setup safe logging before pretty logging
   - Auto-detect Windows console and disable emojis if needed

## Impact

- ✅ **No more Unicode errors**: Windows console logging works correctly
- ✅ **JSON serialization works**: All numpy types converted to Python types
- ✅ **Session saving works**: Human recordings save successfully
- ✅ **Metrics saving works**: Performance metrics save to JSON correctly

## Status

✅ **FIXED** - All Unicode and JSON serialization issues resolved. System should now work correctly on Windows.

