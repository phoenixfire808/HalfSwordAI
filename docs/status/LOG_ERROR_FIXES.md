# Log Error Fixes - November 24, 2025

## Issues Found in Logs

### 1. UnboundLocalError: `time` variable in `input_mux.py`
**Error**: `cannot access local variable 'time' where it is not associated with a value`
**Location**: `half_sword_ai/input/input_mux.py:877`
**Root Cause**: Local `import time` statement inside function shadowed the module-level import, causing variable access issues.

**Fix**: Removed local `import time` statement since `time` is already imported at module level (line 14).

```python
# Before (line 835):
if is_attack_swing and mouse_button_to_hold:
    import time  # ❌ Local import shadows module-level import
    time.sleep(0.05)

# After:
if is_attack_swing and mouse_button_to_hold:
    time.sleep(0.05)  # ✅ Uses module-level import
```

### 2. UnboundLocalError: `human_action_count` variable in `learner.py`
**Error**: `cannot access local variable 'human_action_count' where it is not associated with a value`
**Location**: `half_sword_ai/core/learner.py:210`
**Root Cause**: Variable `human_action_count` was used in log statement before being calculated.

**Fix**: Moved `human_action_count` calculation before its first use.

```python
# Before:
if self.update_count % 10 == 0:
    logger.info(f"... Human: {human_action_count}")  # ❌ Used before definition

recent_experiences = list(self.replay_buffer.buffer)[-100:]
human_action_count = sum(1 for exp in recent_experiences ...)  # Defined here

# After:
recent_experiences = list(self.replay_buffer.buffer)[-100:]
human_action_count = sum(1 for exp in recent_experiences ...)  # ✅ Defined first

if self.update_count % 10 == 0:
    logger.info(f"... Human: {human_action_count}")  # ✅ Used after definition
```

## Impact

These errors were causing:
- **System crashes**: UnboundLocalError exceptions stopped the agent
- **Training failures**: Learner process crashed before completing training steps
- **Input injection failures**: Attack swings failed due to time.sleep() error

## Verification

- ✅ Fixed variable scoping issues
- ✅ Verified imports are correct
- ✅ No linter errors
- ✅ Code follows Python best practices

## Prevention

To prevent similar issues:
1. **Avoid local imports** when module-level import exists
2. **Define variables before use** in all code paths
3. **Use linter** to catch variable scoping issues early
4. **Test error paths** to catch UnboundLocalError exceptions

## Status

✅ **FIXED** - Both errors resolved. System should now run without these crashes.

