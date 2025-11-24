# Mouse Attack Swing Fix

## Problem
In Half Sword, attacks require **holding the mouse button while moving**, not just clicking. The previous implementation only clicked buttons, which doesn't trigger attacks properly.

## Solution
Updated the system to:
1. **Detect attack swings** - Large, rapid mouse movements (>200 pixels magnitude)
2. **Hold button BEFORE movement** - Press LMB/RMB before starting the swing
3. **Move mouse WHILE button held** - Perform the swing gesture while button is pressed
4. **Release button AFTER movement** - Release button after swing completes

## Changes Made

### 1. DirectInput (`half_sword_ai/input/direct_input.py`)
- Added `press_mouse_button()` - Holds button down
- Added `release_mouse_button()` - Releases button
- Updated `click_mouse_button()` - Still available for single clicks
- Tracks mouse button states

### 2. Gesture Engine (`half_sword_ai/input/gesture_engine.py`)
- Updated `perform_macro_action()` to accept `buttons` parameter
- Automatically holds mouse button for attack actions (strikes, slashes, thrusts)
- Releases button after gesture completes

### 3. Input Multiplexer (`half_sword_ai/input/input_mux.py`)
- Detects attack swings based on movement magnitude (>200 pixels)
- Holds appropriate mouse button (left/right) before movement
- Performs movement while button is held
- Releases button after movement completes
- Handles both DirectInput and PyAutoGUI fallback modes

## How It Works

### Attack Detection
```python
movement_magnitude = sqrt(pixel_x² + pixel_y²)
is_attack_swing = movement_magnitude > 200  # Threshold
```

### Attack Swing Sequence
1. **Detect attack** - Large movement detected
2. **Press button** - `press_mouse_button('left')` or `press_mouse_button('right')`
3. **Perform gesture** - `perform_smooth_gesture(dx, dy, duration)` while button held
4. **Release button** - `release_mouse_button('left')` after gesture completes

### Button Selection
- Uses button from `buttons` dict if provided (LMB or RMB)
- Defaults to left button for large attack swings
- Respects user's button choice when specified

## Attack Types Supported

### Strikes (Large Movements)
- **Overhead Strike** - Downward movement (>400px)
- **Horizontal Slash L** - Left-to-right (>400px)
- **Horizontal Slash R** - Right-to-left (>400px)
- **Thrust** - Forward with ALT key held

### Guards (Small Movements)
- Small movements don't trigger attack mode
- Buttons can still be held for positioning

## Testing

To verify attack swings work:
1. Run the agent in autonomous mode
2. Agent should perform large mouse movements
3. Check logs for "Holding left/right button for attack swing"
4. Verify button is held during movement
5. Check logs for "Released button after swing"

## Notes

- **Attack threshold**: 200 pixels movement magnitude
- **Button hold duration**: Matches gesture duration (typically 50ms for strikes)
- **Fallback support**: Works with both DirectInput and PyAutoGUI
- **Safety**: Only injects in AUTONOMOUS mode with safety checks

## Future Improvements

- Adjustable attack threshold based on game sensitivity
- Button hold duration optimization per attack type
- Support for combo attacks (multiple swings in sequence)
- Attack velocity/acceleration tracking for better detection

