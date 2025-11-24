# Detailed Logging Enabled - November 24, 2025

## Changes Made

### 1. Enabled Detailed Logging by Default
**File**: `half_sword_ai/config/__init__.py`
- Changed `DETAILED_LOGGING: bool = False` to `DETAILED_LOGGING: bool = True`
- This enables function names and line numbers in log messages

### 2. Added Comprehensive Logging Throughout Codebase

#### Actor Process (`half_sword_ai/core/actor.py`)
- **Mode Checks**: Logs every 60 frames with current mode, human active status, bot injection status
- **Human Input Detection**: Detailed logs when human input is detected, including detection confidence
- **Mode Switching**: Logs all mode switches (AUTONOMOUS <-> MANUAL) with frame numbers
- **Bot Action Generation**: Logs when bot actions are generated, including action values and inference latency
- **Human Action Recording**: Logs when human actions are recorded, including button states and mouse deltas
- **Injection**: Logs all injection attempts, successes, and blocks with reasons
- **Error Handling**: Enhanced error logging with frame numbers, error types, and context

#### Input Multiplexer (`half_sword_ai/input/input_mux.py`)
- **Human Movement Detection**: Logs detection confidence, consecutive detections, bot injection status
- **Mode Switching**: Detailed logs for all mode switches with timing information
- **Injection Blocking**: Logs all reasons for blocking injection (safety lock, mode mismatch, human movement)
- **Human Input Capture**: Logs captured mouse deltas, normalized values, and active buttons
- **Idle Detection**: Logs idle detection with timing details

### 3. Logging Format

All logs now include:
- **Prefix tags**: `[MODE SWITCH]`, `[HUMAN INPUT]`, `[BOT CONTROL]`, `[INJECTION]`, etc.
- **Frame numbers**: For tracking timing and sequence
- **Context values**: Mode, detection confidence, button states, deltas, etc.
- **Error details**: Error types, arguments, and stack traces

### 4. Log Levels

- **DEBUG**: Detailed information for debugging (mode checks, input capture, etc.)
- **INFO**: Important events (mode switches, action generation, etc.)
- **WARNING**: Non-critical issues (injection blocked, frame capture failures, etc.)
- **ERROR**: Errors with full context and stack traces

## Example Log Output

```
[INPUT_MUX] Human movement detected | Confidence: 0.85 | Switching AUTONOMOUS -> MANUAL
[INPUT_MUX] Detection details: consecutive_detections=3, bot_injecting=False
[INPUT_MUX] Mode switched | human_override_count=1, bot_injecting=False
[MODE SWITCH] Human input detected - switching AUTONOMOUS -> MANUAL | Frame: 1234
[HUMAN CONTROL] Frame 1234 | Mode: manual | Recording actions, bot paused
[HUMAN INPUT] Frame 1234 | dx=0.123, dy=-0.456, buttons={'left': True, 'w': True}
[HUMAN BUTTONS] Frame 1234 | Active buttons: ['left', 'w']
[HUMAN RECORD] Frame 1234 | Recording action: mouse_delta=(49.2, -182.4), buttons={'left': True, 'w': True}, reward=0.0234
[INJECTION BLOCKED] Frame 1234 | Mode changed to manual - skipping injection
```

## Benefits

✅ **Easy Debugging**: All critical events are logged with context
✅ **Timing Analysis**: Frame numbers help track sequence and timing
✅ **State Tracking**: Mode, button states, and detection confidence logged
✅ **Error Diagnosis**: Full error context with stack traces
✅ **Performance Monitoring**: Inference and injection latencies logged

## Performance Impact

- Logging is throttled (every 60 frames for mode checks, every 50 frames for actions)
- DEBUG logs can be disabled by setting `DETAILED_LOGGING = False` in config
- File logging uses buffered I/O for performance
- Console logging uses safe handlers to prevent Unicode errors

## Status

✅ **ENABLED** - Detailed logging is now active throughout the codebase for easier debugging.

