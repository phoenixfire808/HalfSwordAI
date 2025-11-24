# Half Sword Controls - Complete Reference

This document details all controls used in Half Sword and how they're implemented in the AI agent.

## Control Scheme

### Mouse Controls
- **Mouse Movement (X-axis)**: Controls torso rotation/yaw - rotates character left/right
- **Mouse Movement (Y-axis)**: Controls torso pitch - tilts character up/down
- **Left Mouse Button (LMB)**: Left hand control - grips/swings with left hand
- **Right Mouse Button (RMB)**: Right hand control / Half-swording mode - grips blade for thrusting

### Movement Keys (WASD)
- **W**: Forward movement - applies force forward
- **A**: Left strafe - applies force left
- **S**: Backward movement - applies force backward  
- **D**: Right strafe - applies force right
- **Diagonal Movement**: Holding multiple keys normalizes to 45-degree angles

### Grab Mechanics
- **Q**: Left hand grab - independent left hand control
- **E**: Right hand grab - independent right hand control

### Combat Controls
- **SPACE**: Jump/dodge - evasive movement
- **ALT**: Thrust/half-swording - enables thrust attacks when held
- **G**: Surrender - gives up control

### Optional Controls
- **SHIFT**: Sprint (if implemented in game)
- **CTRL**: Crouch (if implemented in game)

## Implementation in AI Agent

### Button State Tracking
All controls are tracked in real-time via:
- `half_sword_ai/input/input_mux.py` - `_get_current_button_states()`
- Uses Windows API (`win32api.GetAsyncKeyState`) for low-latency detection
- Falls back to `keyboard` library if win32api unavailable

### Action Recording
All controls are logged in:
- `half_sword_ai/learning/human_recorder.py` - `record_action()`
- Records:
  - Mouse delta (dx, dy) in pixels and normalized
  - All button states (WASD, Q/E, mouse buttons, etc.)
  - Movement vector (normalized WASD direction)
  - Button hold durations
  - Button press/release events

### Action Discretization
Sword movements are discretized into macro-actions:
- `half_sword_ai/input/action_discretizer.py`
- Movement keys (WASD) remain continuous
- Allows combining movement with sword techniques

### Learning Integration
The learning system (`half_sword_ai/core/learner.py`) uses:
- **Behavioral Cloning**: Learns from all recorded human actions
- **DQN**: Uses discrete sword actions + continuous movement
- **Expert Buffer**: Stores all human demonstrations with full control context

## Data Structure

Each recorded action includes:
```python
{
    'timestamp': float,
    'relative_time': float,
    # Mouse movement
    'mouse_delta': (dx, dy),           # Pixels
    'normalized_delta': (dx_norm, dy_norm),  # [-1, 1]
    'mouse_magnitude': float,
    # Movement keys (WASD)
    'movement_vector': {'x': float, 'y': float},  # Normalized direction
    'movement_x': float,  # -1 to 1
    'movement_y': float,  # -1 to 1
    # All button states
    'buttons': {
        'left': bool, 'right': bool,
        'w': bool, 'a': bool, 's': bool, 'd': bool,
        'q': bool, 'e': bool,
        'space': bool, 'alt': bool, 'g': bool,
        'shift': bool, 'ctrl': bool
    },
    'button_changes': {...},  # Press/release events
    'button_hold_durations': {...},  # How long each button held
    # Motion tracking
    'velocity': float,
    'acceleration': float,
    # Game state
    'game_state': {...},
    'frame_shape': tuple,
    'pattern_context': {...}
}
```

## Usage in Training

1. **Human Demonstration**: All controls are recorded during human gameplay
2. **Behavioral Cloning**: Model learns to predict all controls from game state
3. **Reinforcement Learning**: Agent learns optimal control combinations
4. **Pattern Matching**: System recognizes control patterns (e.g., "thrust while moving forward")

## Notes

- Movement keys (WASD) are **continuous** - can be held for variable durations
- Mouse movement is **continuous** - captures exact movement patterns
- Button states are **binary** - pressed or not pressed
- The system captures **combinations** - e.g., moving forward while thrusting (W + ALT)
- All controls are logged at **60 FPS** for high-fidelity learning

