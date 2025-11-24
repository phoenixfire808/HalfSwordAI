# Human-in-the-Loop Learning Flow

## Overview

The Half Sword AI Agent implements a complete human-in-the-loop learning system that:
1. **Starts autonomously** - Bot learns from its own gameplay
2. **Detects human input** - Automatically switches to manual mode when you move mouse/keyboard
3. **Records everything** - Captures all human actions with vision data
4. **Learns immediately** - Integrates human demonstrations into the model in real-time
5. **Takes back control** - Returns to autonomous mode after you stop moving

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STARTUP                                  │
│  Bot starts in AUTONOMOUS mode                             │
│  - Generates actions from its own model                    │
│  - Learns from its own gameplay                            │
│  - Builds its own AI from scratch                          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              AUTONOMOUS MODE (Bot Control)                  │
│                                                              │
│  • Bot generates actions from neural network               │
│  • Actions injected via Interception/DirectInput            │
│  • Bot's actions recorded for self-learning                │
│  • Model trains continuously in background                 │
│                                                              │
│  Monitoring:                                                │
│  • Continuously checks for human input                      │
│  • Ignores bot's own movements (cooldown protection)       │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ (Human moves mouse/keyboard)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           HUMAN INPUT DETECTED                               │
│                                                              │
│  Detection Algorithm:                                      │
│  • Velocity-based detection                                 │
│  • Acceleration analysis                                    │
│  • Pattern recognition                                      │
│  • Confidence threshold: 0.7                                │
│                                                              │
│  Action:                                                    │
│  • Immediately switch to MANUAL mode                        │
│  • Bot stops injecting actions                              │
│  • Human takes full control                                 │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              MANUAL MODE (Human Control)                    │
│                                                              │
│  Recording System:                                          │
│  • Records EVERY frame when human is active                 │
│  • Captures:                                                │
│    - Mouse movement (dx, dy) in pixels                      │
│    - All keyboard presses (WASD, mouse buttons, etc.)      │
│    - Current game frame (vision data)                       │
│    - Game state (health, position, enemy info)              │
│    - Movement patterns and context                         │
│                                                              │
│  Storage:                                                   │
│  • HumanActionRecorder: Expert buffer                       │
│  • ReplayBuffer: High-priority experiences                 │
│  • Pattern storage: For emulation                           │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ (Human stops moving for 0.5s)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           HUMAN IDLE DETECTED                                │
│                                                              │
│  Timeout Check:                                             │
│  • HUMAN_TIMEOUT = 0.5 seconds                              │
│  • Checks if mouse/keyboard idle                            │
│  • Verifies no recent human input                           │
│                                                              │
│  Action:                                                    │
│  • Switch back to AUTONOMOUS mode                           │
│  • Mark "just_switched_to_autonomous" flag                   │
│  • Reset human input timers                                 │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         INTEGRATION PHASE (First 2 seconds)                 │
│                                                              │
│  Immediate Replay:                                          │
│  • Replays last 10-15 human actions                        │
│  • Directly copies human movements (95% weight)            │
│  • Copies button presses exactly                           │
│                                                              │
│  Pattern Matching:                                          │
│  • Matches current game state to human patterns             │
│  • Blends human patterns with model predictions            │
│  • Gradually transitions from 95% → 70% human weight      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         CONTINUOUS LEARNING                                  │
│                                                              │
│  Background Training:                                       │
│  • Learner process runs continuously                        │
│  • Samples from replay buffer                               │
│  • Prioritizes human actions (HIGH priority)                │
│                                                              │
│  Loss Calculation:                                          │
│  • DQN Loss: Standard reinforcement learning               │
│  • BC Loss: Behavioral cloning from human actions          │
│  • BC Weight: 2x higher when human actions present         │
│                                                              │
│  Model Updates:                                             │
│  • Updates happen in real-time                              │
│  • Actor immediately uses new weights                       │
│  • Continuous improvement from human demonstrations         │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Input Multiplexer (`input_mux.py`)
- **Mode Management**: Switches between AUTONOMOUS and MANUAL
- **Human Detection**: Multi-algorithm detection (velocity, acceleration, patterns)
- **Bot Protection**: Cooldown prevents detecting bot's own movements
- **Timeout Handling**: Returns to autonomous after 0.5s of inactivity

### 2. Human Action Recorder (`human_recorder.py`)
- **Full Context Recording**: Frame + game_state + all controls
- **Expert Buffer**: Stores all human demonstrations
- **Pattern Analysis**: Extracts movement patterns in real-time
- **Session Management**: Saves sessions for later analysis

### 3. Actor Process (`actor.py`)
- **Action Generation**: Generates bot actions from model
- **Human Recording**: Records human actions with full context
- **Replay Integration**: Replays human actions when switching back
- **Pattern Matching**: Matches and blends human patterns

### 4. Learner Process (`learner.py`)
- **Continuous Training**: Trains model in background
- **Priority Sampling**: Prioritizes human actions (HIGH priority)
- **BC Loss**: Behavioral cloning from human demonstrations
- **Adaptive Weighting**: Higher BC weight when human actions present

## Data Flow

### When Human Takes Control:
```
Human Input → Input Mux Detection → Switch to MANUAL
    ↓
Actor Loop Detects MANUAL Mode
    ↓
Record Every Frame:
    • Frame (vision)
    • Game State (health, position, etc.)
    • Mouse Delta (dx, dy)
    • All Button States (WASD, mouse buttons, etc.)
    • Movement Patterns
    ↓
Store in:
    • HumanActionRecorder.expert_buffer
    • ReplayBuffer (HIGH priority, human_intervention=True)
```

### When Bot Takes Back Control:
```
Human Idle Detected → Switch to AUTONOMOUS
    ↓
Immediate Replay:
    • Get last 10-15 human actions
    • Replay them directly (95% weight)
    • Gradually blend with model (95% → 70%)
    ↓
Pattern Matching:
    • Match current state to human patterns
    • Blend human patterns with model predictions
    ↓
Continuous Learning:
    • Learner samples human actions (HIGH priority)
    • Trains with BC loss (2x weight)
    • Updates model in real-time
```

## Configuration

### Key Settings (`config.py`):
```python
HUMAN_TIMEOUT = 0.5  # Seconds before switching back to autonomous
NOISE_THRESHOLD = 2.0  # Minimum movement to detect as human
BATCH_SIZE = 32  # Training batch size
LEARNING_RATE = 1e-4  # Model learning rate
```

### Detection Thresholds (`input_mux.py`):
```python
detection_confidence = 0.7  # Minimum confidence to detect human
movement_magnitude = 5.0  # Minimum pixels to detect movement
bot_injection_cooldown = 0.5  # Ignore detection after bot injection
```

## Learning Integration

### Behavioral Cloning (BC):
- **Purpose**: Learn to imitate human actions
- **Loss**: Distance between predicted action and human action
- **Weight**: 2x higher when human actions present
- **Priority**: Human actions sampled more frequently

### Pattern Matching:
- **Purpose**: Recognize and replay human movement patterns
- **Method**: Matches current game state to stored patterns
- **Blending**: 95% human → 70% human → 30% human over 2 seconds
- **Context**: Considers health, enemy position, threat level

### Replay Buffer Priority:
- **HIGH Priority**: Human actions (sampled 5x more often)
- **LOW Priority**: Bot actions
- **TD Error**: Used for additional prioritization
- **Temporal Weight**: Newer experiences slightly more important

## Verification Checklist

✅ **Autonomous Start**: Bot starts in AUTONOMOUS mode
✅ **Human Detection**: Detects mouse/keyboard movement
✅ **Mode Switch**: Switches to MANUAL immediately
✅ **Recording**: Records all human actions with vision data
✅ **Learning**: Learns from human actions with high priority
✅ **Switch Back**: Returns to AUTONOMOUS after timeout
✅ **Integration**: Replays and blends human actions
✅ **Continuous**: Model updates continuously in background

## Current Status

The system is **fully functional** and implements all required features:
- ✅ Autonomous operation
- ✅ Human detection and mode switching
- ✅ Complete action recording with vision
- ✅ Real-time learning integration
- ✅ Automatic return to autonomous mode
- ✅ Pattern matching and emulation

## Usage Tips

1. **Let bot start**: Bot will begin autonomously learning
2. **Take control**: Move mouse/keyboard to take over
3. **Demonstrate**: Show the bot what you want it to learn
4. **Stop moving**: After 0.5s of inactivity, bot takes back control
5. **Watch it learn**: Bot will immediately try to emulate your actions

The bot learns from **everything** you do:
- Mouse movements
- Mouse clicks (left/right)
- Keyboard presses (WASD, space, alt, etc.)
- Movement patterns
- Timing and rhythm
- Context (when to do what)

