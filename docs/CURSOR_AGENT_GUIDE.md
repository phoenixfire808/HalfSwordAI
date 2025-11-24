# Cursor AI Agent Guide - Half Sword AI Project

## 🎯 Project Overview

This is a **modular autonomous learning agent** for the Half Sword game. The system uses:
- **Neural networks** (CNN + MLP) for action prediction
- **YOLO object detection** for enemy/threat detection
- **Ollama/Qwen LLM** for strategic decision-making
- **Continuous online learning** with DAgger + PPO
- **Human-in-the-loop** recording for imitation learning

## 📁 Project Structure

```
ai butler 2/
├── main.py                    # Entry point - START HERE
├── requirements.txt           # Dependencies
├── half_sword_ai/            # Main modular codebase
│   ├── config/               # Configuration (Config class)
│   ├── core/                 # Core components
│   │   ├── agent.py          # Main orchestrator (HalfSwordAgent)
│   │   ├── actor.py          # Real-time inference loop
│   │   ├── learner.py        # Background training
│   │   └── model.py          # Neural network architecture
│   ├── perception/           # Vision and detection
│   │   ├── vision.py         # Screen capture, memory reading
│   │   ├── yolo_detector.py  # Object detection
│   │   └── yolo_self_learning.py  # Self-learning YOLO
│   ├── learning/             # Learning components
│   │   ├── replay_buffer.py  # Experience replay
│   │   ├── model_tracker.py  # Training tracking
│   │   └── human_recorder.py # Human action recording
│   ├── input/                # Input handling
│   │   ├── input_mux.py      # Human/bot control switching
│   │   └── kill_switch.py    # Emergency stop
│   ├── monitoring/           # Monitoring and observability
│   │   ├── performance_monitor.py  # Performance metrics
│   │   ├── watchdog.py       # Game state monitoring
│   │   └── dashboard/        # Web dashboard
│   └── llm/                  # LLM integration
│       └── ollama_integration.py  # Ollama/Qwen API
├── data/                     # Training data (human sessions)
├── logs/                     # Runtime logs and reports
├── models/                   # Model checkpoints
├── archive/                  # Old/unused files (ignore)
└── yolov8n.pt               # YOLO pretrained model
```

## 🚀 Quick Start for Agents

### 1. Understanding the Flow

**Entry Point**: `main.py` → `HalfSwordAgent` (in `half_sword_ai/core/agent.py`)

**Main Flow**:
1. Agent initializes all components
2. Actor runs inference loop (real-time)
3. Learner trains in background (continuous)
4. Performance monitor tracks metrics
5. Dashboard provides web UI

### 2. Key Components to Know

#### Configuration (`half_sword_ai/config/__init__.py`)
- **Single source of truth** for all settings
- Access via: `from half_sword_ai.config import config`
- Key settings:
  - `GAME_EXECUTABLE_PATH`: Path to game executable
  - `AUTO_LAUNCH_GAME`: Auto-launch if not detected
  - `CAPTURE_FPS`: Screen capture frame rate
  - `LEARNING_RATE`: Training learning rate
  - `OLLAMA_ENABLED`: Enable/disable LLM

#### Core Agent (`half_sword_ai/core/agent.py`)
- **Main orchestrator** - manages all components
- Initializes everything in `initialize()`
- Runs main loop in `start()`
- Handles shutdown in `shutdown()`

#### Actor Process (`half_sword_ai/core/actor.py`)
- **Real-time inference loop** at game frame rate
- Captures frames → runs YOLO → runs model inference → injects actions
- Handles human override detection
- Records actions for training

#### Learner Process (`half_sword_ai/core/learner.py`)
- **Background training** - runs continuously
- Uses DAgger + PPO hybrid loss
- Prioritizes human actions 5x
- Updates model in real-time

#### Perception (`half_sword_ai/perception/vision.py`)
- `ScreenCapture`: High-speed screen capture (DXCam/MSS)
- `MemoryReader`: Direct memory access (Pymem)
- `VisionProcessor`: Combines screen + memory + YOLO

## 🔧 Common Tasks

### Adding a New Feature

1. **Identify the module** it belongs to:
   - Vision/detection → `perception/`
   - Learning/training → `learning/`
   - Input handling → `input/`
   - Monitoring → `monitoring/`
   - Core logic → `core/`

2. **Create/update the file** in the appropriate module

3. **Update `__init__.py`** in that module to export new classes/functions

4. **Update imports** in files that use it

5. **Add configuration** if needed in `config/__init__.py`

### Modifying Configuration

**Location**: `half_sword_ai/config/__init__.py`

**Pattern**:
```python
@dataclass
class Config:
    SETTING_NAME: type = default_value
```

**Usage**:
```python
from half_sword_ai.config import config
config.SETTING_NAME = new_value
```

### Adding a New Metric to Performance Monitor

1. **Add tracking** in `half_sword_ai/monitoring/performance_monitor.py`:
   - Add deque for storing values
   - Add `record_*()` method
   - Update `get_current_stats()` to include it

2. **Update dashboard** in `half_sword_ai/monitoring/dashboard/server.py`:
   - Add to `_get_performance_data()` or appropriate endpoint
   - Update HTML template if needed

### Debugging Issues

1. **Check logs** in `logs/` directory
2. **Check performance reports** in `logs/performance_report_*.txt`
3. **Use dashboard** at `http://localhost:5000` (if running)
4. **Enable detailed logging**: Set `config.DETAILED_LOGGING = True`

### Testing Changes

1. **Run the system**: `python main.py`
2. **Check console output** for errors
3. **Monitor dashboard** for metrics
4. **Check logs** for detailed information

## 📝 Important Conventions

### Import Patterns

**Always use full module paths**:
```python
# ✅ CORRECT
from half_sword_ai.core.model import create_model
from half_sword_ai.config import config

# ❌ WRONG
from model import create_model
from config import config
```

### File Organization

- **One class per file** (generally)
- **Related classes** can be in same file
- **Keep modules focused** - don't mix concerns

### Configuration

- **All settings** go in `config/__init__.py`
- **No hardcoded values** in code
- **Use descriptive names** for settings

### Error Handling

- **Log errors** with context
- **Use try/except** for optional features
- **Graceful degradation** when dependencies missing

## 🎯 Key Design Principles

1. **Modularity**: Each component is self-contained
2. **Separation of Concerns**: Clear boundaries between modules
3. **Configuration-Driven**: Settings in config, not code
4. **Real-time**: Actor runs at game frame rate
5. **Continuous Learning**: Training happens in background
6. **Human-in-the-Loop**: Always allow human override

## 🔍 Finding Things

### Where is X?

- **Game launch logic**: `half_sword_ai/perception/vision.py` → `MemoryReader._launch_game_if_needed()`
- **Action injection**: `half_sword_ai/core/actor.py` → `_inject_action()`
- **Training loop**: `half_sword_ai/core/learner.py` → `_training_step()`
- **YOLO detection**: `half_sword_ai/perception/yolo_detector.py` → `YOLODetector.detect()`
- **Performance metrics**: `half_sword_ai/monitoring/performance_monitor.py`
- **Dashboard API**: `half_sword_ai/monitoring/dashboard/server.py`

### Common Patterns

- **Component initialization**: Check `agent.py` → `initialize()`
- **Main loops**: Check `actor.py` → `start()` and `learner.py` → `start()`
- **State management**: Check `performance_monitor.py` and `model_tracker.py`

## ⚠️ Important Warnings

1. **Don't modify archived files** - they're old versions
2. **Always update imports** when moving files
3. **Test after changes** - system is complex
4. **Check dependencies** - some features require optional packages
5. **Game path** - Update `GAME_EXECUTABLE_PATH` if game location changes

## 🐛 Common Issues

### Import Errors
- **Solution**: Check import paths use `half_sword_ai.` prefix
- **Check**: All `__init__.py` files export correctly

### Game Not Detected
- **Check**: `GAME_EXECUTABLE_PATH` is correct
- **Check**: `AUTO_LAUNCH_GAME` is enabled
- **Check**: Game process name matches `GAME_PROCESS_NAME`

### Performance Issues
- **Check**: `CAPTURE_FPS` - lower if needed
- **Check**: `YOLO_DETECTION_INTERVAL` - increase if needed
- **Check**: `DETAILED_LOGGING` - disable for performance

### Training Not Working
- **Check**: Replay buffer has data (`len(replay_buffer) > 0`)
- **Check**: Human actions being recorded
- **Check**: Model is being updated (`update_count` in learner)

## 📚 Additional Resources

- **Architecture docs**: `archive/docs/ARCHITECTURE.md` (historical)
- **Modular structure**: `archive/docs/MODULAR_STRUCTURE.md` (historical)
- **Performance improvements**: `archive/docs/PERFORMANCE_IMPROVEMENTS.md`

## 🎓 Learning the Codebase

### Recommended Reading Order

1. **`main.py`** - Understand entry point
2. **`half_sword_ai/config/__init__.py`** - Understand configuration
3. **`half_sword_ai/core/agent.py`** - Understand initialization
4. **`half_sword_ai/core/actor.py`** - Understand inference loop
5. **`half_sword_ai/core/learner.py`** - Understand training
6. **Other modules** as needed

### Key Concepts

- **Actor-Critic**: Actor generates actions, Learner trains policy
- **DAgger**: Dataset Aggregation - learns from human demonstrations
- **PPO**: Proximal Policy Optimization - policy gradient method
- **Prioritized Replay**: Important experiences sampled more often
- **Human-in-the-Loop**: Human can override bot at any time

## 🚨 Emergency Procedures

### System Hanging
- **Kill switch**: Press `F8` (configurable)
- **Check**: Kill switch in `half_sword_ai/input/kill_switch.py`

### Game Crashes
- **Watchdog**: Automatically detects and restarts
- **Check**: `half_sword_ai/monitoring/watchdog.py`

### Memory Issues
- **Check**: Replay buffer size (`REPLAY_BUFFER_SIZE`)
- **Check**: Frame stack size (`FRAME_STACK_SIZE`)

## 💡 Tips for Agents

1. **Read the code** before making changes
2. **Check existing patterns** - follow conventions
3. **Test incrementally** - small changes, test often
4. **Use the dashboard** - great for debugging
5. **Check logs** - detailed information there
6. **Ask questions** - if something is unclear, investigate

## 📞 Quick Reference

```python
# Import patterns
from half_sword_ai.config import config
from half_sword_ai.core import HalfSwordAgent
from half_sword_ai.perception import ScreenCapture, YOLODetector
from half_sword_ai.learning import PrioritizedReplayBuffer
from half_sword_ai.input import InputMultiplexer, KillSwitch
from half_sword_ai.monitoring import PerformanceMonitor

# Configuration access
config.CAPTURE_FPS = 60
config.LEARNING_RATE = 1e-3

# Running the system
python main.py
```

---

**Last Updated**: After modular refactoring
**Maintainer**: Cursor AI Agents
**Status**: Active Development

