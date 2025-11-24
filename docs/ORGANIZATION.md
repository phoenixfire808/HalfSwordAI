# Project Organization Guide

This document describes the organization and structure of the Half Sword AI Agent project.

## Directory Structure

```
half_sword_ai/
├── __init__.py                 # Package root with main exports
├── config/                      # Configuration management
│   └── __init__.py             # Config class and global instance
├── core/                        # Core system components
│   ├── __init__.py             # Core exports
│   ├── agent.py                # Main orchestrator (HalfSwordAgent)
│   ├── actor.py                # Real-time inference loop (ActorProcess)
│   ├── learner.py              # Background training (LearnerProcess)
│   ├── model.py                # Neural network architecture
│   ├── dqn_model.py            # DQN-specific model implementation
│   ├── environment.py          # Gym environment wrapper
│   └── error_handler.py       # Error detection and recovery
├── perception/                  # Vision and detection
│   ├── __init__.py             # Perception exports
│   ├── vision.py               # Screen capture, memory reading
│   ├── yolo_detector.py        # YOLO object detection
│   ├── yolo_self_learning.py   # YOLO self-learning system
│   ├── screen_reward_detector.py # Screen-based reward detection
│   ├── ocr_reward_tracker.py   # OCR-based score tracking
│   └── terminal_state_detector.py # Death detection
├── learning/                    # Learning components
│   ├── __init__.py             # Learning exports
│   ├── replay_buffer.py        # Prioritized experience replay
│   ├── model_tracker.py        # Training progress tracking
│   ├── human_recorder.py        # Human action recording
│   └── reward_shaper.py         # Reward shaping and curriculum
├── input/                       # Input handling
│   ├── __init__.py             # Input exports
│   ├── input_mux.py            # Input multiplexer (human/bot switching)
│   ├── kill_switch.py           # Emergency stop mechanism
│   ├── direct_input.py          # DirectInput API wrapper
│   ├── gesture_engine.py        # Smooth gesture generation
│   └── action_discretizer.py   # Action discretization for DQN
├── monitoring/                  # Monitoring and observability
│   ├── __init__.py             # Monitoring exports
│   ├── performance_monitor.py  # Performance metrics tracking
│   ├── watchdog.py             # Game state monitoring
│   └── dashboard/               # Web dashboard
│       ├── __init__.py
│       ├── server.py            # Flask dashboard server
│       ├── dashboard_templates/ # HTML templates
│       └── dashboard_static/    # Static assets (CSS, JS)
├── llm/                         # LLM integration (optional)
│   ├── __init__.py             # LLM exports
│   └── ollama_integration.py   # Ollama/Qwen integration
├── utils/                       # Shared utilities
│   ├── __init__.py             # Utils exports
│   ├── logger.py               # Logging utilities
│   ├── pretty_logger.py        # Colored terminal logger
│   ├── file_utils.py            # File I/O utilities
│   ├── time_utils.py            # Time-related utilities
│   ├── math_utils.py            # Mathematical utilities
│   └── process_utils.py        # Process management utilities
└── tools/                       # Development tools
    ├── dataset_builder.py      # Dataset building utilities
    ├── dataset_utils.py         # Dataset manipulation
    ├── find_memory_patterns.py # Memory pattern finding
    └── visualize_model.py      # Model visualization

docs/                            # Documentation
├── guides/                      # User guides
│   ├── HALF_SWORD_CONTROLS.md
│   ├── INTERCEPTION_INSTALL.md
│   ├── DATASET_GUIDE.md
│   ├── QUICK_START.md
│   └── Rewards for Half Sword Bot Learning.txt
├── integration/                 # Integration guides
│   ├── SCRIMBRAIN_INTEGRATION.md
│   └── Guide To Scrimbrain Half Sword Integration.txt
├── status/                      # Status and changelog documents
│   ├── IMPLEMENTATION_STATUS.md
│   ├── LAUNCH_STATUS.md
│   ├── LLM_REMOVAL_SUMMARY.md
│   ├── PLACEHOLDER_DATA_REMOVAL.md
│   └── ATTACK_SWING_FIX.md
├── ARCHITECTURE.md              # Architecture documentation
├── MODULAR_STRUCTURE.md         # Modular structure guide
├── CURSOR_AGENT_GUIDE.md        # Cursor AI agent guide
├── PERFORMANCE_IMPROVEMENTS.md  # Performance notes
├── FIXES_APPLIED.md             # Fix documentation
├── LOG_IMPROVEMENTS.md          # Logging improvements
├── SYSTEM_ALIGNMENT.md          # System alignment notes
└── ORGANIZATION.md              # This file

drivers/                         # Driver files
└── interception/                # Interception driver
    ├── interception_driver/    # Driver files
    ├── Interception-master/    # Source code
    ├── interception_driver.zip # Driver archive
    └── install_interception_driver.bat # Installation script

scripts/                         # Utility scripts
├── start_agent.bat             # Windows batch launcher
├── start_agent.ps1              # PowerShell launcher
├── start_agent.py               # Python launcher
├── install_interception.py      # Interception installer
├── check_interception.py        # Interception checker
├── build_dataset.py             # Dataset builder
└── inspect_dataset.py           # Dataset inspector

tests/                           # Test files
├── test_system.py              # System integration tests
├── test_input_injection.py     # Input injection tests
├── test_kill_switch.py         # Kill switch tests
├── safety_check.py             # Safety verification
├── alignment_check.py          # System alignment checks
└── verify_integration.py       # Integration verification

memory-bank/                     # AI context persistence
├── projectbrief.md             # Project goals and constraints
├── productContext.md           # User stories and workflows
├── systemPatterns.md           # Architecture patterns
├── activeContext.md            # Current focus
├── progress.md                 # Completed features
└── decisionLog.md              # Architectural decisions
```

## Module Organization Principles

### 1. **Single Responsibility**
Each module has a clear, single responsibility:
- `core/`: Orchestration and main processes
- `perception/`: Vision and detection
- `learning/`: Learning algorithms and data structures
- `input/`: Input handling and control
- `monitoring/`: Observability and metrics
- `utils/`: Shared utilities

### 2. **Clear Public API**
Each module exposes a clean public API through `__init__.py`:
```python
# Example: half_sword_ai/core/__init__.py
from half_sword_ai.core.agent import HalfSwordAgent
from half_sword_ai.core.actor import ActorProcess
__all__ = ['HalfSwordAgent', 'ActorProcess', ...]
```

### 3. **Consolidated Code**
Related functionality is kept together:
- All vision code in `perception/`
- All learning code in `learning/`
- All input code in `input/`
- Avoids excessive file proliferation

### 4. **Documentation Organization**
Documentation is organized by purpose:
- `docs/guides/`: User-facing guides
- `docs/integration/`: Integration documentation
- `docs/status/`: Status updates and changelogs
- Root `docs/`: Architecture and technical docs

## Import Patterns

### Preferred Import Style
```python
# Use module-level imports from __init__.py
from half_sword_ai.core import HalfSwordAgent, ActorProcess
from half_sword_ai.perception import ScreenCapture, YOLODetector
from half_sword_ai.learning import PrioritizedReplayBuffer
from half_sword_ai.input import InputMultiplexer, KillSwitch
from half_sword_ai.monitoring import PerformanceMonitor
from half_sword_ai.utils import setup_logger, Timer
from half_sword_ai.config import config
```

### Direct Imports (when needed)
```python
# For specific implementations
from half_sword_ai.core.model import create_model
from half_sword_ai.perception.vision import ScreenCapture
```

## File Naming Conventions

- **Modules**: `snake_case.py` (e.g., `yolo_detector.py`)
- **Classes**: `PascalCase` (e.g., `HalfSwordAgent`)
- **Functions**: `snake_case` (e.g., `capture_screen`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `CAPTURE_WIDTH`)
- **Private**: Leading underscore (e.g., `_internal_method`)

## Code Organization Rules

1. **One Major Class Per File**: Each major class gets its own file
2. **Consolidated Utilities**: Related utilities stay together
3. **Type Hints**: All function signatures have type hints
4. **Docstrings**: Google-style docstrings for public APIs
5. **No Circular Dependencies**: Modules import only from lower-level modules

## Documentation Structure

- **README.md**: Main project overview and quick start
- **docs/ARCHITECTURE.md**: System architecture details
- **docs/guides/**: User guides and tutorials
- **docs/integration/**: Integration guides
- **docs/status/**: Status updates and changelogs
- **memory-bank/**: AI context for Cursor AI agent

## Testing Structure

- **tests/**: All test files
- **Integration tests**: Test full workflows
- **Unit tests**: Test individual components
- **No mock data**: Always use real game data

## Configuration

- **Single Source**: `half_sword_ai/config/__init__.py`
- **Global Instance**: `config` singleton
- **No Duplicates**: Old `config.py` removed

## Maintenance Guidelines

1. **Keep modules focused**: Don't mix responsibilities
2. **Update __init__.py**: When adding new public APIs
3. **Document changes**: Update relevant docs
4. **Follow conventions**: Stick to naming and structure patterns
5. **Test imports**: Ensure imports work from package root

