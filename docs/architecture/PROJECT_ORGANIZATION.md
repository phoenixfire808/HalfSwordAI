# Half Sword AI Agent - Complete Project Organization

## 📁 Directory Structure

```
half_sword_ai/
├── __init__.py                 # Package root with main exports
├── config/                      # Configuration management
│   └── __init__.py             # Config class and global instance
├── core/                        # Core system components
│   ├── __init__.py             # Core exports
│   ├── agent.py                 # Main orchestrator (HalfSwordAgent)
│   ├── actor.py                 # Real-time inference loop (ActorProcess)
│   ├── learner.py               # Background training (LearnerProcess)
│   ├── model.py                 # Neural network architecture
│   ├── dqn_model.py             # DQN-specific model implementation
│   ├── environment.py           # Gym environment wrapper
│   └── error_handler.py         # Error detection and recovery
├── perception/                  # Vision and detection
│   ├── __init__.py             # Perception exports
│   ├── vision.py               # Screen capture, memory reading, vision processor
│   ├── yolo_detector.py        # YOLO object detection
│   ├── yolo_self_learning.py   # YOLO self-learning system
│   ├── yolo_feature_extractor.py # YOLO feature extraction
│   ├── screen_reward_detector.py # Screen-based reward detection
│   ├── ocr_reward_tracker.py   # OCR-based score tracking
│   └── terminal_state_detector.py # Death detection
├── learning/                    # Learning components
│   ├── __init__.py             # Learning exports
│   ├── replay_buffer.py        # Prioritized experience replay
│   ├── model_tracker.py        # Training progress tracking
│   ├── human_recorder.py        # Human action recording
│   ├── autonomous_learner.py   # Autonomous learning manager
│   ├── reward_shaper.py         # Reward shaping and curriculum
│   ├── enhanced_reward_shaper.py # Enhanced reward shaper
│   ├── pattern_recognition.py   # Pattern recognition
│   ├── pattern_matcher.py       # Pattern matching
│   └── data_augmentation.py     # Data augmentation utilities
├── input/                       # Input handling
│   ├── __init__.py             # Input exports
│   ├── input_mux.py            # Input multiplexer (human/bot switching)
│   ├── kill_switch.py           # Emergency kill switch (F8)
│   ├── physics_controller.py   # Physics-based mouse control
│   ├── direct_input.py          # DirectInput input injection
│   ├── gesture_engine.py        # Gesture recognition
│   ├── movement_emulator.py     # Movement emulation
│   ├── movement_replicator.py   # Movement replication
│   └── action_discretizer.py    # Action discretization for DQN
├── monitoring/                  # Monitoring and observability
│   ├── __init__.py             # Monitoring exports
│   ├── performance_monitor.py  # Performance metrics tracking
│   ├── watchdog.py             # Game state monitoring
│   ├── gui_dashboard.py        # Unified GUI dashboard (tkinter)
│   ├── yolo_proof.py           # YOLO learning proof tracker
│   ├── yolo_overlay.py         # YOLO overlay (deprecated - integrated into GUI)
│   ├── data_verification.py     # Data verification utilities
│   ├── yolo_usage_verifier.py   # YOLO usage verification
│   └── dashboard/               # Web dashboard (legacy)
│       ├── __init__.py
│       ├── server.py
│       └── dashboard_templates/
├── tools/                       # Utility tools and scripts
│   ├── __init__.py
│   ├── dataset_builder.py       # Dataset building utilities
│   ├── half_sword_dataset_builder.py # Half Sword specific dataset builder
│   ├── dataset_utils.py         # Dataset utilities
│   ├── ue4ss_integration.py     # UE4SS integration
│   ├── historical_reward_shaper.py # Historical reward shaping
│   ├── find_memory_patterns.py  # Memory pattern finding
│   ├── visualize_model.py       # Model visualization
│   └── verify_learning.py       # Learning verification
├── utils/                       # Shared utilities
│   ├── __init__.py             # Utils exports
│   ├── safe_logger.py          # Safe logging (Windows Unicode handling)
│   ├── pretty_logger.py        # Pretty colored logging
│   ├── logger.py                # Basic logger utilities
│   ├── enhanced_logger.py      # Enhanced logging features
│   ├── terminal_formatter.py   # Terminal formatting
│   ├── file_utils.py            # File I/O utilities
│   ├── time_utils.py            # Time utilities
│   ├── math_utils.py            # Math utilities
│   ├── process_utils.py         # Process management utilities
│   ├── window_finder.py         # Window finding utilities
│   ├── metrics_reporter.py      # Metrics reporting
│   └── error_collector.py      # Error collection utilities
└── llm/                         # LLM integration (optional)
    ├── __init__.py
    └── ollama_integration.py    # Ollama/Qwen integration

scripts/                         # Utility scripts
├── start_agent.py               # Main agent launcher
├── start_agent.bat              # Windows batch launcher
├── start_agent.ps1              # PowerShell launcher
├── verify_learning.py           # Learning verification script
├── monitor_yolo_learning.py    # YOLO learning monitor
├── train_yolo_model.py          # YOLO model training
├── build_dataset.py             # Dataset building
└── [other utility scripts]

tests/                           # Test files
├── test_kill_switch.py          # Kill switch tests
├── test_input_injection.py      # Input injection tests
├── test_system.py               # System integration tests
└── [other test files]

docs/                            # Documentation
├── ARCHITECTURE.md              # System architecture
├── ORGANIZATION.md              # Project organization guide
├── MODULAR_STRUCTURE.md         # Modular structure details
├── guides/                      # User guides
│   ├── QUICK_START.md
│   ├── DATASET_GUIDE.md
│   └── [other guides]
├── integration/                 # Integration documentation
│   ├── SCRIMBRAIN_INTEGRATION.md
│   ├── UE4SS_INTEGRATION.md
│   └── [other integration docs]
└── status/                      # Status updates
    └── [status update files]

memory-bank/                     # AI context persistence
├── projectbrief.md              # Project goals and constraints
├── productContext.md            # User stories and workflows
├── systemPatterns.md            # Architecture patterns
├── activeContext.md             # Current focus
├── progress.md                  # Completed features
└── decisionLog.md               # Architectural decisions

data/                            # Data storage
├── models/                      # Model checkpoints
├── logs/                        # Log files
└── [other data directories]

models/                          # Model storage (symlink or copy)
logs/                            # Log storage
```

## 🎯 Module Organization Principles

### 1. **Single Responsibility**
Each module has one clear purpose:
- `core/`: System orchestration and main processes
- `perception/`: Vision and detection
- `learning/`: Learning algorithms and data management
- `input/`: Input handling and control
- `monitoring/`: Observability and dashboards
- `utils/`: Shared utilities

### 2. **Clear Dependencies**
- Lower-level modules don't depend on higher-level modules
- `utils/` has no dependencies on other modules
- `core/` depends on all other modules
- Modules import from `config` for configuration

### 3. **Consolidated Code**
- Related functionality kept together
- Avoids excessive file proliferation
- Utilities grouped by purpose

### 4. **Proper Exports**
Each `__init__.py` exports only public APIs:
- Main classes and functions
- Configuration objects
- Public constants

## 📦 Import Patterns

### Preferred Style
```python
# Use module-level imports from __init__.py
from half_sword_ai.core import HalfSwordAgent, ActorProcess
from half_sword_ai.perception import ScreenCapture, YOLODetector
from half_sword_ai.learning import PrioritizedReplayBuffer
from half_sword_ai.input import InputMultiplexer, KillSwitch
from half_sword_ai.monitoring import PerformanceMonitor, GUIDashboard
from half_sword_ai.config import config
```

### Direct Imports (when needed)
```python
# For specific implementations
from half_sword_ai.core.model import create_model
from half_sword_ai.perception.vision import ScreenCapture
```

## 📝 File Naming Conventions

- **Modules**: `snake_case.py` (e.g., `yolo_detector.py`)
- **Classes**: `PascalCase` (e.g., `HalfSwordAgent`)
- **Functions**: `snake_case` (e.g., `capture_screen`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `CAPTURE_WIDTH`)
- **Private**: Leading underscore (e.g., `_internal_method`)

## 🗂️ Documentation Organization

### Root Documentation
- `README.md`: Main project overview and quick start
- `AGENTS.md`: Context for AI assistants
- `requirements.txt`: Python dependencies

### docs/ Directory
- `ARCHITECTURE.md`: System architecture details
- `ORGANIZATION.md`: Project organization guide
- `MODULAR_STRUCTURE.md`: Modular structure details
- `guides/`: User-facing guides
- `integration/`: Integration documentation
- `status/`: Status updates and changelogs

### memory-bank/ Directory
- AI context persistence for cross-session continuity
- Project goals, patterns, and decisions

## 🧹 Cleanup Rules

### Removed Files
- Empty duplicate files (`error_aggregator.py` in both `core/` and `utils/`)
- Deprecated components (separate YOLO overlay window)

### Moved Files
- Test files from root → `tests/`
- Documentation files from root → `docs/`

### Organized Structure
- All scripts in `scripts/`
- All tests in `tests/`
- All docs in `docs/`
- All data in `data/` or `models/` or `logs/`

## ✅ Organization Checklist

- [x] Removed duplicate files
- [x] Organized test files
- [x] Organized documentation
- [x] Updated `__init__.py` files with proper exports
- [x] Created comprehensive organization document
- [x] Unified GUI interface (no separate windows)
- [x] Clear module boundaries
- [x] Proper import patterns

## 🚀 Quick Start

```python
from half_sword_ai.core import HalfSwordAgent

agent = HalfSwordAgent()
agent.initialize()
agent.start()
```

## 📚 Key Files

- **Entry Point**: `main.py` or `scripts/start_agent.py`
- **Main Agent**: `half_sword_ai/core/agent.py`
- **Configuration**: `half_sword_ai/config/__init__.py`
- **GUI Dashboard**: `half_sword_ai/monitoring/gui_dashboard.py`

## 🔄 Maintenance

When adding new code:
1. Place in appropriate module directory
2. Update `__init__.py` with exports
3. Follow naming conventions
4. Add docstrings
5. Update this document if structure changes

