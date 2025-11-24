# Project Structure

## Root Directory

```
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Main documentation
├── .gitignore                # Git ignore rules
└── PROJECT_STRUCTURE.md      # This file
```

## Main Package

```
half_sword_ai/                 # Main package
├── __init__.py
├── config/                    # Configuration
│   └── __init__.py
├── core/                      # Core components
│   ├── actor.py              # Actor process (inference loop)
│   ├── agent.py              # Main agent orchestrator
│   ├── error_handler.py      # Error detection and recovery
│   ├── learner.py            # Learner process (training)
│   └── model.py              # Neural network model
├── input/                     # Input handling
│   ├── input_mux.py          # Input multiplexer
│   └── kill_switch.py        # Emergency kill switch
├── learning/                  # Learning components
│   ├── human_recorder.py     # Human action recording
│   ├── model_tracker.py      # Model training tracker
│   └── replay_buffer.py      # Prioritized replay buffer
├── llm/                       # LLM integration
│   └── ollama_integration.py # Ollama Qwen agent
├── monitoring/                # Monitoring and dashboard
│   ├── dashboard/
│   │   ├── server.py         # Flask dashboard server
│   │   └── dashboard_templates/
│   ├── performance_monitor.py # Performance tracking
│   └── watchdog.py           # Game state watchdog
├── perception/                # Perception layer
│   ├── screen_reward_detector.py # Screen-based rewards
│   ├── vision.py             # Screen capture and memory
│   ├── yolo_detector.py      # YOLO object detection
│   └── yolo_self_learning.py # YOLO self-learning
├── tools/                     # Utility tools
│   └── visualize_model.py    # Model visualization
└── utils/                     # Utility functions
    ├── file_utils.py
    ├── logger.py
    ├── math_utils.py
    ├── process_utils.py
    └── time_utils.py
```

## Organized Directories

```
docs/                          # Documentation
├── *.txt                      # Text documentation files
├── *.md                       # Markdown documentation
└── README.md                  # Documentation index

scripts/                       # Launcher scripts
├── start_agent.bat           # Windows batch launcher (recommended)
├── start_agent.py            # Python launcher
├── start_agent.ps1           # PowerShell launcher
├── start_agent.vbs           # VBS launcher
└── README.md                 # Launcher documentation

tests/                         # Test and utility scripts
├── test_*.py                 # Test scripts
├── verify_*.py               # Verification scripts
├── setup_*.py                # Setup scripts
└── README.md                 # Test documentation

models/                        # Model files
└── yolov8n.pt                # YOLO model weights

archive/                       # Archived files
└── old_code/                 # Old code files (legacy)
```

## Generated Directories

```
logs/                          # Log files (auto-generated)
data/                          # Training data (auto-generated)
models/checkpoints/            # Model checkpoints (auto-generated)
```

## File Organization Principles

1. **Root directory**: Only essential files (main.py, README.md, requirements.txt)
2. **half_sword_ai/**: Main package with modular structure
3. **docs/**: All documentation files
4. **scripts/**: Launcher scripts
5. **tests/**: Test and utility scripts
6. **models/**: Model weights and checkpoints
7. **archive/**: Old/legacy code for reference

## Launching the Agent

The easiest way is to use the launcher scripts in `scripts/`:

- **Windows**: Double-click `scripts/start_agent.bat`
- **Command line**: `python main.py`
- **From scripts**: `python scripts/start_agent.py`

All launchers automatically navigate to the correct directory.

