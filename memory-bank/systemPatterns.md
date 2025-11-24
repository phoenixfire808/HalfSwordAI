# System Patterns: Architectural Blueprint

## Architecture Overview

**Pattern**: Modular Monolith with Clear Separation of Concerns

The system is organized as a single Python package (`half_sword_ai/`) with distinct modules for different responsibilities. This balances maintainability with performance (no inter-process communication overhead for core components).

## Tech Stack & Versions

- **Python**: 3.11 (strict requirement - use `.venv311`)
- **PyTorch**: >=2.0.0 (neural network framework)
- **Ultralytics YOLO**: >=8.0.0 (object detection)
- **Flask**: >=3.0.0 (dashboard server)
- **OpenCV**: >=4.8.0 (image processing)
- **dxcam**: >=0.1.0 (Windows screen capture)
- **pymem**: >=1.10.0 (memory reading)

## Folder Structure

```
half_sword_ai/
├── config/              # Configuration management (Config class, global instance)
├── core/               # Core agent components
│   ├── agent.py       # Main orchestrator (HalfSwordAgent)
│   ├── actor.py       # Real-time inference loop (ActorProcess)
│   ├── learner.py     # Background training (LearnerProcess)
│   └── model.py       # Neural network architecture
├── perception/         # Vision and detection
│   ├── vision.py      # Screen capture, memory reading
│   ├── yolo_detector.py
│   └── yolo_self_learning.py
├── learning/          # Learning components
│   ├── replay_buffer.py
│   ├── model_tracker.py
│   └── human_recorder.py
├── input/             # Input handling
│   ├── input_mux.py
│   └── kill_switch.py
├── monitoring/        # Observability
│   ├── performance_monitor.py
│   ├── watchdog.py
│   └── dashboard/
├── llm/               # LLM integration (Ollama/Qwen)
└── utils/             # Shared utilities
```

## Design Patterns

### 1. Configuration Pattern
- **Single Source of Truth**: `half_sword_ai.config.Config` class
- **Global Instance**: `config` singleton accessed throughout codebase
- **Location**: `half_sword_ai/config/__init__.py`
- **Rationale**: Centralized configuration prevents inconsistencies

### 2. Process-Based Architecture
- **Actor Process**: Separate process for real-time inference (low latency)
- **Learner Process**: Separate process for training (can be slower)
- **Rationale**: Prevents training from blocking inference, ensures responsive gameplay

### 3. Input Multiplexer Pattern
- **Human/Bot Switching**: Seamless transition between control modes
- **Mouse Detection**: Automatic detection of manual override
- **Rationale**: Enables human-in-the-loop learning without restarting system

### 4. Modular Imports
- Each module has `__init__.py` with explicit exports
- Clear public API per module
- Rationale: Prevents circular dependencies, improves IDE support

## Conventions

### Naming Schemes
- **Files**: snake_case (e.g., `yolo_detector.py`)
- **Classes**: PascalCase (e.g., `HalfSwordAgent`)
- **Functions/Methods**: snake_case (e.g., `capture_screen`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CAPTURE_WIDTH`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Code Organization
- **One Class Per File**: Major classes get their own file
- **Consolidated Utilities**: Keep related utilities together, avoid excessive files
- **Type Hints**: Use Python type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public functions/classes

### Testing Strategy
- **Integration Tests**: Test full system workflows
- **Unit Tests**: Test individual components in isolation
- **No Mock Data**: Always use real game data in tests
- **Location**: `tests/` directory at project root

### Logging
- **Structured Logging**: Use Python `logging` module
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Location**: `logs/` directory
- **Rotation**: Log files rotated by date/size

### Error Handling
- **Graceful Degradation**: System continues operating when non-critical components fail
- **Error Recovery**: Automatic retry for transient failures
- **Kill Switch**: Always functional, even during errors
- **Error Logging**: All errors logged with full context

## Architectural Decisions

### Why Modular Monolith?
- **Performance**: No IPC overhead for core components
- **Simplicity**: Easier debugging than microservices
- **Scalability**: Can extract modules to separate processes if needed

### Why Separate Actor/Learner Processes?
- **Latency**: Inference must be real-time (<16ms per frame)
- **Throughput**: Training can be slower without affecting gameplay
- **Stability**: Training crashes don't affect inference

### Why YOLO for Object Detection?
- **Real-time**: Fast enough for 60 FPS gameplay
- **Accuracy**: Good detection of enemies/threats
- **Self-learning**: Can improve with custom training data

### Why Flask Dashboard?
- **Lightweight**: Minimal overhead
- **Real-time**: WebSocket support for live updates
- **Accessible**: Browser-based, no installation needed

## Performance Requirements

- **Inference Latency**: <16ms per frame (60 FPS target)
- **Training**: Can be slower, runs in background
- **Memory**: Efficient use of replay buffer (prioritized experience replay)
- **CPU/GPU**: Utilize GPU for neural network inference when available

## Security & Safety

- **Kill Switch**: F8 key always functional
- **Input Validation**: All inputs validated before execution
- **Error Boundaries**: Failures contained to prevent system-wide crashes
- **No External Network**: Agent operates locally (except optional Ollama LLM)

