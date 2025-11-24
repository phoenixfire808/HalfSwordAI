# Decision Log: Institutional Memory

## Architecture Decisions

### Decision: Modular Monolith over Microservices
**Date**: Initial architecture design
**Context**: Need for real-time performance with clear separation of concerns
**Decision**: Single Python package with modular structure
**Rationale**: 
- No IPC overhead for core components
- Easier debugging and development
- Can extract to separate processes if needed
**Alternatives Considered**: Microservices, monolithic single file
**Status**: Implemented

### Decision: Separate Actor and Learner Processes
**Date**: Performance optimization phase
**Context**: Training blocking inference, causing lag
**Decision**: Separate processes for inference (Actor) and training (Learner)
**Rationale**:
- Inference must be real-time (<16ms per frame)
- Training can be slower without affecting gameplay
- Process isolation prevents crashes from affecting each other
**Alternatives Considered**: Single process with threading, async training
**Status**: Implemented

### Decision: DQN for Discrete Actions, PPO for Continuous
**Date**: Algorithm selection
**Context**: Need to support both discrete (keyboard) and continuous (mouse) action spaces
**Decision**: DQN for discrete, PPO for continuous (configurable)
**Rationale**:
- DQN proven effective for discrete action spaces
- PPO handles continuous actions well
- ScrimBrain integration favors discrete actions
**Alternatives Considered**: Pure DQN, Pure PPO, SAC
**Status**: Implemented

### Decision: YOLO for Object Detection
**Date**: Computer vision integration
**Context**: Need real-time enemy/threat detection
**Decision**: Ultralytics YOLO v8
**Rationale**:
- Fast enough for 60 FPS gameplay
- Good accuracy out of the box
- Supports custom training for self-learning
**Alternatives Considered**: Faster R-CNN, SSD, custom CNN
**Status**: Implemented

### Decision: Flask for Dashboard
**Date**: Monitoring system design
**Context**: Need real-time metrics visualization
**Decision**: Flask with WebSocket support
**Rationale**:
- Lightweight, minimal overhead
- Easy to extend with custom endpoints
- Browser-based, no installation needed
**Alternatives Considered**: FastAPI, Django, standalone GUI
**Status**: Implemented

### Decision: Python 3.11 Requirement
**Date**: Environment setup
**Context**: User preference and performance requirements
**Decision**: Strict Python 3.11 requirement with `.venv311`
**Rationale**:
- User preference explicitly stated
- Performance optimizations in 3.11
- Type hint improvements
**Alternatives Considered**: Python 3.10, 3.12
**Status**: Enforced

### Decision: No Synthetic/Mock Data
**Date**: Testing strategy
**Context**: User requirement for real data only
**Decision**: Always use real game data, never mocks
**Rationale**:
- Real data ensures accurate testing
- Prevents overfitting to synthetic patterns
- User explicit requirement
**Alternatives Considered**: Mock data for unit tests, synthetic data generation
**Status**: Enforced

### Decision: Consolidated Code Structure
**Date**: Code organization
**Context**: User preference for minimal file proliferation
**Decision**: Keep related code together, avoid excessive scripts
**Rationale**:
- Easier to navigate and maintain
- Reduces import complexity
- User explicit preference
**Alternatives Considered**: Many small files, micro-modules
**Status**: Enforced

### Decision: Memory Bank Pattern for Cursor
**Date**: Cursor IDE integration
**Context**: Need for context persistence across AI sessions
**Decision**: Implement Memory Bank pattern with .cursor/rules
**Rationale**:
- Enables context-aware AI assistance
- Future-proofs project for new models
- Improves development efficiency
**Alternatives Considered**: Single .cursorrules file, no context system
**Status**: Implementing

## Technology Choices

### PyTorch over TensorFlow
**Rationale**: More Pythonic, better for research, easier debugging

### dxcam over mss/pyautogui for screen capture
**Rationale**: Faster on Windows, lower latency, better for real-time

### pydirectinput over pyautogui for input
**Rationale**: More reliable for game input, bypasses some OS restrictions

## Rejected Approaches

### freqai Framework
**Reason**: User explicit preference against it

### Virtual Environment (venv)
**Reason**: User preference for .venv311 specifically, no standard venv

### Bash Scripts
**Reason**: User preference - use PowerShell/batch instead

### Multiple Separate Scripts
**Reason**: User preference for consolidated code

