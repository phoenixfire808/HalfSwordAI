# AGENTS.md - Context for AI Assistants

## Project Overview

This is a **Half Sword AI Agent** - an autonomous reinforcement learning agent that learns to play Half Sword, a physics-based combat game. The system combines deep reinforcement learning (DQN/PPO), computer vision (YOLO object detection), and human-in-the-loop learning.

See `memory-bank/projectbrief.md` for detailed goals and constraints.

## Operational Commands

### Development
- **Install Dependencies**: `pip install -r requirements.txt` (use Python 3.11)
- **Run Agent**: `python main.py` or `scripts/start_agent.bat` (Windows)
- **Run Tests**: `python -m pytest tests/` (if pytest installed)
- **Start Dashboard**: Automatically starts with agent at http://localhost:5000

### Environment
- **Python Version**: 3.11 (strict requirement)
- **Virtual Environment**: `.venv311` (user preference)
- **Package Manager**: pip (standard Python)

## Code Style Summary

### Python Standards
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case for files/functions, PascalCase for classes
- **Imports**: Absolute imports from `half_sword_ai` package

### Architecture
- **Modular Structure**: Code organized in `half_sword_ai/` package
- **Consolidated Code**: Keep related code together, avoid excessive files
- **Configuration**: Use `half_sword_ai.config.config` singleton
- **No Synthetic Data**: Always use real game data, never mocks

### Key Constraints
- **Python 3.11**: Strict requirement
- **Real-time Performance**: <16ms latency per frame target
- **Safety**: Emergency kill switch (F8) must always work
- **No Mock Data**: Only real data from actual sources

## Documentation Architecture

This project uses a **Memory Bank** for context persistence across AI sessions.

### Memory Bank Files
- `memory-bank/projectbrief.md`: Core goals and constraints
- `memory-bank/productContext.md`: User stories and workflows
- `memory-bank/systemPatterns.md`: Architecture and conventions
- `memory-bank/activeContext.md`: Current focus and next steps
- `memory-bank/progress.md`: Completed features and milestones
- `memory-bank/decisionLog.md`: Architectural decisions and rationale

### Cursor Rules
- `.cursor/rules/memory-bank-manager.mdc`: Mandatory Memory Bank updates
- `.cursor/rules/workflow-protocol.mdc`: Plan/Act workflow enforcement
- `.cursor/rules/python-standards.mdc`: Python coding standards

### Critical Directives
1. **Always Update Memory Bank**: After completing tasks, update relevant Memory Bank files
2. **Read Active Context**: Check `memory-bank/activeContext.md` at session start
3. **Follow Architecture**: Refer to `memory-bank/systemPatterns.md` for patterns
4. **Document Decisions**: Record significant decisions in `memory-bank/decisionLog.md`

## Project Structure

```
half_sword_ai/          # Main package
├── config/            # Configuration management
├── core/              # Core agent components
├── perception/        # Vision and detection
├── learning/          # Learning components
├── input/             # Input handling
├── monitoring/        # Observability
├── llm/               # LLM integration
└── utils/             # Shared utilities

memory-bank/           # AI context persistence
.cursor/rules/         # Cursor IDE rules
tests/                 # Test files
models/                # Model checkpoints
logs/                  # Log files
data/                  # Training data
```

## Tech Stack

- **Python**: 3.11
- **PyTorch**: >=2.0.0 (neural networks)
- **YOLO**: Ultralytics >=8.0.0 (object detection)
- **Flask**: >=3.0.0 (dashboard)
- **OpenCV**: >=4.8.0 (image processing)
- **dxcam**: >=0.1.0 (Windows screen capture)

## Getting Started

1. **Read Memory Bank**: Start with `memory-bank/activeContext.md` to understand current state
2. **Check Architecture**: Review `memory-bank/systemPatterns.md` for patterns
3. **Follow Workflow**: Use Plan/Act protocol for complex tasks
4. **Update Documentation**: Always update Memory Bank after changes

## Important Notes

- This project uses **real data only** - no synthetic/mock data
- Code should be **consolidated** - avoid excessive file proliferation
- **Kill switch (F8)** must always be functional
- System runs in **live mode** - no demo/test modes
- **Python 3.11** is a strict requirement

