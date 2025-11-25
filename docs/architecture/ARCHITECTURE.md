# Half Sword AI - Modular Architecture Guide

## Overview

The system has been refactored into a clean, modular architecture for better organization, maintainability, and scalability.

## Directory Structure

```
half_sword_ai/
├── __init__.py                    # Package root
├── config/                        # Configuration
│   └── __init__.py               # Config class and global instance
├── core/                          # Core system components
│   ├── __init__.py
│   ├── agent.py                   # Main orchestrator
│   ├── actor.py                   # Real-time inference loop
│   ├── learner.py                 # Background training
│   └── model.py                   # Neural network architecture
├── perception/                     # Perception and vision
│   ├── __init__.py
│   ├── vision.py                  # Screen capture, memory, vision processor
│   ├── yolo_detector.py          # Object detection
│   └── yolo_self_learning.py     # Self-learning YOLO
├── learning/                       # Learning components
│   ├── __init__.py
│   ├── replay_buffer.py           # Experience replay
│   ├── model_tracker.py           # Training tracking
│   └── human_recorder.py          # Human action recording
├── input/                          # Input handling
│   ├── __init__.py
│   ├── input_mux.py               # Input multiplexer
│   └── kill_switch.py             # Emergency stop
├── monitoring/                     # Monitoring and observability
│   ├── __init__.py
│   ├── performance_monitor.py    # Performance metrics
│   ├── watchdog.py                # Game state monitoring
│   └── dashboard/                 # Web dashboard
│       ├── __init__.py
│       ├── server.py
│       ├── dashboard_templates/
│       └── dashboard_static/
└── llm/                            # LLM integration
    ├── __init__.py
    └── ollama_integration.py      # Ollama/Qwen API
```

## Module Responsibilities

### Core (`half_sword_ai.core`)
- **Agent**: Orchestrates all components, manages lifecycle
- **Actor**: Real-time inference at game frame rate
- **Learner**: Continuous online training in background
- **Model**: Neural network architecture

### Perception (`half_sword_ai.perception`)
- **Vision**: Screen capture, memory reading, vision processing
- **YOLO Detector**: Object detection for enemies/threats
- **YOLO Self-Learning**: Self-labeling and improvement

### Learning (`half_sword_ai.learning`)
- **Replay Buffer**: Prioritized experience storage
- **Model Tracker**: Training progress and checkpoints
- **Human Recorder**: Captures human demonstrations

### Input (`half_sword_ai.input`)
- **Input Mux**: Seamless human/bot control switching
- **Kill Switch**: Emergency stop mechanism

### Monitoring (`half_sword_ai.monitoring`)
- **Performance Monitor**: Comprehensive metrics tracking
- **Watchdog**: Game state monitoring and recovery
- **Dashboard**: Real-time web-based monitoring

### LLM (`half_sword_ai.llm`)
- **Ollama Integration**: Strategic decision-making via Qwen

## Usage Examples

### Basic Usage
```python
from half_sword_ai.core.agent import HalfSwordAgent

agent = HalfSwordAgent()
agent.initialize()
agent.start()
```

### Importing Specific Components
```python
# Using module __init__ exports
from half_sword_ai.core import HalfSwordAgent, ActorProcess
from half_sword_ai.perception import ScreenCapture, YOLODetector
from half_sword_ai.learning import PrioritizedReplayBuffer
from half_sword_ai.input import InputMultiplexer, KillSwitch
from half_sword_ai.monitoring import PerformanceMonitor, DashboardServer
from half_sword_ai.llm import OllamaQwenAgent

# Or direct imports
from half_sword_ai.core.model import create_model
from half_sword_ai.perception.vision import VisionProcessor
```

### Configuration
```python
from half_sword_ai.config import config

# Modify configuration
config.CAPTURE_FPS = 60
config.LEARNING_RATE = 1e-3
```

## Benefits

1. **Clear Organization**: Logical grouping by functionality
2. **Easy Navigation**: Find components quickly
3. **Scalability**: Add features without cluttering
4. **Maintainability**: Isolated, testable modules
5. **Reusability**: Components can be used independently
6. **Documentation**: Structure documents architecture

## Migration Notes

- Old flat structure files moved to appropriate modules
- All imports updated to use new paths
- `main.py` is now a simple entry point
- Backward compatibility maintained through proper exports

## Entry Point

Run the system from the project root:
```bash
python main.py
```

The entry point imports from the modular structure and initializes the agent.

