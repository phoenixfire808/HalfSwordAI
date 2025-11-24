# Half Sword AI - Modular Architecture

## Directory Structure

```
half_sword_ai/
├── __init__.py                 # Package initialization
├── config/
│   └── __init__.py             # Configuration (Config, config)
├── core/
│   ├── __init__.py             # Core exports
│   ├── agent.py                 # Main orchestrator (HalfSwordAgent)
│   ├── actor.py                 # Actor process (ActorProcess)
│   ├── learner.py               # Learner process (LearnerProcess)
│   └── model.py                 # Neural network (HalfSwordPolicyNetwork, create_model)
├── perception/
│   ├── __init__.py             # Perception exports
│   ├── vision.py                # Screen capture, memory reader, vision processor
│   ├── yolo_detector.py         # YOLO object detection
│   └── yolo_self_learning.py    # YOLO self-learning system
├── learning/
│   ├── __init__.py             # Learning exports
│   ├── replay_buffer.py         # Prioritized experience replay
│   ├── model_tracker.py         # Training progress tracking
│   └── human_recorder.py        # Human action recording
├── input/
│   ├── __init__.py             # Input exports
│   ├── input_mux.py            # Input multiplexer
│   └── kill_switch.py           # Emergency kill switch
├── monitoring/
│   ├── __init__.py             # Monitoring exports
│   ├── performance_monitor.py   # Performance metrics
│   ├── watchdog.py              # Game state watchdog
│   └── dashboard/
│       ├── __init__.py         # Dashboard exports
│       ├── server.py           # Dashboard server
│       ├── dashboard_templates/ # HTML templates
│       └── dashboard_static/   # Static assets
└── llm/
    ├── __init__.py             # LLM exports
    └── ollama_integration.py   # Ollama/Qwen integration
```

## Module Organization

### Core (`half_sword_ai.core`)
- **agent.py**: Main orchestrator that initializes and manages all components
- **actor.py**: Real-time inference loop running at game frame rate
- **learner.py**: Background training process with continuous online learning
- **model.py**: Neural network architecture (CNN + MLP)

### Perception (`half_sword_ai.perception`)
- **vision.py**: Screen capture (DXCam/MSS), memory reading (Pymem), vision processor
- **yolo_detector.py**: YOLO object detection for enemies and threats
- **yolo_self_learning.py**: Self-labeling and reward-based YOLO improvement

### Learning (`half_sword_ai.learning`)
- **replay_buffer.py**: Prioritized experience replay buffer
- **model_tracker.py**: Training progress and model checkpointing
- **human_recorder.py**: Human action recording for DAgger learning

### Input (`half_sword_ai.input`)
- **input_mux.py**: Input multiplexer for human/bot control switching
- **kill_switch.py**: Emergency stop mechanism

### Monitoring (`half_sword_ai.monitoring`)
- **performance_monitor.py**: Comprehensive performance tracking
- **watchdog.py**: Game state monitoring and restart handling
- **dashboard/**: Web-based real-time monitoring interface

### LLM (`half_sword_ai.llm`)
- **ollama_integration.py**: Ollama/Qwen API integration for strategic decision-making

## Usage

### Basic Usage
```python
from half_sword_ai.core.agent import HalfSwordAgent

agent = HalfSwordAgent()
agent.initialize()
agent.start()
```

### Importing Specific Components
```python
# Core components
from half_sword_ai.core import HalfSwordAgent, ActorProcess, LearnerProcess
from half_sword_ai.core.model import create_model

# Perception
from half_sword_ai.perception import ScreenCapture, MemoryReader, YOLODetector

# Learning
from half_sword_ai.learning import PrioritizedReplayBuffer, HumanActionRecorder

# Input
from half_sword_ai.input import InputMultiplexer, KillSwitch

# Monitoring
from half_sword_ai.monitoring import PerformanceMonitor, DashboardServer

# LLM
from half_sword_ai.llm import OllamaQwenAgent
```

### Configuration
```python
from half_sword_ai.config import config

# Access configuration
print(config.CAPTURE_FPS)
print(config.LEARNING_RATE)
```

## Benefits of Modular Structure

1. **Clear Organization**: Each component has a logical place
2. **Easy Navigation**: Find components quickly by category
3. **Scalability**: Add new features without cluttering root directory
4. **Maintainability**: Isolated modules are easier to test and modify
5. **Reusability**: Components can be imported independently
6. **Documentation**: Structure itself documents the system architecture

## Migration from Old Structure

The old flat structure has been reorganized:
- `main.py` → `half_sword_ai/core/agent.py` + new `main.py` (entry point)
- `neural_network.py` → `half_sword_ai/core/model.py`
- `actor_process.py` → `half_sword_ai/core/actor.py`
- `learner_process.py` → `half_sword_ai/core/learner.py`
- `perception_layer.py` → `half_sword_ai/perception/vision.py`
- All other files moved to appropriate modules

## Running the System

```bash
# From project root
python main.py
```

The entry point (`main.py`) imports from the modular structure and runs the agent.

