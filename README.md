# Half Sword AI Agent ⚔️

**Autonomous reinforcement learning agent for Half Sword combat game**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project implements a complete AI agent system that learns to play [Half Sword](https://halfswordgames.com/), a physics-based medieval combat game. The system combines deep reinforcement learning (DQN/PPO), computer vision (YOLO), and human-in-the-loop learning (DAgger) to master the complex physics-based combat mechanics.

This is a machine learning project for Half Sword which utilizes a human-in-the-loop component in addition to machine learning.

## 🎯 Features

- **Real-time Reinforcement Learning**: Continuous online training with DQN/PPO algorithms
- **Human-in-the-Loop**: Learn from human gameplay demonstrations using DAgger
- **Computer Vision**: YOLO object detection with self-learning capabilities
- **Strategic AI**: Ollama Qwen integration for high-level combat strategy
- **Live Dashboard**: Real-time monitoring at http://localhost:5000
- **Modular Architecture**: Clean, scalable codebase organized by functionality
- **Safety First**: Emergency kill switch (F8) always functional
- **Physics-Based**: Handles active ragdoll dynamics and half-swording mechanics

## 🚀 Quick Start

### Prerequisites

- **Python 3.11** (strict requirement)
- **Windows 10/11** (for game integration)
- **Half Sword Demo** installed via Steam
- **Git** for cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/phoenixfire808/HalfSwordAI.git
   cd HalfSwordAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the agent**
   
   **Windows (Recommended)**: Double-click `scripts/start_agent.bat`
   
   **Command Line**: 
   ```bash
   python main.py
   ```

4. **Access the dashboard**
   
   Open http://localhost:5000 in your browser to monitor the agent's performance.

## 📁 Project Structure

```
half_sword_ai/          # Main package
├── core/              # Core agent, actor, learner, model
│   ├── agent.py       # Main orchestrator
│   ├── actor.py       # Real-time inference loop
│   ├── learner.py     # Background training process
│   └── model.py       # Neural network architecture
├── config/            # Configuration management
├── input/             # Input multiplexer, kill switch
├── learning/          # Replay buffer, human recorder, model tracker
├── llm/               # Ollama Qwen integration
├── monitoring/        # Performance monitor, dashboard, watchdog
├── perception/        # Vision, YOLO detection, screen rewards
├── tools/             # Model visualization tools
└── utils/             # Utility functions

scripts/               # Launcher scripts
docs/                  # Comprehensive documentation
├── guides/           # User guides
├── integration/      # Integration docs
├── status/           # Status updates
└── *.md              # Architecture & design docs
tests/                 # Test files
models/                # Model checkpoints and weights
memory-bank/           # AI context for Cursor AI
```

## 🎮 Controls

- **F8**: Emergency kill switch (immediately stops all bot actions)
- **Mouse Movement**: Take manual control
- **Stop Mouse**: Return to bot control (0.5s delay)
- **Ctrl+C**: Graceful shutdown

## 🧠 How It Works

### Architecture Overview

The agent operates in a multi-process architecture:

1. **Actor Process**: Runs real-time inference, capturing game state and executing actions
2. **Learner Process**: Trains the neural network on collected experiences in the background
3. **Main Agent**: Orchestrates both processes and manages shared resources

### Learning Pipeline

1. **Observation**: Screen capture + YOLO object detection + memory reading
2. **Action Selection**: Neural network predicts optimal action (epsilon-greedy exploration)
3. **Execution**: Input injection via pydirectinput/pyautogui
4. **Reward Calculation**: Based on damage dealt, edge alignment, gap targeting
5. **Training**: Experience stored in replay buffer, learner trains periodically

### Key Technologies

- **PyTorch**: Deep learning framework for neural networks
- **YOLO (Ultralytics)**: Real-time object detection
- **Flask**: Web dashboard for monitoring
- **dxcam**: High-performance screen capture
- **pydirectinput**: Low-level input injection

## 📚 Documentation

### Quick Start & Guides
- **[docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)** - Detailed quick start guide
- **[docs/guides/HALF_SWORD_CONTROLS.md](docs/guides/HALF_SWORD_CONTROLS.md)** - Complete control reference
- **[docs/guides/INTERCEPTION_INSTALL.md](docs/guides/INTERCEPTION_INSTALL.md)** - Driver installation guide
- **[docs/guides/DATASET_GUIDE.md](docs/guides/DATASET_GUIDE.md)** - Dataset building guide

### Architecture & Design
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture overview
- **[docs/MODULAR_STRUCTURE.md](docs/MODULAR_STRUCTURE.md)** - Code organization principles
- **[docs/ORGANIZATION.md](docs/ORGANIZATION.md)** - Project organization guide
- **[docs/INDEX.md](docs/INDEX.md)** - Complete documentation index

### Integration
- **[docs/integration/SCRIMBRAIN_INTEGRATION.md](docs/integration/SCRIMBRAIN_INTEGRATION.md)** - ScrimBrain integration details

## 🔧 Configuration

Configuration is managed through `half_sword_ai.config.config`. Key settings:

- **Model Type**: DQN (discrete actions) or PPO (continuous)
- **Frame Size**: 224x224 (ScrimBrain standard)
- **Frame Stack**: 4 frames for temporal context
- **Frame Skip**: 2 frames for physics stability
- **Learning Rate**: Adaptive based on performance

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

Key test files:
- `test_kill_switch.py` - Verify emergency stop functionality
- `test_system.py` - System integration tests
- `safety_check.py` - Safety and error handling verification

## 🐛 Troubleshooting

### Common Issues

1. **Game not detected**: Ensure Half Sword is running and visible on screen
2. **Input not working**: Check if Interception driver is installed (see docs/guides/INTERCEPTION_INSTALL.md)
3. **Performance issues**: Reduce frame size or increase frame skip in config
4. **Kill switch not working**: Press F8 multiple times, check logs for errors

### Debug Mode

Enable verbose logging by setting environment variable:
```bash
set HALF_SWORD_AI_DEBUG=1
python main.py
```

## 🤝 Contributing

This is a research project focused on reinforcement learning for physics-based games. Contributions are welcome!

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

- **Half Sword Games** for creating an amazing physics-based combat game
- **Ultralytics** for YOLO object detection
- **PyTorch** team for the deep learning framework
- **OpenAI** for inspiration on RL architectures

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Ensure you comply with Half Sword's terms of service when using this agent.
