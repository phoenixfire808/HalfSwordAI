# Project Brief: Half Sword AI Agent

## Core Value Proposition

An autonomous reinforcement learning agent that learns to play Half Sword, a physics-based combat game, through continuous online training. The agent combines deep reinforcement learning (DQN/PPO), computer vision (YOLO object detection), and human-in-the-loop learning to master combat mechanics.

## Target Audience

- AI researchers and developers interested in game AI
- Reinforcement learning practitioners
- Computer vision enthusiasts
- Game automation developers

## Non-Negotiable Constraints

1. **Modular Architecture**: All code must be organized in the `half_sword_ai/` package with clear separation of concerns
2. **Real-time Performance**: Must operate at game frame rates (30-60 FPS) without lag
3. **Safety First**: Emergency kill switch (F8) must always be functional
4. **Python 3.11**: Strict requirement - use `.venv311` environment
5. **No Synthetic Data**: Always use real game data, never mock or simulated data
6. **Consolidated Code**: Keep files consolidated, avoid excessive scripts
7. **Live Mode Only**: System runs in live mode, not demo/test mode
8. **No LLM Dependencies**: Avoid freqai and similar frameworks

## Core Technologies

- **Deep Learning**: PyTorch (CNN + MLP architecture)
- **Computer Vision**: YOLO (Ultralytics) for object detection
- **Reinforcement Learning**: DQN for discrete actions, PPO for continuous
- **Input Control**: pydirectinput, pyautogui, pynput
- **Monitoring**: Flask dashboard for real-time metrics
- **Game Integration**: Screen capture + memory reading for state observation

## Success Criteria

- Agent learns effective combat strategies through self-play
- Human-in-the-loop recording improves learning efficiency
- Real-time inference with <16ms latency per frame
- Continuous online learning without performance degradation
- Dashboard provides actionable insights into agent behavior

