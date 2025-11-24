# Product Context: Half Sword AI Agent

## User Stories

### Primary User: AI Developer/Researcher
- **As a developer**, I want the agent to learn combat strategies autonomously so that I can observe emergent behaviors
- **As a researcher**, I want to record human gameplay sessions so that the agent can learn from expert demonstrations
- **As a developer**, I want real-time performance metrics so that I can diagnose training issues
- **As a user**, I want an emergency kill switch so that I can immediately stop the agent if it behaves unexpectedly

## Feature Descriptions

### Core Features

1. **Autonomous Learning Agent**
   - Continuous online reinforcement learning
   - DQN for discrete action spaces (ScrimBrain integration)
   - PPO for continuous action spaces
   - Frame stacking for temporal awareness

2. **Computer Vision Pipeline**
   - Real-time screen capture (dxcam for Windows)
   - YOLO object detection for enemies/threats
   - Self-learning YOLO that improves detection over time
   - Screen-based reward detection via OCR

3. **Human-in-the-Loop Learning**
   - Record human gameplay sessions
   - Imitation learning from expert demonstrations
   - DAgger algorithm for combining human and agent data
   - Seamless switching between human and bot control

4. **Input Management**
   - Input multiplexer for human/bot control switching
   - Emergency kill switch (F8 key)
   - Mouse movement detection for manual override
   - Action space: discrete (keyboard) or continuous (mouse)

5. **Monitoring & Observability**
   - Flask dashboard at http://localhost:5000
   - Real-time performance metrics
   - Training progress visualization
   - Game state watchdog for error detection

6. **Error Handling & Recovery**
   - Automatic error detection
   - Graceful degradation
   - Recovery mechanisms for common failure modes

## User Workflows

### Workflow 1: Starting the Agent
1. Launch Half Sword game
2. Run `python main.py` or use `scripts/start_agent.bat`
3. Agent initializes and begins learning
4. Monitor dashboard at http://localhost:5000

### Workflow 2: Human-in-the-Loop Training
1. Start agent in learning mode
2. Press F8 to pause agent
3. Play game manually (agent records actions)
4. Press F8 again to resume agent control
5. Agent incorporates human data into training

### Workflow 3: Emergency Stop
1. Press F8 key at any time
2. Agent immediately stops all actions
3. System enters safe state
4. Can resume or shutdown gracefully

## Expected Behaviors

- Agent should explore combat mechanics autonomously
- Learning should improve performance over time
- Dashboard should show clear metrics (reward, loss, FPS)
- Kill switch should respond instantly (<100ms)
- System should handle game crashes gracefully

