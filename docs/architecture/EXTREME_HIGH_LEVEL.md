# Extreme High-Level Organization
## Half Sword AI Agent - System Overview

---

## 🎯 ONE-SENTENCE SUMMARY

**Autonomous RL agent that learns Half Sword combat through continuous online training using vision (YOLO), deep learning (PyTorch), and human demonstrations.**

---

## 🏗️ SYSTEM ARCHITECTURE (5-Layer View)

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: GAME                           │
│                  Half Sword (UE5)                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Screen/Memory
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 2: PERCEPTION                      │
│         Vision Capture → YOLO Detection → Features          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ State Vector
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 3: INTELLIGENCE                    │
│    Actor (Inference) ←→ Model ←→ Learner (Training)         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Actions
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 4: CONTROL                         │
│         Input Multiplexer → Game Input Injection            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Feedback Loop
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: OBSERVABILITY                   │
│         Monitoring → Dashboard → Logging                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 CORE EXECUTION FLOW

```
START
  │
  ├─→ Initialize Components
  │   ├─ Vision System (Screen Capture + YOLO)
  │   ├─ Neural Network Model
  │   ├─ Input Controller
  │   └─ Learning System
  │
  ├─→ Launch Two Processes
  │   ├─ ACTOR PROCESS (Real-time)
  │   │   └─ Loop: See → Think → Act → Collect
  │   │
  │   └─ LEARNER PROCESS (Background)
  │       └─ Loop: Sample → Train → Update
  │
  └─→ Monitor & Control
      ├─ Performance Tracking
      ├─ Kill Switch (F8)
      └─ Dashboard
```

---

## 📦 MODULE BREAKDOWN (8 Categories)

| Category | Purpose | Key Files |
|----------|---------|-----------|
| **CORE** | Orchestration & Execution | `agent.py`, `actor.py`, `learner.py`, `model.py` |
| **PERCEPTION** | Vision & Detection | `vision.py`, `yolo_detector.py` |
| **INPUT** | Control & Actions | `input_mux.py`, `kill_switch.py`, `physics_controller.py` |
| **LEARNING** | Training & Memory | `replay_buffer.py`, `reward_shaper.py`, `human_recorder.py` |
| **MONITORING** | Observability | `performance_monitor.py`, `dashboard/`, `watchdog.py` |
| **TOOLS** | Development Tools | `dataset_builder.py`, `ue4ss_integration.py` |
| **UTILS** | Shared Utilities | `logger.py`, `error_collector.py`, `math_utils.py` |
| **CONFIG** | Configuration | `config/__init__.py` (singleton) |

---

## 🎯 KEY COMPONENTS (Top 10)

1. **HalfSwordAgent** (`core/agent.py`)
   - Main orchestrator, manages everything

2. **ActorProcess** (`core/actor.py`)
   - Real-time inference loop (<16ms per frame)

3. **LearnerProcess** (`core/learner.py`)
   - Background training, updates model

4. **Neural Network** (`core/model.py`)
   - CNN + MLP architecture (DQN/PPO)

5. **ScreenCapture** (`perception/vision.py`)
   - Captures game frames (DXCam/MSS)

6. **YOLODetector** (`perception/yolo_detector.py`)
   - Detects enemies/threats in frames

7. **InputMultiplexer** (`input/input_mux.py`)
   - Switches between human/bot control

8. **ReplayBuffer** (`learning/replay_buffer.py`)
   - Stores experiences for training

9. **RewardShaper** (`learning/reward_shaper.py`)
   - Calculates rewards from game state

10. **KillSwitch** (`input/kill_switch.py`)
    - Emergency stop (F8 key)

---

## 🔀 DATA FLOW (Simplified)

```
GAME STATE
    │
    ├─→ Screen Capture ──→ YOLO ──→ Features
    │                                    │
    │                                    ▼
    │                              STATE VECTOR
    │                                    │
    │                                    ▼
    │                              NEURAL NETWORK
    │                                    │
    │                                    ▼
    │                                ACTIONS
    │                                    │
    │                                    ▼
    └─← Input Injection ←───────────────┘
            │
            ▼
        GAME STATE (updated)
            │
            ▼
        REWARD CALCULATION
            │
            ▼
        EXPERIENCE STORAGE
            │
            ▼
        MODEL TRAINING
            │
            ▼
        MODEL UPDATE
            │
            └─→ (loop continues)
```

---

## ⚙️ PROCESS ARCHITECTURE

```
MAIN PROCESS (HalfSwordAgent)
│
├─→ ACTOR PROCESS (Separate Process)
│   ├─ Purpose: Real-time inference
│   ├─ Priority: Low latency (<16ms)
│   └─ Loop: Capture → Infer → Act
│
├─→ LEARNER PROCESS (Separate Process)
│   ├─ Purpose: Model training
│   ├─ Priority: Throughput (can be slower)
│   └─ Loop: Sample → Train → Update
│
└─→ MONITORING (Threads)
    ├─ Performance tracking
    ├─ Dashboard server
    └─ Watchdog
```

**Why Separate Processes?**
- Actor must be fast (game frame rate)
- Learner can be slower (background training)
- Isolation prevents crashes from affecting inference

---

## 🎨 DESIGN PATTERNS

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Singleton** | `config/` | Single configuration source |
| **Process Separation** | `core/actor.py`, `core/learner.py` | Real-time performance |
| **Multiplexer** | `input/input_mux.py` | Human/bot switching |
| **Observer** | `monitoring/` | System observability |
| **Strategy** | `learning/reward_shaper.py` | Reward calculation strategies |

---

## 📊 TECHNOLOGY STACK

```
┌─────────────────────────────────────┐
│         APPLICATION LAYER           │
│  Python 3.11 + Modular Architecture │
└─────────────────────────────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌──────────┐
│ PyTorch │   │  YOLO    │
│  (RL)   │   │ (Vision) │
└─────────┘   └──────────┘
    │               │
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌──────────┐
│  Game   │   │  System  │
│ Capture │   │  Input   │
└─────────┘   └──────────┘
```

---

## 🚀 EXECUTION SEQUENCE

```
1. START
   └─→ main.py

2. INITIALIZE
   └─→ HalfSwordAgent.initialize()
       ├─ Load config
       ├─ Create model
       ├─ Setup vision
       ├─ Setup input
       └─ Setup learning

3. LAUNCH
   └─→ HalfSwordAgent.start()
       ├─ Start Actor Process
       ├─ Start Learner Process
       ├─ Start Dashboard
       └─ Enable Kill Switch

4. RUN
   ├─ Actor: See → Think → Act (loop)
   └─ Learner: Sample → Train → Update (loop)

5. STOP
   └─→ HalfSwordAgent.stop()
       ├─ Stop processes
       ├─ Save checkpoints
       └─ Cleanup
```

---

## 🔑 KEY CONSTRAINTS

- **Python 3.11** (strict requirement)
- **Real-time Performance** (<16ms inference latency)
- **No Synthetic Data** (real game data only)
- **Kill Switch** (F8 always functional)
- **Modular Architecture** (consolidated code)
- **Live Mode Only** (no demo/test modes)

---

## 📈 SCALABILITY MODEL

```
Current: Single Machine, 2 Processes
    │
    ├─→ Scale Up: Better GPU/CPU
    │
    ├─→ Scale Out: Multiple Actor Processes
    │
    └─→ Scale Components: Extract modules to separate processes
```

---

## 🎯 SUCCESS METRICS

- **Latency**: <16ms per frame (60 FPS)
- **Learning**: Continuous improvement
- **Stability**: No crashes, graceful degradation
- **Safety**: Kill switch always works

---

## 📝 SUMMARY

**System Type**: Autonomous Reinforcement Learning Agent

**Architecture**: Modular Monolith with Process Separation

**Core Flow**: Perception → Intelligence → Control → Feedback

**Key Innovation**: Real-time inference + background training separation

**Total Size**: ~71 Python files, 8 major modules

**Complexity**: Medium-High (RL + Vision + Real-time + Learning)

