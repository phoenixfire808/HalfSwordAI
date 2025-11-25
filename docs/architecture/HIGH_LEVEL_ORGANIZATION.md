# High-Level Program Organization
## Half Sword AI Agent - Complete System Architecture

---

## 🎯 SYSTEM OVERVIEW

**Half Sword AI Agent** is an autonomous reinforcement learning system that learns to play Half Sword (physics-based combat game) through continuous online training. The system combines deep RL, computer vision, and human-in-the-loop learning.

---

## 📊 ARCHITECTURAL LAYERS

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT & ORCHESTRATION                  │
│                         main.py                                 │
│                    HalfSwordAgent (core/agent.py)              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   PERCEPTION  │    │     CORE      │    │    INPUT      │
│   LAYER       │    │   EXECUTION   │    │   CONTROL     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   LEARNING    │    │  MONITORING   │    │     TOOLS     │
│   SYSTEM      │    │   & OBSERV    │    │   & UTILS     │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## 🏗️ MODULE HIERARCHY

### **LEVEL 1: ENTRY & ORCHESTRATION**

```
main.py
├── Unicode-safe stdout wrapper (Windows compatibility)
├── Safe logging initialization
└── HalfSwordAgent initialization & startup
```

**Responsibilities:**
- System entry point
- Environment setup (logging, encoding)
- Agent lifecycle management

---

### **LEVEL 2: CORE EXECUTION LAYER**

#### **`half_sword_ai/core/`** - System Brain

```
core/
├── agent.py          → Main orchestrator (HalfSwordAgent)
│   ├── Component initialization
│   ├── Process management (Actor/Learner)
│   ├── Lifecycle control (start/stop/shutdown)
│   └── Error recovery & monitoring
│
├── actor.py          → Real-time inference (ActorProcess)
│   ├── Frame capture loop
│   ├── Model inference (<16ms target)
│   ├── Action execution
│   └── Experience collection
│
├── learner.py        → Background training (LearnerProcess)
│   ├── Replay buffer sampling
│   ├── Model training (DQN/PPO)
│   ├── Model checkpointing
│   └── Training metrics
│
├── model.py          → Neural network architecture
│   ├── CNN encoder (vision processing)
│   ├── MLP head (action prediction)
│   ├── DQN discrete actions
│   └── PPO continuous actions
│
├── dqn_model.py      → DQN-specific implementation
├── environment.py    → RL environment wrapper
└── error_handler.py  → Centralized error handling
```

**Key Design Pattern:** Process-based separation (Actor/Learner) for real-time performance

---

### **LEVEL 2: PERCEPTION LAYER**

#### **`half_sword_ai/perception/`** - Eyes & Vision

```
perception/
├── vision.py              → Screen capture & memory reading
│   ├── ScreenCapture (DXCam/MSS)
│   ├── MemoryReader (Pymem)
│   └── VisionProcessor (frame preprocessing)
│
├── yolo_detector.py       → Object detection (YOLO)
│   ├── Enemy detection
│   ├── Threat identification
│   └── Bounding box extraction
│
├── yolo_self_learning.py  → Self-improving YOLO
│   ├── Auto-labeling
│   ├── Reward-based learning
│   └── Model fine-tuning
│
├── yolo_feature_extractor.py → Feature extraction from detections
├── screen_reward_detector.py  → Reward signal from screen
├── ocr_reward_tracker.py      → OCR-based reward tracking
└── terminal_state_detector.py → Game state from terminal
```

**Key Design Pattern:** Modular vision pipeline with self-learning capability

---

### **LEVEL 2: INPUT CONTROL LAYER**

#### **`half_sword_ai/input/`** - Hands & Control

```
input/
├── input_mux.py           → Input multiplexer
│   ├── Human/Bot switching
│   ├── Mouse detection (manual override)
│   └── Seamless mode transitions
│
├── kill_switch.py         → Emergency stop (F8)
│   ├── Global keyboard listener
│   ├── Immediate shutdown
│   └── Safety guarantee
│
├── direct_input.py        → Low-level input injection
├── gesture_engine.py      → Gesture-based actions
├── physics_controller.py  → Physics-based movement (PID)
├── movement_emulator.py   → Movement pattern emulation
├── movement_replicator.py → Human movement replication
└── action_discretizer.py  → Action space discretization
```

**Key Design Pattern:** Multiplexed input with human-in-the-loop capability

---

### **LEVEL 3: LEARNING SYSTEM**

#### **`half_sword_ai/learning/`** - Intelligence & Memory

```
learning/
├── replay_buffer.py           → Experience storage
│   ├── Prioritized experience replay
│   ├── Frame stacking
│   └── Efficient sampling
│
├── enhanced_reward_shaper.py  → Advanced reward shaping
│   ├── Frame-by-frame rewards
│   ├── Granular components (survival, engagement, etc.)
│   └── Reward normalization
│
├── reward_shaper.py           → Basic reward shaping
├── human_recorder.py          → Human demonstration capture
│   ├── Action recording
│   ├── DAgger integration
│   └── Dataset building
│
├── model_tracker.py           → Training tracking
│   ├── Checkpoint management
│   ├── Training metrics
│   └── Model versioning
│
├── autonomous_learner.py      → Autonomous learning logic
├── pattern_recognition.py     → Pattern detection
├── pattern_matcher.py         → Pattern matching
└── data_augmentation.py       → Data augmentation
```

**Key Design Pattern:** Modular reward system with human-in-the-loop learning

---

### **LEVEL 3: MONITORING & OBSERVABILITY**

#### **`half_sword_ai/monitoring/`** - Eyes on System

```
monitoring/
├── performance_monitor.py  → Performance metrics
│   ├── Latency tracking
│   ├── FPS monitoring
│   └── Resource usage
│
├── watchdog.py             → System watchdog
│   ├── Game state monitoring
│   ├── Crash detection
│   └── Auto-recovery
│
├── gui_dashboard.py        → GUI monitoring interface
├── dashboard/              → Web dashboard
│   ├── server.py          → Flask server
│   └── templates/         → HTML templates
│
├── yolo_overlay.py         → YOLO visualization overlay
├── yolo_proof.py           → YOLO verification
├── yolo_usage_verifier.py  → YOLO usage verification
└── data_verification.py    → Data quality checks
```

**Key Design Pattern:** Multi-layer monitoring (GUI + Web + Logs)

---

### **LEVEL 3: TOOLS & UTILITIES**

#### **`half_sword_ai/tools/`** - Development & Analysis Tools

```
tools/
├── dataset_builder.py           → Dataset construction
├── half_sword_dataset_builder.py → Advanced dataset builder
│   ├── Physics state extraction
│   ├── HEMA pose classification
│   ├── Edge alignment calculation
│   └── Gap target detection
│
├── historical_reward_shaper.py  → Historical reward functions
├── ue4ss_integration.py          → UE4SS game integration
│   ├── Lua scripting
│   ├── SDK generation
│   └── Function hooking
│
├── verify_learning.py            → Learning verification
├── visualize_model.py            → Model visualization
├── find_memory_patterns.py       → Memory pattern analysis
└── dataset_utils.py              → Dataset utilities
```

**Key Design Pattern:** Standalone tools for development and analysis

---

#### **`half_sword_ai/utils/`** - Shared Infrastructure

```
utils/
├── logger.py              → Basic logging
├── safe_logger.py         → Unicode-safe logging (Windows)
├── pretty_logger.py       → Formatted logging with colors
├── enhanced_logger.py     → Enhanced logging features
├── terminal_formatter.py  → Terminal formatting utilities
│
├── error_collector.py     → Error aggregation
├── metrics_reporter.py    → Metrics reporting
├── process_utils.py       → Process management utilities
├── time_utils.py          → Time utilities
├── math_utils.py          → Math utilities
├── file_utils.py          → File I/O utilities
└── window_finder.py       → Window detection utilities
```

**Key Design Pattern:** Reusable utilities with Windows compatibility focus

---

### **LEVEL 3: CONFIGURATION & LLM**

#### **`half_sword_ai/config/`** - Configuration Management

```
config/
└── __init__.py  → Config singleton
    ├── All system configuration
    ├── Hyperparameters
    ├── Paths & directories
    └── Feature flags
```

**Key Design Pattern:** Single source of truth for configuration

---

#### **`half_sword_ai/llm/`** - LLM Integration

```
llm/
└── ollama_integration.py  → Ollama/Qwen integration
    ├── Strategic decision-making
    └── High-level planning
```

**Key Design Pattern:** Optional LLM integration for strategic planning

---

## 🔄 DATA FLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    GAME (Half Sword)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   PERCEPTION LAYER            │
        │  - Screen Capture             │
        │  - Memory Reading             │
        │  - YOLO Detection             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   ACTOR PROCESS               │
        │  - Frame Processing           │
        │  - Model Inference            │
        │  - Action Selection           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   INPUT LAYER                 │
        │  - Input Multiplexer          │
        │  - Action Execution           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   GAME (Half Sword)           │
        └───────────────────────────────┘
                        │
                        │ (Experience)
                        ▼
        ┌───────────────────────────────┐
        │   REPLAY BUFFER               │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   LEARNER PROCESS             │
        │  - Experience Sampling        │
        │  - Model Training             │
        │  - Checkpointing              │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   MODEL UPDATE                │
        └───────────────────────────────┘
```

---

## 🎯 KEY DESIGN PRINCIPLES

### 1. **Modular Monolith**
- Single Python package (`half_sword_ai/`)
- Clear module boundaries
- No excessive file proliferation
- Consolidated related code

### 2. **Process Separation**
- **Actor Process**: Real-time inference (<16ms latency)
- **Learner Process**: Background training (can be slower)
- Prevents training from blocking gameplay

### 3. **Configuration Singleton**
- Single `config` instance
- Centralized configuration
- No scattered config files

### 4. **Safety First**
- Kill switch (F8) always functional
- Error recovery mechanisms
- Graceful degradation

### 5. **Real Data Only**
- No synthetic/mock data
- Always use real game data
- Real-time learning only

---

## 📦 EXTERNAL DEPENDENCIES

```
PyTorch          → Neural networks
Ultralytics YOLO → Object detection
Flask            → Web dashboard
OpenCV           → Image processing
dxcam            → Windows screen capture
pymem            → Memory reading
pydirectinput    → Input injection
```

---

## 🚀 EXECUTION FLOW

1. **Initialization** (`main.py` → `HalfSwordAgent.initialize()`)
   - Load configuration
   - Initialize components (Vision, Input, Learning)
   - Create model
   - Setup monitoring

2. **Startup** (`HalfSwordAgent.start()`)
   - Launch Actor Process (inference)
   - Launch Learner Process (training)
   - Start monitoring dashboard
   - Enable kill switch

3. **Runtime Loop**
   - **Actor**: Capture → Process → Infer → Act → Collect
   - **Learner**: Sample → Train → Update → Checkpoint
   - **Monitor**: Track metrics → Display → Alert

4. **Shutdown** (`HalfSwordAgent.stop()`)
   - Stop processes gracefully
   - Save checkpoints
   - Cleanup resources

---

## 📊 MODULE INTERDEPENDENCIES

```
config (singleton)
    ↑
    ├── core (agent, actor, learner, model)
    ├── perception (vision, yolo)
    ├── input (input_mux, kill_switch)
    ├── learning (replay_buffer, reward_shaper)
    ├── monitoring (performance_monitor, dashboard)
    └── utils (logger, error_collector)

core/agent
    ↑
    ├── core/actor
    ├── core/learner
    ├── perception/vision
    ├── input/input_mux
    ├── learning/replay_buffer
    └── monitoring/*

core/actor
    ↑
    ├── core/model
    ├── perception/vision
    ├── perception/yolo_detector
    ├── input/input_mux
    └── learning/replay_buffer

core/learner
    ↑
    ├── core/model
    ├── learning/replay_buffer
    └── learning/reward_shaper
```

---

## 🎨 CODE ORGANIZATION RULES

1. **One Class Per File**: Major classes get their own file
2. **Consolidated Utilities**: Related utilities together
3. **Type Hints**: Required for all functions
4. **Docstrings**: Google-style for public APIs
5. **Naming**: snake_case (files/functions), PascalCase (classes)
6. **Imports**: Absolute imports from `half_sword_ai` package

---

## 📈 SCALABILITY CONSIDERATIONS

- **Modular Design**: Easy to extract modules to separate processes
- **Configuration-Driven**: Feature flags enable/disable components
- **Process-Based**: Can scale Actor/Learner independently
- **Tool Separation**: Tools are standalone, don't affect core system

---

## 🔒 SAFETY & RELIABILITY

- **Kill Switch**: Always functional (F8)
- **Error Handling**: Centralized error handler
- **Watchdog**: Monitors system health
- **Graceful Degradation**: Continues operating on non-critical failures
- **Checkpointing**: Regular model saves prevent data loss

---

## 📝 SUMMARY

**Total Modules**: ~71 Python files organized into 8 major categories:
- **Core**: 7 files (orchestration, execution, models)
- **Perception**: 7 files (vision, detection)
- **Input**: 8 files (control, multiplexing)
- **Learning**: 9 files (training, rewards, memory)
- **Monitoring**: 8 files (metrics, dashboard, watchdog)
- **Tools**: 7 files (development tools)
- **Utils**: 12 files (shared utilities)
- **Config/LLM**: 2 files (configuration, LLM)

**Architecture**: Modular monolith with process-based separation for real-time performance

**Key Strength**: Clear separation of concerns with consolidated, maintainable code structure

