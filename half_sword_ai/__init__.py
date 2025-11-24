"""
Half Sword AI Agent - Modular Architecture
Autonomous learning agent for Half Sword combat game

This package provides a complete AI agent system for learning to play Half Sword,
a physics-based combat game. The system uses deep reinforcement learning (DQN/PPO),
computer vision (YOLO), and human-in-the-loop learning (DAgger).

Main Components:
    - Core: Agent orchestrator, actor (inference), learner (training)
    - Perception: Screen capture, memory reading, YOLO detection
    - Learning: Replay buffer, human action recording, model tracking
    - Input: Input multiplexer, kill switch, gesture engine
    - Monitoring: Performance tracking, watchdog, dashboard
    - Utils: Shared utilities for logging, file I/O, math, etc.

Quick Start:
    >>> from half_sword_ai.core import HalfSwordAgent
    >>> agent = HalfSwordAgent()
    >>> agent.initialize()
    >>> agent.start()
"""

__version__ = "1.0.0"

# Main exports for convenience
from half_sword_ai.core import HalfSwordAgent, ActorProcess, LearnerProcess
from half_sword_ai.config import config

__all__ = [
    'HalfSwordAgent',
    'ActorProcess', 
    'LearnerProcess',
    'config',
]




