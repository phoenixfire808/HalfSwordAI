# Documentation Index

Complete guide to all documentation in the Half Sword AI Agent project.

## Quick Start

- **[README.md](../../README.md)**: Project overview and quick start
- **[QUICK_START.md](./guides/QUICK_START.md)**: Detailed quick start guide

## Architecture & Design

- **[ARCHITECTURE.md](./ARCHITECTURE.md)**: System architecture overview
- **[MODULAR_STRUCTURE.md](./MODULAR_STRUCTURE.md)**: Modular structure guide
- **[ORGANIZATION.md](./ORGANIZATION.md)**: Project organization guide
- **[SYSTEM_ALIGNMENT.md](./SYSTEM_ALIGNMENT.md)**: System alignment notes

## User Guides

- **[HALF_SWORD_CONTROLS.md](./guides/HALF_SWORD_CONTROLS.md)**: Complete control reference
- **[INTERCEPTION_INSTALL.md](./guides/INTERCEPTION_INSTALL.md)**: Driver installation guide
- **[DATASET_GUIDE.md](./guides/DATASET_GUIDE.md)**: Dataset building guide

## Integration

- **[SCRIMBRAIN_INTEGRATION.md](./integration/SCRIMBRAIN_INTEGRATION.md)**: ScrimBrain integration
- **Guide To Scrimbrain Half Sword Integration.txt**: Detailed integration reference

## Development

- **[CURSOR_AGENT_GUIDE.md](./CURSOR_AGENT_GUIDE.md)**: Cursor AI agent guide
- **[PERFORMANCE_IMPROVEMENTS.md](./PERFORMANCE_IMPROVEMENTS.md)**: Performance notes
- **[FIXES_APPLIED.md](./FIXES_APPLIED.md)**: Fix documentation
- **[LOG_IMPROVEMENTS.md](./LOG_IMPROVEMENTS.md)**: Logging improvements

## Status & Changelog

- **[IMPLEMENTATION_STATUS.md](./status/IMPLEMENTATION_STATUS.md)**: Implementation status
- **[LAUNCH_STATUS.md](./status/LAUNCH_STATUS.md)**: Launch readiness
- **[LLM_REMOVAL_SUMMARY.md](./status/LLM_REMOVAL_SUMMARY.md)**: LLM removal notes
- **[ATTACK_SWING_FIX.md](./status/ATTACK_SWING_FIX.md)**: Attack swing fix

## Project Structure

```
docs/
├── INDEX.md                    # This file
├── ARCHITECTURE.md             # Architecture overview
├── MODULAR_STRUCTURE.md        # Modular structure
├── ORGANIZATION.md             # Organization guide
├── guides/                     # User guides
│   ├── QUICK_START.md
│   ├── HALF_SWORD_CONTROLS.md
│   ├── INTERCEPTION_INSTALL.md
│   └── DATASET_GUIDE.md
├── integration/                # Integration docs
│   └── SCRIMBRAIN_INTEGRATION.md
└── status/                     # Status updates
    ├── IMPLEMENTATION_STATUS.md
    └── LAUNCH_STATUS.md
```

## Memory Bank (AI Context)

The `memory-bank/` directory contains context for AI assistants:
- `projectbrief.md`: Project goals and constraints
- `productContext.md`: User stories and workflows
- `systemPatterns.md`: Architecture patterns
- `activeContext.md`: Current focus
- `progress.md`: Completed features
- `decisionLog.md`: Architectural decisions

## Code Documentation

- See `half_sword_ai/` package for code-level documentation
- Each module has docstrings and `__init__.py` exports
- Type hints throughout for IDE support

