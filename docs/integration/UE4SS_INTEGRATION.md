# UE4SS Integration Guide

## Overview

UE4SS (Unreal Engine 4/5 Scripting System) provides internal automation capabilities for Half Sword, enabling zero-latency access to game state and direct control via Lua scripting.

## Architecture

### Internal vs External Automation

**Internal (UE4SS/Lua)**:
- Zero-latency data access
- Direct memory reading (health, stamina, position)
- Function hooking (auto-parry)
- Object spawning (training dummies)
- Runs inside game process

**External (Python/CV)**:
- Screen capture + YOLO detection
- PyDirectInput for mouse/keyboard
- More robust (undetectable)
- Current implementation

**Hybrid (Recommended)**:
- UE4SS for state awareness
- Python for ML/control logic
- State bridge (Lua → Python JSON)

## Setup

### 1. Install UE4SS

```python
from half_sword_ai.tools.ue4ss_integration import UE4SSIntegration, get_default_config

config = get_default_config()
ue4ss = UE4SSIntegration(config)

# Check if installed
if not ue4ss.check_installation():
    # Install from UE4SS release
    ue4ss.install_ue4ss("path/to/ue4ss/release")
```

### 2. Generate SDK

```python
# Enable SDK generation in UE4SS settings
ue4ss.generate_sdk()

# Run game to generate headers
# Headers will be in: Mods/UE4SS/SDK/
```

### 3. Create Bot Mod

```python
# Create Lua mod for bot automation
ue4ss.create_bot_mod("HalfSwordBot")

# Mod will be created at:
# Mods/HalfSwordBot/Scripts/main.lua
```

## Key Features

### Enemy Scanning

```lua
-- Scan for enemies within radius
function Bot:ScanForEnemies()
    local enemies = {}
    -- Iterate GWorld actors
    -- Filter BP_HalfSwordCharacter_C
    -- Check hostility and distance
    return enemies
end
```

### Health Reading

```lua
-- Read health directly from memory
function Bot:GetHealth(actor)
    return actor.Health  -- Direct property access
end
```

### Auto-Parry

```lua
-- Hook enemy attack event
RegisterHook("/Game/Blueprints/Characters/BP_Enemy:StartAttack", function(enemy)
    -- Calculate parry vector
    -- Set input to parry stance
    Bot:Parry(enemy)
end)
```

### Training Dummy Spawning

```lua
-- Spawn passive NPC for training
function Bot:SpawnTrainingDummy()
    -- Use Trainer Mod integration
    UWorld:SpawnActor(BP_TrainingDummy_C, location)
end
```

## State Bridge

### Lua Side

```lua
-- Export state to JSON
local state = {
    player_health = Bot:GetHealth(Bot.Player),
    enemy_count = #Bot.Enemies
}
-- Write to shared file
```

### Python Side

```python
import json

# Read state from Lua
with open("C:/Temp/half_sword_state.json") as f:
    state = json.load(f)
    
health = state["player_health"]
```

## Recommended Repositories

- **UE4SS-RE/RE-UE4SS**: Core engine hook
- **massclown/HalfSwordTrainerMod**: Reference implementation
- **massclown/HalfSwordModInstaller**: Deployment tool
- **massclown/HalfSwordTrainerMod-playtest**: Playtest version support

## Physics Controller Integration

The `PhysicsMouseController` can be used with UE4SS for hybrid control:

```python
from half_sword_ai.input.physics_controller import PhysicsMouseController

controller = PhysicsMouseController()

# Get target from UE4SS state
target = np.array([state["enemy_x"], state["enemy_y"]])

# Calculate smooth swing path
swing_path = controller.calculate_swing_path(target, current, "horizontal")

# Execute via PyDirectInput
for point in swing_path:
    delta = point - current
    pydirectinput.moveRel(int(delta[0]), int(delta[1]))
    current = point
```

## Implementation Roadmap

1. **Phase 1**: Install UE4SS, verify injection
2. **Phase 2**: Generate SDK, identify key classes
3. **Phase 3**: Implement basic Lua bot (enemy scan, health read)
4. **Phase 4**: Add auto-parry hook
5. **Phase 5**: Create state bridge for Python integration
6. **Phase 6**: Hybrid architecture (UE4SS state + Python ML)

## References

See `docs/research/github_ecosystem_analysis.txt` for comprehensive technical analysis.

