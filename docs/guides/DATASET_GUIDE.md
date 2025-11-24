# Half Sword Dataset Builder Guide

## Overview

The dataset builder collects comprehensive training data for the Half Sword AI agent. It records:
- **Frames**: 84x84x3 RGB images from the game
- **Actions**: Mouse movements and keyboard inputs (normalized)
- **Game State**: Health, stamina, position, enemy info, etc.
- **Rewards**: Calculated rewards based on gameplay
- **Terminal Flags**: Episode end markers (death, round end)

## Quick Start

### 1. Start Dataset Collection

**Windows:**
```bash
scripts\build_dataset.bat
```

**Python:**
```bash
python scripts/build_dataset.py
```

**With custom options:**
```bash
python -m half_sword_ai.tools.dataset_builder --name my_dataset --fps 60
```

### 2. Play the Game

- Make sure Half Sword is running
- Play normally - all your actions will be recorded
- The system captures frames at 60 FPS
- Press **Ctrl+C** to stop and save

### 3. Dataset Saved

Datasets are saved to `data/datasets/`:
- `half_sword_dataset_<timestamp>.npz` - Main dataset file
- `half_sword_dataset_<timestamp>_metadata.json` - Metadata and statistics

## Dataset Format

### .npz File Structure

```python
{
    'frames': np.array,              # (N, 84, 84, 3) RGB images
    'actions': np.array,             # (N, 6) [dx, dy, left, right, space, alt]
    'rewards': np.array,             # (N,) reward values
    'dones': np.array,               # (N,) boolean terminal flags
    'episode_ids': np.array,         # (N,) episode identifiers
    'frame_ids': np.array,           # (N,) frame identifiers within episode
    'timestamps': np.array,          # (N,) Unix timestamps
    'game_state_features': np.array, # (N, 8) normalized game state features
    'metadata': dict                 # Dataset metadata
}
```

### Action Format

Actions are normalized arrays:
- `[0]`: Mouse X delta (-1 to 1)
- `[1]`: Mouse Y delta (-1 to 1)
- `[2]`: Left mouse button (0 or 1)
- `[3]`: Right mouse button (0 or 1)
- `[4]`: Space key (0 or 1)
- `[5]`: Alt key (0 or 1)

## Inspecting Datasets

### View Dataset Information

```bash
python scripts/inspect_dataset.py data/datasets/half_sword_dataset_1234567890.npz
```

This shows:
- Total frames and episodes
- Reward statistics
- Episode lengths and rewards
- Action distribution
- Dataset metadata

### Load Dataset in Python

```python
from half_sword_ai.tools.dataset_utils import load_dataset

# Load dataset
dataset = load_dataset('data/datasets/half_sword_dataset_1234567890.npz')

# Access data
frames = dataset['frames']        # (N, 84, 84, 3)
actions = dataset['actions']      # (N, 6)
rewards = dataset['rewards']      # (N,)
dones = dataset['dones']          # (N,)
metadata = dataset['metadata']     # dict
```

## Combining Datasets

Combine multiple datasets into one:

```python
from half_sword_ai.tools.dataset_utils import combine_datasets

combine_datasets(
    dataset_paths=[
        'data/datasets/dataset1.npz',
        'data/datasets/dataset2.npz',
        'data/datasets/dataset3.npz',
    ],
    output_path='data/datasets/combined_dataset.npz',
    shuffle=True  # Shuffle combined data
)
```

## Best Practices

### 1. Collect Diverse Data

- Play different game modes (Abyss, Duel, etc.)
- Use different weapons and techniques
- Include both successful and failed attempts
- Record various combat scenarios

### 2. Dataset Size Recommendations

- **Minimum**: 10,000 frames (~3 minutes at 60 FPS)
- **Recommended**: 100,000+ frames (~30 minutes)
- **Ideal**: 500,000+ frames (~2.5 hours) for robust training

### 3. Quality Tips

- Play naturally - don't try to "teach" the bot
- Include variety in movements and techniques
- Record multiple sessions for diversity
- Check dataset statistics after collection

### 4. Episode Structure

- Each episode should be a complete fight/round
- Terminal flags mark episode ends (death, round end)
- Episodes are automatically detected from game state

## Dataset Statistics

After collection, check the metadata file for:
- Total frames and episodes
- Average episode length
- Reward distribution
- Action frequency
- Session duration

## Troubleshooting

### No Frames Captured

- Ensure Half Sword is running
- Check that the game window is visible
- Verify screen capture is working

### Low Frame Rate

- Close other applications
- Reduce recording FPS if needed: `--fps 30`
- Check system resources

### Dataset Too Large

- Datasets are compressed (.npz format)
- Large datasets (>1GB) are normal for long sessions
- Consider splitting into multiple smaller datasets

## Advanced Usage

### Custom Dataset Name

```bash
python -m half_sword_ai.tools.dataset_builder --name my_custom_dataset
```

### Custom Output Directory

```bash
python -m half_sword_ai.tools.dataset_builder --output /path/to/output
```

### Lower FPS Recording

```bash
python -m half_sword_ai.tools.dataset_builder --fps 30
```

## Integration with Training

The collected datasets can be used for:
1. **Behavioral Cloning**: Pre-train the policy on expert data
2. **Offline RL**: Train from collected data without interaction
3. **Data Analysis**: Understand gameplay patterns
4. **Model Evaluation**: Test on recorded scenarios

## Next Steps

1. Collect multiple datasets from different sessions
2. Combine datasets for larger training set
3. Use datasets for behavioral cloning pre-training
4. Analyze action patterns and reward distributions

