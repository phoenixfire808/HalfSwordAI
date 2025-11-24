"""
Dataset Utilities for Half Sword
Load, inspect, and combine datasets
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> Dict:
    """
    Load a Half Sword dataset from .npz file
    
    Args:
        dataset_path: Path to .npz dataset file
        
    Returns:
        Dictionary containing:
            - frames: (N, 84, 84, 3) RGB images
            - actions: (N, 6) action arrays
            - rewards: (N,) reward values
            - dones: (N,) terminal flags
            - episode_ids: (N,) episode identifiers
            - frame_ids: (N,) frame identifiers
            - timestamps: (N,) timestamps
            - game_state_features: (N, 8) game state features
            - metadata: Dataset metadata
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    dataset = {
        'frames': data['frames'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'dones': data['dones'],
        'episode_ids': data['episode_ids'],
        'frame_ids': data['frame_ids'],
        'timestamps': data['timestamps'],
        'game_state_features': data['game_state_features'],
        'metadata': data['metadata'].item() if 'metadata' in data else {}
    }
    
    logger.info(f"✅ Loaded dataset: {len(dataset['frames'])} frames, {dataset['metadata'].get('total_episodes', 0)} episodes")
    return dataset

def inspect_dataset(dataset_path: str):
    """
    Print detailed information about a dataset
    
    Args:
        dataset_path: Path to .npz dataset file
    """
    dataset = load_dataset(dataset_path)
    
    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print()
    
    # Basic info
    print("Basic Information:")
    print(f"  Total Frames: {len(dataset['frames'])}")
    print(f"  Total Episodes: {dataset['metadata'].get('total_episodes', 'Unknown')}")
    print(f"  Frame Shape: {dataset['frames'].shape}")
    print(f"  Action Shape: {dataset['actions'].shape}")
    print()
    
    # Statistics
    print("Statistics:")
    print(f"  Total Reward: {dataset['rewards'].sum():.2f}")
    print(f"  Avg Reward per Frame: {dataset['rewards'].mean():.4f}")
    print(f"  Min Reward: {dataset['rewards'].min():.4f}")
    print(f"  Max Reward: {dataset['rewards'].max():.4f}")
    print(f"  Terminal Frames: {dataset['dones'].sum()}")
    print()
    
    # Episode statistics
    episode_ids = dataset['episode_ids']
    unique_episodes = np.unique(episode_ids)
    if len(unique_episodes) > 0:
        episode_lengths = [np.sum(episode_ids == ep_id) for ep_id in unique_episodes]
        episode_rewards = [dataset['rewards'][episode_ids == ep_id].sum() for ep_id in unique_episodes]
        
        print("Episode Statistics:")
        print(f"  Number of Episodes: {len(unique_episodes)}")
        print(f"  Avg Episode Length: {np.mean(episode_lengths):.1f} frames")
        print(f"  Min Episode Length: {np.min(episode_lengths)} frames")
        print(f"  Max Episode Length: {np.max(episode_lengths)} frames")
        print(f"  Avg Episode Reward: {np.mean(episode_rewards):.2f}")
        print(f"  Min Episode Reward: {np.min(episode_rewards):.2f}")
        print(f"  Max Episode Reward: {np.max(episode_rewards):.2f}")
        print()
    
    # Action statistics
    actions = dataset['actions']
    print("Action Statistics:")
    print(f"  Action Dimensions: {actions.shape[1]}")
    print(f"  Avg Action Magnitude: {np.linalg.norm(actions, axis=1).mean():.4f}")
    print(f"  Action Components:")
    for i, name in enumerate(['dx', 'dy', 'left', 'right', 'space', 'alt']):
        print(f"    {name}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}")
    print()
    
    # Metadata
    if dataset['metadata']:
        print("Metadata:")
        for key, value in dataset['metadata'].items():
            if key != 'config':
                print(f"  {key}: {value}")
        print()
    
    print("=" * 80)

def combine_datasets(dataset_paths: List[str], output_path: str, shuffle: bool = True):
    """
    Combine multiple datasets into one
    
    Args:
        dataset_paths: List of paths to .npz dataset files
        output_path: Path to save combined dataset
        shuffle: Whether to shuffle the combined dataset
    """
    logger.info(f"Combining {len(dataset_paths)} datasets...")
    
    all_frames = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_episode_ids = []
    all_frame_ids = []
    all_timestamps = []
    all_game_state_features = []
    
    max_episode_id = 0
    
    for i, dataset_path in enumerate(dataset_paths):
        logger.info(f"Loading dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        dataset = load_dataset(dataset_path)
        
        # Adjust episode IDs to be unique across datasets
        episode_ids = dataset['episode_ids'] + max_episode_id
        max_episode_id = episode_ids.max() + 1
        
        all_frames.append(dataset['frames'])
        all_actions.append(dataset['actions'])
        all_rewards.append(dataset['rewards'])
        all_dones.append(dataset['dones'])
        all_episode_ids.append(episode_ids)
        all_frame_ids.append(dataset['frame_ids'])
        all_timestamps.append(dataset['timestamps'])
        all_game_state_features.append(dataset['game_state_features'])
    
    # Concatenate all arrays
    logger.info("Concatenating datasets...")
    combined_frames = np.concatenate(all_frames, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    combined_rewards = np.concatenate(all_rewards, axis=0)
    combined_dones = np.concatenate(all_dones, axis=0)
    combined_episode_ids = np.concatenate(all_episode_ids, axis=0)
    combined_frame_ids = np.concatenate(all_frame_ids, axis=0)
    combined_timestamps = np.concatenate(all_timestamps, axis=0)
    combined_game_state_features = np.concatenate(all_game_state_features, axis=0)
    
    # Shuffle if requested
    if shuffle:
        logger.info("Shuffling combined dataset...")
        indices = np.random.permutation(len(combined_frames))
        combined_frames = combined_frames[indices]
        combined_actions = combined_actions[indices]
        combined_rewards = combined_rewards[indices]
        combined_dones = combined_dones[indices]
        combined_episode_ids = combined_episode_ids[indices]
        combined_frame_ids = combined_frame_ids[indices]
        combined_timestamps = combined_timestamps[indices]
        combined_game_state_features = combined_game_state_features[indices]
    
    # Save combined dataset
    logger.info(f"Saving combined dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        frames=combined_frames,
        actions=combined_actions,
        rewards=combined_rewards,
        dones=combined_dones,
        episode_ids=combined_episode_ids,
        frame_ids=combined_frame_ids,
        timestamps=combined_timestamps,
        game_state_features=combined_game_state_features,
        metadata={
            'dataset_name': 'combined_dataset',
            'total_frames': len(combined_frames),
            'total_episodes': len(np.unique(combined_episode_ids)),
            'source_datasets': dataset_paths,
            'shuffled': shuffle,
        }
    )
    
    logger.info(f"✅ Combined dataset saved: {len(combined_frames)} frames, {len(np.unique(combined_episode_ids))} episodes")

def list_datasets(data_dir: str = None) -> List[str]:
    """
    List all available datasets in the data directory
    
    Args:
        data_dir: Directory to search (default: data/datasets/)
        
    Returns:
        List of dataset file paths
    """
    from half_sword_ai.config import config
    
    if data_dir is None:
        data_dir = Path(config.DATA_SAVE_PATH) / "datasets"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        return []
    
    datasets = list(data_dir.glob("*.npz"))
    return [str(d) for d in sorted(datasets)]

