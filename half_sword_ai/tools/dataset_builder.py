"""
Dataset Builder for Half Sword
Comprehensive data collection system for training the AI agent

Records:
- Frames (84x84x3 RGB images)
- Actions (mouse deltas, button presses)
- Game State (health, stamina, position, etc.)
- Rewards (calculated from game state)
- Terminal flags (death, round end)

Saves to .npz format for efficient training data loading
"""
import numpy as np
import time
import logging
import os
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from half_sword_ai.config import config
from half_sword_ai.perception.vision import ScreenCapture, MemoryReader
from half_sword_ai.input.input_mux import InputMultiplexer
from half_sword_ai.learning.reward_shaper import RewardShaper
from half_sword_ai.perception.ocr_reward_tracker import OCRRewardTracker
from half_sword_ai.perception.terminal_state_detector import TerminalStateDetector

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

@dataclass
class DatasetEntry:
    """Single entry in the dataset"""
    frame: np.ndarray  # (84, 84, 3) RGB image
    action: np.ndarray  # [dx, dy, left, right, space, alt] normalized
    game_state: Dict  # Health, stamina, position, etc.
    reward: float
    done: bool
    timestamp: float
    episode_id: int
    frame_id: int

class DatasetBuilder:
    """
    Comprehensive dataset builder for Half Sword
    Records all gameplay data for training
    """
    
    def __init__(self, output_dir: str = None, dataset_name: str = None):
        """
        Initialize dataset builder
        
        Args:
            output_dir: Directory to save datasets (default: data/datasets/)
            dataset_name: Name for this dataset (default: auto-generated)
        """
        self.output_dir = Path(output_dir or os.path.join(config.DATA_SAVE_PATH, "datasets"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset name if not provided
        if dataset_name is None:
            timestamp = int(time.time())
            dataset_name = f"half_sword_dataset_{timestamp}"
        self.dataset_name = dataset_name
        
        # Initialize components
        logger.info("Initializing dataset builder components...")
        self.screen_capture = ScreenCapture()
        self.memory_reader = MemoryReader()
        self.input_mux = InputMultiplexer()
        self.input_mux.start()
        
        # Initialize reward calculation
        self.reward_shaper = RewardShaper()
        self.ocr_tracker = OCRRewardTracker() if config.OCR_ENABLED else None
        self.terminal_detector = TerminalStateDetector() if config.TERMINAL_STATE_DETECTION else None
        
        # Dataset storage
        self.entries: List[DatasetEntry] = []
        self.current_episode_id = 0
        self.current_frame_id = 0
        self.session_start_time = time.time()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_episodes': 0,
            'total_actions': 0,
            'total_reward': 0.0,
            'episode_lengths': [],
            'episode_rewards': [],
            'action_distribution': np.zeros(6),  # Track action frequency
        }
        
        # Recording state
        self.recording = False
        self.last_frame = None
        self.last_game_state = None
        
        logger.info(f"✅ Dataset builder initialized: {self.dataset_name}")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def start_recording(self):
        """Start recording dataset"""
        if self.recording:
            logger.warning("Already recording")
            return
        
        self.recording = True
        self.session_start_time = time.time()
        self.current_episode_id = 0
        self.current_frame_id = 0
        self.entries = []
        
        logger.info("🎥 Dataset recording started!")
        logger.info("   - Play the game normally")
        logger.info("   - All frames, actions, and game state will be recorded")
        logger.info("   - Press Ctrl+C to stop and save")
    
    def stop_recording(self):
        """Stop recording and save dataset"""
        if not self.recording:
            return
        
        self.recording = False
        logger.info("🛑 Stopping dataset recording...")
        
        if len(self.entries) == 0:
            logger.warning("No data recorded!")
            return
        
        # Save dataset
        self._save_dataset()
        
        # Print statistics
        self._print_statistics()
    
    def record_frame(self) -> bool:
        """
        Record a single frame of gameplay
        
        Returns:
            True if frame was recorded, False if recording stopped
        """
        if not self.recording:
            return False
        
        try:
            # Capture frame
            frame = self.screen_capture.get_latest_frame()
            if frame is None:
                if self.last_frame is not None:
                    frame = self.last_frame  # Use last valid frame
                else:
                    return True  # Skip if no frame available
            
            self.last_frame = frame
            
            # Get game state
            game_state = self.memory_reader.get_state()
            self.last_game_state = game_state
            
            # Check for terminal state (new episode)
            is_terminal = False
            if self.terminal_detector:
                terminal_result = self.terminal_detector.detect_death_screen(frame)
                is_terminal = terminal_result.get('is_terminal', False)
            
            if is_terminal or game_state.get('is_dead', False):
                # Episode ended
                if len(self.entries) > 0 and self.entries[-1].episode_id == self.current_episode_id:
                    # Calculate episode reward
                    episode_reward = sum(e.reward for e in self.entries if e.episode_id == self.current_episode_id)
                    episode_length = sum(1 for e in self.entries if e.episode_id == self.current_episode_id)
                    self.stats['episode_rewards'].append(episode_reward)
                    self.stats['episode_lengths'].append(episode_length)
                    logger.info(f"📊 Episode {self.current_episode_id} ended: {episode_length} frames, reward: {episode_reward:.2f}")
                
                self.current_episode_id += 1
                self.current_frame_id = 0
                self.stats['total_episodes'] += 1
            
            # Get human action (if available)
            human_action = self.input_mux.get_last_human_input()
            
            if human_action is None:
                # No human input - use neutral action
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                # Convert human action to array format
                if isinstance(human_action, tuple):
                    delta_x = human_action[0] if len(human_action) > 0 else 0.0
                    delta_y = human_action[1] if len(human_action) > 1 else 0.0
                    buttons = human_action[2] if len(human_action) > 2 else {}
                    action = np.array([
                        delta_x,
                        delta_y,
                        1.0 if buttons.get('left', False) else 0.0,
                        1.0 if buttons.get('right', False) else 0.0,
                        1.0 if buttons.get('space', False) else 0.0,
                        1.0 if buttons.get('alt', False) else 0.0
                    ])
                else:
                    action = np.array(human_action)
            
            # Calculate reward
            reward = self._calculate_reward(game_state, frame, action)
            
            # Create dataset entry
            entry = DatasetEntry(
                frame=frame.copy(),  # Copy frame to avoid reference issues
                action=action.copy(),
                game_state=game_state.copy(),
                reward=reward,
                done=is_terminal or game_state.get('is_dead', False),
                timestamp=time.time(),
                episode_id=self.current_episode_id,
                frame_id=self.current_frame_id
            )
            
            # Store entry
            self.entries.append(entry)
            
            # Update statistics
            self.stats['total_frames'] += 1
            self.stats['total_actions'] += 1
            self.stats['total_reward'] += reward
            self.stats['action_distribution'] += np.abs(action)  # Track action magnitude
            
            self.current_frame_id += 1
            
            # Log progress every 1000 frames
            if self.stats['total_frames'] % 1000 == 0:
                logger.info(f"📊 Recorded {self.stats['total_frames']} frames | "
                          f"{len(self.entries)} entries | "
                          f"Episodes: {self.stats['total_episodes']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording frame: {e}", exc_info=True)
            return True  # Continue recording despite errors
    
    def _calculate_reward(self, game_state: Dict, frame: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for current state"""
        try:
            # Use reward shaper
            reward = self.reward_shaper.calculate_reward(
                game_state=game_state,
                frame=frame,
                action=action,
                previous_state=self.last_game_state
            )
            
            # Add OCR reward if available
            if self.ocr_tracker:
                ocr_result = self.ocr_tracker.update(frame)
                if ocr_result.get('success', False):
                    reward += ocr_result.get('reward', 0.0)
            
            return float(reward)
        except Exception as e:
            logger.debug(f"Reward calculation error: {e}")
            return 0.0
    
    def _save_dataset(self):
        """Save dataset to .npz file"""
        if len(self.entries) == 0:
            logger.warning("No entries to save!")
            return
        
        logger.info(f"💾 Saving dataset with {len(self.entries)} entries...")
        
        # Prepare arrays
        frames = []
        actions = []
        rewards = []
        dones = []
        episode_ids = []
        frame_ids = []
        timestamps = []
        
        # Extract game state features
        game_state_features = []
        
        for entry in self.entries:
            frames.append(entry.frame)
            actions.append(entry.action)
            rewards.append(entry.reward)
            dones.append(entry.done)
            episode_ids.append(entry.episode_id)
            frame_ids.append(entry.frame_id)
            timestamps.append(entry.timestamp)
            
            # Extract key game state features
            gs = entry.game_state
            features = [
                gs.get('health', 100.0) / 100.0,
                gs.get('stamina', 100.0) / 100.0,
                gs.get('enemy_health', 100.0) / 100.0,
                1.0 if gs.get('is_dead', False) else 0.0,
                1.0 if gs.get('enemy_dead', False) else 0.0,
                gs.get('position', {}).get('x', 0.0) / 1000.0,
                gs.get('position', {}).get('y', 0.0) / 1000.0,
                gs.get('position', {}).get('z', 0.0) / 1000.0,
            ]
            game_state_features.append(features)
        
        # Convert to numpy arrays
        frames_array = np.array(frames)  # (N, 84, 84, 3)
        actions_array = np.array(actions)  # (N, 6)
        rewards_array = np.array(rewards)  # (N,)
        dones_array = np.array(dones, dtype=bool)  # (N,)
        episode_ids_array = np.array(episode_ids)  # (N,)
        frame_ids_array = np.array(frame_ids)  # (N,)
        timestamps_array = np.array(timestamps)  # (N,)
        game_state_features_array = np.array(game_state_features)  # (N, 8)
        
        # Save to .npz file
        dataset_path = self.output_dir / f"{self.dataset_name}.npz"
        
        np.savez_compressed(
            dataset_path,
            frames=frames_array,
            actions=actions_array,
            rewards=rewards_array,
            dones=dones_array,
            episode_ids=episode_ids_array,
            frame_ids=frame_ids_array,
            timestamps=timestamps_array,
            game_state_features=game_state_features_array,
            metadata={
                'dataset_name': self.dataset_name,
                'total_frames': len(self.entries),
                'total_episodes': self.stats['total_episodes'],
                'session_duration': time.time() - self.session_start_time,
                'frame_shape': frames_array.shape,
                'action_shape': actions_array.shape,
                'config': {
                    'capture_width': config.CAPTURE_WIDTH,
                    'capture_height': config.CAPTURE_HEIGHT,
                    'frame_stack_size': config.FRAME_STACK_SIZE,
                }
            }
        )
        
        file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Dataset saved: {dataset_path}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Frames: {len(self.entries)}")
        logger.info(f"   Episodes: {self.stats['total_episodes']}")
        
        # Also save metadata as JSON for easy inspection
        metadata_path = self.output_dir / f"{self.dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'total_frames': len(self.entries),
                'total_episodes': self.stats['total_episodes'],
                'session_duration': time.time() - self.session_start_time,
                'statistics': {
                    'total_reward': float(self.stats['total_reward']),
                    'avg_reward_per_frame': float(self.stats['total_reward'] / max(1, len(self.entries))),
                    'episode_lengths': self.stats['episode_lengths'],
                    'episode_rewards': [float(r) for r in self.stats['episode_rewards']],
                    'avg_episode_length': float(np.mean(self.stats['episode_lengths'])) if self.stats['episode_lengths'] else 0.0,
                    'avg_episode_reward': float(np.mean(self.stats['episode_rewards'])) if self.stats['episode_rewards'] else 0.0,
                },
                'action_distribution': self.stats['action_distribution'].tolist(),
            }, f, indent=2)
        
        logger.info(f"✅ Metadata saved: {metadata_path}")
    
    def _print_statistics(self):
        """Print dataset statistics"""
        logger.info("=" * 80)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Frames: {self.stats['total_frames']}")
        logger.info(f"Total Episodes: {self.stats['total_episodes']}")
        logger.info(f"Total Reward: {self.stats['total_reward']:.2f}")
        logger.info(f"Avg Reward per Frame: {self.stats['total_reward'] / max(1, self.stats['total_frames']):.4f}")
        
        if self.stats['episode_lengths']:
            logger.info(f"Avg Episode Length: {np.mean(self.stats['episode_lengths']):.1f} frames")
            logger.info(f"Min Episode Length: {np.min(self.stats['episode_lengths'])} frames")
            logger.info(f"Max Episode Length: {np.max(self.stats['episode_lengths'])} frames")
        
        if self.stats['episode_rewards']:
            logger.info(f"Avg Episode Reward: {np.mean(self.stats['episode_rewards']):.2f}")
            logger.info(f"Min Episode Reward: {np.min(self.stats['episode_rewards']):.2f}")
            logger.info(f"Max Episode Reward: {np.max(self.stats['episode_rewards']):.2f}")
        
        logger.info("=" * 80)
    
    def run_recording_loop(self, target_fps: int = 60):
        """
        Run main recording loop
        
        Args:
            target_fps: Target frames per second for recording
        """
        self.start_recording()
        
        frame_time = 1.0 / target_fps
        logger.info(f"🎥 Recording at {target_fps} FPS (frame_time: {frame_time*1000:.2f}ms)")
        
        try:
            while self.recording:
                loop_start = time.time()
                
                # Record frame
                self.record_frame()
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))
                    
        except KeyboardInterrupt:
            logger.info("\n⚠️ Recording interrupted by user")
        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
        finally:
            self.stop_recording()

def main():
    """Main entry point for dataset collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Half Sword Dataset")
    parser.add_argument("--name", type=str, help="Dataset name (default: auto-generated)")
    parser.add_argument("--output", type=str, help="Output directory (default: data/datasets/)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS (default: 60)")
    
    args = parser.parse_args()
    
    # Create dataset builder
    builder = DatasetBuilder(output_dir=args.output, dataset_name=args.name)
    
    # Run recording loop
    builder.run_recording_loop(target_fps=args.fps)

if __name__ == "__main__":
    main()

