"""
Enhanced Half Sword Dataset Builder
Implements comprehensive data collection per "Dataset For Half Swords Bot.txt"

Features:
- Physics data extraction (torque, angular velocity, CoM, support polygon)
- HEMA pose classification (Fiore guards)
- Edge alignment calculation
- Gap targeting (armor weak points)
- Weapon state tracking (grip states)
- Historical reward shaping
- Dataset schema matching Table 1
"""
import numpy as np
import pandas as pd
import time
import logging
import os
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import cv2

from half_sword_ai.config import config
from half_sword_ai.perception.vision import ScreenCapture, MemoryReader, VisionProcessor
from half_sword_ai.input.input_mux import InputMultiplexer
from half_sword_ai.learning.reward_shaper import RewardShaper
from half_sword_ai.tools.historical_reward_shaper import HistoricalRewardShaper

logger = logging.getLogger(__name__)

@dataclass
class PhysicsState:
    """Physics state data for active ragdoll"""
    # Center of Mass
    com_x: float = 0.0
    com_y: float = 0.0
    com_z: float = 0.0
    com_projection_x: float = 0.0  # Projection onto ground plane
    com_projection_y: float = 0.0
    
    # Support Polygon
    support_polygon_area: float = 0.0
    stability_margin: float = 0.0  # Distance from CoM to edge of support polygon
    
    # Joint States (24 joints × 4 quaternion + 3 angular velocity = 168 dims)
    joint_rotations: np.ndarray = None  # (24, 4) quaternions
    joint_angular_velocities: np.ndarray = None  # (24, 3) angular velocities
    
    # Weapon Physics
    weapon_tip_velocity: np.ndarray = None  # (3,) velocity vector
    weapon_tip_position: np.ndarray = None  # (3,) position
    weapon_edge_normal: np.ndarray = None  # (3,) blade plane normal
    weapon_mass: float = 1.0
    weapon_angular_damping: float = 0.0
    weapon_linear_damping: float = 0.0
    
    def __post_init__(self):
        if self.joint_rotations is None:
            self.joint_rotations = np.zeros((24, 4))
        if self.joint_angular_velocities is None:
            self.joint_angular_velocities = np.zeros((24, 3))
        if self.weapon_tip_velocity is None:
            self.weapon_tip_velocity = np.zeros(3)
        if self.weapon_tip_position is None:
            self.weapon_tip_position = np.zeros(3)
        if self.weapon_edge_normal is None:
            self.weapon_edge_normal = np.array([0.0, 0.0, 1.0])

@dataclass
class HEMAPose:
    """HEMA pose classification"""
    pose_type: str = "unknown"  # "posta_di_breve", "posta_di_vera_croce", "posta_di_serpentino", "unknown"
    confidence: float = 0.0  # 0-1 match confidence
    joint_angles: np.ndarray = None  # Key joint angles for pose matching
    
    def __post_init__(self):
        if self.joint_angles is None:
            self.joint_angles = np.zeros(12)  # Key joints: shoulders, elbows, wrists, hips, knees

@dataclass
class WeaponState:
    """Weapon grip and state"""
    grip_state: int = 0  # 0=Standard, 1=Half-Sword, 2=Mordhau
    is_half_sword: bool = False
    is_mordhau: bool = False
    hand_positions: Dict[str, np.ndarray] = None  # Left/right hand positions
    
    def __post_init__(self):
        if self.hand_positions is None:
            self.hand_positions = {
                'left': np.zeros(3),
                'right': np.zeros(3)
            }

@dataclass
class CombatState:
    """Combat state and targeting"""
    opponent_distance: float = 0.0  # Distance to opponent
    opponent_armor: Dict[str, str] = None  # {"head": "plate", "chest": "mail", "legs": "cloth"}
    gap_targets: List[str] = None  # ["face", "armpit_l", "armpit_r", "groin"]
    edge_alignment_score: float = 0.0  # Dot product of velocity and blade plane
    damage_dealt: float = 0.0  # Structural damage inflicted
    
    def __post_init__(self):
        if self.opponent_armor is None:
            self.opponent_armor = {"head": "unknown", "chest": "unknown", "legs": "unknown"}
        if self.gap_targets is None:
            self.gap_targets = []

class HEMAPoseClassifier:
    """
    Classifies poses based on Fiore de'i Liberi guards
    Uses joint angle matching to identify historical guards
    """
    
    def __init__(self):
        # Define reference poses (normalized joint angles)
        # These are approximations - real values would come from pose estimation
        self.reference_poses = {
            "posta_di_breve": {
                "description": "Short Guard - sword close to chest, point forward",
                "joint_angles": np.array([
                    0.0, 0.0, 0.0,  # Left shoulder (pitch, yaw, roll)
                    0.3, 0.0, 0.0,  # Left elbow
                    0.5, 0.0, 0.0,  # Left wrist
                    0.0, 0.0, 0.0,  # Right shoulder
                    0.2, 0.0, 0.0,  # Right elbow
                    0.4, 0.0, 0.0,  # Right wrist
                ])
            },
            "posta_di_vera_croce": {
                "description": "True Cross - receiving strikes, transitioning to grapples",
                "joint_angles": np.array([
                    0.2, 0.0, 0.0,  # Left shoulder
                    0.4, 0.0, 0.0,  # Left elbow
                    0.3, 0.0, 0.0,  # Left wrist
                    0.2, 0.0, 0.0,  # Right shoulder
                    0.4, 0.0, 0.0,  # Right elbow
                    0.3, 0.0, 0.0,  # Right wrist
                ])
            },
            "posta_di_serpentino": {
                "description": "Serpent Guard - withdrawn guard for rapid thrusts",
                "joint_angles": np.array([
                    -0.2, 0.0, 0.0,  # Left shoulder
                    0.3, 0.0, 0.0,  # Left elbow
                    0.2, 0.0, 0.0,  # Left wrist
                    -0.1, 0.0, 0.0,  # Right shoulder
                    0.2, 0.0, 0.0,  # Right elbow
                    0.1, 0.0, 0.0,  # Right wrist
                ])
            }
        }
    
    def classify_pose(self, joint_angles: np.ndarray) -> HEMAPose:
        """
        Classify pose based on joint angles
        
        Args:
            joint_angles: Array of joint angles (12 key joints)
            
        Returns:
            HEMAPose with classification
        """
        if joint_angles is None or len(joint_angles) < 12:
            return HEMAPose(pose_type="unknown", confidence=0.0, joint_angles=joint_angles)
        
        best_match = "unknown"
        best_confidence = 0.0
        
        # Calculate cosine similarity for each reference pose
        for pose_name, pose_data in self.reference_poses.items():
            ref_angles = pose_data["joint_angles"]
            
            # Normalize vectors
            ref_norm = np.linalg.norm(ref_angles)
            angles_norm = np.linalg.norm(joint_angles)
            
            if ref_norm > 0 and angles_norm > 0:
                # Cosine similarity
                similarity = np.dot(ref_angles, joint_angles) / (ref_norm * angles_norm)
                
                # Also check angular distance
                angle_diff = np.linalg.norm(ref_angles - joint_angles)
                angle_score = 1.0 / (1.0 + angle_diff)  # Inverse distance
                
                # Combined confidence
                confidence = (similarity + angle_score) / 2.0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pose_name
        
        return HEMAPose(
            pose_type=best_match,
            confidence=best_confidence,
            joint_angles=joint_angles.copy()
        )

class EdgeAlignmentCalculator:
    """Calculates edge alignment score for blade strikes"""
    
    @staticmethod
    def calculate_alignment(velocity_vector: np.ndarray, blade_normal: np.ndarray, threshold_degrees: float = 10.0) -> float:
        """
        Calculate edge alignment score
        
        Args:
            velocity_vector: (3,) weapon tip velocity
            blade_normal: (3,) blade plane normal vector
            threshold_degrees: Maximum angle for perfect alignment (default 10°)
            
        Returns:
            Alignment score: 1.0 if aligned within threshold, 0.0 if perpendicular
        """
        if np.linalg.norm(velocity_vector) < 0.01 or np.linalg.norm(blade_normal) < 0.01:
            return 0.0
        
        # Normalize vectors
        vel_norm = velocity_vector / np.linalg.norm(velocity_vector)
        blade_norm = blade_normal / np.linalg.norm(blade_normal)
        
        # Dot product gives cosine of angle
        cos_angle = np.dot(vel_norm, blade_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Score: 1.0 if within threshold, decays linearly to 0.0 at 90°
        if angle_deg <= threshold_degrees:
            return 1.0
        elif angle_deg >= 90.0:
            return 0.0
        else:
            # Linear decay from threshold to 90°
            return 1.0 - (angle_deg - threshold_degrees) / (90.0 - threshold_degrees)

class GapTargetDetector:
    """Detects armor gaps and weak points for targeting"""
    
    def __init__(self):
        # Define gap locations relative to opponent
        self.gap_locations = {
            "face": {"offset": np.array([0.0, 0.0, 1.6]), "armor_type": "head"},
            "armpit_l": {"offset": np.array([-0.3, 0.0, 1.4]), "armor_type": "chest"},
            "armpit_r": {"offset": np.array([0.3, 0.0, 1.4]), "armor_type": "chest"},
            "groin": {"offset": np.array([0.0, 0.0, 0.9]), "armor_type": "legs"},
        }
    
    def detect_gaps(self, opponent_position: np.ndarray, opponent_armor: Dict[str, str], 
                    weapon_tip_position: np.ndarray) -> List[str]:
        """
        Detect accessible gaps based on armor and proximity
        
        Args:
            opponent_position: (3,) opponent position
            opponent_armor: Dict of armor types {"head": "plate", ...}
            weapon_tip_position: (3,) weapon tip position
            
        Returns:
            List of accessible gap names
        """
        accessible_gaps = []
        
        for gap_name, gap_data in self.gap_locations.items():
            armor_type = gap_data["armor_type"]
            armor_material = opponent_armor.get(armor_type, "unknown")
            
            # Gaps are accessible if:
            # 1. Armor is cloth/mail (not plate)
            # 2. Or weapon tip is close to gap location
            gap_world_pos = opponent_position + gap_data["offset"]
            distance_to_gap = np.linalg.norm(weapon_tip_position - gap_world_pos)
            
            if armor_material in ["cloth", "mail"] or distance_to_gap < 0.5:
                accessible_gaps.append(gap_name)
        
        return accessible_gaps

class HalfSwordDatasetBuilder:
    """
    Enhanced dataset builder for Half Sword training
    Implements comprehensive data collection per document specifications
    """
    
    def __init__(self, output_dir: str = None, dataset_name: str = None):
        """
        Initialize enhanced dataset builder
        
        Args:
            output_dir: Directory to save datasets
            dataset_name: Name for this dataset
        """
        self.output_dir = Path(output_dir or os.path.join(config.DATA_SAVE_PATH, "half_sword_datasets"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_name is None:
            timestamp = int(time.time())
            dataset_name = f"half_sword_comprehensive_{timestamp}"
        self.dataset_name = dataset_name
        
        # Initialize components
        logger.info("Initializing enhanced dataset builder...")
        self.screen_capture = ScreenCapture()
        self.memory_reader = MemoryReader()
        self.vision_processor = VisionProcessor(self.screen_capture)
        self.input_mux = InputMultiplexer()
        self.input_mux.start()
        
        # Specialized components
        self.hema_classifier = HEMAPoseClassifier()
        self.edge_calculator = EdgeAlignmentCalculator()
        self.gap_detector = GapTargetDetector()
        self.historical_reward_shaper = HistoricalRewardShaper()
        
        # Dataset storage
        self.entries: List[Dict] = []
        self.current_episode_id = 0
        self.current_frame_id = 0
        self.session_start_time = time.time()
        
        # Physics state tracking
        self.last_physics_state: Optional[PhysicsState] = None
        self.last_weapon_state: Optional[WeaponState] = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_episodes': 0,
            'half_sword_frames': 0,
            'mordhau_frames': 0,
            'hema_pose_matches': 0,
            'edge_aligned_strikes': 0,
            'gap_targets_hit': 0,
        }
        
        # Recording state
        self.recording = False
        
        logger.info(f"[OK] Enhanced dataset builder initialized: {self.dataset_name}")
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
        
        logger.info("[REC] Enhanced dataset recording started!")
        logger.info("   - Recording physics data, HEMA poses, edge alignment, gap targeting")
        logger.info("   - Press Ctrl+C to stop and save")
    
    def stop_recording(self):
        """Stop recording and save dataset"""
        if not self.recording:
            return
        
        self.recording = False
        logger.info("[STOP] Stopping dataset recording...")
        
        if len(self.entries) == 0:
            logger.warning("No data recorded!")
            return
        
        # Save dataset
        self._save_dataset()
        self._print_statistics()
    
    def record_frame(self) -> bool:
        """
        Record a single frame with comprehensive data
        
        Returns:
            True if frame was recorded, False if stopped
        """
        if not self.recording:
            return False
        
        try:
            # Capture frame
            frame = self.screen_capture.get_latest_frame()
            if frame is None:
                return True  # Skip if no frame
            
            # Get game state
            game_state = self.memory_reader.get_state()
            
            # Get input state
            human_action = self.input_mux.get_last_human_input()
            
            # Extract physics state (estimated from visual/memory)
            physics_state = self._extract_physics_state(game_state, frame)
            
            # Classify HEMA pose
            hema_pose = self._classify_hema_pose(physics_state)
            
            # Get weapon state
            weapon_state = self._extract_weapon_state(game_state, human_action)
            
            # Calculate edge alignment
            edge_alignment = self._calculate_edge_alignment(physics_state, weapon_state)
            
            # Detect gap targets
            combat_state = self._extract_combat_state(game_state, physics_state, weapon_state)
            
            # Build dataset entry matching Table 1 schema
            entry = self._build_dataset_entry(
                frame=frame,
                game_state=game_state,
                human_action=human_action,
                physics_state=physics_state,
                hema_pose=hema_pose,
                weapon_state=weapon_state,
                combat_state=combat_state,
                edge_alignment=edge_alignment
            )
            
            self.entries.append(entry)
            
            # Update statistics
            self.stats['total_frames'] += 1
            if weapon_state.is_half_sword:
                self.stats['half_sword_frames'] += 1
            if weapon_state.is_mordhau:
                self.stats['mordhau_frames'] += 1
            if hema_pose.confidence > 0.7:
                self.stats['hema_pose_matches'] += 1
            if edge_alignment > 0.8:
                self.stats['edge_aligned_strikes'] += 1
            
            self.current_frame_id += 1
            
            # Log progress
            if self.stats['total_frames'] % 1000 == 0:
                logger.info(f"[STATS] Recorded {self.stats['total_frames']} frames | "
                          f"Half-Sword: {self.stats['half_sword_frames']} | "
                          f"HEMA Matches: {self.stats['hema_pose_matches']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording frame: {e}", exc_info=True)
            return True
    
    def _extract_physics_state(self, game_state: Dict, frame: np.ndarray) -> PhysicsState:
        """
        Extract physics state from game state and frame
        Note: Full implementation requires UE4SS for actual bone positions
        """
        physics = PhysicsState()
        
        # Estimate from available data
        position = game_state.get('position', {})
        physics.com_x = position.get('x', 0.0) or 0.0
        physics.com_y = position.get('y', 0.0) or 0.0
        physics.com_z = position.get('z', 0.0) or 0.0
        
        # Project CoM onto ground plane (Z=0)
        physics.com_projection_x = physics.com_x
        physics.com_projection_y = physics.com_y
        
        # Estimate support polygon (simplified - would need foot positions)
        physics.support_polygon_area = 0.1  # Placeholder
        physics.stability_margin = 0.05  # Placeholder
        
        # Joint rotations/velocities would come from skeleton extraction
        # For now, use zeros (will be populated when UE4SS is configured)
        
        return physics
    
    def _classify_hema_pose(self, physics_state: PhysicsState) -> HEMAPose:
        """Classify HEMA pose from joint angles"""
        # Extract key joint angles (would come from skeleton)
        # For now, use placeholder - real implementation needs pose estimation
        joint_angles = np.zeros(12)
        
        return self.hema_classifier.classify_pose(joint_angles)
    
    def _extract_weapon_state(self, game_state: Dict, human_action: Dict) -> WeaponState:
        """Extract weapon grip state"""
        weapon_state = WeaponState()
        
        # Check if half-sword is active (RMB held)
        if human_action and isinstance(human_action, tuple) and len(human_action) > 2:
            buttons = human_action[2] if len(human_action) > 2 else {}
            weapon_state.is_half_sword = buttons.get('right', False)  # RMB
            weapon_state.is_mordhau = buttons.get('x', False)  # Context key
        
        # Set grip state
        if weapon_state.is_mordhau:
            weapon_state.grip_state = 2
        elif weapon_state.is_half_sword:
            weapon_state.grip_state = 1
        else:
            weapon_state.grip_state = 0
        
        return weapon_state
    
    def _calculate_edge_alignment(self, physics_state: PhysicsState, weapon_state: WeaponState) -> float:
        """Calculate edge alignment score"""
        if physics_state.weapon_tip_velocity is None or physics_state.weapon_edge_normal is None:
            return 0.0
        
        return self.edge_calculator.calculate_alignment(
            physics_state.weapon_tip_velocity,
            physics_state.weapon_edge_normal
        )
    
    def _extract_combat_state(self, game_state: Dict, physics_state: PhysicsState, 
                              weapon_state: WeaponState) -> CombatState:
        """Extract combat state and gap targets"""
        combat = CombatState()
        
        # Estimate opponent distance (would come from YOLO or memory)
        combat.opponent_distance = 2.0  # Placeholder
        
        # Get opponent armor (would come from detection)
        combat.opponent_armor = {"head": "unknown", "chest": "unknown", "legs": "unknown"}
        
        # Detect gaps
        if physics_state.weapon_tip_position is not None:
            opponent_pos = np.array([combat.opponent_distance, 0.0, 1.5])  # Placeholder
            combat.gap_targets = self.gap_detector.detect_gaps(
                opponent_pos,
                combat.opponent_armor,
                physics_state.weapon_tip_position
            )
        
        return combat
    
    def _build_dataset_entry(self, frame: np.ndarray, game_state: Dict, human_action: Dict,
                            physics_state: PhysicsState, hema_pose: HEMAPose,
                            weapon_state: WeaponState, combat_state: CombatState,
                            edge_alignment: float) -> Dict:
        """
        Build dataset entry matching Table 1 schema
        """
        # Extract input
        input_mouse_x = 0.0
        input_mouse_y = 0.0
        if human_action and isinstance(human_action, tuple):
            input_mouse_x = human_action[0] if len(human_action) > 0 else 0.0
            input_mouse_y = human_action[1] if len(human_action) > 1 else 0.0
        
        # Get game version (placeholder)
        game_version = "v0.4 Playtest"  # Would read from game
        
        # Build entry matching Table 1
        entry = {
            'timestamp': int(time.time() * 1000),  # Milliseconds
            'game_version': game_version,
            'input_mouse_x': float(input_mouse_x),
            'input_mouse_y': float(input_mouse_y),
            'is_half_sword': bool(weapon_state.is_half_sword),
            'hand_l_pos_x': float(weapon_state.hand_positions['left'][0]),
            'hand_l_pos_y': float(weapon_state.hand_positions['left'][1]),
            'hand_l_pos_z': float(weapon_state.hand_positions['left'][2]),
            'hand_r_pos_x': float(weapon_state.hand_positions['right'][0]),
            'hand_r_pos_y': float(weapon_state.hand_positions['right'][1]),
            'hand_r_pos_z': float(weapon_state.hand_positions['right'][2]),
            'weapon_tip_vel_x': float(physics_state.weapon_tip_velocity[0]),
            'weapon_tip_vel_y': float(physics_state.weapon_tip_velocity[1]),
            'weapon_tip_vel_z': float(physics_state.weapon_tip_velocity[2]),
            'edge_align_score': float(edge_alignment),
            'target_dist': float(combat_state.opponent_distance),
            'opponent_armor_head': str(combat_state.opponent_armor.get('head', 'unknown')),
            'opponent_armor_chest': str(combat_state.opponent_armor.get('chest', 'unknown')),
            'opponent_armor_legs': str(combat_state.opponent_armor.get('legs', 'unknown')),
            'historical_pose': str(hema_pose.pose_type),
            'pose_confidence': float(hema_pose.confidence),
            'damage_dealt': float(combat_state.damage_dealt),
            'com_x': float(physics_state.com_x),
            'com_y': float(physics_state.com_y),
            'com_z': float(physics_state.com_z),
            'stability_margin': float(physics_state.stability_margin),
            'grip_state': int(weapon_state.grip_state),
            'gap_targets': ','.join(combat_state.gap_targets),
            'episode_id': int(self.current_episode_id),
            'frame_id': int(self.current_frame_id),
        }
        
        # Calculate historical reward
        gap_hit = len(combat_state.gap_targets) > 0 and combat_state.damage_dealt > 0
        historical_reward = self.historical_reward_shaper.calculate_reward(
            damage_dealt=combat_state.damage_dealt,
            edge_alignment_score=edge_alignment,
            gap_target_hit=gap_hit,
            is_half_sword=weapon_state.is_half_sword,
            opponent_distance=combat_state.opponent_distance,
            opponent_armor_type=combat_state.opponent_armor.get('chest', 'unknown')
        )
        entry['historical_reward'] = float(historical_reward)
        
        return entry
    
    def _save_dataset(self):
        """Save dataset to CSV/Parquet format matching Table 1"""
        if len(self.entries) == 0:
            logger.warning("No entries to save!")
            return
        
        logger.info(f"[SAVE] Saving dataset with {len(self.entries)} entries...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.entries)
        
        # Save as CSV
        csv_path = self.output_dir / f"{self.dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"[OK] CSV saved: {csv_path}")
        
        # Save as Parquet (more efficient)
        parquet_path = self.output_dir / f"{self.dataset_name}.parquet"
        df.to_parquet(parquet_path, index=False, compression='snappy')
        logger.info(f"[OK] Parquet saved: {parquet_path}")
        
        # Save metadata
        metadata_path = self.output_dir / f"{self.dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'total_frames': len(self.entries),
                'total_episodes': self.stats['total_episodes'],
                'session_duration': time.time() - self.session_start_time,
                'statistics': {
                    'half_sword_frames': self.stats['half_sword_frames'],
                    'mordhau_frames': self.stats['mordhau_frames'],
                    'hema_pose_matches': self.stats['hema_pose_matches'],
                    'edge_aligned_strikes': self.stats['edge_aligned_strikes'],
                    'gap_targets_hit': self.stats['gap_targets_hit'],
                },
                'schema': list(df.columns),
            }, f, indent=2)
        
        logger.info(f"[OK] Metadata saved: {metadata_path}")
    
    def _print_statistics(self):
        """Print dataset statistics"""
        logger.info("=" * 80)
        logger.info("ENHANCED DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Frames: {self.stats['total_frames']}")
        logger.info(f"Total Episodes: {self.stats['total_episodes']}")
        logger.info(f"Half-Sword Frames: {self.stats['half_sword_frames']} ({100*self.stats['half_sword_frames']/max(1, self.stats['total_frames']):.1f}%)")
        logger.info(f"Mordhau Frames: {self.stats['mordhau_frames']} ({100*self.stats['mordhau_frames']/max(1, self.stats['total_frames']):.1f}%)")
        logger.info(f"HEMA Pose Matches: {self.stats['hema_pose_matches']} ({100*self.stats['hema_pose_matches']/max(1, self.stats['total_frames']):.1f}%)")
        logger.info(f"Edge-Aligned Strikes: {self.stats['edge_aligned_strikes']}")
        logger.info(f"Gap Targets Hit: {self.stats['gap_targets_hit']}")
        logger.info("=" * 80)
    
    def run_recording_loop(self, target_fps: int = 60):
        """Run main recording loop"""
        self.start_recording()
        
        frame_time = 1.0 / target_fps
        logger.info(f"[REC] Recording at {target_fps} FPS")
        
        try:
            while self.recording:
                loop_start = time.time()
                self.record_frame()
                
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))
                    
        except KeyboardInterrupt:
            logger.info("\n[WARN] Recording interrupted by user")
        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
        finally:
            self.stop_recording()

def main():
    """Main entry point for enhanced dataset collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Enhanced Half Sword Dataset")
    parser.add_argument("--name", type=str, help="Dataset name")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS")
    
    args = parser.parse_args()
    
    builder = HalfSwordDatasetBuilder(output_dir=args.output, dataset_name=args.name)
    builder.run_recording_loop(target_fps=args.fps)

if __name__ == "__main__":
    main()

