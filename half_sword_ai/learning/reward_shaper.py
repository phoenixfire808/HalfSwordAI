"""
Comprehensive Reward Function Architecture for Half Sword Physics-Based Combat Agent
Implements multi-layered reward system based on technical compendium specifications.

This module implements:
- Layer 1: Locomotion and Stability Rewards
- Layer 2: Offensive Biomechanics and Lethality
- Layer 3: Defensive Dynamics and Spacing
- Layer 4: Imitation Learning and Style
- Potential-Based Reward Shaping (PBRS)
- Curriculum Learning Support
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CurriculumPhase(Enum):
    """Curriculum learning phases"""
    TODDLER = "toddler"  # Locomotion only
    SWORDSMAN = "swordsman"  # Static targets
    DUELIST = "duelist"  # Defensive drilling
    MASTER = "master"  # Full combat


class RewardShaper:
    """
    Comprehensive reward shaping system for Half Sword physics-based combat.
    Implements Potential-Based Reward Shaping (PBRS) to ensure optimal policy preservation.
    """
    
    def __init__(self, 
                 curriculum_phase: CurriculumPhase = CurriculumPhase.MASTER,
                 gamma: float = 0.99,
                 enable_pbrs: bool = True):
        """
        Initialize reward shaper.
        
        Args:
            curriculum_phase: Current curriculum learning phase
            gamma: Discount factor for PBRS
            enable_pbrs: Whether to use Potential-Based Reward Shaping
        """
        self.curriculum_phase = curriculum_phase
        self.gamma = gamma
        self.enable_pbrs = enable_pbrs
        
        # Previous state for PBRS (potential function calculation)
        self.prev_state: Optional[Dict] = None
        
        # Reward component weights (can be adjusted per curriculum phase)
        self.weights = self._get_weights_for_phase(curriculum_phase)
        
        # Edge alignment power (harsh filter for misalignment)
        self.alignment_power = 3.0
        
        # Balance reward parameters
        self.balance_k = 0.1  # Exponential decay constant
        
        # Recovery tracking
        self.prev_head_height: Optional[float] = None
        self.recovery_time_penalty = 0.01
        
        # Energy efficiency tracking
        self.prev_torque: Optional[np.ndarray] = None
        
        # Spacing parameters
        self.optimal_range_epsilon = 0.1
        
        logger.info(f"RewardShaper initialized: phase={curriculum_phase.value}, PBRS={enable_pbrs}")
    
    def _get_weights_for_phase(self, phase: CurriculumPhase) -> Dict[str, float]:
        """Get reward component weights for curriculum phase"""
        weights = {
            # Layer 1: Locomotion and Stability
            'balance': 0.0,
            'anti_cross': 0.0,
            'recovery': 0.0,
            
            # Layer 2: Offensive Biomechanics
            'edge_alignment': 0.0,
            'kinetic_energy': 0.0,
            'armor_discrimination': 0.0,
            
            # Layer 3: Defensive Dynamics
            'parry': 0.0,
            'spacing': 0.0,
            
            # Layer 4: Imitation Learning
            'imitation': 0.0,
            'energy_efficiency': 0.0,
            
            # PBRS potentials
            'health_potential': 0.0,
            'positional_potential': 0.0,
        }
        
        if phase == CurriculumPhase.TODDLER:
            weights['balance'] = 1.0
            weights['anti_cross'] = 0.5
            weights['recovery'] = 1.0
        elif phase == CurriculumPhase.SWORDSMAN:
            weights['balance'] = 0.3
            weights['edge_alignment'] = 1.0
            weights['kinetic_energy'] = 1.0
            weights['imitation'] = 0.5
            weights['energy_efficiency'] = 0.3
        elif phase == CurriculumPhase.DUELIST:
            weights['balance'] = 0.2
            weights['parry'] = 1.0
            weights['spacing'] = 0.8
            weights['recovery'] = 0.5
        else:  # MASTER
            weights['balance'] = 0.2
            weights['anti_cross'] = 0.3
            weights['recovery'] = 0.3
            weights['edge_alignment'] = 1.0
            weights['kinetic_energy'] = 1.0
            weights['armor_discrimination'] = 1.0
            weights['parry'] = 0.8
            weights['spacing'] = 0.8
            weights['imitation'] = 0.5  # Increased to encourage copying human movements
            weights['energy_efficiency'] = 0.2
            weights['health_potential'] = 0.5
            weights['positional_potential'] = 0.3
        
        return weights
    
    def calculate_reward(self, 
                        game_state: Dict,
                        action: Optional[np.ndarray] = None,
                        previous_state: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate comprehensive reward signal.
        
        Args:
            game_state: Current game state
            action: Current action (for energy efficiency calculation)
            previous_state: Previous state (for PBRS)
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        reward = 0.0
        breakdown = {}
        
        # Layer 1: Locomotion and Stability
        if self.weights['balance'] > 0:
            balance_reward = self._calculate_balance_reward(game_state)
            reward += balance_reward * self.weights['balance']
            breakdown['balance'] = balance_reward
        
        if self.weights['anti_cross'] > 0:
            cross_penalty = self._calculate_anti_cross_penalty(game_state)
            reward += cross_penalty * self.weights['anti_cross']
            breakdown['anti_cross'] = cross_penalty
        
        if self.weights['recovery'] > 0:
            recovery_reward = self._calculate_recovery_reward(game_state, previous_state)
            reward += recovery_reward * self.weights['recovery']
            breakdown['recovery'] = recovery_reward
        
        # Layer 2: Offensive Biomechanics
        if self.weights['edge_alignment'] > 0:
            alignment_reward = self._calculate_edge_alignment_reward(game_state)
            reward += alignment_reward * self.weights['edge_alignment']
            breakdown['edge_alignment'] = alignment_reward
        
        if self.weights['kinetic_energy'] > 0:
            kinetic_reward = self._calculate_kinetic_energy_reward(game_state)
            reward += kinetic_reward * self.weights['kinetic_energy']
            breakdown['kinetic_energy'] = kinetic_reward
        
        if self.weights['armor_discrimination'] > 0:
            armor_reward = self._calculate_armor_discrimination_reward(game_state)
            reward += armor_reward * self.weights['armor_discrimination']
            breakdown['armor_discrimination'] = armor_reward
        
        # Layer 3: Defensive Dynamics
        if self.weights['parry'] > 0:
            parry_reward = self._calculate_parry_reward(game_state)
            reward += parry_reward * self.weights['parry']
            breakdown['parry'] = parry_reward
        
        if self.weights['spacing'] > 0:
            spacing_reward = self._calculate_spacing_reward(game_state)
            reward += spacing_reward * self.weights['spacing']
            breakdown['spacing'] = spacing_reward
        
        # Layer 4: Imitation Learning and Style
        if self.weights['imitation'] > 0:
            imitation_reward = self._calculate_imitation_reward(game_state)
            reward += imitation_reward * self.weights['imitation']
            breakdown['imitation'] = imitation_reward
        
        if self.weights['energy_efficiency'] > 0 and action is not None:
            energy_penalty = self._calculate_energy_efficiency_penalty(action, previous_state)
            reward += energy_penalty * self.weights['energy_efficiency']
            breakdown['energy_efficiency'] = energy_penalty
        
        # Potential-Based Reward Shaping
        if self.enable_pbrs:
            pbrs_reward = self._calculate_pbrs_reward(game_state, previous_state)
            reward += pbrs_reward
            breakdown['pbrs'] = pbrs_reward
        
        # Update previous state for next iteration
        self.prev_state = game_state.copy() if game_state else None
        
        breakdown['total'] = reward
        return reward, breakdown
    
    # ========== Layer 1: Locomotion and Stability ==========
    
    def _calculate_balance_reward(self, game_state: Dict) -> float:
        """
        Center of Mass (CoM) Projection Reward.
        Rewards keeping projected CoM close to centroid of support polygon.
        
        Formula: r_balance = e^(-k * ||P_CoM - C_poly||^2)
        """
        # Approximate CoM position from game state
        # If we have position data, use it; otherwise estimate from health/stability
        position = game_state.get('position', {})
        if isinstance(position, dict):
            pos_x = position.get('x')
            pos_y = position.get('y')
            # Handle None values - default to 0.0 if not available
            if pos_x is None:
                pos_x = 0.0
            if pos_y is None:
                pos_y = 0.0
        else:
            pos_x, pos_y = 0.0, 0.0
        
        # Estimate support polygon centroid (assume centered for now)
        # In full implementation, would calculate from foot positions
        centroid_x, centroid_y = 0.0, 0.0
        
        # Calculate distance from CoM to centroid
        distance_sq = (pos_x - centroid_x)**2 + (pos_y - centroid_y)**2
        
        # Exponential reward (closer = higher reward)
        balance_reward = np.exp(-self.balance_k * distance_sq)
        
        # Penalize if agent is falling (low health or terminal state)
        health = game_state.get('health')
        health = 100 if health is None else health
        if game_state.get('is_dead', False) or health < 10:
            balance_reward *= 0.1
        
        return float(balance_reward)
    
    def _calculate_anti_cross_penalty(self, game_state: Dict) -> float:
        """
        Anti-Cross-Stepping Constraint.
        Penalizes leg crossing which causes tripping.
        
        Formula: p_cross = -1.0 if left foot is to right of right foot, else 0
        """
        # Approximate foot positions from movement pattern
        # If we have movement data, check for crossing
        movement_pattern = game_state.get('movement_pattern', [])
        
        if len(movement_pattern) >= 2:
            # Check if movement suggests leg crossing (rapid direction changes)
            recent_movements = movement_pattern[-2:]
            if len(recent_movements) >= 2:
                dx1, dy1 = recent_movements[0].get('dx', 0), recent_movements[0].get('dy', 0)
                dx2, dy2 = recent_movements[1].get('dx', 0), recent_movements[1].get('dy', 0)
                
                # Detect rapid direction reversal (sign of leg crossing)
                if (dx1 * dx2 < 0 and abs(dx1) > 0.5 and abs(dx2) > 0.5):
                    return -1.0
        
        return 0.0
    
    def _calculate_recovery_reward(self, game_state: Dict, previous_state: Optional[Dict]) -> float:
        """
        Recovery Efficiency Reward.
        Rewards upward head movement when fallen (ragdoll state).
        
        Formula: r_recovery = (h_head(t) - h_head(t-1)) * w_up - w_time
        """
        # Check if agent is in ragdoll/fallen state
        health = game_state.get('health')
        health = 100 if health is None else health
        is_fallen = game_state.get('is_dead', False) or health < 20
        
        if not is_fallen:
            return 0.0
        
        # Estimate head height from health/stability metrics
        # In full implementation, would use bone transform data
        current_head_height = health / 100.0
        
        if previous_state:
            prev_health = previous_state.get('health')
            prev_health = 0 if prev_health is None else prev_health
            prev_head_height = prev_health / 100.0
            height_delta = current_head_height - prev_head_height
            
            # Reward upward movement
            recovery_reward = height_delta * 10.0 - self.recovery_time_penalty
            
            return float(max(0.0, recovery_reward))
        
        return -self.recovery_time_penalty
    
    # ========== Layer 2: Offensive Biomechanics ==========
    
    def _calculate_edge_alignment_reward(self, game_state: Dict) -> float:
        """
        Edge Alignment Dot Product Reward.
        Rewards proper blade alignment for cutting.
        
        Formula: r_align = 1 - |v̂ · n̂|
        Where v is velocity vector, n is blade normal vector.
        """
        # Approximate from weapon velocity and action
        # In full implementation, would use weapon tip velocity and blade normal
        
        # Estimate velocity from movement pattern
        movement_pattern = game_state.get('movement_pattern', [])
        if len(movement_pattern) < 2:
            return 0.0
        
        # Get recent movement velocity
        recent = movement_pattern[-1]
        dx = recent.get('dx', 0.0)
        dy = recent.get('dy', 0.0)
        velocity_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Only reward if velocity is significant (prevent slow alignment rewards)
        if velocity_magnitude < 2.0:  # Threshold in normalized units
            return 0.0
        
        # Approximate alignment: assume good alignment if movement is smooth and directed
        # Perfect alignment would have velocity perpendicular to blade normal
        # For approximation: reward smooth, directed movements
        velocity_normalized = np.array([dx, dy]) / max(velocity_magnitude, 1e-6)
        
        # Assume blade normal is perpendicular to movement direction (simplified)
        # In reality, would calculate from weapon mesh orientation
        blade_normal = np.array([-dy, dx]) / max(velocity_magnitude, 1e-6)
        
        # Dot product: v̂ · n̂
        dot_product = np.abs(np.dot(velocity_normalized, blade_normal))
        
        # Alignment reward: 1 - |dot_product|
        # When moving edge-on (cutting), dot_product ≈ 0, reward ≈ 1
        # When moving flat-on, dot_product ≈ 1, reward ≈ 0
        alignment_reward = 1.0 - dot_product
        
        return float(max(0.0, alignment_reward))
    
    def _calculate_kinetic_energy_reward(self, game_state: Dict) -> float:
        """
        Kinetic Energy and Momentum Transfer Reward.
        Rewards high velocity with proper edge alignment.
        
        Formula: r_kinetic = 0.5 * m_weapon * ||v_tip||^2 * r_align^k
        """
        # Get alignment reward
        alignment_reward = self._calculate_edge_alignment_reward(game_state)
        
        # Estimate weapon mass (assume medium weapon = 1.0 normalized)
        weapon_mass = game_state.get('weapon_mass', 1.0)
        
        # Estimate tip velocity from movement pattern
        movement_pattern = game_state.get('movement_pattern', [])
        if len(movement_pattern) < 1:
            return 0.0
        
        recent = movement_pattern[-1]
        dx = recent.get('dx', 0.0)
        dy = recent.get('dy', 0.0)
        tip_velocity = np.sqrt(dx**2 + dy**2)
        
        # Kinetic energy: 0.5 * m * v^2
        kinetic_energy = 0.5 * weapon_mass * (tip_velocity ** 2)
        
        # Apply alignment filter (raised to power for harsh filtering)
        kinetic_reward = kinetic_energy * (alignment_reward ** self.alignment_power)
        
        return float(max(0.0, kinetic_reward))
    
    def _calculate_armor_discrimination_reward(self, game_state: Dict) -> float:
        """
        Armor Discrimination and Targeting Reward.
        Context-aware impact reward based on target material.
        """
        # Check if we have impact data
        impact_data = game_state.get('impact', {})
        if not impact_data:
            return 0.0
        
        target_material = impact_data.get('material', 'flesh')
        action_type = impact_data.get('action_type', 'slash')  # slash, thrust, blunt
        
        # Material-Action Reward Matrix (from document Table 2)
        reward_matrix = {
            'flesh': {'slash': 1.0, 'thrust': 1.0, 'blunt': 0.8},
            'gambeson': {'slash': 0.5, 'thrust': 0.8, 'blunt': 0.9},
            'chainmail': {'slash': 0.1, 'thrust': 0.6, 'blunt': 1.0},
            'plate': {'slash': -0.1, 'thrust': 1.5, 'blunt': 1.0},
        }
        
        reward = reward_matrix.get(target_material, {}).get(action_type, 0.0)
        
        # Bonus for hitting armor gaps (if detected)
        if impact_data.get('hit_gap', False):
            reward *= 1.5
        
        return float(reward)
    
    # ========== Layer 3: Defensive Dynamics ==========
    
    def _calculate_parry_reward(self, game_state: Dict) -> float:
        """
        Parry Volume Reward.
        Rewards intercepting incoming attacks with proper vector interaction.
        
        Formula: r_parry = I(Collides) * max(0, v_self · -v_opp)
        """
        # Check if parry occurred
        parry_data = game_state.get('parry', {})
        if not parry_data.get('success', False):
            return 0.0
        
        # Get relative velocities
        self_velocity = parry_data.get('self_velocity', np.array([0.0, 0.0]))
        opp_velocity = parry_data.get('opponent_velocity', np.array([0.0, 0.0]))
        
        # Dot product: moving weapon against incoming attack
        parry_strength = np.dot(self_velocity, -opp_velocity)
        
        # Reward active parry (moving into attack) vs passive (static)
        parry_reward = max(0.0, parry_strength)
        
        return float(parry_reward)
    
    def _calculate_spacing_reward(self, game_state: Dict) -> float:
        """
        Spacing (Footsies) Reward.
        Rewards maintaining optimal engagement range.
        
        Formula: r_spacing = e^(-(D_actual - D_opt)^2)
        """
        # Get distance to opponent
        distance = game_state.get('enemy_distance', None)
        if distance is None:
            return 0.0
        
        # Estimate weapon reach (assume medium weapon = 1.0 normalized)
        weapon_reach = game_state.get('weapon_reach', 1.0)
        opponent_reach = game_state.get('enemy_weapon_reach', 1.0)
        
        # Optimal range: just inside hitting range
        optimal_range = weapon_reach - self.optimal_range_epsilon
        
        # Distance reward (Gaussian around optimal range)
        distance_error = distance - optimal_range
        spacing_reward = np.exp(-(distance_error ** 2))
        
        # Penalty for being in danger zone (between own reach and opponent's longer reach)
        if opponent_reach > weapon_reach:
            danger_zone_min = weapon_reach
            danger_zone_max = opponent_reach
            if danger_zone_min <= distance <= danger_zone_max:
                # In danger zone without attacking - apply penalty
                if not game_state.get('is_attacking', False):
                    spacing_reward *= 0.5
        
        return float(spacing_reward)
    
    # ========== Layer 4: Imitation Learning ==========
    
    def _calculate_imitation_reward(self, game_state: Dict) -> float:
        """
        DeepMimic-style Imitation Reward.
        Rewards matching reference poses and velocities.
        """
        # Get reference pose/pattern if available
        reference_pattern = game_state.get('reference_pattern', None)
        if not reference_pattern:
            return 0.0
        
        # Calculate pose similarity (simplified)
        # In full implementation, would compare joint angles
        current_pattern = game_state.get('movement_pattern', [])
        if len(current_pattern) < 1 or len(reference_pattern) < 1:
            return 0.0
        
        # Simple similarity: compare movement directions
        current_dir = np.array([current_pattern[-1].get('dx', 0), current_pattern[-1].get('dy', 0)])
        ref_dir = np.array([reference_pattern[-1].get('dx', 0), reference_pattern[-1].get('dy', 0)])
        
        # Normalize
        current_norm = np.linalg.norm(current_dir)
        ref_norm = np.linalg.norm(ref_dir)
        
        if current_norm < 1e-6 or ref_norm < 1e-6:
            return 0.0
        
        current_dir = current_dir / current_norm
        ref_dir = ref_dir / ref_norm
        
        # Cosine similarity
        similarity = np.dot(current_dir, ref_dir)
        
        return float(max(0.0, similarity))
    
    def _calculate_energy_efficiency_penalty(self, 
                                           action: np.ndarray,
                                           previous_state: Optional[Dict]) -> float:
        """
        Energy Efficiency and Smoothness Penalty.
        Penalizes high torque and jerk (rate of change of torque).
        
        Formula: p_energy = sum(||τ||^2 + ||τ̇||^2)
        """
        if action is None or len(action) < 2:
            return 0.0
        
        if previous_state is None or self.prev_torque is None:
            # Estimate torque from action magnitude
            current_torque = float(np.linalg.norm(action[:2]))  # Movement components
            self.prev_torque = current_torque
            return 0.0
        
        # Estimate current torque from action
        current_torque = float(np.linalg.norm(action[:2]))
        
        # Handle None prev_torque
        if self.prev_torque is None:
            self.prev_torque = current_torque
            return 0.0
        
        # Torque magnitude penalty
        torque_penalty = current_torque ** 2
        
        # Jerk penalty (rate of change of torque)
        torque_delta = current_torque - self.prev_torque
        jerk_penalty = torque_delta ** 2
        
        # Total energy penalty (negative)
        energy_penalty = -(torque_penalty + jerk_penalty)
        
        self.prev_torque = current_torque
        
        return float(energy_penalty)
    
    # ========== Potential-Based Reward Shaping ==========
    
    def _calculate_pbrs_reward(self, 
                              current_state: Dict,
                              previous_state: Optional[Dict]) -> float:
        """
        Potential-Based Reward Shaping.
        Ensures optimal policy preservation.
        
        Formula: F(s, s') = γ * Φ(s') - Φ(s)
        """
        if not self.enable_pbrs or previous_state is None:
            return 0.0
        
        # Calculate potentials
        current_potential = self._calculate_potential(current_state)
        previous_potential = self._calculate_potential(previous_state)
        
        # PBRS shaping reward
        pbrs_reward = self.gamma * current_potential - previous_potential
        
        return float(pbrs_reward)
    
    def _calculate_potential(self, state: Dict) -> float:
        """
        Calculate potential function for PBRS.
        Combines health potential and positional potential.
        """
        potential = 0.0
        
        # Health Potential: Φ_hp = (Health_self - Health_opponent)
        if self.weights['health_potential'] > 0:
            self_health = state.get('health')
            self_health = 100.0 if self_health is None else self_health
            enemy_health = state.get('enemy_health')
            enemy_health = 100.0 if enemy_health is None else enemy_health
            health_potential = (self_health - enemy_health) / 100.0
            potential += health_potential * self.weights['health_potential']
        
        # Positional Potential: Φ_pos = -|Distance - OptimalRange|
        if self.weights['positional_potential'] > 0:
            distance = state.get('enemy_distance', None)
            if distance is not None:
                weapon_reach = state.get('weapon_reach', 1.0)
                optimal_range = weapon_reach - self.optimal_range_epsilon
                distance_error = abs(distance - optimal_range)
                positional_potential = -distance_error
                potential += positional_potential * self.weights['positional_potential']
        
        return float(potential)
    
    def set_curriculum_phase(self, phase: CurriculumPhase):
        """Update curriculum phase and adjust weights"""
        self.curriculum_phase = phase
        self.weights = self._get_weights_for_phase(phase)
        logger.info(f"Curriculum phase updated to: {phase.value}")

