"""
Historical Reward Shaper for Half Sword
Implements reward functions based on HEMA treatises (Fiore, Mair, Talhoffer)

Reward Components:
- R_damage: Damage dealt
- R_edge: Edge alignment bonus
- R_gap: Gap targeting bonus (armor weak points)
- R_stance: Stance appropriateness (half-sword when close/armored)
"""
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HistoricalRewardShaper:
    """
    Reward shaping based on historical HEMA treatises
    Implements the reward function from the document:
    R_t = R_damage + λ₁R_edge + λ₂R_gap + λ₃R_stance
    """
    
    def __init__(self):
        # Reward weights (lambda values)
        self.lambda_edge = 1.0  # Edge alignment weight
        self.lambda_gap = 1.0   # Gap targeting weight
        self.lambda_stance = 0.1  # Stance appropriateness weight
        
        # Edge alignment threshold (degrees)
        self.edge_threshold_degrees = 10.0
        
        # Close play distance threshold (meters)
        self.close_play_distance = 1.0
    
    def calculate_reward(self, 
                         damage_dealt: float,
                         edge_alignment_score: float,
                         gap_target_hit: bool,
                         is_half_sword: bool,
                         opponent_distance: float,
                         opponent_armor_type: str = "unknown") -> float:
        """
        Calculate historical reward based on HEMA principles
        
        Args:
            damage_dealt: Structural damage inflicted
            edge_alignment_score: Edge alignment score (0-1)
            gap_target_hit: Whether a gap target was hit
            is_half_sword: Whether half-sword stance is active
            opponent_distance: Distance to opponent (meters)
            opponent_armor_type: Opponent armor type ("plate", "mail", "cloth", "unknown")
            
        Returns:
            Total reward value
        """
        # Base damage reward
        R_damage = damage_dealt
        
        # Edge alignment reward (Fiore: clean cuts)
        R_edge = self._calculate_edge_reward(edge_alignment_score)
        
        # Gap targeting reward (Mair: precision strikes)
        R_gap = self._calculate_gap_reward(gap_target_hit, opponent_armor_type)
        
        # Stance appropriateness reward (Fiore: Gioco Stretto)
        R_stance = self._calculate_stance_reward(
            is_half_sword, 
            opponent_distance, 
            opponent_armor_type
        )
        
        # Combined reward
        total_reward = (
            R_damage +
            self.lambda_edge * R_edge +
            self.lambda_gap * R_gap +
            self.lambda_stance * R_stance
        )
        
        return total_reward
    
    def _calculate_edge_reward(self, edge_alignment_score: float) -> float:
        """
        Reward for edge alignment (clean cuts)
        Based on Fiore's emphasis on proper blade orientation
        """
        # +1 if aligned within threshold, decays to 0
        if edge_alignment_score >= 0.9:  # Within 10° threshold
            return 1.0
        else:
            return edge_alignment_score  # Linear decay
    
    def _calculate_gap_reward(self, gap_target_hit: bool, opponent_armor_type: str) -> float:
        """
        Reward for hitting armor gaps (weak points)
        Based on Mair's precision techniques
        """
        if not gap_target_hit:
            return 0.0
        
        # Higher reward for hitting gaps in plate armor
        if opponent_armor_type == "plate":
            return 1.0  # Critical hit through plate
        elif opponent_armor_type == "mail":
            return 0.5  # Good hit through mail
        else:
            return 0.2  # Standard hit
    
    def _calculate_stance_reward(self, is_half_sword: bool, opponent_distance: float, 
                                 opponent_armor_type: str) -> float:
        """
        Reward for appropriate stance selection
        Based on Fiore's Gioco Largo (Wide Play) vs Gioco Stretto (Close Play)
        
        Rule: Half-sword is optimal when:
        - Distance < 1.0m AND opponent is armored
        """
        # Check if we're in close play range
        in_close_play = opponent_distance < self.close_play_distance
        
        # Check if opponent is armored
        is_armored = opponent_armor_type in ["plate", "mail"]
        
        if in_close_play and is_armored:
            # Half-sword is appropriate
            if is_half_sword:
                return 0.1  # +0.1 per frame for holding half-sword
            else:
                return -0.1  # -0.1 per frame for normal grip (encourages switching)
        else:
            # Normal grip is fine for wide play
            return 0.0
    
    def calculate_mordhau_reward(self, is_mordhau: bool, opponent_armor_type: str, 
                                 damage_dealt: float) -> float:
        """
        Reward for Mordhau (inverted sword) technique
        Based on Talhoffer's Mordschlag
        Optimal against heavy helmets
        """
        if not is_mordhau:
            return 0.0
        
        # Mordhau is highly effective against plate helmets
        if opponent_armor_type == "plate":
            return damage_dealt * 2.0  # Double damage reward
        else:
            return damage_dealt * 0.5  # Less effective against lighter armor

