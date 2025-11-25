"""
Physics-Based Mouse Controller
Implements PID control and momentum management for physics-based combat

Based on GitHub ecosystem analysis: Physics-based active ragdolls require
smooth, accelerating mouse movements to generate realistic momentum.

Key Features:
- PID controller for smooth targeting
- Momentum management (prevents physics glitches)
- Bezier curve smoothing
- Swing path calculation
- Recovery logic for stuck weapons
"""

import numpy as np
import time
import logging
from typing import Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIDParams:
    """PID controller parameters"""
    kp: float = 0.5  # Proportional gain
    ki: float = 0.0  # Integral gain (usually 0 for mouse control)
    kd: float = 0.2  # Derivative gain
    max_output: float = 1.0  # Maximum output magnitude
    min_output: float = 0.01  # Minimum output to prevent drift


class PIDController:
    """
    PID Controller for smooth mouse movement
    
    Prevents physics engine glitches from sudden movements:
    - Infinite jerk causes "clang" (physics decoupling)
    - Smooth acceleration maximizes kinetic energy on impact
    """
    
    def __init__(self, params: PIDParams = None):
        self.params = params or PIDParams()
        self.integral = np.array([0.0, 0.0])
        self.last_error = np.array([0.0, 0.0])
        self.last_time = time.time()
        self.compute_count = 0
        logger.debug(f"[PIDController] Initialized with params: kp={self.params.kp:.3f}, ki={self.params.ki:.3f}, kd={self.params.kd:.3f}, max_output={self.params.max_output:.3f}")
    
    def compute(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Compute PID output for mouse movement
        
        Args:
            target: Target position (2D: x, y)
            current: Current position (2D: x, y)
            
        Returns:
            Mouse delta vector (2D: dx, dy)
        """
        self.compute_count += 1
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.001)  # Prevent division by zero
        self.last_time = current_time
        
        # Calculate error
        error = target - current
        error_magnitude = np.linalg.norm(error)
        
        logger.debug(f"[PIDController] Compute #{self.compute_count} | dt={dt*1000:.2f}ms | "
                    f"current=({current[0]:.2f}, {current[1]:.2f}) | "
                    f"target=({target[0]:.2f}, {target[1]:.2f}) | "
                    f"error_mag={error_magnitude:.3f}")
        
        # Proportional term
        p_term = self.params.kp * error
        
        # Integral term (accumulate error)
        self.integral += error * dt
        # Anti-windup: clamp integral
        max_integral = self.params.max_output / (self.params.ki + 1e-6)
        integral_before_clamp = self.integral.copy()
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.params.ki * self.integral
        
        if np.any(integral_before_clamp != self.integral):
            logger.debug(f"[PIDController] Integral windup clamped: {integral_before_clamp} -> {self.integral}")
        
        # Derivative term (rate of change)
        error_rate = (error - self.last_error) / dt
        d_term = self.params.kd * error_rate
        
        # Combine terms
        output = p_term + i_term - d_term  # Negative derivative (damping)
        
        logger.debug(f"[PIDController] Terms | P=({p_term[0]:.3f}, {p_term[1]:.3f}) | "
                    f"I=({i_term[0]:.3f}, {i_term[1]:.3f}) | "
                    f"D=({d_term[0]:.3f}, {d_term[1]:.3f}) | "
                    f"Output_raw=({output[0]:.3f}, {output[1]:.3f})")
        
        # Clamp output
        output_magnitude = np.linalg.norm(output)
        was_clamped = False
        if output_magnitude > self.params.max_output:
            output = output * (self.params.max_output / output_magnitude)
            was_clamped = True
            logger.debug(f"[PIDController] Output clamped to max: {output_magnitude:.3f} -> {self.params.max_output:.3f}")
        elif output_magnitude < self.params.min_output and output_magnitude > 0:
            # Apply minimum output in direction of error
            output = error * (self.params.min_output / (np.linalg.norm(error) + 1e-6))
            was_clamped = True
            logger.debug(f"[PIDController] Output boosted to min: {output_magnitude:.3f} -> {self.params.min_output:.3f}")
        
        self.last_error = error
        
        logger.debug(f"[PIDController] Final output=({output[0]:.3f}, {output[1]:.3f}) | "
                    f"magnitude={np.linalg.norm(output):.3f} | "
                    f"clamped={was_clamped}")
        
        return output


class BezierSmoother:
    """
    Bezier curve smoothing for mouse movements
    
    Generates smooth, accelerating arcs that maximize kinetic energy on impact.
    Critical for physics-based combat where damage is velocity-dependent.
    """
    
    def __init__(self, control_points: int = 4):
        self.control_points = control_points
    
    def generate_path(self, start: np.ndarray, end: np.ndarray, 
                     control_offset: float = 0.3) -> np.ndarray:
        """
        Generate Bezier curve path from start to end
        
        Args:
            start: Starting position (2D)
            end: Ending position (2D)
            control_offset: Offset for control points (0-1)
            
        Returns:
            Array of points along the curve
        """
        # Calculate control points for smooth arc
        direction = end - start
        perpendicular = np.array([-direction[1], direction[0]])  # 90° rotation
        
        # Control points create an arc
        p0 = start
        p1 = start + direction * control_offset + perpendicular * control_offset * 0.5
        p2 = end - direction * control_offset + perpendicular * control_offset * 0.5
        p3 = end
        
        # Generate points along Bezier curve
        t_values = np.linspace(0, 1, self.control_points)
        points = []
        
        for t in t_values:
            # Cubic Bezier formula: (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            point = ((1-t)**3 * p0 + 
                     3 * (1-t)**2 * t * p1 + 
                     3 * (1-t) * t**2 * p2 + 
                     t**3 * p3)
            points.append(point)
        
        return np.array(points)


class PhysicsMouseController:
    """
    Physics-aware mouse controller for Half Sword
    
    Manages momentum, prevents physics glitches, and generates realistic
    swing trajectories for physics-based combat.
    """
    
    def __init__(self, pid_params: PIDParams = None, use_bezier: bool = True):
        self.pid = PIDController(pid_params)
        self.bezier = BezierSmoother() if use_bezier else None
        self.velocity_history = deque(maxlen=10)  # Track velocity for momentum
        self.last_position = np.array([0.0, 0.0])
        self.last_time = time.time()
        self.stuck_threshold = 0.01  # Velocity threshold for "stuck" detection
        self.recovery_mode = False
        self.compute_count = 0
        self.stuck_count = 0
        self.recovery_count = 0
        logger.info(f"[PhysicsMouseController] Initialized | Bezier={use_bezier} | "
                   f"PID params: kp={self.pid.params.kp:.3f}, kd={self.pid.params.kd:.3f} | "
                   f"Stuck threshold={self.stuck_threshold:.4f}")
    
    def compute_movement(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Compute smooth mouse movement toward target
        
        Args:
            target: Target position (screen coordinates or normalized)
            current: Current position
            
        Returns:
            Mouse delta (dx, dy)
        """
        self.compute_count += 1
        logger.debug(f"[PhysicsController] compute_movement #{self.compute_count} | "
                    f"target=({target[0]:.2f}, {target[1]:.2f}) | "
                    f"current=({current[0]:.2f}, {current[1]:.2f}) | "
                    f"recovery_mode={self.recovery_mode}")
        
        # Use PID controller for smooth movement
        delta = self.pid.compute(target, current)
        delta_before_bezier = delta.copy()
        
        # Apply Bezier smoothing if enabled
        bezier_applied = False
        if self.bezier and np.linalg.norm(delta) > 0.1:
            # Generate smooth path
            path = self.bezier.generate_path(current, current + delta)
            if len(path) > 1:
                # Use first step of path
                delta = path[1] - path[0]
                bezier_applied = True
                logger.debug(f"[PhysicsController] Bezier smoothing applied | "
                           f"delta_before=({delta_before_bezier[0]:.3f}, {delta_before_bezier[1]:.3f}) | "
                           f"delta_after=({delta[0]:.3f}, {delta[1]:.3f}) | "
                           f"path_points={len(path)}")
        
        # Track velocity for momentum management
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.001)
        velocity = delta / dt
        velocity_magnitude = np.linalg.norm(velocity)
        
        self.velocity_history.append(velocity)
        self.last_position = current
        self.last_time = current_time
        
        logger.debug(f"[PhysicsController] Velocity tracking | "
                    f"velocity=({velocity[0]:.3f}, {velocity[1]:.3f}) | "
                    f"magnitude={velocity_magnitude:.4f} | "
                    f"dt={dt*1000:.2f}ms | "
                    f"history_size={len(self.velocity_history)}")
        
        # Check for stuck weapon (low velocity during swing)
        is_stuck = self._is_stuck(velocity)
        if is_stuck:
            self.stuck_count += 1
            logger.warning(f"[PhysicsController] WEAPON STUCK DETECTED (#{self.stuck_count}) | "
                         f"velocity={velocity_magnitude:.4f} | "
                         f"threshold={self.stuck_threshold:.4f} | "
                         f"entering recovery mode")
            self.recovery_mode = True
            # Generate recovery movement (retract)
            recovery_delta = self._generate_recovery_movement(delta)
            logger.debug(f"[PhysicsController] Recovery delta generated | "
                        f"original=({delta[0]:.3f}, {delta[1]:.3f}) | "
                        f"recovery=({recovery_delta[0]:.3f}, {recovery_delta[1]:.3f})")
            return recovery_delta
        
        if self.recovery_mode and velocity_magnitude > self.stuck_threshold * 2:
            self.recovery_count += 1
            logger.info(f"[PhysicsController] Weapon freed (#{self.recovery_count}) | "
                       f"velocity={velocity_magnitude:.4f} | "
                       f"exiting recovery mode")
            self.recovery_mode = False
        
        momentum = self.get_momentum()
        logger.debug(f"[PhysicsController] Final delta=({delta[0]:.3f}, {delta[1]:.3f}) | "
                    f"magnitude={np.linalg.norm(delta):.3f} | "
                    f"momentum={momentum:.4f} | "
                    f"bezier={bezier_applied} | "
                    f"recovery_mode={self.recovery_mode}")
        
        return delta
    
    def _is_stuck(self, velocity: np.ndarray) -> bool:
        """Check if weapon is stuck (low velocity during expected movement)"""
        if len(self.velocity_history) < 3:
            logger.debug(f"[PhysicsController] _is_stuck: Insufficient history ({len(self.velocity_history)} < 3)")
            return False
        
        # Check if velocity dropped suddenly (collision)
        recent_velocities = list(self.velocity_history)[-3:]
        avg_velocity = np.mean([np.linalg.norm(v) for v in recent_velocities])
        current_velocity = np.linalg.norm(velocity)
        
        logger.debug(f"[PhysicsController] _is_stuck check | "
                    f"avg_velocity={avg_velocity:.4f} | "
                    f"current_velocity={current_velocity:.4f} | "
                    f"threshold={self.stuck_threshold:.4f} | "
                    f"threshold*2={self.stuck_threshold * 2:.4f}")
        
        # If velocity dropped significantly, weapon might be stuck
        if avg_velocity > self.stuck_threshold * 2 and current_velocity < self.stuck_threshold:
            logger.warning(f"[PhysicsController] STUCK DETECTED | "
                          f"avg={avg_velocity:.4f} > {self.stuck_threshold * 2:.4f} AND "
                          f"current={current_velocity:.4f} < {self.stuck_threshold:.4f}")
            return True
        
        return False
    
    def _generate_recovery_movement(self, original_delta: np.ndarray) -> np.ndarray:
        """
        Generate recovery movement to free stuck weapon
        
        Retracts in opposite direction with slight variation
        """
        # Retract in opposite direction
        recovery = -original_delta * 0.5
        
        # Add slight perpendicular component to break free
        perpendicular = np.array([-recovery[1], recovery[0]]) * 0.3
        recovery += perpendicular
        
        return recovery
    
    def calculate_swing_path(self, target_position: np.ndarray, 
                            current_position: np.ndarray,
                            swing_type: str = "horizontal") -> np.ndarray:
        """
        Calculate optimal swing path for physics-based combat
        
        In Half Sword, placing crosshair on target isn't enough.
        Must drag mouse THROUGH target to generate velocity.
        
        Args:
            target_position: Target enemy position
            current_position: Current reticle position
            swing_type: "horizontal", "vertical", "overhead", "thrust"
            
        Returns:
            Array of mouse deltas for swing path
        """
        direction = target_position - current_position
        
        if swing_type == "horizontal":
            # Horizontal slash: move past target
            swing_end = target_position + np.array([direction[0] * 1.5, 0])
        elif swing_type == "vertical":
            # Vertical slash: move through target downward
            swing_end = target_position + np.array([0, abs(direction[1]) * 1.5])
        elif swing_type == "overhead":
            # Overhead smash: start high, swing down through target
            swing_start = current_position + np.array([0, -abs(direction[1]) * 0.5])
            swing_end = target_position + np.array([0, abs(direction[1]) * 0.5])
            return self._generate_swing_path(swing_start, swing_end)
        elif swing_type == "thrust":
            # Thrust: linear movement through target
            swing_end = target_position + direction * 1.2
        else:
            swing_end = target_position + direction * 1.5
        
        return self._generate_swing_path(current_position, swing_end)
    
    def _generate_swing_path(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Generate smooth swing path using Bezier curves"""
        if self.bezier:
            return self.bezier.generate_path(start, end, control_offset=0.4)
        else:
            # Linear path if Bezier disabled
            steps = 10
            return np.linspace(start, end, steps)
    
    def get_momentum(self) -> float:
        """Get current momentum magnitude"""
        if len(self.velocity_history) == 0:
            return 0.0
        
        recent_velocities = list(self.velocity_history)[-5:]
        avg_velocity = np.mean([np.linalg.norm(v) for v in recent_velocities])
        return avg_velocity
    
    def reset(self):
        """Reset controller state"""
        logger.info(f"[PhysicsController] RESET | "
                   f"compute_count={self.compute_count} | "
                   f"stuck_count={self.stuck_count} | "
                   f"recovery_count={self.recovery_count} | "
                   f"recovery_mode={self.recovery_mode}")
        self.pid.integral = np.array([0.0, 0.0])
        self.pid.last_error = np.array([0.0, 0.0])
        self.pid.compute_count = 0
        self.velocity_history.clear()
        self.recovery_mode = False
        self.last_time = time.time()
        self.compute_count = 0
        self.stuck_count = 0
        self.recovery_count = 0
        logger.debug("[PhysicsController] Reset complete - all counters cleared")

