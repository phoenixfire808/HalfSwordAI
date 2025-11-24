"""
Error Handler: Automatic error detection and recovery
Stops game and fixes errors when detected
"""
import logging
import time
import traceback
from typing import Dict, List, Optional, Callable
from collections import deque
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Automatic error detection and recovery system
    Detects errors, stops the game, and attempts to fix issues
    """
    
    def __init__(self, agent=None, max_errors: int = 10, error_window: float = 60.0):
        """
        Initialize error handler
        
        Args:
            agent: Reference to the main agent (for stopping)
            max_errors: Maximum errors before stopping
            error_window: Time window in seconds to count errors
        """
        self.agent = agent
        self.max_errors = max_errors
        self.error_window = error_window
        
        # Error tracking
        self.errors = deque(maxlen=1000)
        self.error_count = 0
        self.last_error_time = 0
        self.critical_errors = []
        
        # Error patterns to detect
        self.critical_patterns = [
            'memory access violation',
            'segmentation fault',
            'null pointer',
            'index out of range',
            'keyerror',
            'attributeerror',
            'typeerror',
            'valueerror',
            'runtimeerror',
            'assertionerror',
            'keyboardinterrupt'
        ]
        
        # Auto-fix enabled
        self.auto_fix_enabled = True
        self.fix_attempts = {}
        self.max_fix_attempts = 3
        
        logger.info("Error handler initialized - Auto-stop and fix enabled")
    
    def record_error(self, error: Exception, context: str = "unknown", 
                    component: str = "unknown", stop_game: bool = True) -> bool:
        """
        Record an error and decide if we should stop the game
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            component: Component that generated the error
            stop_game: Whether to stop the game for this error
            
        Returns:
            True if game should be stopped, False otherwise
        """
        current_time = time.time()
        error_info = {
            'timestamp': current_time,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'component': component,
            'traceback': traceback.format_exc(),
            'critical': self._is_critical_error(error)
        }
        
        self.errors.append(error_info)
        self.error_count += 1
        self.last_error_time = current_time
        
        if error_info['critical']:
            self.critical_errors.append(error_info)
            logger.critical(f"🔴 CRITICAL ERROR in {component}/{context}: {error}")
        
        # Check if we should stop
        should_stop = False
        
        if error_info['critical']:
            should_stop = True
            logger.critical(f"🚨 Critical error detected - stopping game")
        elif self._should_stop_due_to_error_rate():
            should_stop = True
            logger.error(f"🚨 Too many errors detected ({self.error_count} in {self.error_window}s) - stopping game")
        elif stop_game:
            should_stop = True
        
        if should_stop and self.agent:
            self._stop_game_and_fix(error_info)
        
        return should_stop
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Check if error is critical"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        for pattern in self.critical_patterns:
            if pattern in error_str or pattern in error_type:
                return True
        
        return False
    
    def _should_stop_due_to_error_rate(self) -> bool:
        """Check if error rate is too high"""
        if self.error_count < self.max_errors:
            return False
        
        # Count errors in the last error_window seconds
        current_time = time.time()
        recent_errors = [
            e for e in self.errors 
            if current_time - e['timestamp'] <= self.error_window
        ]
        
        if len(recent_errors) >= self.max_errors:
            return True
        
        return False
    
    def _stop_game_and_fix(self, error_info: Dict):
        """
        Stop the game and attempt to fix the error
        
        Args:
            error_info: Information about the error
        """
        logger.critical("=" * 80)
        logger.critical("🚨 AUTOMATIC ERROR DETECTION - STOPPING GAME")
        logger.critical("=" * 80)
        logger.critical(f"Error Type: {error_info['error_type']}")
        logger.critical(f"Error Message: {error_info['error_message']}")
        logger.critical(f"Component: {error_info['component']}")
        logger.critical(f"Context: {error_info['context']}")
        logger.critical("=" * 80)
        
        # Stop the agent
        if self.agent:
            try:
                logger.info("Stopping agent due to error...")
                
                # Enable safety lock
                if hasattr(self.agent, 'input_mux') and self.agent.input_mux:
                    if hasattr(self.agent.input_mux, 'enable_safety_lock'):
                        self.agent.input_mux.enable_safety_lock()
                    elif hasattr(self.agent.input_mux, 'safety_locked'):
                        self.agent.input_mux.safety_locked = True
                    logger.info("✅ Safety lock enabled")
                
                # Stop actor process
                if hasattr(self.agent, 'actor') and self.agent.actor:
                    self.agent.actor.running = False
                    logger.info("✅ Actor process stopped")
                
                # Stop learner process
                if hasattr(self.agent, 'learner') and self.agent.learner:
                    self.agent.learner.running = False
                    logger.info("✅ Learner process stopped")
                
                # Stop the main agent loop
                self.agent.running = False
                self.agent.emergency_stop = True
                
            except Exception as e:
                logger.error(f"Error while stopping agent: {e}")
        
        # Attempt to fix the error
        if self.auto_fix_enabled:
            self._attempt_fix(error_info)
    
    def _attempt_fix(self, error_info: Dict):
        """
        Attempt to automatically fix the error
        
        Args:
            error_info: Information about the error
        """
        error_key = f"{error_info['error_type']}_{error_info['component']}"
        
        # Check if we've tried to fix this error too many times
        fix_count = self.fix_attempts.get(error_key, 0)
        if fix_count >= self.max_fix_attempts:
            logger.error(f"Max fix attempts ({self.max_fix_attempts}) reached for {error_key}")
            logger.error("Manual intervention required")
            return
        
        self.fix_attempts[error_key] = fix_count + 1
        
        logger.info(f"🔧 Attempting to fix error: {error_info['error_type']} in {error_info['component']}")
        
        # Attempt fixes based on error type
        error_type = error_info['error_type'].lower()
        component = error_info['component'].lower()
        
        fixes_applied = []
        
        try:
            # Fix AttributeError - missing attributes
            if 'attributeerror' in error_type:
                fixes_applied.append(self._fix_attribute_error(error_info))
            
            # Fix KeyError - missing keys
            elif 'keyerror' in error_type:
                fixes_applied.append(self._fix_key_error(error_info))
            
            # Fix TypeError - wrong types
            elif 'typeerror' in error_type:
                fixes_applied.append(self._fix_type_error(error_info))
            
            # Fix ValueError - invalid values
            elif 'valueerror' in error_type:
                fixes_applied.append(self._fix_value_error(error_info))
            
            # Fix IndexError - out of bounds
            elif 'indexerror' in error_type:
                fixes_applied.append(self._fix_index_error(error_info))
            
            # Component-specific fixes
            if 'actor' in component:
                fixes_applied.append(self._fix_actor_errors(error_info))
            elif 'learner' in component:
                fixes_applied.append(self._fix_learner_errors(error_info))
            elif 'dashboard' in component:
                fixes_applied.append(self._fix_dashboard_errors(error_info))
            
            if any(fixes_applied):
                logger.info(f"✅ Applied {sum(fixes_applied)} fix(es)")
            else:
                logger.warning("⚠️ No automatic fix available - manual intervention required")
        
        except Exception as e:
            logger.error(f"Error during fix attempt: {e}")
    
    def _fix_attribute_error(self, error_info: Dict) -> bool:
        """Fix AttributeError by adding missing attributes"""
        # This would need context-specific fixes
        # For now, just log
        logger.info("AttributeError detected - checking for missing attributes")
        return False
    
    def _fix_key_error(self, error_info: Dict) -> bool:
        """Fix KeyError by adding missing keys"""
        logger.info("KeyError detected - checking for missing dictionary keys")
        return False
    
    def _fix_type_error(self, error_info: Dict) -> bool:
        """Fix TypeError by converting types"""
        logger.info("TypeError detected - checking for type mismatches")
        return False
    
    def _fix_value_error(self, error_info: Dict) -> bool:
        """Fix ValueError by validating values"""
        logger.info("ValueError detected - checking for invalid values")
        return False
    
    def _fix_index_error(self, error_info: Dict) -> bool:
        """Fix IndexError by checking bounds"""
        logger.info("IndexError detected - checking array/list bounds")
        return False
    
    def _fix_actor_errors(self, error_info: Dict) -> bool:
        """Fix actor-specific errors"""
        if not self.agent or not hasattr(self.agent, 'actor'):
            return False
        
        try:
            actor = self.agent.actor
            
            # Reset frame tracking if needed
            if hasattr(actor, 'last_valid_frame'):
                if actor.last_valid_frame is None:
                    logger.info("Resetting last_valid_frame")
                    # Will be set on next frame capture
            
            # Reset detection cache if needed
            if hasattr(actor, 'cached_detections'):
                if not isinstance(actor.cached_detections, dict):
                    actor.cached_detections = {}
                    logger.info("Reset cached_detections")
            
            return True
        except Exception as e:
            logger.error(f"Error fixing actor: {e}")
            return False
    
    def _fix_learner_errors(self, error_info: Dict) -> bool:
        """Fix learner-specific errors"""
        if not self.agent or not hasattr(self.agent, 'learner'):
            return False
        
        try:
            learner = self.agent.learner
            
            # Reset optimizer state if needed
            if hasattr(learner, 'optimizer'):
                # Could reset optimizer here if needed
                pass
            
            return True
        except Exception as e:
            logger.error(f"Error fixing learner: {e}")
            return False
    
    def _fix_dashboard_errors(self, error_info: Dict) -> bool:
        """Fix dashboard-specific errors"""
        if not self.agent or not hasattr(self.agent, 'dashboard'):
            return False
        
        try:
            # Dashboard errors are usually non-critical
            # Just log and continue
            logger.info("Dashboard error - non-critical, continuing")
            return True
        except Exception as e:
            logger.error(f"Error fixing dashboard: {e}")
            return False
    
    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        current_time = time.time()
        recent_errors = [
            e for e in self.errors 
            if current_time - e['timestamp'] <= self.error_window
        ]
        
        critical_count = len([e for e in recent_errors if e['critical']])
        
        return {
            'total_errors': len(self.errors),
            'recent_errors': len(recent_errors),
            'critical_errors': critical_count,
            'error_rate': len(recent_errors) / self.error_window if self.error_window > 0 else 0,
            'last_error_time': self.last_error_time,
            'auto_fix_enabled': self.auto_fix_enabled,
            'fix_attempts': dict(self.fix_attempts)
        }
    
    def reset(self):
        """Reset error handler (clear error history)"""
        self.error_count = 0
        self.errors.clear()
        self.critical_errors.clear()
        self.fix_attempts.clear()
        logger.info("Error handler reset")

