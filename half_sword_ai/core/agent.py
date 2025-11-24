"""
Main Orchestrator: Half Sword AI Agent
Brings together all components for autonomous learning agent
"""
import torch
import multiprocessing
import signal
import sys
import time
import logging
import os
from half_sword_ai.config import config
from half_sword_ai.core.model import create_model
from half_sword_ai.learning.replay_buffer import PrioritizedReplayBuffer
from half_sword_ai.input.input_mux import InputMultiplexer
from half_sword_ai.perception.vision import ScreenCapture, MemoryReader
from half_sword_ai.monitoring.watchdog import Watchdog
from half_sword_ai.core.actor import ActorProcess
from half_sword_ai.core.learner import LearnerProcess
# LLM integration removed
from half_sword_ai.monitoring.performance_monitor import PerformanceMonitor
from half_sword_ai.input.kill_switch import KillSwitch
from half_sword_ai.monitoring.dashboard.server import DashboardServer
from half_sword_ai.core.error_handler import ErrorHandler
from half_sword_ai.tools.half_sword_dataset_builder import HalfSwordDatasetBuilder

# Setup comprehensive logging with safe Unicode handling
from half_sword_ai.utils.safe_logger import SafeStreamHandler
log_level = logging.DEBUG if config.DETAILED_LOGGING else logging.INFO

# Create file handler (no Unicode issues with files)
file_handler = logging.FileHandler(f'{config.LOG_PATH}/agent_{int(time.time())}.log')
file_handler.setLevel(log_level)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

# Create safe console handler (handles Windows Unicode issues)
# Auto-detect Windows console encoding
import sys
is_windows = sys.platform == 'win32'
try:
    encoding = sys.stdout.encoding or 'utf-8'
    strip_emojis = is_windows and encoding.lower() in ('cp1252', 'ascii', 'latin-1')
except:
    strip_emojis = is_windows
console_handler = SafeStreamHandler(sys.stdout, strip_emojis=strip_emojis)
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
console_handler.setFormatter(console_formatter)

# Setup root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

class HalfSwordAgent:
    """
    Main agent orchestrator
    Manages all processes and components
    """
    
    def __init__(self):
        self.model = None
        self.replay_buffer = None
        self.input_mux = None
        self.screen_capture = None
        self.memory_reader = None
        self.watchdog = None
        self.actor = None
        self.learner = None
        # LLM removed
        self.performance_monitor = None
        self.kill_switch = None
        self.dashboard = None
        self.error_handler = None
        self.dataset_builder = None
        self.running = False
        self.emergency_stop = False
        self.emergency_stop = False
        
    def initialize(self):
        """Initialize all components"""
        logger.info("="*80)
        logger.info("Initializing Half Sword AI Agent...")
        logger.info("="*80)
        
        # Initialize performance monitor first
        logger.info("Initializing performance monitor...")
        self.performance_monitor = PerformanceMonitor()
        logger.info("Performance monitor ready")
        
        # Initialize error handler early (other components can use it)
        logger.info("Initializing error handler...")
        self.error_handler = ErrorHandler(agent=self, max_errors=10, error_window=60.0)
        logger.info("✅ Error handler initialized - Auto-stop and fix enabled")
        
        # LLM integration removed
        
        # Initialize model (ScrimBrain integration)
        logger.info("Creating neural network model...")
        use_dqn = config.USE_DISCRETE_ACTIONS
        self.model = create_model(use_dqn=use_dqn)
        model_type = "DQN (ScrimBrain-style)" if use_dqn else "PPO (Continuous)"
        logger.info(f"Model created: {model_type} on device: {config.DEVICE}")
        
        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer()
        logger.info("Replay buffer initialized")
        
        # Initialize perception
        logger.info("Initializing perception layer...")
        self.screen_capture = ScreenCapture()
        self.memory_reader = MemoryReader()
        
        # Check if game is running, launch if needed
        if config.AUTO_LAUNCH_GAME and not self.memory_reader.is_process_running():
            logger.info("Game process not detected. Attempting to launch game...")
            self.memory_reader._launch_game_if_needed()
            # Wait a bit for game to start
            import time
            time.sleep(3.0)
            # Try attaching again
            if hasattr(self.memory_reader, '_attach_to_process'):
                self.memory_reader._attach_to_process()
        
        logger.info("Perception layer ready")
        
        # Initialize input
        logger.info("Initializing input multiplexer...")
        self.input_mux = InputMultiplexer()
        
        # Log interception status
        from half_sword_ai.input.input_mux import INTERCEPTION_AVAILABLE
        if INTERCEPTION_AVAILABLE and hasattr(self.input_mux, 'interception') and self.input_mux.interception:
            logger.info("✅ Input multiplexer ready - Using Interception driver (kernel-level)")
        else:
            logger.info("✅ Input multiplexer ready - Using DirectInput (fallback mode)")
            logger.info("   Note: Interception driver not installed (optional)")
            logger.info("   See INTERCEPTION_INSTALL.md if you want kernel-level control")
        
        # Initialize watchdog
        logger.info("Initializing watchdog...")
        self.watchdog = Watchdog(self.memory_reader, self.screen_capture)
        logger.info("Watchdog ready")
        
        # Initialize actor
        logger.info("Initializing actor process...")
        self.actor = ActorProcess(
            self.model, self.replay_buffer, self.input_mux,
            self.memory_reader, self.screen_capture, self.watchdog,
            self.performance_monitor
        )
        # Pass error handler to actor (if initialized)
        if self.error_handler:
            self.actor.error_handler = self.error_handler
        logger.info("Actor process ready")
        
        # Initialize learner (before dashboard so dashboard can access it)
        logger.info("Initializing learner process...")
        self.learner = LearnerProcess(self.model, self.replay_buffer, self.performance_monitor)
        # Pass error handler to learner (if initialized)
        if self.error_handler:
            self.learner.error_handler = self.error_handler
        logger.info("Learner process ready")
        
        # Start performance monitoring
        self.performance_monitor.start_episode()
        logger.info("Performance monitoring started")
        
        # Initialize kill switch
        logger.info("Initializing kill switch...")
        self.kill_switch = KillSwitch(
            kill_callback=self._emergency_kill,
            hotkey=config.KILL_BUTTON
        )
        self.kill_switch.start()
        logger.info(f"✅ Kill switch active - Press {config.KILL_BUTTON.upper()} for emergency stop")
        
        # Initialize enhanced dataset builder (if enabled)
        if config.ENABLE_DATASET_COLLECTION:
            logger.info("Initializing enhanced dataset builder...")
            try:
                dataset_name = f"{config.DATASET_NAME_PREFIX}_{int(time.time())}"
                self.dataset_builder = HalfSwordDatasetBuilder(
                    output_dir=os.path.join(config.DATA_SAVE_PATH, "half_sword_datasets"),
                    dataset_name=dataset_name
                )
                # Pass dataset builder to actor
                self.actor.dataset_builder = self.dataset_builder
                # Start recording based on mode
                if config.DATASET_COLLECTION_MODE == "continuous":
                    self.dataset_builder.start_recording()
                    logger.info("[OK] Dataset collection started - Continuous mode")
                else:
                    logger.info("[OK] Dataset builder ready - Will start on trigger")
            except Exception as e:
                logger.error(f"Failed to initialize dataset builder: {e}", exc_info=True)
                self.dataset_builder = None
        else:
            logger.info("Dataset collection disabled (set ENABLE_DATASET_COLLECTION=True to enable)")
        
        # Initialize dashboard (after all components are ready)
        logger.info("Initializing dashboard server...")
        try:
            self.dashboard = DashboardServer(
                performance_monitor=self.performance_monitor,
                input_mux=self.input_mux,
                actor=self.actor,
                kill_switch=self.kill_switch,
                vision_processor=self.actor.vision_processor if hasattr(self.actor, 'vision_processor') else None,
                learner=self.learner
            )
            
            # Set human recorder reference if available
            if hasattr(self.actor, 'human_recorder'):
                self.dashboard.human_recorder = self.actor.human_recorder
            
            # Start dashboard server
            if self.dashboard.app is not None:
                self.dashboard.start()
                # Wait a moment to verify it started
                import time
                time.sleep(0.5)
                if self.dashboard.running:
                    logger.info("✅ Dashboard server started successfully")
                else:
                    logger.warning("⚠️  Dashboard server started but may not be running")
            else:
                logger.warning("⚠️  Dashboard server app not initialized - check Flask installation")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}", exc_info=True)
            self.dashboard = None
        
        # Final integration checks
        logger.info("Performing integration checks...")
        
        # Verify input mux integration
        if self.input_mux and hasattr(self.actor, 'human_recorder'):
            if not hasattr(self.input_mux, 'human_action_recorder'):
                logger.warning("Input mux missing human_action_recorder reference")
            else:
                logger.info("✅ Input mux ↔ Human recorder integration verified")
        
        # Verify learner integration with model tracker
        if self.learner and hasattr(self.learner, 'model_tracker'):
            logger.info("✅ Learner ↔ Model tracker integration verified")
        
        # Verify replay buffer integration
        if self.replay_buffer and self.actor and self.learner:
            logger.info(f"✅ Replay buffer shared: {len(self.replay_buffer)} experiences")
        
        logger.info("Initialization complete!")
        logger.info("")
        logger.info("🔗 SYSTEM INTEGRATION STATUS:")
        logger.info(f"  - Input System: Enhanced with adaptive detection")
        logger.info(f"  - Learning System: Advanced DAgger + Curriculum Learning")
        logger.info(f"  - Performance Monitoring: Comprehensive metrics tracking")
        logger.info(f"  - All components: Fully integrated and ready")
        logger.info("")
    
    def start(self):
        """Start the agent"""
        if self.running:
            logger.warning("Agent already running")
            return
        
        print("\n" + "="*80)
        logger.info("🚀 Starting Half Sword AI Agent...")
        print("="*80 + "\n")
        
        print("📖 CONTROLS:")
        print("   🖱️  Move mouse → Take manual control (bot pauses)")
        print("   ⏸️  Stop moving → Bot resumes after 0.5s")
        print("   ⌨️  Ctrl+C → Stop and generate report\n")
        
        print("🎥 HUMAN-IN-THE-LOOP RECORDING:")
        print("   ✅ ALL your actions are being recorded!")
        print("   ✅ Mouse movements, clicks, keyboard presses captured")
        print("   ✅ Bot learns from your gameplay to emulate your style")
        print("   ✅ Recordings saved to data/ directory")
        print("   ✅ Expert buffer stores recent actions for immediate learning\n")
        
        print("🔄 CONTINUOUS ONLINE LEARNING:")
        print("   ⚡ Model trains in REAL-TIME as you play!")
        print("   ⚡ Your actions train the bot immediately")
        print("   ⚡ Model updates applied instantly")
        print("   ⚡ Human actions prioritized 5x for faster learning")
        print("   ⚡ Training every 0.05s (20 Hz) when human actions present\n")
        
        print(f"🛑 EMERGENCY KILL SWITCH:")
        print(f"   Press {config.KILL_BUTTON.upper()} → IMMEDIATELY stop bot")
        print("   Works instantly, even during active bot control\n")
        
        print(f"🌐 DASHBOARD:")
        print(f"   http://localhost:{self.dashboard.port if self.dashboard else 5000}")
        print("   Real-time monitoring of all metrics\n")
        
        print("🛡️ SAFETY FEATURES:")
        print("   ✅ Bot NEVER injects when you move your mouse")
        print("   ✅ Human input detection prevents interference")
        print("   ✅ Kill switch for instant emergency stop")
        print("   ✅ Performance monitoring tracks all metrics\n")
        
        print("="*80 + "\n")
        
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start learner in separate thread
            import threading
            learner_thread = threading.Thread(target=self.learner.start, daemon=True)
            learner_thread.start()
            
            # Start actor (main loop) - it runs in its own thread
            actor_thread = threading.Thread(target=self.actor.start, daemon=True)
            actor_thread.start()
            
            # Main monitoring loop - check for kill switch
            while self.running and not self.emergency_stop:
                # Check kill switch status periodically
                if self.kill_switch and self.kill_switch.is_killed():
                    logger.critical("Kill switch detected - initiating emergency shutdown")
                    self._emergency_kill()
                    break
                
                # Check error handler for critical errors (auto-stop on errors)
                if self.error_handler and len(self.error_handler.critical_errors) > 0:
                    logger.critical("🚨 Critical errors detected - stopping game automatically")
                    self.emergency_stop = True
                    self._emergency_kill()
                    break
                
                # Check if actor is still running
                if not self.actor.running:
                    logger.warning("Actor process stopped unexpectedly")
                    break
                
                time.sleep(0.1)  # Small delay to allow kill switch and error checking
            
            # If we get here, either normal shutdown or emergency stop
            if self.kill_switch and self.kill_switch.is_killed():
                logger.critical("Emergency stop completed")
            
            # Wait for threads to finish
            actor_thread.join(timeout=2.0)
            learner_thread.join(timeout=2.0)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            # Record fatal error in error handler
            if self.error_handler:
                self.error_handler.record_error(e, context="agent_main_loop", component="agent", stop_game=True)
        finally:
            self.shutdown()
    
    def _emergency_kill(self):
        """Emergency kill function - called by kill switch"""
        logger.critical("="*80)
        logger.critical("EMERGENCY KILL ACTIVATED")
        logger.critical("="*80)
        
        self.emergency_stop = True
        self.running = False
        
        # Immediately enable safety lock
        if self.input_mux:
            logger.critical("Enabling safety lock - bot input completely disabled")
            self.input_mux.enable_safety_lock()
            self.input_mux.force_manual_mode()
            self.input_mux.stop()
        
        # Stop all processes
        if self.actor:
            logger.critical("Stopping actor process...")
            self.actor.stop()
        
        if self.learner:
            logger.critical("Stopping learner process...")
            self.learner.stop()
        
        if self.screen_capture:
            self.screen_capture.stop()
        
        # Record kill event
        if self.performance_monitor:
            self.performance_monitor.record_warning("EMERGENCY KILL ACTIVATED", "kill_switch")
            self.performance_monitor.end_episode("emergency_kill")
        
        # Stop dataset builder if active
        if self.dataset_builder and self.dataset_builder.recording:
            logger.info("Stopping dataset collection...")
            self.dataset_builder.stop_recording()
        
        logger.critical("Emergency shutdown complete - bot is now safe")
        logger.critical("="*80)
    
    def shutdown(self):
        """Shutdown all components"""
        if self.emergency_stop:
            logger.info("Shutdown already initiated by kill switch")
            return
        
        logger.info("Shutting down agent...")
        self.running = False
        
        # Stop kill switch
        if self.kill_switch:
            self.kill_switch.stop()
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop()
        
        if self.actor:
            self.actor.stop()
        
        if self.learner:
            self.learner.stop()
        
        if self.input_mux:
            self.input_mux.stop()
        
        if self.screen_capture:
            self.screen_capture.stop()
        
        # Stop and save dataset builder if active
        if self.dataset_builder and self.dataset_builder.recording:
            logger.info("Stopping dataset collection and saving...")
            self.dataset_builder.stop_recording()
        
        # Save model
        if self.model:
            self._save_model()
        
        # Generate final performance report
        if self.performance_monitor:
            logger.info("Generating final performance report...")
            final_report = self.performance_monitor.generate_report(
                f"{config.LOG_PATH}/final_performance_report.txt"
            )
            logger.info(final_report)
            
            # Save metrics JSON
            self.performance_monitor.save_metrics_json(
                f"{config.LOG_PATH}/final_metrics.json"
            )
        
        logger.info("Shutdown complete")
    
    def _save_model(self):
        """Save model checkpoint"""
        try:
            import os
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            save_path = f"{config.MODEL_SAVE_PATH}/model_checkpoint.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': config.__dict__
            }, save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Half Sword AI Agent - Autonomous Learning Agent      ║
    ║     Autonomous Learning Agent for Physics Combat         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    agent = HalfSwordAgent()
    agent.initialize()
    agent.start()

if __name__ == "__main__":
    main()

