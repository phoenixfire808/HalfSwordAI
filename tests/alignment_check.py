"""
Alignment Check: Verify all components are properly integrated
Checks imports, dependencies, and component connections
"""
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_imports():
    """Check all critical imports"""
    modules = [
        'config',
        'neural_network',
        'replay_buffer',
        'input_mux',
        'perception_layer',
        'watchdog',
        'actor_process',
        'learner_process',
        'ollama_integration',
        'performance_monitor',
        'kill_switch',
        'dashboard_server',
        'human_recorder',
        'yolo_detector',
        'model_tracker'
    ]
    
    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
            logger.info(f"✅ {module}")
        except Exception as e:
            logger.error(f"❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def check_config():
    """Check configuration completeness"""
    from config import config
    
    required_attrs = [
        'OLLAMA_BASE_URL', 'OLLAMA_MODEL',
        'CAPTURE_FPS', 'CAPTURE_WIDTH', 'CAPTURE_HEIGHT',
        'YOLO_ENABLED', 'YOLO_CONFIDENCE_THRESHOLD',
        'MOUSE_SENSITIVITY', 'NOISE_THRESHOLD', 'HUMAN_TIMEOUT',
        'KILL_BUTTON',
        'BATCH_SIZE', 'LEARNING_RATE', 'BETA_BC',
        'MIN_BATCH_SIZE_FOR_TRAINING', 'TRAINING_FREQUENCY',
        'HUMAN_ACTION_PRIORITY_MULTIPLIER',
        'ALWAYS_RECORD_HUMAN', 'RECORDING_MODE',
        'MODEL_SAVE_PATH', 'DATA_SAVE_PATH', 'LOG_PATH',
        'DETAILED_LOGGING', 'PERFORMANCE_REPORT_INTERVAL'
    ]
    
    missing = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing.append(attr)
            logger.error(f"❌ Missing config: {attr}")
        else:
            logger.debug(f"✅ Config: {attr}")
    
    return len(missing) == 0

def check_component_integration():
    """Check component integration"""
    issues = []
    
    try:
        from config import config
        from neural_network import create_model
        from replay_buffer import PrioritizedReplayBuffer
        from input_mux import InputMultiplexer
        from perception_layer import ScreenCapture, MemoryReader, VisionProcessor
        from watchdog import Watchdog
        from actor_process import ActorProcess
        from learner_process import LearnerProcess
        from ollama_integration import OllamaQwenAgent
        from performance_monitor import PerformanceMonitor
        from kill_switch import KillSwitch
        from dashboard_server import DashboardServer
        from human_recorder import HumanActionRecorder
        from yolo_detector import YOLODetector
        from model_tracker import ModelTracker
        
        logger.info("✅ All components importable")
        
        # Check model creation
        try:
            model = create_model()
            logger.info("✅ Model creation works")
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            issues.append("model_creation")
        
        # Check buffer creation
        try:
            buffer = PrioritizedReplayBuffer()
            logger.info("✅ Replay buffer creation works")
        except Exception as e:
            logger.error(f"❌ Buffer creation failed: {e}")
            issues.append("buffer_creation")
        
    except Exception as e:
        logger.error(f"❌ Component integration check failed: {e}")
        issues.append("integration")
    
    return len(issues) == 0

def main():
    """Run all alignment checks"""
    print("="*80)
    print("ALIGNMENT CHECK - Verifying System Integration")
    print("="*80)
    print()
    
    results = {
        'imports': check_imports(),
        'config': check_config(),
        'integration': check_component_integration()
    }
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print()
        print("✅ All alignment checks passed!")
        print("System is ready to run.")
    else:
        print()
        print("❌ Some alignment checks failed.")
        print("Please review the errors above.")
        sys.exit(1)
    
    print("="*80)

if __name__ == "__main__":
    main()

