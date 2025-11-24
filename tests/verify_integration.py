"""
Integration Verification Script
Verifies all components are properly connected and working together
"""
import sys
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_imports():
    """Verify all required modules can be imported"""
    logger.info("Checking imports...")
    try:
        from yolo_self_learning import YOLOSelfLearner
        logger.info("✅ YOLOSelfLearner imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import YOLOSelfLearner: {e}")
        return False
    
    try:
        from actor_process import ActorProcess
        logger.info("✅ ActorProcess imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import ActorProcess: {e}")
        return False
    
    try:
        from dashboard_server import DashboardServer
        logger.info("✅ DashboardServer imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import DashboardServer: {e}")
        return False
    
    return True

def verify_config():
    """Verify configuration is set correctly"""
    logger.info("Checking configuration...")
    
    checks = [
        ("YOLO_SELF_LEARNING_ENABLED", config.YOLO_SELF_LEARNING_ENABLED),
        ("YOLO_MIN_REWARD_FOR_LABELING", config.YOLO_MIN_REWARD_FOR_LABELING),
        ("YOLO_SELF_TRAINING_INTERVAL", config.YOLO_SELF_TRAINING_INTERVAL),
        ("YOLO_CONFIDENCE_ADJUSTMENT_ENABLED", config.YOLO_CONFIDENCE_ADJUSTMENT_ENABLED),
    ]
    
    all_ok = True
    for name, value in checks:
        logger.info(f"  {name}: {value}")
        if value is None:
            logger.warning(f"  ⚠️  {name} is None")
            all_ok = False
    
    return all_ok

def verify_structure():
    """Verify component structure"""
    logger.info("Checking component structure...")
    
    # Check YOLOSelfLearner has required methods
    from yolo_self_learning import YOLOSelfLearner
    
    required_methods = [
        'record_detection_action_pair',
        'adjust_detection_confidence',
        'get_action_guidance_from_detections',
        'save_training_data',
        'train_on_self_labels',
        'get_stats'
    ]
    
    for method in required_methods:
        if hasattr(YOLOSelfLearner, method):
            logger.info(f"  ✅ YOLOSelfLearner.{method} exists")
        else:
            logger.error(f"  ❌ YOLOSelfLearner.{method} missing")
            return False
    
    return True

def verify_dashboard_endpoints():
    """Verify dashboard has required endpoints"""
    logger.info("Checking dashboard endpoints...")
    
    from dashboard_server import DashboardServer
    
    dashboard = DashboardServer()
    
    if not dashboard.app:
        logger.warning("  ⚠️  Dashboard app not initialized (Flask may not be available)")
        return True  # Not a critical error
    
    # Check for YOLO self-learning endpoint
    routes = [str(rule) for rule in dashboard.app.url_map.iter_rules()]
    
    if '/api/yolo_self_learning' in routes:
        logger.info("  ✅ /api/yolo_self_learning endpoint exists")
    else:
        logger.error("  ❌ /api/yolo_self_learning endpoint missing")
        return False
    
    return True

def main():
    """Run all verification checks"""
    logger.info("="*60)
    logger.info("Integration Verification")
    logger.info("="*60)
    
    results = []
    
    results.append(("Imports", verify_imports()))
    results.append(("Configuration", verify_config()))
    results.append(("Structure", verify_structure()))
    results.append(("Dashboard", verify_dashboard_endpoints()))
    
    logger.info("="*60)
    logger.info("Verification Results:")
    logger.info("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("✅ All checks passed! System is ready.")
        return 0
    else:
        logger.error("❌ Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

