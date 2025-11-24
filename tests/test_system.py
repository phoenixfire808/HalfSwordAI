"""
Quick system test - verify all components work
"""
import sys
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports"""
    logger.info("Testing imports...")
    try:
        from main import HalfSwordAgent
        from actor_process import ActorProcess
        from ollama_integration import OllamaQwenAgent
        from dashboard_server import DashboardServer
        from yolo_self_learning import YOLOSelfLearner
        from perception_layer import MemoryReader
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_initialization():
    """Test component initialization"""
    logger.info("Testing initialization...")
    try:
        from main import HalfSwordAgent
        agent = HalfSwordAgent()
        logger.info("✅ Agent created")
        
        # Test initialization (but don't start)
        logger.info("Initializing components...")
        agent.initialize()
        logger.info("✅ Initialization successful")
        
        # Cleanup
        agent.shutdown()
        return True
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_endpoints():
    """Test dashboard endpoints exist"""
    logger.info("Testing dashboard endpoints...")
    try:
        from dashboard_server import DashboardServer
        dashboard = DashboardServer()
        
        if not dashboard.app:
            logger.warning("⚠️  Dashboard app not initialized (Flask may not be available)")
            return True
        
        routes = [str(rule) for rule in dashboard.app.url_map.iter_rules()]
        required_routes = [
            '/api/stats',
            '/api/game_status',
            '/api/llm_communication',
            '/api/yolo_self_learning'
        ]
        
        missing = [r for r in required_routes if r not in routes]
        if missing:
            logger.error(f"❌ Missing routes: {missing}")
            return False
        
        logger.info("✅ All required dashboard endpoints exist")
        return True
    except Exception as e:
        logger.error(f"❌ Dashboard test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("="*60)
    logger.info("SYSTEM TEST")
    logger.info("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dashboard Endpoints", test_dashboard_endpoints()))
    # Skip initialization test as it requires full system
    # results.append(("Initialization", test_initialization()))
    
    logger.info("="*60)
    logger.info("TEST RESULTS:")
    logger.info("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

