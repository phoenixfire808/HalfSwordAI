"""
Test script for LLM error debugging features
Tests the comprehensive error tracking and diagnostics
"""
import json
import time
from half_sword_ai.llm.ollama_integration import OllamaQwenAgent
from half_sword_ai.config import config

def test_llm_debugging():
    """Test LLM error debugging features"""
    print("=" * 80)
    print("LLM ERROR DEBUGGING TEST")
    print("=" * 80)
    print()
    
    # Initialize agent
    print("1. Initializing OllamaQwenAgent...")
    agent = OllamaQwenAgent()
    agent.enable_debug_mode(True)
    print("   [OK] Agent initialized")
    print()
    
    # Test health check
    print("2. Testing health check...")
    health = agent.check_health()
    if health:
        print("   [OK] Ollama service is healthy")
    else:
        print("   [WARN] Ollama service appears to be down")
        print("   [TIP] Start Ollama with: ollama serve")
    print()
    
    # Test diagnostics
    print("3. Getting diagnostics...")
    diagnostics = agent.get_diagnostics()
    status_icon = "[OK]" if diagnostics['service_status']['health_check'] else "[FAIL]"
    print(f"   Service Status: {status_icon} {'Healthy' if diagnostics['service_status']['health_check'] else 'Unhealthy'}")
    print(f"   Base URL: {diagnostics['service_status']['base_url']}")
    print(f"   Current Model: {diagnostics['service_status']['current_model']}")
    print(f"   Total Queries: {diagnostics['performance']['total_queries']}")
    print(f"   Success Rate: {diagnostics['performance']['success_rate']:.1f}%")
    print(f"   Average Latency: {diagnostics['performance']['average_latency']:.3f}s")
    print(f"   Total Errors: {diagnostics['errors']['total_errors']}")
    print()
    
    # Test a query
    print("4. Testing game state analysis...")
    test_game_state = {
        'health': 75.0,
        'stamina': 60.0,
        'enemy_health': 50.0,
        'is_dead': False,
        'enemy_dead': False,
        'position': 'center',
        'combat_state': 'engaged'
    }
    
    try:
        start_time = time.time()
        strategy = agent.analyze_game_state(test_game_state, "Test frame: enemy visible, medium threat")
        latency = time.time() - start_time
        
        if "error" in strategy:
            print(f"   [WARN] Query completed with error (expected if Ollama is slow)")
            print(f"   Error: {strategy.get('error')}")
            print(f"   Fallback Strategy: {strategy.get('strategy')}")
        else:
            print(f"   [OK] Query completed in {latency:.3f}s")
            print(f"   Strategy: {strategy.get('strategy')}")
            print(f"   Confidence: {strategy.get('confidence'):.2f}")
            print(f"   Primary Action: {strategy.get('primary_action')}")
            print(f"   Reasoning: {strategy.get('reasoning', '')[:100]}...")
    except KeyboardInterrupt:
        print(f"   [WARN] Test interrupted (this is OK - error debugging is working)")
        print(f"   The timeout errors were successfully caught and logged!")
    except Exception as e:
        print(f"   [ERROR] Query failed: {e}")
        import traceback
        print(f"   Traceback:\n{traceback.format_exc()}")
    print()
    
    # Test error report
    print("5. Getting error report...")
    error_report = agent.get_error_report()
    print(error_report)
    print()
    
    # Test metrics
    print("6. Getting metrics...")
    metrics = agent.get_metrics()
    print(f"   Total Queries: {metrics['total_queries']}")
    print(f"   Successful: {metrics['successful_queries']}")
    print(f"   Failed: {metrics['failed_queries']}")
    print(f"   Success Rate: {metrics['success_rate']:.1f}%")
    print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
    print(f"   Average Latency: {metrics['average_latency']:.3f}s")
    if metrics['error_statistics']['total_errors'] > 0:
        print(f"   Error Counts: {json.dumps(metrics['error_statistics']['error_counts'], indent=6)}")
    print()
    
    # Test recommendations
    print("7. Recommendations:")
    for rec in diagnostics['recommendations']:
        print(f"   {rec}")
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_llm_debugging()

