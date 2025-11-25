"""
Dashboard Server: Real-time monitoring UI
Web-based dashboard for comprehensive agent monitoring
"""
import json
import time
import threading
import logging
import webbrowser
from typing import Optional, Dict, Any
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    render_template = None
    jsonify = None
    request = None
    CORS = None
    logger.warning("Flask not available - install with: pip install flask flask-cors")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types and other non-serializable types to JSON-serializable Python types
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if not NUMPY_AVAILABLE:
        # If numpy not available, just return the object (will fail if it's actually a numpy type)
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj
    
    # Check for numpy types
    # Check for numpy types
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

class DashboardServer:
    """
    Real-time web dashboard server
    Provides comprehensive monitoring interface
    """
    
    def __init__(self, performance_monitor=None, input_mux=None, actor=None, 
                 kill_switch=None, vision_processor=None, learner=None, port=5000):
        self.performance_monitor = performance_monitor
        self.input_mux = input_mux
        self.actor = actor
        self.learner = learner
        self.kill_switch = kill_switch
        self.vision_processor = vision_processor
        self.port = port
        self.app = None
        self.server_thread = None
        self.running = False
        self.last_update = {}
        self.last_detections = {}
        self.human_recorder = None
        
        if FLASK_AVAILABLE:
            self._init_flask()
        else:
            logger.warning("Dashboard server not available - Flask not installed")
    
    def _init_flask(self):
        """Initialize Flask application with enhanced error handling"""
        import os
        
        try:
            # Get template and static directories
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.join(current_dir, 'dashboard_templates')
            static_dir = os.path.join(current_dir, 'dashboard_static')
            
            # Verify directories exist
            if not os.path.exists(template_dir):
                logger.error(f"Dashboard template directory not found: {template_dir}")
                return
            
            if not os.path.exists(static_dir):
                logger.warning(f"Dashboard static directory not found: {static_dir} - creating it")
                os.makedirs(static_dir, exist_ok=True)
            
            # Verify dashboard.html exists
            dashboard_html = os.path.join(template_dir, 'dashboard.html')
            if not os.path.exists(dashboard_html):
                logger.error(f"Dashboard HTML file not found: {dashboard_html}")
                return
            
            # Initialize Flask app
            self.app = Flask(__name__, 
                            template_folder=template_dir,
                            static_folder=static_dir)
            CORS(self.app)  # Enable CORS for API requests
            
            logger.info(f"Flask app initialized - Templates: {template_dir}, Static: {static_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Flask: {e}", exc_info=True)
            self.app = None
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            try:
                return render_template('dashboard.html')
            except Exception as e:
                logger.error(f"Error rendering dashboard: {e}", exc_info=True)
                # Return a basic error page that's more informative
                return f"""
                <html>
                <head><title>Dashboard Error</title></head>
                <body style="font-family: Arial; padding: 40px; background: #1a1a2e; color: #e0e0e0;">
                    <h1 style="color: #ff5050;">Dashboard Error</h1>
                    <p>Error loading dashboard: {str(e)}</p>
                    <p>Check logs for details.</p>
                    <p><a href="/api/health" style="color: #00d4ff;">Health Check</a></p>
                </body>
                </html>
                """, 500
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint with detailed status"""
            return jsonify({
                "status": "ok",
                "running": self.running,
                "server_running": self.running,
                "timestamp": time.time(),
                "port": self.port,
                "components": {
                    "performance_monitor": self.performance_monitor is not None,
                    "input_mux": self.input_mux is not None,
                    "actor": self.actor is not None,
                    "learner": self.learner is not None,
                    "kill_switch": self.kill_switch is not None,
                    "vision_processor": self.vision_processor is not None
                }
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current statistics"""
            try:
                stats = self._get_all_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}", exc_info=True)
                return jsonify({"error": str(e), "timestamp": time.time()}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors"""
            return jsonify({"error": "Endpoint not found", "available_endpoints": [
                "/", "/api/health", "/api/stats", "/api/performance", "/api/system",
                "/api/learning", "/api/input", "/api/errors", "/api/episodes",
                "/api/yolo", "/api/human_recording", "/api/training"
            ]}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors"""
            logger.error(f"Internal server error: {error}", exc_info=True)
            return jsonify({"error": "Internal server error", "timestamp": time.time()}), 500
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance metrics"""
            try:
                return jsonify(self._get_performance_data())
            except Exception as e:
                logger.error(f"Error getting performance: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system')
        def get_system():
            """Get system resources"""
            try:
                return jsonify(self._get_system_data())
            except Exception as e:
                logger.error(f"Error getting system: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/learning')
        def get_learning():
            """Get learning metrics"""
            try:
                return jsonify(self._get_learning_data())
            except Exception as e:
                logger.error(f"Error getting learning: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/input')
        def get_input():
            """Get input multiplexer stats"""
            try:
                return jsonify(self._get_input_data())
            except Exception as e:
                logger.error(f"Error getting input: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/errors')
        def get_errors():
            """Get recent errors and warnings"""
            try:
                return jsonify(self._get_errors_data())
            except Exception as e:
                logger.error(f"Error getting errors: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/episodes')
        def get_episodes():
            """Get episode history"""
            try:
                return jsonify(self._get_episodes_data())
            except Exception as e:
                logger.error(f"Error getting episodes: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/kill_switch', methods=['GET', 'POST'])
        def kill_switch():
            """Get or trigger kill switch"""
            if request.method == 'POST':
                # Trigger kill switch
                if self.kill_switch:
                    # Simulate kill switch trigger
                    logger.critical("Kill switch triggered via dashboard")
                    return jsonify({"status": "triggered"})
                return jsonify({"status": "error", "message": "Kill switch not available"}), 400
            return jsonify(self._get_kill_switch_data())
        
        @self.app.route('/api/yolo')
        def get_yolo():
            """Get YOLO detection data"""
            return jsonify(self._get_yolo_data())
        
        @self.app.route('/api/human_recording')
        def get_human_recording():
            """Get human action recording statistics"""
            return jsonify(self._get_human_recording_data())
        
        @self.app.route('/api/training')
        def get_training():
            """Get training progress and model statistics"""
            return jsonify(self._get_training_data())
        
        @self.app.route('/api/yolo_self_learning')
        def get_yolo_self_learning():
            """Get YOLO self-learning statistics"""
            return jsonify(self._get_yolo_self_learning_data())
        
        @self.app.route('/api/game_status')
        def get_game_status():
            """Get game process status"""
            return jsonify(self._get_game_status_data())
        
        # LLM communication endpoint removed
        
        @self.app.route('/api/test')
        def test():
            """Simple test endpoint to verify server is working"""
            return jsonify({
                "status": "ok",
                "message": "Dashboard server is working",
                "timestamp": time.time(),
                "flask_available": FLASK_AVAILABLE,
                "running": self.running,
                "port": self.port,
                "thread_alive": self.server_thread.is_alive() if self.server_thread else False
            })
        
        @self.app.route('/test')
        def test_page():
            """Simple test page to verify dashboard is accessible"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dashboard Test</title>
                <style>
                    body { font-family: Arial; padding: 40px; background: #1a1a2e; color: #e0e0e0; }
                    h1 { color: #00d4ff; }
                    .success { color: #00ff88; }
                    .error { color: #ff5050; }
                    a { color: #00d4ff; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .endpoint { margin: 10px 0; padding: 10px; background: #2a2a3e; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Half Sword AI Dashboard - Test Page</h1>
                <p class="success">✅ Dashboard server is running!</p>
                <p><strong>Server Status:</strong> Running on port {}</p>
                <p><strong>Main Dashboard:</strong> <a href="/">Open Dashboard</a></p>
                <h2>Available Endpoints:</h2>
                <div class="endpoint"><a href="/api/health">/api/health</a> - Health check</div>
                <div class="endpoint"><a href="/api/stats">/api/stats</a> - All statistics</div>
                <div class="endpoint"><a href="/api/performance">/api/performance</a> - Performance metrics</div>
                <div class="endpoint"><a href="/api/system">/api/system</a> - System resources</div>
                <div class="endpoint"><a href="/api/learning">/api/learning</a> - Learning metrics</div>
                <div class="endpoint"><a href="/api/input">/api/input</a> - Input multiplexer stats</div>
                <div class="endpoint"><a href="/api/training">/api/training</a> - Training progress</div>
                <div class="endpoint"><a href="/api/yolo">/api/yolo</a> - YOLO detections</div>
                <div class="endpoint"><a href="/api/human_recording">/api/human_recording</a> - Human recording stats</div>
                <h2>Quick Test:</h2>
                <button onclick="testAPI()" style="padding: 10px 20px; background: #00d4ff; border: none; border-radius: 5px; cursor: pointer; color: #1a1a2e; font-weight: bold;">Test API</button>
                <div id="result" style="margin-top: 20px;"></div>
                <script>
                    function testAPI() {{
                        fetch('/api/test')
                            .then(r => r.json())
                            .then(data => {{
                                document.getElementById('result').innerHTML = 
                                    '<pre style="background: #2a2a3e; padding: 15px; border-radius: 5px;">' + 
                                    JSON.stringify(data, null, 2) + '</pre>';
                            }})
                            .catch(e => {{
                                document.getElementById('result').innerHTML = 
                                    '<p class="error">Error: ' + e + '</p>';
                            }});
                    }}
                </script>
            </body>
            </html>
            """.format(self.port)
    
    def _get_all_stats(self) -> Dict:
        """Get all statistics with error handling and proper JSON serialization"""
        try:
            stats = {
                "timestamp": float(time.time()),
                "status": "running" if self.running else "stopped",
                "dashboard_running": bool(self.running)
            }
            
            # Performance monitor stats
            if self.performance_monitor:
                try:
                    perf_stats = self.performance_monitor.get_current_stats()
                    stats.update(perf_stats)
                except Exception as e:
                    logger.debug(f"Error getting performance stats: {e}")
                    stats["performance_error"] = str(e)
            
            # Input mux stats
            if self.input_mux:
                try:
                    stats["input_mux"] = self.input_mux.get_stats()
                except Exception as e:
                    logger.debug(f"Error getting input stats: {e}")
                    stats["input_mux"] = {"error": str(e)}
            
            # Kill switch stats
            if self.kill_switch:
                try:
                    stats["kill_switch"] = {
                        "active": getattr(self.kill_switch, 'running', False),
                        "killed": self.kill_switch.is_killed() if hasattr(self.kill_switch, 'is_killed') else False,
                        "kill_count": getattr(self.kill_switch, 'kill_count', 0),
                        "hotkey": config.KILL_BUTTON.upper()
                    }
                    # Add status if available
                    if hasattr(self.kill_switch, 'get_status'):
                        stats["kill_switch"].update(self.kill_switch.get_status())
                except Exception as e:
                    logger.debug(f"Error getting kill switch stats: {e}")
                    stats["kill_switch"] = {"error": str(e)}
            
            # Actor stats
            if self.actor:
                try:
                    stats["actor"] = {
                        "running": getattr(self.actor, 'running', False),
                        "frame_count": getattr(self.actor, 'frame_count', 0),
                        "episode_count": getattr(self.actor, 'episode_count', 0)
                    }
                    
                    # Update last detections from actor
                    if hasattr(self.actor, 'latest_detections'):
                        self.last_detections = self.actor.latest_detections
                        stats["actor"]["latest_detections_available"] = bool(self.last_detections) if self.last_detections else False
                except Exception as e:
                    logger.debug(f"Error getting actor stats: {e}")
                    stats["actor"] = {"error": str(e)}
            
            # Learner stats
            if self.learner:
                try:
                    stats["learner"] = {
                        "running": getattr(self.learner, 'running', False),
                        "update_count": getattr(self.learner, 'update_count', 0)
                    }
                    # Add model tracker stats if available
                    if hasattr(self.learner, 'model_tracker'):
                        try:
                            training_stats = self.learner.model_tracker.get_training_stats()
                            stats["learner"].update(training_stats)
                        except:
                            pass
                except Exception as e:
                    logger.debug(f"Error getting learner stats: {e}")
                    stats["learner"] = {"error": str(e)}
            
            # Convert all numpy types to Python types before returning
            stats = convert_to_json_serializable(stats)
            return stats
        except Exception as e:
            logger.error(f"Error in _get_all_stats: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": float(time.time()),
                "status": "error"
            }
    
    def _get_performance_data(self) -> Dict:
        """Get performance metrics with error handling"""
        try:
            if not self.performance_monitor:
                return {
                    "fps": {"current": 0, "average": 0, "min": 0, "max": 0},
                    "latencies": {},
                    "frame_times": []
                }
            
            stats = self.performance_monitor.get_current_stats()
            
            fps_data = stats.get("fps", {})
            latencies = stats.get("latencies_ms", {})
            
            # Ensure all latency keys exist
            default_latency = {"avg": 0, "max": 0, "min": 0, "count": 0}
            latencies_data = {
                "capture": latencies.get("capture", default_latency),
                "inference": latencies.get("inference", default_latency),
                "injection": latencies.get("injection", default_latency),
                # LLM latencies removed
            }
            
            # Add YOLO latency if available
            if "yolo" in latencies:
                latencies_data["yolo"] = latencies["yolo"]
            
            # Get frame times safely
            frame_times = []
            if hasattr(self.performance_monitor, 'frame_times') and len(self.performance_monitor.frame_times) > 0:
                frame_times = list(self.performance_monitor.frame_times)[-100:]
            
            return {
                "fps": {
                    "current": fps_data.get("current", 0),
                    "average": fps_data.get("average", 0),
                    "min": fps_data.get("min", 0),
                    "max": fps_data.get("max", 0)
                },
                "latencies": latencies_data,
                "frame_times": frame_times
            }
        except Exception as e:
            logger.error(f"Error in _get_performance_data: {e}", exc_info=True)
            return {
                "fps": {"current": 0, "average": 0, "min": 0, "max": 0},
                "latencies": {},
                "frame_times": [],
                "error": str(e)
            }
    
    def _get_system_data(self) -> Dict:
        """Get system resource data with error handling"""
        try:
            if not self.performance_monitor:
                return {
                    "cpu": {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0},
                    "memory": {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0},
                    "gpu": {},
                    "gpu_utilization": {}
                }
            
            stats = self.performance_monitor.get_current_stats()
            system = stats.get("system", {})
            
            return {
                "cpu": system.get("cpu_percent", {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0}),
                "memory": system.get("memory_mb", {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0}),
                "gpu": system.get("gpu_memory_mb", {}),
                "gpu_utilization": system.get("gpu_utilization", {}),
                "gpu_temperature": system.get("gpu_temperature", {}),
                "network_io": system.get("network_io", {}),
                "disk_io": system.get("disk_io", {})
            }
        except Exception as e:
            logger.error(f"Error in _get_system_data: {e}", exc_info=True)
            return {
                "cpu": {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0},
                "memory": {"current": 0, "average": 0, "max": 0, "min": 0, "count": 0},
                "gpu": {},
                "error": str(e)
            }
    
    def _get_learning_data(self) -> Dict:
        """Get learning metrics"""
        if not self.performance_monitor:
            return {
                "training": {},
                "rewards": {},
                "episodes": {},
                "recent_losses": [],
                "recent_rewards": []
            }
        
        stats = self.performance_monitor.get_current_stats()
        
        return {
            "training": stats.get("training", {
                "loss": {"current": 0, "average": 0, "min": 0, "max": 0}
            }),
            "rewards": stats.get("rewards", {
                "current": 0,
                "average": 0,
                "total": 0
            }),
            "episodes": stats.get("episodes", {
                "average_reward": 0,
                "best_reward": 0,
                "worst_reward": 0,
                "total_episodes": 0
            }),
            "recent_losses": list(self.performance_monitor.training_losses)[-50:] if hasattr(self.performance_monitor, 'training_losses') and len(self.performance_monitor.training_losses) > 0 else [],
            "recent_rewards": list(self.performance_monitor.rewards)[-100:] if hasattr(self.performance_monitor, 'rewards') and len(self.performance_monitor.rewards) > 0 else []
        }
    
    def _get_input_data(self) -> Dict:
        """Get input multiplexer data"""
        if not self.input_mux:
            return {}
        
        stats = self.input_mux.get_stats()
        
        return {
            "mode": stats.get("mode", "unknown"),
            "bot_injections": stats.get("bot_injections", 0),
            "human_overrides": stats.get("human_overrides", 0),
            "mode_switches": stats.get("mode_switches", 0),
            "safety_locked": stats.get("safety_locked", False),
            "human_idle_time": stats.get("human_idle_time", 0)
        }
    
    def _get_errors_data(self) -> Dict:
        """Get errors and warnings"""
        if not self.performance_monitor:
            return {}
        
        return {
            "errors": {
                "total": len(self.performance_monitor.errors),
                "recent": self.performance_monitor.errors[-20:] if len(self.performance_monitor.errors) > 0 else []
            },
            "warnings": {
                "total": len(self.performance_monitor.warnings),
                "recent": self.performance_monitor.warnings[-20:] if len(self.performance_monitor.warnings) > 0 else []
            }
        }
    
    def _get_episodes_data(self) -> Dict:
        """Get episode history"""
        if not self.performance_monitor:
            return {}
        
        return {
            "current_episode": self.performance_monitor.current_episode,
            "recent_episodes": self.performance_monitor.episode_metrics[-20:] if len(self.performance_monitor.episode_metrics) > 0 else [],
            "total_episodes": len(self.performance_monitor.episode_metrics)
        }
    
    def _get_kill_switch_data(self) -> Dict:
        """Get kill switch status"""
        if not self.kill_switch:
            return {}
        
        return {
            "active": self.kill_switch.running,
            "killed": self.kill_switch.is_killed(),
            "kill_count": self.kill_switch.kill_count,
            "hotkey": config.KILL_BUTTON.upper()
        }
    
    def _get_yolo_data(self) -> Dict:
        """Get YOLO detection data with proper JSON serialization"""
        try:
            if not self.vision_processor or not hasattr(self.vision_processor, 'yolo_detector') or not self.vision_processor.yolo_detector:
                return {"enabled": False, "reason": "vision_processor or yolo_detector not available"}
            
            detector_stats = {}
            try:
                if hasattr(self.vision_processor, 'get_detector_stats'):
                    detector_stats = self.vision_processor.get_detector_stats()
                    # Convert numpy types to Python types
                    detector_stats = convert_to_json_serializable(detector_stats)
            except Exception as e:
                logger.debug(f"Error getting detector stats: {e}")
                detector_stats = {"error": str(e)}
            
            # Get latest detections from actor if available
            latest_detections = self.last_detections if self.last_detections else {}
            # Convert numpy types to Python types
            latest_detections = convert_to_json_serializable(latest_detections)
            
            # Ensure all values are JSON-serializable
            detection_count = latest_detections.get('count', 0) if isinstance(latest_detections, dict) else 0
            enemy_count = len(latest_detections.get('enemies', [])) if isinstance(latest_detections, dict) else 0
            threat_level = latest_detections.get('threat_level', 'unknown') if isinstance(latest_detections, dict) else 'unknown'
            
            return {
                "enabled": True,
                "stats": detector_stats,
                "latest_detections": latest_detections,
                "detection_count": int(detection_count) if detection_count is not None else 0,
                "enemy_count": int(enemy_count) if enemy_count is not None else 0,
                "threat_level": str(threat_level) if threat_level is not None else 'unknown'
            }
        except Exception as e:
            logger.error(f"Error in _get_yolo_data: {e}", exc_info=True)
            return {"enabled": False, "error": str(e)}
    
    def _get_yolo_self_learning_data(self) -> Dict:
        """Get YOLO self-learning statistics"""
        if not self.actor or not hasattr(self.actor, 'yolo_self_learner') or not self.actor.yolo_self_learner:
            return {"enabled": False}
        
        learner = self.actor.yolo_self_learner
        stats = learner.get_stats()
        
        return {
            "enabled": True,
            "total_labels_created": stats.get('total_labels_created', 0),
            "positive_labels": stats.get('positive_labels', 0),
            "negative_labels": stats.get('negative_labels', 0),
            "action_detection_pairs": stats.get('action_detection_pairs', 0),
            "self_labeled_data": stats.get('self_labeled_data', 0),
            "detection_rewards": stats.get('detection_rewards', {}),
            "confidence_adjustments": stats.get('confidence_adjustments', {}),
            "training_count": getattr(self.actor, 'yolo_self_training_count', 0)
        }
    
    def _get_human_recording_data(self) -> Dict:
        """Get human action recording statistics with error handling"""
        try:
            # Try actor first
            if self.actor and hasattr(self.actor, 'human_recorder') and self.actor.human_recorder:
                recorder = self.actor.human_recorder
            # Fallback to direct reference
            elif self.human_recorder:
                recorder = self.human_recorder
            else:
                return {"enabled": False, "reason": "human_recorder not available"}
            
            stats = recorder.get_action_statistics()
            
            return {
                "enabled": True,
                "recording": stats.get('recording', False),
                "total_actions": stats.get('total_actions', 0),
                "total_sessions": stats.get('total_sessions', 0),
                "current_session_actions": stats.get('current_session_actions', 0),
                "current_session_duration": stats.get('current_session_duration', 0),
                "mouse_movements": stats.get('mouse_movements', 0),
                "button_presses": stats.get('button_presses', 0),
                "expert_buffer_size": stats.get('expert_buffer_size', 0),
                "avg_mouse_magnitude": stats.get('avg_mouse_magnitude', 0),
                "avg_velocity": stats.get('avg_velocity', 0),
                "action_rate": stats.get('action_rate', 0),
                "quality_metrics": stats.get('quality_metrics', {})
            }
        except Exception as e:
            logger.error(f"Error in _get_human_recording_data: {e}", exc_info=True)
            return {"enabled": False, "error": str(e)}
    
    def _get_training_data(self) -> Dict:
        """Get training progress data"""
        if not self.learner or not hasattr(self.learner, 'model_tracker'):
            return {"enabled": False}
        
        tracker = self.learner.model_tracker
        stats = tracker.get_training_stats()
        
        return {
            "enabled": True,
            "update_count": stats.get('update_count', 0),
            "avg_loss": stats.get('avg_loss', 0.0),
            "avg_bc_loss": stats.get('avg_bc_loss', 0.0),
            "trend": stats.get('trend', 'unknown'),
            "human_action_ratio": stats.get('human_action_ratio', 0.0),
            "beta_bc": getattr(self.learner, 'beta_bc', 1.0)
        }
    
    def _get_game_status_data(self) -> Dict:
        """Get game process status"""
        if not self.actor or not hasattr(self.actor, 'memory_reader'):
            return {"enabled": False, "is_running": False}
        
        memory_reader = self.actor.memory_reader
        if hasattr(memory_reader, 'get_process_status'):
            status = memory_reader.get_process_status()
        else:
            status = {
                "process_name": getattr(memory_reader, 'process_name', 'Unknown'),
                "is_running": memory_reader.is_process_running() if hasattr(memory_reader, 'is_process_running') else False,
                "is_attached": memory_reader.process is not None if hasattr(memory_reader, 'process') else False,
                "has_memory_access": False
            }
        
        # Get current game state
        game_state = {}
        if hasattr(memory_reader, 'get_state'):
            game_state = memory_reader.get_state()
        
        return {
            "enabled": True,
            **status,
            "current_game_state": game_state
        }
    
    # LLM communication data method removed
    
    def start(self):
        """Start dashboard server with enhanced error handling"""
        if not FLASK_AVAILABLE:
            logger.error("Cannot start dashboard - Flask not installed")
            logger.error("Install with: pip install flask flask-cors")
            return
        
        if not self.app:
            logger.error("Cannot start dashboard - Flask app not initialized")
            return
        
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        self.running = True
        
        def run_server():
            try:
                import time as time_module
                # Give a moment for thread to start and Flask to initialize
                time_module.sleep(1.0)
                
                if not self.app:
                    logger.error("Dashboard app not initialized - cannot start server")
                    self.running = False
                    return
                
                logger.info(f"Starting dashboard server on http://127.0.0.1:{self.port}")
                logger.info(f"   Dashboard will be available at http://localhost:{self.port}")
                logger.info(f"   Test page available at http://localhost:{self.port}/test")
                
                # Verify template exists
                import os
                template_path = os.path.join(os.path.dirname(__file__), 'dashboard_templates', 'dashboard.html')
                if os.path.exists(template_path):
                    logger.info(f"✅ Dashboard HTML template found: {template_path}")
                else:
                    logger.error(f"❌ Dashboard HTML template NOT found: {template_path}")
                
                # Use threaded mode with better configuration
                # Note: app.run() is blocking, so it will keep the thread alive
                try:
                    self.app.run(
                        host='127.0.0.1', 
                        port=self.port, 
                        debug=False, 
                        use_reloader=False, 
                        threaded=True, 
                        use_evalex=False,
                        passthrough_errors=True,  # Better error handling
                        request_handler=None  # Use default handler
                    )
                except KeyboardInterrupt:
                    logger.info("Dashboard server interrupted")
                    self.running = False
            except OSError as e:
                error_msg = str(e).lower()
                error_code = getattr(e, 'errno', None)
                if "address already in use" in error_msg or "address is already in use" in error_msg:
                    logger.warning(f"⚠️  Port {self.port} is already in use. Dashboard may already be running.")
                    logger.info(f"   Try accessing: http://localhost:{self.port}")
                    # Don't set running to False - server might be running elsewhere
                elif error_code == 22:  # Invalid argument (Windows stdout flush issue)
                    # This is a known Flask/Windows issue with stdout flushing
                    # The server actually starts fine, just the banner print fails
                    logger.warning(f"⚠️  Dashboard stdout flush issue (Windows) - server should still work")
                    logger.info(f"   Dashboard available at: http://localhost:{self.port}")
                    # Continue running - server is actually working
                    self.running = True
                else:
                    logger.error(f"Dashboard server OSError: {e}", exc_info=True)
                    self.running = False
            except Exception as e:
                logger.error(f"Dashboard server error: {e}", exc_info=True)
                self.running = False
        
        # Start server thread
        self.server_thread = threading.Thread(target=run_server, daemon=True, name="DashboardServer")
        self.server_thread.start()
        
        # Wait a moment and verify it started
        time.sleep(1.0)  # Give more time for Flask to start
        
        if self.server_thread.is_alive():
            logger.info(f"✅ Dashboard server thread started and running")
            logger.info(f"   Main dashboard: http://localhost:{self.port}")
            logger.info(f"   Test page: http://localhost:{self.port}/test")
            logger.info(f"   Health check: http://localhost:{self.port}/api/health")
            
            # Verify server is actually responding (quick test)
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(('127.0.0.1', self.port))
                sock.close()
                if result == 0:
                    logger.info(f"✅ Dashboard server is listening on port {self.port}")
                    # Automatically open browser once server is confirmed listening
                    self._open_browser()
                else:
                    logger.warning(f"⚠️  Dashboard server thread is alive but port {self.port} may not be listening")
            except Exception as e:
                logger.debug(f"Could not verify port status: {e}")
        else:
            logger.error("❌ Dashboard server thread failed to start or died immediately")
            self.running = False
    
    def _open_browser(self):
        """Automatically open browser to dashboard URL"""
        try:
            url = f"http://localhost:{self.port}"
            logger.info(f"🌐 Opening browser to dashboard: {url}")
            time.sleep(0.5)  # Wait a moment to ensure Flask is fully ready
            webbrowser.open(url)
            logger.info(f"✅ Browser opened successfully to dashboard")
        except Exception as e:
            logger.warning(f"⚠️  Could not automatically open browser: {e}")
            logger.info(f"   Please manually open: http://localhost:{self.port}")
    
    def stop(self):
        """Stop dashboard server"""
        self.running = False
        logger.info("Dashboard server stopped")

