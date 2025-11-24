"""
Performance Monitor: Advanced performance tracking with alerting and profiling
Tracks metrics, detects anomalies, and provides predictive analytics
"""
import time
import psutil
import torch
import numpy as np
import json
import logging
import threading
from typing import Dict, List, Optional, Callable
from collections import deque, defaultdict
from datetime import datetime
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from half_sword_ai.config import config

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    Tracks FPS, latency, memory, CPU, GPU, and learning metrics
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.episode_metrics = []
        self.current_episode = {}
        
        # Frame timing
        self.frame_times = deque(maxlen=1000)
        self.last_frame_time = time.time()
        
        # Component latencies
        self.capture_latencies = deque(maxlen=1000)
        self.inference_latencies = deque(maxlen=1000)
        self.injection_latencies = deque(maxlen=1000)
        # LLM latencies removed
        
        # System resources
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_usage = deque(maxlen=1000)
        self.gpu_temperature = deque(maxlen=100)
        self.network_io = deque(maxlen=100)
        self.disk_io = deque(maxlen=100)
        
        # Learning metrics
        self.training_losses = deque(maxlen=10000)
        self.rewards = deque(maxlen=10000)
        self.episode_rewards = []
        
        # Error tracking with categorization
        self.errors = []
        self.warnings = []
        self.error_categories = defaultdict(int)
        
        # Performance counters
        self.frame_count = 0
        self.episode_count = 0
        self.total_training_steps = 0
        self.last_training_step_time = time.time()
        self.training_step_times = deque(maxlen=1000)  # Track training step intervals
        
        # Initialize YOLO latency tracking if needed
        self.yolo_latencies = deque(maxlen=1000)
        
        # Alerting system
        self.alert_thresholds = {
            'fps_min': 10.0,
            'cpu_max': 95.0,
            'memory_max_mb': 16384,
            'gpu_memory_max_mb': 8192,
            'latency_max_ms': 1000.0,
            'error_rate_max': 10.0  # errors per minute
        }
        self.active_alerts = {}
        self.alert_callbacks = []
        
        # Performance profiling
        self.profiling_data = {}
        self.bottleneck_history = deque(maxlen=100)
        
        # Statistical analysis
        self.statistical_cache = {}
        self.cache_ttl = 5.0  # seconds
        self.last_stat_update = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
    def record_frame(self):
        """
        Record frame timing with FPS calculation and alerting
        NOTE: This is now called less frequently - frame timing is done directly in actor loop
        This method is kept for backward compatibility but frame times should be recorded directly
        """
        # Frame timing is now handled directly in the actor loop for accuracy
        # This method can still be used for explicit frame recording if needed
        with self.lock:
            # Check FPS alert if we have frame times
            if len(self.frame_times) > 1:
                avg_frame_time = np.mean(self.frame_times)
                fps = 1.0 / max(0.001, avg_frame_time)
                if fps < self.alert_thresholds['fps_min']:
                    self._trigger_alert('low_fps', {'fps': fps})
    
    def record_capture_latency(self, latency: float):
        """Record screen capture latency"""
        self.capture_latencies.append(latency)
        self.metrics['capture_latency_ms'].append(latency * 1000)
    
    def record_inference_latency(self, latency: float):
        """Record model inference latency"""
        self.inference_latencies.append(latency)
        self.metrics['inference_latency_ms'].append(latency * 1000)
    
    def record_injection_latency(self, latency: float):
        """Record input injection latency"""
        self.injection_latencies.append(latency)
        self.metrics['injection_latency_ms'].append(latency * 1000)
    
    # LLM latency recording removed
    
    def record_training_loss(self, loss: float, ppo_loss: float, bc_loss: float,
                            value_loss: float = None, policy_loss: float = None,
                            entropy: float = None, gradient_norm: float = None,
                            learning_rate: float = None):
        """Record training metrics - REAL-TIME tracking with detailed logging"""
        with self.lock:
            current_time = time.time()
            
            # Track training step timing
            if self.last_training_step_time > 0:
                step_interval = current_time - self.last_training_step_time
                self.training_step_times.append(step_interval)
            
            self.last_training_step_time = current_time
            
            # Store training losses
            self.training_losses.append(loss)
            self.total_training_steps += 1
            
            # Ensure metrics dicts exist
            if 'total_loss' not in self.metrics:
                self.metrics['total_loss'] = deque(maxlen=10000)
            if 'ppo_loss' not in self.metrics:
                self.metrics['ppo_loss'] = deque(maxlen=10000)
            if 'bc_loss' not in self.metrics:
                self.metrics['bc_loss'] = deque(maxlen=10000)
            
            self.metrics['total_loss'].append(loss)
            self.metrics['ppo_loss'].append(ppo_loss)
            self.metrics['bc_loss'].append(bc_loss)
            
            # Record additional metrics if provided
            if value_loss is not None:
                if 'value_loss' not in self.metrics:
                    self.metrics['value_loss'] = deque(maxlen=10000)
                self.metrics['value_loss'].append(value_loss)
            
            if policy_loss is not None:
                if 'policy_loss' not in self.metrics:
                    self.metrics['policy_loss'] = deque(maxlen=10000)
                self.metrics['policy_loss'].append(policy_loss)
            
            if entropy is not None:
                if 'entropy' not in self.metrics:
                    self.metrics['entropy'] = deque(maxlen=10000)
                self.metrics['entropy'].append(entropy)
            
            if gradient_norm is not None:
                if 'gradient_norm' not in self.metrics:
                    self.metrics['gradient_norm'] = deque(maxlen=10000)
                self.metrics['gradient_norm'].append(gradient_norm)
            
            if learning_rate is not None:
                if 'learning_rate' not in self.metrics:
                    self.metrics['learning_rate'] = deque(maxlen=10000)
                self.metrics['learning_rate'].append(learning_rate)
            
            # REAL-TIME logging - every training step!
            # Format values properly (can't use ternary in format specifier)
            value_str = f"{value_loss:.4f}" if value_loss is not None else "N/A"
            policy_str = f"{policy_loss:.4f}" if policy_loss is not None else "N/A"
            grad_str = f"{gradient_norm:.4f}" if gradient_norm is not None else "N/A"
            lr_str = f"{learning_rate:.6f}" if learning_rate is not None else "N/A"
            
            logger.info(f"[TRAINING] Step: {self.total_training_steps} | "
                       f"Loss: {loss:.4f} | PPO: {ppo_loss:.4f} | BC: {bc_loss:.4f} | "
                       f"Value: {value_str} | "
                       f"Policy: {policy_str} | "
                       f"Grad: {grad_str} | "
                       f"LR: {lr_str}")
    
    def record_reward(self, reward: float):
        """Record reward"""
        self.rewards.append(reward)
        self.metrics['reward'].append(reward)
        if 'episode_reward' not in self.current_episode:
            self.current_episode['episode_reward'] = 0.0
        self.current_episode['episode_reward'] += reward
    
    def record_error(self, error: Exception, context: str = ""):
        """Record error with categorization"""
        with self.lock:
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            }
            self.errors.append(error_info)
            self.error_categories[type(error).__name__] += 1
            
            # Check error rate alert
            recent_errors = [e for e in self.errors 
                           if time.time() - datetime.fromisoformat(e["timestamp"]).timestamp() < 60]
            error_rate = len(recent_errors)
            if error_rate > self.alert_thresholds['error_rate_max']:
                self._trigger_alert('high_error_rate', {
                    'rate': error_rate,
                    'threshold': self.alert_thresholds['error_rate_max']
                })
            
            logger.error(f"Error recorded: {error_info}")
    
    def record_warning(self, warning: str, context: str = ""):
        """Record warning"""
        warning_info = {
            "timestamp": datetime.now().isoformat(),
            "warning": warning,
            "context": context
        }
        self.warnings.append(warning_info)
        logger.warning(f"Warning recorded: {warning_info}")
    
    def start_episode(self):
        """Start new episode"""
        self.current_episode = {
            "start_time": time.time(),
            "episode_reward": 0.0,
            "frame_count": 0,
            "human_interventions": 0,
            "deaths": 0,
            "victories": 0
        }
        self.episode_count += 1
    
    def end_episode(self, outcome: str = "unknown"):
        """End current episode"""
        if self.current_episode:
            self.current_episode["end_time"] = time.time()
            self.current_episode["duration"] = self.current_episode["end_time"] - self.current_episode["start_time"]
            self.current_episode["outcome"] = outcome
            self.episode_rewards.append(self.current_episode["episode_reward"])
            self.episode_metrics.append(self.current_episode.copy())
            logger.info(f"Episode {self.episode_count} ended: {outcome}, Reward: {self.current_episode['episode_reward']:.2f}")
    
    def update_system_metrics(self):
        """Update system resource metrics with enhanced tracking"""
        try:
            with self.lock:
                # CPU usage (non-blocking)
                cpu_percent = psutil.cpu_percent(interval=None)
                if cpu_percent is not None:
                    self.cpu_usage.append(cpu_percent)
                    self.metrics['cpu_percent'].append(cpu_percent)
                    
                    # Check CPU alert
                    if cpu_percent > self.alert_thresholds['cpu_max']:
                        self._trigger_alert('high_cpu', {'cpu': cpu_percent})
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.memory_usage.append(memory_mb)
                self.metrics['memory_mb'].append(memory_mb)
                
                # Check memory alert
                if memory_mb > self.alert_thresholds['memory_max_mb']:
                    self._trigger_alert('high_memory', {'memory_mb': memory_mb})
                
                # Network I/O
                try:
                    net_io = psutil.net_io_counters()
                    self.network_io.append({
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    })
                except:
                    pass
                
                # Disk I/O
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.disk_io.append({
                            'read_bytes': disk_io.read_bytes,
                            'write_bytes': disk_io.write_bytes,
                            'read_count': disk_io.read_count,
                            'write_count': disk_io.write_count
                        })
                except:
                    pass
                
                # GPU usage (if available)
                if torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        self.gpu_usage.append(gpu_memory)
                        self.metrics['gpu_memory_mb'].append(gpu_memory)
                        
                        # Check GPU memory alert
                        if gpu_memory > self.alert_thresholds['gpu_memory_max_mb']:
                            self._trigger_alert('high_gpu_memory', {'gpu_mb': gpu_memory})
                        
                        # Try to get GPU utilization if available
                        try:
                            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                        except:
                            gpu_util = 0
                        self.metrics['gpu_utilization'].append(gpu_util)
                        
                        # Try to get GPU temperature (requires nvidia-ml-py)
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            self.gpu_temperature.append(temp)
                        except:
                            pass
                    except Exception as gpu_e:
                        logger.debug(f"GPU metrics error: {gpu_e}")
        except Exception as e:
            logger.debug(f"System metrics update error: {e}")
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = {
            "runtime_seconds": time.time() - self.start_time,
            "frame_count": self.frame_count,
            "episode_count": self.episode_count,
            "total_training_steps": self.total_training_steps,
        }
        
        # FPS - Calculate from actual frame times (accurate)
        if len(self.frame_times) > 0:
            # Current FPS from most recent frame time
            current_frame_time = self.frame_times[-1]
            current_fps = 1.0 / max(0.001, current_frame_time)
            
            # Average FPS from mean frame time
            avg_frame_time = np.mean(self.frame_times)
            avg_fps = 1.0 / max(0.001, avg_frame_time)
            
            # Min/Max FPS (inverse of max/min frame times)
            max_frame_time = np.max(self.frame_times)  # Slowest frame
            min_frame_time = np.min(self.frame_times)  # Fastest frame
            min_fps = 1.0 / max(0.001, max_frame_time)  # Min FPS = slowest frame
            max_fps = 1.0 / max(0.001, min_frame_time)  # Max FPS = fastest frame
            
            # Alternative calculation: frames / time (more accurate for overall rate)
            runtime = time.time() - self.start_time
            calculated_fps = self.frame_count / max(0.001, runtime) if runtime > 0 else 0
            
            stats["fps"] = {
                "current": current_fps,
                "average": avg_fps,
                "min": min_fps,
                "max": max_fps,
                "frame_count": self.frame_count,
                "total_runtime": runtime,
                "calculated_fps": calculated_fps  # Alternative calculation based on frame count
            }
        
        # Latencies
        stats["latencies_ms"] = {
            "capture": {
                "avg": np.mean(self.capture_latencies) * 1000 if len(self.capture_latencies) > 0 else 0,
                "max": np.max(self.capture_latencies) * 1000 if len(self.capture_latencies) > 0 else 0,
                "min": np.min(self.capture_latencies) * 1000 if len(self.capture_latencies) > 0 else 0,
                "count": len(self.capture_latencies)
            },
            "inference": {
                "avg": np.mean(self.inference_latencies) * 1000 if len(self.inference_latencies) > 0 else 0,
                "max": np.max(self.inference_latencies) * 1000 if len(self.inference_latencies) > 0 else 0,
                "min": np.min(self.inference_latencies) * 1000 if len(self.inference_latencies) > 0 else 0,
                "count": len(self.inference_latencies)
            },
            "injection": {
                "avg": np.mean(self.injection_latencies) * 1000 if len(self.injection_latencies) > 0 else 0,
                "max": np.max(self.injection_latencies) * 1000 if len(self.injection_latencies) > 0 else 0,
                "min": np.min(self.injection_latencies) * 1000 if len(self.injection_latencies) > 0 else 0,
                "count": len(self.injection_latencies)
            },
            # LLM metrics removed
        }
        
        # Add YOLO latency if tracked
        if hasattr(self, 'yolo_latencies') and len(self.yolo_latencies) > 0:
            stats["latencies_ms"]["yolo"] = {
                "avg": np.mean(self.yolo_latencies) * 1000,
                "max": np.max(self.yolo_latencies) * 1000,
                "min": np.min(self.yolo_latencies) * 1000,
                "count": len(self.yolo_latencies)
            }
        
        # System resources
        stats["system"] = {
            "cpu_percent": {
                "current": self.cpu_usage[-1] if len(self.cpu_usage) > 0 else 0,
                "average": np.mean(self.cpu_usage) if len(self.cpu_usage) > 0 else 0,
                "max": np.max(self.cpu_usage) if len(self.cpu_usage) > 0 else 0,
                "min": np.min(self.cpu_usage) if len(self.cpu_usage) > 0 else 0,
                "count": len(self.cpu_usage)
            },
            "memory_mb": {
                "current": self.memory_usage[-1] if len(self.memory_usage) > 0 else 0,
                "average": np.mean(self.memory_usage) if len(self.memory_usage) > 0 else 0,
                "max": np.max(self.memory_usage) if len(self.memory_usage) > 0 else 0,
                "min": np.min(self.memory_usage) if len(self.memory_usage) > 0 else 0,
                "count": len(self.memory_usage)
            }
        }
        
        if torch.cuda.is_available() and len(self.gpu_usage) > 0:
            stats["system"]["gpu_memory_mb"] = {
                "current": self.gpu_usage[-1],
                "average": np.mean(self.gpu_usage),
                "max": np.max(self.gpu_usage)
            }
            # Add GPU utilization if available
            if 'gpu_utilization' in self.metrics and len(self.metrics['gpu_utilization']) > 0:
                stats["system"]["gpu_utilization"] = {
                    "current": self.metrics['gpu_utilization'][-1] if len(self.metrics['gpu_utilization']) > 0 else 0,
                    "average": np.mean(self.metrics['gpu_utilization']) if len(self.metrics['gpu_utilization']) > 0 else 0,
                    "max": np.max(self.metrics['gpu_utilization']) if len(self.metrics['gpu_utilization']) > 0 else 0
                }
            # Add GPU temperature if available
            if len(self.gpu_temperature) > 0:
                stats["system"]["gpu_temperature"] = {
                    "current": self.gpu_temperature[-1],
                    "average": np.mean(self.gpu_temperature),
                    "max": np.max(self.gpu_temperature)
                }
        
        # Network I/O
        if len(self.network_io) > 0:
            latest_net = self.network_io[-1]
            stats["system"]["network_io"] = latest_net
        
        # Disk I/O
        if len(self.disk_io) > 0:
            latest_disk = self.disk_io[-1]
            stats["system"]["disk_io"] = latest_disk
        
        # System health
        stats["system_health"] = self._calculate_system_health()
        
        # Active alerts
        stats["active_alerts"] = list(self.active_alerts.keys())
        
        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            stats["bottlenecks"] = bottlenecks
        
        # Learning metrics
        if len(self.training_losses) > 0:
            stats["training"] = {
                "loss": {
                    "current": self.training_losses[-1],
                    "average": np.mean(self.training_losses),
                    "min": np.min(self.training_losses),
                    "max": np.max(self.training_losses)
                }
            }
        
        if len(self.rewards) > 0:
            stats["rewards"] = {
                "current": self.rewards[-1],
                "average": np.mean(self.rewards),
                "total": np.sum(self.rewards)
            }
        
        if len(self.episode_rewards) > 0:
            stats["episodes"] = {
                "average_reward": np.mean(self.episode_rewards),
                "best_reward": np.max(self.episode_rewards),
                "worst_reward": np.min(self.episode_rewards),
                "total_episodes": len(self.episode_rewards)
            }
        
        # Error counts
        stats["errors"] = {
            "total": len(self.errors),
            "recent": len([e for e in self.errors if time.time() - datetime.fromisoformat(e["timestamp"]).timestamp() < 60])
        }
        
        stats["warnings"] = {
            "total": len(self.warnings),
            "recent": len([w for w in self.warnings if time.time() - datetime.fromisoformat(w["timestamp"]).timestamp() < 60])
        }
        
        return stats
    
    def generate_report(self, filepath: Optional[str] = None) -> str:
        """Generate comprehensive performance report"""
        stats = self.get_current_stats()
        
        report = f"""
{'='*80}
HALF SWORD AI AGENT - PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

RUNTIME STATISTICS
------------------
Total Runtime: {stats['runtime_seconds']:.2f} seconds ({stats['runtime_seconds']/60:.2f} minutes)
Frames Processed: {stats['frame_count']:,}
Episodes Completed: {stats['episode_count']:,}
Training Steps: {stats['total_training_steps']:,}

PERFORMANCE METRICS
-------------------
FPS:
  Current: {stats.get('fps', {}).get('current', 0):.2f}
  Average: {stats.get('fps', {}).get('average', 0):.2f}
  Min: {stats.get('fps', {}).get('min', 0):.2f}
  Max: {stats.get('fps', {}).get('max', 0):.2f}

LATENCIES (milliseconds):
  Capture:
    Average: {stats.get('latencies_ms', {}).get('capture', {}).get('avg', 0):.2f}
    Max: {stats.get('latencies_ms', {}).get('capture', {}).get('max', 0):.2f}
    Samples: {stats.get('latencies_ms', {}).get('capture', {}).get('count', 0)}
  
  Inference:
    Average: {stats.get('latencies_ms', {}).get('inference', {}).get('avg', 0):.2f}
    Max: {stats.get('latencies_ms', {}).get('inference', {}).get('max', 0):.2f}
    Samples: {stats.get('latencies_ms', {}).get('inference', {}).get('count', 0)}
  
  Injection:
    Average: {stats.get('latencies_ms', {}).get('injection', {}).get('avg', 0):.2f}
    Max: {stats.get('latencies_ms', {}).get('injection', {}).get('max', 0):.2f}
    Samples: {stats.get('latencies_ms', {}).get('injection', {}).get('count', 0)}
  
  # LLM API metrics removed
"""
        
        # Add YOLO latency if available
        if 'yolo' in stats.get('latencies_ms', {}):
            yolo_lat = stats['latencies_ms']['yolo']
            report += f"""
  YOLO Detection:
    Average: {yolo_lat.get('avg', 0):.2f}
    Max: {yolo_lat.get('max', 0):.2f}
    Samples: {yolo_lat.get('count', 0)}
"""
        
        report += f"""
SYSTEM RESOURCES
----------------
CPU Usage:
  Current: {stats.get('system', {}).get('cpu_percent', {}).get('current', 0):.1f}%
  Average: {stats.get('system', {}).get('cpu_percent', {}).get('average', 0):.1f}%
  Peak: {stats.get('system', {}).get('cpu_percent', {}).get('max', 0):.1f}%

Memory Usage:
  Current: {stats.get('system', {}).get('memory_mb', {}).get('current', 0):.0f} MB
  Average: {stats.get('system', {}).get('memory_mb', {}).get('average', 0):.0f} MB
  Peak: {stats.get('system', {}).get('memory_mb', {}).get('max', 0):.0f} MB
"""
        
        if torch.cuda.is_available() and 'gpu_memory_mb' in stats.get('system', {}):
            report += f"""
GPU Memory:
  Current: {stats['system']['gpu_memory_mb']['current']:.0f} MB
  Average: {stats['system']['gpu_memory_mb']['average']:.0f} MB
  Peak: {stats['system']['gpu_memory_mb']['max']:.0f} MB
"""
        
        if 'training' in stats and stats['training']:
            report += f"""
LEARNING METRICS
----------------
Training Loss:
  Current: {stats['training']['loss']['current']:.4f}
  Average: {stats['training']['loss']['average']:.4f}
  Min: {stats['training']['loss']['min']:.4f}
  Max: {stats['training']['loss']['max']:.4f}
  Training Steps: {stats['total_training_steps']:,}
"""
        else:
            report += f"""
LEARNING METRICS
----------------
Training Loss: No training data yet
Training Steps: {stats['total_training_steps']:,}
"""
        
        if 'rewards' in stats and stats['rewards']:
            report += f"""
Rewards:
  Current: {stats['rewards']['current']:.2f}
  Average: {stats['rewards']['average']:.2f}
  Total: {stats['rewards']['total']:.2f}
  Samples: {len(self.rewards):,}
"""
        else:
            report += f"""
Rewards: No reward data yet
"""
        
        if 'episodes' in stats and stats['episodes'] and stats['episodes'].get('total_episodes', 0) > 0:
            report += f"""
Episode Performance:
  Average Reward: {stats['episodes']['average_reward']:.2f}
  Best Reward: {stats['episodes']['best_reward']:.2f}
  Worst Reward: {stats['episodes']['worst_reward']:.2f}
  Total Episodes: {stats['episodes']['total_episodes']}
"""
        else:
            report += f"""
Episode Performance: No completed episodes yet
  Current Episode: {self.episode_count}
"""
        
        report += f"""
ERRORS & WARNINGS
-----------------
Total Errors: {stats['errors']['total']}
Recent Errors (last minute): {stats['errors']['recent']}
Total Warnings: {stats['warnings']['total']}
Recent Warnings (last minute): {stats['warnings']['recent']}
"""
        
        # Add recent errors
        if len(self.errors) > 0:
            report += "\nRecent Errors:\n"
            for error in self.errors[-10:]:
                report += f"  [{error['timestamp']}] {error['error_type']}: {error['error_message']}\n"
                if error['context']:
                    report += f"    Context: {error['context']}\n"
        
        report += f"\n{'='*80}\n"
        
        # Save to file if requested
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(report)
                logger.info(f"Performance report saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def _trigger_alert(self, alert_type: str, data: Dict):
        """Trigger an alert if not already active"""
        if alert_type not in self.active_alerts:
            self.active_alerts[alert_type] = {
                'type': alert_type,
                'data': data,
                'timestamp': time.time(),
                'first_triggered': time.time()
            }
            logger.warning(f"ALERT: {alert_type} - {data}")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, data)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def clear_alert(self, alert_type: str):
        """Clear an active alert"""
        if alert_type in self.active_alerts:
            del self.active_alerts[alert_type]
    
    def register_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_statistical_analysis(self, metric_name: str) -> Dict:
        """Get advanced statistical analysis for a metric"""
        current_time = time.time()
        
        # Check cache
        if (metric_name in self.statistical_cache and 
            current_time - self.statistical_cache[metric_name]['timestamp'] < self.cache_ttl):
            return self.statistical_cache[metric_name]['data']
        
        with self.lock:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                return {}
            
            values = np.array(list(self.metrics[metric_name]))
            
            if len(values) < 2:
                return {}
            
            analysis = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentiles': {
                    'p25': float(np.percentile(values, 25)),
                    'p50': float(np.percentile(values, 50)),
                    'p75': float(np.percentile(values, 75)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99))
                },
                'count': len(values),
                'trend': self._calculate_trend(values),
                'anomalies': self._detect_anomalies(values)
            }
            
            # Cache result
            self.statistical_cache[metric_name] = {
                'data': analysis,
                'timestamp': current_time
            }
            
            return analysis
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Use linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < np.std(values) * 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _detect_anomalies(self, values: np.ndarray) -> List[int]:
        """Detect anomalies using z-score"""
        if len(values) < 10:
            return []
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values - mean) / std)
        threshold = 3.0  # 3 standard deviations
        anomalies = np.where(z_scores > threshold)[0].tolist()
        
        return anomalies
    
    def identify_bottlenecks(self) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        with self.lock:
            # Check latencies
            latencies = {
                'capture': np.mean(self.capture_latencies) * 1000 if len(self.capture_latencies) > 0 else 0,
                'inference': np.mean(self.inference_latencies) * 1000 if len(self.inference_latencies) > 0 else 0,
                'injection': np.mean(self.injection_latencies) * 1000 if len(self.injection_latencies) > 0 else 0,
                'yolo': np.mean(self.yolo_latencies) * 1000 if len(self.yolo_latencies) > 0 else 0
            }
            
            max_latency = max(latencies.values())
            for name, latency in latencies.items():
                if latency > 50:  # 50ms threshold
                    bottlenecks.append({
                        'type': 'latency',
                        'component': name,
                        'value': latency,
                        'severity': 'high' if latency > 100 else 'medium',
                        'impact': (latency / max_latency) * 100 if max_latency > 0 else 0
                    })
            
            # Check FPS
            if len(self.frame_times) > 0:
                avg_fps = 1.0 / max(0.001, np.mean(self.frame_times))
                if avg_fps < 20:
                    bottlenecks.append({
                        'type': 'fps',
                        'component': 'overall',
                        'value': avg_fps,
                        'severity': 'high' if avg_fps < 15 else 'medium',
                        'impact': 100.0
                    })
            
            # Check system resources
            if len(self.cpu_usage) > 0:
                avg_cpu = np.mean(self.cpu_usage)
                if avg_cpu > 90:
                    bottlenecks.append({
                        'type': 'resource',
                        'component': 'cpu',
                        'value': avg_cpu,
                        'severity': 'high',
                        'impact': (avg_cpu / 100) * 50
                    })
        
        # Store in history
        if bottlenecks:
            self.bottleneck_history.append({
                'timestamp': time.time(),
                'bottlenecks': bottlenecks
            })
        
        return bottlenecks
    
    def get_performance_profile(self) -> Dict:
        """Get comprehensive performance profile"""
        profile = {
            'timestamp': time.time(),
            'runtime_seconds': time.time() - self.start_time,
            'bottlenecks': self.identify_bottlenecks(),
            'active_alerts': list(self.active_alerts.keys()),
            'statistics': {},
            'system_health': self._calculate_system_health()
        }
        
        # Add statistical analysis for key metrics
        key_metrics = ['fps', 'cpu_percent', 'memory_mb', 'gpu_memory_mb']
        for metric in key_metrics:
            if metric in self.metrics:
                profile['statistics'][metric] = self.get_statistical_analysis(metric)
        
        return profile
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0-100)"""
        health = 100.0
        
        # Deduct points for issues
        if len(self.active_alerts) > 0:
            health -= len(self.active_alerts) * 10
        
        # Check FPS
        if len(self.frame_times) > 0:
            avg_fps = 1.0 / max(0.001, np.mean(self.frame_times))
            if avg_fps < 20:
                health -= (20 - avg_fps) * 2
        
        # Check error rate
        recent_errors = [e for e in self.errors 
                        if time.time() - datetime.fromisoformat(e["timestamp"]).timestamp() < 60]
        if len(recent_errors) > 5:
            health -= min(30, len(recent_errors) * 2)
        
        return max(0.0, min(100.0, health))
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-JSON types to JSON-serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Try to convert to string as last resort
            return str(obj)
    
    def save_metrics_json(self, filepath: str):
        """Save metrics to JSON file with enhanced data"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": self._make_json_serializable(self.get_current_stats()),
            "performance_profile": self._make_json_serializable(self.get_performance_profile()),
            "episode_metrics": self._make_json_serializable(self.episode_metrics[-100:]),  # Last 100 episodes
            "errors": self._make_json_serializable(self.errors[-50:]),  # Last 50 errors
            "warnings": self._make_json_serializable(self.warnings[-50:]),  # Last 50 warnings
            "error_categories": self._make_json_serializable(dict(self.error_categories)),
            "bottlenecks": self._make_json_serializable(list(self.bottleneck_history)[-20:])  # Last 20 bottleneck reports
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics JSON saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics JSON: {e}", exc_info=True)

