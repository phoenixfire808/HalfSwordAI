"""
Process Management Utilities
Handles process detection, monitoring, and management
"""
import psutil
import subprocess
import logging
from typing import Optional, List, Dict
from half_sword_ai.config import config

logger = logging.getLogger(__name__)

def is_process_running(process_name: str) -> bool:
    """
    Check if a process is running by name
    
    Args:
        process_name: Name of the process (e.g., "HalfSword-Win64-Shipping.exe")
    
    Returns:
        True if process is running, False otherwise
    """
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception as e:
        logger.warning(f"Error checking process {process_name}: {e}")
        return False

def find_process_by_name(process_name: str) -> Optional[psutil.Process]:
    """
    Find process by name and return Process object
    
    Args:
        process_name: Name of the process
    
    Returns:
        Process object if found, None otherwise
    """
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                    return psutil.Process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    except Exception as e:
        logger.warning(f"Error finding process {process_name}: {e}")
        return None

def get_process_info(process: psutil.Process) -> Dict:
    """
    Get comprehensive information about a process
    
    Args:
        process: psutil.Process object
    
    Returns:
        Dictionary with process information
    """
    try:
        return {
            'pid': process.pid,
            'name': process.name(),
            'status': process.status(),
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'create_time': process.create_time(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Error getting process info: {e}")
        return {}

def safe_kill_process(process: psutil.Process, timeout: float = 5.0) -> bool:
    """
    Safely terminate a process with timeout
    
    Args:
        process: psutil.Process object
        timeout: Timeout in seconds before force kill
    
    Returns:
        True if process was terminated, False otherwise
    """
    try:
        # Try graceful termination first
        process.terminate()
        
        # Wait for process to terminate
        try:
            process.wait(timeout=timeout)
            return True
        except psutil.TimeoutExpired:
            # Force kill if timeout
            logger.warning(f"Process {process.pid} did not terminate, force killing...")
            process.kill()
            process.wait(timeout=1.0)
            return True
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Error killing process: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error killing process: {e}")
        return False

def launch_process(
    executable_path: str,
    args: List[str] = None,
    cwd: Optional[str] = None,
    wait: bool = False
) -> Optional[subprocess.Popen]:
    """
    Launch a process
    
    Args:
        executable_path: Path to executable
        args: Optional command-line arguments
        cwd: Optional working directory
        wait: Whether to wait for process to start
    
    Returns:
        Popen object if successful, None otherwise
    """
    try:
        cmd = [executable_path]
        if args:
            cmd.extend(args)
        
        process = subprocess.Popen(
            cmd,
            cwd=cwd or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        if wait:
            import time
            time.sleep(1.0)  # Give process time to start
        
        return process
    except Exception as e:
        logger.error(f"Failed to launch process {executable_path}: {e}")
        return None

def get_all_processes_by_name(process_name: str) -> List[psutil.Process]:
    """
    Get all processes matching a name
    
    Args:
        process_name: Name of the process
    
    Returns:
        List of Process objects
    """
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                    processes.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.warning(f"Error finding processes {process_name}: {e}")
    
    return processes

def get_process_memory_usage(process: psutil.Process) -> Dict[str, float]:
    """
    Get detailed memory usage for a process
    
    Args:
        process: psutil.Process object
    
    Returns:
        Dictionary with memory statistics in MB
    """
    try:
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Error getting memory usage: {e}")
        return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}

def check_process_health(process: psutil.Process) -> Dict[str, any]:
    """
    Check health status of a process
    
    Args:
        process: psutil.Process object
    
    Returns:
        Dictionary with health information
    """
    try:
        status = process.status()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Determine health status
        is_healthy = True
        issues = []
        
        if status != psutil.STATUS_RUNNING:
            is_healthy = False
            issues.append(f"Status: {status}")
        
        if memory_mb > config.MEMORY_LEAK_THRESHOLD / (1024 * 1024):
            is_healthy = False
            issues.append(f"High memory usage: {memory_mb:.2f} MB")
        
        return {
            'healthy': is_healthy,
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'issues': issues
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {
            'healthy': False,
            'status': 'not_found',
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'issues': [str(e)]
        }




