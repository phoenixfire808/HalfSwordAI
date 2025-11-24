"""
File Utilities
Handles file operations, path management, and data serialization
"""
import os
import json
import time
import zlib
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def join_paths(*paths: str) -> str:
    """
    Join path components (cross-platform)
    
    Args:
        *paths: Path components to join
    
    Returns:
        Joined path string
    """
    return os.path.join(*paths)

def exists(path: str) -> bool:
    """
    Check if file or directory exists
    
    Args:
        path: Path to check
    
    Returns:
        True if exists, False otherwise
    """
    return os.path.exists(path)

def get_file_size(path: str) -> int:
    """
    Get file size in bytes
    
    Args:
        path: File path
    
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(path) if exists(path) else 0
    except Exception as e:
        logger.warning(f"Failed to get file size for {path}: {e}")
        return 0

def get_timestamped_filename(prefix: str, extension: str = "", directory: str = "") -> str:
    """
    Generate timestamped filename
    
    Args:
        prefix: Filename prefix
        extension: File extension (with or without dot)
        directory: Optional directory path
    
    Returns:
        Full path to timestamped file
    """
    timestamp = int(time.time())
    if extension and not extension.startswith('.'):
        extension = f'.{extension}'
    
    filename = f"{prefix}_{timestamp}{extension}"
    
    if directory:
        ensure_dir(directory)
        return join_paths(directory, filename)
    
    return filename

def save_json(
    data: Any,
    filepath: str,
    indent: int = 2,
    compress: bool = False,
    ensure_directory: bool = True
) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to serialize (must be JSON serializable)
        filepath: Path to save file
        indent: JSON indentation (0 for compact)
        compress: Whether to compress with zlib
        ensure_directory: Whether to create parent directory if needed
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if ensure_directory:
            directory = os.path.dirname(filepath)
            if directory:
                ensure_dir(directory)
        
        json_str = json.dumps(data, indent=indent, default=str)
        
        if compress:
            # Compress JSON string
            compressed = zlib.compress(json_str.encode('utf-8'))
            with open(filepath, 'wb') as f:
                f.write(compressed)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def load_json(
    filepath: str,
    compressed: bool = False,
    default: Any = None
) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        compressed: Whether file is compressed with zlib
        default: Default value to return if file doesn't exist or load fails
    
    Returns:
        Loaded data, or default value if failed
    """
    try:
        if not exists(filepath):
            logger.warning(f"JSON file not found: {filepath}")
            return default
        
        if compressed:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
                json_str = zlib.decompress(compressed_data).decode('utf-8')
                return json.loads(json_str)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return default

def save_pickle(data: Any, filepath: str, ensure_directory: bool = True) -> bool:
    """
    Save data using pickle
    
    Args:
        data: Data to serialize
        filepath: Path to save file
        ensure_directory: Whether to create parent directory if needed
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import pickle
        
        if ensure_directory:
            directory = os.path.dirname(filepath)
            if directory:
                ensure_dir(directory)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save pickle to {filepath}: {e}")
        return False

def load_pickle(filepath: str, default: Any = None) -> Any:
    """
    Load data from pickle file
    
    Args:
        filepath: Path to pickle file
        default: Default value to return if file doesn't exist or load fails
    
    Returns:
        Loaded data, or default value if failed
    """
    try:
        import pickle
        
        if not exists(filepath):
            logger.warning(f"Pickle file not found: {filepath}")
            return default
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle from {filepath}: {e}")
        return default

def get_directory_size(directory: str) -> int:
    """
    Get total size of directory in bytes
    
    Args:
        directory: Directory path
    
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = join_paths(dirpath, filename)
                total_size += get_file_size(filepath)
    except Exception as e:
        logger.warning(f"Failed to calculate directory size: {e}")
    
    return total_size

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"




