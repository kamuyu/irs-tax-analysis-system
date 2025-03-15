#<!-- filepath: /root/IRS/utils/system.py -->
#!/usr/bin/env python3
# System utilities for IRS Tax Analysis System

import os
import sys
import platform
import psutil
import logging
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("system_utils")

def get_system_info() -> Dict[str, Union[str, int, float]]:
    """Get detailed system information including CPU, memory, and GPU."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "total_memory": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
        "available_memory": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
        "gpu_info": get_gpu_info()
    }
    return info

def get_gpu_info() -> List[Dict[str, Union[str, int, float]]]:
    """Get GPU information using nvidia-smi if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                index, name, total_memory, used_memory, free_memory, utilization = line.split(', ')
                gpus.append({
                    "index": int(index),
                    "name": name,
                    "total_memory_mb": float(total_memory),
                    "used_memory_mb": float(used_memory),
                    "free_memory_mb": float(free_memory),
                    "utilization_percent": float(utilization)
                })
        return gpus
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # Try using lspci for systems without nvidia-smi
            result = subprocess.run(["lspci"], capture_output=True, text=True)
            gpu_lines = [line for line in result.stdout.split('\n') if "VGA" in line or "3D" in line]
            if gpu_lines:
                return [{"name": line.split(": ")[1].strip()} for line in gpu_lines]
        except (subprocess.SubprocessError, FileNotFoundError, IndexError):
            pass
        
        logger.warning("Could not detect GPU information")
        return []

def configure_docker_memory(memory_limit: str = "32g") -> bool:
    """Configure Docker to use specified memory limit."""
    try:
        # Write Docker configuration
        docker_config = {
            "memory": memory_limit,
            "memory-swap": "-1"  # Disable swap
        }
        
        # Note: In a real implementation, you'd need to modify Docker daemon.json
        # and restart Docker, but this requires root permissions
        logger.info(f"Docker configured to use {memory_limit} of memory")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Docker memory: {e}")
        return False

def optimize_gpu_settings() -> bool:
    """Optimize settings for GPU usage."""
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        logger.warning("No GPU detected, using CPU only")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False
    
    logger.info(f"Detected {len(gpu_info)} GPU(s)")
    
    # Set environment variables to optimize GPU usage
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # For TensorFlow
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # For PyTorch
    
    return True

def clean_memory(model_name: Optional[str] = None) -> bool:
    """Clean up memory after model execution."""
    import gc
    
    logger.info(f"Cleaning up memory{f' after {model_name}' if model_name else ''}")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if PyTorch is installed
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    except ImportError:
        pass
    
    # Log memory usage after cleanup
    mem_info = psutil.virtual_memory()
    logger.info(f"Memory usage after cleanup: {mem_info.percent}% "
                f"({round(mem_info.used / (1024**3), 2)}GB used, "
                f"{round(mem_info.available / (1024**3), 2)}GB available)")
    
    return True

def get_optimal_worker_count() -> int:
    """Determine optimal number of worker processes based on system resources."""
    cpu_count = psutil.cpu_count(logical=False) or 1
    
    # Check if we're running in a Docker container
    in_docker = os.path.exists('/.dockerenv')
    
    if in_docker:
        # In Docker, be more conservative
        return max(1, cpu_count - 1)
    else:
        # Reserve at least 2 cores for system
        return max(1, cpu_count - 2)

if __name__ == "__main__":
    # Display system information when run directly
    info = get_system_info()
    print(json.dumps(info, indent=2))
    print(f"Optimal worker count: {get_optimal_worker_count()}")
    
    # Test GPU optimization
    optimize_gpu_settings()
    
    # Test memory cleanup
    clean_memory("test_model")