#<!-- filepath: /root/IRS/utils/memory.py -->
#!/usr/bin/env python3
# Memory management utilities for IRS Tax Analysis System

import os
import gc
import time
import psutil
import logging
from typing import List, Optional, Dict, Union, Callable
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("memory_utils")

class MemoryTracker:
    """Class to track and report on memory usage."""
    
    def __init__(self, log_interval: int = 60, save_history: bool = False):
        """Initialize memory tracker.
        
        Args:
            log_interval: Interval in seconds for automatic logging
            save_history: Whether to save full history (can consume memory)
        """
        self.log_interval = log_interval
        self.save_history = save_history
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.baseline = self.get_current_usage()
        self.peak = self.baseline
        self.history = [self.baseline] if save_history else []
    
    def get_current_usage(self) -> Dict[str, Union[float, int, List]]:
        """Get current memory usage."""
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        
        # Get GPU memory if available
        gpu_memory = self._get_gpu_memory()
        
        return {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "system_percent": vm.percent,
            "system_used_gb": vm.used / (1024 ** 3),
            "system_available_gb": vm.available / (1024 ** 3),
            "process_rss_gb": process.memory_info().rss / (1024 ** 3),
            "process_vms_gb": process.memory_info().vms / (1024 ** 3),
            "gpu_memory": gpu_memory
        }
    
    def _get_gpu_memory(self) -> List[Dict[str, Union[int, float]]]:
        """Get GPU memory usage if available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return []
                
            result = []
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                free = total_mem - reserved
                
                result.append({
                    "device": i,
                    "total_gb": total_mem,
                    "reserved_gb": reserved,
                    "allocated_gb": allocated,
                    "free_gb": free
                })
            return result
        except (ImportError, AttributeError, RuntimeError):
            return []
    
    def check_and_log(self, force: bool = False, label: Optional[str] = None) -> Dict[str, Union[float, int, List]]:
        """Check current memory usage and log if interval exceeded or forced.
        
        Args:
            force: Whether to force logging regardless of interval
            label: Optional label for the log entry
            
        Returns:
            Current memory usage data
        """
        current = self.get_current_usage()
        if self.save_history:
            self.history.append(current)
        
        # Update peak memory
        if current["system_used_gb"] > self.peak["system_used_gb"]:
            self.peak = current
        
        current_time = time.time()
        if force or (current_time - self.last_log_time) >= self.log_interval:
            self.last_log_time = current_time
            
            # Calculate changes
            baseline_used = self.baseline["system_used_gb"]
            current_used = current["system_used_gb"]
            change = current_used - baseline_used
            change_pct = (change / baseline_used) * 100 if baseline_used > 0 else 0
            
            log_prefix = f"{label}: " if label else ""
            logger.info(
                f"{log_prefix}Memory: {current['system_percent']:.1f}% used "
                f"({current_used:.2f} GB), "
                f"Change: {change_pct:+.1f}% ({change:+.2f} GB)"
            )
            
            # Log GPU memory if available
            for gpu in current.get("gpu_memory", []):
                logger.info(
                    f"GPU {gpu['device']}: {gpu['allocated_gb']:.2f} GB allocated, "
                    f"{gpu['free_gb']:.2f} GB free"
                )
        
        return current
    
    def report(self) -> None:
        """Generate a full memory usage report."""
        current = self.get_current_usage()
        
        logger.info("==== Memory Usage Report ====")
        logger.info(f"Elapsed time: {current['elapsed']:.1f} seconds")
        logger.info(f"Current system memory: {current['system_percent']:.1f}% used "
                   f"({current['system_used_gb']:.2f} GB used, "
                   f"{current['system_available_gb']:.2f} GB available)")
        
        logger.info(f"Peak system memory: {self.peak['system_percent']:.1f}% used "
                   f"({self.peak['system_used_gb']:.2f} GB used)")
        
        baseline_used = self.baseline["system_used_gb"]
        current_used = current["system_used_gb"]
        change = current_used - baseline_used
        change_pct = (change / baseline_used) * 100 if baseline_used > 0 else 0
        
        logger.info(f"Memory change: {change_pct:+.1f}% ({change:+.2f} GB)")
        
        logger.info(f"Process memory: {current['process_rss_gb']:.2f} GB (RSS), "
                   f"{current['process_vms_gb']:.2f} GB (VMS)")
        
        # Log GPU memory if available
        for gpu in current.get("gpu_memory", []):
            logger.info(
                f"GPU {gpu['device']}: {gpu['allocated_gb']:.2f} GB allocated, "
                f"{gpu['free_gb']:.2f} GB free out of {gpu['total_gb']:.2f} GB total"
            )
        
        logger.info("============================")

class MemoryOptimizer:
    """Class to optimize memory usage for LLM operations."""
    
    @staticmethod
    def clean(model_name: Optional[str] = None) -> None:
        """Clean up memory after model execution.
        
        Args:
            model_name: Optional name of model that was cleaned up
        """
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"Cleared CUDA cache{f' after {model_name}' if model_name else ''}")
        except (ImportError, AttributeError):
            pass
    
    @staticmethod
    def optimize_for_inference(model_name: str) -> Dict[str, Union[str, int, bool]]:
        """Optimize memory settings for model inference.
        
        Args:
            model_name: Name of the model to optimize for
            
        Returns:
            Dictionary with optimization parameters
        """
        # Default configuration
        config = {
            "batch_size": 1,
            "max_tokens": 2048,
            "use_fp16": True,
            "use_cache": True,
            "offload_to_cpu": False,
            "quantization": None
        }
        
        # Check available memory
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024 ** 3)
        
        # Check for GPU
        has_gpu = False
        gpu_memory_gb = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                device = torch.cuda.current_device()
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        except (ImportError, AttributeError):
            pass
        
        logger.info(f"Optimizing for model: {model_name}")
        logger.info(f"Available system memory: {available_gb:.2f} GB")
        if has_gpu:
            logger.info(f"Available GPU memory: {gpu_memory_gb:.2f} GB")
        
        # Model-specific optimizations based on model name and available resources
        if "llama3:70b" in model_name.lower():
            if not has_gpu or gpu_memory_gb < 40:
                config["offload_to_cpu"] = True
                config["quantization"] = "int8"
            config["max_tokens"] = 1024  # Be more conservative
        
        elif "mixtral:8x7b" in model_name.lower():
            if not has_gpu or gpu_memory_gb < 16:
                config["quantization"] = "int8"
        
        elif "yi:34b" in model_name.lower():
            if not has_gpu or gpu_memory_gb < 24:
                config["offload_to_cpu"] = True
                config["quantization"] = "int8"
        
        # Apply general optimizations based on available memory
        if available_gb < 8:
            logger.warning("Low memory detected, applying aggressive optimizations")
            config["max_tokens"] = min(config["max_tokens"], 512)
            config["quantization"] = "int4" if not config["quantization"] else config["quantization"]
            config["offload_to_cpu"] = True
        
        logger.info(f"Memory optimization config: {config}")
        return config

def memory_usage_decorator(func: Callable) -> Callable:
    """Decorator to track memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        tracker = MemoryTracker()
        
        logger.info(f"Starting {function_name}")
        tracker.check_and_log(force=True, label=f"{function_name} start")
        
        try:
            result = func(*args, **kwargs)
        finally:
            tracker.check_and_log(force=True, label=f"{function_name} end")
            tracker.report()
            MemoryOptimizer.clean()
        
        return result
    
    return wrapper

if __name__ == "__main__":
    # Simple demonstration
    tracker = MemoryTracker(log_interval=10)
    
    # Initial check
    tracker.check_and_log(force=True, label="Initial")
    
    # Simulate memory allocation
    big_list = [0] * 10**8  # Allocate ~800MB
    
    # Check after allocation
    tracker.check_and_log(force=True, label="After allocation")
    
    # Clean memory
    MemoryOptimizer.clean()
    del big_list
    
    # Final report
    tracker.check_and_log(force=True, label="After cleanup")
    tracker.report()
    
    # Example optimization for different models
    for model in ["llama3:8b", "phi4:medium", "mixtral:8x7b", "yi:34b", "llama3:70b"]:
        config = MemoryOptimizer.optimize_for_inference(model)