#<!-- filepath: /root/IRS/utils/debug.py -->
#!/usr/bin/env python3
# Debug utilities for IRS Tax Analysis System

import os
import sys
import time
import logging
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import ModelManager
from utils.system import get_system_info
from utils.memory import MemoryTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("debug")

def check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available and working properly.
    
    Args:
        model_name: Name of model to check
    
    Returns:
        True if model is working, False otherwise
    """
    logger.info(f"Checking Ollama model: {model_name}")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Step 1: Check connectivity to Ollama
    if not model_manager.check_connectivity():
        logger.error("Cannot connect to Ollama. Please ensure Ollama is running.")
        return False
    
    # Step 2: Check if model is available
    if not model_manager.is_model_available(model_name):
        logger.warning(f"Model {model_name} is not available. Attempting to pull it...")
        if not model_manager.pull_model(model_name):
            logger.error(f"Failed to pull model {model_name}")
            return False
        logger.info(f"Successfully pulled model {model_name}")
    else:
        logger.info(f"Model {model_name} is available")
    
    # Step 3: Test generation with a simple prompt
    logger.info(f"Testing model {model_name} with simple prompt...")
    
    # Start memory tracking
    memory_tracker = MemoryTracker()
    memory_tracker.check_and_log(force=True, label="Before model run")
    
    start_time = time.time()
    response = model_manager.generate(
        model_name=model_name,
        prompt="What is Form 1040?",
        options={"temperature": 0.1, "num_predict": 100}  # Small response for testing
    )
    end_time = time.time()
    
    # Check memory after run
    memory_tracker.check_and_log(force=True, label="After model run")
    
    if "ERROR" in response:
        logger.error(f"Model test failed: {response}")
        return False
    
    # Log results
    duration = end_time - start_time
    logger.info(f"Model test completed in {duration:.2f} seconds")
    logger.info(f"Response preview: {response[:100]}...")
    
    return True

def list_available_models() -> List[str]:
    """List all available Ollama models.
    
    Returns:
        List of available model names
    """
    model_manager = ModelManager()
    
    if not model_manager.check_connectivity():
        logger.error("Cannot connect to Ollama. Please ensure Ollama is running.")
        return []
    
    models = model_manager.get_available_models()
    return models

def diagnose_system() -> Dict[str, Any]:
    """Run system diagnostics and checks.
    
    Returns:
        Dictionary with diagnostic results
    """
    results = {}
    
    # Step 1: Get system info
    sys_info = get_system_info()
    results["system_info"] = sys_info
    
    # Step 2: Check GPU availability
    if sys_info.get("gpu_info"):
        results["gpu_available"] = True
        results["gpu_info"] = sys_info["gpu_info"]
    else:
        results["gpu_available"] = False
    
    # Step 3: Check RAM
    ram_gb = sys_info.get("total_memory_gb", 0)
    results["ram_gb"] = ram_gb
    results["ram_sufficient"] = ram_gb >= 8
    
    # Step 4: Check Ollama connectivity
    model_manager = ModelManager()
    ollama_connected = model_manager.check_connectivity()
    results["ollama_connected"] = ollama_connected
    
    if ollama_connected:
        # Get available models
        models = model_manager.get_available_models()
        results["available_models"] = models
    
    return results

def main():
    """Main function to run diagnostic tools."""
    parser = argparse.ArgumentParser(description="IRS Tax Analysis System Debug Tools")
    parser.add_argument("--model", "-m", help="Check specific Ollama model")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run system diagnostics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run diagnostics
    if args.diagnose:
        logger.info("Running system diagnostics...")
        results = diagnose_system()
        
        print("\n=== SYSTEM DIAGNOSTICS ===")
        print(f"CPU: {results['system_info'].get('processor', 'Unknown')}")
        print(f"RAM: {results['system_info'].get('total_memory_gb', 0):.1f} GB")
        print(f"GPU: {'Available' if results.get('gpu_available', False) else 'Not available'}")
        
        if results.get("gpu_available", False):
            for gpu in results["gpu_info"]:
                print(f"  - {gpu.get('name', 'Unknown GPU')}: {gpu.get('total_memory_mb', 0)/1024:.1f} GB")
        
        print(f"Ollama: {'Connected' if results.get('ollama_connected', False) else 'Not connected'}")
        
        if results.get("ollama_connected", False):
            models = results.get("available_models", [])
            print(f"Available models: {', '.join(models) if models else 'None'}")
    
    # List models
    if args.list_models:
        models = list_available_models()
        
        print("\n=== AVAILABLE MODELS ===")
        if models:
            for model in models:
                print(f"- {model}")
        else:
            print("No models available or Ollama is not running")
    
    # Check specific model
    if args.model:
        is_working = check_ollama_model(args.model)
        
        if is_working:
            print(f"\n✅ Model {args.model} is working correctly")
        else:
            print(f"\n❌ Model {args.model} is not working properly")
    
    # Default: show help if no arguments
    if not (args.model or args.list_models or args.diagnose):
        parser.print_help()

if __name__ == "__main__":
    main()