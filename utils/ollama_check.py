#!/usr/bin/env python3
"""
Ollama connectivity checker and troubleshooter.
This utility helps diagnose and fix common Ollama connection issues.
"""

import os
import sys
import time
import subprocess
import requests
import logging
import platform

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ollama_check')

OLLAMA_API_BASE = "http://localhost:11434/api"

def check_ollama_installed():
    """Check if Ollama is installed on the system."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], 
                                   capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "ollama"], 
                                   capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Ollama found at: {result.stdout.strip()}")
            return True, result.stdout.strip()
        else:
            logger.warning("Ollama not found in PATH")
            return False, None
    except Exception as e:
        logger.error(f"Error checking Ollama installation: {e}")
        return False, None

def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/version", timeout=3)
        if response.status_code == 200:
            version = response.json().get('version', 'unknown')
            logger.info(f"Ollama is running (version: {version})")
            return True, version
        else:
            logger.warning(f"Ollama responded with status code {response.status_code}")
            return False, None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama service is not running or not accessible: {e}")
        return False, None

def start_ollama_service():
    """Attempt to start the Ollama service."""
    try:
        if platform.system() == "Windows":
            # On Windows, start as a detached process
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.DETACHED_PROCESS)
        else:
            # On Unix systems, start in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           start_new_session=True)
        
        # Wait for service to start
        logger.info("Starting Ollama service...")
        for _ in range(5):
            time.sleep(2)
            running, _ = check_ollama_running()
            if running:
                return True
        
        logger.error("Ollama service started but failed to respond")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama service: {e}")
        return False

def check_model_availability(model_name):
    """Check if a specific model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            if model_name in model_names:
                logger.info(f"Model '{model_name}' is available")
                return True
            else:
                logger.info(f"Model '{model_name}' is not available. Available models: {', '.join(model_names)}")
                return False
        else:
            logger.warning(f"Failed to get model list, status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking model availability: {e}")
        return False

def pull_model(model_name):
    """Pull a model from Ollama."""
    try:
        logger.info(f"Pulling model '{model_name}'...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        logger.info(f"Successfully pulled model '{model_name}'")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling model '{model_name}': {e}")
        return False

def diagnose_and_fix():
    """Run diagnostics and attempt to fix Ollama connectivity issues."""
    logger.info("Running Ollama diagnostics...")
    
    # Check if Ollama is installed
    installed, path = check_ollama_installed()
    if not installed:
        logger.error("Ollama is not installed or not in PATH")
        logger.info("Please install Ollama from https://ollama.com/download")
        return False
    
    # Check if Ollama is running
    running, version = check_ollama_running()
    if not running:
        logger.warning("Ollama service is not running")
        
        # Try to start Ollama
        logger.info("Attempting to start Ollama service...")
        if start_ollama_service():
            logger.info("Successfully started Ollama service")
        else:
            logger.error("Failed to start Ollama service automatically")
            logger.info("Please start Ollama manually with 'ollama serve'")
            return False
    
    # Check default models
    default_models = ["llama3:8b", "phi4:medium"]
    for model in default_models:
        if not check_model_availability(model):
            logger.info(f"Model {model} not found, attempting to pull...")
            pull_model(model)
    
    logger.info("Ollama diagnostics completed")
    return True

if __name__ == "__main__":
    print("Ollama Connectivity Diagnostics Tool")
    print("===================================")
    success = diagnose_and_fix()
    if success:
        print("\nDiagnostics completed successfully.")
        print("Ollama should now be properly configured and running.")
    else:
        print("\nSome issues were detected and could not be automatically resolved.")
        print("Please check the logs above and follow the manual troubleshooting steps.")
    
    sys.exit(0 if success else 1)