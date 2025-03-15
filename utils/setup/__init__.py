#!/usr/bin/env python3
"""
Setup utilities for the IRS Tax Analysis System.
Handles environment initialization and dependency management.
"""

import os
import sys
import subprocess
import logging
import platform
from pathlib import Path
import requests
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('setup')

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

def create_directories():
    """Create necessary directories for the IRS Tax Analysis System."""
    dirs = [
        ROOT_DIR / "data" / "docs",
        ROOT_DIR / "data" / "chroma_db",
        ROOT_DIR / "data" / "models",
        ROOT_DIR / "logs"
    ]
    
    for directory in dirs:
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def create_virtualenv():
    """Create a Python virtual environment."""
    venv_dir = ROOT_DIR / "venv"
    
    if venv_dir.exists():
        logger.info("Virtual environment already exists")
        return True
    
    try:
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        logger.info(f"Virtual environment created at {venv_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating virtual environment: {e}")
        return False

def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    requirements_file = ROOT_DIR / "requirements.txt"
    venv_pip = ROOT_DIR / "venv" / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"
    
    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        logger.info("Installing dependencies...")
        subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_pip), "install", "-r", str(requirements_file)], check=True)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        # Check if Ollama is installed
        logger.info("Checking Ollama installation...")
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("Ollama is not installed or not in PATH")
            logger.info("Please install Ollama from https://ollama.com/download")
            return False
        
        # Check if Ollama is running
        logger.info("Checking if Ollama is running...")
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=3)
            if response.status_code == 200:
                version = response.json().get('version', 'unknown')
                logger.info(f"Ollama is running (version: {version})")
                return True
        except requests.exceptions.RequestException:
            logger.warning("Ollama service is not running")
            
            # Try to start Ollama
            try:
                logger.info("Attempting to start Ollama service...")
                if platform.system() == "Windows":
                    subprocess.Popen(["ollama", "serve"], 
                                   creationflags=subprocess.DETACHED_PROCESS)
                else:
                    subprocess.Popen(["ollama", "serve"], 
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   start_new_session=True)
                
                # Wait for service to start
                for _ in range(5):
                    time.sleep(2)
                    try:
                        response = requests.get("http://localhost:11434/api/version", timeout=3)
                        if (response.status_code == 200):
                            logger.info("Ollama service started successfully")
                            return True
                    except requests.exceptions.RequestException:
                        pass
                
                logger.error("Failed to start Ollama service automatically")
                logger.info("Please start Ollama manually with 'ollama serve'")
            except Exception as e:
                logger.error(f"Error starting Ollama service: {e}")
            
            return False
        
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        return False

def pull_ollama_models(models=None):
    """Pull Ollama models."""
    if models is None:
        models = ["llama3:8b", "phi4:medium"]
    
    if not isinstance(models, list):
        models = [models]
    
    logger.info(f"Pulling models: {', '.join(models)}")
    
    # Check if Ollama is running first
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        if response.status_code != 200:
            logger.error("Ollama is not running. Start Ollama with 'ollama serve' and try again.")
            return False
    except requests.exceptions.RequestException:
        logger.error("Ollama is not running. Start Ollama with 'ollama serve' and try again.")
        return False
    
    success = True
    for model in models:
        try:
            # Skip invalid models (like script paths that might have been passed incorrectly)
            if "/" in model and os.path.exists(model):
                logger.warning(f"Skipping invalid model name that looks like a file path: {model}")
                continue
                
            logger.info(f"Pulling model: {model}")
            result = subprocess.run(["ollama", "pull", model], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Successfully pulled model: {model}")
            else:
                logger.error(f"Failed to pull model {model}: {result.stderr}")
                success = False
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            success = False
    
    return success

def setup_environment(models=None, retry_ollama=False):
    """
    Set up the environment for the IRS Tax Analysis System.
    
    Args:
        models: List of Ollama models to pull
        retry_ollama: If True, retry Ollama setup if it fails
    """
    logger.info("Setting up the IRS Tax Analysis System environment...")
    
    # Create directories
    create_directories()
    
    # Create virtual environment if it doesn't exist
    if not create_virtualenv():
        logger.error("Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Initialize vector database
    logger.info("Initializing vector database...")
    venv_python = ROOT_DIR / "venv" / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    db_init_script = ROOT_DIR / "core" / "rag.py"
    
    if db_init_script.exists():
        try:
            subprocess.run([str(venv_python), str(db_init_script), "--init"], check=True)
            logger.info("Vector database initialized successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize vector database: {e}")
            logger.info("Creating core directories anyway...")
            (ROOT_DIR / "data" / "chroma_db").mkdir(parents=True, exist_ok=True)
    else:
        logger.error(f"RAG initialization script not found at {db_init_script}")
        logger.info("Creating core directories anyway...")
        (ROOT_DIR / "data" / "chroma_db").mkdir(parents=True, exist_ok=True)
    
    # Check Ollama installation and setup
    ollama_success = check_ollama()
    if ollama_success:
        pull_success = pull_ollama_models(models)
        if not pull_success and retry_ollama:
            # Run the dedicated Ollama diagnostic tool
            logger.info("Running Ollama diagnostics...")
            ollama_check_script = ROOT_DIR / "utils" / "ollama_check.py"
            if ollama_check_script.exists():
                try:
                    subprocess.run([str(venv_python), str(ollama_check_script)], check=True)
                except subprocess.CalledProcessError:
                    logger.error("Failed to fix Ollama issues automatically")
                    logger.info("Please run './irs.sh diagnose ollama' for detailed diagnostics")
    elif retry_ollama:
        # Run the dedicated Ollama diagnostic tool
        logger.info("Running Ollama diagnostics...")
        ollama_check_script = ROOT_DIR / "utils" / "ollama_check.py"
        if ollama_check_script.exists():
            try:
                subprocess.run([str(venv_python), str(ollama_check_script)], check=True)
            except subprocess.CalledProcessError:
                logger.error("Failed to fix Ollama issues automatically")
                logger.info("Please run './irs.sh diagnose ollama' for detailed diagnostics")
    else:
        logger.warning("Ollama setup skipped or failed. Models will not be available.")
        logger.info("To retry Ollama setup: './irs.sh setup --retry'")
    
    logger.info("Setup completed!")
    return True

if __name__ == "__main__":
    retry = "--retry" in sys.argv
    models = ["llama3:8b", "phi4:medium"]  # Default models
    
    # Check for specific model argument
    for arg in sys.argv[1:]:  # Skip first arg (script name)
        if arg not in ["--retry"] and not arg.startswith("--"):
            models = [arg]
            break
    
    setup_environment(models, retry)