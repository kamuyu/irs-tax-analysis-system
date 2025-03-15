#<!-- filepath: /root/IRS/core/models.py -->
#!/usr/bin/env python3
# Model integration for IRS Tax Analysis System

import os
import json
import time
import requests
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import threading
from pathlib import Path
import unittest
import concurrent.futures

# Import custom utilities
from utils.memory import MemoryOptimizer, memory_usage_decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("models")

class ModelManager:
    """Class to manage LLM models via Ollama"""
    
    def __init__(self, api_base: str = "http://localhost:11434"):
        """Initialize with API base URL"""
        self.api_base = api_base
        self.api_endpoints = {
            "generate": f"{api_base}/api/generate",
            "chat": f"{api_base}/api/chat",
            "embeddings": f"{api_base}/api/embeddings",
            "list": f"{api_base}/api/tags",
            "pull": f"{api_base}/api/pull",
        }
        self.available_models = []
        self.model_lock = threading.Lock()
        
    def check_connectivity(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.api_base}/api/version")
            if response.status_code == 200:
                logger.info(f"Connected to Ollama version: {response.json().get('version', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(self.api_endpoints["list"])
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = [model.get("name") for model in models]
                logger.info(f"Available models: {', '.join(self.available_models)}")
                return self.available_models
            else:
                logger.error(f"Failed to retrieve models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        if not self.available_models:
            self.get_available_models()
        
        return model_name in self.available_models
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama"""
        try:
            logger.info(f"Pulling model {model_name}. This might take a while...")
            
            # Use subprocess for more robust model pulling
            result = subprocess.run(["ollama", "pull", model_name], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully pulled model: {model_name}")
                if model_name not in self.available_models:
                    self.available_models.append(model_name)
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    @memory_usage_decorator
    def generate(self, model_name: str, prompt: str, stream: bool = False, options: Dict[str, Any] = None) -> str:
        """Generate text using specified model.
        
        Args:
            model_name: Name of the model to use
            prompt: Text prompt to send to the model
            stream: Whether to stream the response
            options: Additional options for generation
            
        Returns:
            Generated text
        """
        if not self.is_model_available(model_name):
            logger.warning(f"Model {model_name} not found. Attempting to pull it...")
            if not self.pull_model(model_name):
                logger.error(f"Failed to pull model {model_name}")
                return f"ERROR: Model {model_name} not available."
        
        # Default options
        default_options = {
            "temperature": 0.7,
            "num_predict": 1024,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["<|endoftext|>"]
        }
        
        # Apply memory optimization for specific model
        memory_config = MemoryOptimizer.optimize_for_inference(model_name)
        if memory_config.get("use_fp16", True):
            default_options["f16_kv"] = True
        
        if memory_config.get("max_tokens"):
            default_options["num_predict"] = memory_config["max_tokens"]
        
        # Override with provided options
        if options:
            default_options.update(options)
        
        # Prepare request data
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": default_options
        }
        
        start_time = time.time()
        logger.info(f"Generating with model {model_name}")
        
        try:
            with self.model_lock:  # Ensure only one model runs at a time to optimize GPU usage
                if stream:
                    return self._stream_response(request_data)
                else:
                    return self._sync_response(request_data)
        except Exception as e:
            logger.error(f"Error generating with model {model_name}: {e}")
            return f"ERROR: Generation failed - {str(e)}"
        finally:
            duration = time.time() - start_time
            logger.info(f"Generation with {model_name} completed in {duration:.2f} seconds")
    
    def _sync_response(self, request_data: Dict[str, Any]) -> str:
        """Send synchronous request to Ollama API"""
        response = requests.post(
            self.api_endpoints["generate"],
            json=request_data,
            timeout=300  # 5-minute timeout for long responses
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _stream_response(self, request_data: Dict[str, Any]) -> str:
        """Stream response from Ollama API"""
        full_response = []
        
        with requests.post(
            self.api_endpoints["generate"],
            json=request_data,
            stream=True,
            timeout=300  # 5-minute timeout
        ) as response:
            if response.status_code != 200:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            for line in response.iter_lines():
                if line:
                    line_json = json.loads(line)
                    chunk = line_json.get("response", "")
                    full_response.append(chunk)
                    
                    # Print chunk for streaming output
                    print(chunk, end="", flush=True)
                    
                    # Check if done
                    if line_json.get("done", False):
                        break
        
        print()  # Newline at end of streaming
        return "".join(full_response)
    
    def generate_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """Generate embedding for text using Ollama.
        
        Args:
            text: Text to embed
            model_name: Optional model name (will use default embedding model if not specified)
            
        Returns:
            Embedding vector as list of floats
        """
        if model_name and not self.is_model_available(model_name):
            logger.warning(f"Model {model_name} not found for embedding. Using default.")
            model_name = None
        
        # Use default model if none specified
        if not model_name:
            model_name = "llama3:8b"  # Default embedding model
        
        # Prepare request
        request_data = {
            "model": model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(
                self.api_endpoints["embeddings"],
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.error(f"Embedding API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def run_models_in_parallel(self, prompt: str, model_names: List[str], 
                              max_workers: Optional[int] = None) -> Dict[str, str]:
        """Run multiple models in parallel, optimizing for GPU usage.
        
        Args:
            prompt: The prompt to send to all models
            model_names: List of model names to run
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}
        
        # Check available models first
        available_models = []
        for model_name in model_names:
            if self.is_model_available(model_name) or self.pull_model(model_name):
                available_models.append(model_name)
            else:
                results[model_name] = f"ERROR: Model {model_name} not available."
        
        # Run CPU-intensive preprocessing first
        # This is where we would prepare the query, get embeddings, etc.
        
        # Then process models one at a time to optimize GPU usage
        for model_name in available_models:
            try:
                logger.info(f"Running model {model_name}")
                result = self.generate(model_name, prompt)
                results[model_name] = result
                
                # Clean up after each model run
                MemoryOptimizer.clean(model_name)
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                results[model_name] = f"ERROR: {str(e)}"
                
        return results

# Unit tests
class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.model_manager = ModelManager()
    
    def test_connectivity(self):
        try:
            is_connected = self.model_manager.check_connectivity()
            self.assertTrue(is_connected, "Ollama server should be accessible")
        except Exception as e:
            self.skipTest(f"Skipping test because Ollama server is not accessible: {e}")
    
    def test_get_available_models(self):
        try:
            models = self.model_manager.get_available_models()
            self.assertIsInstance(models, list, "Should return a list of available models")
        except Exception as e:
            self.skipTest(f"Skipping test because Ollama server is not accessible: {e}")

if __name__ == "__main__":
    # Simple demonstration
    manager = ModelManager()
    
    # Check connectivity
    if manager.check_connectivity():
        print("Connected to Ollama")
        
        # Show available models
        models = manager.get_available_models()
        print(f"Available models: {models}")
        
        # Generate text with a model if available
        if "llama3:8b" in models:
            result = manager.generate("llama3:8b", "What is a 1040 tax form?", stream=True)
            print("\nFinal result:", result)
    else:
        print("Could not connect to Ollama")