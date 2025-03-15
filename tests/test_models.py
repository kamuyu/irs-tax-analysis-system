#<!-- filepath: /root/IRS/tests/test_models.py -->
#!/usr/bin/env python3
# Unit tests for model integration

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import ModelManager

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.model_manager = ModelManager()
    
    @patch('requests.get')
    def test_check_connectivity_success(self, mock_get):
        """Test successful Ollama connectivity check"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response
        
        # Check connectivity
        result = self.model_manager.check_connectivity()
        
        # Assertions
        self.assertTrue(result)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_check_connectivity_failure(self, mock_get):
        """Test failed Ollama connectivity check"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Check connectivity
        result = self.model_manager.check_connectivity()
        
        # Assertions
        self.assertFalse(result)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_check_connectivity_exception(self, mock_get):
        """Test exception during Ollama connectivity check"""
        # Mock exception
        mock_get.side_effect = Exception("Connection error")
        
        # Check connectivity
        result = self.model_manager.check_connectivity()
        
        # Assertions
        self.assertFalse(result)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_available_models(self, mock_get):
        """Test getting available models"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:8b"},
                {"name": "phi4:medium"},
                {"name": "mixtral:8x7b"}
            ]
        }
        mock_get.return_value = mock_response
        
        # Get available models
        result = self.model_manager.get_available_models()
        
        # Assertions
        self.assertEqual(len(result), 3)
        self.assertIn("llama3:8b", result)
        self.assertIn("phi4:medium", result)
        self.assertIn("mixtral:8x7b", result)
        mock_get.assert_called_once()
    
    @patch('subprocess.run')
    def test_pull_model_success(self, mock_run):
        """Test successful model pull"""
        # Mock successful subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        # Pull model
        result = self.model_manager.pull_model("llama3:8b")
        
        # Assertions
        self.assertTrue(result)
        mock_run.assert_called_once()
        self.assertIn("llama3:8b", self.model_manager.available_models)
    
    @patch('subprocess.run')
    def test_pull_model_failure(self, mock_run):
        """Test failed model pull"""
        # Mock failed subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Model not found"
        mock_run.return_value = mock_result
        
        # Pull model
        result = self.model_manager.pull_model("nonexistent-model")
        
        # Assertions
        self.assertFalse(result)
        mock_run.assert_called_once()
    
    @patch('requests.post')
    def test_sync_response(self, mock_post):
        """Test synchronous response parsing"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response
        
        # Get response
        request_data = {"model": "llama3:8b", "prompt": "test", "stream": False}
        result = self.model_manager._sync_response(request_data)
        
        # Assertions
        self.assertEqual(result, "Test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_sync_response_error(self, mock_post):
        """Test error handling in synchronous response"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        # Get response
        request_data = {"model": "llama3:8b", "prompt": "test", "stream": False}
        
        # Assertions
        with self.assertRaises(Exception):
            self.model_manager._sync_response(request_data)
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()