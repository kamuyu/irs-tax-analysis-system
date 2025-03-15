#<!-- filepath: /root/IRS/tests/test_end_to_end.py -->
#!/usr/bin/env python3
# End-to-end tests for the IRS Tax Analysis System

import os
import sys
import unittest
import shutil
import tempfile
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import ModelManager
from core.rag import DocumentProcessor, VectorDatabaseManager, HybridRetriever
from core.analysis import TaxAnalyzer, FeedbackAnalyzer
from apps.bulk.run import BulkProcessor

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the IRS Tax Analysis System."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.docs_dir = os.path.join(cls.temp_dir, "docs")
        cls.output_dir = os.path.join(cls.temp_dir, "output")
        cls.chroma_dir = os.path.join(cls.temp_dir, "chroma_db")
        
        os.makedirs(cls.docs_dir)
        os.makedirs(cls.output_dir)
        os.makedirs(cls.chroma_dir)
        
        # Create a test document
        cls.test_file = os.path.join(cls.docs_dir, "test_scenario.txt")
        with open(cls.test_file, "w") as f:
            f.write("Test Scenario: Basic Tax Deduction\n")
            f.write("John is a self-employed consultant who earns $100,000 per year.\n")
            f.write("He works from home and uses 20% of his home exclusively for business.\n")
            f.write("His monthly rent is $2,000 and utilities average $300 per month.\n\n")
            f.write("Question 1\n")
            f.write("How much can John deduct for his home office?\n")
            f.write("a) $4,800 per year\n")
            f.write("b) $5,520 per year\n")
            f.write("c) $24,000 per year\n")
            f.write("d) $0 because he rents his home\n\n")
            f.write("Question 2\n")
            f.write("Which tax form should John use to report his business income?\n")
            f.write("a) Schedule A\n")
            f.write("b) Schedule C\n")
            f.write("c) Schedule D\n")
            f.write("d) Form 1099\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def test_model_connectivity(self):
        """Test connection to Ollama API."""
        model_manager = ModelManager()
        try:
            is_connected = model_manager.check_connectivity()
            self.assertTrue(is_connected, "Should connect to Ollama API")
        except Exception as e:
            self.skipTest(f"Skipping test due to Ollama connection error: {e}")
    
    def test_document_processing(self):
        """Test document processing functionality."""
        processor = DocumentProcessor(self.docs_dir)
        documents = processor.load_text_files()
        
        self.assertEqual(len(documents), 1, "Should load one test document")
        
        # Parse scenario and questions
        doc_info = processor.parse_scenario_and_questions(documents[0])
        
        self.assertIn("Test Scenario", doc_info["title"])
        self.assertEqual(len(doc_info["questions"]), 2, "Should find 2 questions")
    
    @unittest.skipIf(not os.environ.get("RUN_SLOW_TESTS"), "Slow test skipped. Set RUN_SLOW_TESTS=1 to run.")
    def test_bulk_processing(self):
        """Test bulk processing with a real model (slow test)."""
        # Try to use a fast model for testing
        model_name = "llama3:8b"
        
        # Check if Ollama is available and the model exists
        model_manager = ModelManager()
        if not model_manager.check_connectivity():
            self.skipTest("Ollama is not available")
        
        if not model_manager.is_model_available(model_name):
            self.skipTest(f"Model {model_name} is not available")
        
        # Initialize bulk processor
        processor = BulkProcessor(
            models=[model_name],
            input_path=self.docs_dir,
            output_path=self.output_dir,
            feedback=False
        )
        
        # Run processing
        results = processor.process()
        
        # Check results
        self.assertIn("Test Scenario", results, "Should process the test scenario")
        self.assertIn(model_name, results["Test Scenario"], "Should have results for the model")
        
        # Check if output file was created
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0, "Should create output files")

if __name__ == "__main__":
    unittest.main()