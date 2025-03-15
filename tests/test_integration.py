#<!-- filepath: /root/IRS/tests/test_integration.py -->
#!/usr/bin/env python3
# Integration tests for the IRS Tax Analysis System

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.rag import DocumentProcessor, VectorDatabaseManager, Document, HybridRetriever
from core.analysis import TaxAnalyzer, FeedbackAnalyzer, AnalysisResult, ScenarioAnalysis
from core.models import ModelManager
from core.knowledge_graph import TaxKnowledgeGraph, TaxEntity

class TestIntegration(unittest.TestCase):
    """Integration tests for the IRS Tax Analysis System."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model manager that doesn't actually call Ollama
        self.model_manager = ModelManager()
        self.model_manager.generate = MagicMock(return_value="Mock answer to the tax question.")
        self.model_manager.check_connectivity = MagicMock(return_value=True)
        self.model_manager.get_available_models = MagicMock(return_value=["llama3:8b", "mixtral:8x7b"])
        self.model_manager.is_model_available = MagicMock(return_value=True)
        
        # Create test document
        self.doc = Document(
            content="Test Scenario: Tax Deduction\nThis is a test scenario.\n\nQuestion 1\nTest question?",
            metadata={"source": "test.txt", "filename": "test.txt"}
        )
        
        # Create document processor
        self.doc_processor = DocumentProcessor()
        
        # Create vector database manager with in-memory database for testing
        self.vector_db = VectorDatabaseManager(db_dir=os.path.join(self.temp_dir, "chroma_db"))
        
        # Create hybrid retriever
        self.retriever = HybridRetriever(self.vector_db)
        
        # Create mock retriever response
        self.retriever.retrieve = MagicMock(return_value=[
            {"text": "Relevant context for the question.", "metadata": {"source": "test.txt"}, "score": 0.95}
        ])
        
        # Create analyzers
        self.tax_analyzer = TaxAnalyzer(self.model_manager, self.retriever)
        self.feedback_analyzer = FeedbackAnalyzer(self.model_manager)
        
        # Create knowledge graph
        self.kg = TaxKnowledgeGraph(save_path=os.path.join(self.temp_dir, "test_kg.json"))
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_document_to_analysis_pipeline(self):
        """Test the pipeline from document to analysis."""
        # Parse document
        doc_info = self.doc_processor.parse_scenario_and_questions(self.doc)
        
        # Check parsing results
        self.assertEqual(doc_info["title"], "Test Scenario: Tax Deduction")
        self.assertIn("This is a test scenario", doc_info["scenario"])
        self.assertEqual(len(doc_info["questions"]), 1)
        
        # Initialize vector database
        self.vector_db.initialize = MagicMock()
        
        # Analyze scenario
        analysis = self.tax_analyzer.analyze_scenario(doc_info, "llama3:8b", self.temp_dir)
        
        # Check analysis results
        self.assertIsInstance(analysis, ScenarioAnalysis)
        self.assertEqual(analysis.