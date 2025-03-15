#<!-- filepath: /root/IRS/tests/test_rag.py -->
#!/usr/bin/env python3
# Unit tests for RAG components

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.rag import DocumentProcessor, Document, TableData, VectorDatabaseManager

class TestDocument(unittest.TestCase):
    """Test cases for Document class"""
    
    def test_document_creation(self):
        """Test creating a document"""
        doc = Document(content="Test content", metadata={"source": "test.txt"})
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["source"], "test.txt")
        self.assertEqual(len(doc.tables), 0)
    
    def test_add_table(self):
        """Test adding a table to a document"""
        doc = Document(content="Test content", metadata={"source": "test.txt"})
        
        # Create a table
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        table = TableData(content=df, page_num=1, source_file="test.txt")
        
        # Add table to document
        doc.add_table(table)
        
        # Assertions
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].page_num, 1)
        self.assertEqual(doc.tables[0].source_file, "test.txt")

class TestTableData(unittest.TestCase):
    """Test cases for TableData class"""
    
    def test_to_markdown(self):
        """Test converting table to markdown"""
        # Create a table
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        table = TableData(
            content=df, 
            page_num=1, 
            source_file="test.txt",
            section_title="Test Section"
        )
        
        # Convert to markdown
        md = table.to_markdown()
        
        # Assertions
        self.assertIn("Table from test.txt", md)
        self.assertIn("Section: Test Section", md)
        self.assertIn("Page 1", md)
        self.assertIn("col1", md)
        self.assertIn("col2", md)
    
    def test_to_dict(self):
        """Test converting table to dict"""
        # Create a table
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        table = TableData(
            content=df, 
            page_num=1, 
            source_file="test.txt",
            context_before="Before text",
            context_after="After text"
        )
        
        # Convert to dict
        data_dict = table.to_dict()
        
        # Assertions
        self.assertEqual(data_dict["page_num"], 1)
        self.assertEqual(data_dict["source_file"], "test.txt")
        self.assertEqual(data_dict["context_before"], "Before text")
        self.assertEqual(data_dict["context_after"], "After text")
        self.assertIsInstance(data_dict["content"], dict)

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test documents
        self.temp_dir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self.temp_dir.name)
        
        # Create a test document file
        self.test_file = self.docs_dir / "test_scenario.txt"
        with open(self.test_file, "w") as f:
            f.write("Advanced Scenario 1: Tax Deductions\n")
            f.write("This is a test scenario about tax deductions.\n\n")
            f.write("Question 1:\n")
            f.write("What is a business deduction?\n")
            f.write("a) Option 1\n")
            f.write("b) Option 2\n\n")
            f.write("Question 2:\n")
            f.write("How much can be deducted?\n")
            f.write("a) $100\n")
            f.write("b) $200\n")
        
        # Initialize document processor
        self.processor = DocumentProcessor(str(self.docs_dir))
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_load_text_files(self):
        """Test loading text files"""
        documents = self.processor.load_text_files()
        
        # Assertions
        self.assertEqual(len(documents), 1)
        self.assertIn("tax deductions", documents[0].content.lower())
        self.assertEqual(documents[0].metadata["filename"], "test_scenario.txt")
    
    def test_parse_scenario_and_questions(self):
        """Test parsing scenario and questions"""
        # Load document first
        documents = self.processor.load_text_files()
        
        # Parse scenario and questions
        result = self.processor.parse_scenario_and_questions(documents[0])
        
        # Assertions
        self.assertEqual(result["title"], "Advanced Scenario 1: Tax Deductions")
        self.assertIn("test scenario about tax deductions", result["scenario"].lower())
        self.assertEqual(len(result["questions"]), 2)
        self.assertIn("business deduction", result["questions"][0].lower())
        self.assertIn("how much", result["questions"][1].lower())

if __name__ == "__main__":
    unittest.main()