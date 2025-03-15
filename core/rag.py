#!/usr/bin/env python3
"""
Retrieval Augmented Generation (RAG) module for IRS Tax Analysis System.
Handles vector database operations, embedding generation, and semantic search.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import shutil
import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict, field
import chromadb
from chromadb.config import Settings
import pandas as pd
import unittest
import time  # new import

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag')

# Define paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
CHROMA_DB_PATH = ROOT_DIR / "data" / "chroma_db"
ANSWERS_DIR = ROOT_DIR / "data" / "answers"
FEEDBACK_DIR = ROOT_DIR / "data" / "feedback"

# Create output directories
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TableData:
    """Class for representing extracted table data"""
    content: pd.DataFrame
    page_num: int
    source_file: str
    context_before: str = ""
    context_after: str = ""
    section_title: str = ""
    
    def to_markdown(self) -> str:
        """Convert the table to markdown format for LLM consumption"""
        md = f"Table from {os.path.basename(self.source_file)}"
        if self.section_title:
            md += f" (Section: {self.section_title})"
        md += f" (Page {self.page_num}):\n\n"
        md += self.content.to_markdown(index=False)
        return md
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content.to_dict(),
            "page_num": self.page_num,
            "source_file": self.source_file,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "section_title": self.section_title
        }

@dataclass
class Document:
    """Class representing a document with its content and metadata"""
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    tables: List[TableData] = field(default_factory=list)
    
    def add_table(self, table: TableData):
        """Add a table to the document"""
        self.tables.append(table)

class DocumentProcessor:
    """Class to process documents, extract text and tables"""
    
    def __init__(self, docs_dir: str = None):
        """Initialize with directory containing documents"""
        if docs_dir is None:
            docs_dir = str(ROOT_DIR / "data" / "docs")
        self.docs_dir = docs_dir
    
    def load_text_files(self) -> List[Document]:
        """Load all text files from the docs directory"""
        documents = []
        
        # Walk through the directory and load all text files
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            doc = Document(
                                content=content,
                                metadata={
                                    "source": file_path,
                                    "filename": file,
                                    "type": "text"
                                }
                            )
                            documents.append(doc)
                            logger.info(f"Loaded text file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")
        
        return documents
    
    def parse_scenario_and_questions(self, document: Document) -> Dict[str, Union[str, List[str]]]:
        """Parse a document into scenario and questions"""
        content = document.content
        lines = content.strip().split("\n")
        
        # Find scenario title and content
        scenario_title = lines[0].strip() if lines else "Unknown Scenario"
        
        # Split by blank lines to separate scenario from questions
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip():
                current_section.append(line)
            elif current_section:
                sections.append("\n".join(current_section))
                current_section = []
        
        if current_section:
            sections.append("\n".join(current_section))
        
        # First section is the scenario
        scenario = sections[0] if sections else ""
        
        # Remaining sections are questions
        questions = sections[1:] if len(sections) > 1 else []
        
        return {
            "title": scenario_title,
            "scenario": scenario,
            "questions": questions,
            "document": document
        }
    
    def process_all_documents(self) -> List[Document]:
        """Process all documents in the docs directory"""
        documents = self.load_text_files()
        return documents

class VectorDatabaseManager:
    """Class to manage vector database operations"""
    
    def __init__(self, db_dir: str = None, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize vector database manager"""
        if db_dir is None:
            db_dir = str(CHROMA_DB_PATH)
        self.db_dir = db_dir
        self.embedding_model = embedding_model
        self.db_client = None
        self.embeddings = None
        
    def initialize(self) -> None:
        """Initialize the vector database and embeddings"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_dir, exist_ok=True)
            
            # Initialize ChromaDB
            self.db_client = chromadb.PersistentClient(
                path=self.db_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embeddings
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer(self.embedding_model)
            
            logger.info(f"Vector database initialized at {self.db_dir}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    # ... remaining VectorDatabaseManager methods ...

class HybridRetriever:
    """Hybrid retrieval system combining RAG with knowledge graph elements"""
    
    def __init__(self, vector_db: VectorDatabaseManager, kg_enabled: bool = False):
        """Initialize hybrid retriever"""
        self.vector_db = vector_db
        self.kg_enabled = kg_enabled
        self.kg = None
        
        # Initialize knowledge graph if enabled
        if kg_enabled:
            import networkx as nx
            self.kg = nx.DiGraph()
    
    # ... remaining HybridRetriever methods ...

def initialize_vector_db():
    """Initialize the vector database for document storage."""
    try:
        # Create the directory if it doesn't exist
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Import necessary libraries
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("Required packages not found. Please install dependencies:")
            logger.error("pip install chromadb sentence-transformers")
            return False
        
        # Initialize the embedding model
        try:
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
        
        # Initialize the ChromaDB client
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            
            # Create a collection (or get existing one)
            collection = client.get_or_create_collection(
                name="tax_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Vector database initialized at {CHROMA_DB_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            return False
    except Exception as e:
        logger.error(f"Error in vector database initialization: {e}")
        return False

def generate_answers(doc: Document, model: str) -> List[str]:
    """Generate answers for the document using the specified model."""
    try:
        import requests
        
        # Parse scenario and questions
        processor = DocumentProcessor()
        parsed = processor.parse_scenario_and_questions(doc)
        
        answers = []
        scenario = parsed["scenario"]
        questions = parsed["questions"]
        
        logger.info(f"Generating answers for {doc.metadata.get('filename')} with {model}")
        
        # Process each question with the model
        for i, question in enumerate(questions):
            # Prepare prompt with scenario and question
            prompt = f"SCENARIO:\n{scenario}\n\nQUESTION:\n{question}\n\nANSWER:"
            
            try:
                # Make a request to Ollama API
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    answer = response.json().get("response", "No answer generated")
                    answers.append(f"Q{i+1}: {question}\n\nA{i+1}: {answer}\n")
                    logger.info(f"Generated answer {i+1} for {doc.metadata.get('filename')}")
                else:
                    logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                    answers.append(f"Q{i+1}: {question}\n\nA{i+1}: Error generating answer\n")
            except Exception as e:
                logger.error(f"Error generating answer for question {i+1}: {e}")
                answers.append(f"Q{i+1}: {question}\n\nA{i+1}: Error: {str(e)}\n")
        
        return answers
    
    except Exception as e:
        logger.error(f"Error in generate_answers: {e}")
        return [f"Error generating answers: {str(e)}"]

def save_answers(doc: Document, answers: List[str], model: str) -> None:
    """Save the generated answers to a file."""
    try:
        # Create answers directory if it doesn't exist
        ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Prepare filename
        filename = f"{model}_{doc.metadata.get('filename', 'unknown')}_answers.txt"
        answers_file = ANSWERS_DIR / filename
        
        # Write answers to file
        with open(answers_file, "w", encoding="utf-8") as f:
            # Add document title/info
            if "title" in doc.metadata:
                f.write(f"DOCUMENT: {doc.metadata['title']}\n\n")
            
            # Write each answer
            f.write("\n---\n\n".join(answers))
        
        logger.info(f"Saved answers to {answers_file}")
        return answers_file
    
    except Exception as e:
        logger.error(f"Error saving answers: {e}")
        return None

def generate_feedback(doc: Document, answers: List[str], model: str) -> List[str]:
    """Generate feedback for the answers using the specified model."""
    try:
        import requests
        
        feedback = []
        joined_answers = "\n".join(answers)
        
        logger.info(f"Generating feedback for {doc.metadata.get('filename')} with {model}")
        
        # Prepare prompt for feedback
        prompt = (
            f"You are an expert tax advisor reviewing answers to tax questions. "
            f"Review the following answers and provide feedback on their accuracy, "
            f"completeness, and clarity:\n\n{joined_answers}\n\n"
            f"FEEDBACK:"
        )
        
        try:
            # Make a request to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                feedback_text = response.json().get("response", "No feedback generated")
                feedback.append(f"ORIGINAL ANSWERS:\n{joined_answers}\n\nFEEDBACK:\n{feedback_text}")
                logger.info(f"Generated feedback for {doc.metadata.get('filename')}")
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                feedback.append("Error generating feedback")
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            feedback.append(f"Error: {str(e)}")
        
        return feedback
    
    except Exception as e:
        logger.error(f"Error in generate_feedback: {e}")
        return [f"Error generating feedback: {str(e)}"]

def save_feedback(doc: Document, feedback: List[str], model: str) -> None:
    """Save the generated feedback to a file."""
    try:
        # Create feedback directory if it doesn't exist
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        
        # Prepare filename
        filename = f"{model}_{doc.metadata.get('filename', 'unknown')}_with_feedback.txt"
        feedback_file = FEEDBACK_DIR / filename
        
        # Write feedback to file
        with open(feedback_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(feedback))
        
        logger.info(f"Saved feedback to {feedback_file}")
        return feedback_file
    
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return None

def process_documents_sequentially(documents: List[Document], models: List[str]) -> None:
    """Process documents one model at a time and generate feedback sequentially."""
    try:
        # Create necessary directories
        ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR = ROOT_DIR / "data" / "metrics"
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        overall_metrics = {}
        
        # Initialize vector database
        vector_db_manager = VectorDatabaseManager()
        vector_db_manager.initialize()
        
        # Process each model one at a time
        for model in models:
            logger.info(f"Processing documents with model: {model}")
            model_metrics = {"processed": 0, "errors": 0, "total_time": 0.0}
            start_model = time.time()
            for doc in documents:
                start_doc = time.time()
                try:
                    logger.info(f"Processing document: {doc.metadata.get('filename', 'unknown')} with model: {model}")
                    
                    # Generate answers
                    answers = generate_answers(doc, model)
                    
                    # Save answers
                    save_answers(doc, answers, model)
                    
                    # Generate feedback
                    feedback = generate_feedback(doc, answers, model)
                    
                    # Save feedback
                    save_feedback(doc, feedback, model)
                    
                    model_metrics["processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('filename')}: {e}")
                    model_metrics["errors"] += 1
                model_metrics["total_time"] += time.time() - start_doc
            model_metrics["average_time_per_doc"] = (model_metrics["total_time"] / model_metrics["processed"]) if model_metrics["processed"] > 0 else 0
            overall_metrics[model] = model_metrics
            logger.info(f"Completed processing with model {model}: {model_metrics}")
        
        metrics_file = METRICS_DIR / "model_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(overall_metrics, mf, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")
        
    except Exception as e:
        logger.error(f"Error in sequential processing: {e}")

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(description="RAG Module for IRS Tax Analysis System")
    parser.add_argument('--init', action='store_true', help='Initialize vector database')
    parser.add_argument('--query', type=str, help='Query the vector database')
    parser.add_argument('--add', type=str, help='Add document to vector database')
    parser.add_argument('--reset', action='store_true', help='Reset the vector database')
    parser.add_argument('--process', action='store_true', help='Process documents sequentially')
    parser.add_argument('--models', nargs='+', default=["llama3:8b"], help='Models to use for processing')
    
    args = parser.parse_args()
    
    if args.init:
        if initialize_vector_db():
            logger.info("Vector database initialization successful")
        else:
            logger.error("Vector database initialization failed")
            sys.exit(1)
    
    if args.reset:
        try:
            if CHROMA_DB_PATH.exists():
                shutil.rmtree(CHROMA_DB_PATH)
                logger.info(f"Removed existing database at {CHROMA_DB_PATH}")
            initialize_vector_db()
        except Exception as e:
            logger.error(f"Error resetting vector database: {e}")
            sys.exit(1)
    
    if args.process:
        try:
            # Load documents
            processor = DocumentProcessor()
            documents = processor.load_text_files()
            
            if not documents:
                logger.error("No documents found to process")
                sys.exit(1)
                
            logger.info(f"Loaded {len(documents)} documents for processing")
            
            # Define models to use (from command line or default)
            models = args.models if args.models else ["llama3:8b", "phi4", "mixtral:8x7b"]  # Changed from phi4:medium to phi4
            
            logger.info(f"Will process with models: {', '.join(models)}")
            
            # Process documents sequentially
            process_documents_sequentially(documents, models)
            
            logger.info("Sequential processing completed successfully")
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # When run directly, execute main
    if "--unittest" in sys.argv:
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        main()

# Unit tests
class TestDocumentProcessor(unittest.TestCase):
    def test_parse_scenario_and_questions(self):
        # Create a test document
        doc = Document(
            content="Advanced Scenario 1: Tax Deductions\nThis scenario involves deductions.\n\nQuestion 1\na) Option 1\nb) Option 2\n\nQuestion 2\na) Option A\nb) Option B",
            metadata={"source": "test.txt", "filename": "test.txt"}
        )
        
        # Create processor
        processor = DocumentProcessor()
        
        # Parse document
        result = processor.parse_scenario_and_questions(doc)
        
        # Check results
        self.assertEqual(result["title"], "Advanced Scenario 1: Tax Deductions")
        self.assertTrue("This scenario involves deductions" in result["scenario"])
        self.assertEqual(len(result["questions"]), 2)
        self.assertTrue("Option 1" in result["questions"][0])
        self.assertTrue("Option A" in result["questions"][1])

if __name__ == "__main__":
    # Run tests when file is executed directly
    unittest.main()