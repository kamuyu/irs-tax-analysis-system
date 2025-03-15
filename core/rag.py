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

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag')

# Define paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
CHROMA_DB_PATH = ROOT_DIR / "data" / "chroma_db"

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
            logger.info(f"Embedding model loaded: {model}")
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

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(description="RAG Module for IRS Tax Analysis System")
    parser.add_argument('--init', action='store_true', help='Initialize vector database')
    parser.add_argument('--query', type=str, help='Query the vector database')
    parser.add_argument('--add', type=str, help='Add document to vector database')
    parser.add_argument('--reset', action='store_true', help='Reset the vector database')
    
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
    
    # Other operations would be implemented here

if __name__ == "__main__":
    main()

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
    
    def __init__(self, docs_dir: str = "./data/docs"):
        """Initialize with directory containing documents"""
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
    
    def process_all_documents(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Process all documents in the docs directory"""
        documents = self.load_text_files()
        processed_docs = []
        
        for doc in documents:
            processed = self.parse_scenario_and_questions(doc)
            processed_docs.append(processed)
            
        return processed_docs

class VectorDatabaseManager:
    """Class to manage vector database operations"""
    
    def __init__(self, db_dir: str = "./data/chroma_db", embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize vector database manager"""
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
    
    def get_or_create_collection(self, collection_name: str = "irs_docs") -> Any:
        """Get or create a collection in the vector database"""
        if not self.db_client:
            self.initialize()
        
        try:
            collections = self.db_client.list_collections()
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                return self.db_client.get_collection(name=collection_name)
            else:
                logger.info(f"Creating new collection: {collection_name}")
                return self.db_client.create_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.embeddings:
            self.initialize()
        
        try:
            embeddings = self.embeddings.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Document], collection_name: str = "irs_docs", chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Add documents to the vector database with chunking"""
        if not documents:
            logger.warning("No documents to add to vector database")
            return
        
        collection = self.get_or_create_collection(collection_name)
        
        # Process each document
        for doc in documents:
            # Apply chunking if the document is large
            chunks = self._create_chunks(doc.content, chunk_size, chunk_overlap)
            
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.metadata.get('filename', 'doc')}_{i}"
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_count": len(chunks)
                })
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadatas.append(chunk_metadata)
            
            # Generate embeddings
            embeddings = self.embed_text(chunk_texts)
            
            # Add to collection
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Added document '{doc.metadata.get('filename', 'unknown')}' to vector database ({len(chunks)} chunks)")
    
    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create overlapping chunks from text"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = min(start + chunk_size, len(text))
            
            # If we're not at the end, try to break at a paragraph or sentence
            if end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find sentence break
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1
            
            chunks.append(text[start:end])
            start = end - chunk_overlap
        
        return chunks
    
    def query(self, query_text: str, collection_name: str = "irs_docs", n_results: int = 5) -> Dict[str, Any]:
        """Query the vector database"""
        collection = self.get_or_create_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self.embed_text([query_text])[0]
        
        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results

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
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant information using hybrid approach"""
        # Get results from vector database
        rag_results = self.vector_db.query(query, n_results=n_results)
        
        # Prepare combined results
        combined_results = []
        
        # Process RAG results
        for i in range(len(rag_results.get("documents", [[]])[0])):
            result = {
                "text": rag_results["documents"][0][i],
                "metadata": rag_results["metadatas"][0][i],
                "score": 1.0 - min(1.0, rag_results["distances"][0][i]),  # Convert distance to similarity score
                "source": "rag"
            }
            combined_results.append(result)
        
        # Add knowledge graph results if enabled
        if self.kg_enabled and self.kg:
            kg_results = self._retrieve_from_kg(query)
            for result in kg_results:
                combined_results.append(result)
        
        # Sort by score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:n_results]
    
    def _retrieve_from_kg(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve information from knowledge graph"""
        # This is a placeholder for actual KG retrieval logic
        # In a real implementation, this would perform entity extraction and graph traversal
        return []

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