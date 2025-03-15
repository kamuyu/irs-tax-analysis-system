#<!-- filepath: /root/IRS/apps/bulk/run.py -->
#!/usr/bin/env python3
# Bulk processing application for IRS Tax Analysis System

import os
import sys
import logging
import time
import argparse
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import concurrent.futures

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.models import ModelManager
from core.rag import DocumentProcessor, VectorDatabaseManager, HybridRetriever
from core.analysis import TaxAnalyzer, FeedbackAnalyzer
from utils.memory import MemoryOptimizer, memory_usage_decorator
from utils.system import get_system_info, get_optimal_worker_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("bulk_processor")

class ModelSelector:
    """Select appropriate models based on system capabilities and desired performance tiers."""
    
    TIERS = {
        "good": ["llama3:8b", "phi4:medium", "mistral:v0.3"],
        "great": ["mixtral:8x7b"],
        "excellent": ["yi:34b", "llama3:70b"]
    }
    
    @staticmethod
    def select_models_by_tier(model_manager: ModelManager, 
                             include_tiers: List[str] = ["good", "great", "excellent"]) -> List[str]:
        """Select one model from each specified performance tier.
        
        Args:
            model_manager: ModelManager instance
            include_tiers: List of tiers to include
            
        Returns:
            List of selected model names
        """
        # Get system info
        sys_info = get_system_info()
        available_ram = sys_info.get("available_memory_gb", 8)
        has_gpu = len(sys_info.get("gpu_info", [])) > 0
        
        # Get available models
        available_models = model_manager.get_available_models()
        if not available_models:
            logger.warning("No models found. Attempting to pull default model.")
            model_manager.pull_model("llama3:8b")
            available_models = model_manager.get_available_models()
        
        # Select models based on tiers and system capabilities
        selected_models = []
        
        for tier in include_tiers:
            tier_models = ModelSelector.TIERS.get(tier, [])
            
            # Filter by available models
            tier_models = [m for m in tier_models if m in available_models]
            
            if tier_models:
                # For "excellent" tier, only select if we have GPU and enough RAM
                if tier == "excellent" and (not has_gpu or available_ram < 24):
                    logger.info("Skipping 'excellent' tier due to hardware limitations")
                    continue
                
                # For "great" tier, only select if we have enough RAM
                if (tier == "great" and available_ram < 16):
                    logger.info("Skipping 'great' tier due to limited RAM")
                    continue
                
                # Select first available model from tier
                selected_models.append(tier_models[0])
        
        # Ensure we have at least one model
        if not selected_models and available_models:
            logger.info("Falling back to first available model")
            selected_models = [available_models[0]]
        
        logger.info(f"Selected models: {', '.join(selected_models)}")
        return selected_models

class BulkProcessor:
    """Process all tax documents in docs directory with automatically selected models."""
    
    def __init__(self, 
                 docs_dir: str = "./data/docs",
                 output_dir: Optional[str] = None,
                 feedback_enabled: bool = True,
                 parallel_docs: Optional[int] = None):
        """Initialize the bulk processor.
        
        Args:
            docs_dir: Directory containing documents to process
            output_dir: Directory for output files (defaults to same as docs_dir)
            feedback_enabled: Whether to generate feedback
            parallel_docs: Number of parallel document processors
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir) if output_dir else self.docs_dir
        self.feedback_enabled = feedback_enabled
        self.parallel_docs = parallel_docs or get_optimal_worker_count()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.doc_processor = DocumentProcessor(str(self.docs_dir))
        self.vector_db = VectorDatabaseManager()
        self.retriever = HybridRetriever(self.vector_db)
        self.analyzer = TaxAnalyzer(self.model_manager, self.retriever)
        self.feedback_analyzer = FeedbackAnalyzer(self.model_manager)
    
    def process_all_documents(self) -> Dict[str, Any]:
        """Process all documents in the docs directory with appropriate models.
        
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Check Ollama connectivity
        if not self.model_manager.check_connectivity():
            logger.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            return {"error": "Ollama not accessible"}
        
        # Select models based on tiers
        selected_models = ModelSelector.select_models_by_tier(self.model_manager)
        
        if not selected_models:
            logger.error("No suitable models found. Please install models with Ollama.")
            return {"error": "No models available"}
        
        # Find all text files in the docs directory
        text_files = list(self.docs_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {self.docs_dir}")
            return {"error": "No text files found"}
        
        logger.info(f"Found {len(text_files)} text files to process")
        
        # Initialize vector database
        logger.info("Initializing vector database")
        self.vector_db.initialize()
        
        # Process each file with each selected model
        results = {}
        
        for file_path in text_files:
            file_results = self._process_file(file_path, selected_models)
            results[file_path.name] = file_results
        
        total_time = time.time() - start_time
        logger.info(f"Completed processing {len(text_files)} files with {len(selected_models)} models "
                    f"in {total_time:.2f} seconds")
        
        return {
            "files_processed": len(text_files),
            "models_used": selected_models,
            "total_time": total_time,
            "results": results
        }
    
    def _process_file(self, file_path: Path, models: List[str]) -> Dict[str, Any]:
        """Process a single file with multiple models.
        
        Args:
            file_path: Path to the file
            models: List of model names
            
        Returns:
            Dictionary with results for each model
        """
        logger.info(f"Processing file: {file_path}")
        
        # Load document
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}
        
        # Create document
        from core.rag import Document
        doc = Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "type": "text"
            }
        )
        
        # Parse scenario and questions
        doc_info = self.doc_processor.parse_scenario_and_questions(doc)
        logger.info(f"Parsed document with title: {doc_info['title']}")
        logger.info(f"Found {len(doc_info['questions'])} questions")
        
        # Add document to vector database for retrieval
        self.vector_db.add_documents([doc])
        
        # Process with each model
        model_results = {}
        
        for model_name in models:
            logger.info(f"Processing {file_path.name} with model {model_name}")
            
            try:
                # Analyze scenario
                analysis = self.analyzer.analyze_scenario(
                    doc_info,
                    model_name,
                    str(self.output_dir)
                )
                
                output_file = self.output_dir / f"{model_name}_{file_path.name}"
                
                # Generate feedback if enabled
                if self.feedback_enabled:
                    feedback = self.feedback_analyzer.generate_feedback(analysis)
                    feedback_path = self.output_dir / f"{model_name}_{file_path.name}_with_feedback.txt"
                    
                    # Save feedback
                    with open(feedback_path, 'w') as f:
                        f.write(feedback)
                    
                    logger.info(f"Saved feedback to {feedback_path}")
                
                # Store result
                model_results[model_name] = {
                    "success": True,
                    "output_file": str(output_file),
                    "feedback_file": str(feedback_path) if self.feedback_enabled else None
                }
                
                # Clean up after processing
                MemoryOptimizer.clean(model_name)
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name} with {model_name}: {e}")
                model_results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return model_results

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="IRS Tax Analysis System - Bulk Processor")
    parser.add_argument("--docs-dir", default="./data/docs", help="Directory containing documents to process")
    parser.add_argument("--output-dir", help="Output directory (defaults to same as docs-dir)")
    parser.add_argument("--no-feedback", action="store_true", help="Disable feedback generation")
    parser.add_argument("--parallel", type=int, help="Number of parallel document processors")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Process all documents
    processor = BulkProcessor(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        feedback_enabled=not args.no_feedback,
        parallel_docs=args.parallel
    )
    
    try:
        results = processor.process_all_documents()
        
        if "error" in results:
            logger.error(f"Bulk processing failed: {results['error']}")
            return 1
        
        logger.info(f"Successfully processed {results['files_processed']} files with {len(results['models_used'])} models")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error during bulk processing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())