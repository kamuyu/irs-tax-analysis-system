#!/usr/bin/env python3
"""
Bulk processing application for the IRS Tax Analysis System.
Handles processing of all documents in the docs directory.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.rag import process_documents_sequentially, DocumentProcessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bulk')

def main():
    """Main entry point for the bulk processing application."""
    parser = argparse.ArgumentParser(description="Bulk process documents with IRS Tax Analysis System")
    parser.add_argument('--model', '-m', type=str, help='Single model to use for processing')
    parser.add_argument('--input', '-i', type=str, help='Input file or directory')
    parser.add_argument('--output', '-o', type=str, help='Output directory for results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce verbosity')
    parser.add_argument('--optimize', '-O', action='store_true', help='Apply hardware optimization')
    parser.add_argument('--feedback', '-f', action='store_true', default=True, help='Enable feedback generation')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load documents
    docs_dir = args.input if args.input else None
    processor = DocumentProcessor(docs_dir)
    documents = processor.load_text_files()
    
    if not documents:
        logger.error("No documents found to process")
        sys.exit(1)
    
    logger.info(f"Loaded {len(documents)} documents for processing")
    
    # Process with specified model or use all default models
    if args.model:
        models = [args.model]
        logger.info(f"Processing with specified model: {args.model}")
    else:
        # Default to using all three models
        models = ["llama3:8b", "phi4", "mixtral:8x7b"]
        logger.info(f"Processing with default models: {', '.join(models)}")
    
    # Process documents sequentially with the selected models
    process_documents_sequentially(documents, models)
    
    logger.info("Bulk processing completed successfully")

if __name__ == "__main__":
    main()