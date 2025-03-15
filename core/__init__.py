#<!-- filepath: /root/IRS/core/__init__.py -->
# Core functionality for IRS Tax Analysis System
# This package contains the main logic for the IRS system

import logging
import os

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format=log_format,
    handlers=[logging.StreamHandler()]
)

# Check for and handle directory/file structure inconsistencies
# The system uses individual files, not directories, for core components
import sys
from pathlib import Path

# Import major components for easy access
try:
    # Import from the individual files
    from .rag import DocumentProcessor, VectorDatabaseManager, HybridRetriever
    from .analysis import TaxAnalyzer, FeedbackAnalyzer
    from .models import ModelManager
    from .knowledge_graph import TaxKnowledgeGraph, TaxEntity
except ImportError as e:
    logging.warning(f"Could not import core components: {e}")