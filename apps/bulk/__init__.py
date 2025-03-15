#<!-- filepath: /root/IRS/apps/bulk/__init__.py -->
# Bulk processing module for IRS Tax Analysis System

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import core modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.models import ModelManager
from core.rag import DocumentProcessor, VectorDatabaseManager, HybridRetriever
from core.analysis import TaxAnalyzer, FeedbackAnalyzer
from utils.memory import MemoryOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("bulk_processor")