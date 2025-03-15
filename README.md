# IRS Tax Analysis System

A comprehensive system for analyzing tax scenarios using Large Language Models with a hybrid approach of Retrieval Augmented Generation (RAG) and Knowledge Graphs.

## Features

- **Hybrid Retrieval System**: Combines vector search with knowledge graph for superior handling of relational data and tables
- **Multi-Model Analysis**: Automatically selects 3 models at different performance tiers
- **Bulk Processing**: Efficiently processes all documents in the docs directory
- **Parallel Processing**: Optimizes CPU and GPU utilization for maximum performance
- **Progressive Save**: Saves answers as they're completed with consistent naming
- **Feedback Mechanism**: Models review their own and others' answers
- **Comprehensive Metrics**: Includes dashboards for performance monitoring
- **Docker Integration**: Full containerization with GPU support

## Directory Structure

- `apps/`: Applications
  - `bulk/`: Bulk analysis application
  - `streamlit/`: Streamlit web interface
  - `metrics/`: Metrics dashboard and visualization
- `core/`: Core functionality
  - `analysis/`: Tax analysis logic
  - `rag.py`: Retrieval augmented generation tools
  - `knowledge_graph.py`: Knowledge graph implementation
  - `models.py`: Model integration with Ollama
- `utils/`: Utility functions
  - `memory.py`: Memory management and optimization
  - `system.py`: System utilities
  - `metrics.py`: Performance metrics collection
  - `debug.py`: Debugging and troubleshooting tools
  - `setup/`: Setup utilities
- `data/`: Data storage
  - `docs/`: Question and answer documents
  - `chroma_db/`: Vector database
  - `models/`: Model files and embeddings
- `config/`: Configuration files
  - `docker/`: Docker configuration files
  - `models.yaml`: Model configuration
  - `system.yaml`: System configuration
- `tests/`: Test suite

## Quick Start Guide

### 1. Setup Environment

Clone the repository and set up the environment:

```bash
# First-time setup (installs dependencies and pulls default models)
bash ./irs.sh setup
```

### 2. Activate Environment (if needed manually)

The irs.sh script automatically activates the virtual environment, but you can do it manually:

```bash
# Activate the virtual environment
source /root/IRS/venv/bin/activate
```

### 3. Place Question Files

Place your tax question files in the `/root/IRS/data/docs` directory with the `.txt` format.

### 4. Run the Analysis

```bash
# Process all question files in data/docs with automatically selected models
bash ./irs.sh bulk
```

The system will:

- Automatically process all .txt files in the data/docs directory
- Select 3 models at different performance levels (good, great, excellent)
- Generate answers and save them alongside the original files
- Create feedback files with model self-evaluations

### 5. View Results

Answer files will be saved in the same directory as the questions with the naming pattern:

- `[model_name]_[original_filename]` - For answers
- `[model_name]_[original_filename]_with_feedback.txt` - For feedback

### 6. Web Interface (Optional)

```bash
# Launch the interactive web interface
bash ./irs.sh web
```

The web interface provides:

- Document upload capabilities
- Real-time answer generation
- Model selection options
- Performance visualization
- Comparison between models
- Feedback analysis

## Models and Performance

The system supports the following models with automatic selection:

| Model | Tier | RAM Required | Speed | Use Case |
|-------|------|-------------|-------|----------|
| llama3:8b | Good | 8GB | Fast | Default choice for most scenarios |
| phi4:medium | Good | 8GB | Fast | Alternative for basic analysis |
| mixtral:8x7b | Great | 16GB | Medium | Complex reasoning tasks |
| yi:34b | Excellent | 24GB | Slow | Highly complex scenarios |

## Additional Commands

| Command | Description |
|---------|-------------|
| `bash ./irs.sh setup` | Set up environment and download models |
| `bash ./irs.sh bulk` | Process all files in data/docs directory |
| `bash ./irs.sh bulk --parallel 4` | Process with 4 parallel threads |
| `bash ./irs.sh web` | Launch Streamlit web interface |
| `bash ./irs.sh metrics` | Launch metrics dashboard |
| `bash ./irs.sh clean` | Run memory cleanup |
| `bash ./irs.sh sysinfo` | Show system information |
| `bash ./irs.sh help` | Show help information |

## System Requirements

- Python 3.8+
- 8GB+ RAM (32GB recommended for optimal performance)
- CUDA-compatible GPU (optional but recommended)
- Docker and docker-compose (for containerized deployment)

## Metrics and Monitoring

The system includes comprehensive metrics dashboards:

- Real-time model performance tracking
- Memory usage optimization
- Processing time per question and model
- System resource utilization
- Model comparison visualization

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical information, including:

- Architecture details and design decisions
- Advanced configuration options
- Troubleshooting guide
- Customization instructions

## Docker Deployment

For containerized deployment with all components:

```bash
# Build and start all containers
bash docker-compose up -d
```

Access services:

- IRS Streamlit interface: [http://localhost:8501](http://localhost:8501)
- Metrics dashboard: [http://localhost:3000](http://localhost:3000) (default login: admin/admin)
- Prometheus metrics: [http://localhost:9090](http://localhost:9090)
