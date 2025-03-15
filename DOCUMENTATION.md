# IRS Tax Analysis System

## Overview

The IRS Tax Analysis System provides tools for processing and analyzing tax-related questions using a hybrid approach of Retrieval Augmented Generation (RAG) and Knowledge Graphs with Large Language Models.

## Step-by-Step Setup Guide

### 1. Prerequisites

- Linux-based system (Ubuntu/Debian recommended) or WSL on Windows
- Python 3.8+ installed
- Minimum 8GB RAM (32GB recommended for optimal performance)
- CUDA-compatible GPU (optional but recommended)
- Ollama installed [here](https://ollama.com/download)
- GitHub CLI installed [here](https://cli.github.com/manual/installation)

### 2. Environment Setup

```bash
# Clone the repository (if using version control)
git clone https://github.com/yourusername/irs-tax-analysis.git
cd irs-tax-analysis

# Run the setup script (creates venv and installs dependencies)
bash ./irs.sh setup
```

This setup script:
- Creates a Python virtual environment in `/root/IRS/venv`
- Installs all required dependencies from requirements.txt
- Pulls the necessary Ollama models
- Creates required directories

### 3. Activating the Environment

The `irs.sh` script automatically activates the virtual environment, but if you need to manually activate it:

```bash
# Activate the virtual environment
source /root/IRS/venv/bin/activate

# When finished, deactivate with
deactivate
```

### 4. Preparing Question Files

Place your tax question files in the `/root/IRS/data/docs` directory with the `.txt` format.

Question files should follow this format:

```
Advanced Scenario X: 
[Scenario description text]

[Question 1]
a) Option 1
b) Option 2
c) Option 3
d) Option 4

[Question 2]
...
```

Questions are separated by blank lines. The first section is considered the scenario, and subsequent sections are individual questions.

## Architecture

The system follows a modular architecture with the following key components:

1. **Core Processing Engine**:
   - Hybrid retrieval combining vector search and knowledge graph
   - Sequential multi-model analysis for stability
   - Progressive answer generation with feedback mechanism

2. **Application Interfaces**:
   - Bulk processing for batch analysis
   - Streamlit web interface for interactive use
   - Metrics dashboard for monitoring

3. **Infrastructure**:
   - Docker containerization for deployment
   - Prometheus for metrics collection
   - Grafana for visualization

### Hybrid Retrieval Approach

The system uses a sophisticated hybrid approach that combines:

1. **Vector-based RAG (Retrieval Augmented Generation)**:
   - Document chunking with overlap optimization
   - Embedding generation using appropriate models
   - Vector search through ChromaDB for semantic similarity
   - Context window optimization for different models

2. **Knowledge Graph Integration**:
   - Automatic entity extraction from tax documents
   - Relationship mapping based on tax regulations
   - Graph-based querying for structured information
   - Table data preservation and relationship modeling

This hybrid approach significantly improves the system's ability to handle complex tax scenarios, especially those involving relational data, tables, and multi-step reasoning.

### Automatic Model Selection

The system sequentially processes documents with models from different performance tiers:

1. **Good tier** (default): `llama3:8b` or `phi4`
   - Fast response time
   - Lower memory requirements
   - Suitable for simpler questions

2. **Great tier**: `mixtral:8x7b`
   - Medium response time
   - Moderate memory requirements
   - Strong reasoning capabilities

3. **Excellent tier**: `yi:34b`
   - Slower response time
   - Higher memory requirements
   - Advanced reasoning for complex questions

Models are processed one at a time to prevent memory issues and container crashes.

### Directory Structure

```
/root/IRS/
├── apps/                    # Application interfaces
│   ├── bulk/                # Bulk analysis application
│   ├── metrics/             # Metrics collection and visualization
│   └── streamlit/           # Streamlit web interface
├── core/                    # Core functionality
│   ├── analysis/            # Tax analysis logic
│   ├── knowledge_graph.py   # Knowledge graph implementation
│   ├── models.py            # Model integration with Ollama
│   └── rag.py               # Retrieval Augmented Generation components
├── data/                    # Data storage
│   ├── chroma_db/           # Vector database storage
│   ├── docs/                # Question and answer documents
│   ├── models/              # Model files and embeddings
│   ├── answers/             # Generated answers
│   └── feedback/            # Generated feedback with self-reviews
├── config/                  # Configuration files
│   ├── docker/              # Docker configuration files
│   ├── models.yaml          # Model configuration
│   └── system.yaml          # System configuration
├── utils/                   # Utility modules
│   ├── debug.py             # Debugging utilities
│   ├── memory.py            # Memory management tools
│   ├── metrics.py           # Metrics collection
│   ├── setup/               # Setup utilities
│   └── system.py            # System utilities
├── tests/                   # Test suite
├── docker-compose.yml       # Docker deployment configuration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── irs.sh                   # Main entry point script
├── README.md                # Project overview
├── DOCUMENTATION.md         # Detailed documentation
└── prompt.md                # Project requirements
```

## Installation

### Prerequisites

- Linux-based system (Ubuntu/Debian recommended) or WSL
- Python 3.8+ installed
- Minimum 8GB RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU (optional but recommended)
- GitHub CLI installed [here](https://cli.github.com/manual/installation)

### Quick Setup

1. **Clone or set up the repository**:

   If you're setting up from scratch, ensure all the files are in the `/root/IRS` directory.

2. **Run the setup script**:

   ```bash
   ./irs.sh setup
   ```

   This will:
   - Install required system dependencies
   - Create a Python virtual environment
   - Install required Python packages
   - Download and set up the default Ollama model (llama3:8b)

3. **Set up with a specific model**:

   ```bash
   ./irs.sh setup mixtral:8x7b
   ```

## Usage

### Unified Command Interface

All functionality is accessible through the main `irs.sh` script:

```bash
./irs.sh [command] [options]
```

Available commands:

| Command | Description |
|---------|-------------|
| `bulk` | Run bulk analysis of tax questions |
| `web` or `streamlit` | Launch Streamlit web interface |
| `metrics` | Launch metrics dashboard |
| `setup` or `install` | Set up environment and dependencies |
| `clean` or `cleanup` | Run memory cleanup |
| `sysinfo` | Show system information |
| `process` | Process documents sequentially with all models |
| `help` | Show help information |

### Bulk Analysis

To process tax questions in batch mode using the default models sequentially, use:

```bash
./irs.sh bulk --input /root/IRS/data/docs/AdvancedQuestions\ 1.txt
```

The bulk command ignores any positional model argument and always cycles through the default models: llama3:8b, phi4, and mixtral:8x7b.

Options:
- `--input/-i`: Input questions file or directory
- `--output/-o`: Output answers file
- `--model/-m`: Ollama model name (alternative to positional argument)
- `--quiet/-q`: Reduce verbosity
- `--max-length`: Maximum text length to process at once
- `--optimize/-O`: Apply automatic hardware optimization
- `--feedback/-f`: Enable feedback generation (default: enabled)
- `--parallel/-p N`: Number of parallel processes (default: auto)

Example:
```bash
./irs.sh bulk mixtral:8x7b --input /root/IRS/data/docs/AdvancedQuestions\ 1.txt
```

### Web Interface

Launch the Streamlit web interface:

```bash
./irs.sh web
```

The interface will be available at http://localhost:8501 by default.

### Metrics Dashboard

Launch the metrics visualization dashboard:

```bash
./irs.sh metrics
```

This provides real-time monitoring of:
- Model performance statistics
- Memory usage and optimization
- Processing time per question and model
- System resource utilization

## Available Models

The following models are supported with their characteristics:

| Model | Reasoning Capability | RAM Required | Speed |
|-------|---------------------|-------------|-------|
| llama3:8b | High | 8GB | Fast |
| phi4 | High | 8GB | Fast |
| mistral:v0.3 | Good | 8GB | Fast |
| mixtral:8x7b | Very high | 16GB | Medium |
| yi:34b | Very high | 24GB | Slow |
| llama3:70b | Excellent | 32GB | Very slow |

## Parallel Processing

The system optimizes performance through parallel processing:

1. **Document Processing**:
   - Multi-threaded document loading and parsing
   - Chunking and embedding generation in parallel
   - Vector database batch operations

2. **Model Execution**:
   - Efficient GPU memory management across models
   - CPU core utilization for preprocessing
   - Automatic batch size optimization

3. **Performance Tuning**:
   - Automatic resource allocation based on hardware
   - Memory monitoring and cleanup between operations
   - Progressive saving to prevent data loss

## Feedback Mechanism

The system includes a sophisticated feedback mechanism:

1. **Self-review**: Models review their own answers for accuracy and completeness
2. **Cross-model review**: Models analyze other models' answers
3. **Reasoning analysis**: Models explain their reasoning process
4. **Correction proposals**: Models suggest improvements to their own answers
5. **Comparison metrics**: Performance comparison across different models

## File Formats

### Question Files

Place question files in the `/root/IRS/data/docs/` directory with the naming convention `AdvancedQuestions N.txt` (where N is a number).

Format:
```
Advanced Scenario X: 
[Scenario description text]

[Question 1]
a) Option 1
b) Option 2
c) Option 3
d) Option 4

[Question 2]
...
```

Questions are separated by blank lines. The first section is considered the scenario, and subsequent sections are individual questions.

### Answer Files

Answers are generated in the format:

```
SCENARIO:
[Original scenario text]

Q1: [Question text]

A1: [Answer text]

---

Q2: [Question text]

A2: [Answer text]

---
```

### Feedback Files

After generating answers, the system creates feedback files with the naming convention `[model_name]_[scenario_name]_with_feedback.txt`. These files contain:

```bash
ORIGINAL ANSWERS:
[Original answers text]

FEEDBACK:
[Model's feedback on its own answers]

CORRECTIONS:
[Corrected answers based on review]

OTHER MODELS:
[Comparison and review of other models' answers]
```

## Metrics and Monitoring

The system includes comprehensive metrics collection and visualization:

1. **Built-in Streamlit Dashboard**:
   - Real-time performance monitoring
   - Model comparison charts
   - Resource utilization graphs
   - Question complexity analysis

2. **Prometheus Integration**:
   - Time-series metrics collection
   - System and application health monitoring
   - Alert configuration for performance issues
   - Custom exporters for model metrics

3. **Grafana Dashboards**:
   - Visualization of performance data
   - Historical trend analysis
   - Resource utilization insights
   - Model comparison visualizations

4. **Performance Tracking**:
   - Model execution time monitoring
   - Memory usage optimization
   - Query performance statistics
   - Hardware utilization efficiency

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

1. Run memory cleanup:

   ```bash
   ./irs.sh clean
   ```

2. Use a smaller model with sequential processing:

   ```bash
   ./irs.sh process --models llama3:8b
   ```

3. Process one model at a time:

   ```bash
   ./irs.sh process --models phi4
   ```

### Import Errors

If you encounter module import errors:

1. Ensure the setup has completed successfully:

   ```bash
   ./irs.sh setup
   ```

2. Check Python environment:

   ```bash
   /root/IRS/venv/bin/python -c "import sys; print(sys.path)"
   ```

3. Verify module structure:

   ```bash
   ls -la /root/IRS/core/rag/
   ls -la /root/IRS/core/analysis/
   ```

### Ollama Connectivity

If you encounter issues connecting to Ollama:

1. Check if Ollama is running:

   ```bash
   curl http://localhost:11434/api/version
   ```

2. Restart Ollama:

   ```bash
   systemctl restart ollama
   # or
   ollama serve
   ```

3. **Common Setup Error**: If you see `Error: could not connect to ollama app, is it running?`:

   ```bash
   # Check if Ollama is installed
   which ollama
   
   # Start Ollama if it's installed but not running
   ollama serve &
   
   # Wait a moment for Ollama to initialize
   sleep 5
   
   # Try pulling a model again
   ./irs.sh setup --retry
   ```

4. If Ollama is not installed:

   ```bash
   # Install Ollama (Linux)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve &
   ```

5. Try the diagnostics utility:

   ```bash
   # Run the Ollama diagnostics tool
   ./irs.sh diagnose ollama
   ```

## Advanced Configuration

### Custom Models

To use custom models:

1. Pull the model using Ollama:

   ```bash
   ollama pull [model_name]
   ```

2. Run with the custom model:

   ```bash
   ./irs.sh bulk [model_name]
   ```

### Hardware Optimization

The system automatically optimizes for your hardware. To manually optimize:

1. Edit `/root/IRS/core/rag/models.py` to adjust parameters
2. Use the `--optimize` flag when running:
   ```bash
   ./irs.sh bulk --optimize
   ```

## Extending the System

### Adding New Applications

To add a new application interface:

1. Create a new directory under `/root/IRS/apps/`
2. Implement the application using the core modules
3. Update the `irs.sh` script to include the new application

### Adding Custom Analysis Logic

To add custom analysis logic:

1. Add new modules under `/root/IRS/core/analysis/`
2. Update the imports in `/root/IRS/core/__init__.py`

### Customizing the Feedback Mechanism

The feedback system can be customized by:

1. Editing the feedback prompts in `/root/IRS/core/analysis/feedback.py`
2. Adjusting the feedback generation strategy
3. Modifying the comparison logic between models

### Docker Integration

The system can be deployed using Docker:

1. Configure through `docker-compose.yml`
2. Adjust resource limits in the Docker configuration
3. Enable GPU passthrough for optimal performance
4. Scale services based on workload requirements

## GitHub CLI Authentication and Pushing Changes

### Authenticating with GitHub CLI

1. **Install GitHub CLI** (if not already installed):
   ```bash
   gh --version
   ```

   If it's not installed, you can install it by following the instructions [here](https://cli.github.com/manual/installation).

2. **Authenticate GitHub CLI**:
   ```bash
   gh auth login
   ```

   Follow the prompts to authenticate with your GitHub account:
   ```
   ? What account do you want to log into? GitHub.com
   ? What is your preferred protocol for Git operations on this host? HTTPS
   ? Authenticate Git with your GitHub credentials? Yes
   ? How would you like to authenticate GitHub CLI? Login with a web browser

   ! First copy your one-time code: C0D4-AEFB
   Press Enter to open github.com in your browser...
   ✓ Authentication complete.
   - gh config set -h github.com git_protocol https
   ✓ Configured git protocol
   ! Authentication credentials saved in plain text
   ✓ Logged in as kamuyu
   ```

### Pushing Changes to GitHub

1. **Add the changes to the staging area**:
   ```bash
   git add .
   ```

2. **Commit the changes with a meaningful commit message**:
   ```bash
   git commit -m "Your commit message here"
   ```

3. **Push the changes to the remote repository**:
   ```bash
   git push
   ```

### Using the `commit_all.sh` Script

You can also use the `commit_all.sh` script to commit and push changes with a single command. Here is how you can do it:

1. **Run the `commit_all.sh` script with a commit message**:
   ```bash
   bash /root/IRS/scripts/commit_all.sh "Fixed bug in data processing script"
   ```

This script will add all changes, commit them with the provided message, and push them to the remote repository if it is configured.
