# IRS Tax Analysis System

A comprehensive system for analyzing tax scenarios using Large Language Models with a hybrid approach of Retrieval Augmented Generation (RAG) and Knowledge Graphs.

## Features

- **Hybrid Retrieval System**: Combines vector search with knowledge graph for superior handling of relational data and tables
- **Multi-Model Analysis**: Automatically selects and runs models sequentially at different performance tiers
- **Bulk Processing**: Efficiently processes all documents in the docs directory
- **Sequential Processing**: Processes documents one model at a time to ensure stability
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
  - `answers/`: Generated answers
  - `feedback/`: Generated feedback with self-reviews
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

### Bulk Analysis

Process tax questions in batch mode using the default models sequentially:

```bash
./irs.sh bulk --input /root/IRS/data/docs/AdvancedQuestions\ 1.txt
```

Note: The bulk command ignores any positional model argument and always uses the default models: llama3:8b, phi4, and mixtral:8x7b.

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
| phi4 | Good | 8GB | Fast | Alternative for basic analysis |
| mixtral:8x7b | Great | 16GB | Medium | Complex reasoning tasks |
| yi:34b | Excellent | 24GB | Slow | Highly complex scenarios |

## Additional Commands

| Command | Description |
|---------|-------------|
| `bash ./irs.sh setup` | Set up environment and download models |
| `bash ./irs.sh bulk` | Process all files in data/docs directory |
| `bash ./irs.sh process` | Process documents sequentially with all models |
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

After processing documents (via bulk or process commands), a metrics report is generated in `/root/IRS/data/metrics/model_metrics.json`. The built-in Streamlit dashboard (launched with `./irs.sh metrics`) uses this report to compare the performance of each model (e.g. number of processed documents and processing times).

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

   Follow the prompts to authenticate with your GitHub account
  
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

### Setting Up Global Git Configuration

**Important: Set up your global git configuration to ensure your commits are properly attributed.**

```bash
git config --global user.email "fkamuyu@gmail.com"
git config --global user.name "Francis Kamuyu"
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
