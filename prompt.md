# Title: IRS Tax Analysis System

## Introduction

The project is a solution to answer questions in text files in the ./docs folder using a hybrid approach combining RAG (Retrieval Augmented Generation) through chroma_db vector database in ./chroma_db folder and a lightweight knowledge graph. The answers should be from 3 models automatically selected from those implemented in Ollama:

* `llama3:8b`: Good reasoning capability and fast (default)
* `phi4`: Good reasoning capability and fast
* `mixtral:8x7b`: Great reasoning capability with medium speed
* `yi:34b`: Excellent reasoning capability but slower

## Architecture

The solution runs in a docker container running in WSL, with GPU acceleration when available, utilizing 32GB of memory allocation. All components run in the venv environment, with dependencies documented in the requirements.txt file.

The solution follows a modular design with minimal, specialized files following best practices. All code is thoroughly tested and documented for maintainability.

## Solution Design

The system processes tax*related questions using these primary components:

1. **Hybrid Retrieval System**: Combines vector search (RAG) with a knowledge graph for superior handling of relational data and tables
2. **Automatic Model Selection**: Automatically selects 3 models at different performance tiers (good, great, excellent)
3. **Bulk Processing**: Automatically processes all text files in the docs directory without requiring specific file input
4. **Sequential Processing**: Processes one model at a time to prevent memory issues and Docker crashes
5. **Progressive Save**: Each answer is written as soon as it's completed, with naming convention [model_name]_[scenario_file]
6. **Feedback Mechanism**: Models review their own answers and others' answers, generating a separate feedback file named [model_name]_[scenario_file]_with_feedback.txt

## Metrics and Monitoring

The system includes comprehensive metrics collection and visualization via:

1. Built*in metrics dashboard using Streamlit
2. Prometheus integration for time*series metrics
3. Grafana dashboards for visualization
4. Performance tracking for model execution, memory usage, and query performance

## Steps

1. **Set up project structure and version control**
   * Create the main project directory structure as outlined in DOCUMENTATION.md
   * Initialize Git repository with milestone/version tracking
   * Create a virtual environment for Python dependencies
   * Set up Docker configuration with appropriate GPU and memory settings

2. **Create core utility scripts**
   * Create `utils/system.py` for system*related utilities (hardware detection, path management)
   * Create `utils/memory.py` for memory management and GPU optimization
   * Create `utils/setup/__init__.py` for environment setup functions
   * Create `utils/debug.py` for troubleshooting and diagnostics

3. **Implement hybrid RAG and Knowledge Graph components**
   * Create `core/rag.py` for vector database interactions
   * Create `core/knowledge_graph.py` for entity and relationship modeling
   * Implement document embedding functionality
   * Set up Chroma DB connection and specialized table handling
   * Create utilities for entity extraction and relationship mapping

4. **Create document parser with table handling**
   * Implement document loading from the docs folder
   * Create functions to parse scenarios and questions
   * Implement specialized table extraction and formatting
   * Create chunk optimization to keep scenarios and questions intact

5. **Implement model integration with GPU optimization**
   * Create connection to Ollama API with robustness features
   * Implement automatic model selection based on performance tiers
   * Create GPU memory optimization for different model sizes
   * Implement memory cleanup between model executions

6. **Build answer generation pipeline**
   * Implement function to process documents with hybrid retrieval
   * Create answer formatting logic for consistent output
   * Implement progressive saving of answers with appropriate naming
   * Add streaming support for real*time observation

7. **Develop feedback mechanism**
   * Create logic for models to review their own answers
   * Implement comparison between different model answers
   * Build feedback generation and correction system
   * Include reasoning analysis in feedback

8. **Create main CLI interface**
   * Create `irs.sh` as the main entry point
   * Implement command line argument parsing
   * Add help and documentation options
   * Create diagnostic and system information commands

9. **Implement sequential processing for stability**
   * Create `apps/bulk/__init__.py` for bulk processing capability
   * Implement automatic processing of all files in docs directory
   * Add progress tracking and reporting
   * Ensure one model processes at a time to prevent crashes

10. **Create Streamlit web interface with real*time analysis**
    * Set up Streamlit interface in `apps/streamlit/`
    * Create automatic model selection interface
    * Implement document upload and results display
    * Add streaming output to show model thinking in real time

11. **Add comprehensive testing, error handling and logging**
    * Implement unit tests for all components
    * Create integration tests across components
    * Add end*to*end test cases
    * Implement comprehensive error handling
    * Add logging system for debugging
    * Create recovery mechanisms for process interruptions

12. **Implement metrics collection and visualization**
    * Create metrics collection system in `utils/metrics.py`
    * Set up Prometheus integration
    * Create Grafana dashboards for visualization
    * Build a Streamlit metrics dashboard
    * Add performance optimization recommendations based on metrics

13. **Create Docker deployment**
    * Create Docker and docker*compose configurations
    * Add Ollama, Prometheus, and Grafana containers
    * Configure volumes and networking
    * Set resource limits and GPU passthrough

14. **Write comprehensive documentation**
    * Complete README.md with clear setup and usage instructions
    * Create detailed DOCUMENTATION.md with architecture details
    * Add architecture diagrams and decision documentation
    * Include troubleshooting guides and examples
    * Document all API endpoints and parameters

15. **Final testing and optimization**
    * Test with various document types and sizes
    * Optimize memory usage for different hardware configurations
    * Benchmark different models and configurations
    * Ensure all features work in Docker environment
    * Validate metrics collection and visualization

## GitHub CLI Authentication and Pushing Changes

### Authenticating with GitHub CLI

1. **Install GitHub CLI** (if not already installed):

   ```bash
   gh **version
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

   ! First copy your one*time code: C0D4*AEFB
   Press Enter to open github.com in your browser...
   ✓ Authentication complete.
   * gh config set *h github.com git_protocol https
   ✓ Configured git protocol
   ! Authentication credentials saved in plain text
   ✓ Logged in as kamuyu

### Setting Up Global Git Configuration

**Important: Set up your global git configuration to ensure your commits are properly attributed.**

```bash
git config **global user.email "fkamuyu@gmail.com"
git config **global user.name "Francis Kamuyu"
```

### Pushing Changes to GitHub

1. **Add the changes to the staging area**:

   ```bash
   git add .
   ```

2. **Commit the changes with a meaningful commit message**:

   ```bash
   git commit *m "Your commit message here"
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
