#!/bin/bash
# Main entry point for IRS Tax Analysis System

# Set the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a function to activate the virtual environment
function activate_venv {
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    elif [ -f "$ROOT_DIR/venv/Scripts/activate" ]; then
        source "$ROOT_DIR/venv/Scripts/activate"
    else
        echo "Error: Virtual environment not found"
        echo "Please run 'bash ./irs.sh setup' first"
        exit 1
    fi
}

# Display help information
function show_help {
    echo "Usage: $0 [command] [options]"
    echo
    echo "Commands:"
    echo "  bulk                   Run bulk analysis of tax questions"
    echo "  web, streamlit         Launch Streamlit web interface"
    echo "  metrics                Launch metrics dashboard"
    echo "  setup, install         Set up environment and dependencies"
    echo "  clean, cleanup         Run memory cleanup"
    echo "  sysinfo                Show system information"
    echo "  process                Process documents sequentially (prevents crashes)"
    echo "  diagnose [component]   Run diagnostics on specific components"
    echo "  help                   Show this help message"
    echo
    echo "Options:"
    echo "  --input, -i FILE       Input file or directory"
    echo "  --output, -o FILE      Output file or directory"
    echo "  --model, -m MODEL      Specify model to use"
    echo "  --models MODELS        Specify models for sequential processing"
    echo "  --quiet, -q            Reduce verbosity"
    echo "  --optimize, -O         Apply hardware optimization"
    echo "  --feedback, -f         Enable feedback generation (default: enabled)"
    echo "  --parallel, -p N       Set number of parallel processes"
    echo "  --retry                Retry operation (for setup/diagnose)"
    echo
    echo "Examples:"
    echo "  $0 setup               Set up the environment"
    echo "  $0 bulk                Run bulk analysis on all documents"
    echo "  $0 process             Process documents sequentially with all models"
    echo "  $0 web                 Start the Streamlit web interface"
    echo "  $0 diagnose ollama     Run diagnostics on Ollama"
}

# Run diagnostics on specific components
function run_diagnostics {
    component=$1
    
    case $component in
        ollama)
            echo "Running Ollama diagnostics..."
            if [ -f "$ROOT_DIR/venv/bin/python" ]; then
                "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/utils/ollama_check.py"
            elif [ -f "$ROOT_DIR/venv/Scripts/python" ]; then
                "$ROOT_DIR/venv/Scripts/python" "$ROOT_DIR/utils/ollama_check.py"
            else
                echo "Error: Python environment not found"
                echo "Please run 'bash ./irs.sh setup' first"
                exit 1
            fi
            ;;
        all)
            echo "Running all diagnostics..."
            run_diagnostics ollama
            # Add more diagnostic components as they're developed
            ;;
        *)
            echo "Unknown component: $component"
            echo "Available diagnostic components: ollama, all"
            exit 1
            ;;
    esac
}

# Check if any command was provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Parse the command
COMMAND="$1"
shift

# Process the command
case "$COMMAND" in
    setup|install)
        # Check if the retry flag is provided
        RETRY=""
        MODEL=""
        
        for arg in "$@"; do
            if [ "$arg" == "--retry" ]; then
                RETRY="--retry"
            elif [ "$arg" != "--help" ] && [ "${arg:0:1}" != "-" ]; then
                MODEL="$arg"
            fi
        done
        
        if [ -f "$ROOT_DIR/venv/bin/python" ]; then
            echo "Using existing virtual environment"
            activate_venv
            if [ -n "$MODEL" ]; then
                python "$ROOT_DIR/utils/setup/__init__.py" "$MODEL" $RETRY
            else
                python "$ROOT_DIR/utils/setup/__init__.py" $RETRY
            fi
        elif [ -f "$ROOT_DIR/venv/Scripts/python" ]; then
            echo "Using existing virtual environment"
            activate_venv
            if [ -n "$MODEL" ]; then
                python "$ROOT_DIR/utils/setup/__init__.py" "$MODEL" $RETRY
            else
                python "$ROOT_DIR/utils/setup/__init__.py" $RETRY
            fi
        else
            echo "Creating new virtual environment..."
            if command -v python3 &> /dev/null; then
                python3 -m venv "$ROOT_DIR/venv"
            elif command -v python &> /dev/null; then
                python -m venv "$ROOT_DIR/venv"
            else
                echo "Error: Python 3 is required but not found"
                exit 1
            fi
            
            activate_venv
            pip install --upgrade pip
            pip install -r "$ROOT_DIR/requirements.txt"
            
            if [ -n "$MODEL" ]; then
                python "$ROOT_DIR/utils/setup/__init__.py" "$MODEL" $RETRY
            else
                python "$ROOT_DIR/utils/setup/__init__.py" $RETRY
            fi
        fi
        ;;
    
    diagnose)
        # Check if a component was specified
        if [ $# -eq 0 ]; then
            echo "Error: No component specified for diagnostics"
            echo "Available components: ollama, all"
            exit 1
        fi
        
        # Activate the virtual environment if it exists
        if [ -f "$ROOT_DIR/venv/bin/activate" ] || [ -f "$ROOT_DIR/venv/Scripts/activate" ]; then
            activate_venv
            run_diagnostics "$1"
        else
            echo "Error: Virtual environment not found"
            echo "Please run 'bash ./irs.sh setup' first"
            exit 1
        fi
        ;;
    
    bulk)
        echo "Running bulk analysis on all documents in data/docs..."
        activate_venv
        
        # Always use the default models (llama3:8b, phi4, mixtral:8x7b) for bulk processing.
        echo "Processing with default models: llama3:8b, phi4, mixtral:8x7b."
        python "$ROOT_DIR/core/rag.py" --process --models "llama3:8b" "phi4" "mixtral:8x7b" "$@"
        ;;
    
    web|streamlit)
        echo "Starting web interface..."
        activate_venv
        
        # Check if a specific model was provided
        MODEL_ARG=""
        for arg in "$@"; do
            if [[ "$arg" == "--model" || "$arg" == "-m" ]]; then
                MODEL_FLAG=true
            elif [[ "$MODEL_FLAG" == true ]]; then
                MODEL_ARG="--model $arg"
                echo "Using specified model: $arg"
                MODEL_FLAG=false
                break
            fi
        done
        
        if [[ -z "$MODEL_ARG" ]]; then
            # Pass the default models to streamlit
            streamlit run "$ROOT_DIR/apps/streamlit/app.py" -- --default_models "llama3:8b" "phi4" "mixtral:8x7b" "$@"
        else
            # Pass the specified model to streamlit
            streamlit run "$ROOT_DIR/apps/streamlit/app.py" -- "$@"
        fi
        ;;
    
    metrics)
        echo "Starting metrics dashboard..."
        streamlit run "$ROOT_DIR/apps/metrics/dashboard.py" -- "$@"
        ;;
    
    clean|cleanup)
        echo "Cleaning up memory..."
        python -c "from utils.memory import MemoryOptimizer; MemoryOptimizer.clean_all()"
        ;;
    
    sysinfo)
        echo "Getting system information..."
        python -c "from utils.system import get_system_info, print_system_info; print_system_info()"
        ;;
    
    process)
        echo "Processing documents sequentially..."
        activate_venv
        # Extract models from arguments
        MODELS=("llama3:8b")
        for arg in "$@"; do
            if [[ "$arg" == "--models" ]]; then
                MODELS=()
                MODEL_FLAG=true
            elif [[ "$MODEL_FLAG" == true ]]; then
                if [[ "$arg" == --* ]]; then
                    MODEL_FLAG=false
                else
                    MODELS+=("$arg")
                fi
            fi
        done
        
        # Pass models to rag.py
        if [ ${#MODELS[@]} -gt 0 ]; then
            python "$ROOT_DIR/core/rag.py" --process --models "${MODELS[@]}"
        else
            python "$ROOT_DIR/core/rag.py" --process
        fi
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac