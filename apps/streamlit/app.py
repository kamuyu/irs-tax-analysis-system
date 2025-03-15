#<!-- filepath: /root/IRS/apps/streamlit/app.py -->
#!/usr/bin/env python3
# Streamlit web interface for IRS Tax Analysis System

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import threading

# Add parent directory to path so we can import project modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import required modules
import streamlit as st
from core.models import ModelManager
from core.rag import DocumentProcessor, VectorDatabaseManager, HybridRetriever, Document
from core.analysis import TaxAnalyzer, FeedbackAnalyzer
from utils.memory import MemoryOptimizer
from utils.system import clean_memory, optimize_gpu_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("streamlit_app")

# Initialize session state
def init_session_state():
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'available_models' not in st.session_state:
        st.session_state.available_models = st.session_state.model_manager.get_available_models()
    
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabaseManager()
        st.session_state.vector_db.initialize()
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = HybridRetriever(st.session_state.vector_db)
    
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

# Function to process a scenario with models
def process_scenario(scenario_text, questions, selected_models):
    """Process a scenario with selected models"""
    st.session_state.processing = True
    st.session_state.answers = {}
    st.session_state.feedback = {}
    
    try:
        # Create document
        doc = Document(
            content=scenario_text + "\n\n" + "\n\n".join(questions),
            metadata={
                "source": "streamlit_input",
                "filename": "user_query.txt",
                "type": "text"
            }
        )
        
        # Parse scenario and questions
        doc_processor = DocumentProcessor()
        doc_info = doc_processor.parse_scenario_and_questions(doc)
        
        # Initialize analyzer
        analyzer = TaxAnalyzer(st.session_state.model_manager, st.session_state.retriever)
        feedback_analyzer = FeedbackAnalyzer(st.session_state.model_manager)
        
        # Process with each model
        for model_name in selected_models:
            st.session_state.answers[model_name] = {"status": "processing", "results": []}
            
            # Run model analysis in a separate thread
            def analyze_with_model(model_name=model_name):
                try:
                    # Analyze scenario
                    analysis = analyzer.analyze_scenario(doc_info, model_name)
                    
                    # Update session state
                    st.session_state.answers[model_name] = {
                        "status": "completed",
                        "results": [result.to_dict() for result in analysis.results],
                        "analysis": analysis
                    }
                    
                    # Generate feedback
                    other_analyses = []
                    for other_model in selected_models:
                        if other_model != model_name and other_model in st.session_state.answers:
                            if "analysis" in st.session_state.answers[other_model]:
                                other_analyses.append(st.session_state.answers[other_model]["analysis"])
                    
                    # If we have other analyses, generate feedback
                    if other_analyses:
                        feedback = feedback_analyzer.generate_feedback(analysis, other_analyses)
                        st.session_state.feedback[model_name] = feedback
                    
                    # Clean memory
                    clean_memory(model_name)
                    
                except Exception as e:
                    logger.error(f"Error processing with model {model_name}: {e}")
                    st.session_state.answers[model_name] = {"status": "error", "message": str(e)}
            
            # Start analysis thread
            threading.Thread(target=analyze_with_model).start()
    
    except Exception as e:
        logger.error(f"Error setting up processing: {e}")
        st.error(f"Error: {e}")
    
    st.session_state.processing = False

# Main app
def main():
    # Page config
    st.set_page_config(
        page_title="IRS Tax Analysis System",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # Title and description
    st.title("IRS Tax Analysis System")
    st.markdown("""
    Analyze tax scenarios using multiple AI models with Retrieval Augmented Generation (RAG).
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Select Models")
        default_models = ["llama3:8b"]
        
        # Refresh models button
        if st.button("Refresh Available Models"):
            st.session_state.available_models = st.session_state.model_manager.get_available_models()
        
        # Show available models as checkboxes
        model_options = st.session_state.available_models if st.session_state.available_models else ["llama3:8b"]
        selected_models = []
        
        for model in model_options:
            if st.checkbox(model, value=model in default_models):
                selected_models.append(model)
        
        st.session_state.selected_models = selected_models
        
        # Model info
        st.subheader("Model Information")
        st.markdown("""
        - **llama3:8b**: Fast with high reasoning capability
        - **phi4:medium**: Fast with high reasoning capability
        - **mixtral:8x7b**: Very high reasoning with medium speed
        - **yi:34b**: Very high reasoning but slower
        """)
        
        # Memory management
        if st.button("Clean Memory"):
            clean_memory()
            st.success("Memory cleaned")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Option to upload a file or enter text
        input_option = st.radio("Input Method", ["Enter Text", "Upload File"])
        
        if input_option == "Enter Text":
            scenario = st.text_area("Enter Scenario", height=200, help="Enter the tax scenario here")
            
            # Questions input
            st.subheader("Questions")
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=1)
            
            questions = []
            for i in range(num_questions):
                question = st.text_area(f"Question {i+1}", height=100, key=f"question_{i}")
                questions.append(question)
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
            
            if uploaded_file is not None:
                # Read and parse the file
                content = uploaded_file.getvalue().decode("utf-8")
                
                # Create temporary file to process
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                    temp_file.write(content.encode())
                    temp_file_path = temp_file.name
                
                # Process the file
                try:
                    doc_processor = DocumentProcessor()
                    doc = Document(
                        content=content,
                        metadata={
                            "source": uploaded_file.name,
                            "filename": uploaded_file.name,
                            "type": "text"
                        }
                    )
                    
                    doc_info = doc_processor.parse_scenario_and_questions(doc)
                    scenario = doc_info["scenario"]
                    questions = doc_info["questions"]
                    
                    # Display parsed content
                    st.subheader("Parsed Scenario")
                    st.write(scenario)
                    
                    st.subheader(f"Parsed Questions ({len(questions)})")
                    for i, q in enumerate(questions):
                        st.text_area(f"Question {i+1}", value=q, height=100, key=f"q_{i}")
                    
                finally:
                    # Remove temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
        
        # Process button
        if st.button("Process with Selected Models", disabled=len(st.session_state.selected_models) == 0):
            if not st.session_state.selected_models:
                st.error("Please select at least one model")
            elif not scenario or not any(questions):
                st.error("Please enter a scenario and at least one question")
            else:
                process_scenario(scenario, questions, st.session_state.selected_models)
    
    with col2:
        st.header("Results")
        
        # Show results
        tabs = st.tabs(["Results"] + st.session_state.selected_models + ["Feedback"])
        
        # First tab shows all results
        with tabs[0]:
            st.subheader("All Model Results")
            
            for model_name in st.session_state.selected_models:
                if model_name in st.session_state.answers:
                    model_result = st.session_state.answers[model_name]
                    
                    if model_result["status"] == "processing":
                        st.info(f"{model_name}: Processing...")
                    
                    elif model_result["status"] == "completed":
                        st.success(f"{model_name}: Completed")
                        
                        # Display results
                        for i, result in enumerate(model_result["results"]):
                            with st.expander(f"Question {i+1}"):
                                st.write(f"Q: {result['question']}")
                                st.write(f"A: {result['answer']}")
                    
                    elif model_result["status"] == "error":
                        st.error(f"{model_name}: Error - {model_result.get('message', 'Unknown error')}")
        
        # Tab for each model
        for i, model_name in enumerate(st.session_state.selected_models):
            with tabs[i+1]:
                st.subheader(f"{model_name} Results")
                
                if model_name in st.session_state.answers:
                    model_result = st.session_state.answers[model_name]
                    
                    if model_result["status"] == "processing":
                        st.info("Processing...")
                        st.spinner()
                    
                    elif model_result["status"] == "completed":
                        # Display detailed results
                        for i, result in enumerate(model_result["results"]):
                            st.write(f"### Question {i+1}")
                            st.write(f"**Q:** {result['question']}")
                            st.write(f"**A:** {result['answer']}")
                            
                            if result.get('reasoning'):
                                with st.expander("Reasoning"):
                                    st.write(result['reasoning'])
                            
                            if result.get('sources'):
                                with st.expander("Sources"):
                                    for src in result['sources']:
                                        st.write(f"- {src}")
                            
                            st.write("---")
                    
                    elif model_result["status"] == "error":
                        st.error(f"Error: {model_result.get('message', 'Unknown error')}")
        
        # Feedback tab
        with tabs[-1]:
            st.subheader("Model Feedback")
            
            for model_name in st.session_state.selected_models:
                if model_name in st.session_state.feedback:
                    with st.expander(f"{model_name} Feedback"):
                        st.write(st.session_state.feedback[model_name])

if __name__ == "__main__":
    main()