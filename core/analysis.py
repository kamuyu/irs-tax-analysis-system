#<!-- filepath: /root/IRS/core/analysis.py -->
#!/usr/bin/env python3
# Tax analysis logic for IRS Tax Analysis System

import os
import re
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import unittest
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("analysis")

@dataclass
class AnalysisResult:
    """Class representing analysis results for a question"""
    question: str
    answer: str
    confidence: float = 0.0
    reasoning: str = ""
    sources: List[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "execution_time": self.execution_time
        }
    
    def to_text(self) -> str:
        """Convert to formatted text"""
        result = f"Q: {self.question}\n\n"
        result += f"A: {self.answer}\n\n"
        
        if self.reasoning:
            result += f"Reasoning: {self.reasoning}\n\n"
        
        if self.sources:
            result += "Sources:\n"
            for src in self.sources:
                result += f"- {src}\n"
        
        return result

@dataclass
class ScenarioAnalysis:
    """Class representing analysis of a whole scenario"""
    scenario: str
    results: List[AnalysisResult]
    model_name: str
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "scenario": self.scenario,
            "results": [result.to_dict() for result in self.results],
            "model_name": self.model_name,
            "total_time": self.total_time
        }
    
    def to_text(self) -> str:
        """Convert to formatted text"""
        result = f"MODEL: {self.model_name}\n\n"
        result += f"SCENARIO:\n{self.scenario}\n\n"
        
        for i, analysis_result in enumerate(self.results, 1):
            result += f"Q{i}: {analysis_result.question}\n\n"
            result += f"A{i}: {analysis_result.answer}\n\n"
            result += "---\n\n"
        
        result += f"Analysis completed in {self.total_time:.2f} seconds"
        return result
    
    def save_to_file(self, output_dir: str, filename: Optional[str] = None) -> str:
        """Save analysis results to file"""
        if filename is None:
            # Create filename from model name and timestamp
            timestamp = int(time.time())
            filename = f"{self.model_name}_{timestamp}.txt"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path
        file_path = os.path.join(output_dir, filename)
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_text())
        
        logger.info(f"Saved analysis results to {file_path}")
        return file_path

class TaxAnalyzer:
    """Class to perform tax analysis using LLMs"""
    
    def __init__(self, model_manager, retriever):
        """Initialize with model manager and retriever"""
        self.model_manager = model_manager
        self.retriever = retriever
    
    def analyze_scenario(self, scenario_data: Dict[str, Any], model_name: str, output_dir: str = "./data/docs") -> ScenarioAnalysis:
        """Analyze a full scenario"""
        start_time = time.time()
        
        scenario = scenario_data["scenario"]
        questions = scenario_data["questions"]
        results = []
        
        # Process each question
        for question in questions:
            logger.info(f"Processing question: {question[:50]}...")
            result = self.analyze_question(scenario, question, model_name)
            results.append(result)
            
            # Progressive saving - save after each question
            partial_analysis = ScenarioAnalysis(
                scenario=scenario,
                results=results,
                model_name=model_name,
                total_time=time.time() - start_time
            )
            
            # Create filename from original document name
            original_filename = scenario_data.get("document", {}).metadata.get("filename", "unknown")
            filename = f"{model_name}_{original_filename}"
            
            # Save to file
            partial_analysis.save_to_file(output_dir, filename)
        
        # Complete analysis
        total_time = time.time() - start_time
        
        analysis = ScenarioAnalysis(
            scenario=scenario,
            results=results,
            model_name=model_name,
            total_time=total_time
        )
        
        return analysis
    
    def analyze_question(self, scenario: str, question: str, model_name: str) -> AnalysisResult:
        """Analyze a single question"""
        start_time = time.time()
        
        # Retrieve relevant information
        context = self._get_context_for_question(scenario, question)
        
        # Prepare prompt
        prompt = self._create_prompt(scenario, question, context)
        
        # Get answer from model
        response = self.model_manager.generate(model_name, prompt)
        
        # Parse response
        answer = self._parse_response(response)
        
        # Extract reasoning if available
        reasoning = self._extract_reasoning(response)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return AnalysisResult(
            question=question,
            answer=answer,
            confidence=0.85,  # Placeholder confidence
            reasoning=reasoning,
            sources=[c["metadata"]["source"] for c in context if "metadata" in c and "source" in c["metadata"]],
            execution_time=execution_time
        )
    
    def _get_context_for_question(self, scenario: str, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a question"""
        # Combine scenario and question for retrieval
        full_query = f"{scenario}\n{question}"
        
        # Retrieve relevant passages
        retrieval_results = self.retriever.retrieve(full_query, n_results=5)
        
        return retrieval_results
    
    def _create_prompt(self, scenario: str, question: str, context: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM"""
        prompt = "You are a tax expert assistant. Analyze the following tax scenario and question.\n\n"
        prompt += f"SCENARIO:\n{scenario}\n\n"
        prompt += f"QUESTION:\n{question}\n\n"
        
        prompt += "RELEVANT INFORMATION:\n"
        for ctx in context:
            prompt += f"---\n{ctx.get('text', '')}\n"
        
        prompt += "\nBased on the scenario, question, and relevant information, provide a detailed answer. "
        prompt += "Include your reasoning process, cite specific tax rules when applicable, "
        prompt += "and be precise in your conclusions."
        
        return prompt
    
    def _parse_response(self, response: str) -> str:
        """Parse the model response to extract the answer"""
        # For simple implementation, return the full response
        # In a more complex implementation, we'd extract the direct answer from reasoning
        return response.strip()
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from the model response"""
        # Look for reasoning sections
        reasoning_patterns = [
            r"(?:Reasoning|Analysis):\s*(.*?)(?=\n\n|\Z)",
            r"(?:Here's my reasoning|Let me analyze this):\s*(.*?)(?=\n\n|\Z)",
            r"(?:Let me think through this|My thought process):\s*(.*?)(?=\n\n|\Z)"
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit reasoning section, return empty string
        return ""

class FeedbackAnalyzer:
    """Class to analyze and provide feedback on model answers"""
    
    def __init__(self, model_manager):
        """Initialize with model manager"""
        self.model_manager = model_manager
    
    def generate_feedback(self, original_analysis: ScenarioAnalysis, other_analyses: List[ScenarioAnalysis] = None) -> str:
        """Generate feedback on the original analysis"""
        model_name = original_analysis.model_name
        logger.info(f"Generating feedback for {model_name}'s analysis")
        
        # Create feedback prompt
        prompt = self._create_feedback_prompt(original_analysis, other_analyses)
        
        # Get feedback from the same model
        feedback = self.model_manager.generate(model_name, prompt)
        
        return feedback
    
    def _create_feedback_prompt(self, original_analysis: ScenarioAnalysis, other_analyses: List[ScenarioAnalysis] = None) -> str:
        """Create a prompt for feedback generation"""
        prompt = "You are a tax expert tasked with reviewing and providing feedback on your previous analysis. "
        prompt += "First review your original answers, then provide specific feedback on each answer, "
        prompt += "noting any errors, omissions, or areas for improvement.\n\n"
        
        prompt += "SCENARIO:\n"
        prompt += original_analysis.scenario + "\n\n"
        
        prompt += "YOUR ORIGINAL ANSWERS:\n"
        for i, result in enumerate(original_analysis.results, 1):
            prompt += f"Q{i}: {result.question}\n\n"
            prompt += f"A{i}: {result.answer}\n\n"
        
        # Add other models' analyses if provided
        if other_analyses:
            prompt += "ANSWERS FROM OTHER MODELS:\n\n"
            for analysis in other_analyses:
                prompt += f"MODEL: {analysis.model_name}\n"
                for i, result in enumerate(analysis.results, 1):
                    prompt += f"Q{i}: {result.question}\n\n"
                    prompt += f"A{i}: {result.answer}\n\n"
                prompt += "---\n\n"
        
        prompt += "Based on this review, please provide:\n"
        prompt += "1. Specific feedback on each of your answers\n"
        prompt += "2. Any corrections or improvements you would make\n"
        prompt += "3. A comparative analysis against other models (if provided)\n"
        prompt += "4. Your final, revised answers incorporating all feedback\n"
        
        return prompt
    
    def save_feedback(self, original_analysis: ScenarioAnalysis, feedback: str, output_dir: str) -> str:
        """Save feedback to a file"""
        # Create filename
        timestamp = int(time.time())
        original_filename = f"{original_analysis.model_name}_{timestamp}"
        filename = f"{original_analysis.model_name}_{timestamp}_with_feedback.txt"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path
        file_path = os.path.join(output_dir, filename)
        
        # Format the content
        content = f"MODEL: {original_analysis.model_name}\n\n"
        content += f"ORIGINAL ANALYSIS:\n{original_analysis.to_text()}\n\n"
        content += f"FEEDBACK:\n{feedback}\n"
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Saved feedback to {file_path}")
        return file_path

# Unit tests
class TestAnalysis(unittest.TestCase):
    def test_analysis_result(self):
        # Create a test analysis result
        result = AnalysisResult(
            question="What is a tax deduction?",
            answer="A tax deduction is an expense that reduces your taxable income.",
            confidence=0.9,
            reasoning="Tax deductions reduce the amount of income that is subject to taxation.",
            sources=["IRS Publication 17"]
        )
        
        # Check text formatting
        text = result.to_text()
        self.assertTrue("tax deduction" in text.lower())
        self.assertTrue("reasoning" in text.lower())
        self.assertTrue("irs publication" in text.lower())

if __name__ == "__main__":
    # Run tests when file is executed directly
    unittest.main()