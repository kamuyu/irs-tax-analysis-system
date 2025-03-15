#<!-- filepath: /root/IRS/utils/metrics.py -->
#!/usr/bin/env python3
# Metrics collection for IRS Tax Analysis System

import warnings
warnings.filterwarnings("ignore")

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any, Callable
import json
from pathlib import Path
import unittest
import socket
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("metrics")

class MetricsCollector:
    """Class to collect and store system metrics."""
    
    def __init__(self, metrics_dir: str = "./logs/metrics"):
        """Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics = {}
        self.start_time = time.time()
        self.hostname = socket.gethostname()
    
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record a metrics event.
        
        Args:
            event_type: Type of event (e.g., 'model_run', 'query', 'error')
            data: Event data dictionary
        """
        timestamp = time.time()
        
        # Create event record
        event = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "hostname": self.hostname,
            "event_type": event_type,
            "elapsed_seconds": timestamp - self.start_time,
            "data": data
        }
        
        # Log the event
        logger.debug(f"Metrics event: {event_type} - {json.dumps(data)}")
        
        # Store the event in the appropriate file
        self._store_event(event)
    
    def _store_event(self, event: Dict[str, Any]) -> None:
        """Store an event in the metrics file.
        
        Args:
            event: Event data to store
        """
        # Determine file path based on date and event type
        date_str = datetime.fromtimestamp(event["timestamp"]).strftime("%Y-%m-%d")
        event_type = event["event_type"]
        file_path = self.metrics_dir / f"{date_str}_{event_type}.jsonl"
        
        # Append event to file
        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Error storing metrics event: {e}")
    
    def record_model_run(self, model_name: str, prompt_tokens: int, completion_tokens: int, 
                       duration_ms: float, success: bool = True, error: str = None) -> None:
        """Record a model run event.
        
        Args:
            model_name: Name of the model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            duration_ms: Duration in milliseconds
            success: Whether the run was successful
            error: Error message if not successful
        """
        data = {
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": duration_ms,
            "tokens_per_second": ((prompt_tokens + completion_tokens) / duration_ms * 1000) if duration_ms > 0 else 0,
            "success": success
        }
        
        if error:
            data["error"] = error
        
        self.record_event("model_run", data)
    
    def record_query(self, query_type: str, query_text: str, num_results: int, 
                   duration_ms: float, success: bool = True, error: str = None) -> None:
        """Record a database query event.
        
        Args:
            query_type: Type of query (e.g., 'vector', 'knowledge_graph')
            query_text: Query text
            num_results: Number of results returned
            duration_ms: Duration in milliseconds
            success: Whether the query was successful
            error: Error message if not successful
        """
        data = {
            "query_type": query_type,
            "query_length": len(query_text),
            "num_results": num_results,
            "duration_ms": duration_ms,
            "success": success
        }
        
        if error:
            data["error"] = error
        
        self.record_event("query", data)
    
    def record_error(self, component: str, error_type: str, message: str, details: Dict[str, Any] = None) -> None:
        """Record an error event.
        
        Args:
            component: Component where the error occurred
            error_type: Type of error
            message: Error message
            details: Additional error details
        """
        data = {
            "component": component,
            "error_type": error_type,
            "message": message
        }
        
        if details:
            data["details"] = details
        
        self.record_event("error", data)

class PrometheusBridge:
    """Bridge to export metrics to Prometheus."""
    
    def __init__(self, port: int = 8000):
        """Initialize Prometheus bridge.
        
        Args:
            port: Port to expose Prometheus metrics
        """
        self.port = port
        self._metrics = {}
        self._running = False
        self._server_thread = None
    
    def start_server(self) -> None:
        """Start the Prometheus metrics server."""
        try:
            from prometheus_client import start_http_server, Counter, Gauge, Histogram
            
            # Create metrics
            self._metrics["model_runs_total"] = Counter(
                "irs_model_runs_total", 
                "Total number of model runs",
                ["model_name", "success"]
            )
            
            self._metrics["model_tokens_total"] = Counter(
                "irs_model_tokens_total", 
                "Total number of tokens processed",
                ["model_name", "token_type"]
            )
            
            self._metrics["model_duration_seconds"] = Histogram(
                "irs_model_duration_seconds", 
                "Duration of model runs in seconds",
                ["model_name"],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
            )
            
            self._metrics["tokens_per_second"] = Gauge(
                "irs_tokens_per_second", 
                "Tokens processed per second",
                ["model_name"]
            )
            
            self._metrics["query_duration_seconds"] = Histogram(
                "irs_query_duration_seconds", 
                "Duration of queries in seconds",
                ["query_type"],
                buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
            )
            
            self._metrics["error_total"] = Counter(
                "irs_error_total", 
                "Total number of errors",
                ["component", "error_type"]
            )
            
            # Start server
            start_http_server(self.port)
            self._running = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
            
        except ImportError:
            logger.warning("prometheus_client not installed. Prometheus metrics not available.")
            self._running = False
        except Exception as e:
            logger.error(f"Error starting Prometheus server: {e}")
            self._running = False
    
    def record_model_run(self, model_name: str, prompt_tokens: int, completion_tokens: int, 
                       duration_sec: float, success: bool = True) -> None:
        """Record a model run in Prometheus metrics.
        
        Args:
            model_name: Name of the model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            duration_sec: Duration in seconds
            success: Whether the run was successful
        """
        if not self._running:
            return
            
        success_str = "true" if success else "false"
        self._metrics["model_runs_total"].labels(model_name=model_name, success=success_str).inc()
        self._metrics["model_tokens_total"].labels(model_name=model_name, token_type="prompt").inc(prompt_tokens)
        self._metrics["model_tokens_total"].labels(model_name=model_name, token_type="completion").inc(completion_tokens)
        self._metrics["model_duration_seconds"].labels(model_name=model_name).observe(duration_sec)
        
        tokens_per_second = (prompt_tokens + completion_tokens) / duration_sec if duration_sec > 0 else 0
        self._metrics["tokens_per_second"].labels(model_name=model_name).set(tokens_per_second)
    
    def record_query(self, query_type: str, duration_sec: float) -> None:
        """Record a query in Prometheus metrics.
        
        Args:
            query_type: Type of query (e.g., 'vector', 'knowledge_graph')
            duration_sec: Duration in seconds
        """
        if not self._running:
            return
            
        self._metrics["query_duration_seconds"].labels(query_type=query_type).observe(duration_sec)
    
    def record_error(self, component: str, error_type: str) -> None:
        """Record an error in Prometheus metrics.
        
        Args:
            component: Component where the error occurred
            error_type: Type of error
        """
        if not self._running:
            return
            
        self._metrics["error_total"].labels(component=component, error_type=error_type).inc()

# Unit tests for metrics
class TestMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path("./test_metrics")
        self.temp_dir.mkdir(exist_ok=True)
        self.collector = MetricsCollector(str(self.temp_dir))
    
    def tearDown(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_record_event(self):
        self.collector.record_event("test", {"value": 123})
        
        # Check if a file was created
        files = list(self.temp_dir.glob("*.jsonl"))
        self.assertGreater(len(files), 0, "Should create a metrics file")
        
        # Check file content
        with open(files[0], 'r') as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["event_type"], "test")
            self.assertEqual(data["data"]["value"], 123)
    
    def test_record_model_run(self):
        self.collector.record_model_run("test_model", 100, 50, 1500, True)
        
        # Check if a file was created with correct data
        files = list(self.temp_dir.glob("*model_run.jsonl"))
        self.assertGreater(len(files), 0, "Should create a model_run metrics file")
        
        with open(files[0], 'r') as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["data"]["model_name"], "test_model")
            self.assertEqual(data["data"]["prompt_tokens"], 100)
            self.assertEqual(data["data"]["completion_tokens"], 50)

def load_model_metrics():
    metrics_file = Path(__file__).parent.parent / "data" / "metrics" / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as mf:
            metrics = json.load(mf)
        return metrics
    else:
        return {}

def report_metrics():
    metrics = load_model_metrics()
    if not metrics:
        print("No metrics data available.")
        return
    print("=== MODEL RUN STATISTICS ===")
    for model, data in metrics.items():
        print(f"Model: {model}")
        print(f"  Processed Documents: {data.get('processed',0)}")
        print(f"  Errors: {data.get('errors',0)}")
        print(f"  Total Processing Time: {data.get('total_time',0):.2f} seconds")
        print(f"  Average Time per Document: {data.get('average_time_per_doc',0):.2f} seconds")
        print("")

if __name__ == "__main__":
    # Simple demonstration
    collector = MetricsCollector()
    
    # Record model run
    collector.record_model_run("llama3:8b", 500, 200, 3500, True)
    
    # Record query
    collector.record_query("vector", "What is a tax deduction?", 5, 120, True)
    
    # Record error
    collector.record_error("rag", "retrieval_error", "Failed to retrieve documents")
    
    print("Metrics recorded successfully")
    
    # Start Prometheus server if prometheus_client is installed
    try:
        bridge = PrometheusBridge(port=8000)
        bridge.start_server()
        
        # Record some Prometheus metrics
        bridge.record_model_run("llama3:8b", 500, 200, 3.5, True)
        bridge.record_query("vector", 0.12)
        
        print("Prometheus metrics server started on port 8000")
        print("Press Ctrl+C to exit")
        
        # Keep the server running
        while True:
            time.sleep(1)
    except ImportError:
        print("prometheus_client not installed. Prometheus metrics not available.")
    except KeyboardInterrupt:
        print("Exiting...")