#<!-- filepath: /root/IRS/apps/metrics/dashboard.py -->
#!/usr/bin/env python3
# Metrics Dashboard for IRS Tax Analysis System

import os
import sys
import logging
from typing import Dict, List, Any
from pathlib import Path
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Streamlit if available
try:
    import streamlit as st
except ImportError:
    st = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("metrics_dashboard")

class MetricsDashboard:
    """Class to provide metrics visualization and analysis."""
    
    def __init__(self, metrics_dir: str = "./logs/metrics"):
        """Initialize metrics dashboard.
        
        Args:
            metrics_dir: Directory containing metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        self.data = {}
        
        # Load metrics data
        self.load_data()
    
    def load_data(self) -> None:
        """Load metrics data from files."""
        if not self.metrics_dir.exists():
            logger.warning(f"Metrics directory not found: {self.metrics_dir}")
            return
        
        # Find all metrics files
        metrics_files = list(self.metrics_dir.glob("*.jsonl"))
        logger.info(f"Found {len(metrics_files)} metrics files")
        
        # Process each file
        for file_path in metrics_files:
            file_name = file_path.name
            
            # Extract event type from filename
            event_type = file_name.split("_", 1)[1].split(".")[0]
            
            # Read file data
            try:
                with open(file_path, "r") as f:
                    events = [json.loads(line) for line in f]
                
                # Store data
                if event_type not in self.data:
                    self.data[event_type] = []
                
                self.data[event_type].extend(events)
                logger.info(f"Loaded {len(events)} events from {file_name}")
            
            except Exception as e:
                logger.error(f"Error loading metrics file {file_path}: {e}")
    
    def get_model_run_stats(self) -> Dict[str, Any]:
        """Get statistics for model runs.
        
        Returns:
            Dictionary of model run statistics
        """
        if "model_run" not in self.data or not self.data["model_run"]:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame([event["data"] for event in self.data["model_run"]])
        
        # Calculate statistics by model
        stats = {}
        for model_name, group in df.groupby("model_name"):
            stats[model_name] = {
                "runs": len(group),
                "success_rate": group["success"].mean() * 100,
                "avg_duration_ms": group["duration_ms"].mean(),
                "avg_tokens": group["total_tokens"].mean(),
                "avg_tokens_per_second": group["tokens_per_second"].mean(),
                "total_tokens": group["total_tokens"].sum()
            }
        
        # Overall statistics
        stats["overall"] = {
            "runs": len(df),
            "success_rate": df["success"].mean() * 100,
            "avg_duration_ms": df["duration_ms"].mean(),
            "avg_tokens": df["total_tokens"].mean(),
            "avg_tokens_per_second": df["tokens_per_second"].mean(),
            "total_tokens": df["total_tokens"].sum()
        }
        
        return stats
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics for queries.
        
        Returns:
            Dictionary of query statistics
        """
        if "query" not in self.data or not self.data["query"]:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame([event["data"] for event in self.data["query"]])
        
        # Calculate statistics by query type
        stats = {}
        for query_type, group in df.groupby("query_type"):
            stats[query_type] = {
                "queries": len(group),
                "success_rate": group["success"].mean() * 100,
                "avg_duration_ms": group["duration_ms"].mean(),
                "avg_results": group["num_results"].mean()
            }
        
        # Overall statistics
        stats["overall"] = {
            "queries": len(df),
            "success_rate": df["success"].mean() * 100,
            "avg_duration_ms": df["duration_ms"].mean(),
            "avg_results": df["num_results"].mean()
        }
        
        return stats
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics.
        
        Returns:
            Dictionary of error counts by component and type
        """
        if "error" not in self.data or not self.data["error"]:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame([event["data"] for event in self.data["error"]])
        
        # Count errors by component
        component_counts = df["component"].value_counts().to_dict()
        
        # Count errors by type
        type_counts = df["error_type"].value_counts().to_dict()
        
        return {
            "by_component": component_counts,
            "by_type": type_counts,
            "total": len(df)
        }
    
    def plot_model_performance(self, save_path: str = None) -> None:
        """Plot model performance metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if "model_run" not in self.data or not self.data["model_run"]:
            logger.warning("No model run data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([event["data"] for event in self.data["model_run"]])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Tokens per second by model
        axes[0, 0].set_title("Tokens per Second by Model")
        df.boxplot(column="tokens_per_second", by="model_name", ax=axes[0, 0])
        axes[0, 0].set_ylabel("Tokens/Second")
        axes[0, 0].set_xlabel("")
        
        # Plot 2: Duration by model
        axes[0, 1].set_title("Duration by Model")
        df.boxplot(column="duration_ms", by="model_name", ax=axes[0, 1])
        axes[0, 1].set_ylabel("Duration (ms)")
        axes[0, 1].set_xlabel("")
        
        # Plot 3: Success rate by model
        axes[1, 0].set_title("Success Rate by Model")
        success_rates = df.groupby("model_name")["success"].mean() * 100
        success_rates.plot.bar(ax=axes[1, 0])
        axes[1, 0].set_ylabel("Success Rate (%)")
        axes[1, 0].set_ylim(0, 100)
        
        # Plot 4: Total tokens by model
        axes[1, 1].set_title("Total Tokens by Model")
        df.groupby("model_name")["total_tokens"].sum().plot.bar(ax=axes[1, 1])
        axes[1, 1].set_ylabel("Total Tokens")
        
        # Adjust layout and add title
        plt.suptitle("Model Performance Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def plot_query_performance(self, save_path: str = None) -> None:
        """Plot query performance metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if "query" not in self.data or not self.data["query"]:
            logger.warning("No query data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([event["data"] for event in self.data["query"]])
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Duration by query type
        axes[0].set_title("Query Duration by Type")
        df.boxplot(column="duration_ms", by="query_type", ax=axes[0])
        axes[0].set_ylabel("Duration (ms)")
        axes[0].set_xlabel("")
        
        # Plot 2: Results count by query type
        axes[1].set_title("Results Count by Query Type")
        df.boxplot(column="num_results", by="query_type", ax=axes[1])
        axes[1].set_ylabel("Number of Results")
        axes[1].set_xlabel("")
        
        # Adjust layout and add title
        plt.suptitle("Query Performance Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_dir: str = "./logs/reports") -> str:
        """Generate a comprehensive metrics report.
        
        Args:
            output_dir: Directory to save the report
        
        Returns:
            Path to the generated report file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"metrics_report_{timestamp}.html"
        
        # Get statistics
        model_stats = self.get_model_run_stats()
        query_stats = self.get_query_stats()
        error_stats = self.get_error_stats()
        
        # Create HTML report
        with open(report_file, "w") as f:
            f.write("<html><head>")
            f.write("<title>IRS Tax Analysis System - Metrics Report</title>")
            f.write("<style>body {font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px;}")
            f.write("table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}")
            f.write("th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}")
            f.write("th {background-color: #f2f2f2;}")
            f.write("h1, h2, h3 {color: #333;}</style>")
            f.write("</head><body>")
            
            # Header
            f.write(f"<h1>IRS Tax Analysis System - Metrics Report</h1>")
            f.write(f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Model statistics
            f.write("<h2>Model Performance</h2>")
            if model_stats:
                f.write("<table><tr><th>Model</th><th>Runs</th><th>Success Rate</th><th>Avg Duration (ms)</th>")
                f.write("<th>Avg Tokens</th><th>Avg Tokens/s</th><th>Total Tokens</th></tr>")
                
                for model_name, stats in model_stats.items():
                    if model_name != "overall":
                        f.write(f"<tr><td>{model_name}</td>")
                        f.write(f"<td>{stats['runs']}</td>")
                        f.write(f"<td>{stats['success_rate']:.1f}%</td>")
                        f.write(f"<td>{stats['avg_duration_ms']:.1f}</td>")
                        f.write(f"<td>{stats['avg_tokens']:.1f}</td>")
                        f.write(f"<td>{stats['avg_tokens_per_second']:.1f}</td>")
                        f.write(f"<td>{stats['total_tokens']}</td></tr>")
                
                # Overall row
                if "overall" in model_stats:
                    stats = model_stats["overall"]
                    f.write(f"<tr style='font-weight: bold;'><td>Overall</td>")
                    f.write(f"<td>{stats['runs']}</td>")
                    f.write(f"<td>{stats['success_rate']:.1f}%</td>")
                    f.write(f"<td>{stats['avg_duration_ms']:.1f}</td>")
                    f.write(f"<td>{stats['avg_tokens']:.1f}</td>")
                    f.write(f"<td>{stats['avg_tokens_per_second']:.1f}</td>")
                    f.write(f"<td>{stats['total_tokens']}</td></tr>")
                
                f.write("</table>")
                
                # Add plots
                plot_file = Path(output_dir) / f"model_performance_{timestamp}.png"
                self.plot_model_performance(str(plot_file))
                if plot_file.exists():
                    f.write(f"<img src='{plot_file.name}' width='100%' />")
            
            else:
                f.write("<p>No model run data available</p>")
            
            # Query statistics
            f.write("<h2>Query Performance</h2>")
            if query_stats:
                f.write("<table><tr><th>Query Type</th><th>Queries</th><th>Success Rate</th>")
                f.write("<th>Avg Duration (ms)</th><th>Avg Results</th></tr>")
                
                for query_type, stats in query_stats.items():
                    if query_type != "overall":
                        f.write(f"<tr><td>{query_type}</td>")
                        f.write(f"<td>{stats['queries']}</td>")
                        f.write(f"<td>{stats['success_rate']:.1f}%</td>")
                        f.write(f"<td>{stats['avg_duration_ms']:.1f}</td>")
                        f.write(f"<td>{stats['avg_results']:.1f}</td></tr>")
                
                # Overall row
                if "overall" in query_stats:
                    stats = query_stats["overall"]
                    f.write(f"<tr style='font-weight: bold;'><td>Overall</td>")
                    f.write(f"<td>{stats['queries']}</td>")
                    f.write(f"<td>{stats['success_rate']:.1f}%</td>")
                    f.write(f"<td>{stats['avg_duration_ms']:.1f}</td>")
                    f.write(f"<td>{stats['avg_results']:.1f}</td></tr>")
                
                f.write("</table>")
                
                # Add plot
                plot_file = Path(output_dir) / f"query_performance_{timestamp}.png"
                self.plot_query_performance(str(plot_file))
                if plot_file.exists():
                    f.write(f"<img src='{plot_file.name}' width='100%' />")
            
            else:
                f.write("<p>No query data available</p>")
            
            # Error statistics
            f.write("<h2>Error Statistics</h2>")
            if error_stats:
                # Errors by component
                f.write("<h3>Errors by Component</h3>")
                f.write("<table><tr><th>Component</th><th>Error Count</th></tr>")
                for component, count in error_stats.get("by_component", {}).items():
                    f.write(f"<tr><td>{component}</td><td>{count}</td></tr>")
                f.write("</table>")
                
                # Errors by type
                f.write("<h3>Errors by Type</h3>")
                f.write("<table><tr><th>Error Type</th><th>Error Count</th></tr>")
                for error_type, count in error_stats.get("by_type", {}).items():
                    f.write(f"<tr><td>{error_type}</td><td>{count}</td></tr>")
                f.write("</table>")
                
                f.write(f"<p>Total errors: {error_stats.get('total', 0)}</p>")
            else:
                f.write("<p>No error data available</p>")
            
            # Footer
            f.write("</body></html>")
        
        logger.info(f"Generated metrics report at {report_file}")
        return str(report_file)

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    if not st:
        logger.error("Streamlit not installed. Cannot run dashboard.")
        return
    
    st.set_page_config(page_title="IRS Tax Analysis - Metrics Dashboard", layout="wide")
    st.title("IRS Tax Analysis System - Metrics Dashboard")
    
    # Initialize dashboard
    metrics_dir = st.sidebar.text_input("Metrics Directory", value="./logs/metrics")
    if st.sidebar.button("Load Data"):
        st.session_state.dashboard = MetricsDashboard(metrics_dir)
        st.success(f"Loaded metrics data from {metrics_dir}")
    
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = MetricsDashboard(metrics_dir)
    
    dashboard = st.session_state.dashboard
    
    # Display tabs
    tabs = st.tabs(["Model Performance", "Query Performance", "Errors", "Raw Data"])
    
    # Model Performance tab
    with tabs[0]:
        st.header("Model Performance Metrics")
        model_stats = dashboard.get_model_run_stats()
        
        if model_stats:
            # Display statistics table
            st.subheader("Statistics by Model")
            model_df = pd.DataFrame.from_dict(
                {model: {k: v for k, v in stats.items() if k != "total_tokens"} 
                 for model, stats in model_stats.items() if model != "overall"},
                orient="index"
            )
            st.dataframe(model_df)
            
            # Display total tokens
            st.subheader("Total Tokens by Model")
            tokens_df = pd.DataFrame(
                {"Total Tokens": {model: stats["total_tokens"] 
                                 for model, stats in model_stats.items() if model != "overall"}}
            )
            st.bar_chart(tokens_df)
            
            # Display success rate
            st.subheader("Success Rate by Model")
            success_df = pd.DataFrame(
                {"Success Rate (%)": {model: stats["success_rate"] 
                                     for model, stats in model_stats.items() if model != "overall"}}
            )
            st.bar_chart(success_df)
            
            # Display performance metrics
            st.subheader("Performance Metrics by Model")
            col1, col2 = st.columns(2)
            
            with col1:
                duration_df = pd.DataFrame(
                    {"Avg Duration (ms)": {model: stats["avg_duration_ms"] 
                                         for model, stats in model_stats.items() if model != "overall"}}
                )
                st.bar_chart(duration_df)
            
            with col2:
                tokens_per_sec_df = pd.DataFrame(
                    {"Avg Tokens/Second": {model: stats["avg_tokens_per_second"] 
                                         for model, stats in model_stats.items() if model != "overall"}}
                )
                st.bar_chart(tokens_per_sec_df)
            
        else:
            st.info("No model performance data available")
    
    # Query Performance tab
    with tabs[1]:
        st.header("Query Performance Metrics")
        query_stats = dashboard.get_query_stats()
        
        if query_stats:
            # Display statistics table
            st.subheader("Statistics by Query Type")
            query_df = pd.DataFrame.from_dict(
                {qtype: stats for qtype, stats in query_stats.items() if qtype != "overall"},
                orient="index"
            )
            st.dataframe(query_df)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Success Rate by Query Type")
                success_df = pd.DataFrame(
                    {"Success Rate (%)": {qtype: stats["success_rate"] 
                                         for qtype, stats in query_stats.items() if qtype != "overall"}}
                )
                st.bar_chart(success_df)
            
            with col2:
                st.subheader("Average Duration by Query Type")
                duration_df = pd.DataFrame(
                    {"Avg Duration (ms)": {qtype: stats["avg_duration_ms"] 
                                         for qtype, stats in query_stats.items() if qtype != "overall"}}
                )
                st.bar_chart(duration_df)
        else:
            st.info("No query performance data available")
    
    # Errors tab
    with tabs[2]:
        st.header("Error Statistics")
        error_stats = dashboard.get_error_stats()
        
        if error_stats:
            st.subheader(f"Total Errors: {error_stats.get('total', 0)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Errors by Component")
                component_df = pd.DataFrame(
                    list(error_stats.get("by_component", {}).items()),
                    columns=["Component", "Count"]
                )
                st.bar_chart(component_df.set_index("Component"))
            
            with col2:
                st.subheader("Errors by Type")
                type_df = pd.DataFrame(
                    list(error_stats.get("by_type", {}).items()),
                    columns=["Type", "Count"]
                )
                st.bar_chart(type_df.set_index("Type"))
        else:
            st.info("No error data available")
    
    # Raw Data tab
    with tabs[3]:
        st.header("Raw Metrics Data")
        
        for event_type, events in dashboard.data.items():
            if events:
                with st.expander(f"{event_type.capitalize()} Events ({len(events)})"):
                    # Convert to DataFrame for display
                    events_df = pd.DataFrame([{**event["data"], "timestamp": event["datetime"]} for event in events])
                    st.dataframe(events_df)
    
    # Generate report
    if st.sidebar.button("Generate HTML Report"):
        report_path = dashboard.generate_report()
        st.sidebar.success(f"Report generated at: {report_path}")
        
        # Provide download link if possible
        try:
            with open(report_path, "r") as f:
                report_content = f.read()
                st.sidebar.download_button(
                    "Download Report", 
                    report_content, 
                    file_name="metrics_report.html",
                    mime="text/html"
                )
        except:
            st.sidebar.info("Report created but cannot be downloaded directly from Streamlit")

def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IRS Tax Analysis System - Metrics Dashboard")
    parser.add_argument("--metrics-dir", default="./logs/metrics", help="Directory containing metrics files")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--output", default="./logs/reports", help="Output directory for report")
    parser.add_argument("--plot", action="store_true", help="Generate and show plots")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit dashboard")
    
    args = parser.parse_args()
    
    if args.streamlit:
        run_streamlit_dashboard()
        return
    
    # Initialize dashboard with metrics data
    dashboard = MetricsDashboard(args.metrics_dir)
    
    # Print statistics
    print("\n=== MODEL RUN STATISTICS ===")
    model_stats = dashboard.get_model_run_stats()
    if model_stats:
        for model, stats in model_stats.items():
            print(f"{model}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("No model run data available")
    
    print("\n=== QUERY STATISTICS ===")
    query_stats = dashboard.get_query_stats()
    if query_stats:
        for query_type, stats in query_stats.items():
            print(f"{query_type}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("No query data available")
    
    print("\n=== ERROR STATISTICS ===")
    error_stats = dashboard.get_error_stats()
    if error_stats:
        print(f"Total errors: {error_stats.get('total', 0)}")
        
        print("\nErrors by component:")
        for component, count in error_stats.get("by_component", {}).items():
            print(f"  {component}: {count}")
        
        print("\nErrors by type:")
        for error_type, count in error_stats.get("by_type", {}).items():
            print(f"  {error_type}: {count}")
    else:
        print("No error data available")
    
    # Generate plots if requested
    if args.plot:
        dashboard.plot_model_performance()
        dashboard.plot_query_performance()
    
    # Generate report if requested
    if args.report:
        report_path = dashboard.generate_report(args.output)
        print(f"\nGenerated report at: {report_path}")

if __name__ == "__main__":
    main()