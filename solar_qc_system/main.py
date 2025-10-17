"""
Main entry point for the Solar Panel Quality Control System

This script serves as the main entry point for running the Streamlit dashboard
and provides command-line interface for data generation and analysis.

Usage:
    python main.py                    # Run Streamlit dashboard
    python main.py --generate-data    # Generate sample data only
    python main.py --run-analysis     # Run analysis on existing data

Author: Solar QC Team
Version: 1.0.0
"""

import sys
import os
import argparse
import logging
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our custom modules
from data_generator import SolarDataGenerator
from six_sigma_analysis import SixSigmaAnalysisEngine
from utils import setup_logging, save_dataframe
from config import *

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample datasets and save to CSV files"""
    try:
        logger.info("Generating sample datasets...")
        
        # Initialize data generator
        generator = SolarDataGenerator()
        
        # Generate datasets
        panel_data = generator.generate_panel_performance_data()
        failure_data = generator.generate_failure_events_data()
        environmental_data = generator.generate_environmental_data()
        
        # Create data directory if it doesn't exist
        data_dir = current_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save datasets
        save_dataframe(panel_data, "panel_performance.csv")
        save_dataframe(failure_data, "failure_events.csv")
        save_dataframe(environmental_data, "environmental_data.csv")
        
        logger.info("Sample datasets generated successfully!")
        print("✅ Sample datasets generated and saved to 'data' directory:")
        print(f"   - panel_performance.csv ({len(panel_data)} records)")
        print(f"   - failure_events.csv ({len(failure_data)} records)")
        print(f"   - environmental_data.csv ({len(environmental_data)} records)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        print(f"❌ Error generating sample data: {e}")
        return False

def run_analysis():
    """Run Six Sigma analysis on existing data"""
    try:
        logger.info("Running Six Sigma analysis...")
        
        # Check if data files exist
        data_dir = current_dir / "data"
        required_files = ["panel_performance.csv", "failure_events.csv", "environmental_data.csv"]
        
        missing_files = []
        for file in required_files:
            if not (data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing data files: {', '.join(missing_files)}")
            print("Please run 'python main.py --generate-data' first to create sample data.")
            return False
        
        # Load datasets
        import pandas as pd
        panel_data = pd.read_csv(data_dir / "panel_performance.csv")
        failure_data = pd.read_csv(data_dir / "failure_events.csv")
        environmental_data = pd.read_csv(data_dir / "environmental_data.csv")
        
        # Initialize analysis engine
        analysis_engine = SixSigmaAnalysisEngine()
        
        # Run comprehensive analysis
        results = analysis_engine.run_comprehensive_analysis(
            panel_data=panel_data,
            failure_data=failure_data,
            environmental_data=environmental_data
        )
        
        # Generate executive summary
        summary = analysis_engine.generate_executive_summary(results)
        
        # Print results
        print("✅ Six Sigma Analysis Results:")
        print("=" * 50)
        
        for key, value in summary.items():
            if key != 'recommendations':
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        if 'recommendations' in summary and summary['recommendations']:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Save detailed results
        results_dir = current_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save summary to text file
        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("SOLAR PANEL QUALITY CONTROL SYSTEM\n")
            f.write("SIX SIGMA ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                if key != 'recommendations':
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            if 'recommendations' in summary and summary['recommendations']:
                f.write("\nRecommendations:\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        print(f"\n📊 Detailed results saved to 'results' directory")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        print(f"❌ Error running analysis: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    try:
        import subprocess
        
        print("🚀 Starting Solar Panel Quality Control Dashboard...")
        print("📊 Dashboard will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard")
        
        # Run streamlit
        dashboard_path = current_dir / "dashboard.py"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Solar Panel Quality Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run interactive dashboard
    python main.py --generate-data    # Generate sample data
    python main.py --run-analysis     # Run analysis on existing data
    python main.py --help            # Show this help message
        """
    )
    
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate sample datasets and save to CSV files"
    )
    
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run Six Sigma analysis on existing data files"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Solar Panel Quality Control System v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("☀️ Solar Panel Quality Control System")
    print("🔬 Six Sigma Analysis for Utility-Scale Solar Power Plants")
    print("=" * 60)
    
    try:
        if args.generate_data:
            success = generate_sample_data()
            sys.exit(0 if success else 1)
        
        elif args.run_analysis:
            success = run_analysis()
            sys.exit(0 if success else 1)
        
        else:
            # Default: run dashboard
            run_dashboard()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()