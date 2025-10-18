"""
Streamlit Dashboard for Solar Panel Quality Control System

This module provides an interactive web-based dashboard for visualizing
and analyzing solar panel performance using Six Sigma methodologies.

Features:
- Interactive data upload and generation
- Real-time SPC control charts
- Process capability analysis
- Pareto charts for failure analysis
- Reliability metrics and trends
- Cost analysis and ROI calculations
- Executive summary and recommendations

Author: Solar QC Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64
import logging
from typing import Dict, List, Optional

# Import our custom modules
from six_sigma_analysis import SixSigmaAnalysisEngine
from config import *
from utils import setup_logging, save_dataframe, load_dataframe

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Solar Panel Quality Control System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SolarQCDashboard:
    """Main dashboard class for the Solar QC System"""
    
    def __init__(self):
        self.analysis_engine = SixSigmaAnalysisEngine()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            ☀️ Solar Panel Quality Control System
            <br><small>Six Sigma Analysis for Utility-Scale Solar Power Plants</small>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.markdown("## 🎛️ Control Panel")
        
        # Data Management Section
        with st.sidebar.expander("📊 Data Management", expanded=True):
            # File upload option
            st.markdown("### Upload Your Data")
            uploaded_files = st.file_uploader(
                "Upload CSV files", 
                accept_multiple_files=True,
                type=['csv']
            )
            
            if uploaded_files:
                self.handle_file_upload(uploaded_files)
            
            # Load existing data button
            if st.button("📂 Load Existing Data", type="primary"):
                self.load_existing_data()
        
        # Analysis Options
        with st.sidebar.expander("🔍 Analysis Options", expanded=True):
            st.markdown("### Select Analysis Type")
            
            analysis_options = st.multiselect(
                "Choose analyses to run:",
                ["SPC Control Charts", "Process Capability", "Pareto Analysis", 
                 "Reliability Analysis", "Cost Analysis"],
                default=["SPC Control Charts", "Process Capability", "Pareto Analysis"]
            )
            
            if st.button("🚀 Run Analysis") and st.session_state.data_loaded:
                self.run_analysis(analysis_options)
        
        # Export Options
        with st.sidebar.expander("💾 Export Options"):
            if st.session_state.data_loaded:
                if st.button("📥 Download Generated Data"):
                    self.download_data()
                
                if st.session_state.analysis_results:
                    if st.button("📊 Download Analysis Report"):
                        self.download_report()
    
    def load_existing_data(self):
        """Load existing data from CSV files"""
        try:
            logger.info("Starting data loading process...")
            data_dir = Path("data")
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                st.error("❌ Data directory not found. Please generate data first using: python main.py --generate-data")
                return
            
            logger.info(f"Data directory found: {data_dir}")
            
            # Expected data files
            expected_files = {
                'panel_performance': 'panel_performance.csv',
                'failure_events': 'failure_events.csv', 
                'environmental_data': 'environmental_data.csv'
            }
            
            datasets = {}
            missing_files = []
            
            for dataset_name, filename in expected_files.items():
                file_path = data_dir / filename
                logger.info(f"Checking file: {file_path}")
                if file_path.exists():
                    logger.info(f"Loading dataset: {dataset_name} from {file_path}")
                    df = load_dataframe(str(file_path))
                    logger.info(f"Dataset {dataset_name} loaded successfully - Shape: {df.shape}, Columns: {list(df.columns)}")
                    logger.info(f"Dataset {dataset_name} data types: {df.dtypes.to_dict()}")
                    logger.info(f"Dataset {dataset_name} null values: {df.isnull().sum().to_dict()}")
                    datasets[dataset_name] = df
                else:
                    logger.warning(f"File not found: {file_path}")
                    missing_files.append(filename)
            
            if missing_files:
                logger.error(f"Missing files: {missing_files}")
                st.warning(f"⚠️ Missing data files: {', '.join(missing_files)}. Please generate data first using: python main.py --generate-data")
                return
            
            # Store in session state
            logger.info(f"Storing {len(datasets)} datasets in session state")
            st.session_state.datasets = datasets
            st.session_state.data_loaded = True
            
            # Debug session state
            logger.info(f"Session state datasets keys: {list(st.session_state.datasets.keys())}")
            logger.info(f"Session state data_loaded: {st.session_state.data_loaded}")
            
            st.success("✅ Successfully loaded existing data!")
            
            # Show data preview
            self.show_data_preview()
            
        except Exception as e:
            logger.error(f"Data loading error: {e}", exc_info=True)
            st.error(f"❌ Error loading data: {str(e)}")
    
    def handle_file_upload(self, uploaded_files):
        """Handle uploaded CSV files"""
        try:
            datasets = {}
            
            for file in uploaded_files:
                df = pd.read_csv(file)
                file_name = file.name.replace('.csv', '')
                datasets[file_name] = df
                st.sidebar.success(f"✅ Loaded {file.name}")
            
            st.session_state.datasets = datasets
            st.session_state.data_loaded = True
            
            self.show_data_preview()
            
        except Exception as e:
            st.error(f"❌ Error uploading files: {str(e)}")
            logger.error(f"File upload error: {e}")
    
    def show_data_preview(self):
        """Show preview of loaded data"""
        st.markdown("## 📋 Data Preview")
        
        if not st.session_state.datasets:
            st.warning("No data available. Please load existing data or upload CSV files first.")
            return
        
        tabs = st.tabs(list(st.session_state.datasets.keys()))
        
        for i, (dataset_name, df) in enumerate(st.session_state.datasets.items()):
            with tabs[i]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Show data types and basic stats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Data Types")
                    # Convert dtypes to string to avoid Arrow serialization issues
                    dtypes_df = df.dtypes.to_frame('Type')
                    dtypes_df['Type'] = dtypes_df['Type'].astype(str)
                    st.dataframe(dtypes_df, width='stretch')
                
                with col2:
                    st.markdown("### Basic Statistics")
                    st.dataframe(df.describe(), width='stretch')
                
                # Show sample data with proper datetime formatting
                st.markdown("### Sample Data")
                display_df = df.head(10).copy()
                
                # Convert datetime columns to string to avoid Arrow conversion issues
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        try:
                            # Try to convert to datetime and then to string with explicit format
                            # Check if it looks like a date column
                            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                                temp_series = pd.to_datetime(display_df[col], format='%Y-%m-%d', errors='coerce')
                                if not temp_series.isna().all():  # If at least some dates were parsed
                                    display_df[col] = temp_series.dt.strftime('%Y-%m-%d')
                        except:
                            # If conversion fails, keep as is
                            pass
                
                st.dataframe(display_df, width='stretch')
    
    def run_analysis(self, analysis_options: List[str]):
        """Run selected analyses"""
        logger.info(f"Starting analysis with options: {analysis_options}")
        
        if not st.session_state.data_loaded:
            st.error("❌ No data available. Please load existing data or upload CSV files first.")
            logger.warning("No data loaded when trying to run analysis")
            return
        
        try:
            with st.spinner("Running Six Sigma analysis..."):
                datasets = st.session_state.datasets
                logger.info(f"Available datasets: {list(datasets.keys()) if datasets else 'None'}")
                
                # Check if we have the required datasets
                if not datasets:
                    st.error("❌ No datasets available. Please load data first.")
                    logger.error("No datasets available in session state")
                    return
                
                # Log dataset shapes for debugging
                for name, df in datasets.items():
                    if isinstance(df, pd.DataFrame):
                        logger.info(f"Dataset {name}: shape={df.shape}, columns={list(df.columns)}")
                    else:
                        logger.warning(f"Dataset {name} is not a DataFrame: type={type(df)}")
                
                # Run comprehensive analysis
                logger.info("Calling run_comprehensive_analysis...")
                results = self.analysis_engine.run_comprehensive_analysis(
                    panel_data=datasets.get('panel_performance', pd.DataFrame()),
                    failure_data=datasets.get('failure_events', pd.DataFrame()),
                    environmental_data=datasets.get('environmental_data', pd.DataFrame())
                )
                
                logger.info(f"Analysis engine returned: type={type(results)}, value={results}")
                
                # Ensure results is not None
                if results is None:
                    st.error("❌ Analysis returned no results. Please check the data and try again.")
                    logger.error("Analysis engine returned None - this is the root cause of the NoneType error")
                    return
                
                # Check for errors in results
                if isinstance(results, dict):
                    if 'error' in results:
                        st.error(f"❌ Analysis Error: {results.get('message', 'Unknown error')}")
                        logger.error(f"Analysis engine error: {results.get('error', 'Unknown')}")
                        if 'traceback' in results:
                            with st.expander("Error Details"):
                                st.code(results['traceback'])
                        return
                    
                    # Check if analysis was successful
                    if not results.get('success', False):
                        st.error(f"❌ Analysis failed: {results.get('message', 'Analysis was not successful')}")
                        logger.error(f"Analysis not successful: {results}")
                        return
                else:
                    st.error("❌ Analysis returned unexpected result format.")
                    logger.error(f"Unexpected result type: {type(results)}")
                    return
                
                logger.info("Analysis completed successfully, storing results")
                st.session_state.analysis_results = results
                
                st.success("✅ Analysis completed successfully!")
                
                # Show results
                self.show_analysis_results(analysis_options)
                
        except Exception as e:
            st.error(f"❌ Error running analysis: {str(e)}")
            logger.error(f"Analysis error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def show_analysis_results(self, analysis_options: List[str]):
        """Display analysis results"""
        if not st.session_state.analysis_results:
            st.warning("No analysis results available.")
            return
        
        results = st.session_state.analysis_results
        
        # Executive Summary
        st.markdown("## 📊 Executive Summary")
        
        summary = self.analysis_engine.generate_executive_summary(results)
        
        if summary:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'process_capability' in summary:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Process Capability</h4>
                        <p>{summary['process_capability']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'reliability' in summary:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⚡ Reliability</h4>
                        <p>{summary['reliability']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'cost_impact' in summary:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>💰 Cost Impact</h4>
                        <p>{summary['cost_impact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'process_control' in summary:
                    status_class = "success-box" if "in statistical control" in summary['process_control'] else "warning-box"
                    st.markdown(f"""
                    <div class="{status_class}">
                        <h4>📈 Process Control</h4>
                        <p>{summary['process_control']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            if 'recommendations' in summary and summary['recommendations']:
                st.markdown("### 💡 Recommendations")
                for i, rec in enumerate(summary['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")
        
        # Detailed Analysis Results
        st.markdown("## 🔍 Detailed Analysis")
        
        # Create tabs for different analyses
        available_analyses = []
        if "SPC Control Charts" in analysis_options and 'spc_efficiency' in results:
            available_analyses.append("SPC Control Charts")
        if "Process Capability" in analysis_options and 'capability_analysis' in results:
            available_analyses.append("Process Capability")
        if "Pareto Analysis" in analysis_options and 'pareto_failures' in results:
            available_analyses.append("Pareto Analysis")
        if "Reliability Analysis" in analysis_options and 'reliability_trend' in results:
            available_analyses.append("Reliability Analysis")
        if "Cost Analysis" in analysis_options and 'cost_dashboard' in results:
            available_analyses.append("Cost Analysis")
        
        if available_analyses:
            tabs = st.tabs(available_analyses)
            
            for i, analysis_type in enumerate(available_analyses):
                with tabs[i]:
                    self.render_analysis_tab(analysis_type, results)
    
    def render_analysis_tab(self, analysis_type: str, results: Dict):
        """Render individual analysis tab"""
        if analysis_type == "SPC Control Charts":
            st.markdown("### Statistical Process Control Charts")
            
            if 'spc_efficiency' in results:
                st.plotly_chart(results['spc_efficiency'], use_container_width=True)
            
            # Show control limits and patterns
            if 'efficiency_limits' in results:
                limits = results['efficiency_limits']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ucl = limits.get('UCL', 0)
                    ucl = 0 if ucl is None else ucl
                    st.metric("Upper Control Limit", f"{ucl:.3f}")
                with col2:
                    cl = limits.get('CL', 0)
                    cl = 0 if cl is None else cl
                    st.metric("Center Line", f"{cl:.3f}")
                with col3:
                    lcl = limits.get('LCL', 0)
                    lcl = 0 if lcl is None else lcl
                    st.metric("Lower Control Limit", f"{lcl:.3f}")
            
            if 'efficiency_patterns' in results:
                patterns = results['efficiency_patterns']
                st.markdown("#### Pattern Detection")
                
                for pattern_type, indices in patterns.items():
                    if indices:
                        st.warning(f"⚠️ {pattern_type.replace('_', ' ').title()}: {len(indices)} points detected")
        
        elif analysis_type == "Process Capability":
            st.markdown("### Process Capability Analysis")
            
            if 'capability_analysis' in results:
                st.plotly_chart(results['capability_analysis'], use_container_width=True)
            
            if 'capability_indices' in results:
                indices = results['capability_indices']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cp = indices.get('Cp', 0)
                    cp = 0 if cp is None else cp
                    st.metric("Cp", f"{cp:.3f}")
                with col2:
                    cpk = indices.get('Cpk', 0)
                    cpk = 0 if cpk is None else cpk
                    st.metric("Cpk", f"{cpk:.3f}")
                with col3:
                    pp = indices.get('Pp', 0)
                    pp = 0 if pp is None else pp
                    st.metric("Pp", f"{pp:.3f}")
                with col4:
                    ppk = indices.get('Ppk', 0)
                    ppk = 0 if ppk is None else ppk
                    st.metric("Ppk", f"{ppk:.3f}")
        
        elif analysis_type == "Pareto Analysis":
            st.markdown("### Pareto Analysis - Failure Modes")
            
            if 'pareto_failures' in results:
                st.plotly_chart(results['pareto_failures'], use_container_width=True)
                
                st.markdown("""
                **Pareto Principle (80/20 Rule)**: Focus on the failure modes that contribute 
                to 80% of the problems for maximum impact on quality improvement.
                """)
        
        elif analysis_type == "Reliability Analysis":
            st.markdown("### Reliability Analysis")
            
            if 'reliability_metrics' in results:
                metrics = results['reliability_metrics']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mtbf = metrics.get('MTBF_hours', 0)
                    mtbf = 0 if mtbf is None else mtbf
                    st.metric("MTBF (Hours)", f"{mtbf:.1f}")
                with col2:
                    mttr = metrics.get('MTTR_hours', 0)
                    mttr = 0 if mttr is None else mttr
                    st.metric("MTTR (Hours)", f"{mttr:.1f}")
                with col3:
                    availability = metrics.get('availability_percent', 0)
                    availability = 0 if availability is None else availability
                    st.metric("Availability (%)", f"{availability:.1f}")
            
            if 'reliability_trend' in results:
                st.plotly_chart(results['reliability_trend'], use_container_width=True)
        
        elif analysis_type == "Cost Analysis":
            st.markdown("### Cost Analysis")
            
            if 'cost_metrics' in results:
                metrics = results['cost_metrics']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_cost = metrics.get('total_repair_cost', 0)
                    total_cost = 0 if total_cost is None else total_cost
                    st.metric("Total Repair Cost", f"${total_cost:,.2f}")
                with col2:
                    avg_cost = metrics.get('avg_repair_cost', 0)
                    avg_cost = 0 if avg_cost is None else avg_cost
                    st.metric("Average Repair Cost", f"${avg_cost:,.2f}")
                with col3:
                    max_cost = metrics.get('max_repair_cost', 0)
                    max_cost = 0 if max_cost is None else max_cost
                    st.metric("Max Repair Cost", f"${max_cost:,.2f}")
            
            if 'cost_dashboard' in results:
                st.plotly_chart(results['cost_dashboard'], use_container_width=True)
    
    def download_data(self):
        """Provide data download functionality"""
        if not st.session_state.datasets:
            st.error("No data to download")
            return
        
        # Create download links for each dataset
        for dataset_name, df in st.session_state.datasets.items():
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{dataset_name}.csv">Download {dataset_name}.csv</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
    
    def download_report(self):
        """Generate and download analysis report"""
        if not st.session_state.analysis_results:
            st.error("No analysis results to download")
            return
        
        # Generate report content
        report_content = self.generate_report_content()
        
        b64 = base64.b64encode(report_content.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="six_sigma_analysis_report.txt">Download Analysis Report</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    
    def generate_report_content(self) -> str:
        """Generate text report content"""
        results = st.session_state.analysis_results
        summary = self.analysis_engine.generate_executive_summary(results)
        
        report = f"""
SOLAR PANEL QUALITY CONTROL SYSTEM
SIX SIGMA ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
"""
        
        for key, value in summary.items():
            if key != 'recommendations':
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        if 'recommendations' in summary:
            report += "\nRECOMMENDATIONS\n===============\n"
            for i, rec in enumerate(summary['recommendations'], 1):
                report += f"{i}. {rec}\n"
        
        # Add detailed metrics
        if 'capability_indices' in results:
            report += "\nPROCESS CAPABILITY INDICES\n==========================\n"
            indices = results['capability_indices']
            for key, value in indices.items():
                if isinstance(value, (int, float)):
                    report += f"{key}: {value:.3f}\n"
        
        if 'reliability_metrics' in results:
            report += "\nRELIABILITY METRICS\n===================\n"
            metrics = results['reliability_metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    report += f"{key}: {value:.2f}\n"
        
        return report
    
    def run(self):
        """Main dashboard runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        if not st.session_state.data_loaded:
            st.markdown("""
            ## 🚀 Welcome to the Solar Panel Quality Control System
            
            This system uses **Six Sigma methodologies** to analyze and improve 
            solar panel performance and reliability for utility-scale installations.
            
            ### Getting Started:
            1. **Load existing data** or **upload your own CSV files** using the sidebar
            2. **Select analysis types** you want to run
            3. **Click "Run Analysis"** to generate insights
            4. **Review results** and implement recommendations
            
            ### Key Features:
            - 📊 **Statistical Process Control (SPC)** charts for real-time monitoring
            - 🎯 **Process Capability Analysis** to assess performance against specifications
            - 📈 **Pareto Analysis** to identify critical failure modes
            - ⚡ **Reliability Analysis** with MTBF/MTTR calculations
            - 💰 **Cost Analysis** for ROI optimization
            
            ### Data Requirements:
            - **Panel Performance Data**: Efficiency, power output, temperature, age
            - **Failure Events Data**: Failure types, repair costs, downtime
            - **Environmental Data**: Weather conditions, irradiance, humidity
            
            ### Note:
            Sample datasets have been pre-generated. Use the "📂 Load Existing Data" button 
            in the sidebar to load them, or upload your own CSV files.
            """)
        else:
            # Show data preview and analysis results
            self.show_data_preview()
            
            if st.session_state.analysis_results:
                self.show_analysis_results(["SPC Control Charts", "Process Capability", 
                                          "Pareto Analysis", "Reliability Analysis", "Cost Analysis"])

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = SolarQCDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Dashboard error: {e}")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details"):
            st.code(str(e))

if __name__ == "__main__":
    main()