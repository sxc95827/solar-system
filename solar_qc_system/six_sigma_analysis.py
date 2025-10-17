"""
Six Sigma Analysis Module for Solar Panel Quality Control System

This module implements comprehensive Six Sigma statistical analysis tools
for monitoring and improving solar panel performance and reliability.

Key Features:
- Statistical Process Control (SPC) charts
- Process capability analysis (Cp, Cpk, Pp, Ppk)
- Pareto analysis for failure modes
- MTBF/MTTR reliability analysis
- Cost analysis and ROI calculations
- Control limit calculations
- Trend analysis and forecasting

Author: Solar QC Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

from config import CONTROL_LIMITS, FAILURE_TYPES
from utils import setup_logging, calculate_process_capability

# Setup logging
logger = setup_logging()

class SPCAnalyzer:
    """Statistical Process Control analyzer for solar panel performance"""
    
    def __init__(self):
        self.control_limits = CONTROL_LIMITS
        
    def calculate_control_limits(self, data: pd.Series, chart_type: str = 'xbar') -> Dict[str, float]:
        """
        Calculate control limits for SPC charts
        
        Args:
            data: Time series data
            chart_type: Type of control chart ('xbar', 'r', 'individuals')
            
        Returns:
            Dictionary with UCL, LCL, and centerline values
        """
        try:
            if chart_type == 'xbar':
                # X-bar chart (subgroup means)
                centerline = data.mean()
                std_dev = data.std()
                
                # A2 factor for subgroup size (assuming n=5)
                A2 = 0.577
                
                ucl = centerline + A2 * std_dev
                lcl = centerline - A2 * std_dev
                
            elif chart_type == 'individuals':
                # Individual measurements chart
                centerline = data.mean()
                moving_range = data.diff().abs().mean()
                
                # Constants for individuals chart
                ucl = centerline + 2.66 * moving_range
                lcl = centerline - 2.66 * moving_range
                
            elif chart_type == 'r':
                # Range chart
                ranges = data.rolling(window=5).max() - data.rolling(window=5).min()
                centerline = ranges.mean()
                
                # D3 and D4 factors for subgroup size n=5
                D3, D4 = 0, 2.114
                
                ucl = D4 * centerline
                lcl = D3 * centerline
                
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
                
            return {
                'UCL': ucl,
                'LCL': lcl,
                'CL': centerline,
                'std_dev': data.std()
            }
            
        except Exception as e:
            logger.error(f"Error calculating control limits: {e}")
            return {}
    
    def create_control_chart(self, data: pd.DataFrame, metric: str, 
                           chart_type: str = 'individuals') -> go.Figure:
        """
        Create interactive SPC control chart
        
        Args:
            data: DataFrame with timestamp and metric columns
            metric: Column name for the metric to chart
            chart_type: Type of control chart
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate control limits
            limits = self.calculate_control_limits(data[metric], chart_type)
            
            if not limits:
                raise ValueError("Could not calculate control limits")
            
            # Create figure
            fig = go.Figure()
            
            # Add data points
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[metric],
                mode='lines+markers',
                name=f'{metric}',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Add control limits
            fig.add_hline(y=limits['UCL'], line_dash="dash", line_color="red",
                         annotation_text=f"UCL: {limits['UCL']:.2f}")
            fig.add_hline(y=limits['LCL'], line_dash="dash", line_color="red",
                         annotation_text=f"LCL: {limits['LCL']:.2f}")
            fig.add_hline(y=limits['CL'], line_dash="solid", line_color="green",
                         annotation_text=f"CL: {limits['CL']:.2f}")
            
            # Identify out-of-control points
            ooc_points = (data[metric] > limits['UCL']) | (data[metric] < limits['LCL'])
            if ooc_points.any():
                fig.add_trace(go.Scatter(
                    x=data.index[ooc_points],
                    y=data[metric][ooc_points],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(color='red', size=8, symbol='x')
                ))
            
            # Update layout
            fig.update_layout(
                title=f'SPC Control Chart - {metric}',
                xaxis_title='Time',
                yaxis_title=metric,
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating control chart: {e}")
            return go.Figure()
    
    def detect_patterns(self, data: pd.Series, limits: Dict[str, float]) -> Dict[str, List[int]]:
        """
        Detect special cause patterns in control chart data
        
        Args:
            data: Time series data
            limits: Control limits dictionary
            
        Returns:
            Dictionary of pattern types and their indices
        """
        patterns = {
            'out_of_control': [],
            'seven_consecutive': [],
            'trend': [],
            'two_of_three': []
        }
        
        try:
            # Rule 1: Points beyond control limits
            ooc_mask = (data > limits['UCL']) | (data < limits['LCL'])
            patterns['out_of_control'] = data.index[ooc_mask].tolist()
            
            # Rule 2: Seven consecutive points on same side of centerline
            centerline = limits['CL']
            above_center = (data > centerline).astype(int)
            below_center = (data < centerline).astype(int)
            
            for i in range(len(data) - 6):
                if above_center.iloc[i:i+7].sum() == 7 or below_center.iloc[i:i+7].sum() == 7:
                    patterns['seven_consecutive'].extend(range(i, i+7))
            
            # Rule 3: Seven consecutive increasing or decreasing points
            for i in range(len(data) - 6):
                window = data.iloc[i:i+7]
                if all(window.iloc[j] < window.iloc[j+1] for j in range(6)) or \
                   all(window.iloc[j] > window.iloc[j+1] for j in range(6)):
                    patterns['trend'].extend(range(i, i+7))
            
            # Rule 4: Two out of three consecutive points beyond 2-sigma
            sigma = limits['std_dev']
            upper_2sigma = centerline + 2 * sigma
            lower_2sigma = centerline - 2 * sigma
            
            beyond_2sigma = (data > upper_2sigma) | (data < lower_2sigma)
            
            for i in range(len(data) - 2):
                if beyond_2sigma.iloc[i:i+3].sum() >= 2:
                    patterns['two_of_three'].extend(range(i, i+3))
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns

class ProcessCapabilityAnalyzer:
    """Process capability analysis for solar panel performance"""
    
    def __init__(self):
        self.capability_thresholds = {
            'excellent': 1.67,
            'adequate': 1.33,
            'marginal': 1.0,
            'inadequate': 0.67
        }
    
    def calculate_capability_indices(self, data: pd.Series, 
                                   usl: float, lsl: float) -> Dict[str, float]:
        """
        Calculate process capability indices (Cp, Cpk, Pp, Ppk)
        
        Args:
            data: Process data
            usl: Upper specification limit
            lsl: Lower specification limit
            
        Returns:
            Dictionary with capability indices
        """
        try:
            mean = data.mean()
            std_dev = data.std()
            
            # Cp (Process Capability)
            cp = (usl - lsl) / (6 * std_dev)
            
            # Cpk (Process Capability Index)
            cpu = (usl - mean) / (3 * std_dev)
            cpl = (mean - lsl) / (3 * std_dev)
            cpk = min(cpu, cpl)
            
            # Pp (Process Performance)
            pp = (usl - lsl) / (6 * std_dev)
            
            # Ppk (Process Performance Index)
            ppu = (usl - mean) / (3 * std_dev)
            ppl = (mean - lsl) / (3 * std_dev)
            ppk = min(ppu, ppl)
            
            return {
                'Cp': cp,
                'Cpk': cpk,
                'Pp': pp,
                'Ppk': ppk,
                'CPU': cpu,
                'CPL': cpl,
                'PPU': ppu,
                'PPL': ppl,
                'mean': mean,
                'std_dev': std_dev
            }
            
        except Exception as e:
            logger.error(f"Error calculating capability indices: {e}")
            return {}
    
    def create_capability_histogram(self, data: pd.Series, usl: float, 
                                  lsl: float, target: Optional[float] = None) -> go.Figure:
        """
        Create process capability histogram with specification limits
        
        Args:
            data: Process data
            usl: Upper specification limit
            lsl: Lower specification limit
            target: Target value (optional)
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate capability indices
            indices = self.calculate_capability_indices(data, usl, lsl)
            
            if not indices:
                raise ValueError("Could not calculate capability indices")
            
            # Create histogram
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                name='Data Distribution',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Add specification limits
            fig.add_vline(x=usl, line_dash="dash", line_color="red",
                         annotation_text=f"USL: {usl}")
            fig.add_vline(x=lsl, line_dash="dash", line_color="red",
                         annotation_text=f"LSL: {lsl}")
            
            # Add target if provided
            if target is not None:
                fig.add_vline(x=target, line_dash="solid", line_color="green",
                             annotation_text=f"Target: {target}")
            
            # Add normal curve overlay
            x_range = np.linspace(data.min(), data.max(), 100)
            normal_curve = norm.pdf(x_range, indices['mean'], indices['std_dev'])
            
            # Scale normal curve to match histogram
            hist_counts, _ = np.histogram(data, bins=30)
            scale_factor = hist_counts.max() / normal_curve.max()
            normal_curve *= scale_factor
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_curve,
                mode='lines',
                name='Normal Curve',
                line=dict(color='orange', width=2)
            ))
            
            # Update layout with capability indices
            capability_text = f"Cp: {indices['Cp']:.3f}<br>Cpk: {indices['Cpk']:.3f}<br>Pp: {indices['Pp']:.3f}<br>Ppk: {indices['Ppk']:.3f}"
            
            fig.update_layout(
                title='Process Capability Analysis',
                xaxis_title='Value',
                yaxis_title='Frequency',
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref='paper',
                        yref='paper',
                        text=capability_text,
                        showarrow=False,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating capability histogram: {e}")
            return go.Figure()

class ParetoAnalyzer:
    """Pareto analysis for failure modes and defects"""
    
    def create_pareto_chart(self, failure_data: pd.DataFrame, 
                          category_col: str, count_col: str = None) -> go.Figure:
        """
        Create Pareto chart for failure analysis
        
        Args:
            failure_data: DataFrame with failure data
            category_col: Column name for failure categories
            count_col: Column name for counts (optional, will count occurrences if None)
            
        Returns:
            Plotly figure object
        """
        try:
            # Count occurrences if count_col not provided
            if count_col is None:
                pareto_data = failure_data[category_col].value_counts().reset_index()
                pareto_data.columns = [category_col, 'count']
            else:
                pareto_data = failure_data.groupby(category_col)[count_col].sum().reset_index()
                pareto_data.columns = [category_col, 'count']
            
            # Sort by count descending
            pareto_data = pareto_data.sort_values('count', ascending=False)
            
            # Calculate cumulative percentage
            pareto_data['cumulative'] = pareto_data['count'].cumsum()
            pareto_data['cumulative_pct'] = (pareto_data['cumulative'] / pareto_data['count'].sum()) * 100
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=pareto_data[category_col],
                    y=pareto_data['count'],
                    name='Count',
                    marker_color='lightblue'
                ),
                secondary_y=False
            )
            
            # Add cumulative percentage line
            fig.add_trace(
                go.Scatter(
                    x=pareto_data[category_col],
                    y=pareto_data['cumulative_pct'],
                    mode='lines+markers',
                    name='Cumulative %',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ),
                secondary_y=True
            )
            
            # Add 80% line
            fig.add_hline(y=80, line_dash="dash", line_color="orange",
                         annotation_text="80%", secondary_y=True)
            
            # Update layout
            fig.update_xaxes(title_text="Failure Category")
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
            
            fig.update_layout(
                title='Pareto Chart - Failure Analysis',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Pareto chart: {e}")
            return go.Figure()

class ReliabilityAnalyzer:
    """Reliability analysis for MTBF/MTTR calculations"""
    
    def calculate_mtbf_mttr(self, failure_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Mean Time Between Failures and Mean Time To Repair
        
        Args:
            failure_data: DataFrame with failure events
            
        Returns:
            Dictionary with reliability metrics
        """
        try:
            # Convert timestamps to datetime if needed
            if 'failure_date' in failure_data.columns:
                failure_data['failure_date'] = pd.to_datetime(failure_data['failure_date'])
            
            # Calculate time between failures
            failure_data_sorted = failure_data.sort_values('failure_date')
            time_between_failures = failure_data_sorted['failure_date'].diff().dt.total_seconds() / 3600  # hours
            
            # MTBF calculation
            mtbf = time_between_failures.mean()
            
            # MTTR calculation
            if 'repair_time_hours' in failure_data.columns:
                mttr = failure_data['repair_time_hours'].mean()
            else:
                mttr = None
            
            # Availability calculation
            if mttr is not None:
                availability = mtbf / (mtbf + mttr) * 100
            else:
                availability = None
            
            # Failure rate (failures per hour)
            failure_rate = 1 / mtbf if mtbf > 0 else 0
            
            return {
                'MTBF_hours': mtbf,
                'MTTR_hours': mttr,
                'availability_percent': availability,
                'failure_rate_per_hour': failure_rate,
                'total_failures': len(failure_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MTBF/MTTR: {e}")
            return {}
    
    def create_reliability_trend(self, failure_data: pd.DataFrame) -> go.Figure:
        """
        Create reliability trend analysis chart
        
        Args:
            failure_data: DataFrame with failure events
            
        Returns:
            Plotly figure object
        """
        try:
            # Group failures by month
            failure_data['failure_date'] = pd.to_datetime(failure_data['failure_date'])
            monthly_failures = failure_data.groupby(failure_data['failure_date'].dt.to_period('M')).size()
            
            # Calculate rolling MTBF
            rolling_mtbf = []
            for i in range(1, len(monthly_failures) + 1):
                subset = monthly_failures.iloc[:i]
                if len(subset) > 1:
                    time_span = (subset.index[-1] - subset.index[0]).n + 1  # months
                    mtbf = (time_span * 30 * 24) / subset.sum()  # hours
                    rolling_mtbf.append(mtbf)
                else:
                    rolling_mtbf.append(None)
            
            # Create figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add monthly failures
            fig.add_trace(
                go.Bar(
                    x=[str(period) for period in monthly_failures.index],
                    y=monthly_failures.values,
                    name='Monthly Failures',
                    marker_color='lightcoral'
                ),
                secondary_y=False
            )
            
            # Add rolling MTBF
            fig.add_trace(
                go.Scatter(
                    x=[str(period) for period in monthly_failures.index],
                    y=rolling_mtbf,
                    mode='lines+markers',
                    name='Rolling MTBF',
                    line=dict(color='blue', width=2)
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_xaxes(title_text="Month")
            fig.update_yaxes(title_text="Number of Failures", secondary_y=False)
            fig.update_yaxes(title_text="MTBF (Hours)", secondary_y=True)
            
            fig.update_layout(
                title='Reliability Trend Analysis',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating reliability trend: {e}")
            return go.Figure()

class CostAnalyzer:
    """Cost analysis and ROI calculations"""
    
    def calculate_failure_costs(self, failure_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various failure-related costs
        
        Args:
            failure_data: DataFrame with failure events and costs
            
        Returns:
            Dictionary with cost metrics
        """
        try:
            costs = {}
            
            # Total repair costs
            if 'repair_cost' in failure_data.columns:
                costs['total_repair_cost'] = failure_data['repair_cost'].sum()
                costs['avg_repair_cost'] = failure_data['repair_cost'].mean()
                costs['max_repair_cost'] = failure_data['repair_cost'].max()
            
            # Downtime costs
            if 'downtime_hours' in failure_data.columns:
                # Assume $100/hour downtime cost
                downtime_cost_per_hour = 100
                costs['total_downtime_cost'] = (failure_data['downtime_hours'] * downtime_cost_per_hour).sum()
                costs['avg_downtime_cost'] = (failure_data['downtime_hours'] * downtime_cost_per_hour).mean()
            
            # Cost by failure type
            if 'failure_type' in failure_data.columns and 'repair_cost' in failure_data.columns:
                cost_by_type = failure_data.groupby('failure_type')['repair_cost'].agg(['sum', 'mean', 'count'])
                costs['cost_by_failure_type'] = cost_by_type.to_dict()
            
            # Monthly cost trend
            if 'failure_date' in failure_data.columns and 'repair_cost' in failure_data.columns:
                failure_data['failure_date'] = pd.to_datetime(failure_data['failure_date'])
                monthly_costs = failure_data.groupby(failure_data['failure_date'].dt.to_period('M'))['repair_cost'].sum()
                costs['monthly_cost_trend'] = monthly_costs.to_dict()
            
            return costs
            
        except Exception as e:
            logger.error(f"Error calculating failure costs: {e}")
            return {}
    
    def create_cost_analysis_dashboard(self, failure_data: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive cost analysis dashboard
        
        Args:
            failure_data: DataFrame with failure events and costs
            
        Returns:
            Plotly figure object with subplots
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cost by Failure Type', 'Monthly Cost Trend', 
                              'Cost vs Downtime', 'Cumulative Cost'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. Cost by failure type
            if 'failure_type' in failure_data.columns and 'repair_cost' in failure_data.columns:
                cost_by_type = failure_data.groupby('failure_type')['repair_cost'].sum().sort_values(ascending=False)
                
                fig.add_trace(
                    go.Bar(x=cost_by_type.index, y=cost_by_type.values, name='Cost by Type'),
                    row=1, col=1
                )
            
            # 2. Monthly cost trend
            if 'failure_date' in failure_data.columns and 'repair_cost' in failure_data.columns:
                failure_data['failure_date'] = pd.to_datetime(failure_data['failure_date'])
                monthly_costs = failure_data.groupby(failure_data['failure_date'].dt.to_period('M'))['repair_cost'].sum()
                
                fig.add_trace(
                    go.Scatter(x=[str(p) for p in monthly_costs.index], y=monthly_costs.values, 
                             mode='lines+markers', name='Monthly Cost'),
                    row=1, col=2
                )
            
            # 3. Cost vs Downtime scatter
            if 'repair_cost' in failure_data.columns and 'downtime_hours' in failure_data.columns:
                fig.add_trace(
                    go.Scatter(x=failure_data['downtime_hours'], y=failure_data['repair_cost'],
                             mode='markers', name='Cost vs Downtime'),
                    row=2, col=1
                )
            
            # 4. Cumulative cost
            if 'failure_date' in failure_data.columns and 'repair_cost' in failure_data.columns:
                failure_data_sorted = failure_data.sort_values('failure_date')
                cumulative_cost = failure_data_sorted['repair_cost'].cumsum()
                
                fig.add_trace(
                    go.Scatter(x=failure_data_sorted['failure_date'], y=cumulative_cost,
                             mode='lines', name='Cumulative Cost'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Cost Analysis Dashboard")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cost analysis dashboard: {e}")
            return go.Figure()

class SixSigmaAnalysisEngine:
    """Main Six Sigma analysis engine that coordinates all analyzers"""
    
    def __init__(self):
        self.spc_analyzer = SPCAnalyzer()
        self.capability_analyzer = ProcessCapabilityAnalyzer()
        self.pareto_analyzer = ParetoAnalyzer()
        self.reliability_analyzer = ReliabilityAnalyzer()
        self.cost_analyzer = CostAnalyzer()
        
    def run_comprehensive_analysis(self, panel_data: pd.DataFrame, 
                                 failure_data: pd.DataFrame, 
                                 environmental_data: pd.DataFrame) -> Dict:
        """
        Run comprehensive Six Sigma analysis on all datasets
        
        Args:
            panel_data: Panel performance data
            failure_data: Failure events data
            environmental_data: Environmental monitoring data
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        try:
            logger.info("Starting comprehensive Six Sigma analysis...")
            
            # SPC Analysis
            if 'efficiency' in panel_data.columns:
                results['spc_efficiency'] = self.spc_analyzer.create_control_chart(
                    panel_data, 'efficiency', 'individuals'
                )
                
                # Control limits for efficiency
                limits = self.spc_analyzer.calculate_control_limits(panel_data['efficiency'])
                results['efficiency_limits'] = limits
                
                # Pattern detection
                patterns = self.spc_analyzer.detect_patterns(panel_data['efficiency'], limits)
                results['efficiency_patterns'] = patterns
            
            # Process Capability Analysis
            if 'efficiency' in panel_data.columns:
                # Assuming efficiency should be between 15% and 25%
                usl, lsl = 25.0, 15.0
                results['capability_analysis'] = self.capability_analyzer.create_capability_histogram(
                    panel_data['efficiency'], usl, lsl, target=20.0
                )
                
                capability_indices = self.capability_analyzer.calculate_capability_indices(
                    panel_data['efficiency'], usl, lsl
                )
                results['capability_indices'] = capability_indices
            
            # Pareto Analysis
            if 'failure_type' in failure_data.columns:
                results['pareto_failures'] = self.pareto_analyzer.create_pareto_chart(
                    failure_data, 'failure_type'
                )
            
            # Reliability Analysis
            reliability_metrics = self.reliability_analyzer.calculate_mtbf_mttr(failure_data)
            results['reliability_metrics'] = reliability_metrics
            
            results['reliability_trend'] = self.reliability_analyzer.create_reliability_trend(failure_data)
            
            # Cost Analysis
            cost_metrics = self.cost_analyzer.calculate_failure_costs(failure_data)
            results['cost_metrics'] = cost_metrics
            
            results['cost_dashboard'] = self.cost_analyzer.create_cost_analysis_dashboard(failure_data)
            
            logger.info("Comprehensive Six Sigma analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def generate_executive_summary(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate executive summary of Six Sigma analysis
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Dictionary with summary metrics and recommendations
        """
        summary = {}
        
        try:
            # Process Capability Summary
            if 'capability_indices' in analysis_results:
                indices = analysis_results['capability_indices']
                cpk = indices.get('Cpk', 0)
                
                if cpk >= 1.67:
                    capability_status = "Excellent"
                elif cpk >= 1.33:
                    capability_status = "Adequate"
                elif cpk >= 1.0:
                    capability_status = "Marginal"
                else:
                    capability_status = "Inadequate"
                
                summary['process_capability'] = f"Process capability is {capability_status} (Cpk = {cpk:.3f})"
            
            # Reliability Summary
            if 'reliability_metrics' in analysis_results:
                metrics = analysis_results['reliability_metrics']
                mtbf = metrics.get('MTBF_hours', 0)
                availability = metrics.get('availability_percent', 0)
                
                summary['reliability'] = f"MTBF: {mtbf:.1f} hours, Availability: {availability:.1f}%"
            
            # Cost Summary
            if 'cost_metrics' in analysis_results:
                costs = analysis_results['cost_metrics']
                total_cost = costs.get('total_repair_cost', 0)
                avg_cost = costs.get('avg_repair_cost', 0)
                
                summary['cost_impact'] = f"Total repair cost: ${total_cost:,.2f}, Average: ${avg_cost:,.2f}"
            
            # Control Chart Summary
            if 'efficiency_patterns' in analysis_results:
                patterns = analysis_results['efficiency_patterns']
                ooc_count = len(patterns.get('out_of_control', []))
                
                if ooc_count > 0:
                    summary['process_control'] = f"Warning: {ooc_count} out-of-control points detected"
                else:
                    summary['process_control'] = "Process is in statistical control"
            
            # Recommendations
            recommendations = []
            
            if 'capability_indices' in analysis_results:
                cpk = analysis_results['capability_indices'].get('Cpk', 0)
                if cpk < 1.33:
                    recommendations.append("Improve process capability through variation reduction")
            
            if 'reliability_metrics' in analysis_results:
                mtbf = analysis_results['reliability_metrics'].get('MTBF_hours', 0)
                if mtbf < 1000:  # Less than 1000 hours
                    recommendations.append("Implement preventive maintenance to improve MTBF")
            
            if 'efficiency_patterns' in analysis_results:
                ooc_count = len(analysis_results['efficiency_patterns'].get('out_of_control', []))
                if ooc_count > 0:
                    recommendations.append("Investigate and eliminate special causes of variation")
            
            summary['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            summary['error'] = str(e)
        
        return summary