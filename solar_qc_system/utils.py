"""
Utility Functions for Solar Panel Quality Control System
========================================================

This module contains utility functions and helper methods used throughout
the solar panel quality control system.
"""

import os
import logging
import logging.config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

from config import LOGGING_CONFIG, DATA_DIR, OUTPUT_DIR

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration for the application."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info("Logging system initialized")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
        
    Returns:
    --------
    bool
        True if all required columns are present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True


def calculate_age_in_months(installation_date: Union[str, datetime], 
                          current_date: Optional[Union[str, datetime]] = None) -> float:
    """
    Calculate the age of a solar panel in months.
    
    Parameters:
    -----------
    installation_date : Union[str, datetime]
        Installation date of the solar panel
    current_date : Optional[Union[str, datetime]]
        Current date for age calculation (defaults to today)
        
    Returns:
    --------
    float
        Age in months
    """
    if isinstance(installation_date, str):
        installation_date = pd.to_datetime(installation_date)
    
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)
    
    age_days = (current_date - installation_date).days
    age_months = age_days / 30.44  # Average days per month
    
    return max(0, age_months)


def calculate_expected_power(irradiance: float, 
                           temperature: float,
                           panel_efficiency: float,
                           panel_area: float = 2.0,
                           temperature_coefficient: float = -0.004) -> float:
    """
    Calculate expected power output based on environmental conditions.
    
    Parameters:
    -----------
    irradiance : float
        Solar irradiance in W/m²
    temperature : float
        Panel temperature in °C
    panel_efficiency : float
        Panel efficiency as a percentage
    panel_area : float
        Panel area in m² (default: 2.0)
    temperature_coefficient : float
        Temperature coefficient (%/°C, default: -0.004)
        
    Returns:
    --------
    float
        Expected power output in kW
    """
    # Standard Test Conditions
    stc_irradiance = 1000  # W/m²
    stc_temperature = 25   # °C
    
    # Temperature correction factor
    temp_correction = 1 + temperature_coefficient * (temperature - stc_temperature)
    
    # Power calculation
    power_kw = (irradiance / stc_irradiance) * (panel_efficiency / 100) * panel_area * temp_correction
    
    return max(0, power_kw)


def generate_time_series(start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        freq: str = 'H') -> pd.DatetimeIndex:
    """
    Generate a time series between two dates.
    
    Parameters:
    -----------
    start_date : Union[str, datetime]
        Start date
    end_date : Union[str, datetime]
        End date
    freq : str
        Frequency string (default: 'H' for hourly)
        
    Returns:
    --------
    pd.DatetimeIndex
        Time series index
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def add_seasonal_variation(base_value: float, 
                          timestamp: datetime,
                          amplitude: float = 0.2) -> float:
    """
    Add seasonal variation to a base value.
    
    Parameters:
    -----------
    base_value : float
        Base value to modify
    timestamp : datetime
        Timestamp for seasonal calculation
    amplitude : float
        Amplitude of seasonal variation (default: 0.2)
        
    Returns:
    --------
    float
        Value with seasonal variation
    """
    # Calculate day of year (0-365)
    day_of_year = timestamp.timetuple().tm_yday
    
    # Seasonal factor (peaks in summer, lowest in winter)
    seasonal_factor = amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    return base_value * (1 + seasonal_factor)


def add_daily_variation(base_value: float,
                       timestamp: datetime,
                       amplitude: float = 0.3) -> float:
    """
    Add daily variation to a base value (solar irradiance pattern).
    
    Parameters:
    -----------
    base_value : float
        Base value to modify
    timestamp : datetime
        Timestamp for daily calculation
    amplitude : float
        Amplitude of daily variation (default: 0.3)
        
    Returns:
    --------
    float
        Value with daily variation
    """
    # Hour of day (0-23)
    hour = timestamp.hour
    
    # Daily pattern (peaks at noon, zero at night)
    if 6 <= hour <= 18:  # Daylight hours
        daily_factor = amplitude * np.sin(np.pi * (hour - 6) / 12)
    else:  # Night hours
        daily_factor = 0
    
    return base_value * (1 + daily_factor)


def calculate_process_capability(data: np.ndarray,
                               usl: float,
                               lsl: float,
                               target: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate process capability indices (Cp, Cpk, Pp, Ppk).
    
    Parameters:
    -----------
    data : np.ndarray
        Process data
    usl : float
        Upper Specification Limit
    lsl : float
        Lower Specification Limit
    target : Optional[float]
        Target value (defaults to midpoint of USL and LSL)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing capability indices
    """
    if target is None:
        target = (usl + lsl) / 2
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    
    # Process capability indices
    cp = (usl - lsl) / (6 * std) if std > 0 else np.inf
    cpu = (usl - mean) / (3 * std) if std > 0 else np.inf
    cpl = (mean - lsl) / (3 * std) if std > 0 else np.inf
    cpk = min(cpu, cpl)
    
    # Performance indices (using actual standard deviation)
    pp = cp  # Same as Cp for this implementation
    ppk = cpk  # Same as Cpk for this implementation
    
    return {
        'Cp': cp,
        'Cpk': cpk,
        'Cpu': cpu,
        'Cpl': cpl,
        'Pp': pp,
        'Ppk': ppk,
        'mean': mean,
        'std': std,
        'target': target
    }


def save_dataframe(df: pd.DataFrame, 
                  filename: str,
                  directory: Optional[Path] = None,
                  file_format: str = 'csv') -> Path:
    """
    Save DataFrame to file with proper formatting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Filename (without extension)
    directory : Optional[Path]
        Directory to save to (defaults to DATA_DIR)
    file_format : str
        File format ('csv', 'excel', 'json')
        
    Returns:
    --------
    Path
        Path to saved file
    """
    if directory is None:
        directory = DATA_DIR
    
    # Ensure directory exists
    directory.mkdir(parents=True, exist_ok=True)
    
    # Add appropriate extension
    if file_format.lower() == 'csv':
        # Ensure filename doesn't have double .csv extension
        if filename.endswith('.csv'):
            filepath = directory / filename
        else:
            filepath = directory / f"{filename}.csv"
        df.to_csv(filepath, index=False)
    elif file_format.lower() == 'excel':
        filepath = directory / f"{filename}.xlsx"
        df.to_excel(filepath, index=False)
    elif file_format.lower() == 'json':
        filepath = directory / f"{filename}.json"
        df.to_json(filepath, orient='records', date_format='iso')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    logger.info(f"DataFrame saved to {filepath}")
    return filepath


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from file with automatic format detection.
    
    Parameters:
    -----------
    filepath : Union[str, Path]
        Path to file
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect file format and load accordingly
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif filepath.suffix.lower() == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"DataFrame loaded from {filepath}")
    return df


def create_summary_statistics(df: pd.DataFrame, 
                            numeric_columns: Optional[List[str]] = None) -> Dict:
    """
    Create comprehensive summary statistics for a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    numeric_columns : Optional[List[str]]
        List of numeric columns to analyze (defaults to all numeric columns)
        
    Returns:
    --------
    Dict
        Dictionary containing summary statistics
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max() if 'timestamp' in df.columns else None
        },
        'numeric_summary': df[numeric_columns].describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary


def format_number(value: float, decimal_places: int = 2) -> str:
    """
    Format a number for display with appropriate decimal places.
    
    Parameters:
    -----------
    value : float
        Number to format
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1000000:
        return f"{value/1000000:.{decimal_places}f}M"
    elif abs(value) >= 1000:
        return f"{value/1000:.{decimal_places}f}K"
    else:
        return f"{value:.{decimal_places}f}"


def create_color_palette(n_colors: int) -> List[str]:
    """
    Create a color palette for visualizations.
    
    Parameters:
    -----------
    n_colors : int
        Number of colors needed
        
    Returns:
    --------
    List[str]
        List of hex color codes
    """
    # Professional color palette
    base_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Repeat colors if needed
    colors = (base_colors * ((n_colors // len(base_colors)) + 1))[:n_colors]
    
    return colors