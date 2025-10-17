"""
Solar Panel Quality Control System
==================================

A comprehensive quality control system for utility-scale solar power plants
using Six Sigma methodologies to track and mitigate solar panel failures.

Developed for NYSERDA (New York State Energy Research & Development Authority)
to better understand the lifespan remaining for existing utility-scale solar farms.

Modules:
--------
- data_generator: Synthetic data generation for testing and demonstration
- six_sigma_analysis: Core Six Sigma statistical analysis methods
- dashboard: Streamlit-based interactive dashboard
- utils: Utility functions and helpers

Author: Six Sigma Hackathon Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Six Sigma Hackathon Team"
__email__ = "team@sixsigmahackathon.com"

# Import main modules
from . import data_generator
from . import six_sigma_analysis
from . import utils

__all__ = [
    'data_generator',
    'six_sigma_analysis', 
    'utils'
]