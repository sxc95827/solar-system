"""
Configuration file for Solar Panel Quality Control System
=========================================================

This module contains all configuration parameters, constants, and settings
used throughout the solar panel quality control system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Solar Panel Technical Specifications
SOLAR_PANEL_SPECS = {
    "efficiency_range": (15.0, 22.0),  # Efficiency percentage range
    "power_rating_kw": (0.3, 0.5),     # Power rating range in kW
    "degradation_rate": 0.007,          # Annual degradation rate (0.7%/year)
    "expected_lifespan_years": 25,      # Expected operational lifespan
    "temperature_coefficient": -0.004,  # Power temperature coefficient (%/°C)
    "irradiance_stc": 1000,            # Standard Test Conditions irradiance (W/m²)
    "temperature_stc": 25,             # Standard Test Conditions temperature (°C)
}

# Environmental Parameters
ENVIRONMENTAL_PARAMS = {
    "temperature_range": (-10, 45),     # Ambient temperature range (°C)
    "humidity_range": (20, 95),         # Relative humidity range (%)
    "wind_speed_range": (0, 15),        # Wind speed range (m/s)
    "irradiance_range": (0, 1200),      # Solar irradiance range (W/m²)
    "dust_levels": [1, 2, 3, 4, 5],     # Dust level categories
    "weather_conditions": ["sunny", "partly_cloudy", "cloudy", "rainy"],
}

# Failure Types and Probabilities
FAILURE_TYPES = {
    "hot_spots": {"probability": 0.25, "severity_range": (2, 4), "repair_cost_range": (200, 800)},
    "micro_cracks": {"probability": 0.20, "severity_range": (1, 3), "repair_cost_range": (150, 600)},
    "connector_failure": {"probability": 0.15, "severity_range": (2, 5), "repair_cost_range": (100, 400)},
    "inverter_issues": {"probability": 0.15, "severity_range": (3, 5), "repair_cost_range": (500, 2000)},
    "soiling": {"probability": 0.10, "severity_range": (1, 2), "repair_cost_range": (50, 200)},
    "corrosion": {"probability": 0.08, "severity_range": (2, 4), "repair_cost_range": (300, 1200)},
    "electrical_degradation": {"probability": 0.07, "severity_range": (3, 5), "repair_cost_range": (400, 1500)},
}

# Six Sigma Control Limits
CONTROL_LIMITS = {
    "efficiency": {
        "ucl_factor": 3,  # Upper Control Limit factor (3-sigma)
        "lcl_factor": 3,  # Lower Control Limit factor (3-sigma)
        "target": 18.5,   # Target efficiency percentage
        "usl": 22.0,      # Upper Specification Limit
        "lsl": 15.0,      # Lower Specification Limit
    },
    "power_output": {
        "ucl_factor": 3,
        "lcl_factor": 3,
        "target": 0.4,    # Target power output in kW
        "usl": 0.5,       # Upper Specification Limit
        "lsl": 0.3,       # Lower Specification Limit
    }
}

# Data Generation Parameters
DATA_GENERATION = {
    "num_panels": 1000,                 # Number of solar panels to simulate
    "num_sites": 3,                     # Number of solar farm sites
    "simulation_months": 12,            # Simulation period in months
    "hourly_records": True,             # Generate hourly performance data
    "failure_rate_per_year": 0.05,      # Annual failure rate (5%)
    "maintenance_interval_months": 6,    # Regular maintenance interval
}

# Site Information
SITES = {
    "SITE_A": {
        "name": "Albany Solar Farm",
        "location": "Albany, NY",
        "capacity_mw": 50,
        "num_panels": 400,
        "installation_year": 2019,
        "climate_zone": "continental"
    },
    "SITE_B": {
        "name": "Buffalo Solar Farm", 
        "location": "Buffalo, NY",
        "capacity_mw": 75,
        "num_panels": 400,
        "installation_year": 2020,
        "climate_zone": "continental"
    },
    "SITE_C": {
        "name": "Syracuse Solar Farm",
        "location": "Syracuse, NY", 
        "capacity_mw": 60,
        "num_panels": 200,
        "installation_year": 2021,
        "climate_zone": "continental"
    }
}

# Panel Types
PANEL_TYPES = {
    "monocrystalline": {
        "efficiency_range": (18, 22),
        "cost_per_kw": 1200,
        "degradation_rate": 0.006,
        "market_share": 0.6
    },
    "polycrystalline": {
        "efficiency_range": (15, 18),
        "cost_per_kw": 1000,
        "degradation_rate": 0.008,
        "market_share": 0.3
    },
    "thin_film": {
        "efficiency_range": (12, 16),
        "cost_per_kw": 800,
        "degradation_rate": 0.010,
        "market_share": 0.1
    }
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "page_title": "Solar Panel Quality Control Dashboard",
    "page_icon": "☀️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#FF6B35",
        "backgroundColor": "#FFFFFF", 
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730"
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "solar_qc.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}