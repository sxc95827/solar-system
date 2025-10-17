"""
Synthetic Data Generator for Solar Panel Quality Control System
==============================================================

This module generates realistic synthetic datasets for testing and demonstrating
the solar panel quality control system. It creates three complementary datasets:

1. Panel Performance Data - Real-time monitoring data for SPC analysis
2. Failure Events Data - Historical failure records for reliability analysis  
3. Environmental Data - Weather and environmental monitoring data

The generated data follows realistic patterns based on solar panel physics,
industry standards, and Six Sigma quality control requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from config import (
    SOLAR_PANEL_SPECS, ENVIRONMENTAL_PARAMS, FAILURE_TYPES, 
    DATA_GENERATION, SITES, PANEL_TYPES, DATA_DIR
)
from utils import (
    calculate_expected_power, add_seasonal_variation, add_daily_variation,
    calculate_age_in_months, save_dataframe, setup_logging
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class SolarDataGenerator:
    """
    Comprehensive synthetic data generator for solar panel quality control system.
    
    This class generates three types of datasets:
    - Panel performance monitoring data
    - Failure event records
    - Environmental monitoring data
    
    All data is generated with realistic patterns and correlations to support
    Six Sigma analysis and quality control demonstrations.
    """
    
    def __init__(self, 
                 start_date: str = "2023-01-01",
                 end_date: str = "2023-12-31",
                 random_seed: int = 42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        start_date : str
            Start date for data generation (YYYY-MM-DD)
        end_date : str
            End date for data generation (YYYY-MM-DD)
        random_seed : int
            Random seed for reproducible results
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Generate time series
        # Fix the deprecated 'H' frequency warning
        self.hourly_timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='h'  # Use 'h' instead of 'H'
        )
        
        logger.info(f"Data generator initialized for period {start_date} to {end_date}")
        logger.info(f"Total time points: {len(self.hourly_timestamps)}")
    
    def generate_panel_performance_data(self) -> pd.DataFrame:
        """
        Generate synthetic solar panel performance monitoring data.
        
        This dataset contains hourly performance metrics for all panels across
        all sites, including power output, efficiency, temperature, and other
        key performance indicators needed for SPC analysis.
        
        Returns:
        --------
        pd.DataFrame
            Panel performance data with columns:
            - panel_id, timestamp, power_output_kw, efficiency_percent,
            - panel_temperature_c, irradiance_w_m2, panel_age_months,
            - maintenance_flag, site_id, panel_type, installation_date
        """
        logger.info("Generating panel performance data...")
        
        performance_data = []
        
        # Calculate total panels for progress tracking
        total_panels = sum(site_info['num_panels'] for site_info in SITES.values())
        panel_count = 0
        
        # Generate data for each site
        for site_id, site_info in SITES.items():
            num_panels = site_info['num_panels']
            installation_year = site_info['installation_year']
            
            logger.info(f"Processing site {site_id} with {num_panels} panels...")
            
            # Generate panel IDs for this site
            panel_ids = [f"{site_id}_P{i:03d}" for i in range(1, num_panels + 1)]
            
            for i, panel_id in enumerate(panel_ids):
                panel_count += 1
                if panel_count % 50 == 0 or panel_count == total_panels:
                    progress = (panel_count / total_panels) * 100
                    logger.info(f"Progress: {panel_count}/{total_panels} panels ({progress:.1f}%)")
                # Assign panel type based on market share
                panel_type = np.random.choice(
                    list(PANEL_TYPES.keys()),
                    p=[PANEL_TYPES[pt]['market_share'] for pt in PANEL_TYPES.keys()]
                )
                
                # Installation date (random within the year)
                installation_date = pd.to_datetime(f"{installation_year}-01-01") + \
                                  timedelta(days=np.random.randint(0, 365))
                
                # Panel specifications
                panel_specs = PANEL_TYPES[panel_type]
                base_efficiency = np.random.uniform(*panel_specs['efficiency_range'])
                degradation_rate = panel_specs['degradation_rate']
                
                # Generate hourly data for this panel (sample subset for faster generation)
                # For demo purposes, generate data every 6 hours instead of every hour
                valid_timestamps = [ts for ts in self.hourly_timestamps if ts >= installation_date]
                sample_timestamps = valid_timestamps[::6]  # Every 6th timestamp
                
                for j, timestamp in enumerate(sample_timestamps):
                    # Progress logging for large datasets
                    if len(sample_timestamps) > 100 and j % 100 == 0:
                        ts_progress = (j / len(sample_timestamps)) * 100
                        logger.debug(f"Panel {panel_id}: {j}/{len(sample_timestamps)} timestamps ({ts_progress:.1f}%)")
                    
                    # Calculate panel age
                    age_months = calculate_age_in_months(installation_date, timestamp)
                    
                    # Apply degradation over time
                    current_efficiency = base_efficiency * (1 - degradation_rate * (age_months / 12))
                    
                    # Generate environmental conditions
                    irradiance = self._generate_irradiance(timestamp)
                    ambient_temp = self._generate_temperature(timestamp)
                    
                    # Panel temperature (higher than ambient due to solar heating)
                    panel_temperature = ambient_temp + np.random.normal(15, 5)
                    panel_temperature = max(ambient_temp, panel_temperature)
                    
                    # Calculate expected power output
                    expected_power = calculate_expected_power(
                        irradiance=irradiance,
                        temperature=panel_temperature,
                        panel_efficiency=current_efficiency,
                        panel_area=2.0
                    )
                    
                    # Add realistic noise and variations
                    power_noise = np.random.normal(0, 0.02)  # 2% noise
                    actual_power = max(0, expected_power * (1 + power_noise))
                    
                    # Efficiency calculation
                    if irradiance > 100:  # Only calculate efficiency during daylight
                        actual_efficiency = (actual_power / 2.0) / (irradiance / 1000) * 100
                    else:
                        actual_efficiency = 0
                    
                    # Maintenance flag (scheduled maintenance every 6 months)
                    maintenance_flag = 1 if (age_months % 6 < 0.1 and 
                                           timestamp.hour == 10) else 0
                    
                    # Add some random maintenance events
                    if np.random.random() < 0.001:  # 0.1% chance per hour
                        maintenance_flag = 1
                    
                    performance_data.append({
                        'panel_id': panel_id,
                        'timestamp': timestamp,
                        'power_output_kw': round(actual_power, 3),
                        'efficiency_percent': round(actual_efficiency, 2),
                        'panel_temperature_c': round(panel_temperature, 1),
                        'irradiance_w_m2': round(irradiance, 1),
                        'panel_age_months': round(age_months, 1),
                        'maintenance_flag': maintenance_flag,
                        'site_id': site_id,
                        'panel_type': panel_type,
                        'installation_date': installation_date.date()
                    })
        
        df = pd.DataFrame(performance_data)
        logger.info(f"Generated {len(df):,} performance records for {len(df['panel_id'].unique())} panels")
        logger.info(f"Data generation completed. Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def generate_failure_events_data(self) -> pd.DataFrame:
        """
        Generate synthetic failure event records.
        
        This dataset contains historical failure events with details about
        failure types, severity, costs, and repair information needed for
        reliability analysis and FMEA.
        
        Returns:
        --------
        pd.DataFrame
            Failure events data with columns:
            - event_id, panel_id, failure_date, failure_type, severity_level,
            - detection_method, repair_cost_usd, downtime_hours, repair_date,
            - root_cause, preventable, warranty_covered
        """
        logger.info("Generating failure events data...")
        
        failure_events = []
        event_id = 1
        
        # Get all panel IDs from sites
        all_panels = []
        for site_id, site_info in SITES.items():
            num_panels = site_info['num_panels']
            panel_ids = [f"{site_id}_P{i:03d}" for i in range(1, num_panels + 1)]
            all_panels.extend([(panel_id, site_id, site_info['installation_year']) 
                             for panel_id in panel_ids])
        
        # Generate failures for each panel
        for panel_id, site_id, installation_year in all_panels:
            installation_date = pd.to_datetime(f"{installation_year}-01-01") + \
                              timedelta(days=np.random.randint(0, 365))
            
            # Calculate number of failures based on panel age and failure rate
            panel_age_years = (self.end_date - installation_date).days / 365.25
            expected_failures = max(1, int(panel_age_years * DATA_GENERATION['failure_rate_per_year']))
            
            # Add some randomness
            num_failures = np.random.poisson(expected_failures)
            
            if num_failures == 0:
                continue
            
            # Generate failure dates
            failure_dates = []
            current_date = installation_date + timedelta(days=30)  # No failures in first month
            
            for _ in range(num_failures):
                # Time between failures (exponential distribution)
                days_to_next_failure = np.random.exponential(365 / DATA_GENERATION['failure_rate_per_year'])
                current_date += timedelta(days=days_to_next_failure)
                
                if current_date <= self.end_date:
                    failure_dates.append(current_date)
            
            # Generate failure events
            for failure_date in failure_dates:
                # Select failure type based on probabilities
                failure_type = np.random.choice(
                    list(FAILURE_TYPES.keys()),
                    p=[FAILURE_TYPES[ft]['probability'] for ft in FAILURE_TYPES.keys()]
                )
                
                failure_info = FAILURE_TYPES[failure_type]
                
                # Generate failure details
                severity_level = np.random.randint(*failure_info['severity_range'])
                repair_cost = np.random.uniform(*failure_info['repair_cost_range'])
                
                # Downtime depends on severity and failure type
                base_downtime = {1: 2, 2: 8, 3: 24, 4: 72, 5: 168}[severity_level]
                downtime_hours = max(1, int(np.random.normal(base_downtime, base_downtime * 0.3)))
                
                # Repair date
                repair_date = failure_date + timedelta(hours=downtime_hours)
                
                # Detection method
                detection_methods = ['automated_monitoring', 'routine_inspection', 'performance_alert']
                detection_weights = [0.6, 0.3, 0.1]
                detection_method = np.random.choice(detection_methods, p=detection_weights)
                
                # Root cause analysis
                root_causes = {
                    'hot_spots': ['manufacturing_defect', 'soiling', 'shading'],
                    'micro_cracks': ['thermal_stress', 'mechanical_stress', 'manufacturing_defect'],
                    'connector_failure': ['corrosion', 'loose_connection', 'manufacturing_defect'],
                    'inverter_issues': ['component_failure', 'overheating', 'power_surge'],
                    'soiling': ['dust_accumulation', 'bird_droppings', 'pollen'],
                    'corrosion': ['moisture_ingress', 'salt_exposure', 'material_degradation'],
                    'electrical_degradation': ['aging', 'thermal_cycling', 'uv_exposure']
                }
                root_cause = np.random.choice(root_causes[failure_type])
                
                # Preventable flag
                preventable_prob = {'manufacturing_defect': 0.2, 'soiling': 0.8, 
                                  'aging': 0.1, 'thermal_stress': 0.4}.get(root_cause, 0.5)
                preventable = np.random.random() < preventable_prob
                
                # Warranty coverage (decreases with age)
                panel_age_years = (failure_date - installation_date).days / 365.25
                warranty_prob = max(0, 1 - panel_age_years / 10)  # 10-year warranty
                warranty_covered = np.random.random() < warranty_prob
                
                failure_events.append({
                    'event_id': f"EVT_{event_id:05d}",
                    'panel_id': panel_id,
                    'failure_date': failure_date.date(),
                    'failure_type': failure_type,
                    'severity_level': severity_level,
                    'detection_method': detection_method,
                    'repair_cost_usd': round(repair_cost, 2),
                    'downtime_hours': downtime_hours,
                    'repair_date': repair_date.date(),
                    'root_cause': root_cause,
                    'preventable': preventable,
                    'warranty_covered': warranty_covered,
                    'site_id': site_id
                })
                
                event_id += 1
        
        df = pd.DataFrame(failure_events)
        logger.info(f"Generated {len(df)} failure events")
        
        return df
    
    def generate_environmental_data(self) -> pd.DataFrame:
        """
        Generate synthetic environmental monitoring data.
        
        This dataset contains hourly environmental conditions that affect
        solar panel performance, including temperature, humidity, wind speed,
        dust levels, and weather conditions.
        
        Returns:
        --------
        pd.DataFrame
            Environmental data with columns:
            - timestamp, site_id, ambient_temperature_c, humidity_percent,
            - wind_speed_ms, dust_level, weather_condition, uv_index,
            - precipitation_mm, air_quality_index
        """
        logger.info("Generating environmental data...")
        
        environmental_data = []
        
        for site_id in SITES.keys():
            for timestamp in self.hourly_timestamps:
                # Generate correlated environmental variables
                ambient_temp = self._generate_temperature(timestamp)
                humidity = self._generate_humidity(timestamp, ambient_temp)
                wind_speed = self._generate_wind_speed(timestamp)
                
                # Weather condition affects other variables
                weather_condition = self._generate_weather_condition(timestamp, humidity)
                
                # Dust level (higher in dry conditions)
                dust_level = self._generate_dust_level(humidity, wind_speed)
                
                # UV index (depends on weather and time of day)
                uv_index = self._generate_uv_index(timestamp, weather_condition)
                
                # Precipitation
                precipitation = self._generate_precipitation(weather_condition)
                
                # Air quality index
                air_quality = self._generate_air_quality(timestamp, weather_condition)
                
                environmental_data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'ambient_temperature_c': round(ambient_temp, 1),
                    'humidity_percent': round(humidity, 1),
                    'wind_speed_ms': round(wind_speed, 1),
                    'dust_level': dust_level,
                    'weather_condition': weather_condition,
                    'uv_index': round(uv_index, 1),
                    'precipitation_mm': round(precipitation, 1),
                    'air_quality_index': round(air_quality, 0)
                })
        
        df = pd.DataFrame(environmental_data)
        logger.info(f"Generated {len(df)} environmental records")
        
        return df
    
    def _generate_irradiance(self, timestamp: pd.Timestamp) -> float:
        """Generate realistic solar irradiance values."""
        # Base irradiance with seasonal and daily variations
        base_irradiance = 800  # W/m²
        
        # Add seasonal variation
        irradiance = add_seasonal_variation(base_irradiance, timestamp, amplitude=0.3)
        
        # Add daily variation (solar pattern)
        irradiance = add_daily_variation(irradiance, timestamp, amplitude=0.8)
        
        # Add weather effects
        hour = timestamp.hour
        if 6 <= hour <= 18:  # Daylight hours
            # Add some random weather variation
            weather_factor = np.random.uniform(0.7, 1.0)
            irradiance *= weather_factor
        else:
            irradiance = 0
        
        return max(0, irradiance)
    
    def _generate_temperature(self, timestamp: pd.Timestamp) -> float:
        """Generate realistic ambient temperature values."""
        # Base temperature with seasonal variation
        base_temp = 15  # °C
        temp = add_seasonal_variation(base_temp, timestamp, amplitude=0.6)
        
        # Daily variation
        hour = timestamp.hour
        daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp += daily_variation
        
        # Add random noise
        temp += np.random.normal(0, 2)
        
        return temp
    
    def _generate_humidity(self, timestamp: pd.Timestamp, temperature: float) -> float:
        """Generate humidity correlated with temperature."""
        # Base humidity (inversely correlated with temperature)
        base_humidity = 70 - (temperature - 15) * 1.5
        
        # Add seasonal variation
        humidity = add_seasonal_variation(base_humidity, timestamp, amplitude=0.2)
        
        # Add random noise
        humidity += np.random.normal(0, 5)
        
        return np.clip(humidity, 20, 95)
    
    def _generate_wind_speed(self, timestamp: pd.Timestamp) -> float:
        """Generate wind speed values."""
        base_wind = 3  # m/s
        
        # Add seasonal variation (windier in winter)
        wind = add_seasonal_variation(base_wind, timestamp, amplitude=-0.3)
        
        # Add random variation
        wind += np.random.exponential(2)
        
        return np.clip(wind, 0, 15)
    
    def _generate_weather_condition(self, timestamp: pd.Timestamp, humidity: float) -> str:
        """Generate weather conditions based on humidity and season."""
        # Weather probabilities based on humidity
        if humidity < 40:
            probs = [0.8, 0.15, 0.05, 0.0]  # sunny, partly_cloudy, cloudy, rainy
        elif humidity < 60:
            probs = [0.5, 0.3, 0.15, 0.05]
        elif humidity < 80:
            probs = [0.2, 0.3, 0.35, 0.15]
        else:
            probs = [0.1, 0.2, 0.4, 0.3]
        
        return np.random.choice(ENVIRONMENTAL_PARAMS['weather_conditions'], p=probs)
    
    def _generate_dust_level(self, humidity: float, wind_speed: float) -> int:
        """Generate dust level based on humidity and wind."""
        # Lower humidity and higher wind = more dust
        base_dust = 3
        
        if humidity < 40:
            base_dust += 1
        if wind_speed > 5:
            base_dust += 1
        
        dust_level = base_dust + np.random.randint(-1, 2)
        return np.clip(dust_level, 1, 5)
    
    def _generate_uv_index(self, timestamp: pd.Timestamp, weather_condition: str) -> float:
        """Generate UV index based on time and weather."""
        hour = timestamp.hour
        
        if 6 <= hour <= 18:
            # Base UV index with daily pattern
            base_uv = 8 * np.sin(np.pi * (hour - 6) / 12)
            
            # Weather effects
            weather_factors = {
                'sunny': 1.0,
                'partly_cloudy': 0.8,
                'cloudy': 0.5,
                'rainy': 0.2
            }
            base_uv *= weather_factors[weather_condition]
            
            # Seasonal variation
            base_uv = add_seasonal_variation(base_uv, timestamp, amplitude=0.4)
        else:
            base_uv = 0
        
        return max(0, base_uv)
    
    def _generate_precipitation(self, weather_condition: str) -> float:
        """Generate precipitation based on weather condition."""
        if weather_condition == 'rainy':
            return np.random.exponential(2)
        elif weather_condition == 'cloudy':
            return np.random.exponential(0.5) if np.random.random() < 0.2 else 0
        else:
            return 0
    
    def _generate_air_quality(self, timestamp: pd.Timestamp, weather_condition: str) -> float:
        """Generate air quality index."""
        base_aqi = 50  # Good air quality
        
        # Seasonal variation (worse in winter)
        aqi = add_seasonal_variation(base_aqi, timestamp, amplitude=-0.3)
        
        # Weather effects
        if weather_condition == 'rainy':
            aqi *= 0.7  # Rain cleans air
        elif weather_condition == 'sunny':
            aqi *= 1.2  # More pollution on sunny days
        
        # Add random variation
        aqi += np.random.normal(0, 10)
        
        return max(0, aqi)
    
    def generate_all_datasets(self, save_to_files: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate all three datasets and optionally save to files.
        
        Parameters:
        -----------
        save_to_files : bool
            Whether to save datasets to CSV files
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing all generated datasets
        """
        logger.info("Starting generation of all datasets...")
        
        # Generate datasets
        datasets = {
            'panel_performance': self.generate_panel_performance_data(),
            'failure_events': self.generate_failure_events_data(),
            'environmental_data': self.generate_environmental_data()
        }
        
        if save_to_files:
            logger.info("Saving datasets to files...")
            
            # Save each dataset
            for name, df in datasets.items():
                filepath = save_dataframe(df, name, DATA_DIR, 'csv')
                logger.info(f"Saved {name} to {filepath}")
        
        logger.info("All datasets generated successfully!")
        
        # Print summary statistics
        for name, df in datasets.items():
            logger.info(f"{name}: {len(df)} records, {len(df.columns)} columns")
        
        return datasets


def main():
    """
    Main function to generate all datasets.
    This can be run as a standalone script.
    """
    # Initialize generator
    generator = SolarDataGenerator(
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=42
    )
    
    # Generate all datasets
    datasets = generator.generate_all_datasets(save_to_files=True)
    
    print("\n" + "="*60)
    print("SOLAR PANEL QUALITY CONTROL - SYNTHETIC DATA GENERATION")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Date Range: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
        
        if name == 'panel_performance':
            print(f"  Unique Panels: {df['panel_id'].nunique()}")
            print(f"  Sites: {df['site_id'].nunique()}")
        elif name == 'failure_events':
            print(f"  Total Failures: {len(df)}")
            print(f"  Affected Panels: {df['panel_id'].nunique()}")
    
    print(f"\nDatasets saved to: {DATA_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()