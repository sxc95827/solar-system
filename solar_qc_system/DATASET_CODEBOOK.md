# 📊 Dataset Codebook - Solar Panel Quality Control System

**Comprehensive Data Dictionary for Six Sigma Analysis**

This codebook provides detailed documentation for all datasets used in the Solar Panel Quality Control System. Each dataset is designed to accurately represent real-world solar panel operations and support comprehensive Six Sigma analysis.

---

## 🔍 Dataset Overview

The system uses three interconnected datasets:

1. **Panel Performance Data** - Real-time operational metrics
2. **Failure Events Data** - Comprehensive failure tracking
3. **Environmental Data** - Environmental monitoring data

All datasets span a 6-month period (January 2024 - June 2024) and are generated with realistic statistical distributions based on industry standards.

---

## 📈 Dataset 1: Panel Performance Data

**File:** `panel_performance.csv`  
**Purpose:** Monitor individual solar panel performance metrics for SPC analysis  
**Records:** ~131,400 (1,000 panels × 6 months × hourly data during daylight)  
**Update Frequency:** Hourly during daylight hours (6 AM - 8 PM)

### Variable Definitions

| Variable | Data Type | Description | Valid Range | Units | Notes |
|----------|-----------|-------------|-------------|-------|-------|
| `panel_id` | String | Unique panel identifier | P001-P1000 | - | Sequential numbering |
| `timestamp` | DateTime | Measurement timestamp | 2024-01-01 to 2024-06-30 | ISO 8601 | Hourly intervals |
| `efficiency` | Float | Panel conversion efficiency | 15.0-25.0 | % | Target: 20% ±2.5% |
| `power_output_kw` | Float | Current power generation | 0.0-1.2 | kW | Rated capacity: 1.0 kW |
| `voltage` | Float | DC output voltage | 25.0-45.0 | V | Nominal: 35V |
| `current` | Float | DC output current | 0.0-15.0 | A | Nominal: 10A |
| `temperature_c` | Float | Panel surface temperature | -15.0-70.0 | °C | Affects efficiency |
| `age_years` | Float | Time since installation | 0.0-10.0 | Years | Degradation factor |
| `site_id` | String | Installation site | Site_A, Site_B, Site_C | - | Geographic location |
| `panel_type` | String | Technology type | See below | - | Manufacturing type |

### Panel Type Categories

| Type | Description | Market Share | Efficiency Range |
|------|-------------|--------------|------------------|
| `Monocrystalline` | Single crystal silicon | 60% | 18-25% |
| `Polycrystalline` | Multi-crystal silicon | 30% | 15-20% |
| `Thin-film` | Amorphous silicon | 10% | 12-18% |

### Site Characteristics

| Site ID | Location Type | Climate | Typical Irradiance |
|---------|---------------|---------|-------------------|
| `Site_A` | Desert | Hot, dry | High (1000+ W/m²) |
| `Site_B` | Coastal | Moderate, humid | Medium (800 W/m²) |
| `Site_C` | Mountain | Cool, variable | Variable (600-1000 W/m²) |

### Data Generation Logic

- **Efficiency:** Normal distribution with age-related degradation (0.5%/year)
- **Power Output:** Calculated from efficiency, irradiance, and temperature
- **Temperature:** Ambient + 20°C (typical panel heating)
- **Voltage/Current:** Derived from power using Ohm's law with realistic variations

---

## ⚠️ Dataset 2: Failure Events Data

**File:** `failure_events.csv`  
**Purpose:** Track all panel failures for reliability and Pareto analysis  
**Records:** 200 failure events  
**Time Span:** 6 months (distributed randomly)

### Variable Definitions

| Variable | Data Type | Description | Valid Range | Units | Notes |
|----------|-----------|-------------|-------------|-------|-------|
| `failure_id` | String | Unique failure identifier | F001-F200 | - | Sequential numbering |
| `panel_id` | String | Associated panel ID | P001-P1000 | - | Links to performance data |
| `failure_date` | DateTime | When failure occurred | 2024-01-01 to 2024-06-30 | ISO 8601 | Random distribution |
| `failure_type` | String | Category of failure | See below | - | Primary classification |
| `severity` | String | Impact level | Low, Medium, High, Critical | - | Business impact |
| `repair_cost` | Float | Total repair cost | 50.0-5000.0 | USD | Parts + labor |
| `downtime_hours` | Float | Hours out of service | 1.0-168.0 | Hours | Until repair complete |
| `repair_time_hours` | Float | Active repair time | 0.5-48.0 | Hours | Technician time |
| `root_cause` | String | Underlying cause | See below | - | Root cause analysis |
| `preventable` | Boolean | Could have been prevented | True/False | - | Maintenance opportunity |

### Failure Type Distribution

| Failure Type | Frequency | Description | Typical Cost | Typical Downtime |
|--------------|-----------|-------------|--------------|------------------|
| `Hot spots` | 25% | Localized overheating | $200-800 | 4-12 hours |
| `Corrosion` | 20% | Material degradation | $300-1500 | 8-24 hours |
| `Physical damage` | 15% | Cracks, breaks | $500-3000 | 12-48 hours |
| `Electrical issues` | 15% | Wiring, connections | $100-600 | 2-8 hours |
| `Soiling` | 10% | Dirt, debris buildup | $50-200 | 1-4 hours |
| `Inverter problems` | 10% | Power conversion | $800-5000 | 24-168 hours |
| `Other` | 5% | Miscellaneous | $100-1000 | 2-12 hours |

### Severity Levels

| Severity | Impact | Repair Priority | Cost Range | Downtime Range |
|----------|--------|-----------------|------------|----------------|
| `Low` | <5% power loss | 7 days | $50-300 | 1-8 hours |
| `Medium` | 5-15% power loss | 3 days | $200-1000 | 4-24 hours |
| `High` | 15-50% power loss | 1 day | $500-3000 | 12-72 hours |
| `Critical` | >50% power loss | Immediate | $1000-5000 | 24-168 hours |

### Root Cause Categories

| Root Cause | Description | Preventable % | Common Failures |
|------------|-------------|---------------|-----------------|
| `Manufacturing` | Design/production defects | 30% | Hot spots, electrical |
| `Environmental` | Weather/climate damage | 60% | Corrosion, physical |
| `Maintenance` | Inadequate upkeep | 90% | Soiling, connections |
| `Installation` | Poor initial setup | 80% | Wiring, mounting |
| `Age` | Normal wear and tear | 20% | General degradation |

---

## 🌤️ Dataset 3: Environmental Data

**File:** `environmental_data.csv`  
**Purpose:** Monitor environmental factors affecting panel performance  
**Records:** ~4,380 (6 months × 30 days × 24 hours × 3 sites)  
**Update Frequency:** Hourly

### Variable Definitions

| Variable | Data Type | Description | Valid Range | Units | Notes |
|----------|-----------|-------------|-------------|-------|-------|
| `timestamp` | DateTime | Measurement time | 2024-01-01 to 2024-06-30 | ISO 8601 | Hourly intervals |
| `site_id` | String | Monitoring location | Site_A, Site_B, Site_C | - | Same as panel data |
| `ambient_temp_c` | Float | Air temperature | -20.0-50.0 | °C | Weather station data |
| `humidity_percent` | Float | Relative humidity | 10.0-100.0 | % | Affects corrosion |
| `wind_speed_ms` | Float | Wind speed | 0.0-25.0 | m/s | Cooling effect |
| `irradiance_wm2` | Float | Solar irradiance | 0.0-1400.0 | W/m² | Direct + diffuse |
| `dust_level` | String | Dust accumulation | Low, Medium, High | - | Soiling factor |
| `weather_condition` | String | General weather | See below | - | Categorical |
| `precipitation_mm` | Float | Rainfall amount | 0.0-50.0 | mm/hour | Cleaning effect |

### Weather Conditions

| Condition | Frequency | Irradiance Impact | Dust Impact |
|-----------|-----------|-------------------|-------------|
| `Clear` | 40% | Maximum | Accumulation |
| `Partly cloudy` | 30% | Reduced 20-50% | Moderate |
| `Cloudy` | 20% | Reduced 50-80% | Low |
| `Rainy` | 8% | Reduced 70-90% | Cleaning |
| `Snowy` | 2% | Reduced 80-95% | Cleaning |

### Dust Level Impact

| Level | Description | Efficiency Loss | Cleaning Frequency |
|-------|-------------|-----------------|-------------------|
| `Low` | Minimal accumulation | 0-2% | Monthly |
| `Medium` | Moderate buildup | 2-8% | Bi-weekly |
| `High` | Heavy soiling | 8-20% | Weekly |

### Seasonal Patterns

- **Temperature:** Sinusoidal variation with site-specific baselines
- **Irradiance:** Day/night cycles with seasonal amplitude changes
- **Humidity:** Inverse correlation with temperature
- **Wind:** Random with seasonal trends
- **Precipitation:** Seasonal patterns with random events

---

## 🔗 Data Relationships

### Cross-Dataset Linkages

1. **Panel Performance ↔ Environmental Data**
   - Linked by `timestamp` and `site_id`
   - Temperature affects efficiency
   - Irradiance drives power output
   - Dust level impacts performance

2. **Panel Performance ↔ Failure Events**
   - Linked by `panel_id`
   - Performance degradation precedes failures
   - Age correlation with failure probability

3. **Failure Events ↔ Environmental Data**
   - Linked by `failure_date` and site location
   - Weather conditions influence failure types
   - Environmental stress accelerates failures

### Data Quality Assurance

- **Temporal Consistency:** All timestamps align with realistic patterns
- **Physical Constraints:** All values within physically possible ranges
- **Statistical Validity:** Distributions match industry benchmarks
- **Referential Integrity:** All foreign keys have valid references

---

## 📊 Statistical Characteristics

### Panel Performance Statistics

| Metric | Mean | Std Dev | Min | Max | Distribution |
|--------|------|---------|-----|-----|--------------|
| Efficiency (%) | 20.0 | 2.1 | 15.2 | 24.8 | Normal |
| Power (kW) | 0.85 | 0.18 | 0.12 | 1.15 | Log-normal |
| Temperature (°C) | 35.2 | 12.4 | -8.3 | 68.7 | Normal |

### Failure Event Statistics

| Metric | Mean | Std Dev | Min | Max | Distribution |
|--------|------|---------|-----|-----|--------------|
| Repair Cost ($) | 847 | 623 | 52 | 4,850 | Log-normal |
| Downtime (hours) | 18.3 | 24.7 | 1.2 | 156.8 | Exponential |
| Repair Time (hours) | 8.7 | 9.2 | 0.5 | 45.3 | Gamma |

### Environmental Statistics

| Metric | Mean | Std Dev | Min | Max | Distribution |
|--------|------|---------|-----|-----|--------------|
| Temperature (°C) | 18.5 | 11.2 | -18.7 | 47.3 | Normal |
| Humidity (%) | 62.4 | 18.9 | 12.1 | 98.7 | Beta |
| Irradiance (W/m²) | 425 | 380 | 0 | 1,350 | Bimodal |

---

## 🎯 Six Sigma Applications

### Control Charts (SPC)
- **X-bar charts:** Panel efficiency monitoring
- **Individual charts:** Single panel tracking
- **Range charts:** Process variation analysis

### Process Capability
- **Specification limits:** 18-22% efficiency
- **Target value:** 20% efficiency
- **Capability indices:** Cp, Cpk, Pp, Ppk

### Pareto Analysis
- **Failure types:** Cost and frequency ranking
- **Root causes:** Prevention prioritization
- **Sites:** Performance comparison

### Reliability Analysis
- **MTBF calculation:** Time between failures
- **MTTR calculation:** Repair time analysis
- **Availability:** Uptime percentage

---

## 🔧 Data Usage Guidelines

### For Analysis Tools
1. **Load all three datasets** for comprehensive analysis
2. **Join on common keys** (panel_id, site_id, timestamp)
3. **Handle missing values** appropriately for each metric
4. **Validate data ranges** before analysis

### For Visualization
1. **Time series plots** for trend analysis
2. **Scatter plots** for correlation analysis
3. **Histograms** for distribution analysis
4. **Box plots** for comparative analysis

### For Six Sigma Analysis
1. **Establish control limits** from historical data
2. **Define specification limits** based on requirements
3. **Calculate capability indices** for process assessment
4. **Identify improvement opportunities** from patterns

---

## 📝 Data Generation Notes

### Realism Factors
- **Industry benchmarks** used for all parameters
- **Physical constraints** enforced throughout
- **Seasonal variations** included in all time series
- **Correlation structures** maintained between variables

### Validation Methods
- **Statistical tests** confirm distribution shapes
- **Domain expert review** validates parameter ranges
- **Cross-validation** ensures relationship consistency
- **Benchmark comparison** with published industry data

### Update Procedures
- **Regenerate data** using `python main.py --generate-data`
- **Modify parameters** in `config.py` for customization
- **Validate output** using built-in quality checks
- **Document changes** in version control

---

**This codebook ensures complete understanding and proper usage of all datasets in the Solar Panel Quality Control System for effective Six Sigma analysis.**