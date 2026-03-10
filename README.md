# PULS3

PULS3 is a Streamlit-based crime monitoring platform designed to visualize crime trends, detect abnormal spikes, and identify high-risk zones within a monitored city.

The system combines data analytics, time-based comparisons, and automated alerts to provide a simplified early-warning dashboard for crime pattern monitoring.

## Table of Contents

- [Highlights](#highlights)
- [Tech Stack](#tech-stack)
- [Data and Modeling](#data-and-modeling)
- [Dashboard Analytics Pipeline](#dashboard-analytics-pipeline)
- [Example Use Cases](#example-use-cases)
- [Future Improvements](#future-improvements)

## Highlights

- Streamlit-based interactive crime monitoring dashboard
- Branded login screen and onboarding setup flow
- Crime trend cards comparing recent activity vs historical periods
- Automatic spike detection for unusual crime increases
- 14-day trend visualization for selected crime types
- Growth insights and behavioral pattern indicators
- High-risk hotspot zone identification
- Alert system for significant crime increases
- Exportable crime reports

## Tech Stack

Python libraries and frameworks used in the project:

- Streamlit: web dashboard interface
- Pandas: data manipulation and analysis
- NumPy: numerical operations
- Scikit-learn: machine learning models
- Statsmodels: ARIMA forecasting for time-series prediction

## Data and Modeling

The dashboard analyzes historical police incident data to detect changes in crime patterns.

Key analytical steps include:

- Data cleaning and preprocessing
- Time-based aggregation of incidents
- Feature engineering for temporal analysis
- Crime frequency comparisons across time windows
- Spike detection using percentage change thresholds
- ARIMA-based short-term crime forecasting

Example features used in modeling:

- Incident date
- Crime type
- Temporal indicators (hour, day of week)
- Rolling crime counts
- Lagged crime activity

## Dashboard Analytics Pipeline

The system processes crime data through the following stages:

### 1. Data Loading

Crime incident data is loaded from the source dataset and formatted for analysis. The dataset includes incident timestamps, crime categories, and geographic identifiers.

### 2. Data Cleaning

Invalid or missing values are removed, and incident timestamps are converted into a standard datetime format.

### 3. Feature Engineering

Time-based features are extracted from incident dates, including:

- Hour of day
- Day of week
- Month
- Week of year

These features allow the system to analyze temporal crime patterns.

### 4. Temporal Aggregation

Crime incidents are grouped into time windows to calculate crime frequencies. This enables comparison between recent crime activity and historical patterns.

### 5. Spike Detection

The system calculates percentage change in crime counts over recent periods. When crime activity increases significantly beyond expected thresholds, the platform flags the event as a spike.

### 6. Risk Zone Identification

Crime frequencies are analyzed across geographic zones. Areas with consistently higher crime activity are flagged as high-risk zones.

### 7. Forecasting

An ARIMA time-series model is used to forecast short-term crime activity trends based on historical data patterns.

### 8. Visualization and Alerts

The processed data is displayed in the Streamlit dashboard using:

- Crime trend cards
- Time-series graphs
- Risk zone indicators
- Alert notifications

## Example Use Cases

PULS3 can be used by several stakeholders:

- Public safety analysts monitoring crime trends
- Researchers studying urban crime patterns
- City planners evaluating neighborhood risk levels
- Community organizations interested in public safety awareness

## Future Improvements

Potential enhancements to the platform include:

- Real-time data streaming from public safety APIs
- Geospatial crime heatmaps
- Deep learning models for improved forecasting
- Multi-city monitoring capabilities
- Automated anomaly detection models
