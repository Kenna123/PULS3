# PULS3

PULS3 is a Streamlit-based crime monitoring platform designed to help visualize crime trends, detect abnormal spikes, and identify high-risk zones within a monitored city.

The system combines data analytics, time-based comparisons, and automated alerts to provide a simplified early-warning dashboard for crime pattern monitoring.

---

# Highlights

- Streamlit-based interactive crime monitoring dashboard
- Branded login screen and onboarding setup flow
- Crime trend cards comparing recent activity vs historical periods
- Automatic spike detection for unusual crime increases
- 14-day trend visualization for selected crime types
- Growth insights and behavioral pattern indicators
- High-risk hotspot zone identification
- Alert system for significant crime increases
- Exportable crime reports

---

# Tech Stack

- Streamlit
- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels (ARIMA forecasting)

---

# Data + Modeling

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

---

# Dashboard Analytics Pipeline

The system processes crime data through the following steps:
