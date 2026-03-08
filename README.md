# PULS3

PULS3 is a Streamlit-based crime monitoring platform that combines predictive alerts, trend analytics, and high-risk zone visibility in a single dashboard.

## Highlights

- Branded login screen, setup flow, and operations dashboard
- Configurable monitoring area and crime categories
- Predictive district risk scoring based on your model logic (`PULS3_Model.ipynb`)
- Real-time style alert feed and crime trend cards
- Spike analytics panel with 14-day trend and signal summaries
- High-risk zones ranked by model probability
- Offline demo fallback when live data feed is unavailable

## Tech Stack

- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost (with RandomForest fallback)

## Data + Modeling

- Live source: Chicago Open Data (`ijzp-q8t2` endpoint)
- Feature engineering and training pattern replicate your notebook:
  - weekly district aggregation
  - lag features (`crime_lag1`, `crime_lag2`, `crime_lag3`)
  - rolling windows (`crime_roll4`, `crime_roll8`)
  - temporal features (`hour`, `day_of_week`)

## Local Run

```bash
cd "/Users/agbug/Downloads/PULS3 Project"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8505
```

Open: [http://localhost:8505](http://localhost:8505)

## Project Files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - Python/Streamlit ignore rules

## Current Status

Core PULS3 UI and model-backed monitoring flow are implemented and runnable locally.
