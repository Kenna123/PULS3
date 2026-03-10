import datetime as dt
import base64
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import streamlit as st
import joblib
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None
try:
    from statsmodels.tsa.arima.model import ARIMA

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    ARIMA = None
try:
    from fpdf import FPDF

    HAS_FPDF = True
except Exception:
    HAS_FPDF = False
    FPDF = None

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="PULS3", page_icon="🛡️", layout="wide")

THEME = {
    "bg": "#f2eff1",
    "panel": "#ffffff",
    "line": "#e3dfe1",
    "text": "#1f2937",
    "muted": "#6b7280",
    "brand": "#8b1d2c",
    "brand_dark": "#6f1321",
    "good": "#0f9f6e",
    "warn": "#f39c12",
    "bad": "#b4233f",
}

CRIME_OPTIONS = ["Assault", "Robbery", "Theft", "Battery", "Burglary", "Motor Vehicle Theft"]


def canonical_crime_name(value: str) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "assault": "Assault",
        "robbery": "Robbery",
        "theft": "Theft",
    }
    return mapping.get(text, str(value or "").strip().title())


@dataclass
class ModelBundle:
    model: object
    encoder: Optional[LabelEncoder]
    weekly_data: pd.DataFrame
    features: List[str]
    source: str = "little_rock"


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
ALERT_DB_PATH = DATA_DIR / "puls3_alerts.db"


def _normalize_lr_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure source datetime column is parsed even before normalization.
    if "INCIDENT_DATE" in df.columns:
        df["INCIDENT_DATE"] = pd.to_datetime(
            df["INCIDENT_DATE"], errors="coerce", format="%m/%d/%Y %I:%M:%S %p"
        )
        if df["INCIDENT_DATE"].isna().mean() > 0.2:
            df["INCIDENT_DATE"] = pd.to_datetime(df["INCIDENT_DATE"], errors="coerce")

    rename_map = {
        "date": "Date",
        "incident_date": "Date",
        "incident_date": "Date",
        "occurred_at": "Date",
        "timestamp": "Date",
        "crime_type": "Primary_Type",
        "offense": "Primary_Type",
        "offense_type": "Primary_Type",
        "offense_description": "Primary_Type",
        "primary_type": "Primary_Type",
        "district": "District",
        "location_district": "District",
        "zone": "District",
        "beat": "District",
        "neighborhood": "Neighborhood",
        "area": "Neighborhood",
        "location": "Neighborhood",
        "incident_location": "Neighborhood",
        "zip": "Zipcode",
        "zipcode": "Zipcode",
        "zip_code": "Zipcode",
        "city": "City",
    }
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    rename_actual = {}
    for k, v in rename_map.items():
        actual = lower_to_actual.get(k)
        if actual and v not in df.columns:
            rename_actual[actual] = v
    if rename_actual:
        df = df.rename(columns=rename_actual)

    if "Date" not in df.columns or "Primary_Type" not in df.columns:
        raise ValueError("Little Rock dataset must include Date and crime type columns.")

    if "District" not in df.columns:
        if "Neighborhood" in df.columns:
            df["District"] = df["Neighborhood"].astype(str)
        else:
            df["District"] = "Unknown"

    if "Neighborhood" not in df.columns:
        df["Neighborhood"] = "District " + df["District"].astype(str)
    if "City" not in df.columns:
        df["City"] = "Little Rock"
    if "Zipcode" not in df.columns:
        df["Zipcode"] = ""

    df = df.dropna(subset=["Date", "Primary_Type", "District"]).copy()
    # Parse normalized datetime column; keep robust fallback if format varies.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%m/%d/%Y %I:%M:%S %p")
    if df["Date"].isna().mean() > 0.2:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Primary_Type"] = df["Primary_Type"].astype(str).str.upper()
    # Map LRPD offense descriptions to core monitoring categories used by UI.
    def _map_primary_type(v: str) -> str:
        if "ASSAULT" in v:
            return "Assault"
        if "ROBBERY" in v:
            return "Robbery"
        if "THEFT" in v or "LARCENY" in v or "BURGLARY" in v or "STOLEN" in v:
            return "Theft"
        return v.title()

    df["Primary_Type"] = df["Primary_Type"].apply(_map_primary_type)
    df["District"] = df["District"].astype(str)
    df["Neighborhood"] = df["Neighborhood"].astype(str)
    df["City"] = df["City"].astype(str)
    df["Zipcode"] = df["Zipcode"].astype(str)
    return df.sort_values("Date")


@st.cache_data(show_spinner=False)
def load_little_rock_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No Little Rock dataset found. Place your exported CSV in ./data/ (e.g., data/little_rock_crime.csv)."
        )
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    df = pd.read_csv(latest)
    return _normalize_lr_columns(df)


@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame) -> ModelBundle:
    data = df.copy()
    data["crime_occurred"] = 1
    data["hour"] = data["Date"].dt.hour
    data["day_of_week"] = data["Date"].dt.dayofweek
    data["week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["year"] = data["Date"].dt.year

    weekly = (
        data.groupby(["District", "year", "week"], as_index=False)
        .agg({"crime_occurred": "sum", "hour": "mean", "day_of_week": "mean"})
        .sort_values(["District", "year", "week"])
    )

    weekly["high_crime_next_week"] = (
        weekly.groupby("District")["crime_occurred"].shift(-1) > weekly["crime_occurred"].median()
    ).astype(float)

    weekly["crime_lag1"] = weekly.groupby("District")["crime_occurred"].shift(1)
    weekly["crime_lag2"] = weekly.groupby("District")["crime_occurred"].shift(2)
    weekly["crime_lag3"] = weekly.groupby("District")["crime_occurred"].shift(3)
    weekly["crime_roll4"] = (
        weekly.groupby("District")["crime_occurred"].rolling(4).mean().reset_index(0, drop=True)
    )
    weekly["crime_roll8"] = (
        weekly.groupby("District")["crime_occurred"].rolling(8).mean().reset_index(0, drop=True)
    )

    le = LabelEncoder()
    weekly["district_encoded"] = le.fit_transform(weekly["District"])

    features = [
        "district_encoded",
        "crime_occurred",
        "crime_lag1",
        "crime_lag2",
        "crime_lag3",
        "crime_roll4",
        "crime_roll8",
        "hour",
        "day_of_week",
    ]

    train_df = weekly.dropna(subset=features + ["high_crime_next_week"]).copy()
    X = train_df[features]
    y = train_df["high_crime_next_week"].astype(int)

    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42)

    model.fit(X, y)
    return ModelBundle(model=model, encoder=le, weekly_data=weekly, features=features, source="little_rock_trained")


@st.cache_resource(show_spinner=False)
def load_model_bundle(df: pd.DataFrame) -> ModelBundle:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_candidates = list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.pkl"))
    if artifact_candidates:
        artifact = sorted(artifact_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        loaded = joblib.load(artifact)
        if isinstance(loaded, ModelBundle):
            return loaded
        if isinstance(loaded, dict):
            required = {"model", "weekly_data", "features"}
            if required.issubset(set(loaded.keys())):
                return ModelBundle(
                    model=loaded["model"],
                    encoder=loaded.get("encoder"),
                    weekly_data=loaded["weekly_data"],
                    features=list(loaded["features"]),
                    source=f"artifact:{artifact.name}",
                )
    return build_model(df)


def get_latest_zone_risk(bundle: ModelBundle) -> pd.DataFrame:
    weekly = bundle.weekly_data.copy()
    latest_rows = weekly.sort_values(["District", "year", "week"]).groupby("District", as_index=False).tail(1)
    latest_rows = latest_rows.dropna(subset=bundle.features).copy()

    probs = bundle.model.predict_proba(latest_rows[bundle.features])[:, 1]
    latest_rows["risk_prob"] = probs

    def severity(prob: float) -> str:
        if prob >= 0.8:
            return "Critical"
        if prob >= 0.6:
            return "Elevated"
        if prob >= 0.4:
            return "Moderate"
        return "Low"

    latest_rows["severity"] = latest_rows["risk_prob"].apply(severity)
    return latest_rows[["District", "risk_prob", "severity"]].sort_values("risk_prob", ascending=False)


def init_state() -> None:
    defaults = {
        "page": "login",
        "email": "",
        "location": "",
        "selected_crimes": ["Assault"],
        "selected_crime": "",
        "selected_crime_user_selected": False,
        "active_crime": "Assault",
        "logged_in": False,
        "monitoring_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
            .brand {{ color: {THEME['brand']}; font-weight: 800; letter-spacing: 0.03em; }}
            .card {{
                background: {THEME['panel']}; border: 1px solid {THEME['line']};
                border-radius: 16px; padding: 22px;
            }}
            .banner {{
                background: {THEME['brand']}; color: white; border-radius: 12px;
                padding: 12px 16px; font-weight: 700;
            }}
            .metric {{
                background: {THEME['panel']}; border: 1px solid {THEME['line']};
                border-radius: 14px; padding: 18px;
            }}
            .metric h4 {{ margin: 0; color: {THEME['muted']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em; }}
            .metric .v {{ margin-top: 8px; font-size: 42px; font-weight: 800; color: {THEME['brand']}; }}
            .pill {{ border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 700; display: inline-block; }}
            .pill-critical {{ background: #fde8ec; color: {THEME['bad']}; }}
            .pill-elevated {{ background: #fff2e1; color: {THEME['warn']}; }}
            .pill-moderate {{ background: #fff7d6; color: #a67c00; }}
            .pill-low {{ background: #e9f7f1; color: {THEME['good']}; }}
            .footer {{ color: {THEME['muted']}; font-size: 12px; text-align: center; padding: 10px 0; }}
            .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; }}
            .stButton > button {{
                background: {THEME['brand']}; color: white; border: none; border-radius: 10px;
                padding: 0.6rem 1.2rem; font-weight: 700;
            }}
            .stButton > button:hover {{ background: {THEME['brand_dark']}; color: white; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login() -> None:
    st.markdown(
        """
        <style>
            html, body {
                background: #f6f6f7 !important;
            }
            .stApp {
                background: #f6f6f7 !important;
            }
            .block-container {
                position: relative;
                width: 100%;
                max-width: none !important;
                min-height: 100vh;
                margin: 0 auto !important;
                background: #f6f6f7;
                padding-top: 80px !important;
                padding-bottom: 80px !important;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow: visible;
            }
            .block-container > div[data-testid="stVerticalBlock"] {
                width: 100%;
                max-width: 420px;
                margin: 0 auto;
                text-align: center;
            }
            .brand-stack {
                position: relative;
                z-index: 1;
                text-align: center;
                margin: 0 auto 40px;
                width: 100%;
            }
            .brand-logo {
                width: 220px;
                max-width: 100%;
                margin: 0 auto;
                display: block;
                margin-bottom: 12px;
            }
            .brand-stack .tagline {
                display: block;
                text-align: center;
                max-width: 400px;
                margin: 0 auto;
            }
            .brand-stack .alerts-label {
                margin: 4px auto 0;
                text-align: center;
            }
            .brand-row {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin-bottom: 12px;
            }
            .fallback-shield {
                width: 25px;
                height: 29px;
                border-radius: 6px;
                background: #7a1e24;
                position: relative;
            }
            .fallback-shield::after {
                content: "";
                position: absolute;
                left: 6px;
                right: 6px;
                top: 9px;
                height: 2px;
                background: #fff;
                box-shadow: 0 -3px 0 0 #fff, 0 3px 0 0 #fff;
                opacity: 0.9;
            }
            .brand-word {
                margin: 0;
                font-size: 44px;
                line-height: 1;
                font-weight: 800;
                letter-spacing: 0.02em;
                color: #7a1e24;
            }
            .tagline {
                margin: 0;
                color: #1f1f22;
                font-size: 19px;
                font-weight: 500;
                line-height: 1.3;
            }
            .tagline .hi {
                color: #7a1e24;
                font-weight: 700;
            }
            .alerts-label {
                color: #7a1e24;
                letter-spacing: 0.03em;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
            }
            div[data-testid="stForm"] {
                position: relative;
                z-index: 1;
                width: 100%;
                margin: 0 auto 32px auto;
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid #ebe7ea;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
                padding: 28px !important;
            }
            div[data-testid="stForm"] form {
                border: 0 !important;
                padding: 0 !important;
            }
            div[data-testid="stForm"] .stTextInput label,
            div[data-testid="stForm"] .stTextInput p {
                color: #1f1f22 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
            }
            div[data-testid="stForm"] .stTextInput > div {
                width: 100% !important;
            }
            div[data-testid="stForm"] .stTextInput [data-baseweb="base-input"] {
                width: 100% !important;
                position: relative !important;
            }
            div[data-testid="stForm"] .stTextInput input {
                width: 100% !important;
                height: 48px !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 8px !important;
                padding: 0 12px !important;
                background: #ffffff !important;
                color: #1f1f22 !important;
                font-size: 16px !important;
                box-shadow: none !important;
            }
            div[data-testid="stForm"] .stTextInput input:focus {
                border-color: #7a1e24 !important;
                box-shadow: 0 0 0 3px rgba(122, 30, 36, 0.14) !important;
                outline: none !important;
            }
            div[data-testid="stForm"] .stTextInput input[type="password"] {
                padding-right: 42px !important;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='19' height='19' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
                background-repeat: no-repeat !important;
                background-position: right 12px center !important;
                background-size: 18px 18px !important;
            }
            div[data-testid="stForm"] [data-baseweb="input"],
            div[data-testid="stForm"] [data-baseweb="base-input"] {
                background: #ffffff !important;
                border: 0 !important;
                box-shadow: none !important;
            }
            div[data-testid="stForm"] [data-baseweb="base-input"] > div {
                background: #ffffff !important;
            }
            div[data-testid="stForm"] [data-baseweb="base-input"] button,
            div[data-testid="stForm"] [data-baseweb="input"] button {
                background: transparent !important;
                border: 0 !important;
                color: #9ca3af !important;
                box-shadow: none !important;
                position: absolute !important;
                right: 12px !important;
                top: 50% !important;
                transform: translateY(-50%) !important;
                height: 20px !important;
                width: 20px !important;
                padding: 0 !important;
                margin: 0 !important;
                z-index: 3 !important;
                cursor: pointer !important;
            }
            div[data-testid="stForm"] .stTextInput input::placeholder {
                color: #9ca3af !important;
                opacity: 1;
            }
            div[data-testid="stForm"] .stTextInput:nth-of-type(1) {
                margin-bottom: 20px !important;
            }
            div[data-testid="stForm"] .stTextInput:nth-of-type(2) {
                margin-bottom: 20px !important;
            }
            div[data-testid="stForm"] .stButton > button,
            div[data-testid="stForm"] .stFormSubmitButton > button {
                height: 48px !important;
                width: 100% !important;
                border: 0 !important;
                border-radius: 8px !important;
                background: #7a1e24 !important;
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                box-shadow: 0 8px 16px rgba(122, 30, 36, 0.28) !important;
            }
            div[data-testid="stForm"] .stButton > button:hover,
            div[data-testid="stForm"] .stFormSubmitButton > button:hover {
                background: #6b1a1f !important;
            }
            .login-footer {
                position: relative;
                z-index: 1;
                margin-top: 0;
                text-align: center;
            }
            .account-row {
                color: #1f1f22;
                font-size: 14px;
                margin-bottom: 16px;
            }
            .account-row .cta {
                color: #7a1e24;
                font-weight: 700;
            }
            .legal-row {
                display: flex;
                justify-content: center;
                gap: 28px;
                color: rgba(31, 31, 34, 0.58);
                font-size: 12px;
                letter-spacing: 0.08em;
                font-weight: 600;
                text-transform: uppercase;
                white-space: nowrap;
                flex-wrap: nowrap;
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-top: 48px !important;
                    padding-bottom: 48px !important;
                }
                .brand-stack {
                    margin-bottom: 32px;
                }
                .tagline { font-size: 17px; }
                .alerts-label { font-size: 12px; }
                .block-container > div[data-testid="stVerticalBlock"] {
                    max-width: 420px;
                }
                div[data-testid="stForm"] { width: 100%; }
                .account-row {
                    font-size: 13px;
                }
                .legal-row { gap: 14px; font-size: 10px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_path = Path("assets/puls3-logo.png")
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        brand_top = f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='PULS3 logo' />"
    else:
        brand_top = """
        <div class="brand-row">
            <span class="fallback-shield"></span>
            <h1 class="brand-word">PULS3</h1>
        </div>
        """
    st.markdown(
        f"""
        <div class="brand-stack">
            {brand_top}
            <p class="tagline">When <span class="hi">Time</span>, <span class="hi">Place</span>, and <span class="hi">Risk</span> Align</p>
            <p class="alerts-label">PULS3 Alerts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login_form", clear_on_submit=False, border=False):
        email = st.text_input("Email Address", value=st.session_state.email, placeholder="name@agency.gov")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if submitted:
        if email.strip() and password.strip():
            st.session_state.email = email.strip()
            st.session_state.logged_in = True
            st.session_state.page = "setup"
            st.rerun()
        else:
            st.warning("Please provide email and password.")

    st.markdown(
        """
        <div class="login-footer">
            <div class="account-row">Don't have an account? <span class="cta">Create Account</span></div>
            <div class="legal-row">
                <span>Privacy Policy</span>
                <span>Terms of Service</span>
                <span>Support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_setup(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
            .stApp, html, body {
                background: #f6f6f7 !important;
            }
            .block-container {
                max-width: 1100px !important;
                min-height: 100vh;
                margin: 0 auto !important;
                padding-top: 56px !important;
                padding-bottom: 56px !important;
            }
            .setup-fixed-logo {
                position: relative;
                display: block;
                margin: 0 auto 24px auto;
                height: 44px;
            }
            .setup-title {
                margin: 0 0 20px;
                color: #1F2937;
                font-size: 34px;
                line-height: 1.1;
                font-weight: 700;
                text-align: center;
            }
            .setup-sub {
                margin: 0 0 20px;
                color: #6B7280;
                font-size: 14px;
                text-align: center;
            }
            div[data-testid="stForm"] {
                max-width: 720px;
                margin: 0 auto !important;
                background: #ffffff;
                border-radius: 16px;
                padding: 32px !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
                border: 1px solid #f0f0f0;
                text-align: left;
            }
            div[data-testid="stForm"], div.stForm {
                width: 100%;
            }
            div[data-testid="stForm"] form, div.stForm form {
                border: 0 !important;
                padding: 0 !important;
            }
            .section-label {
                margin: 0 0 20px;
                color: #374151;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            div[data-testid="stForm"] .stTextInput [data-baseweb="input"],
            div[data-testid="stForm"] .stTextInput [data-baseweb="base-input"] {
                border: 1px solid #E5E7EB !important;
                background: #F9FAFB !important;
                border-radius: 8px !important;
                box-shadow: none !important;
            }
            div[data-testid="stForm"] .stTextInput input {
                width: 100% !important;
                height: 48px !important;
                border-radius: 8px !important;
                border: 0 !important;
                background: #F9FAFB !important;
                padding: 0 14px 0 40px !important;
                font-size: 14px !important;
                line-height: 48px !important;
                color: #374151 !important;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E");
                background-repeat: no-repeat !important;
                background-position: 12px center !important;
            }
            div[data-testid="stForm"] .stTextInput input::placeholder {
                color: #9CA3AF !important;
            }
            .helper-text {
                margin: 6px 0 20px;
                font-size: 12px;
                color: #9CA3AF;
            }
            div[data-testid="stForm"] .stMultiSelect {
                width: 100% !important;
                margin-bottom: 14px;
            }
            div[data-baseweb="select"] > div {
                background: #F9FAFB !important;
                border: 1px solid #E5E7EB !important;
                border-radius: 8px !important;
                min-height: 48px !important;
            }
            div[data-testid="stForm"] .stMultiSelect [data-baseweb="select"] > div {
                min-height: 48px !important;
                align-items: center !important;
            }
            div[data-baseweb="select"] input {
                color: #374151 !important;
                -webkit-text-fill-color: #374151 !important;
            }
            div[data-testid="stForm"] .stMultiSelect input::placeholder {
                color: #9CA3AF !important;
                opacity: 1 !important;
            }
            div[data-baseweb="select"] span {
                color: #374151 !important;
            }
            div[data-baseweb="tag"],
            div[data-testid="stForm"] .stMultiSelect [data-baseweb="tag"] {
                background: #7A1E24 !important;
                border-radius: 6px !important;
                border: none !important;
                color: #ffffff !important;
                font-weight: 600 !important;
            }
            div[data-baseweb="tag"] *,
            div[data-testid="stForm"] .stMultiSelect [data-baseweb="tag"] * {
                color: #ffffff !important;
            }
            div[role="listbox"] {
                background: #ffffff !important;
                color: #374151 !important;
                border: 1px solid #E5E7EB !important;
                border-radius: 10px !important;
            }
            div[role="option"] {
                background: #ffffff !important;
                color: #374151 !important;
            }
            div[role="option"]:hover {
                background: #FDF4F5 !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton > button {
                width: 100% !important;
                height: 56px !important;
                margin-top: 24px !important;
                background: #7A1E24 !important;
                color: #ffffff !important;
                border: 0 !important;
                border-radius: 10px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                box-shadow: 0 12px 22px rgba(122,30,36,0.25) !important;
            }
            .terms {
                margin-top: 12px;
                text-align: center;
                font-size: 12px;
                color: #9CA3AF;
            }
            .setup-copyright {
                text-align: center;
                color: #9CA3AF;
                font-size: 12px;
                margin-top: 40px;
            }
            @media (max-width: 640px) {
                .setup-fixed-logo {
                    margin-bottom: 20px;
                    height: 28px;
                }
                .setup-title {
                    font-size: 32px;
                }
                .setup-sub {
                    font-size: 18px;
                }
                div[data-testid="stForm"] {
                    margin-top: 0;
                    padding: 24px !important;
                }
                .setup-copyright {
                    margin-top: 56px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = Path("assets/puls3-logo.png")
    logo_markup = ""
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_markup = f"<img class='setup-fixed-logo' src='data:image/png;base64,{logo_b64}' alt='PULS3 logo' />"
    st.markdown(logo_markup, unsafe_allow_html=True)

    district_options = sorted(df["District"].dropna().unique().tolist())
    city_options = ["Little Rock"]
    default_city_idx = 0
    if "selected_districts" not in st.session_state:
        st.session_state["selected_districts"] = district_options
    setup_crime_options = ["Assault", "Robbery", "Theft"]
    selected_default = [crime for crime in setup_crime_options if crime in st.session_state.selected_crimes]
    if not selected_default:
        selected_default = ["Assault"]

    with st.form("setup_monitoring_form", clear_on_submit=False, border=False):
        st.markdown("<h2 class='setup-title'>Set Up Monitoring</h2>", unsafe_allow_html=True)
        st.markdown("<p class='setup-sub'>Choose your area and select crimes to monitor</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-label'>Monitoring Area</p>", unsafe_allow_html=True)
        selected_city = st.selectbox(
            "Monitoring Area",
            options=city_options,
            index=default_city_idx,
            label_visibility="collapsed",
        )
        st.markdown("<div class='helper-text'>City-level monitoring</div>", unsafe_allow_html=True)

        st.markdown("<p class='section-label'>Crime Types To Monitor</p>", unsafe_allow_html=True)
        selected_now = st.multiselect(
            "Crime Types To Monitor",
            options=setup_crime_options,
            default=selected_default,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Start Monitoring \u2192", use_container_width=True)
        st.markdown(
            "<div class='terms'>By proceeding, you agree to our <u>Terms of Service</u></div>",
            unsafe_allow_html=True,
        )

    if submitted:
        city_mask = df["City"].astype(str).str.upper().str.contains("LITTLE ROCK", na=False)
        scoped = df[city_mask]
        scoped_districts = sorted(scoped["District"].dropna().astype(str).unique().tolist())
        if not scoped_districts:
            st.error(f"No monitored districts found for city {selected_city}.")
            return

        st.session_state.location = "Little Rock"
        normalized_selection = [canonical_crime_name(c) for c in selected_now]
        st.session_state.selected_crimes = normalized_selection
        st.session_state.selected_crime = ""
        st.session_state.selected_crime_user_selected = False
        if normalized_selection:
            st.session_state.active_crime = normalized_selection[0]
        st.session_state["selected_districts"] = scoped_districts

        if not normalized_selection:
            st.warning("Select at least one crime type.")
        else:
            st.session_state.monitoring_started = True
            st.session_state.page = "dashboard"
            st.rerun()

    st.markdown(
        "<div class='setup-copyright'>© 2024 PULS3 Civic Technology Systems. All rights reserved.</div>",
        unsafe_allow_html=True,
    )


def crime_type_trends(df: pd.DataFrame, selected_crimes: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [{"crime": c, "current": 0, "previous": 0, "pct_change": 0.0, "low_data": True} for c in selected_crimes]
        )

    # Ensure timestamp column is datetime before window math.
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            return pd.DataFrame(
                [{"crime": c, "current": 0, "previous": 0, "pct_change": 0.0, "low_data": True} for c in selected_crimes]
            )

    today = df["Date"].max()
    current_start = today - pd.Timedelta(days=30)
    previous_start = today - pd.Timedelta(days=60)
    current_period = df[(df["Date"] >= current_start) & (df["Date"] <= today)]
    previous_period = df[(df["Date"] >= previous_start) & (df["Date"] < current_start)]

    rows = []
    for c in selected_crimes:
        crime_name = canonical_crime_name(c)
        current = int((current_period["Primary_Type"] == crime_name).sum())
        previous = int((previous_period["Primary_Type"] == crime_name).sum())
        pct = ((current - previous) / max(previous, 1)) * 100
        rows.append(
            {
                "crime": crime_name,
                "current": current,
                "previous": previous,
                "pct_change": pct,
                "low_data": False,
            }
        )
    return pd.DataFrame(rows)


def get_alert_log(df: pd.DataFrame, selected_crimes: List[str], districts: List[str]) -> pd.DataFrame:
    filtered = df[df["Primary_Type"].isin(selected_crimes)].copy()
    if districts:
        filtered = filtered[filtered["District"].isin(districts)]

    rec = filtered.sort_values("Date", ascending=False).head(5).copy()
    rec["Time"] = rec["Date"].dt.strftime("%I:%M %p")
    rec["Type"] = rec["Primary_Type"]
    rec["Location"] = "District " + rec["District"].astype(str)

    severity_map = {
        "Assault": "Critical",
        "Battery": "Elevated",
        "Robbery": "Elevated",
        "Theft": "Low",
        "Burglary": "Moderate",
        "Motor Vehicle Theft": "Moderate",
    }
    rec["Severity"] = rec["Type"].map(severity_map).fillna("Moderate")
    return rec[["Time", "Type", "Location", "Severity"]]


def resolve_location_filter(df: pd.DataFrame, location_text: str) -> pd.DataFrame:
    q = (location_text or "").strip().lower()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for col in ["City", "Zipcode", "Neighborhood", "District"]:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
    filtered = df[mask]
    return filtered


def normalize_zip(value: str) -> str:
    m = re.search(r"\b(\d{5})\b", str(value or ""))
    return m.group(1) if m else ""


def get_little_rock_zipcodes(df: pd.DataFrame) -> set:
    z = (
        df.get("Zipcode", pd.Series([], dtype=str))
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
        .dropna()
    )
    return set(z.tolist())


def get_zone_counts(df: pd.DataFrame, crime: str, days: int = 14) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["zone", "count", "severity"])
    end = df["Date"].max()
    start = end - pd.Timedelta(days=days)
    scoped = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    scoped = scoped[scoped["Primary_Type"] == crime]
    zone_col = "Neighborhood" if "Neighborhood" in scoped.columns else "District"
    counts = scoped.groupby(zone_col).size().reset_index(name="count").sort_values("count", ascending=False)
    if counts.empty:
        return pd.DataFrame(columns=["zone", "count", "severity"])
    counts = counts.rename(columns={zone_col: "zone"})
    counts = counts.head(3).reset_index(drop=True)
    severity_scale = ["Critical", "Elevated", "Moderate"]
    counts["severity"] = [severity_scale[i] for i in range(len(counts))]
    return counts


def get_peak_hours_and_pattern(df: pd.DataFrame, crime: str) -> Tuple[str, str]:
    scoped = df[df["Primary_Type"] == crime].copy()
    if scoped.empty:
        return "8PM-11PM", "Pattern unavailable"
    scoped["hour"] = scoped["Date"].dt.hour
    peak_hour = int(scoped["hour"].mode().iloc[0])
    end_hour = (peak_hour + 3) % 24
    peak_hours = f"{peak_hour:02d}:00-{end_hour:02d}:00"
    weekend_share = scoped["Date"].dt.dayofweek.isin([5, 6]).mean()
    night_share = scoped["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3]).mean()
    if weekend_share > 0.4:
        pattern = "Weekend clustering"
    elif night_share > 0.45:
        pattern = "Night-time spike"
    else:
        pattern = "Weekday spread"
    return peak_hours, pattern


def arima_forecast_next_7(df: pd.DataFrame, crime: str) -> Tuple[pd.Series, pd.Series]:
    scoped = df[df["Primary_Type"] == crime].copy()
    if scoped.empty:
        empty_idx = pd.date_range(pd.Timestamp.now().normalize(), periods=7, freq="D")
        z = pd.Series([0.0] * 7, index=empty_idx, name="forecast")
        return z, z
    daily = scoped.set_index("Date").resample("D").size().astype(float).rename("incidents")
    daily = daily.tail(90)
    if HAS_STATSMODELS:
        try:
            model = ARIMA(daily, order=(2, 1, 2))
            fit = model.fit()
            fc = fit.forecast(steps=7).rename("forecast")
            fc[fc < 0] = 0.0
        except Exception:
            avg = float(daily.tail(14).mean()) if not daily.empty else 0.0
            idx = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=7, freq="D")
            fc = pd.Series([avg] * 7, index=idx, name="forecast")
    else:
        avg = float(daily.tail(14).mean()) if not daily.empty else 0.0
        idx = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=7, freq="D")
        fc = pd.Series([avg] * 7, index=idx, name="forecast")
    return daily, fc


def ensure_alert_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ALERT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                crime_type TEXT NOT NULL,
                location TEXT NOT NULL,
                severity TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_alert(crime_type: str, location: str, severity: str) -> None:
    ensure_alert_db()
    with sqlite3.connect(ALERT_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO alerts(created_at, crime_type, location, severity) VALUES (?, ?, ?, ?)",
            (dt.datetime.now().isoformat(), crime_type, location, severity),
        )
        conn.commit()


def fetch_recent_alerts(limit: int = 12) -> pd.DataFrame:
    ensure_alert_db()
    with sqlite3.connect(ALERT_DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT created_at, crime_type, location, severity FROM alerts ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    if df.empty:
        return pd.DataFrame(columns=["Time", "Type", "Location", "Severity"])
    dt_col = pd.to_datetime(df["created_at"], errors="coerce")
    out = pd.DataFrame(
        {
            "Time": dt_col.dt.strftime("%I:%M %p").fillna(""),
            "Type": df["crime_type"].astype(str),
            "Location": df["location"].astype(str),
            "Severity": df["severity"].astype(str),
        }
    )
    return out


def build_pdf_report(
    location: str,
    selected_crime: str,
    percent_change: float,
    current_count: int,
    previous_count: int,
    spike_detected: bool,
    high_risk_zones: pd.DataFrame,
    peak_hours: str,
    pattern: str,
    forecast_summary: str,
    trend_series: pd.Series,
) -> bytes:
    if not HAS_FPDF:
        raise RuntimeError("PDF generation requires the 'fpdf' package.")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PULS3 Crime Risk Report", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %I:%M %p')}", ln=True)
    pdf.cell(0, 8, f"Location: {location or 'Little Rock'}", ln=True)
    pdf.cell(0, 8, f"Crime Type: {selected_crime}", ln=True)
    pdf.cell(0, 8, f"30-day Change: {percent_change:+.1f}% (Current: {current_count} | Previous: {previous_count})", ln=True)
    pdf.cell(0, 8, f"Spike Status: {'Spike Detected' if spike_detected else 'No Spike'}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        8,
        f"{selected_crime} incidents changed {percent_change:+.1f}% in the last 30 days.\n"
        f"Peak incident hours: {peak_hours}.\n"
        f"Pattern detected: {pattern}.\n"
        f"{forecast_summary}",
    )
    if HAS_MATPLOTLIB and (not trend_series.empty):
        temp_chart_path = None
        try:
            fig, ax = plt.subplots(figsize=(6.4, 2.4))
            ax.plot(trend_series.index, trend_series.values, color="#8B1D2C", linewidth=2.5)
            ax.set_title(f"{selected_crime} - Past 14 Days Trend", fontsize=10)
            ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_chart_path = tmp.name
            fig.savefig(temp_chart_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            pdf.ln(2)
            pdf.image(temp_chart_path, w=180)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
        finally:
            if temp_chart_path:
                try:
                    Path(temp_chart_path).unlink(missing_ok=True)
                except Exception:
                    pass
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "High-Risk Zones", ln=True)
    pdf.set_font("Arial", "", 11)
    if high_risk_zones.empty:
        pdf.cell(0, 8, "No hotspot zones available.", ln=True)
    else:
        for _, row in high_risk_zones.head(3).iterrows():
            pdf.cell(0, 8, f"{row['zone']} - {row['severity']}", ln=True)
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="ignore")
    return bytes(out)


def render_dashboard(df: pd.DataFrame, bundle: ModelBundle) -> None:
    selected_crimes = ["Assault", "Robbery", "Theft"]
    st.session_state.selected_crimes = selected_crimes
    selected_city = (st.session_state.location or "Little Rock").strip()
    if "LITTLE ROCK" not in selected_city.upper():
        selected_city = "Little Rock"
        st.session_state.location = "Little Rock"
    location = "Little Rock"
    city_series = df["City"].astype(str)
    location_df = df[city_series.str.contains("LITTLE ROCK", case=False, na=False)].copy()
    if location_df.empty:
        location_df = df.copy()
    selected_districts = sorted(location_df["District"].dropna().astype(str).unique().tolist())
    if not selected_districts:
        selected_districts = st.session_state.get("selected_districts", [])
    # Overview cards should be computed on city data; crime-level splits are done inside crime_type_trends.
    filtered_df = location_df[
        location_df["Primary_Type"].astype(str).str.strip().str.lower().isin([c.lower() for c in selected_crimes])
    ].copy()
    if filtered_df.empty:
        filtered_df = location_df.copy()

    risk_df = get_latest_zone_risk(bundle)
    if selected_districts:
        risk_df = risk_df[risk_df["District"].isin(selected_districts)]
    system_status = "NORMAL"

    trends = crime_type_trends(filtered_df, selected_crimes)
    positive_trends = trends[trends["pct_change"] > 0].copy()
    if not positive_trends.empty:
        critical_row = positive_trends.sort_values("pct_change", ascending=False).iloc[0]
    elif not trends.empty:
        critical_row = trends.sort_values("pct_change", ascending=False).iloc[0]
    else:
        critical_row = pd.Series({"crime": "Assault", "pct_change": 0.0, "current": 0})
    most_critical_crime = canonical_crime_name(critical_row.get("crime", "Assault"))
    most_critical_spike = (
        float(critical_row.get("pct_change", 0.0)) >= 10.0
        and int(critical_row.get("current", 0)) >= 5
    )

    selected_crime_state = canonical_crime_name(st.session_state.get("selected_crime", ""))
    user_selected = bool(st.session_state.get("selected_crime_user_selected", False))
    if (not user_selected) and (selected_crime_state not in selected_crimes):
        selected_crime_state = most_critical_crime
        st.session_state.selected_crime = selected_crime_state
    if selected_crime_state not in selected_crimes:
        selected_crime_state = most_critical_crime if most_critical_crime in selected_crimes else selected_crimes[0]
        st.session_state.selected_crime = selected_crime_state
    selected_crime = selected_crime_state
    st.session_state.active_crime = selected_crime
    selected_crime_df = location_df[location_df["Primary_Type"] == selected_crime].copy()

    alerts = fetch_recent_alerts(limit=5)
    if alerts.empty:
        alerts = get_alert_log(filtered_df, selected_crimes, selected_districts)

    focus = trends[trends["crime"] == selected_crime].head(1)
    if focus.empty:
        focus_row = pd.Series({"crime": selected_crime, "current": 0, "previous": 0, "pct_change": 0.0})
    else:
        focus_row = focus.iloc[0]
    selected_spike_detected = (
        float(focus_row["pct_change"]) >= 10.0
        and int(focus_row["current"]) >= 5
    )
    system_status = "ELEVATED" if most_critical_spike else "NORMAL"

    history_daily, forecast_7 = arima_forecast_next_7(selected_crime_df, selected_crime)
    trend_src = history_daily.tail(14).rename("incidents")
    if trend_src.empty:
        trend_src = pd.Series([0, 1, 0, 2, 3, 2, 4], index=pd.date_range(dt.datetime.now(), periods=7, freq="D"), name="incidents")

    if len(history_daily) >= 14:
        prev7 = float(history_daily.iloc[-14:-7].sum())
        last7 = float(history_daily.iloc[-7:].sum())
        growth = ((last7 - prev7) / max(prev7, 1.0)) * 100
    else:
        growth = 0.0

    peak_hours_txt, pattern_txt = get_peak_hours_and_pattern(selected_crime_df, selected_crime)
    zone_counts = get_zone_counts(selected_crime_df, selected_crime, days=14)
    top_zone = zone_counts.iloc[0]["zone"] if not zone_counts.empty else "Little Rock Core"
    top_sev = zone_counts.iloc[0]["severity"] if not zone_counts.empty else "Moderate"
    forecast_summary = (
        f"Expected incidents next week: {forecast_7.sum():.1f} total ({forecast_7.mean():.1f}/day average)."
        if not forecast_7.empty
        else "ARIMA forecasting unavailable due to insufficient data."
    )
    logo_path = Path("assets/puls3-logo.png")
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        topbar_logo = f"<img class='topbar-logo-img' src='data:image/png;base64,{logo_b64}' alt='PULS3 logo' />"
    else:
        topbar_logo = "<span class='topbar-logo-fallback'>PULS3</span>"

    st.markdown(
        """
        <style>
            .topbar {
                display: flex; align-items: center; justify-content: space-between;
                padding: 14px 40px; background: white; border-bottom: 1px solid #E5E7EB;
                border-radius: 0; margin: -1.2rem calc(50% - 50vw) 18px;
            }
            .topbar-logo-wrap{
                display:flex; align-items:center;
            }
            .topbar-logo-img{
                height: 34px; width: auto; display: block;
            }
            .topbar-logo-fallback{
                font-weight: 800; color: #7A1E24; font-size: 28px; letter-spacing: 0.01em;
            }
            .topbar-city{
                color: #374151; font-size: 19px; font-weight: 600; letter-spacing: 0.01em;
            }
            .status-banner{
                background:#8B1D2C; color:white; padding:16px 22px; border-radius:12px; margin-top:18px; margin-bottom: 18px;
                display:flex; justify-content:space-between; align-items:center; font-weight:600;
            }
            .trend-wrap-title {
                margin: 10px 0 14px; color: #1f2937; font-size: 34px; font-weight: 700;
            }
            .trend-card {
                background: white; border: 1px solid #E5E7EB; border-radius: 14px; padding: 22px;
                min-height: 120px; position: relative;
            }
            .trend-card.active { border: 2px solid #8B1D2C; box-shadow: 0 0 0 2px rgba(139,29,44,0.08) inset; }
            .trend-select {
                margin-top: 8px;
            }
            .trend-select button {
                width: 100% !important;
                background: #ffffff !important;
                color: #7A1E24 !important;
                border: 1px solid #E5E7EB !important;
                border-radius: 10px !important;
                font-weight: 700 !important;
            }
            .trend-title {
                font-size: 12px; letter-spacing: .08em; color: #6B7280; font-weight: 700; text-transform: uppercase;
            }
            .trend-value { font-size: 32px; font-weight: 800; margin-top: 10px; }
            .trend-meta { margin-top: 6px; color: #6B7280; font-size: 13px; }
            .trend-icon {
                position: absolute; right: 18px; top: 18px; font-size: 18px; color: #8B1D2C;
            }
            .dash-card {
                border: 1px solid #e4dfe1; border-radius: 14px; background: #fff; overflow: hidden;
                margin-top: 16px;
            }
            .dash-card-head {
                padding: 16px 22px; border-bottom: 1px solid #ece8ea; display:flex; justify-content:space-between; align-items:center;
                font-size: 32px; font-weight: 700; color: #1f2937;
            }
            .dash-table { width: 100%; border-collapse: collapse; }
            .dash-table th {
                text-align: left; font-size: 17px; color: #7a6b70; text-transform: uppercase; letter-spacing: .04em; padding: 14px 22px;
                background: #f8f6f7;
            }
            .dash-table td { padding: 14px 22px; border-top: 1px solid #f0ecee; font-size: 27px; color: #202126; }
            .sev-chip { border-radius: 999px; font-size: 16px; font-weight: 700; padding: 4px 10px; display: inline-block; }
            .sev-critical { background:#f9e6ea; color:#a71935; }
            .sev-elevated { background:#fff0df; color:#cf5c00; }
            .sev-moderate { background:#fff8de; color:#ac7b00; }
            .sev-low { background:#e8f6ef; color:#0f9f6e; }
            .analysis-wrap { padding: 20px 22px 22px; }
            .analysis-title { margin: 0; font-size: 46px; color:#201f22; font-weight: 800; }
            .analysis-sub { margin: 2px 0 16px; color:#7a6b70; font-size: 27px; }
            .risk-title { margin: 0 0 12px; font-size: 23px; letter-spacing: .06em; text-transform: uppercase; color:#7a6b70; }
            .risk-row {
                display:flex; justify-content:space-between; border-radius: 10px; padding: 10px 12px;
                margin-bottom: 10px; border: 1px solid #eee7ea; font-size: 26px;
            }
            .kpi {
                border: 1px solid #ece8ea; border-radius: 12px; background: #fff; padding: 12px 14px;
                margin-top: 10px;
            }
            .kpi-label { color:#7a6b70; font-size: 16px; text-transform: uppercase; letter-spacing:.04em; font-weight:700; }
            .kpi-val { color:#1f2937; font-size: 30px; font-weight:700; margin-top:4px; }
            .dash-footer {
                margin-top: 22px; color: #7a6b70; font-size: 17px; display: flex; justify-content: space-between;
                border-top: 1px solid #e5e1e3; padding-top: 14px;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]{
                background: #FFFFFF;
                border-radius: 20px;
                border: 1px solid #EEE7EA !important;
                box-shadow: 0 8px 24px rgba(25, 18, 24, 0.05);
                padding: 20px 18px 20px 18px;
                margin-top: 20px;
            }
            .spike-header{
                margin-bottom: 18px;
            }
            .spike-header h2{
                margin: 0;
                color: #1F2937;
                font-size: 24px;
                line-height: 1.2;
                font-weight: 800;
            }
            .spike-sub{
                color: #6B7280;
                margin-top: 6px;
                font-size: 15px;
            }
            .spike-actions-row{
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                align-items: center;
            }
            .download-link{
                width: 100%;
                min-height: 52px;
                border-radius: 12px;
                border: 1px solid #DDD6DA;
                background: #FFFFFF;
                color: #1F2937;
                font-size: 18px;
                font-weight: 700;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                box-sizing: border-box;
            }
            .download-link:hover,
            .download-link:focus,
            .download-link:active{
                background: #FFFFFF;
                color: #1F2937;
                text-decoration: none;
                border: 1px solid #DDD6DA;
            }
            .download-link.disabled{
                background: #F8FAFC;
                color: #7D8590;
                border: 1px solid #D6DCE3;
                cursor: not-allowed;
                pointer-events: none;
            }
            .trend-box{
                background: #FCFBFC;
                border: 1px dashed #E8E1E5;
                border-radius: 14px;
                padding: 16px 16px 8px 16px;
            }
            .trend-svg-wrap{
                width: 100%;
                height: 320px;
                border-radius: 10px;
                overflow: hidden;
                background: #FCFBFC;
            }
            .trend-svg-wrap svg{
                width: 100%;
                height: 100%;
                display: block;
            }
            .trend-label{
                font-size: 12px;
                letter-spacing: .08em;
                color: #6B7280;
                margin-bottom: 10px;
                text-transform: uppercase;
                font-weight: 700;
            }
            .risk-title{
                font-size: 12px;
                letter-spacing: .08em;
                color: #6B7280;
                margin-bottom: 14px;
                text-transform: uppercase;
                font-weight: 700;
            }
            .risk-item{
                display: flex;
                justify-content: space-between;
                padding: 12px 14px;
                border-radius: 10px;
                margin-bottom: 14px;
                font-weight: 600;
                border: 1px solid transparent;
            }
            .risk-critical{
                background: #FDECEC;
                color: #991B1B;
                border-left: 5px solid #991B1B;
            }
            .risk-elevated{
                background: #FFF3E8;
                color: #C2410C;
                border-left: 5px solid #F97316;
            }
            .risk-moderate{
                background: #FFF8E1;
                color: #B45309;
                border-left: 5px solid #F59E0B;
            }
            .risk-low{
                background: #ECFDF5;
                color: #047857;
                border-left: 5px solid #10B981;
            }
            .risk-badge{
                font-weight: 800;
                letter-spacing: .02em;
            }
            .insight-card{
                background: #FFFFFF;
                border-radius: 12px;
                padding: 14px 14px;
                border: 1px solid #E5E7EB;
                display: flex;
                align-items: center;
                gap: 10px;
                min-height: 78px;
            }
            .insight-icon{
                width: 34px;
                height: 34px;
                border-radius: 999px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: #F7ECEF;
                color: #8B1D2C;
                font-size: 16px;
                font-weight: 700;
                flex: 0 0 34px;
            }
            .insight-copy{
                display: flex;
                flex-direction: column;
            }
            .insight-title{
                font-size: 11px;
                color: #6B7280;
                letter-spacing: .08em;
                text-transform: uppercase;
                font-weight: 700;
            }
            .insight-value{
                font-size: 22px;
                font-weight: 700;
                margin-top: 3px;
                color: #1f2937;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] > button{
                border-radius: 12px !important;
                min-height: 52px !important;
                font-size: 18px !important;
                font-weight: 700 !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stDownloadButton"] > button,
            div[data-testid="stVerticalBlockBorderWrapper"] .stDownloadButton > button{
                border-radius: 12px !important;
                min-height: 52px !important;
                font-size: 18px !important;
                font-weight: 700 !important;
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
                opacity: 1 !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
            }
            .spike-actions-row div[data-testid="stDownloadButton"] > button{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
            }
            .spike-actions-row .stDownloadButton button,
            .spike-actions-row div[data-testid="stDownloadButton"] button{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
                -webkit-text-fill-color: #1F2937 !important;
                opacity: 1 !important;
            }
            .spike-actions-row button[kind="secondary"],
            .spike-actions-row button[data-testid="baseButton-secondary"]{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
                -webkit-text-fill-color: #1F2937 !important;
                opacity: 1 !important;
            }
            .spike-actions-row .stDownloadButton button *,
            .spike-actions-row div[data-testid="stDownloadButton"] button *{
                color: inherit !important;
                -webkit-text-fill-color: currentColor !important;
            }
            .spike-actions-row div[data-testid="stDownloadButton"] > button:hover,
            .spike-actions-row div[data-testid="stDownloadButton"] > button:focus,
            .spike-actions-row div[data-testid="stDownloadButton"] > button:active{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
            }
            .spike-actions-row button[kind="secondary"]:hover,
            .spike-actions-row button[kind="secondary"]:focus,
            .spike-actions-row button[kind="secondary"]:active{
                background: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #DDD6DA !important;
                box-shadow: none !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] > button:disabled,
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stDownloadButton"] > button:disabled{
                opacity: 1 !important;
                cursor: not-allowed !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stDownloadButton"] > button:disabled{
                background: #FFFFFF !important;
                color: #9CA3AF !important;
                border: 1px solid #E5E7EB !important;
                -webkit-text-fill-color: #9CA3AF !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]:disabled{
                background: #F8FAFC !important;
                color: #7D8590 !important;
                border: 1px solid #D6DCE3 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #7D8590 !important;
            }
            .spike-actions-row .stDownloadButton button:disabled,
            .spike-actions-row button[kind="secondary"]:disabled{
                background: #F8FAFC !important;
                color: #7D8590 !important;
                border: 1px solid #D6DCE3 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #7D8590 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-logo-wrap">{topbar_logo}</div>
            <div class="topbar-city">Little Rock</div>
        </div>
        <div class="status-banner">
            <div class="status-left">🛡 SYSTEM STATUS: {system_status}</div>
            <div class="status-right">{'Early Warning: ' + most_critical_crime + ' - ' + location if most_critical_spike else 'No elevated warning - ' + location}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("alert_message"):
        st.success(st.session_state["alert_message"])
        st.session_state["alert_message"] = ""

    st.markdown("<div class='trend-wrap-title'>Crime Type Trends (Last 30 Days)</div>", unsafe_allow_html=True)
    st.caption("Compared to previous 30-day period.")
    cols = st.columns(3)
    for i, row in trends.iterrows():
        if i > 2:
            break
        with cols[i]:
            is_up = row["pct_change"] > 0
            change_color = "#8B1D2C" if is_up else "#1F9D66"
            if row["pct_change"] > 0:
                direction = "Increase"
                trend_icon = "↗"
            elif row["pct_change"] < 0:
                direction = "Decrease"
                trend_icon = "↘"
            else:
                direction = "No Change"
                trend_icon = "→"
            active_class = " active" if row["crime"] == selected_crime else ""
            st.markdown(
                f"""
                <div class='trend-card{active_class}'>
                    <div class='trend-title'>{row['crime'].upper()}</div>
                    <div class='trend-icon' style='color:{change_color};'>{trend_icon}</div>
                    <div class='trend-value' style='color:{change_color};'>
                        {f"{abs(row['pct_change']):.0f}% {direction}"}
                    </div>
                    <div class='trend-meta'>
                        Current: {int(row['current'])} | Previous: {int(row['previous'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='trend-select'>", unsafe_allow_html=True)
            if st.button(f"Select {row['crime']}", key=f"select_crime_{row['crime']}", use_container_width=True):
                st.session_state.selected_crime = row["crime"]
                st.session_state.selected_crime_user_selected = True
                st.session_state.active_crime = row["crime"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Selected Crime: {selected_crime}")

    def render_spike_header() -> None:
        if selected_spike_detected:
            title_txt = f"+{abs(focus_row['pct_change']):.0f}% Spike Detected - {selected_crime}"
            sub_txt = f"{selected_crime} incidents are increasing faster than expected"
        elif float(focus_row["pct_change"]) < 0:
            title_txt = f"{abs(focus_row['pct_change']):.0f}% Decrease - {selected_crime}"
            sub_txt = f"{selected_crime} incidents are lower than the previous 30-day period."
        else:
            title_txt = f"No Significant Spike - {selected_crime}"
            sub_txt = f"{selected_crime} incident activity is currently within expected range."
        h1, h2 = st.columns([1.9, 1.45])
        with h1:
            st.markdown(
                f"""
                <div class="spike-header">
                    <h2>{title_txt}</h2>
                    <p class="spike-sub">{sub_txt}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown("<div class='spike-actions-row'>", unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            with a1:
                if st.button(
                    "🛡 Alert Police Department",
                    key=f"alert_police_{selected_crime}",
                    use_container_width=True,
                    disabled=not selected_spike_detected,
                ):
                    insert_alert(selected_crime, str(top_zone), str(top_sev))
                    st.session_state["alert_message"] = "Police Department Alert Triggered"
                    st.rerun()
            with a2:
                if HAS_FPDF:
                    report_bytes = build_pdf_report(
                        location=location or "Little Rock",
                        selected_crime=selected_crime,
                        percent_change=float(focus_row["pct_change"]),
                        current_count=int(focus_row["current"]),
                        previous_count=int(focus_row["previous"]),
                        spike_detected=selected_spike_detected,
                        high_risk_zones=zone_counts,
                        peak_hours=peak_hours_txt,
                        pattern=pattern_txt,
                        forecast_summary=forecast_summary,
                        trend_series=trend_src,
                    )
                    report_b64 = base64.b64encode(report_bytes).decode("ascii")
                    st.markdown(
                        f"""
                        <a class="download-link"
                           style="width:100%;min-height:52px;border-radius:12px;border:1px solid #DDD6DA;background:#FFFFFF;color:#1F2937;font-size:18px;font-weight:700;text-decoration:none;display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;"
                           download="puls3_crime_report.pdf"
                           href="data:application/pdf;base64,{report_b64}">
                           ↓ Download Report
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <span class="download-link disabled"
                              style="width:100%;min-height:52px;border-radius:12px;border:1px solid #D6DCE3;background:#F8FAFC;color:#7D8590;font-size:18px;font-weight:700;text-decoration:none;display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;">
                              ↓ Download Report
                        </span>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            if not HAS_FPDF:
                st.caption("Install `fpdf2` to enable PDF report downloads.")

    def render_spike_body() -> None:
        chart_col, risk_col = st.columns([2.2, 1])
        with chart_col:
            st.markdown(
                """
                <div class="trend-box">
                <div class="trend-label">PAST 14 DAYS TREND</div>
                """,
                unsafe_allow_html=True,
            )
            smooth_src = trend_src.rolling(3, min_periods=1).mean()
            if HAS_MATPLOTLIB and not smooth_src.empty:
                fig, ax = plt.subplots(figsize=(8.4, 3.1), facecolor="#FCFBFC")
                ax.set_facecolor("#FCFBFC")
                x = np.arange(len(smooth_src))
                y = smooth_src.values.astype(float)
                ax.plot(
                    x,
                    y,
                    color="#8B1D2C",
                    linewidth=5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )
                ax.scatter([x[-1]], [y[-1]], s=130, color="#8B1D2C", zorder=5)
                ax.set_xlim(x.min(), x.max())
                for spine in ["top", "right", "left", "bottom"]:
                    ax.spines[spine].set_visible(False)
                ax.grid(False)
                ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
                fig.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True, clear_figure=True)
            else:
                st.line_chart(smooth_src, color="#8B1D2C")
            if not forecast_7.empty:
                st.caption(
                    f"Expected incidents next week: {forecast_7.sum():.1f}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with risk_col:
            st.markdown('<div class="risk-title">HIGH-RISK ZONES</div>', unsafe_allow_html=True)
            if zone_counts.empty:
                st.caption("No hotspot locations available.")
            for _, row in zone_counts.head(3).iterrows():
                sev = row["severity"]
                color_class = {
                    "Critical": "risk-critical",
                    "Elevated": "risk-elevated",
                    "Moderate": "risk-moderate",
                    "Low": "risk-low",
                }.get(sev, "risk-low")
                st.markdown(
                    f"""
                    <div class="risk-item {color_class}">
                        <span>{row['zone']}</span>
                        <span class="risk-badge">{sev.upper()}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def render_spike_insights() -> None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-icon">↗</div>
                    <div class="insight-copy">
                        <div class="insight-title">GROWTH RATE</div>
                        <div class="insight-value">{growth:+.1f}% last 7 days</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-icon">◷</div>
                    <div class="insight-copy">
                        <div class="insight-title">PEAK HOURS</div>
                        <div class="insight-value">{peak_hours_txt}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-icon">▣</div>
                    <div class="insight-copy">
                        <div class="insight-title">PATTERN</div>
                        <div class="insight-value">{pattern_txt}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        render_spike_header()
        render_spike_body()
        render_spike_insights()

    severity_css = {
        "critical": "sev-critical",
        "elevated": "sev-elevated",
        "moderate": "sev-moderate",
        "low": "sev-low",
    }
    alert_rows = ""
    for _, row in alerts.head(5).iterrows():
        sev = str(row["Severity"]).lower()
        alert_rows += (
            f"<tr><td>{row['Time']}</td><td>{row['Type']}</td><td>{row['Location']}</td>"
            f"<td><span class='sev-chip {severity_css.get(sev, 'sev-moderate')}'>{row['Severity']}</span></td></tr>"
        )
    st.markdown(
        f"""
        <div class='dash-card'>
            <div class='dash-card-head'><span>Recent Alert Log</span><span style='font-size:16px;color:#7A1E24;font-weight:700;'>View All</span></div>
            <table class='dash-table'>
                <thead><tr><th>Time</th><th>Type</th><th>Location</th><th>Severity</th></tr></thead>
                <tbody>{alert_rows}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='dash-footer'><span>PULS3 SYSTEM V2.4.1</span><span>Internal Security Monitoring - Access restricted to authorized analysts only.</span><span>System Status &nbsp;&nbsp; API Docs &nbsp;&nbsp; Privacy Protocol</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    inject_css()

    try:
        with st.spinner("Loading crime data and prediction model..."):
            df = load_little_rock_data()
            bundle = load_model_bundle(df)
            st.session_state["data_source"] = "little_rock"
    except Exception as exc:
        st.error(
            "Could not load Little Rock dataset/model. Add your exported Little Rock CSV under ./data and model artifact under ./models."
        )
        st.exception(exc)
        st.stop()

    page = st.session_state.page
    if page == "login":
        render_login()
    elif page == "setup":
        render_setup(df)
    else:
        render_dashboard(df, bundle)


if __name__ == "__main__":
    main()
