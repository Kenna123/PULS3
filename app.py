import datetime as dt
import io
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import certifi
import numpy as np
import pandas as pd
import requests
import streamlit as st

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


@dataclass
class ModelBundle:
    model: object
    encoder: LabelEncoder
    weekly_data: pd.DataFrame
    features: List[str]


@st.cache_data(show_spinner=False)
def load_crime_data(limit: int = 200000) -> pd.DataFrame:
    url = f"https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit={limit}"
    try:
        resp = requests.get(url, timeout=45, verify=certifi.where())
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to download Chicago crime data over HTTPS. Check network/SSL certificates."
        ) from exc

    rename_map = {
        "date": "Date",
        "primary_type": "Primary_Type",
        "district": "District",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    required = ["Date", "Primary_Type", "District"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    df = df.dropna(subset=required).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["District"] = df["District"].astype(str)
    df["Primary_Type"] = df["Primary_Type"].astype(str).str.title()
    df = df[df["Date"].dt.year >= 2018].copy()
    df = df.sort_values("Date")
    return df


@st.cache_data(show_spinner=False)
def generate_offline_crime_data(days: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end = pd.Timestamp.now().floor("h")
    start = end - pd.Timedelta(days=days)

    districts = [str(i) for i in range(1, 26)]
    crime_weights = np.array([0.2, 0.15, 0.28, 0.2, 0.1, 0.07])
    crime_weights = crime_weights / crime_weights.sum()

    rows = []
    for ts in pd.date_range(start, end, freq="h"):
        base_events = rng.poisson(5)
        for _ in range(base_events):
            rows.append(
                {
                    "Date": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
                    "Primary_Type": rng.choice(CRIME_OPTIONS, p=crime_weights),
                    "District": rng.choice(districts),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("Date")
    return df


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
    return ModelBundle(model=model, encoder=le, weekly_data=weekly, features=features)


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
        location_input = st.text_input(
            "Monitoring Area",
            value="",
            placeholder="Search for city or zip code",
            label_visibility="collapsed",
        )
        st.markdown("<div class='helper-text'>Example: Little Rock, AR</div>", unsafe_allow_html=True)

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
        st.session_state.location = location_input.strip() or "Little Rock, AR"
        st.session_state.selected_crimes = selected_now
        st.session_state["selected_districts"] = district_options

        if not st.session_state.selected_crimes:
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
    now = df["Date"].max()
    last_24h = df[df["Date"] >= now - pd.Timedelta(hours=24)]
    prev_24h = df[(df["Date"] < now - pd.Timedelta(hours=24)) & (df["Date"] >= now - pd.Timedelta(hours=48))]

    rows = []
    for c in selected_crimes:
        now_count = int((last_24h["Primary_Type"] == c).sum())
        prev_count = int((prev_24h["Primary_Type"] == c).sum())
        if prev_count == 0:
            pct = 100.0 if now_count > 0 else 0.0
        else:
            pct = ((now_count - prev_count) / prev_count) * 100
        rows.append({"crime": c, "current": now_count, "previous": prev_count, "pct_change": pct})
    return pd.DataFrame(rows)


def get_alert_log(df: pd.DataFrame, selected_crimes: List[str], districts: List[str]) -> pd.DataFrame:
    filtered = df[df["Primary_Type"].isin(selected_crimes)].copy()
    if districts:
        filtered = filtered[filtered["District"].isin(districts)]

    rec = filtered.sort_values("Date", ascending=False).head(8).copy()
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


def render_dashboard(df: pd.DataFrame, bundle: ModelBundle) -> None:
    selected_crimes = st.session_state.selected_crimes
    location = st.session_state.location
    selected_districts = st.session_state.get("selected_districts", [])
    filtered_df = df[df["Primary_Type"].isin(selected_crimes)].copy()
    if filtered_df.empty:
        filtered_df = df.copy()

    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
            <h2 style='margin:0;' class='brand'>PULS3</h2>
            <div style='color:{THEME['muted']}; font-weight:600;'>📍 {location}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("data_source") == "offline":
        st.info("Offline demo mode: live Chicago feed unavailable, using simulated crime data.")

    risk_df = get_latest_zone_risk(bundle)
    if selected_districts:
        risk_df = risk_df[risk_df["District"].isin(selected_districts)]

    top = risk_df.iloc[0] if not risk_df.empty else None
    warning_msg = (
        f"Early Warning: {selected_crimes[0]} risk rising in District {top['District']}"
        if top is not None
        else "Early Warning: Insufficient model signal"
    )
    st.markdown(f"<div class='banner'>SYSTEM STATUS: ELEVATED &nbsp;&nbsp; | &nbsp;&nbsp; {warning_msg}</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Crime Type Trends (Last 24h)")
    trends = crime_type_trends(filtered_df, selected_crimes)
    cols = st.columns(max(1, len(selected_crimes)))
    for i, row in trends.iterrows():
        with cols[i % len(cols)]:
            color = THEME["bad"] if row["pct_change"] > 0 else THEME["good"]
            direction = "Increase" if row["pct_change"] > 0 else "Decrease"
            st.markdown(
                f"""
                <div class='metric'>
                    <h4>{row['crime']}</h4>
                    <div style='font-size:34px; font-weight:800; color:{color};'>{abs(row['pct_change']):.0f}% {direction}</div>
                    <div style='color:{THEME['muted']};'>Current: {int(row['current'])} | Previous: {int(row['previous'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Recent Alert Log")
    alerts = get_alert_log(filtered_df, selected_crimes, selected_districts)
    st.dataframe(alerts, use_container_width=True, hide_index=True)

    st.write("")
    if not trends.empty:
        focus = trends.sort_values("pct_change", ascending=False).iloc[0]
    else:
        focus = pd.Series({"crime": "Assault", "pct_change": 0.0})

    left, right = st.columns([2, 1.1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {focus['pct_change']:+.0f}% Spike Detected - {focus['crime']}")
        st.caption(f"{focus['crime']} incidents are changing vs prior 24-hour baseline.")

        trend_src = (
            filtered_df[filtered_df["Primary_Type"] == focus["crime"]]
            .set_index("Date")
            .resample("D")
            .size()
            .tail(14)
            .rename("incidents")
        )
        st.line_chart(trend_src)

        c1, c2, c3 = st.columns(3)
        if len(trend_src) >= 8:
            prev7 = float(trend_src.iloc[-8:-1].sum())
            last7 = float(trend_src.iloc[-7:].sum())
            growth = ((last7 - prev7) / prev7 * 100) if prev7 > 0 else 0.0
        else:
            growth = 0.0

        focus_df = filtered_df[filtered_df["Primary_Type"] == focus["crime"]]
        peak_hour = int(focus_df["Date"].dt.hour.mode().iloc[0]) if not focus_df.empty else 20

        c1.metric("Growth Rate", f"{growth:+.1f}%")
        c2.metric("Peak Hour", f"{peak_hour:02d}:00-{(peak_hour + 3) % 24:02d}:00")
        c3.metric("Pattern", "Weekend clustering" if focus_df["Date"].dt.dayofweek.isin([5, 6]).mean() > 0.35 else "Weekday spread")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### High-Risk Zones")
        if risk_df.empty:
            st.caption("No zone risks available.")
        else:
            for _, row in risk_df.head(5).iterrows():
                sev = str(row["severity"]).lower()
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; border:1px solid {THEME['line']}; border-radius:10px; padding:10px; margin-bottom:8px; background:white;'>
                        <div>District {row['District']} — Risk: {row['risk_prob']:.2f}</div>
                        <span class='pill pill-{sev}'>{row['severity']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>PULS3 SYSTEM v2.4.1 | Internal Security Monitoring</div>", unsafe_allow_html=True)


def main() -> None:
    init_state()
    inject_css()

    try:
        with st.spinner("Loading crime data and prediction model..."):
            try:
                df = load_crime_data()
                st.session_state["data_source"] = "live"
            except Exception:
                df = generate_offline_crime_data()
                st.session_state["data_source"] = "offline"
            bundle = build_model(df)
    except Exception as exc:
        st.error(
            "Could not load the crime dataset/model. Please ensure required packages are installed."
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
