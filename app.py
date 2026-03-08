import datetime as dt
import io
from dataclasses import dataclass
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
        "location": "Chicago, IL",
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
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h1 class='brand'>PULS3</h1>", unsafe_allow_html=True)
        st.markdown("### When Time, Place, and Risk Align")
        st.caption("PULS3 Alerts")

        email = st.text_input("Email Address", value=st.session_state.email, placeholder="name@agency.gov")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Log In", use_container_width=True):
            if email.strip() and password.strip():
                st.session_state.email = email.strip()
                st.session_state.logged_in = True
                st.session_state.page = "setup"
                st.rerun()
            else:
                st.warning("Please provide email and password.")

        st.markdown("</div>", unsafe_allow_html=True)


def render_setup(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Set Up Monitoring")
        st.caption("Choose your area and select crimes to monitor.")

        district_options = sorted(df["District"].dropna().unique().tolist())
        location_mode = st.radio("Monitoring Area Type", ["City", "District"], horizontal=True)

        if location_mode == "City":
            st.session_state.location = st.text_input("Monitoring Area", value=st.session_state.location, placeholder="City, ST")
            selected_districts = district_options
        else:
            selected_districts = st.multiselect(
                "Districts",
                options=district_options,
                default=district_options[:3],
                help="Model is district-based using Chicago public data.",
            )
            st.session_state.location = f"Chicago Districts ({len(selected_districts)})"

        selected = st.multiselect("Crime Types to Monitor", options=CRIME_OPTIONS, default=st.session_state.selected_crimes)
        if selected:
            st.session_state.selected_crimes = selected

        st.session_state["selected_districts"] = selected_districts

        if st.button("Start Monitoring", use_container_width=True):
            if not st.session_state.selected_crimes:
                st.warning("Select at least one crime type.")
            elif not selected_districts:
                st.warning("Select at least one district.")
            else:
                st.session_state.monitoring_started = True
                st.session_state.page = "dashboard"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


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
    selected_districts = st.session_state.get("selected_districts", [])

    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
            <h2 style='margin:0;' class='brand'>PULS3</h2>
            <div style='color:{THEME['muted']}; font-weight:600;'>📍 {st.session_state.location}</div>
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
        f"Early Warning: {selected_crimes[0]} - District {top['District']}"
        if top is not None
        else "Early Warning: Insufficient model signal"
    )
    st.markdown(f"<div class='banner'>SYSTEM STATUS: ELEVATED &nbsp;&nbsp; | &nbsp;&nbsp; {warning_msg}</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Crime Type Trends (Last 24h)")
    trends = crime_type_trends(df, selected_crimes)
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
    alerts = get_alert_log(df, selected_crimes, selected_districts)
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
            df[df["Primary_Type"] == focus["crime"]]
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

        focus_df = df[df["Primary_Type"] == focus["crime"]]
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
            for _, row in risk_df.head(8).iterrows():
                sev = str(row["severity"]).lower()
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; border:1px solid {THEME['line']}; border-radius:10px; padding:10px; margin-bottom:8px; background:white;'>
                        <div>District {row['District']}</div>
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
