import streamlit as pd
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
from datetime import datetime

# --- CONFIGURATION & CONSTANTS ---
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="SurakshaNet AI Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    .metric-box {
        background-color: #1b1e24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS TO FETCH BACKEND DATA ---
def fetch_summary_stats():
    try:
        response = requests.get(f"{FASTAPI_URL}/logs/summary", timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return None

def fetch_recent_logs(limit=20):
    try:
        response = requests.get(f"{FASTAPI_URL}/logs/recent?limit={limit}", timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return []
    return []

# --- APP LAYOUT ---
st.title("🛡️ SurakshaNet: AI Site Safety & PPE Compliance")
st.subheader("Automated Real-Time Site Safety Inspection Console")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Mode", ["📊 Live Analytics Dashboard", "📷 Real-Time Camera Inference"])
auto_refresh = st.sidebar.checkbox("Auto-Refresh Logs (Every 5s)", value=True)

# --- MODE 1: LIVE ANALYTICS DASHBOARD ---
if app_mode == "📊 Live Analytics Dashboard":
    st.header("📈 Real-Time Compliance Analytics")
    
    # Trigger refresh if active
    if auto_refresh:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} (Auto-refresh active)")
        # This acts like a soft rerun to fetch fresh data from SQLite via FastAPI
        st.button("🔄 Force Refresh Data", key="refresh_btn")
    
    # Fetch Data from Backend
    summary_data = fetch_summary_stats()
    recent_logs = fetch_recent_logs()

    if summary_data is None:
        st.error("❌ Cannot connect to SurakshaNet FastAPI Backend (Port 8000). Please ensure your FastAPI server is running (`uvicorn main:app --reload`).")
    else:
        # 1. Metrics Row
        total_violations = summary_data.get("total_violations", 0)
        by_class = summary_data.get("by_class", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class='metric-box'>
                    <h3 style='color: #ff4b4b; margin:0;'>⚠️ Total Violations</h3>
                    <p style='font-size: 32px; font-weight: bold; margin:0;'>{total_violations}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class='metric-box' style='border-left-color: #f1c40f;'>
                    <h3 style='color: #f1c40f; margin:0;'>👤 Bare Heads Flagged</h3>
                    <p style='font-size: 32px; font-weight: bold; margin:0;'>{by_class.get('Head', 0)}</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class='metric-box' style='border-left-color: #3498db;'>
                    <h3 style='color: #3498db; margin:0;'>👷 Status</h3>
                    <p style='font-size: 24px; font-weight: bold; margin:5px 0 0 0;'>{'⚠️ BREACH' if total_violations > 0 else '✅ SECURE'}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Charts & Insights Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📊 Violation Breakdown by Class")
            if by_class:
                df_chart = pd.DataFrame(list(by_class.items()), columns=["Object Class", "Count"])
                fig = px.bar(df_chart, x="Object Class", y="Count", color="Object Class",
                             color_discrete_map={"Head": "#ff4b4b", "Person": "#3498db"},
                             template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No logs recorded yet. Safe site environments generate no charts!")

        with col_chart2:
            st.subheader("🕒 Safety Breach Log Stream (Live Feed)")
            if recent_logs:
                # Convert logs list into a clean scannable Pandas DataFrame
                df_logs = pd.DataFrame(recent_logs)
                # Rename columns for presentation
                df_logs.columns = ["ID", "Timestamp", "Detected Issue", "Confidence Score", "BBox Matrix"]
                # Display dataframe with custom styling
                st.dataframe(df_logs[["Timestamp", "Detected Issue", "Confidence Score"]].sort_values(by="Timestamp", ascending=False), 
                             use_container_width=True, height=280)
            else:
                st.info("Zero active breaches logged in the database.")

# --- MODE 2: REAL-TIME CAMERA INFERENCE ---
elif app_mode == "📷 Real-Time Camera Inference":
    st.header("📸 Live Web Camera Stream Calibration")
    st.write("Upload or stream frames directly to the backend validator pipeline.")

    # File Uploader as a test sandbox
    uploaded_file = st.file_uploader("Upload a test frame to diagnose labels...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        # Send raw bytes to FastAPI endpoint /detect
        with st.spinner("Processing framework detections via FastAPI..."):
            try:
                response = requests.post(f"{FASTAPI_URL}/detect", files={f"file": file_bytes})
                if response.status_code == 200:
                    result = response.json()
                    
                    # Convert to OpenCV layout to plot boxes if needed
                    st.json(result) # Display structured compliance payload
                    
                    if result["is_compliant"]:
                        st.success("✅ Target frame is Fully Compliant! Helmet protection detected properly.")
                    else:
                        st.error(f"🚨 Security Breach! {result['total_violations']} safety protocols violated.")
                else:
                    st.error("Backend pipeline returned an error code.")
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")