#!/usr/bin/env python3
"""
Working Medical Face Mask Detection - Monitoring Dashboard
Streamlit dashboard based on the working app structure
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
import requests
from pathlib import Path
import sys
from src.monitoring.metrics_collector import MetricsCollector

# MLflow integration for dashboard summary logging
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
MLFLOW_EXPERIMENT_NAME = "MedicalDashboard"
def get_or_create_mlflow_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
MLFLOW_EXPERIMENT_ID = get_or_create_mlflow_experiment(MLFLOW_EXPERIMENT_NAME)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Page configuration with medical theme
st.set_page_config(
    page_title="🔬 Medical Face Mask Monitoring System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Medical Cyberpunk Theme CSS
MEDICAL_DASHBOARD_CSS = """
<style>
/* ... (keep your CSS here, omitted for brevity) ... */
</style>
"""

def get_api_stats():
    """Get statistics from the API."""
    try:
        response = requests.get("http://localhost:8001/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def get_api_health():
    """Get health status from the API."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            health_data['api_status'] = 'online'
            return health_data
    except Exception:
        pass
    return {
        'api_status': 'offline',
        'status': 'API Offline',
        'uptime_minutes': 0
    }

def load_webcam_detections():
    """
    Load detections from webcam predictions in data/collected/webcam_detections/images and labels.
    Each detection is a dict with: timestamp, image_file, label_file, class, confidence.
    """
    images_dir = Path("data/collected/webcam_detections/images")
    labels_dir = Path("data/collected/webcam_detections/labels")
    detections = []
    if not images_dir.exists() or not labels_dir.exists():
        return detections

    for img_file in sorted(images_dir.glob("*.jpg")):
        base = img_file.stem
        label_file = labels_dir / f"{base}.txt"
        detection = {
            "timestamp": img_file.stat().st_mtime,
            "image_file": str(img_file),
            "label_file": str(label_file),
            "class": "unknown",
            "confidence": 0.0
        }
        # Try to parse label file (YOLO format: class_id x_center y_center w h [confidence])
        if label_file.exists():
            try:
                with open(label_file, "r") as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        class_map = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
                        detection["class"] = class_map.get(class_id, "unknown")
                        # If confidence is present (YOLOv5+), it's the 6th column
                        if len(parts) >= 6:
                            detection["confidence"] = float(parts[5])
                        else:
                            detection["confidence"] = 1.0  # Assume 1.0 if not present
            except Exception:
                pass
        detections.append(detection)
    # Sort by timestamp descending
    detections.sort(key=lambda x: x["timestamp"], reverse=True)
    # Convert timestamp to readable string
    for d in detections:
        d["timestamp"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    return detections

def aggregate_webcam_stats(detections):
    """Aggregate stats from webcam detections."""
    total = len(detections)
    with_mask = sum(1 for d in detections if d["class"] == "with_mask")
    without_mask = sum(1 for d in detections if d["class"] == "without_mask")
    incorrect_mask = sum(1 for d in detections if d["class"] == "mask_weared_incorrect")
    confidences = [d["confidence"] for d in detections if d["confidence"]]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    return {
        "total_scans": total,
        "with_mask": with_mask,
        "without_mask": without_mask,
        "incorrect_mask": incorrect_mask,
        "average_confidence": avg_confidence,
        "recent_activity": detections[:10]
    }

def main():
    metrics_collector.start_background_collection()
    try:
        st.markdown(MEDICAL_DASHBOARD_CSS, unsafe_allow_html=True)
        st.markdown("""
        <div class="main-header">
            <h1 class="header-title">🔬 MEDICAL MONITORING SYSTEM</h1>
            <p class="header-subtitle">Real-time Face Mask Compliance Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        # Try API first
        stats = get_api_stats()
        health = get_api_health()

        # If API is offline, use webcam detections as fallback
        if not stats or health.get('api_status') != 'online':
            webcam_detections = load_webcam_detections()
            stats = {
                "detection_stats": {
                    "total_scans": len(webcam_detections),
                    "with_mask": sum(1 for d in webcam_detections if d["class"] == "with_mask"),
                    "without_mask": sum(1 for d in webcam_detections if d["class"] == "without_mask"),
                    "incorrect_mask": sum(1 for d in webcam_detections if d["class"] == "mask_weared_incorrect"),
                    "average_confidence": (
                        sum(d["confidence"] for d in webcam_detections if d["confidence"]) / len(webcam_detections)
                        if webcam_detections else 0
                    )
                },
                "recent_activity": webcam_detections[:10]
            }

        detection_stats = stats.get('detection_stats', {})
        total_scans = detection_stats.get('total_scans', 0)
        with_mask = detection_stats.get('with_mask', 0)
        without_mask = detection_stats.get('without_mask', 0)
        incorrect_mask = detection_stats.get('incorrect_mask', 0)
        total = with_mask + without_mask + incorrect_mask
        compliance_rate = (with_mask / total * 100) if total > 0 else 0
        avg_confidence = detection_stats.get('average_confidence', 0) * 100  # percent

        # Log summary metrics to MLflow (optional, e.g. once per dashboard load)
        with mlflow.start_run(run_name="dashboard_summary", experiment_id=MLFLOW_EXPERIMENT_ID):
            mlflow.log_metric("total_scans", total_scans)
            mlflow.log_metric("compliance_rate", compliance_rate)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.log_metric("with_mask", with_mask)
            mlflow.log_metric("without_mask", without_mask)
            mlflow.log_metric("incorrect_mask", incorrect_mask)
            mlflow.set_tag("dashboard", "streamlit")

        # Sidebar controls
        st.sidebar.markdown("## 🏥 System Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 2, 30, 5)
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()

        # API Status
        api_status_text = health.get('api_status', 'unknown')
        if api_status_text == 'online':
            api_status = "🟢 API ONLINE"
            api_color = "status-online"
        elif total_scans > 0:
            api_status = "🟡 API OFFLINE - FILE DATA"
            api_color = "status-warning"
        else:
            api_status = "🔴 API OFFLINE"
            api_color = "status-error"
        st.sidebar.markdown(f'<p class="{api_color}">Status: {api_status}</p>', unsafe_allow_html=True)

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_scans}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{compliance_rate:.1f}%</div>
                <div class="metric-label">Compliance Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            uptime = f"{health.get('uptime_minutes', 0):.0f}m"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{uptime}</div>
                <div class="metric-label">System Uptime</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_confidence:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Real-Time Compliance")
            if total > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[datetime.now()],
                    y=[compliance_rate],
                    mode='lines+markers',
                    name='Compliance Rate',
                    line=dict(color='#00fff9', width=3),
                    marker=dict(size=10, color='#39ff14')
                ))
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Compliance Rate (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detection data available. Upload images or run webcam detection.")

        with col2:
            st.markdown("### 🥧 Detection Distribution")
            if total > 0:
                labels = ['With Mask', 'Without Mask', 'Incorrect Mask']
                values = [with_mask, without_mask, incorrect_mask]
                colors = ['#39ff14', '#ff073a', '#ff6b35']
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=colors,
                    textfont=dict(color='white', size=14)
                )])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detection data available.")

        # Recent Activity
        st.markdown("## 📊 Recent Detection Activity")
        recent_activity = stats.get("recent_activity", []) if stats else []
        if recent_activity:
            df = pd.DataFrame(recent_activity)
            # For webcam detections, rename columns for clarity
            if "image_file" in df.columns:
                df = df.rename(columns={"image_file": "Image", "label_file": "Label", "class": "Class", "confidence": "Confidence", "timestamp": "Time"})
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent detection activity. Upload images or run webcam detection.")

        # System Information
        st.markdown("## 🖥️ System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Detection System:**")
            st.write(f"📊 Total Detections: {total_scans}")
        with col2:
            st.markdown("**API Status:**")
            st.write(f"Status: {health.get('status', 'Unknown')}")
            st.write(f"Port: 8001")

        # Instructions for users
        if total == 0:
            st.markdown("## 🚀 Getting Started")
            st.info("""
            To see live data in this dashboard:

            1. **Start the API**: `python src/inference/api.py`
            2. **Run the webcam app**: `python src/realtime_webcam_app.py`
            3. **Upload images to API** or **use webcam detection**

            This dashboard shows real detection data from the system!
            """)

        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    finally:
        metrics_collector.stop_background_collection()

if __name__ == "__main__":
    main()