#!/usr/bin/env python3
"""
Working Medical Face Mask Detection - Monitoring Dashboard
Streamlit dashboard based on the working app structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import cv2
from PIL import Image
import base64
import sys
from src.monitoring.metrics_collector import MetricsCollector

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Global metrics collector instance
metrics_collector = MetricsCollector()

def load_real_detections():
    """Load real detection data from detections directory."""
    detections_dir = Path("detections")
    if not detections_dir.exists():
        return []
    
    detections = []
    for json_file in detections_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                detection_data = json.load(f)
                detection_data['filename'] = json_file.stem
                detections.append(detection_data)
        except Exception as e:
            continue
    
    # Sort by timestamp
    detections.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return detections

def get_live_detection_stats():
    """Get live detection statistics from real detection files."""
    detections = load_real_detections()
    
    if not detections:
        return {
            'total_detections': 0,
            'with_mask': 0,
            'without_mask': 0,
            'mask_weared_incorrect': 0,
            'average_confidence': 0,
            'recent_detections': []
        }
    
    # Calculate statistics
    total = len(detections)
    with_mask = sum(1 for d in detections if d.get('class') == 'with_mask')
    without_mask = sum(1 for d in detections if d.get('class') == 'without_mask')
    incorrect = sum(1 for d in detections if d.get('class') == 'mask_weared_incorrect')
    
    confidences = [d.get('confidence', 0) for d in detections if d.get('confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total_detections': total,
        'with_mask': with_mask,
        'without_mask': without_mask,
        'mask_weared_incorrect': incorrect,
        'average_confidence': avg_confidence,
        'recent_detections': detections[:10]  # Last 10 detections
    }

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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* Global Theme Variables */
:root {
    --medical-cyan: #00fff9;
    --medical-green: #39ff14;
    --medical-orange: #ff6b35;
    --medical-red: #ff073a;
    --bg-dark: #0a0a0a;
    --bg-card: #1a1a2e;
    --bg-panel: #16213e;
    --text-primary: #ffffff;
    --text-secondary: #b3b3b3;
}

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-card) 50%, var(--bg-panel) 100%);
    color: var(--text-primary);
    font-family: 'Rajdhani', sans-serif;
}

/* Medical Grid Background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 255, 249, 0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 249, 0.1) 1px, transparent 1px);
    background-size: 30px 30px;
    z-index: -1;
    animation: medicalGridPulse 4s ease-in-out infinite;
    pointer-events: none;
}

@keyframes medicalGridPulse {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 0.4; }
}

/* Header Styling */
.main-header {
    background: linear-gradient(90deg, var(--bg-card), var(--bg-panel));
    border: 2px solid var(--medical-cyan);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(0, 255, 249, 0.3);
    position: relative;
    overflow: hidden;
    text-align: center;
}

.main-header::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 249, 0.1), transparent);
    animation: scanLine 3s infinite;
}

@keyframes scanLine {
    0% { left: -100%; }
    100% { left: 100%; }
}

.header-title {
    font-family: 'Orbitron', monospace;
    font-size: 3em;
    font-weight: 900;
    color: var(--medical-cyan);
    text-shadow: 0 0 20px var(--medical-cyan);
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 3px;
}

.header-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3em;
    color: var(--text-secondary);
    margin-top: 10px;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-panel));
    border: 1px solid var(--medical-cyan);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 0 20px rgba(0, 255, 249, 0.2);
    text-align: center;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.5em;
    font-weight: 700;
    color: var(--medical-cyan);
    text-shadow: 0 0 10px var(--medical-cyan);
}

.metric-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1em;
    color: var(--text-secondary);
    margin-top: 5px;
}

/* Status Indicators */
.status-online {
    color: var(--medical-green);
    text-shadow: 0 0 10px var(--medical-green);
}

.status-warning {
    color: var(--medical-orange);
    text-shadow: 0 0 10px var(--medical-orange);
}

.status-error {
    color: var(--medical-red);
    text-shadow: 0 0 10px var(--medical-red);
}

/* Charts */
.chart-container {
    background: rgba(26, 26, 46, 0.8);
    border: 1px solid rgba(0, 255, 249, 0.3);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 15px rgba(0, 255, 249, 0.1);
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, var(--bg-card), var(--bg-panel));
    border-right: 2px solid var(--medical-cyan);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(45deg, var(--medical-cyan), var(--medical-green));
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 5px 15px rgba(0, 255, 249, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 249, 0.5);
}

/* Data Tables */
.dataframe {
    background: rgba(26, 26, 46, 0.9);
    color: var(--text-primary);
    border: 1px solid var(--medical-cyan);
    border-radius: 10px;
}

</style>
"""

def get_api_stats():
    """Get statistics from the API with fallback to real detection data"""
    try:
        response = requests.get("http://localhost:8001/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback to real detection data
    return get_live_detection_stats()

def get_api_health():
    """Get health status from the API with real detection enhancement"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            health_data['api_status'] = 'online'
            return health_data
    except:
        pass
    
    # API offline, return basic info
    return {
        'api_status': 'offline',
        'status': 'API Offline - Using detection files',
        'uptime_minutes': 0
    }

def create_compliance_chart(stats):
    """Create compliance rate chart with real data only"""
    if not stats or 'detection_stats' not in stats:
        # No data available - show empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Please run some detections first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='white', size=16)
        )
    else:
        detection_stats = stats.get('detection_stats', {})
        total = detection_stats.get('with_mask', 0) + detection_stats.get('without_mask', 0) + detection_stats.get('incorrect_mask', 0)
        if total > 0:
            compliance_rate = (detection_stats.get('with_mask', 0) / total) * 100
        else:
            compliance_rate = 0
        
        # Use current time for real data point
        dates = [datetime.now()]
        compliance_rates = [compliance_rate]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=compliance_rates,
            mode='lines+markers',
            name='Compliance Rate',
            line=dict(color='#00fff9', width=3),
            marker=dict(size=8, color='#39ff14')
        ))
    
    fig.update_layout(
        title="Real-Time Medical Compliance Rate",
        xaxis_title="Time",
        yaxis_title="Compliance Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#00fff9'),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_detection_breakdown(stats):
    """Create detection breakdown pie chart with real data only"""
    if not stats or 'detection_stats' not in stats:
        # No data available - show message
        fig = go.Figure()
        fig.add_annotation(
            text="No detections yet. Upload images to the API or use webcam.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='white', size=16)
        )
    else:
        detection_stats = stats.get('detection_stats', {})
        labels = ['With Mask', 'Without Mask', 'Incorrect Mask']
        values = [
            detection_stats.get('with_mask', 0),
            detection_stats.get('without_mask', 0),
            detection_stats.get('incorrect_mask', 0)
        ]
        
        # Only show chart if there's actual data
        if sum(values) > 0:
            colors = ['#39ff14', '#ff073a', '#ff6b35']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textfont=dict(color='white', size=14)
            )])
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No detections recorded yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color='white', size=16)
            )
    
    fig.update_layout(
        title="Real Detection Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#00fff9')
    )
    
    return fig

def main():
    """Main dashboard function with real-time detection data and metrics API integration"""
    # Start background metrics collection
    metrics_collector.start_background_collection()
    try:
        # Apply CSS
        st.markdown(MEDICAL_DASHBOARD_CSS, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1 class="header-title">🔬 MEDICAL MONITORING SYSTEM</h1>
            <p class="header-subtitle">Real-time Face Mask Compliance Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get live detection data
        detections = load_real_detections()
        stats = get_live_detection_stats()
        health = get_api_health()

        # --- Metrics API Integration ---
        # Example: Fetch last 10 model metrics from DB
        model_metrics = metrics_collector.get_metrics('model', limit=10)
        # Example: Fetch last 10 drift metrics if available (data/business)
        data_metrics = metrics_collector.get_metrics('data', limit=10)
        # You can display these in the dashboard as needed
        # ...existing code...
        # (Insert Streamlit widgets to show metrics if desired)
        # ...existing code...
        
        # Sidebar controls
        st.sidebar.markdown("## 🏥 System Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 2, 30, 5)
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Detection file status
        st.sidebar.markdown("### 📁 Detection Files")
        st.sidebar.metric("Total Detection Files", len(detections))
        if detections:
            latest_detection = detections[0]
            latest_time = latest_detection.get('timestamp', 'Unknown')
            if latest_time != 'Unknown':
                try:
                    # Parse timestamp and format nicely
                    dt = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))
                    latest_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            st.sidebar.write(f"Latest: {latest_time}")
        
        # API Status
        api_status_text = health.get('api_status', 'unknown')
        if api_status_text == 'online':
            api_status = "🟢 API ONLINE"
            api_color = "status-online"
        elif len(detections) > 0:
            api_status = "🟡 API OFFLINE - DETECTION FILES AVAILABLE"
            api_color = "status-warning"
        else:
            api_status = "🔴 NO DATA"
            api_color = "status-error"
        
        st.sidebar.markdown(f'<p class="{api_color}">Status: {api_status}</p>', unsafe_allow_html=True)
        
        # Main metrics with real data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_scans = stats.get('total_detections', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_scans}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with_mask = stats.get('with_mask', 0)
            without_mask = stats.get('without_mask', 0)
            total = with_mask + without_mask + stats.get('mask_weared_incorrect', 0)
            compliance_rate = (with_mask / total * 100) if total > 0 else 0
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
            avg_confidence = stats.get('average_confidence', 0) * 100  # Convert to percentage
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_confidence:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts with real data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Real-Time Compliance")
            if total > 0:
                # Show real compliance data
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
                st.info("No detection data available. Run webcam app to generate data.")
        
        with col2:
            st.markdown("### 🥧 Detection Distribution")
            if total > 0:
                labels = ['With Mask', 'Without Mask', 'Incorrect Mask']
                values = [with_mask, without_mask, stats.get('mask_weared_incorrect', 0)]
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
        
        if detections:
            # Show recent detections in a nice table
            recent_data = []
            for det in detections[:10]:  # Last 10 detections
                timestamp = det.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime('%H:%M:%S')
                    except:
                        pass
            
                recent_data.append({
                    'Time': timestamp,
                    'Detection': det.get('class', 'Unknown'),
                    'Confidence': f"{det.get('confidence', 0):.1%}",
                    'File': det.get('filename', 'Unknown')
                })
            
            df = pd.DataFrame(recent_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent detection activity. Start the webcam app to see real-time detections.")
        
        # System Information
        st.markdown("## 🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection System:**")
            st.write(f"✅ Files Available: {len(detections)}")
            st.write(f"📊 Total Detections: {total}")
            
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
        
        # Auto refresh without infinite loop
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
            
            with col2:
                st.markdown("**Uptime:**")
                st.write(f"{health.get('uptime_minutes', 0) / 60:.1f} hours")
                
                st.markdown("**Start Time:**")
                start_time = health.get('start_time', 'Unknown')
                if start_time != 'Unknown':
                    try:
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass
                st.write(start_time)
        else:
            st.info("API not available. Please start the API server to see real-time data.")
            st.code("cd src && python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000")
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    finally:
        # Stop background metrics collection on dashboard shutdown
        metrics_collector.stop_background_collection()

if __name__ == "__main__":
    main()
