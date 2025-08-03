"""
Streamlit dashboard for DeFi fraud detection monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional

from ..utils.config import config
from ..utils.logger import setup_logger

# Setup logging
logger = setup_logger("dashboard")

# Page configuration
st.set_page_config(
    page_title="DeFi Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = f"http://{config.get('api.host', 'localhost')}:{config.get('api.port', 8000)}"


def get_api_health() -> Dict[str, Any]:
    """Get API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        return response.json()
    except Exception as e:
        logger.error("Failed to get API health", error=str(e))
        return {"status": "error", "error": str(e)}


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=5)
        return response.json()
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        return {"error": str(e)}


def get_api_metrics() -> Dict[str, Any]:
    """Get API metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/metrics", timeout=5)
        return response.json()
    except Exception as e:
        logger.error("Failed to get API metrics", error=str(e))
        return {"error": str(e)}


def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration."""
    dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='H')
    
    data = {
        'timestamp': dates,
        'transactions': np.random.poisson(100, len(dates)),
        'fraud_detected': np.random.binomial(100, 0.05, len(dates)),
        'avg_gas_price': np.random.normal(20, 5, len(dates)),
        'avg_transaction_value': np.random.normal(0.1, 0.05, len(dates)),
        'model_accuracy': np.random.normal(0.95, 0.02, len(dates))
    }
    
    return pd.DataFrame(data)


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 DeFi Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Real-time Monitoring", "Model Performance", "Audit Trail", "Settings"]
    )
    
    # API Status Check
    health_status = get_api_health()
    
    if health_status.get('status') == 'healthy':
        st.sidebar.markdown('<div class="success-alert">✅ API Connected</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="fraud-alert">❌ API Disconnected</div>', unsafe_allow_html=True)
    
    # Page routing
    if page == "Overview":
        show_overview_page()
    elif page == "Real-time Monitoring":
        show_monitoring_page()
    elif page == "Model Performance":
        show_model_performance_page()
    elif page == "Audit Trail":
        show_audit_trail_page()
    elif page == "Settings":
        show_settings_page()


def show_overview_page():
    """Show the overview page."""
    st.header("📊 Overview")
    
    # Get API data
    health_status = get_api_health()
    model_info = get_model_info()
    api_metrics = get_api_metrics()
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="API Status",
            value=health_status.get('status', 'Unknown'),
            delta="✅" if health_status.get('status') == 'healthy' else "❌"
        )
    
    with col2:
        model_status = health_status.get('model_status', 'unknown')
        st.metric(
            label="Model Status",
            value=model_status,
            delta="✅" if model_status == 'loaded' else "⚠️"
        )
    
    with col3:
        uptime = api_metrics.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        st.metric(
            label="Uptime (hours)",
            value=f"{uptime_hours:.1f}",
            delta="🟢"
        )
    
    # Sample data for demonstration
    sample_data = create_sample_data()
    
    # Create charts
    st.subheader("📈 Transaction Volume & Fraud Detection")
    
    # Transaction volume chart
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Transaction Volume', 'Fraud Detection Rate'),
        vertical_spacing=0.1
    )
    
    fig1.add_trace(
        go.Scatter(x=sample_data['timestamp'], y=sample_data['transactions'],
                  name='Transactions', line=dict(color='blue')),
        row=1, col=1
    )
    
    fraud_rate = (sample_data['fraud_detected'] / sample_data['transactions'] * 100)
    fig1.add_trace(
        go.Scatter(x=sample_data['timestamp'], y=fraud_rate,
                  name='Fraud Rate (%)', line=dict(color='red')),
        row=2, col=1
    )
    
    fig1.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Model information
    if 'error' not in model_info:
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Algorithm:** {model_info.get('algorithm', 'N/A')}")
            st.write(f"**Version:** {model_info.get('version', 'N/A')}")
            st.write(f"**Feature Count:** {model_info.get('feature_count', 'N/A')}")
        
        with col2:
            metrics = model_info.get('performance_metrics', {})
            if metrics:
                st.write("**Performance Metrics:**")
                for metric, value in metrics.items():
                    st.write(f"- {metric.title()}: {value:.3f}")
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    
    # Create sample recent transactions
    recent_transactions = pd.DataFrame({
        'Time': pd.date_range(start=datetime.now() - timedelta(hours=6), periods=10, freq='H'),
        'Transaction Hash': [f"0x{i:064x}" for i in range(10)],
        'Value (ETH)': np.random.uniform(0.01, 1.0, 10),
        'Fraud Risk': np.random.uniform(0, 1, 10),
        'Status': np.random.choice(['Normal', 'Suspicious', 'Fraud'], 10, p=[0.7, 0.2, 0.1])
    })
    
    st.dataframe(recent_transactions, use_container_width=True)


def show_monitoring_page():
    """Show the real-time monitoring page."""
    st.header("📡 Real-time Monitoring")
    
    # Auto-refresh
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Create tabs for different monitoring aspects
    tab1, tab2, tab3 = st.tabs(["Transaction Flow", "Network Metrics", "Alerts"])
    
    with tab1:
        st.subheader("Transaction Flow")
        
        # Sample real-time data
        time_range = pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq='H')
        
        # Transaction volume
        tx_volume = np.random.poisson(100, 24)
        fig = px.line(x=time_range, y=tx_volume, title="Transaction Volume (Last 24 Hours)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gas price trends
        gas_prices = np.random.normal(20, 5, 24)
        fig2 = px.line(x=time_range, y=gas_prices, title="Average Gas Price (Gwei)")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Network Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Gas Price", "25.4 Gwei", "2.1")
            st.metric("Network Congestion", "Medium", "-5%")
            st.metric("Block Time", "12.1s", "0.2s")
        
        with col2:
            st.metric("Pending Transactions", "15,234", "1,234")
            st.metric("Average Block Size", "85%", "3%")
            st.metric("Network Hashrate", "1.2 TH/s", "0.1 TH/s")
    
    with tab3:
        st.subheader("🚨 Fraud Alerts")
        
        # Sample alerts
        alerts = [
            {"time": "2 minutes ago", "severity": "High", "description": "Unusual transaction pattern detected", "address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"},
            {"time": "5 minutes ago", "severity": "Medium", "description": "High gas price anomaly", "address": "0x1234567890abcdef1234567890abcdef12345678"},
            {"time": "10 minutes ago", "severity": "Low", "description": "New address with high transaction volume", "address": "0xabcdef1234567890abcdef1234567890abcdef12"}
        ]
        
        for alert in alerts:
            color = {"High": "red", "Medium": "orange", "Low": "yellow"}[alert["severity"]]
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0;">
                <strong>{alert['severity']} Alert</strong><br>
                {alert['description']}<br>
                <small>Address: {alert['address']}</small><br>
                <small>Time: {alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)


def show_model_performance_page():
    """Show the model performance page."""
    st.header("🤖 Model Performance")
    
    # Get model info
    model_info = get_model_info()
    
    if 'error' in model_info:
        st.error("Failed to load model information")
        return
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = model_info.get('performance_metrics', {})
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    
    # Performance charts
    st.subheader("Performance Over Time")
    
    # Sample performance data
    dates = pd.date_range(start='2023-11-01', end='2023-12-01', freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'accuracy': np.random.normal(0.95, 0.02, len(dates)),
        'precision': np.random.normal(0.92, 0.03, len(dates)),
        'recall': np.random.normal(0.88, 0.04, len(dates)),
        'f1_score': np.random.normal(0.90, 0.025, len(dates))
    })
    
    fig = px.line(performance_data, x='date', y=['accuracy', 'precision', 'recall', 'f1_score'],
                  title="Model Performance Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix (sample)
    st.subheader("Confusion Matrix")
    
    # Sample confusion matrix
    confusion_matrix = np.array([[850, 50], [30, 70]])
    
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Normal", "Fraud"],
        y=["Normal", "Fraud"],
        title="Confusion Matrix",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("Feature Importance")
    
    # Sample feature importance
    features = model_info.get('feature_names', [])
    if features:
        importance = np.random.uniform(0, 1, len(features))
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                    title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)


def show_audit_trail_page():
    """Show the audit trail page."""
    st.header("📋 Audit Trail")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        event_type = st.selectbox("Event Type", ["All", "Predictions", "Model Changes", "Data Access"])
    
    with col2:
        date_range = st.date_input("Date Range", value=(datetime.now().date(), datetime.now().date()))
    
    with col3:
        severity = st.selectbox("Severity", ["All", "High", "Medium", "Low"])
    
    # Sample audit trail data
    audit_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), periods=100, freq='H'),
        'event_type': np.random.choice(['prediction', 'model_change', 'data_access'], 100),
        'user_id': np.random.choice(['user1', 'user2', 'system'], 100),
        'description': np.random.choice([
            'Transaction prediction made',
            'Model version updated',
            'Training data accessed',
            'Fraud alert triggered'
        ], 100),
        'severity': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.6, 0.3, 0.1]),
        'transaction_hash': [f"0x{i:064x}" for i in np.random.randint(0, 1000000, 100)]
    })
    
    # Filter data
    if event_type != "All":
        audit_data = audit_data[audit_data['event_type'] == event_type.lower().replace(' ', '_')]
    
    if severity != "All":
        audit_data = audit_data[audit_data['severity'] == severity]
    
    # Display audit trail
    st.dataframe(audit_data, use_container_width=True)
    
    # Export functionality
    if st.button("📥 Export Audit Trail"):
        csv = audit_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def show_settings_page():
    """Show the settings page."""
    st.header("⚙️ Settings")
    
    # Configuration settings
    st.subheader("API Configuration")
    
    api_host = st.text_input("API Host", value=config.get('api.host', 'localhost'))
    api_port = st.number_input("API Port", value=config.get('api.port', 8000), min_value=1, max_value=65535)
    
    if st.button("Save API Settings"):
        st.success("API settings saved!")
    
    # Model settings
    st.subheader("Model Configuration")
    
    algorithm = st.selectbox("Model Algorithm", ["random_forest", "logistic_regression", "svm"])
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", value=42)
    
    if st.button("Save Model Settings"):
        st.success("Model settings saved!")
    
    # Monitoring settings
    st.subheader("Monitoring Configuration")
    
    refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 30, 10)
    alert_threshold = st.slider("Fraud Alert Threshold", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Save Monitoring Settings"):
        st.success("Monitoring settings saved!")
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**API Status:**", "🟢 Connected" if get_api_health().get('status') == 'healthy' else "🔴 Disconnected")
        st.write("**Model Status:**", "🟢 Loaded" if get_model_info().get('algorithm') else "🔴 Not Loaded")
    
    with col2:
        st.write("**Uptime:**", f"{api_metrics.get('uptime_seconds', 0) / 3600:.1f} hours")
        st.write("**Version:**", "1.0.0")


if __name__ == "__main__":
    main() 