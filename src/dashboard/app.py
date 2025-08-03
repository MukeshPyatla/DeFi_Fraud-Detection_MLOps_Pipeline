"""
Standalone Streamlit dashboard for DeFi fraud detection monitoring.
This version works independently without requiring the full ML pipeline.
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
import os
import sys

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

# API configuration - use environment variable or default to demo mode
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

def get_api_health() -> Dict[str, Any]:
    """Get API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_api_metrics() -> Dict[str, Any]:
    """Get API metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/metrics", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration."""
    dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='H')
    
    data = {
        'timestamp': dates,
        'transaction_hash': [f'0x{i:064x}' for i in range(len(dates))],
        'from_address': [f'0x{i:040x}' for i in range(len(dates))],
        'to_address': [f'0x{i:040x}' for i in range(len(dates))],
        'value_eth': np.random.uniform(0.001, 10, len(dates)),
        'gas_price': np.random.uniform(10, 100, len(dates)),
        'gas_used': np.random.uniform(21000, 100000, len(dates)),
        'block_number': np.random.randint(18000000, 19000000, len(dates)),
        'fraud_score': np.random.uniform(0, 1, len(dates)),
        'is_fraud': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], len(dates), p=[0.7, 0.25, 0.05])
    }
    
    return pd.DataFrame(data)

def predict_fraud_demo(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Demo fraud prediction function."""
    # Simple demo logic based on transaction characteristics
    value = transaction_data.get('value_eth', 0)
    gas_price = transaction_data.get('gas_price', 0)
    
    # Simple risk scoring
    risk_score = 0.0
    
    # Higher value transactions are riskier
    if value > 5:
        risk_score += 0.3
    elif value > 1:
        risk_score += 0.1
    
    # Higher gas prices might indicate urgency (potential risk)
    if gas_price > 50:
        risk_score += 0.2
    
    # Add some randomness for demo
    risk_score += np.random.uniform(0, 0.3)
    risk_score = min(risk_score, 1.0)
    
    return {
        'fraud_score': risk_score,
        'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
        'is_fraud': risk_score > 0.6,
        'confidence': np.random.uniform(0.7, 0.95)
    }

def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">🔍 DeFi Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Real-time Monitoring", "Model Performance", "Fraud Prediction", "Audit Trail", "Settings"]
    )
    
    # Check API connection
    api_health = get_api_health()
    api_connected = api_health.get('status') == 'healthy'
    
    if not api_connected:
        st.warning("⚠️ API not connected. Running in demo mode.")
    
    # Page routing
    if page == "Overview":
        show_overview_page(api_connected)
    elif page == "Real-time Monitoring":
        show_monitoring_page(api_connected)
    elif page == "Model Performance":
        show_model_performance_page(api_connected)
    elif page == "Fraud Prediction":
        show_fraud_prediction_page(api_connected)
    elif page == "Audit Trail":
        show_audit_trail_page(api_connected)
    elif page == "Settings":
        show_settings_page()

def show_overview_page(api_connected: bool):
    """Show overview page."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "1,234,567", "↑ 12%")
    
    with col2:
        st.metric("Fraud Detected", "1,234", "↓ 5%")
    
    with col3:
        st.metric("Success Rate", "99.2%", "↑ 0.3%")
    
    with col4:
        st.metric("API Status", "🟢 Connected" if api_connected else "🔴 Disconnected")
    
    # Recent activity chart
    st.subheader("📈 Recent Transaction Activity")
    sample_data = create_sample_data()
    
    fig = px.line(
        sample_data.head(100),
        x='timestamp',
        y='value_eth',
        title='Transaction Values Over Time',
        labels={'value_eth': 'Value (ETH)', 'timestamp': 'Time'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Fraud alerts
    st.subheader("🚨 Recent Fraud Alerts")
    fraud_data = sample_data[sample_data['is_fraud'] == 1].head(5)
    
    if not fraud_data.empty:
        for _, row in fraud_data.iterrows():
            st.markdown(f"""
            <div class="fraud-alert">
                <strong>High Risk Transaction Detected</strong><br>
                Hash: {row['transaction_hash'][:20]}...<br>
                Value: {row['value_eth']:.4f} ETH<br>
                Risk Score: {row['fraud_score']:.2f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No fraud alerts in the last 24 hours")

def show_monitoring_page(api_connected: bool):
    """Show real-time monitoring page."""
    st.header("🔍 Real-time Monitoring")
    
    # Auto-refresh
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Live transaction feed
    st.subheader("📡 Live Transaction Feed")
    
    # Create sample live data
    sample_data = create_sample_data().tail(20)
    
    for _, row in sample_data.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.text(f"Hash: {row['transaction_hash'][:20]}...")
        
        with col2:
            st.text(f"{row['value_eth']:.4f} ETH")
        
        with col3:
            risk_color = "🔴" if row['fraud_score'] > 0.7 else "🟡" if row['fraud_score'] > 0.3 else "🟢"
            st.text(f"{risk_color} {row['fraud_score']:.2f}")
        
        with col4:
            st.text(row['risk_level'])
    
    # Network statistics
    st.subheader("📊 Network Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gas price distribution
        fig = px.histogram(sample_data, x='gas_price', title='Gas Price Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Transaction value distribution
        fig = px.histogram(sample_data, x='value_eth', title='Transaction Value Distribution')
        st.plotly_chart(fig, use_container_width=True)

def show_model_performance_page(api_connected: bool):
    """Show model performance page."""
    st.header("📊 Model Performance")
    
    if api_connected:
        model_info = get_model_info()
        if 'error' not in model_info:
            st.json(model_info)
        else:
            st.error("Failed to get model information")
    else:
        st.info("📋 Model Performance (Demo Mode)")
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    # Create sample performance data
    dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'accuracy': np.random.uniform(0.85, 0.98, len(dates)),
        'precision': np.random.uniform(0.80, 0.95, len(dates)),
        'recall': np.random.uniform(0.75, 0.90, len(dates)),
        'f1_score': np.random.uniform(0.80, 0.92, len(dates))
    })
    
    fig = px.line(performance_data, x='date', y=['accuracy', 'precision', 'recall', 'f1_score'],
                  title='Model Performance Over Time')
    st.plotly_chart(fig, use_container_width=True)

def show_fraud_prediction_page(api_connected: bool):
    """Show fraud prediction page."""
    st.header("🔮 Fraud Prediction")
    
    st.subheader("📝 New Transaction Analysis")
    
    # Input form
    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_hash = st.text_input("Transaction Hash", value="0x1234567890abcdef...")
            from_address = st.text_input("From Address", value="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")
            to_address = st.text_input("To Address", value="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")
            value_eth = st.number_input("Value (ETH)", min_value=0.0, value=1.5, step=0.1)
        
        with col2:
            gas_price = st.number_input("Gas Price (Gwei)", min_value=0, value=20, step=1)
            gas_used = st.number_input("Gas Used", min_value=21000, value=21000, step=1000)
            block_number = st.number_input("Block Number", min_value=0, value=18000000, step=1)
        
        submitted = st.form_submit_button("🔍 Analyze Transaction")
        
        if submitted:
            # Create transaction data
            transaction_data = {
                'transaction_hash': transaction_hash,
                'from_address': from_address,
                'to_address': to_address,
                'value_eth': value_eth,
                'gas_price': gas_price,
                'gas_used': gas_used,
                'block_number': block_number
            }
            
            # Get prediction
            if api_connected:
                try:
                    response = requests.post(f"{API_BASE_URL}/api/v1/predict", 
                                          json=transaction_data, timeout=10)
                    prediction = response.json()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
                    prediction = predict_fraud_demo(transaction_data)
            else:
                prediction = predict_fraud_demo(transaction_data)
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_color = "🔴" if prediction['is_fraud'] else "🟢"
                st.metric("Risk Level", f"{risk_color} {prediction['risk_level']}")
            
            with col2:
                st.metric("Fraud Score", f"{prediction['fraud_score']:.3f}")
            
            with col3:
                st.metric("Confidence", f"{prediction['confidence']:.1%}")
            
            # Detailed analysis
            st.subheader("🔍 Detailed Analysis")
            
            if prediction['is_fraud']:
                st.markdown("""
                <div class="fraud-alert">
                    <strong>⚠️ HIGH RISK TRANSACTION DETECTED</strong><br>
                    This transaction has been flagged as potentially fraudulent based on our analysis.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-alert">
                    <strong>✅ LOW RISK TRANSACTION</strong><br>
                    This transaction appears to be legitimate based on our analysis.
                </div>
                """, unsafe_allow_html=True)

def show_audit_trail_page(api_connected: bool):
    """Show audit trail page."""
    st.header("📋 Audit Trail")
    
    st.subheader("🔍 Recent Predictions")
    
    # Create sample audit data
    audit_data = create_sample_data().tail(50)
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"])
    
    with col2:
        fraud_filter = st.selectbox("Filter by Fraud Status", ["All", "Legitimate", "Fraudulent"])
    
    # Apply filters
    filtered_data = audit_data.copy()
    
    if risk_filter != "All":
        filtered_data = filtered_data[filtered_data['risk_level'] == risk_filter]
    
    if fraud_filter == "Fraudulent":
        filtered_data = filtered_data[filtered_data['is_fraud'] == 1]
    elif fraud_filter == "Legitimate":
        filtered_data = filtered_data[filtered_data['is_fraud'] == 0]
    
    # Display audit trail
    st.dataframe(
        filtered_data[['timestamp', 'transaction_hash', 'value_eth', 'fraud_score', 'risk_level', 'is_fraud']].head(20),
        use_container_width=True
    )

def show_settings_page():
    """Show settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Configuration")
    
    # API settings
    st.write("**API Configuration**")
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    # Display settings
    st.write("**Display Settings**")
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=30)
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")
    
    st.subheader("ℹ️ System Information")
    st.write(f"**API Status:** {'🟢 Connected' if get_api_health().get('status') == 'healthy' else '🔴 Disconnected'}")
    st.write(f"**Streamlit Version:** {st.__version__}")
    st.write(f"**Python Version:** {sys.version}")

if __name__ == "__main__":
    main() 