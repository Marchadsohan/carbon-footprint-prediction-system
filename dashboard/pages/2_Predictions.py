"""
ML Predictions Dashboard
LSTM and XGBoost carbon forecasting
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Predictions")
st.markdown("LSTM Time-Series & XGBoost Feature-Based Forecasting")

st.markdown("---")

# Load prediction data
try:
    pred_files = [f for f in os.listdir('data/predictions') if f.endswith('.json')]
    
    if pred_files:
        latest_pred_file = max([os.path.join('data/predictions', f) for f in pred_files],
                              key=os.path.getmtime)
        
        with open(latest_pred_file, 'r') as f:
            predictions = json.load(f)
        
        if predictions:
            latest_prediction = predictions[-1]
            
            # Model comparison
            st.subheader("🔬 Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 LSTM (Time-Series)")
                lstm_pred = latest_prediction.get('lstm_prediction', {})
                
                st.metric("24h Forecast", f"{lstm_pred.get('total_predicted_kg', 0):.6f} kg CO2")
                st.metric("Avg per Hour", f"{lstm_pred.get('average_per_hour_kg', 0):.6f} kg")
                st.metric("Confidence", f"{lstm_pred.get('confidence', 0)*100:.0f}%")
                
                st.info(f"**Model:** {lstm_pred.get('model', 'LSTM')}")
            
            with col2:
                st.markdown("#### 🎯 XGBoost (Feature-Based)")
                xgb_pred = latest_prediction.get('xgboost_prediction', {})
                
                # Handle both possible key names
                avg_pred = xgb_pred.get('average_prediction_kg', xgb_pred.get('predictions', [0])[0] if isinstance(xgb_pred.get('predictions'), list) else 0)
                
                st.metric("Current Forecast", f"{avg_pred:.6f} kg CO2")
                st.metric("Confidence", f"{xgb_pred.get('confidence', 0)*100:.0f}%")
                
                st.info(f"**Model:** {xgb_pred.get('model', 'XGBoost')}")
            
            st.markdown("---")
            
            # 24-hour forecast chart
            st.subheader("📊 24-Hour Carbon Forecast")
            
            hourly_predictions = lstm_pred.get('predictions', [])
            
            if hourly_predictions:
                hours = list(range(len(hourly_predictions)))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=hourly_predictions,
                    mode='lines+markers',
                    name='Predicted Carbon',
                    line=dict(color='#00D26A', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(0, 210, 106, 0.2)'
                ))
                
                fig.update_layout(
                    xaxis_title="Hours Ahead",
                    yaxis_title="Carbon Emissions (kg CO2)",
                    hovermode='x unified',
                    height=400,
                    title="LSTM 24-Hour Forecast"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly predictions available")
            
            st.markdown("---")
            
            # Prediction history
            if len(predictions) > 1:
                st.subheader("📈 Prediction History")
                
                pred_history = []
                for pred in predictions:
                    lstm_data = pred.get('lstm_prediction', {})
                    xgb_data = pred.get('xgboost_prediction', {})
                    
                    # Handle different key formats
                    xgb_val = xgb_data.get('average_prediction_kg', 
                                          xgb_data.get('predictions', [0])[0] if isinstance(xgb_data.get('predictions'), list) else 0)
                    
                    pred_history.append({
                        'timestamp': pred['timestamp'],
                        'lstm_24h': lstm_data.get('total_predicted_kg', 0),
                        'xgboost_current': xgb_val
                    })
                
                df_history = pd.DataFrame(pred_history)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                
                fig_history = go.Figure()
                
                fig_history.add_trace(go.Scatter(
                    x=df_history['timestamp'],
                    y=df_history['lstm_24h'],
                    mode='lines+markers',
                    name='LSTM 24h Forecast',
                    line=dict(color='#667eea', width=2)
                ))
                
                fig_history.add_trace(go.Scatter(
                    x=df_history['timestamp'],
                    y=df_history['xgboost_current'],
                    mode='lines+markers',
                    name='XGBoost Current',
                    line=dict(color='#764ba2', width=2)
                ))
                
                fig_history.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Carbon Emissions (kg CO2)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_history, use_container_width=True)
            
            st.markdown("---")
            
            # Current metrics
            st.subheader("📊 Current System Metrics")
            
            current_metrics = latest_prediction.get('current_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Daily Carbon",
                    f"{current_metrics.get('daily_carbon_kg', 0):.6f} kg"
                )
            
            with col2:
                st.metric(
                    "Instances Monitored",
                    current_metrics.get('instances_monitored', 0)
                )
            
            with col3:
                st.metric(
                    "Avg CPU",
                    f"{current_metrics.get('avg_cpu_utilization', 0):.1f}%"
                )
            
            st.caption(f"📅 Prediction Generated: {latest_prediction['timestamp']}")
        
        else:
            st.warning("⚠️ No predictions generated yet")
    
    else:
        st.warning("⚠️ No prediction data available")
        st.info("ML predictions are generated after collecting sufficient monitoring data")

except Exception as e:
    st.error(f"Error loading predictions: {e}")
    import traceback
    st.code(traceback.format_exc())

# Model information
with st.expander("ℹ️ Model Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **LSTM (Long Short-Term Memory)**
        - Type: Recurrent Neural Network
        - Purpose: Time-series forecasting
        - Prediction Window: 24 hours ahead
        - Features: CPU, Network, Power, Time
        - Best for: Sequential patterns
        """)
    
    with col2:
        st.markdown("""
        **XGBoost (Extreme Gradient Boosting)**
        - Type: Ensemble Decision Trees
        - Purpose: Feature-based prediction
        - Prediction: Current period
        - Features: CPU, Network, Region, Time
        - Best for: Complex relationships
        """)
