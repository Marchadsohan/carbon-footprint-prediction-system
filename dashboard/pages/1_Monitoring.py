"""
Real-Time Monitoring Dashboard
Live AWS CloudWatch metrics and carbon tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import time

st.set_page_config(page_title="Real-Time Monitoring", page_icon="📊", layout="wide")

st.title("📊 Real-Time Monitoring Dashboard")
st.markdown("Live AWS CloudWatch metrics with 5-minute refresh")

# Auto-refresh controls
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### System Status")
with col2:
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if st.button("🔄 Refresh Now"):
        st.rerun()

st.markdown("---")

# Load latest monitoring data
try:
    monitoring_files = [f for f in os.listdir('data/realtime') 
                       if f.startswith('monitoring_') and f.endswith('.csv')]
    
    if monitoring_files:
        latest_file = max([os.path.join('data/realtime', f) for f in monitoring_files],
                         key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        
        # Current metrics
        st.subheader("📈 Current Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🖥️ Instances Monitored",
                len(df),
                help="Number of active EC2 instances"
            )
        
        with col2:
            avg_cpu = df['cpu_utilization'].mean()
            st.metric(
                "💻 Average CPU",
                f"{avg_cpu:.1f}%",
                help="Average CPU utilization across all instances"
            )
        
        with col3:
            current_carbon = df['total_carbon_kg'].sum()
            st.metric(
                "🌍 Current Carbon",
                f"{current_carbon:.6f} kg",
                help="Current period carbon emissions"
            )
        
        with col4:
            daily_carbon = df['carbon_emissions_kg_per_day'].sum()
            st.metric(
                "📅 Daily Projection",
                f"{daily_carbon:.6f} kg/day",
                help="Estimated daily carbon emissions"
            )
        
        st.markdown("---")
        
        # Instance details
        st.subheader("🖥️ Instance Details")
        
        display_df = df[[
            'instance_name', 'instance_type', 'region',
            'cpu_utilization', 'actual_power_watts',
            'carbon_emissions_kg_per_day'
        ]].copy()
        
        display_df.columns = [
            'Instance Name', 'Instance Type', 'Region',
            'CPU %', 'Power (W)', 'Daily Carbon (kg)'
        ]
        
        st.dataframe(
            display_df.style.format({
                'CPU %': '{:.2f}',
                'Power (W)': '{:.2f}',
                'Daily Carbon (kg)': '{:.6f}'
            }).background_gradient(subset=['CPU %'], cmap='RdYlGn_r'),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💻 CPU Utilization by Instance")
            fig_cpu = px.bar(
                df,
                x='instance_name',
                y='cpu_utilization',
                color='cpu_utilization',
                color_continuous_scale='RdYlGn_r',
                title="CPU Utilization"
            )
            fig_cpu.update_layout(showlegend=False)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            st.subheader("🌍 Carbon Emissions by Instance")
            fig_carbon = px.pie(
                df,
                values='carbon_emissions_kg_per_day',
                names='instance_name',
                title="Daily Carbon Distribution"
            )
            st.plotly_chart(fig_carbon, use_container_width=True)
        
        # Optimization potential
        st.markdown("---")
        st.subheader("💚 Optimization Potential")
        
        potential_saving = daily_carbon * 0.898
        potential_cost = potential_saving * 6.70 * 365
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Daily Saving Potential",
                f"{potential_saving:.6f} kg CO2",
                f"{89.8}% reduction"
            )
        
        with col2:
            st.metric(
                "Annual Cost Saving",
                f"${potential_cost:.2f}",
                help="Based on $6.70 per kg CO2"
            )
        
        with col3:
            trees_equivalent = int(potential_saving * 365 * 50)
            st.metric(
                "Trees Equivalent",
                f"{trees_equivalent:,} trees/year",
                help="Estimated tree planting equivalent"
            )
        
        # Last updated
        st.caption(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        st.warning("⚠️ No monitoring data available")
        st.info("Start the monitoring system to see live data")
        st.code("python src/realtime_monitoring/ml_enabled_monitor.py", language="bash")

except Exception as e:
    st.error(f"Error loading monitoring data: {e}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(10)
    st.rerun()
