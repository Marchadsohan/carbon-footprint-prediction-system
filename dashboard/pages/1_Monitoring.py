"""
Real-Time Monitoring Dashboard
Live multicloud metrics and carbon tracking — AWS · GCP · Azure
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import glob
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(page_title="Real-Time Monitoring", page_icon="📊", layout="wide")

st.title("📊 Real-Time Monitoring Dashboard")
st.markdown("Live multicloud metrics — AWS · GCP · Azure · 5-minute refresh")

# ─── Controls ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("### System Status")
with col2:
    if st.button("🔄 Collect Live Data", type="primary"):
        with st.spinner("Collecting from all configured clouds..."):
            import subprocess
            subprocess.run(
                [sys.executable, "run_collector.py"],
                capture_output=True, text=True, timeout=60
            )
        st.success("Collection complete!")
        st.rerun()
with col3:
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if st.button("Refresh Now"):
        st.rerun()

st.markdown("---")


# ─── Smart data loader ────────────────────────────────────────────────────────
def load_latest_data():
    # Priority 1: multicloud master CSV
    files = sorted(glob.glob("data/realtime/multicloud_*.csv"), reverse=True)
    if files:
        df = pd.read_csv(files[0])
        if 'cloud_provider' not in df.columns:
            df['cloud_provider'] = 'AWS'
        return df, files[0], "multicloud"

    # Priority 2: aws_carbon CSV from run_collector
    files = sorted(glob.glob("data/realtime/aws_carbon_*.csv"), reverse=True)
    if files:
        df = pd.read_csv(files[0])
        df['cloud_provider'] = 'AWS'
        return df, files[0], "aws_carbon"

    # Priority 3: legacy monitoring CSV
    try:
        legacy = [f for f in os.listdir('data/realtime')
                  if f.startswith('monitoring_') and f.endswith('.csv')]
        if legacy:
            latest = max([os.path.join('data/realtime', f) for f in legacy],
                         key=os.path.getmtime)
            df = pd.read_csv(latest)
            df['cloud_provider'] = 'AWS'
            return df, latest, "legacy"
    except:
        pass

    return None, None, None


df, source_file, source_type = load_latest_data()

if df is not None:

    # Column aliases
    cpu_col    = next((c for c in ['cpu_utilization', 'cpu_percent']            if c in df.columns), None)
    carbon_col = next((c for c in ['carbon_kg_per_day',
                                   'carbon_emissions_kg_per_day',
                                   'total_carbon_kg']                           if c in df.columns), None)
    power_col  = next((c for c in ['actual_power_watts', 'base_power_watts']    if c in df.columns), None)

    st.caption(
        f"Source: `{os.path.basename(source_file)}` | "
        f"Type: `{source_type}` | "
        f"Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ─── KPI Row ──────────────────────────────────────────────────────────────
    st.subheader("📈 Current Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("☁️ Clouds Active",
                df['cloud_provider'].nunique())
    col2.metric("🖥️ Instances",
                len(df))
    col3.metric("💻 Avg CPU",
                f"{df[cpu_col].mean():.4f}%"     if cpu_col    else "N/A")
    col4.metric("🌍 Daily Carbon",
                f"{df[carbon_col].sum():.6f} kg" if carbon_col else "N/A")
    col5.metric("⚡ Total Power",
                f"{df[power_col].sum():.1f} W"   if power_col  else "N/A")

    st.markdown("---")

    # ─── Multicloud breakdown (shown only if >1 cloud) ────────────────────────
    if df['cloud_provider'].nunique() > 1 and carbon_col and cpu_col:
        st.subheader("☁️ Carbon by Cloud Provider")

        cloud_summary = df.groupby('cloud_provider').agg(
            carbon=(carbon_col, 'sum'),
            cpu=(cpu_col, 'mean'),
            instances=('cloud_provider', 'count')
        ).reset_index()

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_cloud = px.bar(
                cloud_summary, x='cloud_provider', y='carbon',
                color='cloud_provider',
                color_discrete_map={'AWS': '#FF9900', 'GCP': '#4285F4', 'AZURE': '#0089D6'},
                title="Carbon per Cloud (kg CO₂/day)",
                text='carbon'
            )
            fig_cloud.update_traces(texttemplate='%{text:.6f}', textposition='outside')
            fig_cloud.update_layout(
                height=320, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cloud, use_container_width=True)

        with cc2:
            fig_cpu_cloud = px.bar(
                cloud_summary, x='cloud_provider', y='cpu',
                color='cloud_provider',
                color_discrete_map={'AWS': '#FF9900', 'GCP': '#4285F4', 'AZURE': '#0089D6'},
                title="Avg CPU per Cloud (%)"
            )
            fig_cpu_cloud.update_layout(
                height=320, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cpu_cloud, use_container_width=True)

        st.markdown("---")

    # ─── Instance table ───────────────────────────────────────────────────────
    st.subheader("🖥️ Instance Details")

    table_cols = [c for c in
                  ['instance_name', 'cloud_provider', 'instance_type',
                   'region', cpu_col, power_col, carbon_col]
                  if c and c in df.columns]

    fmt = {}
    if cpu_col    in df.columns: fmt[cpu_col]    = '{:.4f}'
    if power_col  in df.columns: fmt[power_col]  = '{:.2f}'
    if carbon_col in df.columns: fmt[carbon_col] = '{:.6f}'

    styled = df[table_cols].style.format(fmt)
    if cpu_col in df.columns:
        styled = styled.background_gradient(subset=[cpu_col], cmap='RdYlGn_r')

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Charts Row ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if cpu_col and 'instance_name' in df.columns:
            st.subheader("💻 CPU Utilization")
            fig_cpu = px.bar(
                df, x='instance_name', y=cpu_col,
                color='cloud_provider',
                color_discrete_map={'AWS': '#FF9900', 'GCP': '#4285F4', 'AZURE': '#0089D6'},
                title="CPU % by Instance"
            )
            fig_cpu.update_layout(
                height=350, showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cpu, use_container_width=True)

    with col2:
        if carbon_col and 'instance_name' in df.columns:
            st.subheader("🌍 Carbon Distribution")
            fig_carbon = px.pie(
                df, values=carbon_col, names='instance_name',
                title="Daily Carbon by Instance",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_carbon.update_layout(height=350)
            st.plotly_chart(fig_carbon, use_container_width=True)

    # ─── Historical CSV files list ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 Collection History")

    all_files = (
        sorted(glob.glob("data/realtime/multicloud_*.csv"), reverse=True) +
        sorted(glob.glob("data/realtime/aws_carbon_*.csv"),  reverse=True)
    )
    if all_files:
        file_info = []
        for fp in all_files[:10]:
            try:
                size = os.path.getsize(fp)
                mtime = datetime.fromtimestamp(os.path.getmtime(fp))
                rows  = sum(1 for _ in open(fp)) - 1
                file_info.append({
                    "File":     os.path.basename(fp),
                    "Records":  rows,
                    "Size":     f"{size/1024:.1f} KB",
                    "Collected": mtime.strftime("%Y-%m-%d %H:%M:%S")
                })
            except:
                pass
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    else:
        st.info("No history yet — click Collect Live Data above")

    # ─── Optimization potential ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💚 Optimization Potential (GPCO — 89.8% Reduction)")

    if carbon_col:
        daily_carbon     = df[carbon_col].sum()
        potential_saving = daily_carbon * 0.898
        potential_cost   = potential_saving * 6.70 * 365

        col1, col2, col3 = st.columns(3)
        col1.metric("Daily Saving Potential",
                    f"{potential_saving:.6f} kg CO2", "89.8% reduction")
        col2.metric("Annual Cost Saving",
                    f"${potential_cost:.2f}", help="Based on $6.70 per kg CO2")
        col3.metric("Trees Equivalent",
                    f"{int(potential_saving * 365 * 50):,} trees/year")

    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.warning("⚠️ No monitoring data available")
    st.info("Click **Collect Live Data** above to fetch from all configured clouds")

    col1, col2 = st.columns(2)
    with col1:
        st.code("python run_collector.py", language="bash")
    with col2:
        st.markdown("""
        **What this does:**
        - Auto-detects AWS/GCP/Azure from `.env`
        - Fetches real CPU from CloudWatch/Monitoring
        - Calculates carbon emissions
        - Saves CSV to `data/realtime/`
        """)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()
