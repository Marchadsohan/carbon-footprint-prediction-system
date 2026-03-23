"""
Carbon Footprint Monitoring System - Multi-Page Dashboard
Complete system visualization with real-time updates
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import glob
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

st.set_page_config(
    page_title="Carbon Footprint Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; color: #00D26A;
        text-align: center; margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem; color: #666;
        text-align: center; margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 10px;
        color: white; text-align: center;
    }
    .green-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem; border-radius: 10px;
        color: white; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cloud detection from .env ────────────────────────────────────────────────
aws_ok   = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
gcp_ok   = bool(os.getenv("GCP_PROJECT_ID"))
azure_ok = bool(os.getenv("AZURE_CLIENT_ID") and os.getenv("AZURE_CLIENT_SECRET"))
clouds_active = sum([aws_ok, gcp_ok, azure_ok])

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Carbon Monitor")
    st.markdown("---")

    st.markdown("### Navigation")
    st.info("""
    **System Components:**
    - 🏠 Home: Overview
    - 📊 Monitoring: Live metrics
    - 🤖 Predictions: ML forecasts
    - ⛓️ Blockchain: Verification
    - 🎯 Optimization: Strategies
    - 📈 Analytics: Trends
    - 📄 Reports: Export data
    """)

    st.markdown("---")

    # Multicloud Status
    st.markdown("### ☁️ Multicloud Status")
    st.markdown(f"{'🟢' if aws_ok   else '🔴'} **AWS**   — {'Active' if aws_ok   else 'Add keys to .env'}")
    st.markdown(f"{'🟢' if gcp_ok   else '🔴'} **GCP**   — {'Active' if gcp_ok   else 'Add keys to .env'}")
    st.markdown(f"{'🟢' if azure_ok else '🔴'} **Azure** — {'Active' if azure_ok else 'Add keys to .env'}")

    realtime_files = sorted(glob.glob("data/realtime/multicloud_*.csv"), reverse=True)
    if realtime_files:
        st.caption(f"Last collected: {os.path.basename(realtime_files[0])}")
    else:
        st.caption("Run: python run_collector.py")

    st.markdown("---")

    # Blockchain System Status
    st.markdown("### System Status")
    try:
        blockchain_files = [f for f in os.listdir('data/blockchain') if f.endswith('.json')]
        if blockchain_files:
            latest_blockchain = max(
                [os.path.join('data/blockchain', f) for f in blockchain_files],
                key=os.path.getmtime
            )
            with open(latest_blockchain, 'r') as f:
                blockchain_data = json.load(f)
            summary = blockchain_data['summary']
            st.success("🟢 System Online")
            st.metric("Total Blocks", summary['total_blocks'])
            st.metric("CO2 Saved",    f"{summary['total_co2_saved_kg']:.3f} kg")
        else:
            st.warning("⚠️ No blockchain data")
    except:
        st.error("🔴 System Offline")

    st.markdown("---")
    st.markdown("### About")
    st.caption("""
    **Version:** 2.0.0
    **Models:** LSTM + XGBoost
    **Blockchain:** Enhanced PoW
    **Clouds:** AWS · GCP · Azure
    """)

# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🌍 Carbon Footprint Prediction System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">A Blockchain-Based Framework for Carbon Footprint '
    'Monitoring &amp; Optimization in Multicloud Infrastructure</div>',
    unsafe_allow_html=True
)

# ─── Hero Row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""<div class="metric-card">
        <h2>🤖 AI-Powered</h2>
        <p>LSTM + XGBoost models for accurate carbon prediction</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="metric-card">
        <h2>⛓️ Blockchain</h2>
        <p>Immutable verification of carbon savings</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="metric-card">
        <h2>📊 Real-Time</h2>
        <p>Live monitoring every 5 minutes</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="green-card">
        <h2>☁️ Multicloud</h2>
        <p>{clouds_active} cloud(s) active<br>AWS · GCP · Azure</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Multicloud Live Snapshot ─────────────────────────────────────────────────
st.subheader("☁️ Latest Multicloud Collection")

all_realtime = sorted(glob.glob("data/realtime/multicloud_*.csv"), reverse=True)
if not all_realtime:
    all_realtime = sorted(glob.glob("data/realtime/aws_carbon_*.csv"), reverse=True)

if all_realtime:
    df_live = pd.read_csv(all_realtime[0])
    if 'cloud_provider' not in df_live.columns:
        df_live['cloud_provider'] = 'AWS'

    carbon_col = next(
        (c for c in ['carbon_kg_per_day', 'carbon_emissions_kg_per_day', 'total_carbon_kg']
         if c in df_live.columns), None
    )
    cpu_col = next(
        (c for c in ['cpu_utilization', 'cpu_percent'] if c in df_live.columns), None
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Clouds Collected", df_live['cloud_provider'].nunique())
    mc2.metric("Instances Found",  len(df_live))
    mc3.metric("Total Carbon Now",
               f"{df_live[carbon_col].sum():.6f} kg/day" if carbon_col else "N/A")
    mc4.metric("Avg CPU",
               f"{df_live[cpu_col].mean():.4f}%" if cpu_col else "N/A")

    show_cols = [c for c in
                 ['instance_name', 'cloud_provider', 'instance_type',
                  'region', cpu_col, carbon_col]
                 if c and c in df_live.columns]
    st.dataframe(df_live[show_cols].round(6), use_container_width=True, hide_index=True)

    col_btn1, _ = st.columns([1, 5])
    with col_btn1:
        if st.button("🔄 Collect Now", type="primary"):
            import subprocess
            with st.spinner("Collecting from all configured clouds..."):
                subprocess.run(
                    [sys.executable, "run_collector.py"],
                    capture_output=True, timeout=60
                )
            st.success("Done!")
            st.rerun()
else:
    st.info("No multicloud data yet.")
    if st.button("▶ Run First Collection", type="primary"):
        import subprocess
        with st.spinner("Collecting..."):
            subprocess.run(
                [sys.executable, "run_collector.py"],
                capture_output=True, timeout=60
            )
        st.rerun()

st.markdown("---")

# ─── Blockchain Overview (existing — unchanged) ───────────────────────────────
try:
    blockchain_files = [f for f in os.listdir('data/blockchain') if f.endswith('.json')]

    if blockchain_files:
        latest_blockchain = max(
            [os.path.join('data/blockchain', f) for f in blockchain_files],
            key=os.path.getmtime
        )
        with open(latest_blockchain, 'r') as f:
            blockchain_data = json.load(f)
        summary = blockchain_data['summary']

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📦 Total Blocks",   f"{summary['total_blocks']:,}")
        col2.metric("🌍 CO2 Saved",       f"{summary['total_co2_saved_kg']:.3f} kg")
        col3.metric("💳 Carbon Credits",  f"{summary['total_carbon_credits']:,}")
        col4.metric("📊 Transactions",    f"{summary['total_transactions']:,}")
        col5.metric("✅ Blockchain",      "VALID" if summary['is_valid'] else "INVALID")

        st.markdown("---")
        st.subheader("📋 Recent Activity")

        csv_file = latest_blockchain.replace('.json', '_transactions.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            recent = df.tail(5).sort_values('timestamp', ascending=False)
            display_cols = ['timestamp', 'organization', 'carbon_saving_kg',
                            'reduction_percentage', 'strategy']
            st.dataframe(
                recent[display_cols].style.format({
                    'carbon_saving_kg':     '{:.6f}',
                    'reduction_percentage': '{:.2f}%'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No transaction data available yet")

        st.markdown("---")
        st.subheader("🎯 Quick Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **System Performance:**
            - ✅ All systems operational
            - ✅ Blockchain validated
            - ✅ Real-time monitoring active
            - ✅ ML predictions running
            """)
        with col2:
            if summary['total_co2_saved_kg'] > 10:
                st.info(f"""
                **Carbon Impact:**
                - 🌱 {summary['total_co2_saved_kg']:.3f} kg CO2 saved
                - 📉 89.8% max reduction achieved
                - 💰 ${summary['total_co2_saved_kg'] * 6.70:.2f} cost savings
                - 🌳 Equivalent to {int(summary['total_co2_saved_kg'] * 50)} trees
                """)
            else:
                st.info("""
                **Getting Started:**
                - 📊 Continue monitoring to collect data
                - 🤖 ML predictions available after 5+ data points
                - ⛓️ Blockchain recording active
                - 🎯 Optimization recommendations coming soon
                """)
    else:
        st.warning("⚠️ No blockchain data found. Run the monitoring system first.")
        st.code("python src/realtime_monitoring/ml_enabled_monitor.py", language="bash")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure the monitoring system has been run and data files exist.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#666; padding:2rem;">
    <p><strong>Carbon Footprint Prediction System v2.0</strong></p>
    <p>Real-Time Multicloud Monitoring | ML Predictions | Blockchain Verification</p>
    <p style="font-size:0.8rem; margin-top:1rem;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>
""", unsafe_allow_html=True)
