"""
Blockchain Visualization Dashboard
Immutable carbon tracking with distributed validation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

st.set_page_config(page_title="Blockchain", page_icon="⛓️", layout="wide")

st.title("⛓️ Blockchain Verification System")
st.markdown("Immutable audit trail of carbon savings with distributed validation")

st.markdown("---")

# Load blockchain
try:
    blockchain_dir = 'data/blockchain'
    
    if not os.path.exists(blockchain_dir):
        st.error("⚠️ Blockchain directory not found!")
        st.info("Generate blockchain data first:")
        st.code("python generate_blockchain_data.py", language="bash")
        st.stop()
    
    blockchain_files = [f for f in os.listdir(blockchain_dir) if f.endswith('.json')]
    
    if not blockchain_files:
        st.warning("⚠️ No blockchain files found!")
        st.info("Generate blockchain data:")
        st.code("python generate_blockchain_data.py", language="bash")
        st.stop()
    
    # Get latest blockchain file
    latest_blockchain = max([os.path.join(blockchain_dir, f) for f in blockchain_files],
                           key=os.path.getmtime)
    
    st.success(f"📂 Loading: {os.path.basename(latest_blockchain)}")
    
    with open(latest_blockchain, 'r') as f:
        blockchain_data = json.load(f)
    
    summary = blockchain_data.get('summary', {})
    chain = blockchain_data.get('chain', [])
    
    if not chain:
        st.error("⚠️ Blockchain is empty!")
        st.stop()
    
    # Rest of the blockchain page code continues here...
    # (Keep all the existing visualization code from before)
    
    # Blockchain summary
    st.subheader("📊 Blockchain Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📦 Total Blocks", summary.get('total_blocks', 0))
    
    with col2:
        st.metric("📋 Transactions", summary.get('total_transactions', 0))
    
    with col3:
        st.metric("✅ Validation", "VALID" if summary.get('is_valid') else "INVALID")
    
    with col4:
        st.metric("⚡ Avg Mining", f"{summary.get('avg_mining_time_sec', 0):.4f}s")
    
    with col5:
        st.metric("🔒 Difficulty", summary.get('difficulty', 0))
    
    st.markdown("---")
    
    # Carbon metrics
    st.subheader("🌍 Carbon Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total CO2 Saved",
            f"{summary.get('total_co2_saved_kg', 0):.3f} kg",
            help="Total carbon emissions saved"
        )
    
    with col2:
        st.metric(
            "Carbon Credits",
            summary.get('total_carbon_credits', 0),
            help="Credits earned (1 per 0.01 kg CO2 saved)"
        )
    
    with col3:
        cost_saving = summary.get('total_co2_saved_kg', 0) * 6.70
        st.metric(
            "Cost Savings",
            f"${cost_saving:.2f}",
            help="Based on $6.70 per kg CO2"
        )
    
    st.markdown("---")
    
    # Show first 10 and last 10 blocks
    st.subheader("🔗 Blockchain Blocks")
    
    if len(chain) <= 20:
        display_blocks = chain
    else:
        display_blocks = chain[:10] + chain[-10:]
        st.info(f"Showing first 10 and last 10 blocks (total: {len(chain)} blocks)")
    
    # Simple block list
    for block in display_blocks[:5]:  # Show first 5 for performance
        with st.expander(f"Block #{block.get('index', 0)} - {block.get('datetime', 'Unknown time')[:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Index", block.get('index', 0))
                st.metric("Transactions", len(block.get('transactions', [])))
            
            with col2:
                st.metric("Proof", block.get('proof', 0))
                st.metric("Validators", len(block.get('validators', [])))
            
            with col3:
                consensus = "✅" if block.get('consensus_achieved') else "❌"
                st.metric("Consensus", consensus)
    
    st.markdown("---")
    
    # Transaction summary
    csv_file = latest_blockchain.replace('.json', '_transactions.csv')
    if os.path.exists(csv_file):
        st.subheader("📊 Transaction Summary")
        df = pd.read_csv(csv_file)
        
        st.metric("Total Transactions", len(df))
        
        # Show sample transactions
        st.dataframe(df.head(10), use_container_width=True)
        
        # Download button
        st.download_button(
            label="📥 Download All Transactions (CSV)",
            data=df.to_csv(index=False),
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.caption(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error(f"❌ Error loading blockchain: {e}")
    import traceback
    st.code(traceback.format_exc())
    
    st.info("To fix this, run:")
    st.code("python generate_blockchain_data.py", language="bash")
