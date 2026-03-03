"""
Analytics Dashboard
Historical trends and comprehensive system analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.title("📈 System Analytics")
st.markdown("Historical Trends & Performance Analysis")

st.markdown("---")

# Load all data
try:
    # Load blockchain
    blockchain_files = [f for f in os.listdir('data/blockchain') if f.endswith('.json')]
    
    if blockchain_files:
        latest_blockchain = max([os.path.join('data/blockchain', f) for f in blockchain_files],
                               key=os.path.getmtime)
        
        with open(latest_blockchain, 'r') as f:
            blockchain_data = json.load(f)
        
        # Load transactions
        csv_file = latest_blockchain.replace('.json', '_transactions.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Overview metrics
            st.subheader("📊 Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(df))
            
            with col2:
                total_carbon_saved = df['carbon_saving_kg'].sum()
                st.metric("Total CO2 Saved", f"{total_carbon_saved:.3f} kg")
            
            with col3:
                avg_reduction = df['reduction_percentage'].mean()
                st.metric("Avg Reduction", f"{avg_reduction:.1f}%")
            
            with col4:
                duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
                st.metric("Data Period", f"{duration_days} days")
            
            st.markdown("---")
            
            # Time series analysis
            st.subheader("📈 Carbon Savings Over Time")
            
            # Resample to daily
            df_daily = df.set_index('timestamp').resample('D')['carbon_saving_kg'].sum().reset_index()
            
            fig_timeseries = go.Figure()
            
            fig_timeseries.add_trace(go.Scatter(
                x=df_daily['timestamp'],
                y=df_daily['carbon_saving_kg'],
                mode='lines+markers',
                name='Daily Savings',
                line=dict(color='#00D26A', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 210, 106, 0.2)'
            ))
            
            # Add trend line
            if len(df_daily) > 1:
                x_numeric = (df_daily['timestamp'] - df_daily['timestamp'].min()).dt.total_seconds()
                z = np.polyfit(x_numeric, df_daily['carbon_saving_kg'], 1)
                p = np.poly1d(z)
                
                fig_timeseries.add_trace(go.Scatter(
                    x=df_daily['timestamp'],
                    y=p(x_numeric),
                    mode='lines',
                    name='Trend',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
            
            fig_timeseries.update_layout(
                xaxis_title="Date",
                yaxis_title="Carbon Saved (kg CO2)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            st.markdown("---")
            
            # Strategy analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Strategy Performance")
                
                strategy_perf = df.groupby('strategy').agg({
                    'carbon_saving_kg': ['sum', 'mean', 'count']
                }).round(6)
                strategy_perf.columns = ['Total Saved', 'Avg Saved', 'Usage Count']
                strategy_perf = strategy_perf.sort_values('Total Saved', ascending=False)
                
                st.dataframe(
                    strategy_perf.style.format({
                        'Total Saved': '{:.6f}',
                        'Avg Saved': '{:.6f}',
                        'Usage Count': '{:.0f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("🏢 Organization Performance")
                
                org_perf = df.groupby('organization').agg({
                    'carbon_saving_kg': 'sum',
                    'credits_earned': 'sum'
                }).sort_values('carbon_saving_kg', ascending=False)
                
                fig_org = px.bar(
                    org_perf.reset_index(),
                    x='organization',
                    y='carbon_saving_kg',
                    title="Total Carbon Saved by Organization",
                    color='carbon_saving_kg',
                    color_continuous_scale='Greens'
                )
                
                fig_org.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_org, use_container_width=True)
            
            st.markdown("---")
            
            # Distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Reduction Distribution")
                
                fig_hist = px.histogram(
                    df,
                    x='reduction_percentage',
                    nbins=20,
                    title="Distribution of Reduction Percentages",
                    color_discrete_sequence=['#667eea']
                )
                
                fig_hist.update_layout(
                    xaxis_title="Reduction Percentage",
                    yaxis_title="Count",
                    height=300
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("💳 Credits Distribution")
                
                fig_credits = px.box(
                    df,
                    y='credits_earned',
                    title="Credits Earned Distribution",
                    color_discrete_sequence=['#764ba2']
                )
                
                fig_credits.update_layout(height=300)
                st.plotly_chart(fig_credits, use_container_width=True)
            
            st.markdown("---")
            
            # Correlation analysis
            st.subheader("🔗 Correlation Analysis")
            
            # Create correlation matrix
            numeric_cols = ['baseline_carbon_kg', 'optimized_carbon_kg', 
                          'carbon_saving_kg', 'reduction_percentage', 'credits_earned']
            
            if all(col in df.columns for col in numeric_cols):
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("---")
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Best Performing Strategy**")
                best_strategy = strategy_perf.index[0]
                best_saving = strategy_perf.iloc[0]['Total Saved']
                st.success(f"{best_strategy}\n\n{best_saving:.6f} kg CO2 saved")
            
            with col2:
                st.markdown("**Top Organization**")
                top_org = org_perf.index[0]
                top_org_saving = org_perf.iloc[0]['carbon_saving_kg']
                st.info(f"{top_org}\n\n{top_org_saving:.6f} kg CO2 saved")
            
            with col3:
                st.markdown("**Daily Average**")
                if duration_days > 0:
                    daily_avg = total_carbon_saved / duration_days
                    st.warning(f"Average Savings\n\n{daily_avg:.6f} kg CO2/day")
                else:
                    st.warning("Insufficient data")
            
        else:
            st.warning("⚠️ No transaction data available")
    
    else:
        st.warning("⚠️ No blockchain data available")

except Exception as e:
    st.error(f"Error loading analytics data: {e}")

# Statistical summary
with st.expander("📊 Statistical Summary"):
    if 'df' in locals() and not df.empty:
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("No data available for statistical analysis")
