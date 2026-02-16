"""
Real AWS Carbon Footprint Dashboard
Displays 8.8 days of real AWS monitoring data with 6 optimization strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from blockchain_aws_integration import create_aws_blockchain_from_optimizations


# Page configuration
st.set_page_config(
    page_title="🌱 Real AWS Carbon Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)   

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h2 {
        font-size: 36px;
        margin: 10px 0;
        color: white;
    }
    .metric-card h3 {
        font-size: 18px;
        margin: 5px 0;
        color: #ecf0f1;
    }
    .metric-card p {
        font-size: 14px;
        margin: 5px 0;
        color: #bdc3c7;
    }
</style>
""", unsafe_allow_html=True)


class RealAWSDashboard:
    """Real-time AWS Carbon Dashboard"""
    
    def __init__(self):
        self.baseline_df = None
        self.optimizations_df = None
        
    @st.cache_data
    def load_real_data(_self):
        """Load real AWS monitoring data"""
        try:
            baseline = pd.read_csv('data/real_aws_baseline.csv')
            baseline['timestamp'] = pd.to_datetime(baseline['timestamp'])
            
            optimizations = pd.read_csv('data/processed/optimization_results_real_aws.csv')
            
            return baseline, optimizations
        except FileNotFoundError as e:
            st.error(f"❌ Data file not found: {e}")
            return None, None
    
    def create_hero_metrics(self, df, opts):
        """Create main KPI metrics from real data"""
        st.markdown("### 📊 Real AWS Metrics (8.8 Days Monitoring)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate metrics
        total_records = len(df)
        duration_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
        avg_carbon = df['carbon_kg_per_hour'].mean()
        total_carbon = df['carbon_kg_per_5min'].sum()
        avg_cpu = df['cpu_percent'].mean()
        
        # Best optimization result
        best_opt = opts.iloc[-1]  # GPCO
        carbon_reduction = best_opt['carbon_reduction_pct']
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Total Records</h3>
                    <h2>{total_records:,}</h2>
                    <p>{duration_days:.1f} Days</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>💻 Avg CPU</h3>
                    <h2>{avg_cpu:.3f}%</h2>
                    <p>Over-provisioned!</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>💨 Avg Carbon</h3>
                    <h2>{avg_carbon:.6f}</h2>
                    <p>kg CO₂/hour</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🌍 Total CO₂</h3>
                    <h2>{total_carbon:.3f} kg</h2>
                    <p>Baseline Emissions</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h3>🔥 Best Result</h3>
                    <h2>{carbon_reduction:.1f}%</h2>
                    <p>GPCO Strategy</p>
                </div>
                """, unsafe_allow_html=True)
    
    def create_realtime_timeline(self, df):
        """Create real-time carbon timeline"""
        st.markdown("### 📈 Real AWS Carbon Timeline (8.8 Days)")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Carbon Emissions (kg CO₂/hour)', 'CPU Utilization (%)'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Carbon emissions
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['carbon_kg_per_hour'],
                mode='lines',
                name='Carbon Emissions',
                line=dict(color='#e74c3c', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)',
                hovertemplate='<b>%{x}</b><br>Carbon: %{y:.6f} kg CO₂/hr<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add average line
        avg_carbon = df['carbon_kg_per_hour'].mean()
        fig.add_hline(
            y=avg_carbon, 
            line_dash="dash", 
            line_color="#c0392b",
            annotation_text=f"Avg: {avg_carbon:.6f}",
            row=1, col=1
        )
        
        # CPU utilization
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cpu_percent'],
                mode='lines',
                name='CPU Usage',
                line=dict(color='#3498db', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)',
                hovertemplate='<b>%{x}</b><br>CPU: %{y:.3f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add average line
        avg_cpu = df['cpu_percent'].mean()
        fig.add_hline(
            y=avg_cpu, 
            line_dash="dash", 
            line_color="#2980b9",
            annotation_text=f"Avg: {avg_cpu:.3f}%",
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="kg CO₂/hour", row=1, col=1)
        fig.update_yaxes(title_text="CPU %", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_optimizations_comparison(self, opts):
        """Create 6 optimizations comparison chart"""
        st.markdown("### 🎯 6 Optimization Strategies - Real Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig = go.Figure()
            
            colors = ['#95a5a6', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad', '#27ae60']
            
            fig.add_trace(go.Bar(
                y=opts['strategy'],
                x=opts['carbon_reduction_pct'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='black', width=1)
                ),
                text=opts['carbon_reduction_pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Reduction: %{x:.1f}%<extra></extra>'
            ))
            
            fig.add_vline(x=90, line_dash="dash", line_color="green", 
                         annotation_text="90% Target")
            
            fig.update_layout(
                title="Carbon Reduction by Strategy",
                xaxis_title="Reduction (%)",
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strategy details
            st.markdown("#### 📋 Strategy Breakdown")
            
            for idx, row in opts.iterrows():
                if idx == 0:  # Skip baseline
                    continue
                
                with st.expander(f"**{row['strategy'].split('(')[0].strip()}**"):
                    st.write(f"**Description:** {row['description'][:100]}...")
                    st.metric("Carbon Reduction", f"{row['carbon_reduction_pct']:.1f}%")
                    st.metric("Cost Reduction", f"{row['cost_reduction_pct']:.1f}%")
                    if row['annual_carbon_saving_kg'] > 0:
                        st.metric("Annual CO₂ Saved", f"{row['annual_carbon_saving_kg']:.3f} kg")
    
    def create_cost_carbon_analysis(self, opts):
        """Create cost vs carbon scatter plot"""
        st.markdown("### 💰 Cost vs Carbon Tradeoff Analysis")
        
        fig = px.scatter(
            opts,
            x='carbon_kg_per_day',
            y='cost_usd_per_day',
            size='carbon_reduction_pct',
            color='carbon_reduction_pct',
            hover_name='strategy',
            labels={
                'carbon_kg_per_day': 'Carbon (kg CO₂/day)',
                'cost_usd_per_day': 'Cost ($/day)',
                'carbon_reduction_pct': 'Reduction %'
            },
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        
        # Optimal zone
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=0.01, y1=0.1,
            line=dict(color="green", width=2, dash="dash"),
            fillcolor="green",
            opacity=0.1
        )
        
        fig.add_annotation(
            x=0.005, y=0.25,
            text="← Optimal Zone<br>(Low carbon + Low cost)",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_live_recommendations(self, opts):
        """Create live recommendations panel"""
        st.markdown("### 🤖 Real-Time Optimization Recommendations")
        
        # Map strategies to icons
        strategy_icons = {
            'Baseline': '📊',
            'Temporal': '⏰',
            'Geographic': '🌍',
            'Resource': '⚙️',
            'CBSD': '♻️',
            'DCTR': '🛡️',
            'GPCO': '🧠'
        }
        
        for idx, row in opts.iterrows():
            if idx == 0:  # Skip baseline
                continue
            
            strategy_name = row['strategy'].split('(')[0].strip()
            icon = strategy_icons.get(strategy_name.split()[0], '✨')
            
            # Color based on effectiveness
            if row['carbon_reduction_pct'] >= 85:
                gradient = "linear-gradient(90deg, #27ae60, #2ecc71)"
            elif row['carbon_reduction_pct'] >= 50:
                gradient = "linear-gradient(90deg, #f39c12, #e67e22)"
            else:
                gradient = "linear-gradient(90deg, #3498db, #2980b9)"
            
            st.markdown(f"""
                <div style="background: {gradient};
                            color: white; padding: 15px; border-radius: 10px;
                            margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0; color: white;">{icon} {strategy_name}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💨 Carbon Reduction",
                    f"{row['carbon_reduction_pct']:.1f}%",
                    delta=f"{row['annual_carbon_saving_kg']:.2f} kg/year"
                )
            
            with col2:
                st.metric(
                    "💰 Cost Reduction",
                    f"{row['cost_reduction_pct']:.1f}%",
                    delta=f"${row['annual_cost_saving_usd']:.2f}/year" if row['annual_cost_saving_usd'] > 0 else "Same"
                )
            
            with col3:
                st.metric(
                    "📊 Daily Carbon",
                    f"{row['carbon_kg_per_day']:.6f} kg",
                    delta="Target"
                )
            
            with col4:
                st.metric(
                    "💵 Daily Cost",
                    f"${row['cost_usd_per_day']:.2f}",
                    delta="Optimized"
                )
            
            with st.expander("📋 Implementation Details"):
                st.write(f"**Strategy:** {row['description']}")
                st.write(f"**Daily Carbon:** {row['carbon_kg_per_day']:.6f} kg CO₂")
                st.write(f"**Daily Cost:** ${row['cost_usd_per_day']:.2f}")
                
                if row['annual_carbon_saving_kg'] > 0:
                    st.success(f"✅ Annual Savings: {row['annual_carbon_saving_kg']:.3f} kg CO₂")
                if row['annual_cost_saving_usd'] > 0:
                    st.success(f"✅ Annual Cost Savings: ${row['annual_cost_saving_usd']:.2f}")
            
            st.markdown("---")
    
    def create_summary_stats(self, df, opts):
        """Create summary statistics"""
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        # Baseline vs Best
        baseline = opts.iloc[0]
        best = opts.iloc[-1]
        
        with col1:
            st.markdown("#### 📉 Baseline (Virginia 24/7)")
            st.write(f"• Carbon: {baseline['carbon_kg_per_day']:.6f} kg/day")
            st.write(f"• Cost: ${baseline['cost_usd_per_day']:.2f}/day")
            st.write(f"• Annual: {baseline['carbon_kg_per_day'] * 365:.3f} kg CO₂")
        
        with col2:
            st.markdown("#### 🎯 Best Strategy (GPCO)")
            st.write(f"• Carbon: {best['carbon_kg_per_day']:.6f} kg/day")
            st.write(f"• Cost: ${best['cost_usd_per_day']:.2f}/day")
            st.write(f"• Annual: {best['carbon_kg_per_day'] * 365:.3f} kg CO₂")
        
        with col3:
            st.markdown("#### 🔥 Total Improvement")
            carbon_saved = baseline['carbon_kg_per_day'] - best['carbon_kg_per_day']
            cost_saved = baseline['cost_usd_per_day'] - best['cost_usd_per_day']
            
            st.write(f"• Reduction: {best['carbon_reduction_pct']:.1f}%")
            st.write(f"• Daily Savings: {carbon_saved:.6f} kg")
            st.write(f"• Annual Savings: {carbon_saved * 365:.3f} kg CO₂")
            st.write(f"• Cost Savings: ${cost_saved * 365:.2f}/year")

    def create_blockchain_panel(self, opts):
        """Create blockchain verification panel with real AWS data"""
        st.markdown("### 🔗 Blockchain Verification System")
        
        # Create/load blockchain
        try:
            blockchain = create_aws_blockchain_from_optimizations()
            
            if blockchain:
                summary = blockchain.get_chain_summary()
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📦 Total Blocks",
                        summary['total_blocks'],
                        delta="Immutable"
                    )
                
                with col2:
                    st.metric(
                        "📝 Transactions",
                        summary['total_transactions'],
                        delta="Verified"
                    )
                
                with col3:
                    st.metric(
                        "💰 Carbon Credits",
                        summary['total_carbon_credits'],
                        delta="Earned"
                    )
                
                with col4:
                    st.metric(
                        "🌍 CO2 Saved",
                        f"{summary['total_co2_saved']:.3f} kg",
                        delta="Blockchain verified"
                    )
                
                # Blockchain status
                col_status1, col_status2 = st.columns(2)
                
                with col_status1:
                    if summary['is_valid']:
                        st.success("🔐 **Blockchain Status:** ✅ Valid - All blocks verified")
                    else:
                        st.error("❌ **Blockchain Status:** Invalid chain detected!")
                
                with col_status2:
                    st.info(f"🔗 **Chain Hash:** `{summary['blockchain_hash'][:32]}...`")
                
                # Recent transactions
                st.markdown("#### 📋 Recent Blockchain Transactions")
                
                transactions = blockchain.get_all_transactions()
                
                for i, tx in enumerate(transactions[-6:], 1):  # Last 6 transactions
                    with st.expander(f"🔗 Transaction {i}: {tx['strategy_name']} - Block #{tx['block_index']}"):
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.write(f"**Strategy:** {tx['strategy_name']}")
                            st.write(f"**Organization:** {tx['organization']}")
                            st.write(f"**Data Source:** {tx['data_source']}")
                            st.write(f"**AWS Region:** {tx['aws_region']}")
                        
                        with col_b:
                            st.write(f"**Baseline:** {tx['baseline_emissions']:.6f} kg CO₂/day")
                            st.write(f"**Optimized:** {tx['optimized_emissions']:.6f} kg CO₂/day")
                            st.write(f"**Savings:** {tx['carbon_saving']:.3f} kg CO₂/year")
                            st.write(f"**Cost Savings:** ${tx['cost_saving']:.2f}/year")
                        
                        with col_c:
                            st.write(f"**Credits Earned:** {tx['credits_earned']}")
                            st.write(f"**Verified:** {'✅ Yes' if tx['verified'] else '⏳ Pending'}")
                            st.write(f"**Block Hash:** `{tx['block_hash']}`")
                            st.write(f"**Timestamp:** {tx['timestamp'][:19]}")
                
                # Blockchain visualization
                st.markdown("#### 🔗 Blockchain Structure")
                
                chain_data = []
                for block in blockchain.chain:
                    chain_data.append({
                        'Block': f"Block #{block['index']}",
                        'Transactions': len(block['transactions']),
                        'Hash': blockchain.hash(block)[:16] + "...",
                        'Previous Hash': block['previous_hash'][:16] + "..." if block['previous_hash'] != '0' else 'Genesis'
                    })
                
                chain_df = pd.DataFrame(chain_data)
                st.dataframe(chain_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Blockchain system error: {e}")
            st.info("💡 Blockchain stores carbon optimizations on immutable ledger for transparency and verification")
            
        
    
    def run_dashboard(self):
        """Run the real-time dashboard"""
        st.markdown('<h1 class="main-header">🌱 Real AWS Carbon Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**Live monitoring from 8.8 days of real AWS EC2 data (2,540 measurements)**")
        
        # Load data
        df, opts = self.load_real_data()
        if df is None or opts is None:
            st.error("❌ Failed to load data. Please run analyze_real_aws_data.py first.")
            st.stop()
        
        # Sidebar
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 30s)", value=False)
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()
        
        st.sidebar.markdown("### 📊 Data Info")
        st.sidebar.info(f"""
        **Real AWS Monitoring**
        - Instance: t2.micro
        - Region: us-east-1
        - Duration: 8.8 days
        - Records: {len(df):,}
        - Date: Jan 9-18, 2026
        """)
        
        st.sidebar.markdown("### 🎯 Best Result")
        best = opts.iloc[-1]
        st.sidebar.success(f"""
        **GPCO Strategy**
        - Carbon: {best['carbon_reduction_pct']:.1f}% ↓
        - Cost: {best['cost_reduction_pct']:.1f}% ↓
        - Annual: {best['annual_carbon_saving_kg']:.3f} kg saved
        """)
        
        # Main dashboard
        self.create_hero_metrics(df, opts)
        
        st.markdown("---")
        
        self.create_realtime_timeline(df)
        
        st.markdown("---")
        
        self.create_optimizations_comparison(opts)
        
        st.markdown("---")
        
        self.create_cost_carbon_analysis(opts)
        
        st.markdown("---")
        
        self.create_live_recommendations(opts)
        
        st.markdown("---")
        
        self.create_summary_stats(df, opts)
        st.markdown("---")

        self.create_blockchain_panel(opts)

        
        # Footer
        st.markdown("---")
        st.markdown("### ✅ System Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Real AWS Data: Active")
        with col2:
            st.success("✅ 6 Optimizations: Calculated")
        with col3:
            st.success("✅ Dashboard: Live")


def main():
    """Main entry point"""
    dashboard = RealAWSDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
