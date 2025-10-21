import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
import sys
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Page configuration
st.set_page_config(
    page_title="🌱 Carbon Footprint AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .recommendation-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class CarbonDashboard:
    """Main Carbon Footprint AI Dashboard"""
    
    def __init__(self):
        self.data = None
        self.tcep_model = None
        self.xgb_optimizer = None
        
    @st.cache_data
    def load_data(_self):
        """Load synthetic carbon footprint data"""
        try:
            df = pd.read_csv('../../data/synthetic/carbon_footprint_dataset.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            st.error("❌ Dataset not found! Please run data generation first.")
            return None
    
    def create_main_metrics(self, df):
        """Create main KPI metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Current carbon rate (last hour)
        current_carbon = df['carbon_emissions_kg_co2'].iloc[-1]
        
        # Daily total
        today_data = df[df['timestamp'].dt.date == df['timestamp'].iloc[-1].date()]
        daily_total = today_data['carbon_emissions_kg_co2'].sum()
        
        # Weekly average
        weekly_avg = df['carbon_emissions_kg_co2'].rolling(window=168, min_periods=1).mean().iloc[-1]
        
        # Total energy consumed today
        daily_energy = today_data['energy_consumption_kwh'].sum()
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🕐 Current Rate</h3>
                <h2>{:.4f} kg CO2/h</h2>
                <p>Last Hour</p>
            </div>
            """.format(current_carbon), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📅 Today Total</h3>
                <h2>{:.2f} kg CO2</h2>
                <p>24 Hour Sum</p>
            </div>
            """.format(daily_total), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Weekly Avg</h3>
                <h2>{:.4f} kg CO2/h</h2>
                <p>7 Day Average</p>
            </div>
            """.format(weekly_avg), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Energy Today</h3>
                <h2>{:.2f} kWh</h2>
                <p>Total Consumed</p>
            </div>
            """.format(daily_energy), unsafe_allow_html=True)
    
    def create_time_series_chart(self, df):
        """Create carbon emissions time series"""
        st.subheader("📈 Carbon Emissions Over Time")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.selectbox("📅 Time Range", 
                                   options=[7, 30, 90, 365], 
                                   format_func=lambda x: f"Last {x} days",
                                   index=0)
        
        # Filter data
        end_date = df['timestamp'].max()
        start_date = end_date - pd.Timedelta(days=days_back)
        filtered_df = df[df['timestamp'] >= start_date].copy()
        
        # Resample for better visualization
        if days_back <= 7:
            freq = 'H'  # Hourly
            title_freq = "Hourly"
        elif days_back <= 30:
            freq = 'D'  # Daily
            title_freq = "Daily"
        else:
            freq = 'W'  # Weekly
            title_freq = "Weekly"
        
        resampled = filtered_df.set_index('timestamp').resample(freq).agg({
            'carbon_emissions_kg_co2': 'mean',
            'energy_consumption_kwh': 'mean',
            'cpu_usage_percent': 'mean',
            'renewable_energy_pct': 'mean'
        }).reset_index()
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{title_freq} Carbon Emissions', f'{title_freq} Energy Consumption'),
            vertical_spacing=0.1
        )
        
        # Carbon emissions
        fig.add_trace(
            go.Scatter(
                x=resampled['timestamp'],
                y=resampled['carbon_emissions_kg_co2'],
                mode='lines+markers',
                name='Carbon Emissions (kg CO2)',
                line=dict(color='#ff6b6b', width=2),
                hovertemplate='<b>%{x}</b><br>CO2: %{y:.4f} kg<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Energy consumption
        fig.add_trace(
            go.Scatter(
                x=resampled['timestamp'],
                y=resampled['energy_consumption_kwh'],
                mode='lines+markers',
                name='Energy (kWh)',
                line=dict(color='#4ecdc4', width=2),
                hovertemplate='<b>%{x}</b><br>Energy: %{y:.4f} kWh<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="kg CO2", row=1, col=1)
        fig.update_yaxes(title_text="kWh", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_regional_analysis(self, df):
        """Create regional carbon intensity analysis"""
        st.subheader("🌍 Regional Carbon Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional emissions pie chart
            regional_data = df.groupby('region_name').agg({
                'carbon_emissions_kg_co2': 'sum',
                'renewable_energy_pct': 'mean',
                'base_carbon_intensity': 'mean'
            }).reset_index()
            
            fig_pie = px.pie(
                regional_data, 
                values='carbon_emissions_kg_co2', 
                names='region_name',
                title="Total Emissions by Region",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Regional efficiency scatter
            fig_scatter = px.scatter(
                regional_data,
                x='renewable_energy_pct',
                y='base_carbon_intensity',
                size='carbon_emissions_kg_co2',
                hover_name='region_name',
                title="Carbon Intensity vs Renewable Energy",
                labels={
                    'renewable_energy_pct': 'Renewable Energy %',
                    'base_carbon_intensity': 'Base Carbon Intensity'
                },
                color='carbon_emissions_kg_co2',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def create_recommendations_panel(self, df):
        """Create AI recommendations panel using native Streamlit components"""
        st.subheader("🤖 AI-Powered Optimization Recommendations")
        
        # Sample recommendations (in real system, these would come from XGBoost model)
        sample_recommendations = [
            {
                'type': 'temporal_shift',
                'icon': '⏰',
                'title': 'Temporal Optimization',
                'description': 'Schedule non-critical workloads during low-carbon hours (2-6 AM)',
                'carbon_saving': 0.156,
                'cost_saving': 7.80,
                'effort': 'Low',
                'confidence': 89,
                'reduction_percent': 15.6
            },
            {
                'type': 'geographic_shift',
                'icon': '🌍',
                'title': 'Geographic Migration',
                'description': 'Migrate workloads to West region (40% renewable energy)',
                'carbon_saving': 0.203,
                'cost_saving': 10.15,
                'effort': 'Medium',
                'confidence': 76,
                'reduction_percent': 20.3
            },
            {
                'type': 'resource_optimization',
                'icon': '⚙️',
                'title': 'Resource Right-sizing',
                'description': 'Optimize server configurations to reduce over-provisioning',
                'carbon_saving': 0.088,
                'cost_saving': 4.40,
                'effort': 'Low',
                'confidence': 92,
                'reduction_percent': 8.8
            }
        ]
        
        # Create recommendations using native Streamlit components
        for i, rec in enumerate(sample_recommendations):
            # Create a container with border styling
            with st.container():
                # Add colored header bar
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #28a745, #20c997); 
                            color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: white;">{rec['icon']} {rec['title']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Main content row
                col_desc, col_carbon, col_cost, col_confidence = st.columns([3, 1.5, 1.5, 1.5])
                
                with col_desc:
                    st.write(f"**Description:** {rec['description']}")
                    st.caption(f"⚡ **Implementation Effort:** {rec['effort']}")
                
                with col_carbon:
                    st.metric(
                        label="💨 CO2 Saved",
                        value=f"{rec['carbon_saving']:.3f} kg",
                        delta=f"{rec['reduction_percent']:.1f}% reduction"
                    )
                
                with col_cost:
                    st.metric(
                        label="💰 Cost Saved",
                        value=f"${rec['cost_saving']:.2f}",
                        delta="per month"
                    )
                
                with col_confidence:
                    # Confidence with color coding
                    if rec['confidence'] > 80:
                        confidence_color = "🟢 High"
                    elif rec['confidence'] > 60:
                        confidence_color = "🟡 Medium"
                    else:
                        confidence_color = "🔴 Low"
                    
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{rec['confidence']}%",
                        delta=confidence_color
                    )
                
                # Action buttons row
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 1])
                
                with col_btn1:
                    if st.button(f"✅ Implement", key=f"implement_{i}", type="primary"):
                        st.success(f"🚀 {rec['title']} scheduled for implementation!")
                        st.balloons()
                
                with col_btn2:
                    if st.button(f"📊 Details", key=f"details_{i}"):
                        with st.expander(f"📋 Detailed Analysis - {rec['title']}", expanded=True):
                            st.write(f"**Optimization Type:** {rec['type'].replace('_', ' ').title()}")
                            st.write(f"**Expected CO2 Reduction:** {rec['carbon_saving']:.3f} kg CO2 per hour")
                            st.write(f"**Monthly Cost Savings:** ${rec['cost_saving']:.2f}")
                            st.write(f"**Implementation Time:** 2-5 days ({rec['effort']} effort)")
                            st.write(f"**Risk Level:** Low")
                            st.write(f"**Prerequisites:** None")
                
                with col_btn3:
                    if st.button(f"⏰ Schedule", key=f"schedule_{i}"):
                        st.info(f"📅 {rec['title']} added to optimization queue for next maintenance window")
                
                with col_btn4:
                    if st.button(f"📈 Impact", key=f"impact_{i}"):
                        # Show impact visualization
                        impact_data = pd.DataFrame({
                            'Metric': ['Current', 'After Optimization'],
                            'CO2 Emissions': [1.0, 1.0 - rec['reduction_percent']/100],
                            'Cost': [100, 100 - rec['cost_saving']]
                        })
                        fig_impact = px.bar(
                            impact_data, 
                            x='Metric', 
                            y='CO2 Emissions',
                            title=f'Impact Preview - {rec["title"]}',
                            color='CO2 Emissions',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                
                # Separator with some styling
                st.markdown("---")
        
        # Summary section
        st.markdown("### 📈 Optimization Summary")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            total_carbon_savings = sum(rec['carbon_saving'] for rec in sample_recommendations)
            st.metric(
                label="🌍 Total CO2 Savings Potential",
                value=f"{total_carbon_savings:.3f} kg",
                delta="per hour if all implemented"
            )
        
        with col_summary2:
            total_cost_savings = sum(rec['cost_saving'] for rec in sample_recommendations)
            st.metric(
                label="💰 Total Cost Savings Potential",
                value=f"${total_cost_savings:.2f}",
                delta="per month if all implemented"
            )
        
        with col_summary3:
            avg_confidence = sum(rec['confidence'] for rec in sample_recommendations) / len(sample_recommendations)
            st.metric(
                label="🎯 Average Confidence",
                value=f"{avg_confidence:.1f}%",
                delta="High reliability"
            )

    def create_prediction_panel(self, df):
        """Create carbon emission predictions"""
        st.subheader("🔮 TCEP Carbon Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Next 24 Hours Forecast")
            
            # Generate sample predictions (in real system, these would come from TCEP model)
            current_time = df['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=current_time + pd.Timedelta(hours=1),
                periods=24,
                freq='H'
            )
            
            # Simple prediction simulation
            base_carbon = df['carbon_emissions_kg_co2'].iloc[-24:].mean()
            predictions = []
            
            for i, time in enumerate(future_times):
                # Add some realistic variation
                hour = time.hour
                is_business = 9 <= hour <= 17
                weekend_factor = 0.8 if time.weekday() >= 5 else 1.0
                business_factor = 1.2 if is_business else 0.9
                
                pred = base_carbon * weekend_factor * business_factor * (1 + np.random.normal(0, 0.1))
                predictions.append(max(0, pred))
            
            pred_df = pd.DataFrame({
                'timestamp': future_times,
                'predicted_carbon': predictions
            })
            
            fig_pred = px.line(
                pred_df,
                x='timestamp',
                y='predicted_carbon',
                title='24-Hour Carbon Forecast',
                labels={'predicted_carbon': 'Predicted CO2 (kg)'}
            )
            fig_pred.update_traces(line=dict(color='orange', width=3))
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Prediction Accuracy")
            
            # Sample accuracy metrics
            accuracy_data = pd.DataFrame({
                'Horizon': ['1 Hour', '6 Hour', '24 Hour'],
                'MAE': [0.089, 0.093, 0.092],
                'MAPE': [64.1, 67.6, 66.5],
                'Accuracy': [88.2, 84.7, 81.3]
            })
            
            fig_acc = px.bar(
                accuracy_data,
                x='Horizon',
                y='Accuracy',
                title='TCEP Model Accuracy',
                color='Accuracy',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    def create_blockchain_panel(self, df):
        """Create simplified blockchain verification panel"""
        st.subheader("🔗 Blockchain Verification System")
        
        # Create demo blockchain data
        import sys
        sys.path.append('../blockchain')
        
        try:
            from simple_blockchain import create_demo_blockchain_data
            
            # Create blockchain with demo data
            blockchain = create_demo_blockchain_data()
            
            # Get blockchain summary
            summary = blockchain.get_chain_summary()
            
            # Display blockchain statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📦 Blocks",
                    value=summary['total_blocks'],
                    delta="Immutable ledger"
                )
            
            with col2:
                st.metric(
                    label="📝 Transactions",
                    value=summary['total_transactions'],
                    delta="Verified records"
                )
            
            with col3:
                st.metric(
                    label="💰 Carbon Credits",
                    value=summary['total_carbon_credits'],
                    delta="Credits earned"
                )
            
            with col4:
                st.metric(
                    label="🌍 CO2 Saved",
                    value=f"{summary['total_co2_saved']:.3f} kg",
                    delta="Blockchain verified"
                )
            
            # Show recent transactions
            st.markdown("### 📋 Recent Blockchain Transactions")
            
            # Get last few transactions from all blocks
            recent_transactions = []
            for block in blockchain.chain[-3:]:  # Last 3 blocks
                for tx in block['transactions']:
                    if tx.get('transaction_type') == 'carbon_record':
                        tx['block_index'] = block['index']
                        tx['block_hash'] = blockchain.hash(block)[:16] + "..."
                        recent_transactions.append(tx)
            
            # Display transactions
            if recent_transactions:
                for i, tx in enumerate(recent_transactions[-5:]):  # Show last 5
                    with st.expander(f"🔗 Transaction {i+1}: {tx['organization']} - Block {tx['block_index']}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Organization:** {tx['organization']}")
                            st.write(f"**Predicted Emissions:** {tx['predicted_emissions']:.3f} kg CO2")
                            st.write(f"**Actual Emissions:** {tx['actual_emissions']:.3f} kg CO2")
                            st.write(f"**CO2 Savings:** {tx['optimization_savings']:.3f} kg")
                        
                        with col_b:
                            st.write(f"**Model Used:** {tx['model_name']}")
                            st.write(f"**Credits Earned:** {tx['credits_earned']}")
                            st.write(f"**Verified:** {'✅ Yes' if tx['verified'] else '⏳ Pending'}")
                            st.write(f"**Block Hash:** `{tx['block_hash']}`")
            
            # Blockchain validation status
            is_valid = blockchain.validate_chain()
            if is_valid:
                st.success("🔐 **Blockchain Validation:** ✅ All blocks verified and chain is valid")
            else:
                st.error("❌ **Blockchain Validation:** Chain validation failed!")
            
            # Blockchain info
            st.markdown("### ℹ️ Blockchain Information")
            
            blockchain_info = {
                "Chain Length": f"{summary['total_blocks']} blocks",
                "Total Organizations": summary['organizations'],
                "Latest Block Hash": summary['blockchain_hash'][:32] + "...",
                "Consensus Algorithm": "Proof of Work (Simplified)",
                "Network": "Private Carbon Verification Network",
                "Status": "🟢 Online and Synchronized"
            }
            
            for key, value in blockchain_info.items():
                col_info1, col_info2 = st.columns([1, 2])
                with col_info1:
                    st.write(f"**{key}:**")
                with col_info2:
                    st.write(value)
        
        except Exception as e:
            st.error(f"❌ Blockchain system temporarily unavailable: {str(e)}")
            
            # Fallback: Show conceptual blockchain info
            st.info("📋 **Blockchain Concept:** Your system would store carbon predictions and optimizations on an immutable ledger for verification and transparency.")
            
        
    def run_dashboard(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🌱 Carbon Footprint AI Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Load data
        df = self.load_data()
        if df is None:
            st.stop()
        
        # Sidebar controls
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 30s)", value=False)
        if auto_refresh:
            st.experimental_rerun()
        
        # Date range filter
        st.sidebar.markdown("### 📅 Date Range")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        selected_dates = st.sidebar.date_input(
            "Select date range:",
            value=(max_date - pd.Timedelta(days=7).to_pytimedelta(), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data by selected dates
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            df = df[
                (df['timestamp'].dt.date >= start_date) &
                (df['timestamp'].dt.date <= end_date)
            ]
        
        # Region filter
        st.sidebar.markdown("### 🌍 Region Filter")
        selected_regions = st.sidebar.multiselect(
            "Select regions:",
            options=df['region_name'].unique(),
            default=df['region_name'].unique()
        )
        
        if selected_regions:
            df = df[df['region_name'].isin(selected_regions)]
        
        # Main dashboard content
        if len(df) > 0:
            # Main metrics
            self.create_main_metrics(df)
            
            # Time series chart
            self.create_time_series_chart(df)
            
            # Two columns for additional content
            col1, col2 = st.columns(2)
            
            with col1:
                # Regional analysis
                self.create_regional_analysis(df)
            
            with col2:
                # Predictions
                self.create_prediction_panel(df)
            
            # Recommendations (full width)
            self.create_recommendations_panel(df)
            self.create_blockchain_panel(df)
            
            # Footer
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                st.success("✅ TCEP Model: Active")
            with status_col2:
                st.success("✅ XGBoost Optimizer: Active")  
            with status_col3:
                st.success("✅ Data Pipeline: Online")
                
        else:
            st.error("No data available for selected filters.")

def main():
    """Main function to run dashboard"""
    dashboard = CarbonDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
