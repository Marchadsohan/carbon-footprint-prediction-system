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

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.xgboost.carbon_optimizer import CarbonOptimizer

# Page configuration
st.set_page_config(
    page_title="🌱 Carbon Footprint AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

    @st.cache_resource
    def load_optimizer(_self):
        """Load trained XGBoost optimizer from disk"""
        model_dir = '../../../models/xgboost'
        model_path = os.path.join(model_dir, 'carbon_optimizer.json')
        columns_path = os.path.join(model_dir, 'feature_columns.pkl')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        importance_path = os.path.join(model_dir, 'feature_importance.pkl')

        optimizer = CarbonOptimizer()

        import xgboost as xgb
        if os.path.exists(model_path):
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(model_path)
            optimizer.model = xgb_model
            if os.path.exists(columns_path):
                optimizer.feature_columns = joblib.load(columns_path)
            if os.path.exists(encoders_path):
                optimizer.label_encoders = joblib.load(encoders_path)
            if os.path.exists(importance_path):
                optimizer.feature_importance = joblib.load(importance_path)
            optimizer.is_trained = True
            return optimizer

        # If no saved model, train on the fly
        df = _self.load_data()
        if df is None:
            return None

        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        optimizer.train_optimizer(df_sample)
        return optimizer

    def create_main_metrics(self, df):
        """Create main KPI metrics"""
        col1, col2, col3, col4 = st.columns(4)

        current_carbon = df['carbon_emissions_kg_co2'].iloc[-1]
        today_data = df[df['timestamp'].dt.date == df['timestamp'].iloc[-1].date()]
        daily_total = today_data['carbon_emissions_kg_co2'].sum()
        weekly_avg = df['carbon_emissions_kg_co2'].rolling(window=168, min_periods=1).mean().iloc[-1]
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

        col1, _ = st.columns(2)
        with col1:
            days_back = st.selectbox(
                "📅 Time Range",
                options=[7, 30, 90, 365],
                format_func=lambda x: f"Last {x} days",
                index=0,
            )

        end_date = df['timestamp'].max()
        start_date = end_date - pd.Timedelta(days=days_back)
        filtered_df = df[df['timestamp'] >= start_date].copy()

        if days_back <= 7:
            freq = 'H'
            title_freq = "Hourly"
        elif days_back <= 30:
            freq = 'D'
            title_freq = "Daily"
        else:
            freq = 'W'
            title_freq = "Weekly"

        resampled = (
            filtered_df.set_index('timestamp')
            .resample(freq)
            .agg({
                'carbon_emissions_kg_co2': 'mean',
                'energy_consumption_kwh': 'mean',
                'cpu_usage_percent': 'mean',
                'renewable_energy_pct': 'mean',
            })
            .reset_index()
        )

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{title_freq} Carbon Emissions', f'{title_freq} Energy Consumption'),
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=resampled['timestamp'],
                y=resampled['carbon_emissions_kg_co2'],
                mode='lines+markers',
                name='Carbon Emissions (kg CO2)',
                line=dict(color='#ff6b6b', width=2),
                hovertemplate='<b>%{x}</b><br>CO2: %{y:.4f} kg<extra></extra>',
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=resampled['timestamp'],
                y=resampled['energy_consumption_kwh'],
                mode='lines+markers',
                name='Energy (kWh)',
                line=dict(color='#4ecdc4', width=2),
                hovertemplate='<b>%{x}</b><br>Energy: %{y:.4f} kWh<extra></extra>',
            ),
            row=2, col=1,
        )

        fig.update_layout(height=600, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="kg CO2", row=1, col=1)
        fig.update_yaxes(title_text="kWh", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def create_regional_analysis(self, df):
        """Create regional carbon intensity analysis"""
        st.subheader("🌍 Regional Carbon Analysis")

        col1, col2 = st.columns(2)

        regional_data = (
            df.groupby('region_name')
            .agg({
                'carbon_emissions_kg_co2': 'sum',
                'renewable_energy_pct': 'mean',
                'base_carbon_intensity': 'mean',
            })
            .reset_index()
        )

        with col1:
            fig_pie = px.pie(
                regional_data,
                values='carbon_emissions_kg_co2',
                names='region_name',
                title="Total Emissions by Region",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                regional_data,
                x='renewable_energy_pct',
                y='base_carbon_intensity',
                size='carbon_emissions_kg_co2',
                hover_name='region_name',
                title="Carbon Intensity vs Renewable Energy",
                labels={
                    'renewable_energy_pct': 'Renewable Energy %',
                    'base_carbon_intensity': 'Base Carbon Intensity',
                },
                color='carbon_emissions_kg_co2',
                color_continuous_scale='RdYlGn_r',
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    def create_recommendations_panel(self, df):
        """Create AI recommendations panel - 6 unique strategies in exact order"""
        st.subheader("🤖 AI-Powered Optimization Recommendations")

        optimizer = self.load_optimizer()
        if optimizer is None or not optimizer.is_trained:
            st.warning("⚠️ Optimizer model not available.")
            return

        # Get latest single record to avoid duplicates
        current_data = df.tail(1).copy()
        
        # Define exact order we want
        desired_order = [
            'temporal_shift', 'geographic_shift', 'resource_optimization',
            'CBSD', 'DCTR', 'GPCO'
        ]
        
        # Get all recommendations
        all_recs = optimizer.generate_recommendations(current_data, top_n=20)
        
        # Group by type, pick best of each
        rec_by_type = {}
        for rec in all_recs:
            rec_type = rec['type']
            if rec_type not in rec_by_type:
                rec_by_type[rec_type] = rec
            elif rec['predicted_carbon_saving'] > rec_by_type[rec_type]['predicted_carbon_saving']:
                rec_by_type[rec_type] = rec
        
        # Create ordered list
        recommendations = []
        for rec_type in desired_order:
            if rec_type in rec_by_type:
                recommendations.append(rec_by_type[rec_type])

        # Metadata for display
        type_meta = {
            'temporal_shift': ('⏰', 'Temporal Optimization'),
            'geographic_shift': ('🌍', 'Geographic Migration'),
            'resource_optimization': ('⚙️', 'Resource Right-sizing'),
            'CBSD': ('♻️', 'Carbon Budget-Aware Degradation'),
            'DCTR': ('🛡️', 'Carbon-Tiered Reliability'),
            'GPCO': ('🧠', 'Peak-Time Carbon Orchestrator'),
        }

        total_carbon_saving = 0.0
        total_cost_saving = 0.0
        confidences = []

        for i, rec in enumerate(recommendations, 1):
            icon, title = type_meta.get(rec['type'], ('✨', rec['type']))

            with st.container():
                st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #28a745, #20c997);
                                color: white; padding: 10px; border-radius: 5px;
                                margin-bottom: 10px;">
                        <h3 style="margin: 0; color: white;">{icon} {title}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                col_desc, col_carbon, col_cost, col_conf = st.columns([3, 1.5, 1.5, 1.5])

                with col_desc:
                    st.write(f"**Description:** {rec['description']}")
                    st.caption(f"⚡ **Implementation Effort:** {rec['implementation_effort']}")

                with col_carbon:
                    st.metric(
                        label="💨 CO2 Saved",
                        value=f"{rec['predicted_carbon_saving']:.3f} kg",
                        delta="Estimated",
                    )

                with col_cost:
                    st.metric(
                        label="💰 Cost Saved",
                        value=f"${rec['estimated_cost_saving']:.2f}",
                        delta="per month (approx)",
                    )

                with col_conf:
                    conf = rec.get('confidence', 70)
                    if conf > 80:
                        conf_label = "🟢 High"
                    elif conf > 60:
                        conf_label = "🟡 Medium"
                    else:
                        conf_label = "🔴 Low"

                    st.metric(
                        label="🎯 Confidence",
                        value=f"{conf:.1f}%",
                        delta=conf_label,
                    )

                st.markdown("---")

            total_carbon_saving += rec['predicted_carbon_saving']
            total_cost_saving += rec['estimated_cost_saving']
            confidences.append(rec.get('confidence', 70))

        st.markdown("### 📈 Optimization Summary")

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.metric(
                label="🌍 Total CO2 Savings Potential",
                value=f"{total_carbon_saving:.3f} kg",
                delta="if all applied",
            )

        with col_s2:
            st.metric(
                label="💰 Total Cost Savings Potential",
                value=f"${total_cost_saving:.2f}",
                delta="approx per month",
            )

        with col_s3:
            avg_conf = np.mean(confidences) if confidences else 0.0
            st.metric(
                label="🎯 Average Confidence",
                value=f"{avg_conf:.1f}%",
                delta="Model estimated",
            )

    def create_prediction_panel(self, df):
        """Create carbon emission predictions"""
        st.subheader("🔮 TCEP Carbon Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Next 24 Hours Forecast")

            current_time = df['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=current_time + pd.Timedelta(hours=1),
                periods=24,
                freq='H',
            )

            base_carbon = df['carbon_emissions_kg_co2'].iloc[-24:].mean()
            predictions = []

            for time in future_times:
                hour = time.hour
                is_business = 9 <= hour <= 17
                weekend_factor = 0.8 if time.weekday() >= 5 else 1.0
                business_factor = 1.2 if is_business else 0.9

                pred = base_carbon * weekend_factor * business_factor * (1 + np.random.normal(0, 0.1))
                predictions.append(max(0, pred))

            pred_df = pd.DataFrame({'timestamp': future_times, 'predicted_carbon': predictions})

            fig_pred = px.line(
                pred_df,
                x='timestamp',
                y='predicted_carbon',
                title='24-Hour Carbon Forecast',
                labels={'predicted_carbon': 'Predicted CO2 (kg)'},
            )
            fig_pred.update_traces(line=dict(color='orange', width=3))
            st.plotly_chart(fig_pred, use_container_width=True)

        with col2:
            st.markdown("### 📈 Prediction Accuracy")

            accuracy_data = pd.DataFrame({
                'Horizon': ['1 Hour', '6 Hour', '24 Hour'],
                'Accuracy': [88.2, 84.7, 81.3],
            })

            fig_acc = px.bar(
                accuracy_data,
                x='Horizon',
                y='Accuracy',
                title='TCEP Model Accuracy (Demo)',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    def create_blockchain_panel(self, df):
        """Create simplified blockchain verification panel"""
        st.subheader("🔗 Blockchain Verification System")

        import sys as _sys
        _sys.path.append('../blockchain')

        try:
            from simple_blockchain import create_demo_blockchain_data

            blockchain = create_demo_blockchain_data()
            summary = blockchain.get_chain_summary()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(label="📦 Blocks", value=summary['total_blocks'], delta="Immutable ledger")

            with col2:
                st.metric(label="📝 Transactions", value=summary['total_transactions'], delta="Verified records")

            with col3:
                st.metric(label="💰 Carbon Credits", value=summary['total_carbon_credits'], delta="Credits earned")

            with col4:
                st.metric(label="🌍 CO2 Saved", value=f"{summary['total_co2_saved']:.3f} kg", delta="Blockchain verified")

            st.markdown("### 📋 Recent Blockchain Transactions")

            recent_transactions = []
            for block in blockchain.chain[-3:]:
                for tx in block['transactions']:
                    if tx.get('transaction_type') == 'carbon_record':
                        tx['block_index'] = block['index']
                        tx['block_hash'] = blockchain.hash(block)[:16] + "..."
                        recent_transactions.append(tx)

            if recent_transactions:
                for i, tx in enumerate(recent_transactions[-5:]):
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

            is_valid = blockchain.validate_chain()
            if is_valid:
                st.success("🔐 **Blockchain Validation:** ✅ All blocks verified and chain is valid")
            else:
                st.error("❌ **Blockchain Validation:** Chain validation failed!")

        except Exception as e:
            st.error(f"❌ Blockchain system temporarily unavailable: {str(e)}")
            st.info("📋 Blockchain concept: storing carbon predictions and optimizations on an immutable ledger for verification and transparency.")

    def run_dashboard(self):
        """Run the main dashboard"""
        st.markdown('<h1 class="main-header">🌱 Carbon Footprint AI Dashboard</h1>', unsafe_allow_html=True)

        df = self.load_data()
        if df is None:
            st.stop()

        st.sidebar.markdown("## 🎛️ Dashboard Controls")

        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 30s)", value=False)
        if auto_refresh:
            st.experimental_rerun()

        st.sidebar.markdown("### 📅 Date Range")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        default_start = max_date - pd.Timedelta(days=7).to_pytimedelta()

        selected_dates = st.sidebar.date_input(
            "Select date range:",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

        st.sidebar.markdown("### 🌍 Region Filter")
        selected_regions = st.sidebar.multiselect(
            "Select regions:",
            options=df['region_name'].unique(),
            default=df['region_name'].unique(),
        )

        if selected_regions:
            df = df[df['region_name'].isin(selected_regions)]

        if len(df) > 0:
            self.create_main_metrics(df)
            self.create_time_series_chart(df)

            col1, col2 = st.columns(2)

            with col1:
                self.create_regional_analysis(df)

            with col2:
                self.create_prediction_panel(df)

            self.create_recommendations_panel(df)
            self.create_blockchain_panel(df)

            st.markdown("---")
            st.markdown("### 📊 System Status")

            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                st.success("✅ TCEP Model: Active (demo)")
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
