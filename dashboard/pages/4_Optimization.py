"""
Optimization Strategies Dashboard
AI-powered recommendations for carbon reduction
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os

st.set_page_config(page_title="Optimization", page_icon="🎯", layout="wide")

st.title("🎯 Carbon Optimization Strategies")
st.markdown("AI-Powered Recommendations Based on Real-Time Analysis")

st.markdown("---")

# Load latest predictions with recommendations
try:
    pred_files = [f for f in os.listdir('data/predictions') if f.endswith('.json')]
    
    if pred_files:
        latest_pred_file = max([os.path.join('data/predictions', f) for f in pred_files],
                              key=os.path.getmtime)
        
        with open(latest_pred_file, 'r') as f:
            predictions = json.load(f)
        
        if predictions:
            latest_prediction = predictions[-1]
            recommendations = latest_prediction['recommendations']
            current_carbon = latest_prediction['current_metrics']['daily_carbon_kg']
            
            # Summary metrics
            st.subheader("📊 Optimization Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Daily Carbon",
                    f"{current_carbon:.6f} kg CO2"
                )
            
            with col2:
                max_reduction = latest_prediction['max_reduction_percentage']
                st.metric(
                    "Max Reduction Potential",
                    f"{max_reduction:.1f}%",
                    help="Maximum achievable reduction"
                )
            
            with col3:
                total_saving = latest_prediction['total_potential_saving_kg']
                st.metric(
                    "Total Saving Potential",
                    f"{total_saving:.6f} kg CO2/day"
                )
            
            with col4:
                annual_cost = total_saving * 6.70 * 365
                st.metric(
                    "Annual Cost Savings",
                    f"${annual_cost:.2f}"
                )
            
            st.markdown("---")
            
            # Strategy comparison chart
            st.subheader("📈 Strategy Comparison")
            
            strategy_data = []
            for rec in recommendations:
                strategy_data.append({
                    'Strategy': rec['strategy'].replace('_', ' '),
                    'Reduction %': rec['reduction_percentage'],
                    'Saving (kg CO2/day)': rec['estimated_saving_kg'],
                    'Priority': rec['priority'],
                    'Difficulty': rec['difficulty']
                })
            
            df_strategies = pd.DataFrame(strategy_data)
            
            fig_comparison = go.Figure()
            
            # Color by priority
            colors = {
                'HIGH': '#FF6B6B',
                'MEDIUM': '#FFA500',
                'LOW': '#4ECDC4'
            }
            
            for priority in ['HIGH', 'MEDIUM', 'LOW']:
                df_priority = df_strategies[df_strategies['Priority'] == priority]
                if not df_priority.empty:
                    fig_comparison.add_trace(go.Bar(
                        x=df_priority['Reduction %'],
                        y=df_priority['Strategy'],
                        name=priority,
                        orientation='h',
                        marker_color=colors[priority]
                    ))
            
            fig_comparison.update_layout(
                xaxis_title="Carbon Reduction (%)",
                yaxis_title="Strategy",
                barmode='group',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed recommendations
            st.subheader("💡 Detailed Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['strategy']} - **{rec['priority']} Priority**", expanded=(i <= 3)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Implementation:**")
                        st.info(rec['implementation'])
                        
                        st.markdown(f"**Expected Impact:**")
                        st.success(f"""
                        - Carbon Reduction: **{rec['reduction_percentage']}%**
                        - Daily Saving: **{rec['estimated_saving_kg']:.6f} kg CO2**
                        - Annual Saving: **{rec['estimated_saving_kg'] * 365:.3f} kg CO2**
                        - Cost Savings: **${rec['estimated_saving_kg'] * 6.70 * 365:.2f}/year**
                        """)
                    
                    with col2:
                        st.markdown("**Metrics**")
                        st.metric("Priority", rec['priority'])
                        st.metric("Difficulty", rec['difficulty'])
                        st.metric("Reduction", f"{rec['reduction_percentage']}%")
                        
                        # Calculate implementation score
                        priority_score = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[rec['priority']]
                        difficulty_score = {'Low': 3, 'Medium': 2, 'High': 1}[rec['difficulty']]
                        impact_score = rec['reduction_percentage'] / 10
                        total_score = (priority_score + difficulty_score + impact_score) / 3
                        
                        st.progress(total_score / 5)
                        st.caption(f"Implementation Score: {total_score:.1f}/5")
            
            st.markdown("---")
            
            # Combined optimization impact
            st.subheader("🚀 Combined Optimization Impact")
            
            # Calculate cumulative impact
            baseline = current_carbon
            optimized_values = [baseline]
            labels = ['Baseline']
            
            cumulative_reduction = 0
            for rec in recommendations[:3]:  # Top 3 strategies
                reduction = baseline * (rec['reduction_percentage'] / 100)
                cumulative_reduction += reduction
                optimized = baseline - cumulative_reduction
                optimized_values.append(max(optimized, 0))
                labels.append(rec['strategy'].split('_')[0])
            
            fig_cumulative = go.Figure()
            
            fig_cumulative.add_trace(go.Waterfall(
                name="Carbon Impact",
                orientation="v",
                x=labels,
                y=[baseline] + [-v for v in [optimized_values[i] - optimized_values[i-1] 
                                             for i in range(1, len(optimized_values))]],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#00D26A"}},
                increasing={"marker": {"color": "#FF6B6B"}},
                totals={"marker": {"color": "#667eea"}}
            ))
            
            fig_cumulative.update_layout(
                title="Cumulative Optimization Impact",
                yaxis_title="Daily Carbon Emissions (kg CO2)",
                height=400
            )
            
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            # Final impact summary
            final_carbon = optimized_values[-1]
            total_reduction_pct = ((baseline - final_carbon) / baseline * 100) if baseline > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Baseline Carbon",
                    f"{baseline:.6f} kg CO2/day"
                )
            
            with col2:
                st.metric(
                    "Optimized Carbon",
                    f"{final_carbon:.6f} kg CO2/day",
                    f"-{total_reduction_pct:.1f}%"
                )
            
            with col3:
                annual_impact = (baseline - final_carbon) * 365 * 6.70
                st.metric(
                    "Annual Impact",
                    f"${annual_impact:.2f}",
                    help="Cost savings from carbon reduction"
                )
            
        else:
            st.warning("⚠️ No recommendations generated yet")
    
    else:
        st.warning("⚠️ No optimization data available")
        st.info("Recommendations are generated during monitoring")

except Exception as e:
    st.error(f"Error loading optimization data: {e}")

# Strategy guide
with st.expander("📚 Strategy Implementation Guide"):
    st.markdown("""
    ### 1. Temporal Shift to Off-Peak Hours
    **Reduction Potential:** 35.8%
    - Schedule workloads during low-carbon periods (00:00-06:00)
    - Use AWS Lambda scheduled functions
    - Implement workload queuing systems
    
    ### 2. Geographic Migration to Low-Carbon Regions
    **Reduction Potential:** 73.5%
    - Migrate to regions with renewable energy (us-west-2, ca-central-1)
    - Use AWS global infrastructure strategically
    - Consider data residency requirements
    
    ### 3. Resource Right-Sizing
    **Reduction Potential:** 24.1%
    - Analyze CPU/memory utilization
    - Downsize over-provisioned instances
    - Use AWS Compute Optimizer recommendations
    
    ### 4. Code Optimization
    **Reduction Potential:** 18.2%
    - Profile application performance
    - Optimize algorithms and queries
    - Reduce unnecessary computations
    
    ### 5. Dynamic Reliability Adjustment
    **Reduction Potential:** 29.8%
    - Adjust replication based on criticality
    - Use single-AZ for non-critical workloads
    - Implement smart failover strategies
    
    ### 6. Peak-Time Orchestration
    **Reduction Potential:** 42.6%
    - Intelligent workload distribution
    - Auto-scaling based on carbon intensity
    - Priority-based task scheduling
    """)
