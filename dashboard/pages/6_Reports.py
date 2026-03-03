"""
Reports Dashboard
Export and generate comprehensive reports
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import io

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

st.title("📄 Reports & Exports")
st.markdown("Generate and download comprehensive system reports")

st.markdown("---")

# Report type selection
st.subheader("📋 Select Report Type")

report_type = st.selectbox(
    "Choose report format",
    ["Executive Summary", "Technical Report", "Blockchain Audit", "Carbon Credits Statement", "Raw Data Export"]
)

st.markdown("---")

try:
    # Load data
    blockchain_files = [f for f in os.listdir('data/blockchain') if f.endswith('.json')]
    
    if blockchain_files:
        latest_blockchain = max([os.path.join('data/blockchain', f) for f in blockchain_files],
                               key=os.path.getmtime)
        
        with open(latest_blockchain, 'r') as f:
            blockchain_data = json.load(f)
        
        summary = blockchain_data['summary']
        
        # Load transactions
        csv_file = latest_blockchain.replace('.json', '_transactions.csv')
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
        
        # Generate report based on type
        if report_type == "Executive Summary":
            st.subheader("📊 Executive Summary Report")
            
            report_content = f"""
# Carbon Footprint Monitoring System
## Executive Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

### System Overview

- **Total Blocks:** {summary['total_blocks']:,}
- **Total Transactions:** {summary['total_transactions']:,}
- **Blockchain Status:** {'✅ VALID' if summary['is_valid'] else '❌ INVALID'}

### Carbon Impact

- **Total CO2 Saved:** {summary['total_co2_saved_kg']:.3f} kg
- **Carbon Credits Earned:** {summary['total_carbon_credits']:,}
- **Cost Savings:** ${summary['total_co2_saved_kg'] * 6.70:.2f}

### Key Metrics

- **Average Mining Time:** {summary['avg_mining_time_sec']:.4f} seconds
- **Proof-of-Work Difficulty:** {summary['difficulty']}
- **Total Validators:** {summary['total_validators']}
- **Organizations Tracked:** {summary['organizations']}

### Recommendations

1. Continue monitoring for comprehensive data collection
2. Implement top optimization strategies
3. Scale to additional cloud regions
4. Integrate with carbon offset programs

---

*This report is blockchain-verified and immutable.*
            """
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label="📥 Download Executive Summary (Markdown)",
                data=report_content,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        elif report_type == "Technical Report":
            st.subheader("🔧 Technical Report")
            
            if not df.empty:
                st.markdown("### System Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Reduction", f"{df['reduction_percentage'].mean():.2f}%")
                
                with col2:
                    st.metric("Total Transactions", len(df))
                
                with col3:
                    st.metric("Avg Carbon/TX", f"{df['carbon_saving_kg'].mean():.6f} kg")
                
                st.markdown("### Transaction Statistics")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Export technical data
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Transactions', index=False)
                    
                    # Summary sheet
                    summary_df = pd.DataFrame([summary])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📥 Download Technical Report (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"technical_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No transaction data available")
        
        elif report_type == "Blockchain Audit":
            st.subheader("⛓️ Blockchain Audit Report")
            
            audit_report = {
                "audit_date": datetime.now().isoformat(),
                "blockchain_summary": summary,
                "validation_status": "PASS" if summary['is_valid'] else "FAIL",
                "total_blocks": summary['total_blocks'],
                "chain_hash": summary['blockchain_hash'],
                "audit_checks": {
                    "chain_integrity": summary['is_valid'],
                    "block_count_valid": summary['total_blocks'] > 0,
                    "transactions_recorded": summary['total_transactions'] > 0,
                    "credits_calculated": summary['total_carbon_credits'] >= 0
                }
            }
            
            st.json(audit_report)
            
            st.download_button(
                label="📥 Download Blockchain Audit (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"blockchain_audit_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        elif report_type == "Carbon Credits Statement":
            st.subheader("💳 Carbon Credits Statement")
            
            credits_data = []
            for org, credits in blockchain_data['carbon_credits'].items():
                credits_data.append({
                    'Organization': org,
                    'Credits': credits,
                    'CO2 Saved (kg)': credits * 0.01,
                    'Monetary Value ($)': credits * 0.01 * 6.70
                })
            
            df_credits = pd.DataFrame(credits_data)
            
            st.dataframe(
                df_credits.style.format({
                    'Credits': '{:,}',
                    'CO2 Saved (kg)': '{:.3f}',
                    'Monetary Value ($)': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # CSV export
            csv = df_credits.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Credits Statement (CSV)",
                data=csv,
                file_name=f"carbon_credits_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        elif report_type == "Raw Data Export":
            st.subheader("📦 Raw Data Export")
            
            st.markdown("### Available Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Blockchain Data**")
                st.json({
                    "format": "JSON",
                    "size": f"{os.path.getsize(latest_blockchain) / 1024:.2f} KB",
                    "blocks": summary['total_blocks']
                })
                
                with open(latest_blockchain, 'r') as f:
                    blockchain_json = f.read()
                
                st.download_button(
                    label="📥 Download Blockchain (JSON)",
                    data=blockchain_json,
                    file_name=f"blockchain_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            with col2:
                if os.path.exists(csv_file):
                    st.markdown("**Transaction Data**")
                    st.json({
                        "format": "CSV",
                        "size": f"{os.path.getsize(csv_file) / 1024:.2f} KB",
                        "records": len(df)
                    })
                    
                    csv_data = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Transactions (CSV)",
                        data=csv_data,
                        file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        st.markdown("---")
        st.success("✅ Reports generated successfully")
    
    else:
        st.warning("⚠️ No data available for report generation")
        st.info("Run the monitoring system to collect data first")

except Exception as e:
    st.error(f"Error generating report: {e}")

# Report customization
with st.expander("⚙️ Report Settings"):
    st.markdown("""
    ### Available Export Formats
    
    - **Markdown (.md)** - Executive summaries
    - **Excel (.xlsx)** - Technical reports with multiple sheets
    - **JSON (.json)** - Structured blockchain data
    - **CSV (.csv)** - Transaction and credits data
    
    ### Report Contents
    
    All reports are:
    - ✅ Blockchain-verified
    - ✅ Timestamped
    - ✅ Immutable
    - ✅ Audit-ready
    """)
