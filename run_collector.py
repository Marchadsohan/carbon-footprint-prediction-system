"""
Multicloud Carbon Collector
Run: python run_collector.py
Automatically collects from ALL clouds configured in .env
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cloud_connectors.connector_factory import get_all_configured_connectors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collector.log', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
os.makedirs('logs', exist_ok=True)


def run():
    print("\nMulticloud Carbon Collector")
    print("=" * 60)

    connectors = get_all_configured_connectors()

    if not connectors:
        print("No clouds configured. Add keys to .env")
        return

    all_dfs = []

    for connector in connectors:
        provider = connector.config.get("provider", "unknown").upper()
        print(f"\nCollecting from {provider}...")

        try:
            df = connector.collect_and_calculate()
            df["cloud_provider"] = provider
            all_dfs.append(df)
            filepath = connector.save_data(df)
            print(f"{provider}: {len(df)} records saved to {filepath}")
        except Exception as e:
            logger.error(f"{provider} collection failed: {e}")

    # Merge all clouds into one master CSV
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_path = f"data/realtime/multicloud_{ts}.csv"
        master_df.to_csv(master_path, index=False)

        print("\n" + "=" * 60)
        print(f"TOTAL: {len(master_df)} records from {len(all_dfs)} cloud(s)")
        print(f"Master CSV: {master_path}")
        print(f"Total carbon: {master_df['carbon_kg'].sum():.6f} kgCO2")
        print("Ready for dashboard and XGBoost!")


if __name__ == "__main__":
    run()
