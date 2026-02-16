"""
AWS Real Data Loader
Replaces synthetic data with real AWS monitoring data (8.8 days, 2,540 records)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class AWSDataLoader:
    """Load and process real AWS EC2 monitoring data"""
    
    def __init__(self, data_path='data/real_aws_baseline.csv'):
        self.data_path = data_path
        self.df = None
        self.metrics = {}
        
    def load_data(self):
        """Load real AWS data from 8.8 days of monitoring"""
        print("🔄 Loading REAL AWS data...")
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"✅ Loaded {len(self.df)} records")
        print(f"📅 Duration: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def get_baseline_metrics(self):
        """Calculate baseline metrics from real AWS data"""
        if self.df is None:
            self.load_data()
        
        duration_hours = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600
        
        self.metrics = {
            'total_records': len(self.df),
            'duration_days': duration_hours / 24,
            'duration_hours': duration_hours,
            'avg_cpu_percent': self.df['cpu_percent'].mean(),
            'max_cpu_percent': self.df['cpu_percent'].max(),
            'avg_memory_percent': self.df['memory_percent'].mean(),
            'avg_carbon_per_hour': self.df['carbon_kg_per_hour'].mean(),
            'total_carbon_kg': self.df['carbon_kg_per_5min'].sum(),
            'region': 'us-east-1',
            'carbon_intensity': 0.415,  # kg CO2/kWh for Virginia
            'instance_type': 't2.micro',
            'instance_cost_per_hour': 0.0116
        }
        
        return self.metrics
    
    def prepare_for_models(self):
        """Prepare data format for TCEP and XGBoost models"""
        if self.df is None:
            self.load_data()
        
        # Add time features for ML models
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        
        # Feature engineering
        self.df['cpu_rolling_avg'] = self.df['cpu_percent'].rolling(window=12, min_periods=1).mean()
        self.df['memory_rolling_avg'] = self.df['memory_percent'].rolling(window=12, min_periods=1).mean()
        
        return self.df
    
    def save_processed(self, output_path='data/processed/aws_baseline_processed.csv'):
        """Save processed data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Saved processed data to {output_path}")

# Test the loader
if __name__ == "__main__":
    loader = AWSDataLoader()
    df = loader.load_data()
    metrics = loader.get_baseline_metrics()
    
    print("\n" + "="*60)
    print("📊 REAL AWS BASELINE METRICS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
