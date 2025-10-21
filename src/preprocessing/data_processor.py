import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class CarbonDataProcessor:
    """
    Comprehensive data preprocessing pipeline for carbon footprint prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
    def load_data(self, file_path):
        """Load the carbon footprint dataset"""
        print(f"📂 Loading data from: {file_path}")
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def create_temporal_features(self, df):
        """Create advanced temporal features for machine learning"""
        print("🕐 Creating temporal features...")
        
        df = df.copy()
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rolling features (moving averages and std)
        for window in [6, 24, 168]:  # 6h, 1d, 1week
            df[f'cpu_rolling_mean_{window}h'] = df['cpu_usage_percent'].rolling(
                window=window, min_periods=1).mean()
            df[f'cpu_rolling_std_{window}h'] = df['cpu_usage_percent'].rolling(
                window=window, min_periods=1).std()
            df[f'energy_rolling_mean_{window}h'] = df['energy_consumption_kwh'].rolling(
                window=window, min_periods=1).mean()
        
        # Lag features (previous values)
        for lag in [1, 6, 24]:  # 1h, 6h, 1d ago
            df[f'cpu_lag_{lag}h'] = df['cpu_usage_percent'].shift(lag)
            df[f'memory_lag_{lag}h'] = df['memory_usage_gb'].shift(lag)
            df[f'carbon_lag_{lag}h'] = df['carbon_emissions_kg_co2'].shift(lag)
        
        # Rate of change features
        df['cpu_change_1h'] = df['cpu_usage_percent'].pct_change(periods=1)
        df['memory_change_1h'] = df['memory_usage_gb'].pct_change(periods=1)
        df['energy_change_1h'] = df['energy_consumption_kwh'].pct_change(periods=1)
        
        print(f"✅ Created temporal features. New shape: {df.shape}")
        return df
    
    def create_efficiency_features(self, df):
        """Create efficiency and ratio features"""
        print("⚡ Creating efficiency features...")
        
        df = df.copy()
        
        # Resource utilization ratios
        df['cpu_memory_ratio'] = df['cpu_usage_percent'] / (df['memory_usage_gb'] + 1)
        df['energy_per_cpu'] = df['energy_consumption_kwh'] / (df['cpu_usage_percent'] + 1)
        df['energy_per_memory'] = df['energy_consumption_kwh'] / (df['memory_usage_gb'] + 1)
        df['carbon_intensity_ratio'] = df['effective_carbon_intensity'] / df['base_carbon_intensity']
        
        # Efficiency scores
        df['resource_efficiency'] = (df['cpu_usage_percent'] * df['memory_usage_gb']) / 100
        df['carbon_efficiency'] = df['carbon_emissions_kg_co2'] / (df['energy_consumption_kwh'] + 0.001)
        df['performance_efficiency'] = df['throughput_req_sec'] / (df['response_time_ms'] + 1)
        
        # Load indicators
        df['high_cpu_load'] = (df['cpu_usage_percent'] > 70).astype(int)
        df['high_memory_load'] = (df['memory_usage_gb'] > 32).astype(int)
        df['peak_carbon_time'] = (df['effective_carbon_intensity'] > df['effective_carbon_intensity'].quantile(0.75)).astype(int)
        
        print(f"✅ Created efficiency features. New shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("🔧 Handling missing values...")
        
        missing_count = df.isnull().sum().sum()
        print(f"   Total missing values: {missing_count}")
        
        if missing_count > 0:
            # Forward fill for short gaps, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # For any remaining NaNs, use median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        print(f"✅ Missing values handled. Remaining: {df.isnull().sum().sum()}")
        return df
    
    def select_features(self, df):
        """Select the best features for machine learning"""
        print("🎯 Selecting features for ML...")
        
        # Core resource features
        core_features = [
            'cpu_usage_percent', 'memory_usage_gb', 'storage_io_ops',
            'network_bandwidth_mbps', 'energy_consumption_kwh',
            'response_time_ms', 'throughput_req_sec', 'service_availability'
        ]
        
        # Temporal features
        temporal_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_business_hours', 'is_weekend'
        ]
        
        # Regional features
        regional_features = [
            'region_id', 'base_carbon_intensity', 'renewable_energy_pct',
            'effective_carbon_intensity'
        ]
        
        # Rolling features
        rolling_features = [
            'cpu_rolling_mean_6h', 'cpu_rolling_mean_24h', 'cpu_rolling_mean_168h',
            'energy_rolling_mean_6h', 'energy_rolling_mean_24h'
        ]
        
        # Lag features
        lag_features = [
            'cpu_lag_1h', 'cpu_lag_6h', 'memory_lag_1h',
            'carbon_lag_1h', 'carbon_lag_6h'
        ]
        
        # Efficiency features
        efficiency_features = [
            'cpu_memory_ratio', 'energy_per_cpu', 'resource_efficiency',
            'carbon_efficiency', 'high_cpu_load', 'peak_carbon_time'
        ]
        
        # Combine all feature groups
        selected_features = (core_features + temporal_features + regional_features + 
                           rolling_features + lag_features + efficiency_features)
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        self.feature_columns = available_features
        
        print(f"✅ Selected {len(available_features)} features for ML")
        return available_features
    
    def prepare_ml_data(self, df):
        """Prepare data for machine learning"""
        print("🤖 Preparing data for machine learning...")
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_efficiency_features(df)
        df = self.handle_missing_values(df)
        
        # Select features
        feature_columns = self.select_features(df)
        
        # Prepare X and y
        X = df[feature_columns].copy()
        y = df['carbon_emissions_kg_co2'].copy()
        
        print(f"✅ ML data prepared:")
        print(f"   Features (X): {X.shape}")
        print(f"   Target (y): {y.shape}")
        
        return X, y, df
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train/validation/test sets with temporal consideration"""
        print("📊 Splitting data into train/validation/test sets...")
        
        # Use sequential split to respect temporal order
        total_size = len(X)
        test_start = int(total_size * (1 - test_size))
        val_start = int(test_start * (1 - val_size / (1 - test_size)))
        
        # Sequential splits
        X_train = X.iloc[:val_start]
        y_train = y.iloc[:val_start]
        
        X_val = X.iloc[val_start:test_start]  
        y_val = y.iloc[val_start:test_start]
        
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]
        
        print(f"✅ Data split completed:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("📏 Scaling features...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames with column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        self.is_fitted = True
        
        print("✅ Feature scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data for later use"""
        print("💾 Saving processed data...")
        
        # Create processed data directory
        processed_dir = os.path.join('..', '..', 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save datasets
        datasets = {
            'X_train.csv': X_train,
            'X_val.csv': X_val, 
            'X_test.csv': X_test,
            'y_train.csv': y_train,
            'y_val.csv': y_val,
            'y_test.csv': y_test
        }
        
        for filename, data in datasets.items():
            filepath = os.path.join(processed_dir, filename)
            if hasattr(data, 'to_csv'):
                data.to_csv(filepath, index=False)
            else:
                pd.Series(data).to_csv(filepath, index=False)
        
        print(f"✅ Processed data saved to: {processed_dir}")
        
    def generate_data_report(self, df, X, y):
        """Generate a comprehensive data report"""
        print("📋 Generating data report...")
        
        report = {
            'dataset_info': {
                'total_records': len(df),
                'total_features': len(X.columns),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'target_statistics': {
                'mean_carbon_emissions': round(y.mean(), 4),
                'std_carbon_emissions': round(y.std(), 4),
                'min_carbon_emissions': round(y.min(), 4),
                'max_carbon_emissions': round(y.max(), 4)
            },
            'feature_groups': {
                'temporal_features': len([f for f in X.columns if any(t in f for t in ['hour', 'day', 'month', 'business', 'weekend'])]),
                'resource_features': len([f for f in X.columns if any(r in f for r in ['cpu', 'memory', 'storage', 'network'])]),
                'efficiency_features': len([f for f in X.columns if any(e in f for e in ['ratio', 'efficiency', 'load'])]),
                'regional_features': len([f for f in X.columns if any(r in f for r in ['region', 'carbon_intensity', 'renewable'])])
            }
        }
        
        return report

def main():
    """Main preprocessing pipeline"""
    
    print("🌱 Carbon Footprint Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize processor
    processor = CarbonDataProcessor()
    
    # Load data
    data_path = '../../data/synthetic/carbon_footprint_dataset.csv'
    df = processor.load_data(data_path)
    
    # Prepare ML data
    X, y, processed_df = processor.prepare_ml_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(X_train, X_val, X_test)
    
    # Save processed data
    processor.save_processed_data(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
    
    # Generate report
    report = processor.generate_data_report(processed_df, X, y)
    
    print("\n📋 DATA PROCESSING REPORT")
    print("=" * 40)
    print(f"📊 Total Records: {report['dataset_info']['total_records']:,}")
    print(f"🔢 Total Features: {report['dataset_info']['total_features']}")
    print(f"📅 Date Range: {report['dataset_info']['date_range']}")
    print(f"💾 Memory Usage: {report['dataset_info']['memory_usage_mb']} MB")
    print(f"\n🎯 Target Statistics:")
    print(f"   Mean CO2: {report['target_statistics']['mean_carbon_emissions']} kg/hour")
    print(f"   Std CO2: {report['target_statistics']['std_carbon_emissions']} kg/hour")
    print(f"   Range: {report['target_statistics']['min_carbon_emissions']} - {report['target_statistics']['max_carbon_emissions']} kg/hour")
    print(f"\n🏗️ Feature Groups:")
    print(f"   Temporal: {report['feature_groups']['temporal_features']} features")
    print(f"   Resource: {report['feature_groups']['resource_features']} features")
    print(f"   Efficiency: {report['feature_groups']['efficiency_features']} features")
    print(f"   Regional: {report['feature_groups']['regional_features']} features")
    
    print(f"\n🎉 Preprocessing complete! Ready for ML model training.")
    print(f"Next steps:")
    print(f"1. Build TCEP neural network")
    print(f"2. Train XGBoost optimizer")
    print(f"3. Create prediction dashboard")

if __name__ == "__main__":
    main()
