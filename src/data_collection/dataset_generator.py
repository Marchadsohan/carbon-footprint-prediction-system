import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_carbon_dataset(num_hours=8760, start_date=None):
    """
    Generate realistic synthetic carbon footprint dataset
    
    Args:
        num_hours: Number of hours to generate (8760 = 1 year)
        start_date: Start date (default: 2024-01-01)
    
    Returns:
        pandas.DataFrame: Synthetic carbon footprint data
    """
    
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    print(f"🌱 Generating {num_hours} hours of carbon footprint data...")
    
    # Define regions with different carbon characteristics
    regions = [
        {'id': 1, 'name': 'Northeast', 'base_carbon': 0.35, 'renewable_pct': 0.20},
        {'id': 2, 'name': 'Southeast', 'base_carbon': 0.65, 'renewable_pct': 0.07},
        {'id': 3, 'name': 'Midwest', 'base_carbon': 0.55, 'renewable_pct': 0.15},
        {'id': 4, 'name': 'Southwest', 'base_carbon': 0.45, 'renewable_pct': 0.30},
        {'id': 5, 'name': 'West', 'base_carbon': 0.25, 'renewable_pct': 0.40}
    ]
    
    # Generate time series
    dates = [start_date + timedelta(hours=i) for i in range(num_hours)]
    
    data = []
    for i, timestamp in enumerate(dates):
        # Select random region
        region = random.choice(regions)
        
        # Extract time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Business patterns
        is_business_hours = 9 <= hour <= 17 and day_of_week < 5
        is_weekend = day_of_week >= 5
        
        # Seasonal effects (higher usage in summer/winter)
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        
        # Generate realistic resource usage
        base_cpu = 25 + 15 * seasonal_factor
        if is_business_hours:
            base_cpu += 25  # Higher usage during business hours
        if not is_weekend:
            base_cpu += 10  # Higher on weekdays
            
        # Add random variation
        cpu_usage = max(5, min(95, base_cpu + random.gauss(0, 12)))
        
        # Memory usage correlates with CPU but has its own pattern
        base_memory = 16 + 0.3 * cpu_usage + random.gauss(0, 4)
        memory_usage = max(4, min(64, base_memory))
        
        # Storage and network usage
        storage_io_ops = random.randint(50, 800)
        
        # ✅ FIXED: Use expovariate instead of exponential
        network_bandwidth = max(1, random.expovariate(1/15))
        
        # Calculate energy consumption (simplified model)
        # CPU: ~100W per core at 100%, Memory: ~5W per GB, Base: ~200W
        cpu_energy = (cpu_usage / 100) * 0.1  # kWh per hour
        memory_energy = memory_usage * 0.005  # kWh per hour  
        base_energy = 0.2  # Base system energy
        
        total_energy = cpu_energy + memory_energy + base_energy
        
        # Apply Power Usage Effectiveness (PUE) - datacenter efficiency
        pue_factor = random.uniform(1.3, 1.6)  # Typical datacenter PUE
        energy_consumption_kwh = total_energy * pue_factor
        
        # Performance metrics
        response_time = 120 + 50 * (cpu_usage / 100) + random.gauss(0, 20)
        response_time = max(50, response_time)
        
        throughput = max(100, 400 - 2 * (cpu_usage - 50) + random.gauss(0, 30))
        
        service_availability = max(98.0, 99.9 - 0.01 * (cpu_usage / 100) + random.gauss(0, 0.1))
        
        # Calculate carbon intensity
        # Higher renewable % = lower effective carbon intensity
        effective_carbon_intensity = region['base_carbon'] * (1 - region['renewable_pct'])
        
        # Add time-of-day carbon variation (solar/wind patterns)
        if 10 <= hour <= 16:  # Solar peak hours
            renewable_boost = 0.1 * region['renewable_pct']
            effective_carbon_intensity *= (1 - renewable_boost)
        
        # Calculate final carbon emissions
        carbon_emissions = energy_consumption_kwh * effective_carbon_intensity
        
        # Create data record
        record = {
            'timestamp': timestamp,
            'cpu_usage_percent': round(cpu_usage, 2),
            'memory_usage_gb': round(memory_usage, 2),
            'storage_io_ops': storage_io_ops,
            'network_bandwidth_mbps': round(network_bandwidth, 2),
            'energy_consumption_kwh': round(energy_consumption_kwh, 4),
            'response_time_ms': round(response_time, 2),
            'throughput_req_sec': round(throughput, 0),
            'service_availability': round(service_availability, 3),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_business_hours': is_business_hours,
            'is_weekend': is_weekend,
            'region_id': region['id'],
            'region_name': region['name'],
            'base_carbon_intensity': region['base_carbon'],
            'renewable_energy_pct': region['renewable_pct'],
            'effective_carbon_intensity': round(effective_carbon_intensity, 4),
            'carbon_emissions_kg_co2': round(carbon_emissions, 6)
        }
        
        data.append(record)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"📊 Generated {i + 1:,} records...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset generation complete!")
    print(f"📈 Shape: {df.shape}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"🌍 Regions: {df['region_name'].unique()}")
    print(f"💨 Avg carbon emissions: {df['carbon_emissions_kg_co2'].mean():.4f} kg CO2/hour")
    
    return df

def save_dataset(df, filename='carbon_footprint_dataset.csv'):
    """Save dataset to CSV file"""
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join('..', '..', 'data', 'synthetic')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save file
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"💾 Dataset saved to: {filepath}")
    return filepath

def main():
    """Main function to generate and save dataset"""
    
    print("🌱 Carbon Footprint Dataset Generator")
    print("=" * 50)
    
    # Generate dataset
    df = generate_carbon_dataset(num_hours=8760)  # 1 year of data
    
    # Display basic statistics
    print("\n📊 Dataset Summary:")
    print("-" * 30)
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Show sample data
    print("\n🔍 Sample Data (first 5 rows):")
    print("-" * 50)
    print(df.head())
    
    # Show statistics
    print("\n📈 Key Statistics:")
    print("-" * 30)
    numeric_cols = ['cpu_usage_percent', 'memory_usage_gb', 'energy_consumption_kwh', 'carbon_emissions_kg_co2']
    print(df[numeric_cols].describe())
    
    # Save dataset
    save_dataset(df)
    
    print("\n🎉 Ready for next steps!")
    print("Now you can build your ML models with this data!")

if __name__ == "__main__":
    main()
