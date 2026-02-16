import pandas as pd

# Load real AWS data
df = pd.read_csv('data/real_aws_baseline.csv')

print("=" * 60)
print("🎉 REAL AWS DATA VERIFICATION")
print("=" * 60)
print(f"\n📊 Total Records: {len(df)}")
print(f"📅 Columns: {list(df.columns)}")
print(f"\n🔍 First 5 rows:")
print(df.head())
print(f"\n📈 Data Summary:")
print(df.describe())
print(f"\n✅ Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"💚 Average Carbon: {df['carbon_kg_per_hour'].mean():.6f} kg/hour")
print(f"💻 Average CPU: {df['cpu_percent'].mean():.3f}%")
print(f"\n📏 Data Shape: {df.shape}")
print(f"\n🎯 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print("\n✅ DATA READY FOR INTEGRATION!")
