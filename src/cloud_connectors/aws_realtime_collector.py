"""
Real-Time AWS Data Collector
Integrates with existing carbon-footprint-prediction-system
Collects live data from AWS CloudWatch and calculates carbon emissions
"""

import boto3
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aws_realtime_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AWSRealtimeCollector:
    """
    Real-time AWS CloudWatch data collector
    Integrates with your existing LSTM/XGBoost models
    """
    
    # Carbon intensity by AWS region (kg CO2/kWh) - from your research
    REGION_CARBON_INTENSITY = {
        'us-east-1': 0.000379,
        'us-east-2': 0.000744,
        'us-west-1': 0.000251,
        'us-west-2': 0.000114,
        'eu-west-1': 0.000316,
        'eu-central-1': 0.000338,
        'ap-southeast-1': 0.000493,
        'ap-northeast-1': 0.000506,
        'ca-central-1': 0.000130,
    }
    
    # Instance power consumption (watts)
    INSTANCE_POWER = {
        't2.micro': 5,
        't2.small': 10,
        't2.medium': 20,
        't2.large': 40,
        't2.xlarge': 80,
        't3.micro': 5,
        't3.small': 10,
        't3.medium': 20,
        't3.large': 40,
        't3.xlarge': 80,
        'm5.large': 50,
        'm5.xlarge': 100,
        'm5.2xlarge': 200,
        'c5.large': 60,
        'c5.xlarge': 120,
        'c5.2xlarge': 240,
    }
    
    def __init__(self, region='us-east-1'):
        """Initialize AWS clients"""
        self.region = region
        
        try:
            # Initialize AWS clients
            self.ec2_client = boto3.client('ec2', region_name=region)
            self.cloudwatch = boto3.client('cloudwatch', region_name=region)
            
            logger.info(f"✅ AWS Realtime Collector initialized for region: {region}")
            logger.info(f"   Carbon intensity: {self.REGION_CARBON_INTENSITY.get(region, 0.0004)} kg CO2/kWh")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {e}")
            logger.info("💡 Make sure AWS credentials are configured:")
            logger.info("   - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
            logger.info("   - Or run: aws configure")
            raise
    
    def get_running_instances(self) -> List[Dict]:
        """Get all running EC2 instances"""
        try:
            response = self.ec2_client.describe_instances(
                Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    
                    # Get instance name from tags
                    instance_name = 'unnamed'
                    if 'Tags' in instance:
                        for tag in instance['Tags']:
                            if tag['Key'] == 'Name':
                                instance_name = tag['Value']
                    
                    instances.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'instance_name': instance_name,
                        'launch_time': instance['LaunchTime'],
                        'region': self.region,
                        'availability_zone': instance['Placement']['AvailabilityZone'],
                        'state': instance['State']['Name']
                    })
            
            logger.info(f"📊 Found {len(instances)} running instances in {self.region}")
            return instances
            
        except Exception as e:
            logger.error(f"❌ Error fetching instances: {e}")
            return []
    
    def get_instance_metrics(self, instance_id: str, period_minutes: int = 5) -> Dict:
        """
        Get CloudWatch metrics for an instance
        Returns: CPU, Network, Disk metrics
        """
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=period_minutes)
        
        metrics_to_fetch = [
            'CPUUtilization',
            'NetworkIn',
            'NetworkOut',
            'DiskReadBytes',
            'DiskWriteBytes'
        ]
        
        metrics_data = {}
        
        try:
            for metric_name in metrics_to_fetch:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=['Average', 'Maximum']
                )
                
                if response['Datapoints']:
                    datapoint = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                    metrics_data[metric_name] = datapoint['Average']
                    metrics_data[f"{metric_name}_Max"] = datapoint.get('Maximum', datapoint['Average'])
                else:
                    metrics_data[metric_name] = 0.0
                    metrics_data[f"{metric_name}_Max"] = 0.0
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching metrics for {instance_id}: {e}")
            return {}
    
    def calculate_carbon_footprint(self, instance_info: Dict, metrics: Dict) -> Dict:
        """Calculate carbon footprint using research methodology"""
        
        instance_type = instance_info['instance_type']
        region = instance_info['region']
        
        base_power_watts = self.INSTANCE_POWER.get(instance_type, 30)
        cpu_util = metrics.get('CPUUtilization', 50) / 100.0
        actual_power_watts = base_power_watts * (0.5 + 0.5 * cpu_util)
        power_kw = actual_power_watts / 1000.0
        time_hours = 5 / 60.0
        energy_kwh = power_kw * time_hours
        carbon_intensity = self.REGION_CARBON_INTENSITY.get(region, 0.0004)
        carbon_kg = energy_kwh * carbon_intensity
        carbon_kg_per_day = carbon_kg * 288
        
        network_gb = (metrics.get('NetworkIn', 0) + metrics.get('NetworkOut', 0)) / (1024**3)
        network_carbon_kg = network_gb * 0.001
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'instance_id': instance_info['instance_id'],
            'instance_name': instance_info['instance_name'],
            'instance_type': instance_type,
            'region': region,
            'availability_zone': instance_info['availability_zone'],
            'cpu_utilization': metrics.get('CPUUtilization', 0),
            'cpu_utilization_max': metrics.get('CPUUtilization_Max', 0),
            'network_in_bytes': metrics.get('NetworkIn', 0),
            'network_out_bytes': metrics.get('NetworkOut', 0),
            'disk_read_bytes': metrics.get('DiskReadBytes', 0),
            'disk_write_bytes': metrics.get('DiskWriteBytes', 0),
            'base_power_watts': base_power_watts,
            'actual_power_watts': actual_power_watts,
            'energy_kwh': energy_kwh,
            'carbon_intensity_kg_per_kwh': carbon_intensity,
            'carbon_emissions_kg': carbon_kg,
            'carbon_emissions_kg_per_day': carbon_kg_per_day,
            'network_carbon_kg': network_carbon_kg,
            'total_carbon_kg': carbon_kg + network_carbon_kg,
        }
    
    def collect_realtime_data(self) -> pd.DataFrame:
        """Main collection method"""
        
        logger.info("🔄 Starting real-time data collection...")
        
        instances = self.get_running_instances()
        
        if not instances:
            logger.warning("⚠️ No running instances found")
            logger.info("💡 To test with real data:")
            logger.info("   1. Launch an EC2 instance in AWS Console")
            logger.info("   2. Wait 5-10 minutes for CloudWatch metrics")
            logger.info("   3. Run this collector again")
            return pd.DataFrame()
        
        carbon_data = []
        
        for instance in instances:
            logger.info(f"📊 Collecting metrics for {instance['instance_id']} ({instance['instance_name']})")
            
            metrics = self.get_instance_metrics(instance['instance_id'])
            
            if metrics:
                carbon_info = self.calculate_carbon_footprint(instance, metrics)
                carbon_data.append(carbon_info)
            else:
                logger.warning(f"⚠️ No metrics available for {instance['instance_id']} yet")
        
        df = pd.DataFrame(carbon_data)
        
        if not df.empty:
            total_carbon = df['total_carbon_kg'].sum()
            daily_carbon = df['carbon_emissions_kg_per_day'].sum()
            
            logger.info(f"✅ Collected data for {len(df)} instances")
            logger.info(f"🌍 Current period carbon: {total_carbon:.6f} kg CO2")
            logger.info(f"📅 Estimated daily carbon: {daily_carbon:.6f} kg CO2/day")
            
            potential_saving = daily_carbon * 0.898
            logger.info(f"💚 Potential daily saving: {potential_saving:.6f} kg CO2 (89.8% optimization)")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Save collected data to CSV"""
        
        if df.empty:
            return
        
        os.makedirs('data/realtime', exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/realtime/aws_carbon_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        logger.info(f"💾 Data saved to {filename}")
        
        return filename


def main():
    """Test the real-time collector"""
    
    print("\n🚀 AWS Real-Time Carbon Collector")
    print("=" * 80)
    
    try:
        collector = AWSRealtimeCollector(region='us-east-1')
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    df = collector.collect_realtime_data()
    
    if not df.empty:
        print("\n📊 Real-Time Carbon Data:")
        print("=" * 80)
        print(df[['instance_name', 'instance_type', 'cpu_utilization', 
                  'actual_power_watts', 'carbon_emissions_kg_per_day']].to_string())
        
        filename = collector.save_data(df)
        print(f"\n💾 Data saved to: {filename}")
        print("\n✅ Ready for ML prediction and optimization!")
        
    else:
        print("\n⚠️ No data collected.")
        print("Launch an EC2 instance and try again in 5-10 minutes.")


if __name__ == "__main__":
    main()
