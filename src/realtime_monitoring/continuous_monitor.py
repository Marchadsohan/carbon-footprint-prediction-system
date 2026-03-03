"""
Continuous Carbon Monitoring Service
Runs 24/7, collecting data every 5 minutes and triggering optimizations
"""

import time
import schedule
import logging
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.cloud_connectors.aws_realtime_collector import AWSRealtimeCollector

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinuousCarbonMonitor:
    """
    24/7 Carbon Monitoring Service
    - Collects real-time data every N minutes
    - Predicts carbon emissions using ML models
    - Triggers optimization when thresholds exceeded
    - Stores data for blockchain verification
    """
    
    def __init__(self, region='us-east-1', interval_minutes=5):
        self.region = region
        self.interval_minutes = interval_minutes
        self.collector = AWSRealtimeCollector(region=region)
        
        # Monitoring state
        self.monitoring_active = False
        self.total_collections = 0
        self.total_carbon_measured = 0.0
        self.total_potential_savings = 0.0
        
        # Data buffer for blockchain
        self.data_buffer = []
        
        logger.info(f"[INIT] Continuous Carbon Monitor initialized")
        logger.info(f"   Region: {region}")
        logger.info(f"   Interval: Every {interval_minutes} minutes")
    
    def collect_and_analyze(self):
        """Main monitoring cycle"""
        
        try:
            self.total_collections += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[COLLECT] Collection #{self.total_collections} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # 1. Collect real-time data
            df = self.collector.collect_realtime_data()
            
            if df.empty:
                logger.warning("[WARN] No data collected in this cycle")
                logger.info("[INFO] Launch an EC2 instance to see real monitoring")
                return
            
            # 2. Calculate metrics
            current_carbon = df['total_carbon_kg'].sum()
            daily_carbon = df['carbon_emissions_kg_per_day'].sum()
            avg_cpu = df['cpu_utilization'].mean()
            
            self.total_carbon_measured += current_carbon
            
            logger.info(f"\n[METRICS] Current Metrics:")
            logger.info(f"   Instances monitored: {len(df)}")
            logger.info(f"   Average CPU: {avg_cpu:.1f}%")
            logger.info(f"   Current carbon: {current_carbon:.6f} kg CO2")
            logger.info(f"   Daily projection: {daily_carbon:.6f} kg CO2/day")
            
            # 3. Calculate optimization potential (based on your 89.8% max reduction)
            potential_daily_saving = daily_carbon * 0.898  # Max from your research
            potential_cost_saving = potential_daily_saving * 6.70  # $6.70/kg CO2
            
            self.total_potential_savings += potential_daily_saving
            
            logger.info(f"\n[OPTIMIZATION] Optimization Potential:")
            logger.info(f"   Daily carbon saving: {potential_daily_saving:.6f} kg CO2")
            logger.info(f"   Daily cost saving: ${potential_cost_saving:.2f}")
            logger.info(f"   Annual saving: ${potential_cost_saving * 365:.2f}")
            
            # 4. Check if optimization needed
            if daily_carbon > 0.10:  # Threshold: 100g CO2/day
                logger.warning(f"\n[ALERT] HIGH CARBON ALERT!")
                logger.info(f"   Daily carbon ({daily_carbon:.3f} kg) exceeds threshold (0.100 kg)")
                self.trigger_optimization(df)
            else:
                logger.info(f"\n[STATUS] Carbon levels within acceptable range")
            
            # 5. Save to buffer for blockchain
            self.data_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'collection_number': self.total_collections,
                'num_instances': len(df),
                'current_carbon_kg': current_carbon,
                'daily_carbon_kg': daily_carbon,
                'potential_saving_kg': potential_daily_saving,
                'avg_cpu_utilization': avg_cpu
            })
            
            # 6. Save detailed data
            os.makedirs('data/realtime', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/realtime/monitoring_{timestamp}.csv'
            df.to_csv(filename, index=False)
            logger.info(f"\n[SAVE] Data saved to: {filename}")
            
            # 7. Summary
            logger.info(f"\n[SUMMARY] Session Statistics:")
            logger.info(f"   Total collections: {self.total_collections}")
            logger.info(f"   Total carbon measured: {self.total_carbon_measured:.6f} kg")
            logger.info(f"   Total potential savings: {self.total_potential_savings:.6f} kg")
            
        except Exception as e:
            logger.error(f"[ERROR] Error in collection cycle: {e}", exc_info=True)
    
    def trigger_optimization(self, df: pd.DataFrame):
        """Trigger optimization strategies"""
        
        logger.info("\n[OPTIMIZE] Triggering Carbon Optimization Analysis...")
        logger.info("="*60)
        
        # Your 6 optimization strategies from research
        strategies = [
            {
                'name': 'Temporal Shift to Off-Peak Hours',
                'reduction': 35.8,
                'description': 'Move workloads to low-carbon time periods'
            },
            {
                'name': 'Geographic Migration to Low-Carbon Regions',
                'reduction': 73.5,
                'description': 'Migrate to regions with renewable energy'
            },
            {
                'name': 'Resource Right-Sizing',
                'reduction': 24.1,
                'description': 'Optimize instance sizes for workload'
            },
            {
                'name': 'Code Optimization',
                'reduction': 18.2,
                'description': 'Improve application efficiency'
            },
            {
                'name': 'Dynamic Reliability Adjustment',
                'reduction': 29.8,
                'description': 'Adjust redundancy based on requirements'
            },
            {
                'name': 'Peak-Time Orchestration',
                'reduction': 42.6,
                'description': 'Intelligent workload scheduling'
            }
        ]
        
        daily_carbon = df['carbon_emissions_kg_per_day'].sum()
        
        logger.info(f"Baseline Daily Carbon: {daily_carbon:.6f} kg CO2/day\n")
        
        for i, strategy in enumerate(strategies, 1):
            optimized = daily_carbon * (1 - strategy['reduction']/100)
            saving = daily_carbon - optimized
            cost_saving = saving * 6.70 * 365  # Annual
            
            logger.info(f"[STRATEGY {i}] {strategy['name']}")
            logger.info(f"  Description: {strategy['description']}")
            logger.info(f"  Reduction: {strategy['reduction']}%")
            logger.info(f"  Optimized: {optimized:.6f} kg CO2/day")
            logger.info(f"  Saving: {saving:.6f} kg CO2/day")
            logger.info(f"  Annual Cost Saving: ${cost_saving:.2f}\n")
        
        # Combined optimization (89.8% - your max reduction)
        combined_optimized = daily_carbon * 0.102
        combined_saving = daily_carbon - combined_optimized
        combined_cost = combined_saving * 6.70 * 365
        
        logger.info(f"[COMBINED] COMBINED OPTIMIZATION (All Strategies):")
        logger.info(f"   Total Reduction: 89.8%")
        logger.info(f"   Optimized Carbon: {combined_optimized:.6f} kg CO2/day")
        logger.info(f"   Total Saving: {combined_saving:.6f} kg CO2/day")
        logger.info(f"   Annual Cost Saving: ${combined_cost:.2f}")
        logger.info("="*60)
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        
        print("\n" + "="*60)
        print("[START] CARBON FOOTPRINT CONTINUOUS MONITORING SERVICE")
        print("="*60)
        print(f"Region: {self.region}")
        print(f"Interval: Every {self.interval_minutes} minutes")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("\nPress Ctrl+C to stop monitoring\n")
        
        self.monitoring_active = True
        
        # Schedule collection
        schedule.every(self.interval_minutes).minutes.do(self.collect_and_analyze)
        
        # Run first collection immediately
        self.collect_and_analyze()
        
        # Continuous loop
        try:
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring service"""
        
        logger.info("\n" + "="*60)
        logger.info("[STOP] Stopping Carbon Monitoring Service...")
        logger.info("="*60)
        
        self.monitoring_active = False
        
        logger.info(f"\n[FINAL] FINAL SESSION STATISTICS:")
        logger.info(f"   Total collections: {self.total_collections}")
        logger.info(f"   Total carbon measured: {self.total_carbon_measured:.6f} kg CO2")
        logger.info(f"   Total potential savings: {self.total_potential_savings:.6f} kg CO2")
        logger.info(f"   Session duration: {self.total_collections * self.interval_minutes} minutes")
        
        # Save session summary
        if self.data_buffer:
            summary_df = pd.DataFrame(self.data_buffer)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = f'data/realtime/session_summary_{timestamp}.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"\n[SAVE] Session summary saved to: {summary_file}")
        
        logger.info("\n[DONE] Monitoring service stopped successfully")
    
    def get_buffer_for_blockchain(self):
        """Get collected data for blockchain recording"""
        return self.data_buffer


def main():
    """Run continuous monitoring"""
    
    # Create monitor instance
    monitor = ContinuousCarbonMonitor(
        region='us-east-1',
        interval_minutes=5  # Collect every 5 minutes
    )
    
    # Start monitoring (will run until Ctrl+C)
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
