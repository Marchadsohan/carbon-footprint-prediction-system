"""
Integration between Continuous Monitor and Enhanced Blockchain
Automatically records monitoring data as blockchain blocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.blockchain.enhanced_carbon_blockchain import EnhancedCarbonBlockchain
from src.realtime_monitoring.continuous_monitor import ContinuousCarbonMonitor
import logging

logger = logging.getLogger(__name__)


class BlockchainEnabledMonitor(ContinuousCarbonMonitor):
    """
    Continuous Monitor with automatic blockchain recording
    Each monitoring collection becomes a blockchain block
    """
    
    def __init__(self, region='us-east-1', interval_minutes=5, 
                 blockchain_difficulty=2):
        super().__init__(region, interval_minutes)
        
        # Initialize blockchain
        self.blockchain = EnhancedCarbonBlockchain(
            difficulty=blockchain_difficulty,
            enable_distributed=True
        )
        
        logger.info(f"[BLOCKCHAIN] Blockchain integration enabled")
        logger.info(f"   Difficulty: {blockchain_difficulty}")
    
    def collect_and_analyze(self):
        """Override to add blockchain recording"""
        
        # Call parent method
        super().collect_and_analyze()
        
        # Record to blockchain if data was collected
        if self.data_buffer:
            latest_data = self.data_buffer[-1]
            
            # Add to blockchain
            self.blockchain.new_carbon_transaction(
                organization="Carbon-Footprint-Research",
                instance_id="monitoring",
                instance_type="continuous",
                region=self.region,
                baseline_carbon=latest_data['daily_carbon_kg'],
                optimized_carbon=latest_data['daily_carbon_kg'] * 0.102,
                carbon_saving=latest_data['potential_saving_kg'],
                strategy="continuous_monitoring",
                cpu_utilization=latest_data['avg_cpu_utilization'],
                timestamp=latest_data['timestamp']
            )
            
            # Mine block
            block = self.blockchain.mine_block()
            
            if block:
                logger.info(f"[BLOCKCHAIN] Block #{block['index']} mined")
                logger.info(f"   Hash: {self.blockchain.hash(block)[:16]}...")
    
    def stop_monitoring(self):
        """Override to save blockchain"""
        
        # Call parent method
        super().stop_monitoring()
        
        # Save blockchain
        filename = self.blockchain.save_blockchain()
        logger.info(f"[BLOCKCHAIN] Final blockchain saved: {filename}")
        
        # Show blockchain summary
        summary = self.blockchain.get_chain_summary()
        logger.info(f"\n[BLOCKCHAIN SUMMARY]:")
        logger.info(f"   Total Blocks: {summary['total_blocks']}")
        logger.info(f"   Valid: {summary['is_valid']}")


def main():
    """Run blockchain-enabled monitoring"""
    
    monitor = BlockchainEnabledMonitor(
        region='us-east-1',
        interval_minutes=5,
        blockchain_difficulty=2
    )
    
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
