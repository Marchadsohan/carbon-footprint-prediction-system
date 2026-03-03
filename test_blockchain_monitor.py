"""
Test blockchain-enabled continuous monitoring
Each collection creates a new blockchain block
"""

import sys
sys.path.append('src')

from blockchain.monitor_blockchain_integration import BlockchainEnabledMonitor

print("\n" + "="*60)
print("Testing Blockchain-Enabled Continuous Monitor")
print("This will collect data and create blockchain blocks")
print("="*60)

# Create blockchain-enabled monitor
monitor = BlockchainEnabledMonitor(
    region='us-east-1',
    interval_minutes=5,
    blockchain_difficulty=2
)

# Run single collection test
print("\n[TEST] Running single collection with blockchain...")
monitor.collect_and_analyze()

# Show blockchain status
summary = monitor.blockchain.get_chain_summary()
print(f"\n[BLOCKCHAIN STATUS]")
print(f"   Total Blocks: {summary['total_blocks']}")
print(f"   Total Transactions: {summary['total_transactions']}")
print(f"   Valid: {summary['is_valid']}")
print(f"   Chain Hash: {summary['blockchain_hash'][:20]}...")

# Save blockchain
filename = monitor.blockchain.save_blockchain('data/blockchain/test_live_blockchain.json')
print(f"\n[SAVED] Blockchain saved to: {filename}")

print("\n[SUCCESS] Test complete!")
print("="*60)
