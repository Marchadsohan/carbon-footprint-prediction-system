"""
Enhanced Carbon Blockchain with Time-Series Monitoring
Scales to 1,000+ blocks with distributed validation
Integrates with continuous monitoring system
"""

import hashlib
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/blockchain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedCarbonBlockchain:
    """
    Production-grade blockchain for carbon footprint tracking
    Features:
    - Time-series carbon monitoring
    - Distributed validation simulation
    - Scales to 1,000+ blocks
    - Configurable proof-of-work difficulty
    """
    
    def __init__(self, difficulty=2, enable_distributed=True):
        self.chain = []
        self.current_transactions = []
        self.carbon_credits = {}
        self.difficulty = difficulty
        self.enable_distributed = enable_distributed
        self.validator_nodes = {}
        
        # Performance metrics
        self.total_validations = 0
        self.total_mining_time = 0
        
        # Create genesis block
        self.new_block(previous_hash='0', proof=100)
        
        logger.info(f"[INIT] Enhanced Carbon Blockchain initialized")
        logger.info(f"   Difficulty: {difficulty} (PoW leading zeros)")
        logger.info(f"   Distributed validation: {enable_distributed}")
        logger.info(f"   Genesis block created")
    
    def new_block(self, proof: int, previous_hash: Optional[str] = None, 
                  validators: List[str] = None) -> Dict:
        """Create a new block in the blockchain"""
        
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
            'validators': validators or [],
            'consensus_achieved': len(validators or []) >= 3 if validators else True
        }
        
        # Reset current transactions
        self.current_transactions = []
        self.chain.append(block)
        
        return block
    
    def new_carbon_transaction(self, organization: str, instance_id: str,
                              instance_type: str, region: str,
                              baseline_carbon: float, optimized_carbon: float,
                              carbon_saving: float, strategy: str = "continuous_monitoring",
                              cpu_utilization: float = 0.0,
                              timestamp: str = None) -> int:
        """
        Add carbon monitoring transaction
        """
        
        # Calculate carbon credits (1 credit per 0.01 kg CO2 saved)
        credits_earned = int(carbon_saving / 0.01) if carbon_saving > 0 else 0
        
        transaction = {
            'organization': organization,
            'timestamp': timestamp or datetime.now().isoformat(),
            'instance_id': instance_id,
            'instance_type': instance_type,
            'region': region,
            'baseline_carbon_kg': baseline_carbon,
            'optimized_carbon_kg': optimized_carbon,
            'carbon_saving_kg': carbon_saving,
            'reduction_percentage': ((baseline_carbon - optimized_carbon) / baseline_carbon * 100) if baseline_carbon > 0 else 0,
            'strategy': strategy,
            'cpu_utilization': cpu_utilization,
            'credits_earned': credits_earned,
            'transaction_type': 'carbon_monitoring',
            'verified': True
        }
        
        self.current_transactions.append(transaction)
        
        # Update carbon credits
        if organization not in self.carbon_credits:
            self.carbon_credits[organization] = 0
        self.carbon_credits[organization] += credits_earned
        
        return len(self.chain) + 1
    
    def mine_block(self, miner_address: str = "carbon_monitor_miner") -> Optional[Dict]:
        """
        Mine a new block with proof of work and distributed validation
        """
        
        if not self.current_transactions:
            return None
        
        start_time = time.time()
        
        last_block = self.last_block
        last_proof = last_block['proof']
        
        # Proof of work
        proof = self.proof_of_work(last_proof)
        
        # Distributed validation
        validators = self.simulate_consensus() if self.enable_distributed else []
        
        # Create new block
        previous_hash = self.hash(last_block)
        block = self.new_block(proof, previous_hash, validators)
        
        # Track performance
        mining_time = time.time() - start_time
        self.total_mining_time += mining_time
        self.total_validations += len(validators)
        
        return block
    
    def proof_of_work(self, last_proof: int) -> int:
        """
        Proof of Work Algorithm
        Find a number that produces a hash with N leading zeros
        """
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof
    
    def valid_proof(self, last_proof: int, proof: int) -> bool:
        """Validate proof with configurable difficulty"""
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:self.difficulty] == "0" * self.difficulty
    
    def simulate_consensus(self, num_validators: int = 5) -> List[str]:
        """
        Simulate distributed consensus validation
        Returns list of validator nodes that approved
        """
        validator_ids = [f"validator_node_{i}" for i in range(num_validators)]
        
        # 95% approval rate (Byzantine fault tolerance)
        approving_validators = [
            vid for vid in validator_ids 
            if np.random.random() > 0.05
        ]
        
        return approving_validators
    
    @staticmethod
    def hash(block: Dict) -> str:
        """Create SHA-256 hash of a block"""
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self) -> Dict:
        """Return the last block in the chain"""
        return self.chain[-1]
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            last_block = self.chain[i-1]
            
            # Check hash linkage
            if current_block['previous_hash'] != self.hash(last_block):
                logger.error(f"[ERROR] Invalid hash at block {i}")
                return False
            
            # Check proof of work
            if not self.valid_proof(last_block['proof'], current_block['proof']):
                logger.error(f"[ERROR] Invalid proof at block {i}")
                return False
        
        return True
    
    def get_chain_summary(self) -> Dict:
        """Get comprehensive blockchain statistics"""
        
        total_blocks = len(self.chain)
        total_transactions = sum(len(block['transactions']) for block in self.chain)
        total_credits = sum(self.carbon_credits.values())
        
        # Calculate total carbon savings
        total_savings = 0
        for block in self.chain:
            for tx in block['transactions']:
                if tx.get('transaction_type') == 'carbon_monitoring':
                    total_savings += tx.get('carbon_saving_kg', 0)
        
        # Average mining time
        avg_mining_time = self.total_mining_time / max(total_blocks - 1, 1)
        
        return {
            'total_blocks': total_blocks,
            'total_transactions': total_transactions,
            'total_carbon_credits': total_credits,
            'total_co2_saved_kg': total_savings,
            'blockchain_hash': self.hash(self.last_block),
            'organizations': len(self.carbon_credits),
            'is_valid': self.validate_chain(),
            'avg_mining_time_sec': avg_mining_time,
            'total_validators': self.total_validations,
            'difficulty': self.difficulty
        }
    
    def get_all_transactions(self) -> List[Dict]:
        """Get all transactions from blockchain"""
        
        all_transactions = []
        
        for block in self.chain:
            for tx in block['transactions']:
                tx_copy = tx.copy()
                tx_copy['block_index'] = block['index']
                tx_copy['block_hash'] = self.hash(block)[:16]
                tx_copy['block_timestamp'] = block['datetime']
                all_transactions.append(tx_copy)
        
        return all_transactions
    
    def get_transactions_dataframe(self) -> pd.DataFrame:
        """Convert blockchain transactions to pandas DataFrame"""
        
        transactions = self.get_all_transactions()
        if not transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(transactions)
        return df
    
    def save_blockchain(self, filename: str = None):
        """Save blockchain to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/blockchain/blockchain_{timestamp}.json'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        blockchain_data = {
            'chain': self.chain,
            'carbon_credits': self.carbon_credits,
            'summary': self.get_chain_summary(),
            'created': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(blockchain_data, f, indent=2)
        
        logger.info(f"[SAVE] Blockchain saved to: {filename}")
        return filename
    
    def load_blockchain(self, filename: str):
        """Load blockchain from JSON file"""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.chain = data['chain']
        self.carbon_credits = data['carbon_credits']
        
        logger.info(f"[LOAD] Blockchain loaded from: {filename}")
        logger.info(f"   Blocks: {len(self.chain)}")


def generate_timeseries_blockchain(days: int = 42, 
                                   measurements_per_day: int = 24,
                                   difficulty: int = 2) -> EnhancedCarbonBlockchain:
    """
    Generate blockchain with time-series carbon monitoring data
    
    Args:
        days: Number of days to simulate (default: 42 days = 1,008 blocks)
        measurements_per_day: Measurements per day (default: 24 hourly)
        difficulty: PoW difficulty (default: 2 leading zeros)
    
    Returns:
        EnhancedCarbonBlockchain with generated data
    """
    
    total_blocks = days * measurements_per_day
    
    logger.info(f"\n[GENERATE] Creating Time-Series Carbon Blockchain")
    logger.info(f"{'='*60}")
    logger.info(f"Duration: {days} days")
    logger.info(f"Measurements/day: {measurements_per_day}")
    logger.info(f"Total blocks: {total_blocks}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"{'='*60}\n")
    
    # Initialize blockchain
    blockchain = EnhancedCarbonBlockchain(difficulty=difficulty, enable_distributed=True)
    
    # Simulation parameters
    start_date = datetime.now() - timedelta(days=days)
    
    organizations = [
        "AWS_US_East_1", "AWS_US_West_2", "AWS_EU_West_1",
        "AWS_AP_Southeast_1", "AWS_CA_Central_1"
    ]
    
    instance_types = ['t2.micro', 't2.small', 't3.medium', 'm5.large']
    
    strategies = [
        "Temporal_Shift_Off_Peak",
        "Geographic_Migration_Low_Carbon",
        "Resource_Right_Sizing",
        "Code_Optimization",
        "Dynamic_Reliability_Adjustment",
        "Peak_Time_Orchestration"
    ]
    
    strategy_effectiveness = {
        "Temporal_Shift_Off_Peak": 0.358,
        "Geographic_Migration_Low_Carbon": 0.735,
        "Resource_Right_Sizing": 0.241,
        "Code_Optimization": 0.182,
        "Dynamic_Reliability_Adjustment": 0.298,
        "Peak_Time_Orchestration": 0.426
    }
    
    # Generate blocks
    for hour in range(total_blocks):
        current_time = start_date + timedelta(hours=hour)
        
        # Simulate realistic hourly patterns
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        # Higher emissions during business hours
        time_multiplier = 1.0
        if 9 <= hour_of_day <= 17 and day_of_week < 5:
            time_multiplier = 1.3
        elif 0 <= hour_of_day <= 6:
            time_multiplier = 0.7
        
        # Select random parameters
        org = organizations[hour % len(organizations)]
        instance_type = instance_types[hour % len(instance_types)]
        strategy = strategies[hour % len(strategies)]
        
        # Calculate carbon
        base_emission = 0.00156 * time_multiplier + np.random.normal(0, 0.0001)
        base_emission = max(base_emission, 0.0001)  # Ensure positive
        
        effectiveness = strategy_effectiveness[strategy]
        optimized = base_emission * (1 - effectiveness)
        carbon_saving = (base_emission - optimized) * 24  # Daily
        
        cpu_util = np.random.uniform(10, 80)
        
        # Add transaction
        blockchain.new_carbon_transaction(
            organization=org,
            instance_id=f"i-{hour:04d}",
            instance_type=instance_type,
            region="us-east-1",
            baseline_carbon=base_emission,
            optimized_carbon=optimized,
            carbon_saving=carbon_saving,
            strategy=strategy,
            cpu_utilization=cpu_util,
            timestamp=current_time.isoformat()
        )
        
        # Mine block
        block = blockchain.mine_block()
        
        # Progress reporting
        if (hour + 1) % 100 == 0:
            progress = (hour + 1) / total_blocks * 100
            logger.info(f"[PROGRESS] {hour + 1}/{total_blocks} blocks ({progress:.1f}%)")
    
    # Final validation
    is_valid = blockchain.validate_chain()
    
    logger.info(f"\n[COMPLETE] Blockchain Generation Complete")
    logger.info(f"{'='*60}")
    
    summary = blockchain.get_chain_summary()
    logger.info(f"[SUMMARY] Blockchain Statistics:")
    logger.info(f"   Total Blocks: {summary['total_blocks']}")
    logger.info(f"   Total Transactions: {summary['total_transactions']}")
    logger.info(f"   Total CO2 Saved: {summary['total_co2_saved_kg']:.3f} kg")
    logger.info(f"   Carbon Credits: {summary['total_carbon_credits']}")
    logger.info(f"   Organizations: {summary['organizations']}")
    logger.info(f"   Validation: {'VALID' if is_valid else 'INVALID'}")
    logger.info(f"   Avg Mining Time: {summary['avg_mining_time_sec']:.4f} sec")
    logger.info(f"   Chain Hash: {summary['blockchain_hash'][:20]}...")
    logger.info(f"{'='*60}\n")
    
    return blockchain


def main():
    """Test enhanced blockchain"""
    
    print("\n[TEST] Enhanced Carbon Blockchain System")
    print("="*60)
    
    # Generate 1,008 block blockchain (42 days x 24 hours)
    blockchain = generate_timeseries_blockchain(
        days=42,
        measurements_per_day=24,
        difficulty=2
    )
    
    # Save blockchain
    filename = blockchain.save_blockchain()
    
    # Export transactions to CSV
    df = blockchain.get_transactions_dataframe()
    if not df.empty:
        csv_file = filename.replace('.json', '_transactions.csv')
        df.to_csv(csv_file, index=False)
        print(f"[EXPORT] Transactions exported to: {csv_file}")
    
    print("\n[SUCCESS] Enhanced blockchain created successfully!")


if __name__ == "__main__":
    main()
