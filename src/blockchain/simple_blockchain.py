import hashlib
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class SimpleCarbonBlockchain:
    """
    Simplified blockchain implementation for carbon footprint verification
    Demonstrates blockchain concepts without requiring complex setup
    """
    
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.carbon_credits = {}
        
        # Create genesis block
        self.new_block(previous_hash='0', proof=100)
        
        print("🔗 Simple Carbon Blockchain initialized")
        
    def new_block(self, proof, previous_hash=None):
        """Create a new Block in the Blockchain"""
        
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        
        # Reset the current list of transactions
        self.current_transactions = []
        
        self.chain.append(block)
        return block
    
    def new_carbon_transaction(self, organization, predicted_emissions, actual_emissions, 
                              optimization_savings, model_name="TCEP"):
        """
        Add a new carbon footprint transaction to the list of transactions
        """
        
        # Calculate carbon credits earned (1 credit per 0.1 kg CO2 saved)
        credits_earned = int(optimization_savings / 0.01) if optimization_savings > 0 else 0
        
        transaction = {
            'organization': organization,
            'timestamp': datetime.now().isoformat(),
            'predicted_emissions': predicted_emissions,
            'actual_emissions': actual_emissions,
            'optimization_savings': optimization_savings,
            'model_name': model_name,
            'credits_earned': credits_earned,
            'transaction_type': 'carbon_record',
            'verified': True  # Auto-verified for demo
        }
        
        # Add to pending transactions
        self.current_transactions.append(transaction)
        
        # Update carbon credits balance
        if organization not in self.carbon_credits:
            self.carbon_credits[organization] = 0
        self.carbon_credits[organization] += credits_earned
        
        print(f"✅ Carbon transaction added: {optimization_savings:.3f} kg CO2 saved → {credits_earned} credits")
        
        return len(self.chain) + 1
    
    def mine_block(self, miner_address):
        """
        Mine a new block (simplified proof of work)
        """
        
        if not self.current_transactions:
            return None
            
        # Simple proof of work (find a number that makes hash start with zeros)
        last_block = self.last_block
        last_proof = last_block['proof']
        
        proof = self.proof_of_work(last_proof)
        
        # Reward the miner with carbon credits
        self.new_carbon_transaction(
            organization=miner_address,
            predicted_emissions=0,
            actual_emissions=0,
            optimization_savings=0.1,  # Mining reward
            model_name="Mining_Reward"
        )
        
        # Create the new Block
        previous_hash = self.hash(last_block)
        block = self.new_block(proof, previous_hash)
        
        print(f"⛏️ Block {block['index']} mined successfully!")
        return block
    
    def proof_of_work(self, last_proof):
        """
        Simple Proof of Work Algorithm:
        Find a number that, when hashed with the previous proof, contains 4 leading zeroes
        """
        
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof):
        """
        Validates the Proof: Does hash(last_proof, proof) contain 4 leading zeroes?
        """
        
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    
    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        """
        
        # Make sure the Dictionary is Ordered, or hashes will be inconsistent
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    def get_chain_summary(self):
        """Get blockchain summary statistics"""
        
        total_blocks = len(self.chain)
        total_transactions = sum(len(block['transactions']) for block in self.chain)
        total_credits = sum(self.carbon_credits.values())
        
        # Calculate total CO2 savings verified on blockchain
        total_savings = 0
        for block in self.chain:
            for tx in block['transactions']:
                if tx.get('transaction_type') == 'carbon_record':
                    total_savings += tx.get('optimization_savings', 0)
        
        return {
            'total_blocks': total_blocks,
            'total_transactions': total_transactions,
            'total_carbon_credits': total_credits,
            'total_co2_saved': total_savings,
            'blockchain_hash': self.hash(self.last_block),
            'organizations': len(self.carbon_credits)
        }
    
    def get_organization_summary(self, organization):
        """Get carbon footprint summary for a specific organization"""
        
        org_transactions = []
        total_savings = 0
        
        for block in self.chain:
            for tx in block['transactions']:
                if tx.get('organization') == organization and tx.get('transaction_type') == 'carbon_record':
                    org_transactions.append(tx)
                    total_savings += tx.get('optimization_savings', 0)
        
        return {
            'organization': organization,
            'total_transactions': len(org_transactions),
            'total_co2_saved': total_savings,
            'carbon_credits_balance': self.carbon_credits.get(organization, 0),
            'transactions': org_transactions[-5:]  # Last 5 transactions
        }
    
    def validate_chain(self):
        """
        Validate the entire blockchain
        """
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            last_block = self.chain[i-1]
            
            # Check if the current block's hash is correct
            if current_block['previous_hash'] != self.hash(last_block):
                return False
                
            # Check if the Proof of Work is correct
            if not self.valid_proof(last_block['proof'], current_block['proof']):
                return False
        
        return True

def create_demo_blockchain_data():
    """Create demonstration blockchain with carbon footprint data"""
    
    print("🔗 Creating Demo Carbon Blockchain")
    print("=" * 50)
    
    # Initialize blockchain
    blockchain = SimpleCarbonBlockchain()
    
    # Demo organizations
    organizations = [
        "TechCorp_Inc", 
        "GreenData_Ltd", 
        "CloudServe_Co", 
        "EcoSystem_LLC",
        "DataCenter_Pro"
    ]
    
    # Simulate carbon footprint transactions
    demo_data = [
        {"org": "TechCorp_Inc", "predicted": 0.156, "actual": 0.134, "savings": 0.045},
        {"org": "GreenData_Ltd", "predicted": 0.203, "actual": 0.167, "savings": 0.067},
        {"org": "CloudServe_Co", "predicted": 0.088, "actual": 0.076, "savings": 0.023},
        {"org": "EcoSystem_LLC", "predicted": 0.134, "actual": 0.112, "savings": 0.031},
        {"org": "DataCenter_Pro", "predicted": 0.167, "actual": 0.141, "savings": 0.052},
        {"org": "TechCorp_Inc", "predicted": 0.145, "actual": 0.121, "savings": 0.038},
        {"org": "GreenData_Ltd", "predicted": 0.189, "actual": 0.158, "savings": 0.055},
    ]
    
    # Add transactions
    for data in demo_data:
        blockchain.new_carbon_transaction(
            organization=data["org"],
            predicted_emissions=data["predicted"],
            actual_emissions=data["actual"],
            optimization_savings=data["savings"],
            model_name="TCEP-v1.0"
        )
    
    # Mine blocks
    miner = "CarbonMiner_001"
    for i in range(3):  # Mine 3 blocks
        blockchain.mine_block(miner)
        print(f"📦 Block {blockchain.last_block['index']} added to chain")
    
    # Validate blockchain
    is_valid = blockchain.validate_chain()
    print(f"🔐 Blockchain validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    return blockchain

def main():
    """Test the simple blockchain"""
    
    # Create demo blockchain
    blockchain = create_demo_blockchain_data()
    
    # Show summary
    summary = blockchain.get_chain_summary()
    
    print(f"\n📊 BLOCKCHAIN SUMMARY")
    print("=" * 30)
    print(f"📦 Total Blocks: {summary['total_blocks']}")
    print(f"📝 Total Transactions: {summary['total_transactions']}")
    print(f"💰 Total Carbon Credits: {summary['total_carbon_credits']}")
    print(f"🌍 Total CO2 Saved: {summary['total_co2_saved']:.3f} kg")
    print(f"🏢 Organizations: {summary['organizations']}")
    print(f"🔐 Chain Hash: {summary['blockchain_hash'][:16]}...")
    
    # Show organization summary
    org_summary = blockchain.get_organization_summary("TechCorp_Inc")
    print(f"\n🏢 ORGANIZATION SUMMARY: {org_summary['organization']}")
    print("=" * 40)
    print(f"📝 Transactions: {org_summary['total_transactions']}")
    print(f"🌍 CO2 Saved: {org_summary['total_co2_saved']:.3f} kg")
    print(f"💰 Carbon Credits: {org_summary['carbon_credits_balance']}")
    
    return blockchain

if __name__ == "__main__":
    main()
