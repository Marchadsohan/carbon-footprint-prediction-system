"""
ML-Enabled Continuous Monitor
Integrates real-time monitoring with ML predictions and blockchain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.blockchain.monitor_blockchain_integration import BlockchainEnabledMonitor
from src.models.model_integration import CarbonPredictionEngine
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MLBlockchainMonitor(BlockchainEnabledMonitor):
    """
    Complete monitoring system with:
    - Real-time AWS data collection
    - ML-based predictions (LSTM + XGBoost)
    - Blockchain recording
    - Automated optimization recommendations
    """
    
    def __init__(self, region='us-east-1', interval_minutes=5, 
                 blockchain_difficulty=2, lstm_model_path=None, xgboost_model_path=None):
        super().__init__(region, interval_minutes, blockchain_difficulty)
        
        # Initialize ML engine
        self.ml_engine = CarbonPredictionEngine(
            lstm_model_path=lstm_model_path,
            xgboost_model_path=xgboost_model_path
        )
        
        self.prediction_history = []
        
        logger.info("[ML] ML integration enabled")
    
    def collect_and_analyze(self):
        """Override to add ML predictions"""
        
        # Call parent (blockchain-enabled monitoring)
        super().collect_and_analyze()
        
        # Generate ML predictions if we have data
        if self.data_buffer and len(self.data_buffer) >= 3:
            
            # Convert buffer to DataFrame
            df_buffer = pd.DataFrame(self.data_buffer)
            
            # Create monitoring DataFrame format
            monitoring_df = pd.DataFrame({
                'timestamp': df_buffer['timestamp'],
                'instance_id': ['monitoring'] * len(df_buffer),
                'instance_type': ['continuous'] * len(df_buffer),
                'region': [self.region] * len(df_buffer),
                'cpu_utilization': df_buffer['avg_cpu_utilization'],
                'cpu_utilization_max': df_buffer['avg_cpu_utilization'] * 1.5,
                'network_in_bytes': [1000000] * len(df_buffer),
                'network_out_bytes': [500000] * len(df_buffer),
                'actual_power_watts': [2.5] * len(df_buffer),
                'carbon_intensity_kg_per_kwh': [0.000379] * len(df_buffer),
                'carbon_emissions_kg': df_buffer['current_carbon_kg'],
                'carbon_emissions_kg_per_day': df_buffer['daily_carbon_kg']
            })
            
            # Generate predictions
            report = self.ml_engine.generate_prediction_report(monitoring_df)
            self.prediction_history.append(report)
            
            # Log predictions
            logger.info(f"\n[ML-PREDICTION] Forecast Summary:")
            logger.info(f"   24h Predicted Carbon: {report['lstm_prediction']['total_predicted_kg']:.6f} kg")
            logger.info(f"   Model Confidence: {report['lstm_prediction']['confidence']*100:.0f}%")
            logger.info(f"   Top Recommendation: {report['recommendations'][0]['strategy']}")
            logger.info(f"   Potential Saving: {report['recommendations'][0]['estimated_saving_kg']:.6f} kg CO2/day")
    
    def stop_monitoring(self):
        """Override to save ML predictions"""
        
        # Call parent
        super().stop_monitoring()
        
        # Save prediction history
        if self.prediction_history:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/predictions/predictions_{timestamp}.json'
            os.makedirs('data/predictions', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)
            
            logger.info(f"[ML] Predictions saved to: {filename}")


def main():
    """Run ML-enabled monitoring"""
    
    monitor = MLBlockchainMonitor(
        region='us-east-1',
        interval_minutes=5,
        blockchain_difficulty=2
    )
    
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
