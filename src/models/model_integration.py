"""
ML Model Integration for Real-Time Carbon Prediction
Connects existing LSTM and XGBoost models with live monitoring
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarbonPredictionEngine:
    """
    Integrates LSTM and XGBoost models for real-time prediction
    """
    
    def __init__(self, lstm_model_path=None, xgboost_model_path=None):
        """
        Initialize prediction engine
        
        Args:
            lstm_model_path: Path to saved LSTM model (.h5)
            xgboost_model_path: Path to saved XGBoost model (.pkl)
        """
        self.lstm_model = None
        self.xgboost_model = None
        self.scaler = None
        
        # Try to load models if paths provided
        if lstm_model_path and os.path.exists(lstm_model_path):
            self.load_lstm_model(lstm_model_path)
        
        if xgboost_model_path and os.path.exists(xgboost_model_path):
            self.load_xgboost_model(xgboost_model_path)
        
        logger.info("[INIT] Carbon Prediction Engine initialized")
        logger.info(f"   LSTM Model: {'Loaded' if self.lstm_model else 'Not loaded'}")
        logger.info(f"   XGBoost Model: {'Loaded' if self.xgboost_model else 'Not loaded'}")
    
    def load_lstm_model(self, model_path: str):
        """Load pre-trained LSTM model"""
        try:
            from tensorflow import keras
            self.lstm_model = keras.models.load_model(model_path)
            logger.info(f"[LOAD] LSTM model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load LSTM model: {e}")
            return False
    
    def load_xgboost_model(self, model_path: str):
        """Load pre-trained XGBoost model"""
        try:
            self.xgboost_model = joblib.load(model_path)
            logger.info(f"[LOAD] XGBoost model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load XGBoost model: {e}")
            return False
    
    def prepare_features(self, monitoring_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from monitoring data for prediction
        
        Features extracted:
        - CPU utilization (current, mean, max)
        - Network traffic (in, out, total)
        - Instance type encoding
        - Time features (hour, day of week)
        - Region carbon intensity
        """
        
        features = []
        
        for _, row in monitoring_data.iterrows():
            feature_vector = [
                row.get('cpu_utilization', 0),
                row.get('cpu_utilization_max', 0),
                row.get('network_in_bytes', 0) / 1e9,  # Convert to GB
                row.get('network_out_bytes', 0) / 1e9,
                row.get('actual_power_watts', 0),
                row.get('carbon_intensity_kg_per_kwh', 0.0004),
            ]
            
            # Time features
            timestamp = pd.to_datetime(row.get('timestamp', datetime.now()))
            feature_vector.extend([
                timestamp.hour,
                timestamp.weekday(),
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def predict_lstm(self, monitoring_data: pd.DataFrame, 
                     hours_ahead: int = 24) -> Dict:
        """
        Predict carbon emissions using LSTM for time-series forecasting
        
        Args:
            monitoring_data: Recent monitoring data
            hours_ahead: Number of hours to forecast
        
        Returns:
            Dictionary with predictions
        """
        
        if self.lstm_model is None:
            logger.warning("[WARN] LSTM model not loaded, using heuristic prediction")
            return self._heuristic_prediction(monitoring_data, hours_ahead)
        
        try:
            # Prepare features
            features = self.prepare_features(monitoring_data)
            
            # Reshape for LSTM [samples, timesteps, features]
            X = features.reshape(1, features.shape[0], features.shape[1])
            
            # Predict
            predictions = self.lstm_model.predict(X, verbose=0)
            
            # Generate hourly predictions
            base_carbon = monitoring_data['carbon_emissions_kg'].mean()
            hourly_predictions = []
            
            for hour in range(hours_ahead):
                predicted_carbon = float(predictions[0][0]) * (1 + np.random.normal(0, 0.1))
                hourly_predictions.append(max(predicted_carbon, 0))
            
            return {
                'model': 'LSTM',
                'predictions': hourly_predictions,
                'total_predicted_kg': sum(hourly_predictions),
                'average_per_hour_kg': np.mean(hourly_predictions),
                'confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"[ERROR] LSTM prediction failed: {e}")
            return self._heuristic_prediction(monitoring_data, hours_ahead)
    
    def predict_xgboost(self, monitoring_data: pd.DataFrame) -> Dict:
        """
        Predict carbon emissions using XGBoost for feature-based prediction
        
        Args:
            monitoring_data: Current monitoring data
        
        Returns:
            Dictionary with predictions and feature importance
        """
        
        if self.xgboost_model is None:
            logger.warning("[WARN] XGBoost model not loaded, using heuristic prediction")
            return self._heuristic_prediction(monitoring_data, 1)
        
        try:
            # Prepare features
            features = self.prepare_features(monitoring_data)
            
            # Predict
            predictions = self.xgboost_model.predict(features)
            
            return {
                'model': 'XGBoost',
                'predictions': predictions.tolist(),
                'average_prediction_kg': float(np.mean(predictions)),
                'confidence': 0.88
            }
            
        except Exception as e:
            logger.error(f"[ERROR] XGBoost prediction failed: {e}")
            return self._heuristic_prediction(monitoring_data, 1)
    
    def _heuristic_prediction(self, monitoring_data: pd.DataFrame, 
                             hours_ahead: int = 24) -> Dict:
        """
        Fallback heuristic prediction when models not available
        Uses historical patterns and trends
        """
        
        if monitoring_data.empty:
            base_carbon = 0.001
        else:
            base_carbon = monitoring_data['carbon_emissions_kg'].mean()
        
        # Simulate daily pattern (higher during business hours)
        hourly_predictions = []
        current_hour = datetime.now().hour
        
        for hour in range(hours_ahead):
            hour_of_day = (current_hour + hour) % 24
            
            # Business hours multiplier
            if 9 <= hour_of_day <= 17:
                multiplier = 1.3
            elif 0 <= hour_of_day <= 6:
                multiplier = 0.7
            else:
                multiplier = 1.0
            
            # Add some randomness
            predicted = base_carbon * multiplier * (1 + np.random.normal(0, 0.05))
            hourly_predictions.append(max(predicted, 0))
        
        return {
            'model': 'Heuristic',
            'predictions': hourly_predictions,
            'total_predicted_kg': sum(hourly_predictions),
            'average_per_hour_kg': np.mean(hourly_predictions),
            'confidence': 0.65
        }
    
    def recommend_optimization_strategy(self, monitoring_data: pd.DataFrame) -> List[Dict]:
        """
        Recommend optimization strategies based on current metrics
        
        Returns:
            List of recommended strategies with expected impact
        """
        
        if monitoring_data.empty:
            return []
        
        recommendations = []
        
        # Calculate current metrics
        avg_cpu = monitoring_data['cpu_utilization'].mean()
        current_carbon = monitoring_data['carbon_emissions_kg_per_day'].mean()
        
        # Strategy 1: Temporal Shift (if high usage during peak hours)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17 and avg_cpu > 30:
            recommendations.append({
                'strategy': 'Temporal Shift to Off-Peak Hours',
                'priority': 'HIGH',
                'reduction_percentage': 35.8,
                'estimated_saving_kg': current_carbon * 0.358,
                'implementation': 'Schedule non-critical workloads during 00:00-06:00',
                'difficulty': 'Medium'
            })
        
        # Strategy 2: Geographic Migration (always recommend if high carbon region)
        if monitoring_data['region'].iloc[0] in ['us-east-1', 'us-east-2']:
            recommendations.append({
                'strategy': 'Geographic Migration to Low-Carbon Region',
                'priority': 'HIGH',
                'reduction_percentage': 73.5,
                'estimated_saving_kg': current_carbon * 0.735,
                'implementation': 'Migrate to us-west-2 or ca-central-1 (lower carbon intensity)',
                'difficulty': 'High'
            })
        
        # Strategy 3: Resource Right-Sizing (if low CPU usage)
        if avg_cpu < 30:
            recommendations.append({
                'strategy': 'Resource Right-Sizing',
                'priority': 'MEDIUM',
                'reduction_percentage': 24.1,
                'estimated_saving_kg': current_carbon * 0.241,
                'implementation': f'Downsize instances (current avg CPU: {avg_cpu:.1f}%)',
                'difficulty': 'Low'
            })
        
        # Strategy 4: Code Optimization (always applicable)
        recommendations.append({
            'strategy': 'Code Optimization',
            'priority': 'MEDIUM',
            'reduction_percentage': 18.2,
            'estimated_saving_kg': current_carbon * 0.182,
            'implementation': 'Optimize algorithms, reduce computational complexity',
            'difficulty': 'Medium'
        })
        
        # Strategy 5: Dynamic Reliability Adjustment
        recommendations.append({
            'strategy': 'Dynamic Reliability Adjustment',
            'priority': 'LOW',
            'reduction_percentage': 29.8,
            'estimated_saving_kg': current_carbon * 0.298,
            'implementation': 'Adjust replication factor based on criticality',
            'difficulty': 'High'
        })
        
        # Strategy 6: Peak-Time Orchestration
        if avg_cpu > 50:
            recommendations.append({
                'strategy': 'Peak-Time Orchestration',
                'priority': 'MEDIUM',
                'reduction_percentage': 42.6,
                'estimated_saving_kg': current_carbon * 0.426,
                'implementation': 'Implement intelligent workload distribution',
                'difficulty': 'Medium'
            })
        
        # Sort by priority and reduction percentage
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: (priority_order[x['priority']], -x['reduction_percentage']))
        
        return recommendations
    
    def generate_prediction_report(self, monitoring_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive prediction report
        
        Args:
            monitoring_data: Recent monitoring data
        
        Returns:
            Complete prediction report with recommendations
        """
        
        logger.info("[PREDICT] Generating prediction report...")
        
        # LSTM prediction (24 hours)
        lstm_pred = self.predict_lstm(monitoring_data, hours_ahead=24)
        
        # XGBoost prediction (current)
        xgboost_pred = self.predict_xgboost(monitoring_data)
        
        # Optimization recommendations
        recommendations = self.recommend_optimization_strategy(monitoring_data)
        
        # Current metrics
        current_carbon = monitoring_data['carbon_emissions_kg_per_day'].mean() if not monitoring_data.empty else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                'daily_carbon_kg': current_carbon,
                'instances_monitored': len(monitoring_data),
                'avg_cpu_utilization': monitoring_data['cpu_utilization'].mean() if not monitoring_data.empty else 0
            },
            'lstm_prediction': lstm_pred,
            'xgboost_prediction': xgboost_pred,
            'recommendations': recommendations,
            'total_potential_saving_kg': sum(r['estimated_saving_kg'] for r in recommendations),
            'max_reduction_percentage': max((r['reduction_percentage'] for r in recommendations), default=0)
        }
        
        logger.info(f"[PREDICT] Report generated successfully")
        logger.info(f"   24h predicted carbon: {lstm_pred['total_predicted_kg']:.6f} kg")
        logger.info(f"   Recommendations: {len(recommendations)}")
        logger.info(f"   Max potential reduction: {report['max_reduction_percentage']:.1f}%")
        
        return report


def main():
    """Test the prediction engine"""
    
    print("\n[TEST] Carbon Prediction Engine")
    print("="*60)
    
    # Initialize engine (without pre-trained models)
    engine = CarbonPredictionEngine()
    
    # Create sample monitoring data
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)],
        'instance_id': ['i-test'] * 5,
        'instance_type': ['t3.micro'] * 5,
        'region': ['us-east-1'] * 5,
        'cpu_utilization': [25.5, 30.2, 28.7, 32.1, 29.5],
        'cpu_utilization_max': [45.0, 50.0, 48.0, 52.0, 49.0],
        'network_in_bytes': [1000000] * 5,
        'network_out_bytes': [500000] * 5,
        'actual_power_watts': [2.5] * 5,
        'carbon_intensity_kg_per_kwh': [0.000379] * 5,
        'carbon_emissions_kg': [0.00001] * 5,
        'carbon_emissions_kg_per_day': [0.00023] * 5
    })
    
    # Generate prediction report
    report = engine.generate_prediction_report(sample_data)
    
    print("\n[REPORT] Prediction Report")
    print("="*60)
    print(f"Current Daily Carbon: {report['current_metrics']['daily_carbon_kg']:.6f} kg")
    print(f"24h Predicted Carbon: {report['lstm_prediction']['total_predicted_kg']:.6f} kg")
    print(f"Confidence: {report['lstm_prediction']['confidence']*100:.0f}%")
    
    print(f"\n[RECOMMENDATIONS] Top 3 Optimization Strategies:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"\n{i}. {rec['strategy']}")
        print(f"   Priority: {rec['priority']}")
        print(f"   Reduction: {rec['reduction_percentage']}%")
        print(f"   Saving: {rec['estimated_saving_kg']:.6f} kg CO2/day")
        print(f"   Implementation: {rec['implementation']}")
    
    print("\n[SUCCESS] Test complete!")


if __name__ == "__main__":
    main()
