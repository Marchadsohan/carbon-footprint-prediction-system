import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CarbonOptimizer:
    """
    XGBoost-based Carbon Footprint Optimizer for recommendation generation
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.optimization_rules = []
        
        print("🎯 Carbon Optimizer initialized with XGBoost engine")
    
    def prepare_optimization_data(self, df):
        """Prepare data for optimization model training"""
        print("📊 Preparing optimization dataset...")
        
        # Create optimization scenarios
        optimization_data = []
        
        for idx, row in df.iterrows():
            current_record = row.to_dict()
            
            # Scenario 1: Temporal shifting (move to low-carbon hours)
            if current_record['is_business_hours']:
                scenario = current_record.copy()
                scenario['is_business_hours'] = False
                scenario['hour_of_day'] = 3  # 3 AM (low carbon)
                scenario['hour_sin'] = np.sin(2 * np.pi * 3 / 24)
                scenario['hour_cos'] = np.cos(2 * np.pi * 3 / 24)
                
                # Estimate carbon reduction
                carbon_reduction = current_record['carbon_emissions_kg_co2'] * 0.20  # 20% reduction
                scenario['carbon_emissions_kg_co2'] = current_record['carbon_emissions_kg_co2'] - carbon_reduction
                scenario['optimization_type'] = 'temporal_shift'
                scenario['carbon_saving'] = carbon_reduction
                scenario['cost_saving'] = carbon_reduction * 50  # $50 per kg CO2
                
                optimization_data.append(scenario)
            
            # Scenario 2: Geographic shifting (move to cleaner region)
            if current_record['renewable_energy_pct'] < 0.3:  # If low renewable
                scenario = current_record.copy()
                scenario['region_id'] = 5  # West region (high renewable)
                scenario['region_name'] = 'West'
                scenario['renewable_energy_pct'] = 0.40
                scenario['base_carbon_intensity'] = 0.25
                scenario['effective_carbon_intensity'] = 0.25 * (1 - 0.40)
                
                # Recalculate carbon emissions
                new_carbon = current_record['energy_consumption_kwh'] * scenario['effective_carbon_intensity']
                carbon_reduction = current_record['carbon_emissions_kg_co2'] - new_carbon
                scenario['carbon_emissions_kg_co2'] = new_carbon
                scenario['optimization_type'] = 'geographic_shift'
                scenario['carbon_saving'] = carbon_reduction
                scenario['cost_saving'] = carbon_reduction * 50
                
                optimization_data.append(scenario)
            
            # Scenario 3: Resource optimization (right-sizing)
            if current_record['cpu_usage_percent'] < 40:  # Underutilized
                scenario = current_record.copy()
                scenario['cpu_usage_percent'] = min(60, current_record['cpu_usage_percent'] * 1.5)
                scenario['memory_usage_gb'] = current_record['memory_usage_gb'] * 0.8
                
                # Recalculate energy and carbon
                cpu_energy = (scenario['cpu_usage_percent'] / 100) * 0.1
                memory_energy = scenario['memory_usage_gb'] * 0.005
                base_energy = 0.15  # Reduced base energy
                total_energy = cpu_energy + memory_energy + base_energy
                scenario['energy_consumption_kwh'] = total_energy * 1.4  # PUE
                
                new_carbon = scenario['energy_consumption_kwh'] * current_record['effective_carbon_intensity']
                carbon_reduction = current_record['carbon_emissions_kg_co2'] - new_carbon
                scenario['carbon_emissions_kg_co2'] = new_carbon
                scenario['optimization_type'] = 'resource_optimization'
                scenario['carbon_saving'] = carbon_reduction
                scenario['cost_saving'] = carbon_reduction * 50 + 200  # Additional cost savings
                
                optimization_data.append(scenario)
            
            # Add baseline scenario (no optimization)
            baseline = current_record.copy()
            baseline['optimization_type'] = 'baseline'
            baseline['carbon_saving'] = 0
            baseline['cost_saving'] = 0
            optimization_data.append(baseline)
        
        opt_df = pd.DataFrame(optimization_data)
        
        print(f"✅ Optimization dataset created:")
        print(f"   Total scenarios: {len(opt_df):,}")
        print(f"   Optimization types: {opt_df['optimization_type'].value_counts().to_dict()}")
        
        return opt_df
    
    def train_optimizer(self, df, target_col='carbon_saving'):
        """Train XGBoost model for carbon optimization"""
        print("🎯 Training XGBoost Carbon Optimizer...")
        
        # Prepare optimization data
        opt_df = self.prepare_optimization_data(df)
        
        # Select features for optimization model
        self.feature_columns = [
            'cpu_usage_percent', 'memory_usage_gb', 'energy_consumption_kwh',
            'hour_of_day', 'day_of_week', 'is_business_hours', 'is_weekend',
            'region_id', 'renewable_energy_pct', 'effective_carbon_intensity',
            'carbon_emissions_kg_co2'
        ]
        
        # Encode categorical variables
        categorical_features = ['optimization_type']
        for feature in categorical_features:
            if feature in opt_df.columns:
                le = LabelEncoder()
                opt_df[f'{feature}_encoded'] = le.fit_transform(opt_df[feature].astype(str))
                self.label_encoders[feature] = le
                self.feature_columns.append(f'{feature}_encoded')
        
        # Prepare features and target
        X = opt_df[self.feature_columns].copy()
        y = opt_df[target_col].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=opt_df['optimization_type']
        )
        
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        
        # XGBoost model with optimized parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**xgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"✅ XGBoost training completed!")
        print(f"   Training R²: {train_r2:.4f}, MAE: {train_mae:.6f}")
        print(f"   Test R²: {test_r2:.4f}, MAE: {test_mae:.6f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"🔍 Top 5 Important Features:")
        for idx, row in self.feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        return self.model
    
    def generate_recommendations(self, current_data, top_n=5):
        """Generate optimization recommendations for current state"""
        if not self.is_trained:
            print("❌ Optimizer not trained! Please train first.")
            return []
        
        print("💡 Generating optimization recommendations...")
        
        recommendations = []
        
        for idx, row in current_data.iterrows():
            current_record = row.to_dict()
            
            # Generate different optimization scenarios
            scenarios = []
            
            # 1. Temporal optimization
            if current_record.get('is_business_hours', False):
                scenario = self._create_temporal_scenario(current_record)
                scenarios.append(scenario)
            
            # 2. Geographic optimization
            if current_record.get('renewable_energy_pct', 0) < 0.3:
                scenario = self._create_geographic_scenario(current_record)
                scenarios.append(scenario)
            
            # 3. Resource optimization
            if current_record.get('cpu_usage_percent', 100) < 50:
                scenario = self._create_resource_scenario(current_record)
                scenarios.append(scenario)
            
            # Predict carbon savings for each scenario
            for scenario in scenarios:
                if len(scenario) > 0:
                    # Prepare features as DataFrame with correct column order
                    feature_data = {}
                    
                    # Fill in all required features
                    for col in self.feature_columns:
                        if col == 'optimization_type_encoded':
                            opt_type = scenario.get('optimization_type', 'baseline')
                            if 'optimization_type' in self.label_encoders:
                                try:
                                    encoded_val = self.label_encoders['optimization_type'].transform([opt_type])[0]
                                except ValueError:
                                    encoded_val = self.label_encoders['optimization_type'].transform(['baseline'])[0]
                                feature_data[col] = encoded_val
                            else:
                                feature_data[col] = 0
                        else:
                            feature_data[col] = scenario.get(col, 0)
                    
                    # Create DataFrame for prediction
                    feature_df = pd.DataFrame([feature_data])[self.feature_columns]
                    
                    # Make prediction
                    predicted_saving = self.model.predict(feature_df)[0]
                    
                    if predicted_saving > 0.001:  # Only show positive savings
                        recommendation = {
                            'type': scenario['optimization_type'],
                            'description': self._get_description(scenario),
                            'predicted_carbon_saving': max(0, predicted_saving),
                            'estimated_cost_saving': max(0, predicted_saving) * 50,
                            'confidence': min(100, abs(predicted_saving) * 10),
                            'implementation_effort': self._get_effort_level(scenario['optimization_type'])
                        }
                        
                        recommendations.append(recommendation)
        
        # Sort by predicted savings
        recommendations = sorted(recommendations, 
                               key=lambda x: x['predicted_carbon_saving'], 
                               reverse=True)[:top_n]
        
        return recommendations
    
    def _create_temporal_scenario(self, record):
        """Create temporal optimization scenario"""
        scenario = record.copy()
        scenario['optimization_type'] = 'temporal_shift'
        scenario['is_business_hours'] = False
        scenario['hour_of_day'] = 3  # 3 AM
        return scenario
    
    def _create_geographic_scenario(self, record):
        """Create geographic optimization scenario"""
        scenario = record.copy()
        scenario['optimization_type'] = 'geographic_shift'
        scenario['region_id'] = 5  # West region
        scenario['renewable_energy_pct'] = 0.40
        scenario['effective_carbon_intensity'] = 0.15
        return scenario
    
    def _create_resource_scenario(self, record):
        """Create resource optimization scenario"""
        scenario = record.copy()
        scenario['optimization_type'] = 'resource_optimization'
        scenario['cpu_usage_percent'] = min(70, record.get('cpu_usage_percent', 30) * 1.4)
        scenario['memory_usage_gb'] = record.get('memory_usage_gb', 16) * 0.8
        return scenario
    
    def _get_description(self, scenario):
        """Get human-readable description for scenario"""
        opt_type = scenario.get('optimization_type', '')
        
        descriptions = {
            'temporal_shift': f"Schedule workload during low-carbon hours (3 AM) instead of business hours",
            'geographic_shift': f"Migrate workload to West region (40% renewable energy)",
            'resource_optimization': f"Right-size resources: CPU to {scenario.get('cpu_usage_percent', 0):.1f}%, Memory to {scenario.get('memory_usage_gb', 0):.1f}GB"
        }
        
        return descriptions.get(opt_type, "Unknown optimization")
    
    def _get_effort_level(self, opt_type):
        """Get implementation effort level"""
        effort_levels = {
            'temporal_shift': 'Low',
            'geographic_shift': 'Medium', 
            'resource_optimization': 'Low'
        }
        return effort_levels.get(opt_type, 'Medium')
    
    def save_optimizer(self, model_dir='../../../models/xgboost'):
        """Save trained optimizer model"""
        print("💾 Saving XGBoost optimizer...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        self.model.save_model(os.path.join(model_dir, 'carbon_optimizer.json'))
        
        # Save other components
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_importance, os.path.join(model_dir, 'feature_importance.pkl'))
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        
        print(f"✅ Optimizer saved to: {model_dir}")

def main():
    """Main function to train carbon optimizer"""
    
    print("🎯 XGBoost Carbon Optimizer Training")
    print("=" * 50)
    
    # Load original dataset for optimization scenarios
    print("📂 Loading synthetic dataset...")
    try:
        df = pd.read_csv('../../../data/synthetic/carbon_footprint_dataset.csv')
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Use a subset for faster training
        df_sample = df.sample(n=1000, random_state=42)
        print(f"   Using sample: {df_sample.shape[0]} records")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize and train optimizer
    optimizer = CarbonOptimizer()
    model = optimizer.train_optimizer(df_sample)
    
    # Generate sample recommendations
    print("\n💡 Testing recommendation generation...")
    sample_data = df_sample.head(5)
    recommendations = optimizer.generate_recommendations(sample_data, top_n=3)
    
    print(f"\n🎯 SAMPLE RECOMMENDATIONS")
    print("=" * 40)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['type'].replace('_', ' ').title()}")
        print(f"   📋 {rec['description']}")
        print(f"   💨 Carbon Saving: {rec['predicted_carbon_saving']:.4f} kg CO2")
        print(f"   💰 Cost Saving: ${rec['estimated_cost_saving']:.2f}")
        print(f"   🎯 Confidence: {rec['confidence']:.1f}%")
        print(f"   ⚡ Effort: {rec['implementation_effort']}")
    
    # Save optimizer
    optimizer.save_optimizer()
    
    print(f"\n🎉 XGBoost optimizer training completed!")
    print("Next steps:")
    print("1. Create blockchain integration")
    print("2. Build prediction dashboard")

if __name__ == "__main__":
    main()
