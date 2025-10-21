import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CarbonAwareLSTM(layers.Layer):
    """
    Custom LSTM layer with carbon intensity awareness for enhanced prediction
    """
    
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        
        # Standard LSTM layer
        self.lstm = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=0.2,
            recurrent_dropout=0.1,
            activation='tanh',
            recurrent_activation='sigmoid'
        )
        
        # Carbon-aware attention mechanism
        self.carbon_attention = layers.Dense(1, activation='sigmoid', name='carbon_attention')
        self.combine = layers.Dense(units, activation='tanh', name='carbon_combine')
        
    def call(self, inputs, training=None):
        # For now, use all features for LSTM
        lstm_output = self.lstm(inputs, training=training)
        return lstm_output

class TCEPModel:
    """
    Temporal Carbon Emission Predictor with advanced LSTM architecture
    """
    
    def __init__(self, sequence_length=24, n_features=30, prediction_horizons=[1, 6, 24]):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_horizons = prediction_horizons
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        
        print(f"🧠 TCEP Model initialized:")
        print(f"   Sequence length: {sequence_length} hours")
        print(f"   Features: {n_features}")
        print(f"   Prediction horizons: {prediction_horizons} hours")
    
    def build_model(self):
        """Build the TCEP neural network architecture"""
        print("🏗️ Building TCEP neural network...")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='sequence_input')
        
        # Multi-scale temporal feature extraction
        # Branch 1: Short-term patterns (1-6 hours)
        short_conv = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        short_pool = layers.MaxPooling1D(pool_size=2)(short_conv)
        
        # Branch 2: Medium-term patterns (6-24 hours)
        medium_conv = layers.Conv1D(64, kernel_size=6, activation='relu', padding='same')(inputs)
        medium_pool = layers.MaxPooling1D(pool_size=2)(medium_conv)
        
        # Branch 3: Long-term patterns (24+ hours) with LSTM
        long_lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        long_lstm2 = layers.LSTM(64, return_sequences=False, dropout=0.2)(long_lstm1)
        
        # Flatten conv branches
        short_flat = layers.GlobalMaxPooling1D()(short_pool)
        medium_flat = layers.GlobalMaxPooling1D()(medium_pool)
        
        # Combine all branches
        combined = layers.concatenate([short_flat, medium_flat, long_lstm2], name='multi_scale_fusion')
        
        # Feature enhancement layer
        enhanced = layers.Dense(256, activation='relu', name='feature_enhancement')(combined)
        enhanced = layers.Dropout(0.3)(enhanced)
        enhanced = layers.BatchNormalization()(enhanced)
        
        # Shared representation layer
        shared_repr = layers.Dense(128, activation='relu', name='shared_representation')(enhanced)
        shared_repr = layers.Dropout(0.2)(shared_repr)
        
        # Multi-horizon prediction heads
        outputs = {}
        for horizon in self.prediction_horizons:
            # Horizon-specific processing
            horizon_dense = layers.Dense(64, activation='relu', name=f'horizon_{horizon}h_dense')(shared_repr)
            horizon_dropout = layers.Dropout(0.1)(horizon_dense)
            
            # Final prediction layer
            horizon_output = layers.Dense(1, activation='linear', name=f'carbon_pred_{horizon}h')(horizon_dropout)
            outputs[f'pred_{horizon}h'] = horizon_output
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='TCEP_Model')
        
        print("✅ TCEP model architecture built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def carbon_focused_loss(self, alpha=2.0):
        """
        Simplified custom loss function focused on carbon intensity
        """
        def loss_fn(y_true, y_pred):
            # Basic MSE loss
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # High carbon penalty (focus on reducing emissions during peak periods)
            mean_carbon = tf.reduce_mean(y_true)
            high_carbon_mask = tf.cast(y_true > mean_carbon, tf.float32)
            high_carbon_penalty = tf.reduce_mean(high_carbon_mask * tf.square(y_true - y_pred))
            
            # Combined loss
            total_loss = mse + alpha * high_carbon_penalty
            
            return total_loss
        
        return loss_fn
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with custom loss and metrics"""
        print("⚙️ Compiling TCEP model...")
        
        # Define losses and metrics for each horizon
        losses = {}
        metrics = {}
        loss_weights = {}
        
        for horizon in self.prediction_horizons:
            losses[f'pred_{horizon}h'] = self.carbon_focused_loss()
            metrics[f'pred_{horizon}h'] = ['mae', 'mse']
            # Weight shorter horizons more heavily
            loss_weights[f'pred_{horizon}h'] = 1.0 / (horizon ** 0.5)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )
        
        print("✅ Model compiled successfully!")
        return self.model
    
    def prepare_sequences(self, X, y):
        """Prepare sequences for LSTM training"""
        print(f"📊 Preparing sequences (length={self.sequence_length})...")
        
        sequences_X = []
        targets = {f'pred_{h}h': [] for h in self.prediction_horizons}
        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
        
        # Create sequences
        max_horizon = max(self.prediction_horizons)
        for i in range(len(X_values) - self.sequence_length - max_horizon + 1):
            # Input sequence
            seq_x = X_values[i:i + self.sequence_length]
            sequences_X.append(seq_x)
            
            # Target values for different horizons
            for horizon in self.prediction_horizons:
                target_idx = i + self.sequence_length + horizon - 1
                if target_idx < len(y_values):
                    targets[f'pred_{horizon}h'].append(y_values[target_idx])
                else:
                    targets[f'pred_{horizon}h'].append(y_values[-1])  # Use last available value
        
        # Convert to numpy arrays
        sequences_X = np.array(sequences_X, dtype=np.float32)
        for horizon in self.prediction_horizons:
            targets[f'pred_{horizon}h'] = np.array(targets[f'pred_{horizon}h'], dtype=np.float32)
        
        print(f"✅ Sequences prepared:")
        print(f"   Input sequences: {sequences_X.shape}")
        for horizon in self.prediction_horizons:
            print(f"   Target {horizon}h: {targets[f'pred_{horizon}h'].shape}")
        
        return sequences_X, targets
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=1):
        """Train the TCEP model"""
        print("🚀 Starting TCEP model training...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        print(f"📈 Training with {len(X_train_seq)} sequences...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.is_trained = True
        print("✅ TCEP training completed!")
        
        return self.history
    
    def predict(self, X_test, return_sequences=False):
        """Make predictions using trained TCEP model"""
        print("🔮 Making TCEP predictions...")
        
        # Prepare test sequences
        X_test_seq, _ = self.prepare_sequences(X_test, np.zeros(len(X_test)))  # Dummy targets
        
        # Make predictions
        predictions = self.model.predict(X_test_seq, verbose=0)
        
        print(f"✅ Predictions completed for {len(X_test_seq)} sequences")
        return predictions
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate TCEP model performance"""
        print("📊 Evaluating TCEP model performance...")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Prepare test sequences for comparison
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)
        
        # Calculate metrics for each horizon
        results = {}
        for horizon in self.prediction_horizons:
            y_true = y_test_seq[f'pred_{horizon}h']
            y_pred = predictions[f'pred_{horizon}h'].flatten()
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            r2 = r2_score(y_true, y_pred)
            
            results[f'{horizon}h'] = {
                'MAE': round(mae, 6),
                'MSE': round(mse, 8),
                'RMSE': round(rmse, 6),
                'MAPE': round(mape, 2),
                'R²': round(r2, 4)
            }
        
        return results
    
    def save_model(self, model_dir='../../../models/tcep'):
        """Save the trained TCEP model"""
        print("💾 Saving TCEP model...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model architecture and weights
        self.model.save(os.path.join(model_dir, 'tcep_model.h5'))
        
        # Save model configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'prediction_horizons': self.prediction_horizons,
            'is_trained': self.is_trained
        }
        
        joblib.dump(config, os.path.join(model_dir, 'tcep_config.pkl'))
        
        print(f"✅ TCEP model saved to: {model_dir}")

def main():
    """Main function to train TCEP model"""
    
    print("🌱 TCEP (Temporal Carbon Emission Predictor) Training")
    print("=" * 60)
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    try:
        X_train = pd.read_csv('../../../data/processed/X_train.csv')
        X_val = pd.read_csv('../../../data/processed/X_val.csv')  
        X_test = pd.read_csv('../../../data/processed/X_test.csv')
        y_train = pd.read_csv('../../../data/processed/y_train.csv').squeeze()
        y_val = pd.read_csv('../../../data/processed/y_val.csv').squeeze()
        y_test = pd.read_csv('../../../data/processed/y_test.csv').squeeze()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please run the preprocessing pipeline first!")
        return
    
    # Initialize TCEP model
    tcep = TCEPModel(
        sequence_length=24,  # 24 hours
        n_features=X_train.shape[1],
        prediction_horizons=[1, 6, 24]  # 1h, 6h, 1day
    )
    
    # Build and compile model
    tcep.build_model()
    tcep.compile_model(learning_rate=0.001)
    
    # Display model architecture
    print("\n🏗️ TCEP Model Architecture:")
    tcep.model.summary()
    
    # Train model
    history = tcep.train(
        X_train, y_train, X_val, y_val,
        epochs=30,  # Reduced for faster training
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating TCEP Performance...")
    results = tcep.evaluate_model(X_test, y_test)
    
    print("\n🎯 TCEP MODEL PERFORMANCE")
    print("=" * 40)
    for horizon, metrics in results.items():
        print(f"\n🔮 {horizon} Prediction:")
        print(f"   MAE: {metrics['MAE']:.6f} kg CO2")
        print(f"   RMSE: {metrics['RMSE']:.6f} kg CO2")  
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   R²: {metrics['R²']:.4f}")
    
    # Save model
    tcep.save_model()
    
    print("\n🎉 TCEP training and evaluation completed!")
    print("Next steps:")
    print("1. Build XGBoost optimization engine")
    print("2. Create blockchain integration")
    print("3. Build prediction dashboard")

if __name__ == "__main__":
    main()
