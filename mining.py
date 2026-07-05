import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import logging
from pathlib import Path
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPricePredictor:
    def __init__(self):
        """Initialize the stock price predictor with multiple models."""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
        }
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_features(self, df):
        """Prepare features for prediction."""
        # Select features
        feature_cols = [
            'vader_compound', 'vader_positive', 'vader_negative',
            'vader_neutral', 'textblob_polarity', 'textblob_subjectivity',
            'tweet_count', 'daily_return', 'volatility'
        ]
        
        # Create lagged features
        for col in ['daily_return', 'vader_compound', 'tweet_count']:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}'] = df.groupby('Stock Name')[col].shift(lag)
        
        # Drop rows with NaN values (due to lag creation)
        df = df.dropna()
        
        # Prepare X and y
        X = df[feature_cols + [col for col in df.columns if 'lag' in col]]
        y = df['daily_return'].shift(-1)  # Predict next day's return
        
        # Drop the last row as we don't have the next day's return
        X = X[:-1]
        y = y[:-1]
        
        return X, y
    
    def train_models(self, X, y):
        """Train all models."""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create results DataFrame
        results_df = pd.DataFrame({'actual': y_test})
        metrics_list = []
        
        # Train and evaluate each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            results_df[f'{name}_pred'] = y_pred
            
            # Calculate metrics
            metrics = {
                'model': name,
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            metrics_list.append(metrics)
            
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test
            }
            
            logger.info(f"{name} model metrics: {metrics}")
        
        # Save results and metrics
        results_df.to_csv('data/processed/model_results.csv', index=False)
        pd.DataFrame(metrics_list).to_csv('data/processed/model_metrics.csv', index=False)
        
        # Save feature importance if available
        feature_importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
        
        if feature_importance:
            pd.DataFrame(feature_importance).to_csv('data/processed/feature_importance.csv')
    
    def compare_models(self):
        """Compare the performance of all models."""
        comparison = pd.DataFrame({
            name: results['metrics']
            for name, results in self.results.items()
        }).T
        
        return comparison
    
    def save_models(self):
        """Save trained models and scaler."""
        output_dir = Path('data/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, results in self.results.items():
            model_path = output_dir / f'{name}_model.joblib'
            joblib.dump(results['model'], model_path)
        
        # Save scaler
        scaler_path = output_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        logger.info("Saved models and scaler")

def main():
    """Main function to run the data mining pipeline."""
    predictor = StockPricePredictor()
    
    try:
        # Load integrated data
        logger.info("Loading integrated data...")
        df = pd.read_csv('data/processed/integrated_data.csv')
        
        # Prepare features
        logger.info("Preparing features...")
        X, y = predictor.prepare_features(df)
        
        # Train models
        logger.info("Training models...")
        predictor.train_models(X, y)
        
        # Compare models
        comparison = predictor.compare_models()
        logger.info("\nModel Comparison:\n" + str(comparison))
        
        # Save models
        predictor.save_models()
        
        logger.info("Data mining pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data mining pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 