"""
Main script for comprehensive time series analysis.

This script demonstrates the complete pipeline including data loading,
preprocessing, model training, evaluation, and visualization.
"""

import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data_utils import DataLoader, DataPreprocessor, DataAnalyzer
from src.lstm_model import LSTMModel, LSTMTrainer, TimeSeriesDataProcessor
from src.forecasting_models import ARIMAModel, ProphetModel, ModelComparator, StatisticalModels
from src.anomaly_detection import AnomalyDetector, AnomalyAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/timeseries.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {
            'data': {'symbol': 'TSLA', 'start_date': '2020-01-01', 'end_date': '2023-01-01'},
            'models': {'lstm': {'hidden_size': 50, 'epochs': 100}},
            'visualization': {'figure_size': [12, 8]}
        }


def load_and_prepare_data(config: Dict[str, Any]) -> tuple:
    """
    Load and prepare data for analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (data, preprocessor, train_data, val_data, test_data)
    """
    logger.info("Loading and preparing data...")
    
    # Load data
    data_config = config['data']
    
    try:
        # Try to load stock data first
        data = DataLoader.load_stock_data(
            symbol=data_config['symbol'],
            start_date=data_config['start_date'],
            end_date=data_config['end_date']
        )
        logger.info(f"Loaded stock data for {data_config['symbol']}: {len(data)} points")
    except Exception as e:
        logger.warning(f"Failed to load stock data: {e}. Generating synthetic data...")
        data = DataLoader.generate_synthetic_data(
            n_points=1000,
            trend=0.001,
            seasonality_period=30,
            seasonality_amplitude=0.5,
            noise_level=0.1
        )
        logger.info(f"Generated synthetic data: {len(data)} points")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaler_type="minmax")
    
    # Scale data
    scaled_data = preprocessor.fit_transform(data)
    
    # Split data
    train_data, val_data, test_data = preprocessor.split_data(
        scaled_data,
        train_ratio=data_config.get('train_split', 0.8),
        val_ratio=data_config.get('validation_split', 0.1)
    )
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return data, preprocessor, train_data, val_data, test_data


def train_lstm_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: Dict[str, Any]
) -> tuple:
    """
    Train LSTM model.
    
    Args:
        train_data: Training data
        val_data: Validation data
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, trainer, history)
    """
    logger.info("Training LSTM model...")
    
    # Get LSTM configuration
    lstm_config = config['models']['lstm']
    sequence_length = config['data']['sequence_length']
    
    # Create sequences
    processor = TimeSeriesDataProcessor(sequence_length=sequence_length)
    
    X_train, y_train = processor.create_sequences(train_data)
    X_val, y_val = processor.create_sequences(val_data)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=lstm_config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=lstm_config.get('batch_size', 32), shuffle=False)
    
    # Initialize model
    model = LSTMModel(
        input_size=1,
        hidden_size=lstm_config['hidden_size'],
        num_layers=lstm_config.get('num_layers', 2),
        dropout=lstm_config.get('dropout', 0.2)
    )
    
    # Initialize trainer
    trainer = LSTMTrainer(
        model,
        learning_rate=lstm_config.get('learning_rate', 0.001)
    )
    
    # Train model
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=lstm_config['epochs'],
        early_stopping_patience=lstm_config.get('early_stopping_patience', 10)
    )
    
    logger.info("LSTM model training completed")
    
    return model, trainer, history


def train_other_models(data: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train ARIMA and Prophet models.
    
    Args:
        data: Time series data
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    logger.info("Training additional models...")
    
    models = {}
    
    # Train ARIMA
    try:
        logger.info("Training ARIMA model...")
        arima_config = config['models']['arima']
        
        arima_model = ARIMAModel(
            seasonal=arima_config.get('seasonal', True),
            stepwise=arima_config.get('stepwise', True)
        )
        arima_model.fit(data)
        models['ARIMA'] = arima_model
        logger.info("ARIMA model trained successfully")
        
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")
    
    # Train Prophet
    try:
        logger.info("Training Prophet model...")
        prophet_config = config['models']['prophet']
        
        prophet_model = ProphetModel(
            yearly_seasonality=prophet_config.get('yearly_seasonality', True),
            weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            daily_seasonality=prophet_config.get('daily_seasonality', False),
            seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative')
        )
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        prophet_model.fit(prophet_df)
        models['Prophet'] = prophet_model
        logger.info("Prophet model trained successfully")
        
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
    
    return models


def run_anomaly_detection(data: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run anomaly detection analysis.
    
    Args:
        data: Time series data
        config: Configuration dictionary
        
    Returns:
        Dictionary of anomaly detection results
    """
    logger.info("Running anomaly detection...")
    
    anomaly_config = config['anomaly_detection']
    analyzer = AnomalyAnalyzer()
    
    # Add detectors
    if 'isolation_forest' in anomaly_config:
        detector = AnomalyDetector("isolation_forest")
        analyzer.add_detector("Isolation Forest", detector)
    
    if 'autoencoder' in anomaly_config:
        detector = AnomalyDetector("autoencoder")
        analyzer.add_detector("Autoencoder", detector)
    
    # Fit detectors
    analyzer.fit_all(data.values.reshape(-1, 1), **anomaly_config)
    
    # Detect anomalies
    predictions = analyzer.detect_all(data.values.reshape(-1, 1))
    scores = analyzer.get_scores_all(data.values.reshape(-1, 1))
    
    logger.info("Anomaly detection completed")
    
    return {
        'predictions': predictions,
        'scores': scores,
        'analyzer': analyzer
    }


def evaluate_models(
    models: Dict[str, Any],
    test_data: np.ndarray,
    preprocessor: DataPreprocessor,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate all trained models.
    
    Args:
        models: Dictionary of trained models
        test_data: Test data
        preprocessor: Data preprocessor
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating models...")
    
    results = {}
    sequence_length = config['data']['sequence_length']
    
    for model_name, model in models.items():
        try:
            if model_name == 'LSTM':
                # LSTM evaluation
                trainer = model['trainer']
                
                # Create test sequences
                processor = TimeSeriesDataProcessor(sequence_length=sequence_length)
                X_test, y_test = processor.create_sequences(test_data)
                
                # Create test data loader
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.FloatTensor(y_test)
                )
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Make predictions
                predictions = trainer.predict(test_loader)
                predictions_unscaled = preprocessor.inverse_transform(predictions.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                actual_values = preprocessor.inverse_transform(test_data.reshape(-1, 1)).flatten()
                actual_values = actual_values[-len(predictions_unscaled):]
                
                mse = np.mean((actual_values - predictions_unscaled) ** 2)
                mae = np.mean(np.abs(actual_values - predictions_unscaled))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - predictions_unscaled) / actual_values)) * 100
                
                results[model_name] = {
                    'predictions': predictions_unscaled,
                    'actual': actual_values,
                    'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}
                }
                
            elif model_name == 'ARIMA':
                # ARIMA evaluation
                n_periods = len(test_data)
                predictions = model.predict(n_periods)
                
                # Calculate metrics
                actual_values = preprocessor.inverse_transform(test_data.reshape(-1, 1)).flatten()
                actual_values = actual_values[-len(predictions):]
                
                mse = np.mean((actual_values - predictions) ** 2)
                mae = np.mean(np.abs(actual_values - predictions))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                
                results[model_name] = {
                    'predictions': predictions,
                    'actual': actual_values,
                    'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}
                }
                
            elif model_name == 'Prophet':
                # Prophet evaluation
                future_df = model.make_future_dataframe(periods=len(test_data))
                forecast = model.predict(future_df)
                
                predictions = forecast['yhat'].tail(len(test_data)).values
                
                # Calculate metrics
                actual_values = preprocessor.inverse_transform(test_data.reshape(-1, 1)).flatten()
                
                mse = np.mean((actual_values - predictions) ** 2)
                mae = np.mean(np.abs(actual_values - predictions))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                
                results[model_name] = {
                    'predictions': predictions,
                    'actual': actual_values,
                    'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}
                }
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    logger.info("Model evaluation completed")
    return results


def create_visualizations(
    data: pd.Series,
    results: Dict[str, Any],
    anomaly_results: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Create comprehensive visualizations.
    
    Args:
        data: Original time series data
        results: Model evaluation results
        anomaly_results: Anomaly detection results
        config: Configuration dictionary
    """
    logger.info("Creating visualizations...")
    
    fig_size = config['visualization']['figure_size']
    
    # 1. Data analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    axes[0, 0].plot(data.index, data.values, linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Original Time Series Data')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution
    axes[0, 1].hist(data.values, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Data Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rolling statistics
    window = 30
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    axes[1, 0].plot(data.index, data.values, alpha=0.3, label='Original')
    axes[1, 0].plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean ({window})')
    axes[1, 0].set_title('Rolling Statistics')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 1].boxplot(data.values)
    axes[1, 1].set_title('Box Plot')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model comparison plot
    if results:
        plt.figure(figsize=(15, 8))
        
        # Plot original data
        plt.plot(data.index, data.values, color='black', linewidth=2, label='Actual', alpha=0.7)
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, result) in enumerate(results.items()):
            predictions = result['predictions']
            # Create test dates
            test_dates = data.index[-len(predictions):]
            plt.plot(test_dates, predictions, color=colors[i % len(colors)], 
                    linewidth=2, label=f'{model_name} Prediction', alpha=0.8)
        
        plt.title('Model Predictions Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Anomaly detection plot
    if anomaly_results and anomaly_results['predictions']:
        predictions = anomaly_results['predictions']
        
        n_detectors = len([p for p in predictions.values() if p is not None])
        if n_detectors > 0:
            fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(15, 4 * (n_detectors + 1)))
            
            if n_detectors == 0:
                axes = [axes]
            
            # Plot original data
            axes[0].plot(data.index, data.values, color='blue', alpha=0.7)
            axes[0].set_title('Original Data')
            axes[0].set_ylabel('Value')
            axes[0].grid(True, alpha=0.3)
            
            # Plot anomaly detection results
            plot_idx = 1
            colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
            
            for i, (name, pred) in enumerate(predictions.items()):
                if pred is not None:
                    # Plot data with anomalies highlighted
                    normal_mask = pred == 1
                    anomaly_mask = pred == -1
                    
                    axes[plot_idx].plot(data.index, data.values, color='blue', alpha=0.3, label='Normal')
                    axes[plot_idx].scatter(
                        data.index[anomaly_mask],
                        data.values[anomaly_mask],
                        color='red',
                        s=50,
                        label='Anomaly',
                        alpha=0.8
                    )
                    
                    axes[plot_idx].set_title(f'{name} - Anomaly Detection')
                    axes[plot_idx].set_ylabel('Value')
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            plt.xlabel('Time')
            plt.tight_layout()
            plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    logger.info("Visualizations created and saved")


def main():
    """Main function to run the complete time series analysis pipeline."""
    logger.info("Starting comprehensive time series analysis...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Load and prepare data
        data, preprocessor, train_data, val_data, test_data = load_and_prepare_data(config)
        
        # Train LSTM model
        lstm_model, lstm_trainer, lstm_history = train_lstm_model(train_data, val_data, config)
        
        # Train other models
        other_models = train_other_models(data, config)
        
        # Combine all models
        all_models = {
            'LSTM': {'model': lstm_model, 'trainer': lstm_trainer, 'history': lstm_history},
            **other_models
        }
        
        # Run anomaly detection
        anomaly_results = run_anomaly_detection(data, config)
        
        # Evaluate models
        evaluation_results = evaluate_models(all_models, test_data, preprocessor, config)
        
        # Create visualizations
        create_visualizations(data, evaluation_results, anomaly_results, config)
        
        # Print summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Models trained: {list(all_models.keys())}")
        logger.info(f"Anomaly detectors used: {list(anomaly_results['predictions'].keys())}")
        
        # Print evaluation metrics
        if evaluation_results:
            logger.info("\nModel Performance Summary:")
            for model_name, result in evaluation_results.items():
                metrics = result['metrics']
                logger.info(f"{model_name}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
                          f"RMSE={metrics['rmse']:.6f}, MAPE={metrics['mape']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
