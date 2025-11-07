"""
Additional forecasting models for time series analysis.

This module provides ARIMA, Prophet, and other forecasting models
for comprehensive comparison with LSTM.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Time series models
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logging.warning("pmdarima not available. ARIMA functionality will be limited.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("prophet not available. Prophet functionality will be disabled.")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Statistical models will be disabled.")

logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    ARIMA model wrapper with automatic parameter selection.
    """
    
    def __init__(self, seasonal: bool = True, stepwise: bool = True):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to use seasonal ARIMA
            stepwise: Whether to use stepwise parameter selection
        """
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: pd.Series, **kwargs) -> 'ARIMAModel':
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
            **kwargs: Additional arguments for auto_arima
            
        Returns:
            Self for method chaining
        """
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required for ARIMA functionality")
            
        logger.info("Fitting ARIMA model...")
        
        self.model = auto_arima(
            data,
            seasonal=self.seasonal,
            stepwise=self.stepwise,
            suppress_warnings=True,
            error_action='ignore',
            **kwargs
        )
        
        self.fitted_model = self.model
        logger.info(f"ARIMA model fitted: {self.model.order}")
        
        return self
    
    def predict(self, n_periods: int, **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            n_periods: Number of periods to predict
            **kwargs: Additional arguments for predict
            
        Returns:
            Predictions array
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = self.fitted_model.predict(n_periods=n_periods, **kwargs)
        return predictions
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Model summary string
        """
        if self.fitted_model is None:
            return "Model not fitted"
        return str(self.fitted_model.summary())


class ProphetModel:
    """
    Prophet model wrapper for time series forecasting.
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "multiplicative"
    ):
        """
        Initialize Prophet model.
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet functionality")
            
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode
        )
        self.fitted_model = None
        
    def fit(self, data: pd.DataFrame) -> 'ProphetModel':
        """
        Fit Prophet model to data.
        
        Args:
            data: DataFrame with 'ds' (datetime) and 'y' (values) columns
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Prophet model...")
        
        # Ensure data has required columns
        if 'ds' not in data.columns or 'y' not in data.columns:
            raise ValueError("Data must have 'ds' and 'y' columns")
            
        self.fitted_model = self.model.fit(data)
        logger.info("Prophet model fitted successfully")
        
        return self
    
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions.
        
        Args:
            future_df: DataFrame with 'ds' column for future dates
            
        Returns:
            DataFrame with predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        forecast = self.fitted_model.predict(future_df)
        return forecast
    
    def make_future_dataframe(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        """
        Create future dataframe for predictions.
        
        Args:
            periods: Number of periods to predict
            freq: Frequency of the time series
            
        Returns:
            Future dataframe
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        return self.fitted_model.make_future_dataframe(periods=periods, freq=freq)


class StatisticalModels:
    """
    Collection of statistical time series models.
    """
    
    @staticmethod
    def seasonal_decomposition(
        data: pd.Series,
        model: str = 'additive',
        period: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition.
        
        Args:
            data: Time series data
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (auto-detected if None)
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for statistical models")
            
        decomposition = seasonal_decompose(data, model=model, period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    
    @staticmethod
    def plot_decomposition(
        decomposition: Dict[str, pd.Series],
        title: str = "Seasonal Decomposition",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot seasonal decomposition.
        
        Args:
            decomposition: Decomposition results
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition['observed'].plot(ax=axes[0], title=f"{title} - Observed")
        decomposition['trend'].plot(ax=axes[1], title="Trend")
        decomposition['seasonal'].plot(ax=axes[2], title="Seasonal")
        decomposition['residual'].plot(ax=axes[3], title="Residual")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelComparator:
    """
    Compare multiple forecasting models.
    """
    
    def __init__(self):
        """Initialize the model comparator."""
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to compare.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        
    def fit_all(self, train_data: pd.Series) -> None:
        """
        Fit all models to training data.
        
        Args:
            train_data: Training time series data
        """
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name} model...")
                
                if hasattr(model, 'fit'):
                    if isinstance(model, ProphetModel):
                        # Prophet requires DataFrame with ds and y columns
                        df = pd.DataFrame({
                            'ds': train_data.index,
                            'y': train_data.values
                        })
                        model.fit(df)
                    else:
                        model.fit(train_data)
                        
                self.results[name] = {'model': model, 'fitted': True}
                logger.info(f"{name} model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}")
                self.results[name] = {'model': model, 'fitted': False, 'error': str(e)}
    
    def predict_all(self, n_periods: int) -> Dict[str, np.ndarray]:
        """
        Make predictions with all fitted models.
        
        Args:
            n_periods: Number of periods to predict
            
        Returns:
            Dictionary of predictions by model name
        """
        predictions = {}
        
        for name, result in self.results.items():
            if not result.get('fitted', False):
                continue
                
            try:
                model = result['model']
                
                if isinstance(model, ProphetModel):
                    # Prophet prediction
                    future_df = model.make_future_dataframe(periods=n_periods)
                    forecast = model.predict(future_df)
                    predictions[name] = forecast['yhat'].tail(n_periods).values
                else:
                    # Other models
                    predictions[name] = model.predict(n_periods=n_periods)
                    
            except Exception as e:
                logger.error(f"Error predicting with {name} model: {e}")
                predictions[name] = None
                
        return predictions
    
    def evaluate_all(
        self,
        test_data: pd.Series,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Evaluate all models.
        
        Args:
            test_data: Test time series data
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with evaluation metrics
        """
        metrics = []
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            try:
                # Calculate metrics
                mse = mean_squared_error(test_data.values, pred)
                mae = mean_absolute_error(test_data.values, pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((test_data.values - pred) / test_data.values)) * 100
                
                metrics.append({
                    'Model': name,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                
        return pd.DataFrame(metrics)
    
    def plot_comparison(
        self,
        actual: pd.Series,
        predictions: Dict[str, np.ndarray],
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of all models.
        
        Args:
            actual: Actual values
            predictions: Dictionary of predictions
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(actual.index, actual.values, label='Actual', linewidth=2, color='black')
        
        # Plot predictions
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for i, (name, pred) in enumerate(predictions.items()):
            if pred is not None:
                plt.plot(
                    actual.index[:len(pred)],
                    pred,
                    label=f'{name}',
                    alpha=0.7,
                    color=colors[i]
                )
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
