"""
Data utilities for time series analysis.

This module provides functions for loading, preprocessing, and generating
synthetic time series data.
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for various time series sources.
    """
    
    @staticmethod
    def load_stock_data(
        symbol: str,
        start_date: str,
        end_date: str,
        column: str = "Close"
    ) -> pd.Series:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            column: Column to extract (default: Close)
            
        Returns:
            Time series data
        """
        try:
            logger.info(f"Loading stock data for {symbol} from {start_date} to {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in data")
                
            return df[column]
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise
    
    @staticmethod
    def generate_synthetic_data(
        n_points: int = 1000,
        trend: float = 0.001,
        seasonality_period: int = 30,
        seasonality_amplitude: float = 0.5,
        noise_level: float = 0.1,
        start_date: Optional[str] = None
    ) -> pd.Series:
        """
        Generate synthetic time series data.
        
        Args:
            n_points: Number of data points
            trend: Linear trend coefficient
            seasonality_period: Period of seasonality
            seasonality_amplitude: Amplitude of seasonality
            noise_level: Standard deviation of noise
            start_date: Start date (if None, uses current date)
            
        Returns:
            Synthetic time series data
        """
        logger.info(f"Generating synthetic data with {n_points} points")
        
        # Generate time index
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_points)
        else:
            start_date = pd.to_datetime(start_date)
            
        dates = pd.date_range(start=start_date, periods=n_points, freq='D')
        
        # Generate synthetic data
        t = np.arange(n_points)
        
        # Trend component
        trend_component = trend * t
        
        # Seasonality component
        seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        
        # Noise component
        noise_component = np.random.normal(0, noise_level, n_points)
        
        # Combine components
        values = trend_component + seasonal_component + noise_component
        
        return pd.Series(values, index=dates, name='synthetic_data')
    
    @staticmethod
    def load_csv_data(
        file_path: str,
        date_column: str,
        value_column: str,
        date_format: Optional[str] = None
    ) -> pd.Series:
        """
        Load time series data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of date column
            value_column: Name of value column
            date_format: Date format string (if None, auto-detect)
            
        Returns:
            Time series data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            if date_column not in df.columns:
                raise ValueError(f"Date column {date_column} not found")
            if value_column not in df.columns:
                raise ValueError(f"Value column {value_column} not found")
                
            # Convert date column
            if date_format:
                df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            else:
                df[date_column] = pd.to_datetime(df[date_column])
                
            # Set date as index
            df = df.set_index(date_column)
            
            return df[value_column]
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise


class DataPreprocessor:
    """
    Data preprocessing utilities for time series.
    """
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        self.is_fitted = False
        
    def fit_transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            data: Input data
            
        Returns:
            Scaled data
        """
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)
            
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        return scaled_data
    
    def transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Input data
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
            
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
            
        return self.scaler.inverse_transform(data)
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_col: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Input data
            sequence_length: Length of sequences
            target_col: Target column index (if None, uses last column)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        if target_col is None:
            target_col = data.shape[1] - 1
            
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, target_col])
            
        return np.array(X), np.array(y)
    
    def split_data(
        self,
        data: Union[pd.Series, np.ndarray],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data (calculated as 1 - train_ratio - val_ratio)
            
        Returns:
            Tuple of (train, val, test) data
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data


class DataAnalyzer:
    """
    Data analysis utilities for time series.
    """
    
    @staticmethod
    def basic_statistics(data: pd.Series) -> Dict[str, Any]:
        """
        Calculate basic statistics for time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'median': data.median(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'missing_values': data.isnull().sum(),
            'duplicate_values': data.duplicated().sum()
        }
        
        return stats
    
    @staticmethod
    def plot_time_series(
        data: pd.Series,
        title: str = "Time Series Plot",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot time series data.
        
        Args:
            data: Time series data
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        plt.plot(data.index, data.values, linewidth=1, alpha=0.8)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_distribution(
        data: pd.Series,
        title: str = "Distribution Plot",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of time series data.
        
        Args:
            data: Time series data
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(data.values, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title(f"{title} - Histogram")
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data.values)
        ax2.set_title(f"{title} - Box Plot")
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_rolling_statistics(
        data: pd.Series,
        window: int = 30,
        title: str = "Rolling Statistics",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot rolling statistics.
        
        Args:
            data: Time series data
            window: Rolling window size
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original data
        axes[0, 0].plot(data.index, data.values, alpha=0.7)
        axes[0, 0].set_title(f"{title} - Original Data")
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling mean
        rolling_mean = data.rolling(window=window).mean()
        axes[0, 1].plot(data.index, data.values, alpha=0.3, label='Original')
        axes[0, 1].plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean ({window})')
        axes[0, 1].set_title(f"{title} - Rolling Mean")
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling std
        rolling_std = data.rolling(window=window).std()
        axes[1, 0].plot(rolling_std.index, rolling_std.values)
        axes[1, 0].set_title(f"{title} - Rolling Standard Deviation")
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling variance
        rolling_var = data.rolling(window=window).var()
        axes[1, 1].plot(rolling_var.index, rolling_var.values)
        axes[1, 1].set_title(f"{title} - Rolling Variance")
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def detect_outliers(
        data: pd.Series,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers in time series data.
        
        Args:
            data: Time series data
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
            
        elif method == "modified_zscore":
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def plot_outliers(
        data: pd.Series,
        outliers: pd.Series,
        title: str = "Outlier Detection",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot outliers in time series data.
        
        Args:
            data: Time series data
            outliers: Boolean series indicating outliers
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot normal data
        normal_data = data[~outliers]
        plt.plot(normal_data.index, normal_data.values, color='blue', alpha=0.7, label='Normal')
        
        # Plot outliers
        outlier_data = data[outliers]
        plt.scatter(outlier_data.index, outlier_data.values, color='red', s=50, label='Outliers', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
