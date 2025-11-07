"""
Unit tests for time series analysis modules.

This module provides comprehensive tests for all components
of the time series analysis pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import modules to test
from src.data_utils import DataLoader, DataPreprocessor, DataAnalyzer
from src.lstm_model import LSTMModel, LSTMTrainer, TimeSeriesDataProcessor
from src.forecasting_models import ARIMAModel, ProphetModel, StatisticalModels, ModelComparator
from src.anomaly_detection import AnomalyDetector, AnomalyAnalyzer, AutoencoderAnomalyDetector


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = DataLoader.generate_synthetic_data(
            n_points=100,
            trend=0.001,
            seasonality_period=10,
            seasonality_amplitude=0.5,
            noise_level=0.1
        )
        
        assert isinstance(data, pd.Series)
        assert len(data) == 100
        assert data.name == 'synthetic_data'
        assert not data.isnull().any()
    
    @patch('src.data_utils.yf.download')
    def test_load_stock_data(self, mock_download):
        """Test stock data loading."""
        # Mock yfinance response
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Open': [99, 100, 101, 102, 103],
            'High': [101, 102, 103, 104, 105],
            'Low': [98, 99, 100, 101, 102]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        mock_download.return_value = mock_df
        
        data = DataLoader.load_stock_data('TEST', '2020-01-01', '2020-01-05')
        
        assert isinstance(data, pd.Series)
        assert len(data) == 5
        assert data.name == 'Close'
    
    def test_load_csv_data(self):
        """Test CSV data loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,value\n')
            f.write('2020-01-01,100\n')
            f.write('2020-01-02,101\n')
            f.write('2020-01-03,102\n')
            temp_file = f.name
        
        try:
            data = DataLoader.load_csv_data(temp_file, 'date', 'value')
            
            assert isinstance(data, pd.Series)
            assert len(data) == 3
            assert data.name == 'value'
        finally:
            os.unlink(temp_file)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor(scaler_type="minmax")
        data = np.array([1, 2, 3, 4, 5])
        
        scaled_data = preprocessor.fit_transform(data)
        
        assert scaled_data.shape == (5, 1)
        assert np.all(scaled_data >= 0)
        assert np.all(scaled_data <= 1)
        assert preprocessor.is_fitted
    
    def test_transform(self):
        """Test transform method."""
        preprocessor = DataPreprocessor(scaler_type="minmax")
        train_data = np.array([1, 2, 3, 4, 5])
        test_data = np.array([6, 7, 8])
        
        preprocessor.fit_transform(train_data)
        scaled_test = preprocessor.transform(test_data)
        
        assert scaled_test.shape == (3, 1)
        assert np.all(scaled_test >= 0)
        assert np.all(scaled_test <= 1)
    
    def test_inverse_transform(self):
        """Test inverse_transform method."""
        preprocessor = DataPreprocessor(scaler_type="minmax")
        data = np.array([1, 2, 3, 4, 5])
        
        scaled_data = preprocessor.fit_transform(data)
        original_data = preprocessor.inverse_transform(scaled_data)
        
        np.testing.assert_array_almost_equal(original_data.flatten(), data)
    
    def test_create_sequences(self):
        """Test create_sequences method."""
        preprocessor = DataPreprocessor()
        data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        sequence_length = 3
        
        X, y = preprocessor.create_sequences(data, sequence_length)
        
        assert X.shape == (7, 3, 1)  # 10 - 3 = 7 sequences
        assert y.shape == (7, 1)
        assert X[0].shape == (3, 1)
    
    def test_split_data(self):
        """Test split_data method."""
        preprocessor = DataPreprocessor()
        data = np.arange(100)
        
        train, val, test = preprocessor.split_data(data, train_ratio=0.6, val_ratio=0.2)
        
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20
        assert len(train) + len(val) + len(test) == len(data)


class TestDataAnalyzer:
    """Test cases for DataAnalyzer class."""
    
    def test_basic_statistics(self):
        """Test basic_statistics method."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        stats = DataAnalyzer.basic_statistics(data)
        
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['count'] == 10
        assert stats['mean'] == 5.5
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])  # 100 is an outlier
        
        outliers = DataAnalyzer.detect_outliers(data, method="iqr")
        
        assert outliers.iloc[-1] == True  # Last value should be detected as outlier
        assert outliers.iloc[:-1].sum() == 0  # Other values should not be outliers


class TestLSTMModel:
    """Test cases for LSTMModel class."""
    
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization."""
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
        
        assert model.input_size == 1
        assert model.hidden_size == 50
        assert model.num_layers == 2
        assert model.output_size == 1
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
        x = torch.randn(32, 30, 1)  # batch_size=32, seq_len=30, input_size=1
        
        output = model(x)
        
        assert output.shape == (32, 1)  # batch_size=32, output_size=1


class TestTimeSeriesDataProcessor:
    """Test cases for TimeSeriesDataProcessor class."""
    
    def test_create_sequences(self):
        """Test create_sequences method."""
        processor = TimeSeriesDataProcessor(sequence_length=3)
        data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        
        X, y = processor.create_sequences(data)
        
        assert X.shape == (7, 3, 1)
        assert y.shape == (7, 1)
        assert X[0].shape == (3, 1)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        processor = TimeSeriesDataProcessor()
        data = np.array([[1], [2], [3], [4], [5]])
        
        scaled_data = processor.fit_transform(data)
        
        assert scaled_data.shape == (5, 1)
        assert processor.is_fitted


class TestLSTMTrainer:
    """Test cases for LSTMTrainer class."""
    
    def test_trainer_initialization(self):
        """Test LSTMTrainer initialization."""
        model = LSTMModel()
        trainer = LSTMTrainer(model, learning_rate=0.001)
        
        assert trainer.model == model
        assert trainer.device is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
    
    def test_train_epoch(self):
        """Test train_epoch method."""
        model = LSTMModel()
        trainer = LSTMTrainer(model)
        
        # Create dummy data
        X = torch.randn(32, 30, 1)
        y = torch.randn(32, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        loss = trainer.train_epoch(dataloader)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def test_isolation_forest_detector(self):
        """Test Isolation Forest anomaly detector."""
        detector = AnomalyDetector(method="isolation_forest")
        
        # Generate normal data with some outliers
        normal_data = np.random.normal(0, 1, (100, 1))
        outlier_data = np.random.normal(10, 1, (10, 1))
        data = np.vstack([normal_data, outlier_data])
        
        detector.fit(data)
        predictions = detector.predict(data)
        
        assert len(predictions) == len(data)
        assert set(predictions) == {-1, 1}  # Anomaly scores are -1 or 1
    
    def test_statistical_detector(self):
        """Test statistical anomaly detector."""
        detector = AnomalyDetector(method="statistical")
        
        # Generate normal data with some outliers
        normal_data = np.random.normal(0, 1, (100, 1))
        outlier_data = np.random.normal(10, 1, (10, 1))
        data = np.vstack([normal_data, outlier_data])
        
        detector.fit(data)
        predictions = detector.predict(data)
        
        assert len(predictions) == len(data)
        assert set(predictions) == {-1, 1}


class TestAnomalyAnalyzer:
    """Test cases for AnomalyAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test AnomalyAnalyzer initialization."""
        analyzer = AnomalyAnalyzer()
        
        assert analyzer.detectors == {}
        assert analyzer.results == {}
    
    def test_add_detector(self):
        """Test add_detector method."""
        analyzer = AnomalyAnalyzer()
        detector = AnomalyDetector(method="isolation_forest")
        
        analyzer.add_detector("test_detector", detector)
        
        assert "test_detector" in analyzer.detectors
        assert analyzer.detectors["test_detector"] == detector


class TestStatisticalModels:
    """Test cases for StatisticalModels class."""
    
    @patch('src.forecasting_models.STATSMODELS_AVAILABLE', True)
    @patch('src.forecasting_models.seasonal_decompose')
    def test_seasonal_decomposition(self, mock_decompose):
        """Test seasonal decomposition."""
        # Mock decomposition result
        mock_result = MagicMock()
        mock_result.trend = pd.Series([1, 2, 3, 4, 5])
        mock_result.seasonal = pd.Series([0.1, 0.2, 0.1, 0.2, 0.1])
        mock_result.resid = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1])
        mock_result.observed = pd.Series([1.2, 2.1, 3.1, 4.1, 5.1])
        
        mock_decompose.return_value = mock_result
        
        data = pd.Series([1, 2, 3, 4, 5])
        decomposition = StatisticalModels.seasonal_decomposition(data)
        
        assert 'trend' in decomposition
        assert 'seasonal' in decomposition
        assert 'residual' in decomposition
        assert 'observed' in decomposition


class TestModelComparator:
    """Test cases for ModelComparator class."""
    
    def test_comparator_initialization(self):
        """Test ModelComparator initialization."""
        comparator = ModelComparator()
        
        assert comparator.models == {}
        assert comparator.results == {}
    
    def test_add_model(self):
        """Test add_model method."""
        comparator = ModelComparator()
        model = MagicMock()
        
        comparator.add_model("test_model", model)
        
        assert "test_model" in comparator.models
        assert comparator.models["test_model"] == model


class TestAutoencoderAnomalyDetector:
    """Test cases for AutoencoderAnomalyDetector class."""
    
    def test_autoencoder_initialization(self):
        """Test AutoencoderAnomalyDetector initialization."""
        model = AutoencoderAnomalyDetector(input_dim=10, encoding_dim=5)
        
        assert model.input_dim == 10
        assert model.encoding_dim == 5
    
    def test_autoencoder_forward_pass(self):
        """Test autoencoder forward pass."""
        model = AutoencoderAnomalyDetector(input_dim=10, encoding_dim=5)
        x = torch.randn(32, 10)
        
        encoded, decoded = model(x)
        
        assert encoded.shape == (32, 5)
        assert decoded.shape == (32, 10)


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_data_loading_and_preprocessing_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Generate synthetic data
        data = DataLoader.generate_synthetic_data(n_points=100)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.fit_transform(data)
        
        # Create sequences
        processor = TimeSeriesDataProcessor(sequence_length=10)
        X, y = processor.create_sequences(scaled_data)
        
        # Split data
        train_data, val_data, test_data = preprocessor.split_data(scaled_data)
        
        assert len(scaled_data) == len(data)
        assert X.shape[0] == len(scaled_data) - 10
        assert len(train_data) + len(val_data) + len(test_data) == len(scaled_data)
    
    def test_lstm_training_pipeline(self):
        """Test complete LSTM training pipeline."""
        # Generate synthetic data
        data = DataLoader.generate_synthetic_data(n_points=200)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.fit_transform(data)
        
        # Split data
        train_data, val_data, test_data = preprocessor.split_data(scaled_data)
        
        # Create sequences
        processor = TimeSeriesDataProcessor(sequence_length=10)
        X_train, y_train = processor.create_sequences(train_data)
        X_val, y_val = processor.create_sequences(val_data)
        
        # Initialize model and trainer
        model = LSTMModel(input_size=1, hidden_size=20, num_layers=1)
        trainer = LSTMTrainer(model, learning_rate=0.01)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train model
        history = trainer.train(train_loader, val_loader, epochs=5, verbose=False)
        
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 5
        assert len(history['val_losses']) == 5


if __name__ == "__main__":
    pytest.main([__file__])
