# Time Series Analysis with LSTM, ARIMA, Prophet, and Anomaly Detection

A comprehensive time series analysis project featuring modern machine learning techniques for forecasting and anomaly detection. This project demonstrates state-of-the-art methods including LSTM neural networks, ARIMA, Prophet, and various anomaly detection algorithms.

## Features

- **Multiple Forecasting Models**: LSTM, ARIMA, Prophet with automatic parameter selection
- **Anomaly Detection**: Isolation Forest, Autoencoders, Statistical methods
- **Interactive Dashboard**: Streamlit-based web interface for data exploration
- **Comprehensive Visualization**: Plotly and Matplotlib visualizations
- **Modern Architecture**: Clean code structure with type hints and documentation
- **Configuration Management**: YAML-based configuration system
- **Unit Testing**: Comprehensive test suite for all components
- **Logging**: Structured logging throughout the pipeline

## Project Structure

```
├── src/                          # Source code modules
│   ├── lstm_model.py            # LSTM implementation with PyTorch
│   ├── forecasting_models.py   # ARIMA, Prophet, and statistical models
│   ├── anomaly_detection.py     # Anomaly detection algorithms
│   └── data_utils.py           # Data loading and preprocessing utilities
├── config/                       # Configuration files
│   └── config.yaml             # Main configuration file
├── data/                        # Data storage directory
├── models/                      # Trained model storage
├── logs/                        # Log files
├── tests/                       # Unit tests
│   └── test_modules.py         # Test suite
├── notebooks/                    # Jupyter notebooks for exploration
├── main.py                      # Main execution script
├── streamlit_app.py            # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-LSTM-ARIMA-Prophet-and-Anomaly-Detection.git
cd Time-Series-Analysis-with-LSTM-ARIMA-Prophet-and-Anomaly-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data models logs
```

## Usage

### Command Line Interface

Run the complete analysis pipeline:

```bash
python main.py
```

This will:
- Load or generate time series data
- Train LSTM, ARIMA, and Prophet models
- Run anomaly detection
- Generate comprehensive visualizations
- Save results and models

### Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard provides:
- Data loading from multiple sources (stock data, synthetic data, CSV upload)
- Interactive data exploration and visualization
- Model training with customizable parameters
- Real-time results visualization
- Anomaly detection analysis

### Configuration

Modify `config/config.yaml` to customize:

- Data source settings (symbol, date range)
- Model parameters (LSTM architecture, ARIMA settings, Prophet configuration)
- Training parameters (epochs, batch size, learning rate)
- Anomaly detection settings
- Visualization preferences

Example configuration:

```yaml
data:
  symbol: "TSLA"
  start_date: "2020-01-01"
  end_date: "2023-01-01"
  sequence_length: 30
  train_split: 0.8
  validation_split: 0.1

models:
  lstm:
    hidden_size: 50
    num_layers: 2
    dropout: 0.2
    learning_rate: 0.001
    batch_size: 32
    epochs: 100
    early_stopping_patience: 10
```

## Models

### LSTM (Long Short-Term Memory)

- **Architecture**: Multi-layer LSTM with dropout regularization
- **Features**: Early stopping, learning rate scheduling, GPU support
- **Use Case**: Complex time series with long-term dependencies

### ARIMA (AutoRegressive Integrated Moving Average)

- **Features**: Automatic parameter selection using pmdarima
- **Seasonality**: Support for seasonal ARIMA (SARIMA)
- **Use Case**: Traditional time series forecasting

### Prophet

- **Features**: Automatic seasonality detection, holiday effects
- **Robustness**: Handles missing data and outliers well
- **Use Case**: Business time series with clear seasonality

### Anomaly Detection

- **Isolation Forest**: Unsupervised anomaly detection
- **Autoencoder**: Deep learning-based reconstruction error
- **Statistical**: Z-score and modified Z-score methods

## Data Sources

### Stock Data
- Yahoo Finance integration via yfinance
- Support for any stock symbol
- Configurable date ranges

### Synthetic Data
- Configurable trend, seasonality, and noise
- Useful for testing and demonstration
- Realistic time series patterns

### CSV Upload
- Support for custom CSV files
- Flexible date format handling
- Automatic data type detection

## Visualization

The project includes comprehensive visualizations:

- **Time Series Plots**: Original data with trend analysis
- **Distribution Analysis**: Histograms, box plots, statistical summaries
- **Rolling Statistics**: Moving averages, standard deviations
- **Model Comparisons**: Side-by-side prediction comparisons
- **Anomaly Detection**: Highlighted anomalies on time series
- **Performance Metrics**: MSE, MAE, RMSE, MAPE comparisons

## Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_modules.py::TestLSTMModel -v
```

Test coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Logging

The project uses structured logging with:

- **File Logging**: Saves to `logs/timeseries.log`
- **Console Output**: Real-time progress updates
- **Configurable Levels**: INFO, DEBUG, WARNING, ERROR
- **Structured Format**: Timestamp, module, level, message

## Performance Optimization

### GPU Support
- Automatic GPU detection and usage
- PyTorch CUDA integration
- Fallback to CPU if GPU unavailable

### Memory Management
- Efficient data loading with DataLoader
- Batch processing for large datasets
- Memory-efficient sequence creation

### Model Persistence
- Automatic model checkpointing
- Best model saving based on validation loss
- Model loading for inference

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Install PyTorch with CUDA support if using GPU
3. **Memory Issues**: Reduce batch size or sequence length
4. **Data Loading**: Check internet connection for stock data

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor training progress:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all classes and methods
- Use meaningful variable names

### Testing Requirements

- Maintain test coverage above 80%
- Add tests for new features
- Update tests when modifying existing code

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Facebook Research for Prophet
- scikit-learn team for machine learning utilities
- Streamlit team for the web framework
- Yahoo Finance for financial data

## Citation

If you use this project in your research, please cite:

```bibtex
@software{timeseries_analysis_2024,
  title={Time Series Analysis with LSTM, ARIMA, Prophet, and Anomaly Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/timeseries-analysis}
}
```

**Note**: This project is for educational and research purposes. Always verify results and consider the limitations of time series forecasting in real-world applications.


# Time-Series-Analysis-with-LSTM-ARIMA-Prophet-and-Anomaly-Detection
