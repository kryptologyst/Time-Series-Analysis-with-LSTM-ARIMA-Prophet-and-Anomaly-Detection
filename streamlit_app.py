"""
Streamlit interface for time series analysis.

This module provides an interactive web interface for exploring
time series data, training models, and visualizing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import DataLoader, DataPreprocessor, DataAnalyzer
from lstm_model import LSTMModel, LSTMTrainer, TimeSeriesDataProcessor
from forecasting_models import ARIMAModel, ProphetModel, ModelComparator, StatisticalModels
from anomaly_detection import AnomalyDetector, AnomalyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file."""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Using default settings.")
        return {
            'data': {'symbol': 'TSLA', 'start_date': '2020-01-01', 'end_date': '2023-01-01'},
            'models': {'lstm': {'hidden_size': 50, 'epochs': 100}},
            'visualization': {'figure_size': [12, 8]}
        }

def main():
    """Main Streamlit application."""
    st.title("📈 Time Series Analysis Dashboard")
    st.markdown("Comprehensive time series analysis with LSTM, ARIMA, Prophet, and anomaly detection")
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Stock Data", "Synthetic Data", "Upload CSV"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    
    # Data loading section
    st.header("📊 Data Loading")
    
    if data_source == "Stock Data":
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol", value=config['data']['symbol'])
        with col2:
            start_date = st.text_input("Start Date", value=config['data']['start_date'])
        with col3:
            end_date = st.text_input("End Date", value=config['data']['end_date'])
        
        if st.button("Load Stock Data"):
            with st.spinner("Loading stock data..."):
                try:
                    data = DataLoader.load_stock_data(symbol, start_date, end_date)
                    st.session_state.data = data
                    st.success(f"Loaded {len(data)} data points for {symbol}")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
    
    elif data_source == "Synthetic Data":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_points = st.number_input("Number of Points", min_value=100, max_value=5000, value=1000)
        with col2:
            trend = st.number_input("Trend", min_value=-0.01, max_value=0.01, value=0.001, step=0.0001)
        with col3:
            seasonality_period = st.number_input("Seasonality Period", min_value=7, max_value=365, value=30)
        
        col4, col5 = st.columns(2)
        with col4:
            seasonality_amplitude = st.number_input("Seasonality Amplitude", min_value=0.0, max_value=2.0, value=0.5)
        with col5:
            noise_level = st.number_input("Noise Level", min_value=0.0, max_value=1.0, value=0.1)
        
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                try:
                    data = DataLoader.generate_synthetic_data(
                        n_points=n_points,
                        trend=trend,
                        seasonality_period=seasonality_period,
                        seasonality_amplitude=seasonality_amplitude,
                        noise_level=noise_level
                    )
                    st.session_state.data = data
                    st.success(f"Generated {len(data)} synthetic data points")
                except Exception as e:
                    st.error(f"Error generating data: {e}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                date_column = st.text_input("Date Column Name")
            with col2:
                value_column = st.text_input("Value Column Name")
            with col3:
                date_format = st.text_input("Date Format (optional)")
            
            if st.button("Load CSV Data") and date_column and value_column:
                with st.spinner("Loading CSV data..."):
                    try:
                        # Save uploaded file temporarily
                        with open("temp_data.csv", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        data = DataLoader.load_csv_data(
                            "temp_data.csv",
                            date_column,
                            value_column,
                            date_format if date_format else None
                        )
                        st.session_state.data = data
                        st.success(f"Loaded {len(data)} data points from CSV")
                        
                        # Clean up
                        os.remove("temp_data.csv")
                    except Exception as e:
                        st.error(f"Error loading CSV data: {e}")
    
    # Data analysis section
    if st.session_state.data is not None:
        st.header("📈 Data Analysis")
        
        data = st.session_state.data
        
        # Basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Statistics")
            stats = DataAnalyzer.basic_statistics(data)
            stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=1)
        ))
        fig.update_layout(
            title="Time Series Plot",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        with col1:
            # Histogram
            fig_hist = px.histogram(
                data,
                title="Distribution Histogram",
                nbins=50
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                y=data.values,
                title="Box Plot"
            )
            fig_box.update_layout(height=300)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Rolling statistics
        window = st.slider("Rolling Window Size", min_value=5, max_value=100, value=30)
        
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        fig_rolling = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Mean', 'Rolling Standard Deviation'),
            vertical_spacing=0.1
        )
        
        fig_rolling.add_trace(
            go.Scatter(x=data.index, y=data.values, mode='lines', name='Original', opacity=0.3),
            row=1, col=1
        )
        fig_rolling.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean.values, mode='lines', name='Rolling Mean'),
            row=1, col=1
        )
        fig_rolling.add_trace(
            go.Scatter(x=rolling_std.index, y=rolling_std.values, mode='lines', name='Rolling Std'),
            row=2, col=1
        )
        
        fig_rolling.update_layout(height=600, title="Rolling Statistics")
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Model training section
    if st.session_state.data is not None:
        st.header("🤖 Model Training")
        
        # Model selection
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["LSTM", "ARIMA", "Prophet"],
            default=["LSTM"]
        )
        
        # Training parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            sequence_length = st.number_input("Sequence Length", min_value=5, max_value=100, value=30)
        with col2:
            train_ratio = st.slider("Training Ratio", min_value=0.5, max_value=0.9, value=0.8)
        with col3:
            val_ratio = st.slider("Validation Ratio", min_value=0.05, max_value=0.3, value=0.1)
        
        if st.button("Train Models"):
            if not models_to_train:
                st.warning("Please select at least one model to train.")
            else:
                with st.spinner("Training models..."):
                    try:
                        data = st.session_state.data
                        
                        # Prepare data
                        preprocessor = DataPreprocessor()
                        scaled_data = preprocessor.fit_transform(data)
                        
                        # Split data
                        train_data, val_data, test_data = preprocessor.split_data(
                            scaled_data, train_ratio, val_ratio
                        )
                        
                        # Create sequences
                        X_train, y_train = preprocessor.create_sequences(train_data, sequence_length)
                        X_val, y_val = preprocessor.create_sequences(val_data, sequence_length)
                        X_test, y_test = preprocessor.create_sequences(test_data, sequence_length)
                        
                        # Train models
                        predictions = {}
                        
                        if "LSTM" in models_to_train:
                            st.write("Training LSTM model...")
                            
                            # LSTM model
                            model = LSTMModel(
                                input_size=1,
                                hidden_size=config['models']['lstm']['hidden_size'],
                                num_layers=2,
                                dropout=0.2
                            )
                            
                            trainer = LSTMTrainer(model, learning_rate=0.001)
                            
                            # Create data loaders
                            import torch
                            from torch.utils.data import DataLoader, TensorDataset
                            
                            train_dataset = TensorDataset(
                                torch.FloatTensor(X_train),
                                torch.FloatTensor(y_train)
                            )
                            val_dataset = TensorDataset(
                                torch.FloatTensor(X_val),
                                torch.FloatTensor(y_val)
                            )
                            
                            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                            
                            # Train
                            history = trainer.train(
                                train_loader,
                                val_loader,
                                epochs=config['models']['lstm']['epochs'],
                                early_stopping_patience=10
                            )
                            
                            # Make predictions
                            test_dataset = TensorDataset(
                                torch.FloatTensor(X_test),
                                torch.FloatTensor(y_test)
                            )
                            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                            
                            lstm_pred = trainer.predict(test_loader)
                            predictions['LSTM'] = preprocessor.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
                        
                        if "ARIMA" in models_to_train:
                            st.write("Training ARIMA model...")
                            
                            try:
                                arima_model = ARIMAModel()
                                arima_model.fit(data)
                                
                                n_periods = len(test_data)
                                arima_pred = arima_model.predict(n_periods)
                                predictions['ARIMA'] = arima_pred
                                
                            except Exception as e:
                                st.warning(f"ARIMA training failed: {e}")
                        
                        if "Prophet" in models_to_train:
                            st.write("Training Prophet model...")
                            
                            try:
                                prophet_model = ProphetModel()
                                
                                # Prepare data for Prophet
                                prophet_df = pd.DataFrame({
                                    'ds': data.index,
                                    'y': data.values
                                })
                                
                                prophet_model.fit(prophet_df)
                                
                                # Make future dataframe
                                future_df = prophet_model.make_future_dataframe(periods=len(test_data))
                                forecast = prophet_model.predict(future_df)
                                
                                # Extract predictions for test period
                                prophet_pred = forecast['yhat'].tail(len(test_data)).values
                                predictions['Prophet'] = prophet_pred
                                
                            except Exception as e:
                                st.warning(f"Prophet training failed: {e}")
                        
                        st.session_state.predictions = predictions
                        st.session_state.models_trained = True
                        st.success("Models trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error training models: {e}")
    
    # Results visualization
    if st.session_state.models_trained and st.session_state.predictions:
        st.header("📊 Results Visualization")
        
        data = st.session_state.data
        predictions = st.session_state.predictions
        
        # Get test data for comparison
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.fit_transform(data)
        train_data, val_data, test_data = preprocessor.split_data(scaled_data, 0.8, 0.1)
        
        # Create test dates
        test_dates = data.index[-len(test_data):]
        
        # Plot predictions
        fig = go.Figure()
        
        # Plot actual data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=test_dates[:len(pred)],
                y=pred,
                mode='lines',
                name=f'{model_name} Prediction',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model evaluation metrics
        st.subheader("Model Performance Metrics")
        
        metrics_data = []
        for model_name, pred in predictions.items():
            if len(pred) > 0:
                # Calculate metrics
                actual_values = data.values[-len(pred):]
                
                mse = np.mean((actual_values - pred) ** 2)
                mae = np.mean(np.abs(actual_values - pred))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - pred) / actual_values)) * 100
                
                metrics_data.append({
                    'Model': model_name,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE (%)': mape
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
    
    # Anomaly detection section
    if st.session_state.data is not None:
        st.header("🔍 Anomaly Detection")
        
        # Anomaly detection methods
        anomaly_methods = st.multiselect(
            "Select Anomaly Detection Methods",
            ["Isolation Forest", "Autoencoder", "Statistical"],
            default=["Isolation Forest"]
        )
        
        if st.button("Run Anomaly Detection"):
            if not anomaly_methods:
                st.warning("Please select at least one anomaly detection method.")
            else:
                with st.spinner("Running anomaly detection..."):
                    try:
                        data = st.session_state.data
                        
                        # Initialize anomaly analyzer
                        analyzer = AnomalyAnalyzer()
                        
                        # Add detectors
                        for method in anomaly_methods:
                            if method == "Isolation Forest":
                                detector = AnomalyDetector("isolation_forest")
                            elif method == "Autoencoder":
                                detector = AnomalyDetector("autoencoder")
                            elif method == "Statistical":
                                detector = AnomalyDetector("statistical")
                            
                            analyzer.add_detector(method, detector)
                        
                        # Fit detectors
                        analyzer.fit_all(data.values.reshape(-1, 1))
                        
                        # Detect anomalies
                        anomaly_predictions = analyzer.detect_all(data.values.reshape(-1, 1))
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Plot original data
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode='lines',
                            name='Original Data',
                            line=dict(color='blue', width=1)
                        ))
                        
                        # Plot anomalies
                        for method, pred in anomaly_predictions.items():
                            if pred is not None:
                                anomaly_mask = pred == -1
                                anomaly_data = data[anomaly_mask]
                                
                                fig.add_trace(go.Scatter(
                                    x=anomaly_data.index,
                                    y=anomaly_data.values,
                                    mode='markers',
                                    name=f'{method} Anomalies',
                                    marker=dict(color='red', size=8)
                                ))
                        
                        fig.update_layout(
                            title="Anomaly Detection Results",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate report
                        report = analyzer.generate_report(data, anomaly_predictions)
                        st.subheader("Anomaly Detection Report")
                        st.dataframe(report, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error running anomaly detection: {e}")

if __name__ == "__main__":
    main()
