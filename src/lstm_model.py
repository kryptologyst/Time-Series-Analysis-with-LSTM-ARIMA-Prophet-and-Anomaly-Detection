"""
Modern LSTM implementation for time series forecasting.

This module provides a comprehensive LSTM model with proper type hints,
error handling, and modern PyTorch practices.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        output_size: Number of output features (default: 1)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TimeSeriesDataProcessor:
    """
    Data processor for time series data with scaling and sequence creation.
    """
    
    def __init__(self, sequence_length: int = 30, scaler: Optional[MinMaxScaler] = None):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Length of input sequences
            scaler: Pre-fitted scaler (optional)
        """
        self.sequence_length = sequence_length
        self.scaler = scaler or MinMaxScaler()
        self.is_fitted = scaler is not None
        
    def create_sequences(
        self, 
        data: np.ndarray, 
        target_col: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data array
            target_col: Column index for target variable (if None, uses last column)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if target_col is None:
            target_col = data.shape[1] - 1
            
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])
            
        return np.array(X), np.array(y)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            data: Input data array
            
        Returns:
            Scaled data array
        """
        if not self.is_fitted:
            scaled_data = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            scaled_data = self.scaler.transform(data)
            
        return scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data array
            
        Returns:
            Original scale data array
        """
        return self.scaler.inverse_transform(data)


class LSTMTrainer:
    """
    Trainer class for LSTM models with comprehensive training and evaluation.
    """
    
    def __init__(
        self,
        model: LSTMModel,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: LSTM model to train
            device: Device to use for training (CPU/GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'models/best_lstm_model.pt')
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Predictions array
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
                
        return np.concatenate(predictions, axis=0)
    
    def evaluate(
        self, 
        data_loader: DataLoader, 
        scaler: MinMaxScaler
    ) -> Dict[str, float]:
        """
        Evaluate the model with multiple metrics.
        
        Args:
            data_loader: Data loader for evaluation
            scaler: Scaler for inverse transformation
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(data_loader)
        
        # Get true values
        true_values = []
        for _, batch_y in data_loader:
            true_values.append(batch_y.numpy())
        true_values = np.concatenate(true_values, axis=0)
        
        # Inverse transform
        predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1))
        true_values_unscaled = scaler.inverse_transform(true_values.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(true_values_unscaled, predictions_unscaled)
        mae = mean_absolute_error(true_values_unscaled, predictions_unscaled)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true_values_unscaled - predictions_unscaled) / true_values_unscaled)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }


def plot_training_history(history: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    if 'val_losses' in history and history['val_losses']:
        ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot predictions vs actual
    ax2.set_title('Predictions vs Actual')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "LSTM Predictions vs Actual",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predictions against actual values.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
