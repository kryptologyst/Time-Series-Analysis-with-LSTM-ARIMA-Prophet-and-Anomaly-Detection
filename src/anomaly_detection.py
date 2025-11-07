"""
Anomaly detection methods for time series data.

This module provides various anomaly detection techniques including
Isolation Forest, Autoencoders, and statistical methods.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AutoencoderAnomalyDetector(nn.Module):
    """
    Autoencoder-based anomaly detector for time series data.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (encoded, decoded) tensors
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to original space."""
        return self.decoder(x)


class AnomalyDetector:
    """
    Comprehensive anomaly detection system.
    """
    
    def __init__(self, method: str = "isolation_forest"):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'autoencoder', 'statistical')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare data for anomaly detection.
        
        Args:
            data: Input data
            
        Returns:
            Prepared data
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    
    def fit(self, data: np.ndarray, **kwargs) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            data: Training data
            **kwargs: Additional arguments for the model
            
        Returns:
            Self for method chaining
        """
        data = self._prepare_data(data)
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=kwargs.get('random_state', 42),
                **kwargs
            )
            self.model.fit(data)
            
        elif self.method == "autoencoder":
            # Scale data
            scaled_data = self.scaler.fit_transform(data)
            
            # Convert to PyTorch tensors
            tensor_data = torch.FloatTensor(scaled_data)
            
            # Initialize autoencoder
            self.model = AutoencoderAnomalyDetector(
                input_dim=data.shape[1],
                encoding_dim=kwargs.get('encoding_dim', 10),
                hidden_dims=kwargs.get('hidden_dims', [64, 32])
            )
            
            # Training parameters
            learning_rate = kwargs.get('learning_rate', 0.001)
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            
            # Train autoencoder
            self._train_autoencoder(tensor_data, learning_rate, epochs, batch_size)
            
        elif self.method == "statistical":
            # Statistical method using Z-score
            self.model = "statistical"
            self.scaler.fit(data)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.is_fitted = True
        logger.info(f"Anomaly detector fitted using {self.method} method")
        
        return self
    
    def _train_autoencoder(
        self,
        data: torch.Tensor,
        learning_rate: float,
        epochs: int,
        batch_size: int
    ) -> None:
        """
        Train autoencoder model.
        
        Args:
            data: Training data
            learning_rate: Learning rate
            epochs: Number of epochs
            batch_size: Batch size
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, in dataloader:
                optimizer.zero_grad()
                _, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.6f}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Anomaly scores (-1 for anomalies, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        data = self._prepare_data(data)
        
        if self.method == "isolation_forest":
            return self.model.predict(data)
            
        elif self.method == "autoencoder":
            # Scale data
            scaled_data = self.scaler.transform(data)
            tensor_data = torch.FloatTensor(scaled_data)
            
            self.model.eval()
            with torch.no_grad():
                _, decoded = self.model(tensor_data)
                reconstruction_error = torch.mean((tensor_data - decoded) ** 2, dim=1)
                
            # Convert reconstruction error to anomaly scores
            threshold = np.percentile(reconstruction_error.numpy(), 90)  # 90th percentile as threshold
            anomaly_scores = np.where(reconstruction_error.numpy() > threshold, -1, 1)
            return anomaly_scores
            
        elif self.method == "statistical":
            # Z-score based detection
            scaled_data = self.scaler.transform(data)
            z_scores = np.abs(scaled_data)
            threshold = 3  # 3-sigma rule
            anomaly_scores = np.where(np.any(z_scores > threshold, axis=1), -1, 1)
            return anomaly_scores
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores.
        
        Args:
            data: Data to analyze
            
        Returns:
            Continuous anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        data = self._prepare_data(data)
        
        if self.method == "isolation_forest":
            return self.model.decision_function(data)
            
        elif self.method == "autoencoder":
            scaled_data = self.scaler.transform(data)
            tensor_data = torch.FloatTensor(scaled_data)
            
            self.model.eval()
            with torch.no_grad():
                _, decoded = self.model(tensor_data)
                reconstruction_error = torch.mean((tensor_data - decoded) ** 2, dim=1)
                
            return reconstruction_error.numpy()
            
        elif self.method == "statistical":
            scaled_data = self.scaler.transform(data)
            z_scores = np.abs(scaled_data)
            return np.max(z_scores, axis=1)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")


class AnomalyAnalyzer:
    """
    Comprehensive anomaly analysis and visualization.
    """
    
    def __init__(self):
        """Initialize the anomaly analyzer."""
        self.detectors = {}
        self.results = {}
        
    def add_detector(self, name: str, detector: AnomalyDetector) -> None:
        """
        Add an anomaly detector.
        
        Args:
            name: Detector name
            detector: Detector instance
        """
        self.detectors[name] = detector
        
    def fit_all(self, data: np.ndarray, **kwargs) -> None:
        """
        Fit all detectors.
        
        Args:
            data: Training data
            **kwargs: Additional arguments
        """
        for name, detector in self.detectors.items():
            try:
                logger.info(f"Fitting {name} detector...")
                detector.fit(data, **kwargs)
                self.results[name] = {'detector': detector, 'fitted': True}
                logger.info(f"{name} detector fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting {name} detector: {e}")
                self.results[name] = {'detector': detector, 'fitted': False, 'error': str(e)}
    
    def detect_all(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run anomaly detection with all fitted detectors.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary of anomaly predictions
        """
        predictions = {}
        
        for name, result in self.results.items():
            if not result.get('fitted', False):
                continue
                
            try:
                detector = result['detector']
                predictions[name] = detector.predict(data)
            except Exception as e:
                logger.error(f"Error detecting anomalies with {name}: {e}")
                predictions[name] = None
                
        return predictions
    
    def get_scores_all(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get anomaly scores from all fitted detectors.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary of anomaly scores
        """
        scores = {}
        
        for name, result in self.results.items():
            if not result.get('fitted', False):
                continue
                
            try:
                detector = result['detector']
                scores[name] = detector.get_anomaly_scores(data)
            except Exception as e:
                logger.error(f"Error getting scores from {name}: {e}")
                scores[name] = None
                
        return scores
    
    def plot_anomalies(
        self,
        data: pd.Series,
        predictions: Dict[str, np.ndarray],
        title: str = "Anomaly Detection Results",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot anomaly detection results.
        
        Args:
            data: Time series data
            predictions: Dictionary of anomaly predictions
            title: Plot title
            save_path: Path to save the plot
        """
        n_detectors = len([p for p in predictions.values() if p is not None])
        fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(15, 4 * (n_detectors + 1)))
        
        if n_detectors == 0:
            axes = [axes]
        
        # Plot original data
        axes[0].plot(data.index, data.values, color='blue', alpha=0.7)
        axes[0].set_title(f"{title} - Original Data")
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Plot anomaly detection results
        plot_idx = 1
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        
        for i, (name, pred) in enumerate(predictions.items()):
            if pred is None:
                continue
                
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
            
            axes[plot_idx].set_title(f"{name} - Anomaly Detection")
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.xlabel('Time')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scores(
        self,
        data: pd.Series,
        scores: Dict[str, np.ndarray],
        title: str = "Anomaly Scores",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot anomaly scores.
        
        Args:
            data: Time series data
            scores: Dictionary of anomaly scores
            title: Plot title
            save_path: Path to save the plot
        """
        n_detectors = len([s for s in scores.values() if s is not None])
        fig, axes = plt.subplots(n_detectors, 1, figsize=(15, 4 * n_detectors))
        
        if n_detectors == 1:
            axes = [axes]
        
        plot_idx = 0
        for name, score in scores.items():
            if score is None:
                continue
                
            axes[plot_idx].plot(data.index, score, color='red', alpha=0.7)
            axes[plot_idx].set_title(f"{name} - Anomaly Scores")
            axes[plot_idx].set_ylabel('Score')
            axes[plot_idx].grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.xlabel('Time')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(
        self,
        data: pd.Series,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Generate anomaly detection report.
        
        Args:
            data: Time series data
            predictions: Dictionary of anomaly predictions
            
        Returns:
            Report DataFrame
        """
        report_data = []
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            n_anomalies = np.sum(pred == -1)
            n_normal = np.sum(pred == 1)
            anomaly_rate = n_anomalies / len(pred) * 100
            
            report_data.append({
                'Detector': name,
                'Total Points': len(pred),
                'Anomalies': n_anomalies,
                'Normal': n_normal,
                'Anomaly Rate (%)': anomaly_rate
            })
            
        return pd.DataFrame(report_data)
