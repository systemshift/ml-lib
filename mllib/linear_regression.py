import numpy as np
from typing import Optional, Tuple
from .base import BaseModel

class LinearRegression(BaseModel):
    """
    Linear Regression implementation using gradient descent optimization.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of gradient descent iterations
    batch_size : int, optional (default=None)
        Size of mini-batches for gradient descent
        If None, uses full batch gradient descent
    
    Attributes:
    -----------
    weights : np.ndarray
        Model weights (coefficients)
    bias : float
        Model bias (intercept)
    loss_history : list
        History of MSE loss during training
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: Optional[int] = None
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
        # For feature scaling
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize model parameters"""
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
    def _get_batch_indices(self, n_samples: int) -> np.ndarray:
        """Get random batch indices for mini-batch gradient descent"""
        if self.batch_size is None:
            return np.arange(n_samples)
        else:
            return np.random.choice(n_samples, self.batch_size, replace=False)
    
    def _scale_features(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scale features and target values"""
        if self.X_mean is None:
            # First time scaling (during fit)
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            if y is not None:
                self.y_mean = np.mean(y)
                self.y_std = np.std(y) + 1e-8
        
        # Scale X
        X_scaled = (X - self.X_mean) / self.X_std
        
        # Scale y if provided
        if y is not None:
            y_scaled = (y - self.y_mean) / self.y_std
            return X_scaled, y_scaled
        
        return X_scaled, None
    
    def _unscale_predictions(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Unscale predictions back to original scale"""
        return y_pred_scaled * self.y_std + self.y_mean
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions"""
        return X @ self.weights + self.bias
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Compute gradients for weights and bias"""
        m = len(X)
        error = y_pred - y
        
        # Gradients for weights and bias
        dw = (2/m) * (X.T @ error)
        db = (2/m) * np.sum(error)
        
        return dw, db
    
    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """Update model parameters using gradients"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LinearRegression
            Fitted model
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Validate input dimensions
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        # Scale features and target
        X_scaled, y_scaled = self._scale_features(X, y)
            
        # Initialize parameters
        self._initialize_parameters(X_scaled.shape[1])
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Get batch indices
            batch_idx = self._get_batch_indices(len(X_scaled))
            X_batch = X_scaled[batch_idx]
            y_batch = y_scaled[batch_idx]
            
            # Forward pass
            y_pred = self._forward(X_batch)
            
            # Compute loss
            mse = np.mean((y_pred - y_batch) ** 2)
            self.loss_history.append(mse)
            
            # Compute and apply gradients
            dw, db = self._compute_gradients(X_batch, y_batch, y_pred)
            self._update_parameters(dw, db)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} features but got {X.shape[1]}")
            
        # Scale features
        X_scaled, _ = self._scale_features(X)
        
        # Make predictions and unscale
        y_pred_scaled = self._forward(X_scaled)
        return self._unscale_predictions(y_pred_scaled)
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing model parameters:
            - 'weights': Feature coefficients (unscaled)
            - 'bias': Intercept term (unscaled)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before getting parameters")
            
        # Unscale weights and bias to original scale
        unscaled_weights = self.weights * (self.y_std / self.X_std)
        unscaled_bias = self.y_mean - np.sum(self.weights * self.X_mean / self.X_std) * self.y_std
            
        return {
            'weights': unscaled_weights,
            'bias': unscaled_bias
        }
