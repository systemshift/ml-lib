from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Any

class BaseModel(ABC):
    """
    Abstract base class for all models in the mllib package.
    
    This class defines the common interface that all models must implement.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : BaseModel
            The fitted model
        """
        pass
    
    @abstractmethod
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
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the model's performance score.
        
        For classification: returns accuracy score
        For regression: returns R² score
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True target values
            
        Returns:
        --------
        score : float
            Model performance score
        """
        y_pred = self.predict(X)
        
        # If regression (numerical targets with many unique values)
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 100:
            # R² score for regression
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            # Accuracy for classification
            return np.mean(y_pred == y)
