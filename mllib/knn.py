import numpy as np
from typing import Union, Literal
from collections import Counter
from .base import BaseModel

class KNearestNeighbors(BaseModel):
    """
    K-Nearest Neighbors algorithm implementation supporting both classification and regression.
    
    Parameters:
    -----------
    k : int
        Number of neighbors to use for prediction
    metric : str, optional (default='euclidean')
        Distance metric to use. Options: 'euclidean', 'manhattan'
    weights : str, optional (default='uniform')
        Weight function used in prediction. Options: 'uniform', 'distance'
    """
    
    def __init__(
        self, 
        k: int = 5, 
        metric: Literal['euclidean', 'manhattan'] = 'euclidean',
        weights: Literal['uniform', 'distance'] = 'uniform'
    ):
        if k < 1:
            raise ValueError("k must be greater than 0")
            
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
    def _validate_data(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """Validate input data"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
            
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if len(y) != len(X):
                raise ValueError("X and y must have the same number of samples")
                
    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate distances between X and training data"""
        if self.metric == 'euclidean':
            # Efficient computation of Euclidean distances using matrix operations
            # Uses the formula: (a-b)^2 = a^2 + b^2 - 2ab
            a2 = np.sum(X**2, axis=1).reshape(-1, 1)
            b2 = np.sum(self.X_train**2, axis=1)
            ab = X @ self.X_train.T
            distances = np.sqrt(a2 + b2 - 2*ab)
        else:  # manhattan
            # Manhattan distance using broadcasting
            distances = np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)
            
        return distances
    
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        """Calculate weights for predictions based on distances"""
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        else:  # 'distance'
            # Add small epsilon to avoid division by zero
            return 1 / (distances + 1e-10)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """
        Fit the k-nearest neighbors classifier/regressor.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : KNearestNeighbors
            The fitted classifier
        """
        self._validate_data(X, y)
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted target values
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Fit the model before making predictions")
            
        self._validate_data(X)
        distances = self._calculate_distances(X)
        weights = self._get_weights(distances)
        
        # Get indices of k nearest neighbors
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        
        # If target is categorical (classification)
        if np.issubdtype(self.y_train.dtype, np.number) and len(np.unique(self.y_train)) > 100:
            # Regression case - weighted average
            neighbor_weights = np.take_along_axis(weights, k_indices, axis=1)
            neighbor_targets = self.y_train[k_indices]
            return np.sum(neighbor_weights * neighbor_targets, axis=1) / np.sum(neighbor_weights, axis=1)
        else:
            # Classification case - weighted voting
            predictions = []
            for idx, w in zip(k_indices, weights):
                neighbor_labels = self.y_train[idx]
                neighbor_weights = w[idx]
                # Get weighted votes for each class
                class_weights = {}
                for label, weight in zip(neighbor_labels, neighbor_weights):
                    class_weights[label] = class_weights.get(label, 0) + weight
                # Select class with highest weighted votes
                predictions.append(max(class_weights.items(), key=lambda x: x[1])[0])
            return np.array(predictions)
