import numpy as np
from typing import List, Tuple, Literal, Optional
from .base import BaseModel

class Activation:
    """Neural Network activation functions and their derivatives"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

class Layer:
    """
    Neural Network Layer
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    output_size : int
        Number of neurons in the layer
    activation : str
        Activation function ('relu', 'sigmoid', or 'tanh')
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Literal['relu', 'sigmoid', 'tanh'] = 'relu'
    ):
        # Initialize weights with He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros(output_size)
        
        # Set activation function and its derivative
        if activation == 'relu':
            self.activation_fn = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation_fn = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        else:  # tanh
            self.activation_fn = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
            
        # For storing intermediate values
        self.input = None
        self.z = None
        self.output = None
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        self.input = X
        self.z = X @ self.weights + self.bias
        self.output = self.activation_fn(self.z)
        return self.output
    
    def backward(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass through the layer"""
        # Calculate gradients
        delta = delta * self.activation_derivative(self.z)
        dW = self.input.T @ delta
        db = np.sum(delta, axis=0)
        # Propagate error to previous layer
        delta_prev = delta @ self.weights.T
        return delta_prev, dW, db

class NeuralNetwork(BaseModel):
    """
    Neural Network implementation with customizable architecture
    
    Parameters:
    -----------
    layer_sizes : List[int]
        List of layer sizes (including input and output layers)
    activations : List[str]
        List of activation functions for each layer (except input)
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of training iterations
    batch_size : int, optional
        Size of mini-batches (None for full batch)
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: Optional[int] = None
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Must have at least input and output layers")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Must provide activation for each layer except input")
            
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.loss_history = []
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            )
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Complete forward pass through the network"""
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def _backward_pass(self, delta: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Complete backward pass through the network"""
        gradients = []
        for layer in reversed(self.layers):
            delta, dW, db = layer.backward(delta)
            gradients.append((dW, db))
        return list(reversed(gradients))
    
    def _update_parameters(self, gradients: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Update network parameters using gradients"""
        for layer, (dW, db) in zip(self.layers, gradients):
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db
    
    def _get_batch_indices(self, n_samples: int) -> np.ndarray:
        """Get random batch indices"""
        if self.batch_size is None:
            return np.arange(n_samples)
        return np.random.choice(n_samples, self.batch_size, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetwork':
        """
        Train the neural network
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples, n_outputs)
            Target values (should be one-hot encoded for classification)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Ensure y is 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Validate dimensions
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(y.shape) != 2:
            raise ValueError("y must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
            
        # Training loop
        for _ in range(self.n_iterations):
            batch_idx = self._get_batch_indices(len(X))
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Forward pass
            predictions = self._forward_pass(X_batch)
            
            # Compute loss (MSE)
            loss = np.mean((predictions - y_batch) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            output_delta = 2 * (predictions - y_batch) / len(batch_idx)
            gradients = self._backward_pass(output_delta)
            
            # Update parameters
            self._update_parameters(gradients)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples, n_outputs)
            Predicted values
        """
        X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        return self._forward_pass(X)

def to_onehot(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """Convert integer labels to one-hot encoded format"""
    if n_classes is None:
        n_classes = len(np.unique(y))
    return np.eye(n_classes)[y]
