"""
mllib - A Machine Learning Library for Learning and Experimentation

This library implements various machine learning algorithms from scratch
to help understand their inner workings.

Available models:
---------------
KNearestNeighbors : K-Nearest Neighbors algorithm for classification and regression
LinearRegression : Linear Regression using gradient descent optimization
NeuralNetwork : Feedforward Neural Network with customizable architecture

Base classes:
------------
BaseModel : Abstract base class that all models inherit from
"""

from .base import BaseModel
from .knn import KNearestNeighbors
from .linear_regression import LinearRegression
from .nn import NeuralNetwork, to_onehot

__all__ = [
    'BaseModel',
    'KNearestNeighbors',
    'LinearRegression',
    'NeuralNetwork',
    'to_onehot',
]
