"""
mllib - A Machine Learning Library for Learning and Experimentation

This library implements various machine learning algorithms from scratch
to help understand their inner workings.

Available models:
---------------
KNearestNeighbors : K-Nearest Neighbors algorithm for classification and regression

Base classes:
------------
BaseModel : Abstract base class that all models inherit from
"""

from .base import BaseModel
from .knn import KNearestNeighbors

__all__ = [
    'BaseModel',
    'KNearestNeighbors',
]
