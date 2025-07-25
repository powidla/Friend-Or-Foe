"""
Friend-Or-Foe: A collection of microbial datasets obtained from metabolic modeling.

This package provides easy access to microbial interaction datasets for machine learning research,
along with utilities and model implementations for predictive modeling of microbial interactions.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data.loader import FriendOrFoeDataLoader
from .models.base import BaseModel

__all__ = [
    "FriendOrFoeDataLoader",
    "BaseModel",
    "__version__",
]
