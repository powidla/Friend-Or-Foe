'''
Friend-Or-Foe: A collection of microbial datasets obtained from metabolic modeling.

This package provides easy access to microbial interaction datasets for machine learning research,
along with utilities and model implementations for predictive modeling of microbial interactions.
'''

__version__ = "1.0.0"

# Data utilities
from .data.loader import FriendOrFoeDataLoader

# Model classes  
from .model.base import (
    BaseModel,
    TabNetModel, 
    XGBoostModel,
    LightGBMModel, 
    CatBoostModel,
    FTTransformerModel,
    TabMModel
)

# Convenience functions
from .data.loader import quick_load, list_all_datasets

__all__ = [
    # Core classes
    "FriendOrFoeDataLoader",
    "BaseModel",
    
    # Model implementations
    "TabNetModel",
    "XGBoostModel", 
    "LightGBMModel",
    "CatBoostModel",
    "FTTransformerModel", 
    "TabMModel",
    
    # Utility functions
    "quick_load",
    "list_all_datasets",
    
    # Package info
    "__version__",
]
