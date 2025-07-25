
"""
Base Model Classes and CLI Interface for Friend-Or-Foe Package
"""

# friend_or_foe/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import torch
import warnings


class BaseModel(ABC):
    """
    Abstract base class for all Friend-Or-Foe models.
    
    This class defines the interface that all models should implement.
    """
    
    def __init__(self, **kwargs):
        """Initialize the model with given parameters."""
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.DataFrame] = None) -> 'BaseModel':
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the given data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities as numpy array
        """
        raise NotImplementedError("predict_proba not implemented for this model")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, 
                task_type: str = "classification") -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        predictions = self.predict(X_test)
        
        if task_type.lower() == "classification":
            return self._classification_metrics(y_test, predictions, X_test)
        else:
            return self._regression_metrics(y_test, predictions)
    
    def _classification_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray, 
                              X: pd.DataFrame) -> Dict[str, float]:
        """Calculate classification metrics."""
        y_true_flat = y_true.values.flatten() if hasattr(y_true, 'values') else y_true
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, y_pred),
            'f1_score': f1_score(y_true_flat, y_pred, average='weighted'),
        }
        
        # Add AUC if we can get probabilities
        try:
            y_proba = self.predict_proba(X)
            if y_proba.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true_flat, y_proba[:, 1])
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true_flat, y_proba, multi_class='ovr')
        except (NotImplementedError, AttributeError, ValueError):
            pass
            
        return metrics
    
    def _regression_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        y_true_flat = y_true.values.flatten() if hasattr(y_true, 'values') else y_true
        
        return {
            'mse': mean_squared_error(y_true_flat, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred)),
            'r2': r2_score(y_true_flat, y_pred),
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        # Implementation depends on specific model type
        raise NotImplementedError("save_model must be implemented by subclasses")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        # Implementation depends on specific model type
        raise NotImplementedError("load_model must be implemented by subclasses")


class TabNetModel(BaseModel):
    """
    TabNet model implementation for Friend-Or-Foe datasets.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        self.TabNetClassifier = TabNetClassifier
        self.TabNetRegressor = TabNetRegressor
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'TabNetModel':
        """Fit TabNet model."""
        
        # Prepare data
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.flatten()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values.flatten()
            eval_set = [(X_val_np, y_val_np)]
        
        # Initialize model
        if task_type.lower() == "classification":
            self.model = self.TabNetClassifier(**self.model_params)
        else:
            self.model = self.TabNetRegressor(**self.model_params)
        
        # Train model
        self.model.fit(
            X_train_np, y_train_np,
            eval_set=eval_set,
            eval_name=['val'] if eval_set else None,
            eval_metric=['accuracy'] if task_type.lower() == "classification" else ['mse'],
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        self.is_fitted = True
        self.training_history = self.model.history
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with TabNet."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_np = X.values.astype(np.float32)
        return self.model.predict(X_np)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with TabNet (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        X_np = X.values.astype(np.float32)
        return self.model.predict_proba(X_np)
    
    def save_model(self, filepath: str):
        """Save TabNet model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load TabNet model."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_params = model_data['model_params']
        self.training_history = model_data.get('training_history', {})
        self.is_fitted = True


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for Friend-Or-Foe datasets.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import xgboost as xgb
        self.xgb = xgb
        
        # Set default parameters if not provided
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'XGBoostModel':
        """Fit XGBoost model."""
        
        # Prepare parameters based on task type
        if task_type.lower() == "classification":
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_train.values.flatten()))
            if n_classes == 2:
                self.model_params['objective'] = 'binary:logistic'
                self.model_params['eval_metric'] = 'logloss'
            else:
                self.model_params['objective'] = 'multi:softprob'
                self.model_params['eval_metric'] = 'mlogloss'
                self.model_params['num_class'] = n_classes
            
            self.model = self.xgb.XGBClassifier(**self.model_params)
        else:
            self.model_params['objective'] = 'reg:squarederror'
            self.model_params['eval_metric'] = 'rmse'
            self.model = self.xgb.XGBRegressor(**self.model_params)
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val.values.flatten())]
        
        # Fit model
        self.model.fit(
            X_train.values, 
            y_train.values.flatten(),
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with XGBoost (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        """Save XGBoost model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # XGBoost has built-in save functionality
        if filepath.endswith('.json'):
            self.model.save_model(filepath)
        else:
            # Use pickle for compatibility
            import pickle
            model_data = {
                'model': self.model,
                'model_params': self.model_params
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load XGBoost model."""
        if filepath.endswith('.json'):
            # Load using XGBoost's native format
            # Need to recreate model with correct type
            if 'Classifier' in str(type(self.model)) or self.model_params.get('objective', '').startswith('binary') or self.model_params.get('objective', '').startswith('multi'):
                self.model = self.xgb.XGBClassifier()
            else:
                self.model = self.xgb.XGBRegressor()
            self.model.load_model(filepath)
        else:
            # Load using pickle
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.model_params = model_data['model_params']
        
        self.is_fitted = True


class LightGBMModel(BaseModel):
    """
    LightGBM model implementation for Friend-Or-Foe datasets.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lightgbm as lgb
        self.lgb = lgb
        
        # Set default parameters
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'LightGBMModel':
        """Fit LightGBM model."""
        
        # Prepare parameters based on task type
        if task_type.lower() == "classification":
            n_classes = len(np.unique(y_train.values.flatten()))
            if n_classes == 2:
                self.model_params['objective'] = 'binary'
                self.model_params['metric'] = 'binary_logloss'
            else:
                self.model_params['objective'] = 'multiclass'
                self.model_params['metric'] = 'multi_logloss'
                self.model_params['num_class'] = n_classes
            
            self.model = self.lgb.LGBMClassifier(**self.model_params)
        else:
            self.model_params['objective'] = 'regression'
            self.model_params['metric'] = 'rmse'
            self.model = self.lgb.LGBMRegressor(**self.model_params)
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val.values.flatten())]
        
        # Fit model
        self.model.fit(
            X_train.values,
            y_train.values.flatten(),
            eval_set=eval_set,
            callbacks=[self.lgb.early_stopping(50), self.lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with LightGBM (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        """Save LightGBM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # LightGBM has built-in save functionality
        if filepath.endswith('.txt'):
            self.model.booster_.save_model(filepath)
        else:
            # Use pickle for compatibility
            import pickle
            model_data = {
                'model': self.model,
                'model_params': self.model_params
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load LightGBM model."""
        if filepath.endswith('.txt'):
            # Load using LightGBM's native format
            if 'Classifier' in str(type(self.model)) or self.model_params.get('objective') in ['binary', 'multiclass']:
                self.model = self.lgb.LGBMClassifier()
            else:
                self.model = self.lgb.LGBMRegressor()
            self.model = self.lgb.Booster(model_file=filepath)
        else:
            # Load using pickle
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.model_params = model_data['model_params']
        
        self.is_fitted = True


class CatBoostModel(BaseModel):
    """
    CatBoost model implementation for Friend-Or-Foe datasets.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import catboost as cb
        self.cb = cb
        
        # Set default parameters
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False,
            'random_seed': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'CatBoostModel':
        """Fit CatBoost model."""
        
        if task_type.lower() == "classification":
            self.model = self.cb.CatBoostClassifier(**self.model_params)
        else:
            self.model = self.cb.CatBoostRegressor(**self.model_params)
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val.values, y_val.values.flatten())
        
        # Fit model
        self.model.fit(
            X_train.values,
            y_train.values.flatten(),
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with CatBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with CatBoost (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        """Save CatBoost model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # CatBoost has built-in save functionality
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load CatBoost model."""
        # CatBoost requires us to know the model type beforehand
        # Try to infer from filepath or use a default
        try:
            self.model = self.cb.CatBoostClassifier()
            self.model.load_model(filepath)
        except:
            try:
                self.model = self.cb.CatBoostRegressor()
                self.model.load_model(filepath)
            except Exception as e:
                raise ValueError(f"Could not load CatBoost model: {e}")
        
        self.is_fitted = True
