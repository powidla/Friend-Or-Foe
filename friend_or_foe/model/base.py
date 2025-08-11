
"""
Friend-Or-Foe package base functions.
Currently supports loading data from hugging face repo: https://huggingface.co/datasets/powidla/Friend-Or-Foe
The following models can be called via abstract class: 
- GBDTs: xbg, lghtgbm, catboost
- NNs: TabNet, FT, TabM
In additon for GBDTs it is possible to perform SHAP analysis and plot basic graphs (drop, waterfall).
"""

# friend_or_foe/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import torch
import pickle
import warnings
import shap
import matplotlib.pyplot as plt
import catboost as cb
from rtdl_revisiting_models import FTTransformer
from .tabm.model import Model


class BaseModel(ABC):
    '''
    Abstract base class for all Friend-Or-Foe models.
    
    This class defines all models .
    '''
    
    def __init__(self, **kwargs):
        '''
        Initialize the model with given parameters.
        '''
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.DataFrame] = None) -> 'BaseModel':
        '''
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Outputs:
            Self for method chaining
        '''
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Make predictions on the given data.
        
        Args:
            X: Features to predict on
            
        Outputs:
            Predictions as numpy array
        '''
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict class probabilities (for classification models).
        
        Args:
            X: Features to predict on
            
        Outputs:
            Class probabilities as numpy array
        '''
        raise NotImplementedError("predict_proba not implemented for this model")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, 
                task_type: str = "classification") -> Dict[str, float]:
        '''
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            task_type: Type of task ('classification' or 'regression')
            
        Outputs:
            Dictionary of evaluation metrics
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        predictions = self.predict(X_test)
        
        if task_type.lower() == "classification":
            return self._classification_metrics(y_test, predictions, X_test)
        else:
            return self._regression_metrics(y_test, predictions)
    
    def _classification_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray, 
                              X: pd.DataFrame) -> Dict[str, float]:
        '''
        Calculate classification metrics.
        '''
        y_true_flat = y_true.values.flatten() if hasattr(y_true, 'values') else y_true
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, y_pred),
            'f1_score': f1_score(y_true_flat, y_pred, average='weighted'),
        }
        
        # Add AUC 
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
        '''
        Calculate regression metrics.
        '''
        y_true_flat = y_true.values.flatten() if hasattr(y_true, 'values') else y_true
        
        return {
            'mse': mean_squared_error(y_true_flat, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred)),
            'r2': r2_score(y_true_flat, y_pred),
        }
    
    def save_model(self, filepath: str):
        '''
        Save the trained model.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        # Implementation depends on specific model type
        raise NotImplementedError("save_model must be implemented by subclasses")
    
    def load_model(self, filepath: str):
        '''
        Load a trained model.
        '''
        # Implementation depends on specific model type
        raise NotImplementedError("load_model must be implemented by subclasses")


class TabNetModel(BaseModel):
    '''
    TabNet model implementation for Friend-Or-Foe datasets.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        self.TabNetClassifier = TabNetClassifier
        self.TabNetRegressor = TabNetRegressor
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'TabNetModel':
        
        # Prepare data
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values
        if task_type.lower() == "regression":
            # Ensure 2D for regression
            if y_train_np.ndim == 1:
                y_train_np = y_train_np.reshape(-1, 1)
        else:
            y_train_np = y_train_np.flatten()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values
            if task_type.lower() == "regression":
                if y_val_np.ndim == 1:
                    y_val_np = y_val_np.reshape(-1, 1)
            else:
                y_val_np = y_val_np.flatten()
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
        '''
        Make predictions with TabNet.
        '''
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        import threading
        
        # Use threading lock to prevent conflicts
        lock = threading.Lock()
        with lock:
            model_data = {
                'model': self.model,
                'model_params': self.model_params,
                'training_history': self.training_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        '''
        Load TabNet model.
        '''
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_params = model_data['model_params']
        self.training_history = model_data.get('training_history', {})
        self.is_fitted = True


class XGBoostModel(BaseModel):
    '''
    XGBoost model implementation for Friend-Or-Foe datasets.
    '''
    
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
        '''
        Fit XGBoost model.
        '''
        
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
        '''
        Make predictions with XGBoost.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict probabilities with XGBoost (classification only).
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        '''
        Save XGBoost model.
        '''
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
        '''
        Load XGBoost model.
        '''
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
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        '''
        Get feature importance from XGBoost.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def shap_analysis(self, X_background: pd.DataFrame, X_explain: pd.DataFrame, 
                     plot_type: str = "summary", max_display: int = 20, 
                     save_path: Optional[str] = None) -> Dict[str, Any]:
        '''
        Perform SHAP analysis for XGBoost model.
        
        Args:
            X_background: Background dataset for SHAP explainer
            X_explain: Dataset to explain
            plot_type: Type of SHAP plot ('summary', 'waterfall', 'force', 'dependence')
            max_display: Maximum number of features to display
            save_path: Path to save the plot
            
        Outputs:
            Dictionary containing SHAP values and explainer
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before SHAP analysis")
        
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib are required for explainability analysis. Install with: pip install shap matplotlib")
        
        # Create SHAP explainer for XGBoost
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_explain.values)
        
        # Handle binary vs multiclass
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Multiclass - use the positive class for binary or first class for multiclass
            shap_values_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_values_plot = shap_values
        
        # Create plots based on type
        if plot_type == "summary":
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_explain.values, 
                            feature_names=X_explain.columns, 
                            max_display=max_display, show=False)
            if save_path:
                plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        elif plot_type == "waterfall":
            if len(X_explain) > 0:
                shap.waterfall_plot(explainer.expected_value, shap_values_plot[0], 
                                  X_explain.iloc[0], feature_names=X_explain.columns,
                                  max_display=max_display, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_waterfall.png", dpi=300, bbox_inches='tight')
                plt.show()
                
        elif plot_type == "force":
            if len(X_explain) > 0:
                shap.force_plot(explainer.expected_value, shap_values_plot[0], 
                              X_explain.iloc[0], feature_names=X_explain.columns,
                              matplotlib=True, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_force.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Calculate feature importance from SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_explain.columns,
            'shap_importance': np.abs(shap_values_plot).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'expected_value': explainer.expected_value
        }


class LightGBMModel(BaseModel):
    '''
    LightGBM model implementation for Friend-Or-Foe datasets.
    '''
    
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
        '''
        Fit LightGBM model.
        '''
        
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
        
        # val
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val.values.flatten())]
            
        callbacks = [self.lgb.log_evaluation(0)]
        if eval_set is not None:
            callbacks.append(self.lgb.early_stopping(50))
        # fit
        self.model.fit(
            X_train.values,
            y_train.values.flatten(),
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Make predictions with LightGBM.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict probabilities with LightGBM (classification only).
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        '''
        Save LightGBM model.
        '''
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
        '''
        Load LightGBM model.
        '''
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
    def get_feature_importance(self, importance_type: str = "split") -> pd.DataFrame:
        '''
        Get feature importance from LightGBM.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def shap_analysis(self, X_background: pd.DataFrame, X_explain: pd.DataFrame, 
                     plot_type: str = "summary", max_display: int = 20, 
                     save_path: Optional[str] = None) -> Dict[str, Any]:
        '''
        Perform SHAP analysis for LightGBM model.
        
        Args:
            X_background: Background dataset for SHAP explainer
            X_explain: Dataset to explain
            plot_type: Type of SHAP plot ('summary', 'waterfall', 'force', 'dependence')
            max_display: Maximum number of features to display
            save_path: Path to save the plot
            
        Outputs:
            Dictionary containing SHAP values and explainer
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before SHAP analysis")
        
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib are required for explainability analysis. Install with: pip install shap matplotlib")
        
        # Create SHAP explainer for LightGBM
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_explain.values)
        
        # Handle binary vs multiclass
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Multiclass - use the positive class for binary or first class for multiclass
            shap_values_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_values_plot = shap_values
        
        # Create plots based on type
        if plot_type == "summary":
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_explain.values, 
                            feature_names=X_explain.columns, 
                            max_display=max_display, show=False)
            if save_path:
                plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        elif plot_type == "waterfall":
            if len(X_explain) > 0:
                shap.waterfall_plot(explainer.expected_value, shap_values_plot[0], 
                                  X_explain.iloc[0], feature_names=X_explain.columns,
                                  max_display=max_display, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_waterfall.png", dpi=300, bbox_inches='tight')
                plt.show()
                
        elif plot_type == "force":
            if len(X_explain) > 0:
                shap.force_plot(explainer.expected_value, shap_values_plot[0], 
                              X_explain.iloc[0], feature_names=X_explain.columns,
                              matplotlib=True, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_force.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Calculate feature importance from SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_explain.columns,
            'shap_importance': np.abs(shap_values_plot).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'expected_value': explainer.expected_value
        }



class CatBoostModel(BaseModel):
    '''
    CatBoost model implementation for Friend-Or-Foe datasets.
    '''
    
    def __init__(self, **kwargs):
        # Handle parameter mapping BEFORE calling super().__init__
        if 'n_estimators' in kwargs:
            kwargs['iterations'] = kwargs.pop('n_estimators')
        
        super().__init__(**kwargs)
        import catboost as cb
        self.cb = cb
        
        # Set default parameters
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'CatBoostModel':
        '''
        Fit CatBoost model.
        '''
        
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
        '''
        Make predictions with CatBoost.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict probabilities with CatBoost (classification only).
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("predict_proba only available for classification")
            
        return self.model.predict_proba(X.values)
    
    def save_model(self, filepath: str):
        '''
        Save CatBoost model.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # CatBoost has built-in save functionality
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        '''
        Load CatBoost model.
        '''
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
    def get_feature_importance(self) -> pd.DataFrame:
        '''
        Get feature importance from CatBoost.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def shap_analysis(self, X_background: pd.DataFrame, X_explain: pd.DataFrame, 
                     plot_type: str = "summary", max_display: int = 20, 
                     save_path: Optional[str] = None) -> Dict[str, Any]:
        '''
        Perform SHAP analysis for CatBoost model.
        
        Args:
            X_background: Background dataset for SHAP explainer
            X_explain: Dataset to explain
            plot_type: Type of SHAP plot ('summary', 'waterfall', 'force', 'dependence')
            max_display: Maximum number of features to display
            save_path: Path to save the plot
            
        Outputs:
            Dictionary containing SHAP values and explainer
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before SHAP analysis")
        
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib are required for explainability analysis. Install with: pip install shap matplotlib")
        
        # Create SHAP 
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_explain.values)
        
        # Handle binary vs multiclass
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Multiclass - use the positive class for binary or first class for multiclass
            shap_values_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_values_plot = shap_values
        
        # Create plots based on type
        if plot_type == "summary":
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_explain.values, 
                            feature_names=X_explain.columns, 
                            max_display=max_display, show=False)
            if save_path:
                plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        elif plot_type == "waterfall":
            if len(X_explain) > 0:
                shap.waterfall_plot(explainer.expected_value, shap_values_plot[0], 
                                  X_explain.iloc[0], feature_names=X_explain.columns,
                                  max_display=max_display, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_waterfall.png", dpi=300, bbox_inches='tight')
                plt.show()
                
        elif plot_type == "force":
            if len(X_explain) > 0:
                shap.force_plot(explainer.expected_value, shap_values_plot[0], 
                              X_explain.iloc[0], feature_names=X_explain.columns,
                              matplotlib=True, show=False)
                if save_path:
                    plt.savefig(f"{save_path}_force.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Calculate feature importance from SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_explain.columns,
            'shap_importance': np.abs(shap_values_plot).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'expected_value': explainer.expected_value
        }

class FTTransformerModel(BaseModel):
    '''
    FT-Transformer model implementation using rtdl_revisiting_models package.
    https://github.com/yandex-research/rtdl
    Uses default parameters only, following the standard usage pattern.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Only allow basic training parameters, not model architecture parameters
        default_params = {
            'max_epochs': 100,
            'patience': 16,
            'batch_size': 256,
            'eval_batch_size': 4096,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'FTTransformerModel':
        """Fit FT-Transformer model using rtdl_revisiting_models with default parameters."""
        
        try:
            import torch
            import torch.nn.functional as F
            import delu
            import numpy as np
            import scipy.special
            import sklearn.preprocessing
            from rtdl_revisiting_models import FTTransformer
            from tqdm import tqdm
            import math
        except ImportError:
            raise ImportError("Required packages missing. Install with: pip install rtdl-revisiting-models delu")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        delu.random.seed(self.model_params['random_state'])
        
        # Prepare data
        print("Preprocessing data for FT-Transformer...")
        
        # Convert to numpy
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.flatten()
        
        X_val_np = X_val.values.astype(np.float32) if X_val is not None else None
        y_val_np = y_val.values.flatten() if y_val is not None else None
        
        # Feature preprocessing (using quantile transformation like in the example)
        noise = np.random.default_rng(self.model_params['random_state']).normal(
            0.0, 1e-5, X_train_np.shape
        ).astype(X_train_np.dtype)
        
        preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(X_train_np) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        ).fit(X_train_np + noise)
        
        X_train_processed = preprocessing.transform(X_train_np)
        X_val_processed = preprocessing.transform(X_val_np) if X_val_np is not None else None
        
        # Determine task type and prepare labels
        if task_type.lower() == "classification":
            n_classes = len(np.unique(y_train_np))
            if n_classes == 2:
                self.task_type = "binclass"
                d_out = 1
                y_train_tensor = torch.FloatTensor(y_train_np).to(device)
                y_val_tensor = torch.FloatTensor(y_val_np).to(device) if y_val_np is not None else None
            else:
                self.task_type = "multiclass"
                d_out = n_classes
                y_train_tensor = torch.LongTensor(y_train_np).to(device)
                y_val_tensor = torch.LongTensor(y_val_np).to(device) if y_val_np is not None else None
        else:
            self.task_type = "regression"
            d_out = 1
            # Normalize labels for regression
            self.Y_mean = y_train_np.mean()
            self.Y_std = y_train_np.std()
            y_train_normalized = (y_train_np - self.Y_mean) / self.Y_std
            y_val_normalized = (y_val_np - self.Y_mean) / self.Y_std if y_val_np is not None else None
            
            y_train_tensor = torch.FloatTensor(y_train_normalized).to(device)
            y_val_tensor = torch.FloatTensor(y_val_normalized).to(device) if y_val_normalized is not None else None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
        X_val_tensor = torch.FloatTensor(X_val_processed).to(device) if X_val_processed is not None else None
        
        data = {
            'train': {'x_cont': X_train_tensor, 'y': y_train_tensor},
        }
        
        if X_val_tensor is not None:
            data['val'] = {'x_cont': X_val_tensor, 'y': y_val_tensor}
        
        # Create model with default parameters only
        print("Creating FT-Transformer model with default parameters...")
        self.model = FTTransformer(
            n_cont_features=X_train.shape[1],
            cat_cardinalities=[],  # No categorical features for Friend-Or-Foe 
            d_out=d_out,
            **FTTransformer.get_default_kwargs(),
        ).to(device)
        
        optimizer = self.model.make_default_optimizer()
        
        # Define loss function
        if self.task_type == "binclass":
            loss_fn = F.binary_cross_entropy_with_logits
        elif self.task_type == "multiclass":
            loss_fn = F.cross_entropy
        else:
            loss_fn = F.mse_loss
        
        # Helper function to apply model
        def apply_model(batch):
            return self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
        
        # Evaluation function
        @torch.no_grad()
        def evaluate(part: str) -> float:
            self.model.eval()
            
            y_pred_list = []
            y_true_list = []
            
            for batch in delu.iter_batches(data[part], self.model_params['eval_batch_size']):
                preds = apply_model(batch)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(batch["y"].cpu().numpy())
            
            y_pred = np.concatenate(y_pred_list)
            y_true = np.concatenate(y_true_list)
            
            if self.task_type == "binclass":
                y_pred_prob = scipy.special.expit(y_pred)
                y_pred_binary = np.round(y_pred_prob)
                score = sklearn.metrics.accuracy_score(y_true, y_pred_binary)
            elif self.task_type == "multiclass":
                y_pred_class = y_pred.argmax(1)
                score = sklearn.metrics.accuracy_score(y_true, y_pred_class)
            else:
                # Regression - return negative RMSE for maximization
                score = -np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
            
            return score
        
        # Training loop
        print("Training FT-Transformer...")
        batch_size = self.model_params['batch_size']
        epoch_size = math.ceil(len(X_train_tensor) / batch_size)
        early_stopping = delu.tools.EarlyStopping(self.model_params['patience'], mode="max")
        
        self.training_history = {
            'train_loss': [],
            'val_score': [],
            'best_epoch': -1,
            'best_val_score': -float('inf')
        }
        
        best_state_dict = None
        
        for epoch in range(self.model_params['max_epochs']):
            # Training
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            progress_bar = tqdm(
                delu.iter_batches(data["train"], batch_size, shuffle=True),
                desc=f"Epoch {epoch+1}/{self.model_params['max_epochs']}",
                total=epoch_size,
                leave=False
            )
            
            for batch in progress_bar:
                optimizer.zero_grad()
                predictions = apply_model(batch)
                
                if self.task_type == "multiclass":
                    loss = loss_fn(predictions, batch["y"])
                else:
                    loss = loss_fn(predictions, batch["y"])
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = epoch_loss / n_batches
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            if 'val' in data:
                val_score = evaluate("val")
                self.training_history['val_score'].append(val_score)
                
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_score={val_score:.4f}")
                
                # Early stopping check
                early_stopping.update(val_score)
                if val_score > self.training_history['best_val_score']:
                    self.training_history['best_val_score'] = val_score
                    self.training_history['best_epoch'] = epoch
                    best_state_dict = self.model.state_dict().copy()
                
                if early_stopping.should_stop():
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best model from epoch {self.training_history['best_epoch']+1}")
        
        self.is_fitted = True
        self.device = device
        self.preprocessing = preprocessing
        self.n_cont_features = X_train.shape[1]
        self.d_out = d_out
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Make predictions with FT-Transformer.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import torch
        import delu
        import numpy as np
        import scipy.special
        
        # Preprocess features
        X_processed = self.preprocessing.transform(X.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            data_dict = {'x_cont': X_tensor}
            for batch in delu.iter_batches(data_dict, self.model_params['eval_batch_size']):
                preds = self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
                predictions.append(preds.cpu().numpy())
        
        y_pred = np.concatenate(predictions)
        
        if self.task_type == "binclass":
            y_pred_prob = scipy.special.expit(y_pred)
            return np.round(y_pred_prob).astype(int)
        elif self.task_type == "multiclass":
            return y_pred.argmax(1)
        else:
            # Regression - denormalize
            return y_pred * self.Y_std + self.Y_mean
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with FT-Transformer (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        
        import torch
        import delu
        import numpy as np
        import scipy.special
        
        # Preprocess features
        X_processed = self.preprocessing.transform(X.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            data_dict = {'x_cont': X_tensor}
            for batch in delu.iter_batches(data_dict, self.model_params['eval_batch_size']):
                preds = self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
                predictions.append(preds.cpu().numpy())
        
        y_pred = np.concatenate(predictions)
        
        if self.task_type == "binclass":
            y_pred_prob = scipy.special.expit(y_pred)
            return np.column_stack([1 - y_pred_prob, y_pred_prob])
        else:
            # Multiclass
            return scipy.special.softmax(y_pred, axis=1)
    
    def save_model(self, filepath: str):
        """Save FT-Transformer model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import torch
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'training_history': self.training_history,
            'task_type': self.task_type,
            'preprocessing': self.preprocessing,
            'device': str(self.device),
            'n_cont_features': self.n_cont_features,
            'd_out': self.d_out,
        }
        
        # Add regression-specific attributes
        if self.task_type == "regression":
            model_data['Y_mean'] = self.Y_mean
            model_data['Y_std'] = self.Y_std
        
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str):
        '''
        Load FT-Transformer model.
        '''
        import torch
        from rtdl_revisiting_models import FTTransformer
        
        model_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        self.model_params = model_data['model_params']
        self.training_history = model_data['training_history']
        self.task_type = model_data['task_type']
        self.preprocessing = model_data['preprocessing']
        self.device = torch.device(model_data['device'])
        self.n_cont_features = model_data['n_cont_features']
        self.d_out = model_data['d_out']
        
        if self.task_type == "regression":
            self.Y_mean = model_data['Y_mean']
            self.Y_std = model_data['Y_std']
        
        # Recreate model with default parameters
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=[],
            d_out=self.d_out,
            **FTTransformer.get_default_kwargs(),
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_fitted = True

class TabMModel(BaseModel):
    '''
    TabM model implementation using the official TabM code.
    https://github.com/yandex-research/tabm
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set default parameters
        default_params = {
            'arch_type': 'tabm',
            'k': 32,
            'n_blocks': 3,
            'd_block': 512,
            'dropout': 0.1,
            'lr': 2e-3,
            'weight_decay': 3e-4,
            'max_epochs': 2000,
            'patience': 200,
            'batch_size': 256,
            'eval_batch_size': 128,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            task_type: str = "classification") -> 'TabMModel':
        '''
        Fit TabM model using the official implementation.
        '''
        
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            import scipy.special
            import sklearn.preprocessing
            import sklearn.metrics
            from tqdm import tqdm
            import math
            import random
            
            # Import tbm
            from .tabm.model import Model, make_parameter_groups
            
        except ImportError:
            raise ImportError("Required packages missing. Install TabM dependencies and ensure model files are in friend_or_foe/models/tabm/")
        
        # Set random seeds
        seed = self.model_params['random_state']
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Preprocessing data for TabM...")
        
        # Convert to numpy
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.flatten()
        
        X_val_np = X_val.values.astype(np.float32) if X_val is not None else None
        y_val_np = y_val.values.flatten() if y_val is not None else None
        
        # Feature preprocessing (quantile transformation)
        noise = np.random.default_rng(seed).normal(
            0.0, 1e-5, X_train_np.shape
        ).astype(X_train_np.dtype)
        
        preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(X_train_np) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        ).fit(X_train_np + noise)
        
        X_train_processed = preprocessing.transform(X_train_np)
        X_val_processed = preprocessing.transform(X_val_np) if X_val_np is not None else None
        
        # Determine task type and prepare labels
        if task_type.lower() == "classification":
            n_classes = len(np.unique(y_train_np))
            if n_classes == 2:
                self.task_type = "binclass"
            else:
                self.task_type = "multiclass"
            self.n_classes = n_classes
            self.regression_label_stats = None
            
            # Convert to int64 for classification
            y_train_processed = y_train_np.astype(np.int64)
            y_val_processed = y_val_np.astype(np.int64) if y_val_np is not None else None
        else:
            self.task_type = "regression"
            self.n_classes = None
            
            # Normalize labels for regression
            self.regression_label_stats = {
                'mean': y_train_np.mean(),
                'std': y_train_np.std()
            }
            y_train_processed = (y_train_np - self.regression_label_stats['mean']) / self.regression_label_stats['std']
            y_val_processed = (y_val_np - self.regression_label_stats['mean']) / self.regression_label_stats['std'] if y_val_np is not None else None
        
        # Prepare data dictionaries
        data_numpy = {
            'train': {'x_cont': X_train_processed, 'y': y_train_processed},
        }
        
        if X_val_processed is not None:
            data_numpy['val'] = {'x_cont': X_val_processed, 'y': y_val_processed}
        
        # Convert to tensors
        data = {
            part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
            for part in data_numpy
        }
        
        # Set correct tensor types
        if self.task_type == "regression":
            for part in data:
                data[part]['y'] = data[part]['y'].float()
        
        # Create model
        print("Creating TabM model...")
        
        # TabM configuration
        self.model = Model(
            n_num_features=X_train.shape[1],
            cat_cardinalities=[],  # No categorical features for Friend-Or-Foe
            n_classes=self.n_classes,
            backbone={
                'type': 'MLP',
                'n_blocks': self.model_params['n_blocks'],
                'd_block': self.model_params['d_block'],
                'dropout': self.model_params['dropout'],
            },
            bins=None,
            num_embeddings=None,
            arch_type=self.model_params['arch_type'],
            k=self.model_params['k'],
        ).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model), 
            lr=self.model_params['lr'], 
            weight_decay=self.model_params['weight_decay']
        )
        
        # Loss function
        base_loss_fn = F.mse_loss if self.task_type == 'regression' else F.cross_entropy
        
        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # TabM produces k predictions per object
            k = y_pred.shape[-1 if self.task_type == 'regression' else -2]
            return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))
        
        # Model application function
        def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
            return (
                self.model(
                    data[part]['x_cont'][idx],
                    None  # No categorical features
                )
                .squeeze(-1)  # Remove last dimension for regression
                .float()
            )
        
        # Evaluation function
        @torch.no_grad()
        def evaluate(part: str) -> float:
            self.model.eval()
            
            eval_batch_size = self.model_params['eval_batch_size']
            y_pred = torch.cat([
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
            ]).cpu().numpy()
            
            if self.task_type == 'regression':
                # Transform predictions back to original space
                y_pred = y_pred * self.regression_label_stats['std'] + self.regression_label_stats['mean']
            
            # Compute mean of k predictions
            if self.task_type != 'regression':
                y_pred = scipy.special.softmax(y_pred, axis=-1)
            y_pred = y_pred.mean(1)
            
            y_true = data[part]['y'].cpu().numpy()
            if self.task_type == 'regression':
                # Transform true values back to original space for evaluation
                y_true = y_true * self.regression_label_stats['std'] + self.regression_label_stats['mean']
                score = -np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
            else:
                score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
            
            return float(score)
        
        # Training loop
        print("Training TabM...")
        
        batch_size = self.model_params['batch_size']
        epoch_size = math.ceil(len(data['train']['y']) / batch_size)
        
        best = {
            'val': -math.inf,
            'test': -math.inf,
            'epoch': -1,
        }
        remaining_patience = self.model_params['patience']
        
        self.training_history = {
            'train_loss': [],
            'val_score': [],
            'best_epoch': -1,
            'best_val_score': -float('inf')
        }
        
        best_state_dict = None
        Y_train_tensor = data['train']['y']
        
        for epoch in range(self.model_params['max_epochs']):
            # Training
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            progress_bar = tqdm(
                torch.randperm(len(data['train']['y']), device=device).split(batch_size),
                desc=f"Epoch {epoch+1}/{self.model_params['max_epochs']}",
                total=epoch_size,
                leave=False
            )
            
            for batch_idx in progress_bar:
                optimizer.zero_grad()
                loss = loss_fn(apply_model('train', batch_idx), Y_train_tensor[batch_idx])
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = epoch_loss / n_batches
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            if 'val' in data:
                val_score = evaluate('val')
                self.training_history['val_score'].append(val_score)
                
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_score={val_score:.4f}")
                
                if val_score > best['val']:
                    print("New best epoch!")
                    best = {'val': val_score, 'epoch': epoch}
                    self.training_history['best_val_score'] = val_score
                    self.training_history['best_epoch'] = epoch
                    best_state_dict = self.model.state_dict().copy()
                    remaining_patience = self.model_params['patience']
                else:
                    remaining_patience -= 1
                
                if remaining_patience < 0:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best model from epoch {self.training_history['best_epoch']+1}")
        
        self.is_fitted = True
        self.device = device
        self.preprocessing = preprocessing
        self.n_features = X_train.shape[1]
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Make predictions with TabM.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import torch
        import numpy as np
        import scipy.special
        
        # Preprocess features
        X_processed = self.preprocessing.transform(X.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            eval_batch_size = self.model_params['eval_batch_size']
            for idx in torch.arange(len(X_tensor), device=self.device).split(eval_batch_size):
                preds = self.model(X_tensor[idx], None).squeeze(-1).float()
                predictions.append(preds.cpu().numpy())
        
        y_pred = np.concatenate(predictions)
        
        if self.task_type == 'regression':
            # Transform back to original space
            y_pred = y_pred * self.regression_label_stats['std'] + self.regression_label_stats['mean']
            # Return mean of k predictions
            return y_pred.mean(1)
        else:
            # For classification, compute mean in probability space
            y_pred = scipy.special.softmax(y_pred, axis=-1)
            y_pred = y_pred.mean(1)
            return y_pred.argmax(1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with TabM (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        
        import torch
        import numpy as np
        import scipy.special
        
        # Preprocess features
        X_processed = self.preprocessing.transform(X.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            eval_batch_size = self.model_params['eval_batch_size']
            for idx in torch.arange(len(X_tensor), device=self.device).split(eval_batch_size):
                preds = self.model(X_tensor[idx], None).squeeze(-1).float()
                predictions.append(preds.cpu().numpy())
        
        y_pred = np.concatenate(predictions)
        
        # Convert to probabilities and average across k predictions
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        return y_pred.mean(1)
    
    def save_model(self, filepath: str):
        '''
        Save TabM model.
        '''
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import torch
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'training_history': self.training_history,
            'task_type': self.task_type,
            'n_classes': self.n_classes,
            'regression_label_stats': self.regression_label_stats,
            'preprocessing': self.preprocessing,
            'device': str(self.device),
            'n_features': self.n_features,
        }
        
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str):
        '''
        Load TabM model.
        '''
    
        model_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        self.model_params = model_data['model_params']
        self.training_history = model_data['training_history']
        self.task_type = model_data['task_type']
        self.n_classes = model_data['n_classes']
        self.regression_label_stats = model_data['regression_label_stats']
        self.preprocessing = model_data['preprocessing']
        self.device = torch.device(model_data['device'])
        self.n_features = model_data['n_features']
        
        # Recreate model
        self.model = Model(
            n_num_features=self.n_features,
            cat_cardinalities=[],
            n_classes=self.n_classes,
            backbone={
                'type': 'MLP',
                'n_blocks': self.model_params['n_blocks'],
                'd_block': self.model_params['d_block'],
                'dropout': self.model_params['dropout'],
            },
            bins=None,
            num_embeddings=None,
            arch_type=self.model_params['arch_type'],
            k=self.model_params['k'],
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_fitted = True
