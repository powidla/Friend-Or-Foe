#!/usr/bin/env python3
"""
Tests all models and loading data.
So far basic functions work.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, List, Tuple

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent))

try:
    from friend_or_foe.model.base import (
        BaseModel, TabNetModel, XGBoostModel, LightGBMModel, 
        CatBoostModel, FTTransformerModel, TabMModel
    )
    from friend_or_foe.data.loader import FriendOrFoeDataLoader
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure the friend_or_foe package is properly installed or in your Python path.")
    sys.exit(1)


class TestColors:
    '''
    ANSI color codes for test output.
    '''
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(message: str):
    '''
    Print a formatted test header.
    '''
    print(f"\n{TestColors.BOLD}{TestColors.BLUE}{'='*60}")
    print(f"{message}")
    print(f"{'='*60}{TestColors.END}")


def print_success(message: str):
    '''
    Print success message.
    '''
    print(f"{TestColors.GREEN} {message}{TestColors.END}")


def print_error(message: str):
    '''
    Print error message.
    '''
    print(f"{TestColors.RED} {message}{TestColors.END}")


def print_warning(message: str):
    '''
    Print warning message.
    '''
    print(f"{TestColors.YELLOW}  {message}{TestColors.END}")


def print_info(message: str):
    '''
    Print info message.
    '''
    print(f"{TestColors.CYAN} {message}{TestColors.END}")


def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 20, 
                           task_type: str = "classification", 
                           n_classes: int = 2) -> Dict[str, pd.DataFrame]:
    '''
    Create synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: 'classification' or 'regression'
        n_classes: Number of classes (for classification)
    
    Returns:
        Dictionary with train/val/test splits
    '''
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    if task_type == "classification":
        # Create classification target
        weights = np.random.randn(n_features)
        linear_combination = X @ weights
        probabilities = 1 / (1 + np.exp(-linear_combination))
        
        if n_classes == 2:
            y = (probabilities > 0.5).astype(int)
        else:
            # Softmax
            logits = np.random.randn(n_samples, n_classes)
            y = np.argmax(logits, axis=1)
    else:
        # Regression
        weights = np.random.randn(n_features)
        noise = np.random.randn(n_samples) * 0.1
        y = X @ weights + noise
    
    # Conver
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=['target'])
    
    # Split into train/val/test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    return {
        'X_train': X_df[:n_train],
        'y_train': y_df[:n_train],
        'X_val': X_df[n_train:n_train+n_val],
        'y_val': y_df[n_train:n_train+n_val],
        'X_test': X_df[n_train+n_val:],
        'y_test': y_df[n_train+n_val:],
    }


def test_model_interface(model_class, model_name: str, task_type: str = "classification") -> bool:
    '''
    Test basic model interface (fit, predict, evaluate).
    
    Args:
        model_class: Model class to test
        model_name: Name of the model for logging
        task_type: 'classification' or 'regression'
    
    Returns:
        True if all tests pass, False otherwise
    '''
    print_info(f"Testing {model_name} interface...")
    
    try:
        # Create synthetic data
        n_classes = 2 if task_type == "classification" else None
        data = create_synthetic_dataset(
            n_samples=500, 
            n_features=10, 
            task_type=task_type,
            n_classes=n_classes
        )
        
        # Initialize model with minimal parameters for speed
        if model_name == "TabNet":
            model = model_class(n_d=8, n_a=8, n_steps=1)
        elif model_name == "FT-Transformer":
            model = model_class(max_epochs=2, patience=1, batch_size=64)
        elif model_name == "TabM":
            model = model_class(max_epochs=2, patience=1, batch_size=64, k=4, d_block=32)
        else:
            # Tree-based models
            model = model_class(n_estimators=10, random_state=42)
        
        # Test 1: Model initialization
        assert hasattr(model, 'fit'), f"{model_name} missing fit method"
        assert hasattr(model, 'predict'), f"{model_name} missing predict method"
        assert hasattr(model, 'evaluate'), f"{model_name} missing evaluate method"
        print_success(f"{model_name} initialized correctly")
        
        # Test 2: Model fitting
        model.fit(
            data['X_train'], 
            data['y_train'], 
            data['X_val'], 
            data['y_val'], 
            task_type=task_type
        )
        assert model.is_fitted, f"{model_name} not marked as fitted"
        print_success(f"{model_name} fitted successfully")
        
        # Test 3: Predictions
        predictions = model.predict(data['X_test'])
        assert isinstance(predictions, np.ndarray), f"{model_name} predictions not numpy array"
        assert len(predictions) == len(data['X_test']), f"{model_name} wrong prediction length"
        print_success(f"{model_name} predictions work")
        
        # Test 4: Evaluation
        metrics = model.evaluate(data['X_test'], data['y_test'], task_type=task_type)
        assert isinstance(metrics, dict), f"{model_name} metrics not dictionary"
        assert len(metrics) > 0, f"{model_name} no metrics returned"
        print_success(f"{model_name} evaluation works")
        
        # Test 5: Probability predictions (classification only)
        if task_type == "classification":
            try:
                probabilities = model.predict_proba(data['X_test'])
                assert isinstance(probabilities, np.ndarray), f"{model_name} probabilities not numpy array"
                assert probabilities.shape[0] == len(data['X_test']), f"{model_name} wrong probability shape"
                print_success(f"{model_name} probability predictions work")
            except NotImplementedError:
                print_warning(f"{model_name} predict_proba not implemented")
        
        return True
        
    except Exception as e:
        print_error(f"{model_name} interface test failed: {e}")
        return False


def test_model_save_load(model_class, model_name: str, task_type: str = "classification") -> bool:
    '''
    Test model save/load functionality.
    
    Args:
        model_class: Model class to test
        model_name: Name of the model for logging
        task_type: 'classification' or 'regression'
    
    Returns:
        True if save/load works, False otherwise
    '''
    print_info(f"Testing {model_name} save/load...")
    
    try:
        # Create and train model
        data = create_synthetic_dataset(
            n_samples=200, 
            n_features=5, 
            task_type=task_type
        )
        
        # Initialize with minimal parameters
        if model_name == "TabNet":
            model = model_class(n_d=4, n_a=4, n_steps=1)
        elif model_name == "FT-Transformer":
            model = model_class(max_epochs=1, patience=1, batch_size=32)
        elif model_name == "TabM":
            model = model_class(max_epochs=1, patience=1, batch_size=32, k=2, d_block=16)
        else:
            model = model_class(n_estimators=5, random_state=42)
        
        # Train model
        model.fit(data['X_train'], data['y_train'], task_type=task_type)
        
        # Get predictions before saving
        predictions_before = model.predict(data['X_test'])
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model.save_model(tmp_file.name)
            
            # Create new model instance and load
            if model_name == "TabNet":
                model_loaded = model_class(n_d=4, n_a=4, n_steps=1)
            elif model_name == "FT-Transformer":
                model_loaded = model_class()
            elif model_name == "TabM":
                model_loaded = model_class()
            else:
                model_loaded = model_class()
            
            model_loaded.load_model(tmp_file.name)
            
            # Get predictions after loading
            predictions_after = model_loaded.predict(data['X_test'])
            
            # Compare predictions
            np.testing.assert_array_almost_equal(
                predictions_before, 
                predictions_after, 
                decimal=4,
                err_msg=f"{model_name} predictions differ after save/load"
            )
            
            # Clean 
            os.unlink(tmp_file.name)
        
        print_success(f"{model_name} save/load works correctly")
        return True
        
    except Exception as e:
        print_error(f"{model_name} save/load test failed: {e}")
        return False


def test_tree_model_shap(model_class, model_name: str) -> bool:
    '''
    Test SHAP analysis for tree-based models.
    
    Args:
        model_class: Tree-based model class
        model_name: Name of the model
    
    Returns:
        True if SHAP analysis works, False otherwise
    '''
    print_info(f"Testing {model_name} SHAP analysis...")
    
    try:
        # Create data
        data = create_synthetic_dataset(n_samples=200, n_features=8, task_type="classification")
        
        # Train model
        model = model_class(n_estimators=10, random_state=42)
        model.fit(data['X_train'], data['y_train'], task_type="classification")
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, pd.DataFrame), f"{model_name} importance not DataFrame"
        assert 'feature' in importance.columns, f"{model_name} missing feature column"
        assert 'importance' in importance.columns, f"{model_name} missing importance column"
        print_success(f"{model_name} feature importance works")
        
        # Test SHAP analysis
        shap_results = model.shap_analysis(
            X_background=data['X_train'].head(50),
            X_explain=data['X_test'].head(20),
            plot_type="summary",
            max_display=5
        )
        
        assert isinstance(shap_results, dict), f"{model_name} SHAP results not dictionary"
        assert 'feature_importance' in shap_results, f"{model_name} missing SHAP feature importance"
        assert 'shap_values' in shap_results, f"{model_name} missing SHAP values"
        print_success(f"{model_name} SHAP analysis works")
        
        return True
        
    except ImportError:
        print_warning(f"{model_name} SHAP test skipped (SHAP not installed)")
        return True  # Not a failure if SHAP is not installed
    except Exception as e:
        print_error(f"{model_name} SHAP test failed: {e}")
        return False


def test_data_loader() -> bool:
    '''
    Test the data loader functionality.
    '''
    print_info("Testing FriendOrFoeDataLoader...")
    
    try:
        loader = FriendOrFoeDataLoader(verbose=False)
        
        # Test initialization
        assert hasattr(loader, 'load_dataset'), "DataLoader missing load_dataset method"
        assert hasattr(loader, 'list_available_datasets'), "DataLoader missing list_available_datasets method"
        print_success("DataLoader initialized correctly")
        
        # Test listing datasets (this requires internet)
        try:
            datasets = loader.list_available_datasets()
            assert isinstance(datasets, dict), "list_available_datasets should return dict"
            print_success("DataLoader can list datasets")
        except Exception:
            print_warning("DataLoader listing test skipped (requires internet)")
        
        return True
        
    except Exception as e:
        print_error(f"DataLoader test failed: {e}")
        return False


def run_all_tests() -> bool:
    '''
    Run comprehensive test suite for all models.
    
    Returns:
        True if all tests pass, False otherwise
    '''
    print_test_header("FRIEND-OR-FOE COMPREHENSIVE TEST SUITE")
    
    # Test configuration
    models_to_test = [
        (XGBoostModel, "XGBoost"),
        (LightGBMModel, "LightGBM"),
        (CatBoostModel, "CatBoost"),
        (TabNetModel, "TabNet"),
    ]
    
    # models
    try:
        models_to_test.append((FTTransformerModel, "FT-Transformer"))
    except Exception:
        print_warning("FT-Transformer not available (missing dependencies)")
    
    try:
        models_to_test.append((TabMModel, "TabM"))
    except Exception:
        print_warning("TabM not available (missing dependencies or model files)")
    
    tree_models = [
        (XGBoostModel, "XGBoost"),
        (LightGBMModel, "LightGBM"),
        (CatBoostModel, "CatBoost"),
    ]
    
    # Track test results
    test_results = []
    
    # Test 1: Data Loader
    print_test_header("TESTING DATA LOADER")
    test_results.append(("DataLoader", test_data_loader()))
    
    # Test 2: Model Interfaces - Classification
    print_test_header("TESTING MODEL INTERFACES - CLASSIFICATION")
    for model_class, model_name in models_to_test:
        result = test_model_interface(model_class, model_name, "classification")
        test_results.append((f"{model_name} (Classification)", result))
    
    # Test 3: Model Interfaces - Regression
    print_test_header("TESTING MODEL INTERFACES - REGRESSION")
    for model_class, model_name in models_to_test:
        result = test_model_interface(model_class, model_name, "regression")
        test_results.append((f"{model_name} (Regression)", result))
    
    # Test 4: Save/Load Functionality
    print_test_header("TESTING SAVE/LOAD FUNCTIONALITY")
    for model_class, model_name in models_to_test:
        result = test_model_save_load(model_class, model_name, "classification")
        test_results.append((f"{model_name} (Save/Load)", result))
    
    # Test 5: SHAP Analysis (Tree models only)
    print_test_header("TESTING SHAP ANALYSIS")
    for model_class, model_name in tree_models:
        result = test_tree_model_shap(model_class, model_name)
        test_results.append((f"{model_name} (SHAP)", result))
    
    # Summary
    print_test_header("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        color = TestColors.GREEN if result else TestColors.RED
        print(f"{color}{status:>6}{TestColors.END} | {test_name}")
    
    print(f"\n{TestColors.BOLD}Overall Results: {passed}/{total} tests passed{TestColors.END}")
    
    if passed == total:
        print_success("All tests passed!")
        return True
    else:
        print_error(f"{total - passed} tests failed. Please check the errors above.")
        return False


def main():
    '''
    Main tests.
    '''
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\n Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
