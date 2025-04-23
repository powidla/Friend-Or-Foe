import numpy as np
import pandas as pd
import os
import shutil
import joblib
import pickle
import json

# ml frameworks
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification # for test of funcs
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

#time management
from tqdm import tqdm
import time

#stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from itertools import combinations


from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

device = 'cpu'
print(f"Using device: {device}")

X_train_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/X_train_BC-I-100.csv")
X_val_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/X_val_BC-I-100.csv")
X_test_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/X_test_BC-I-100.csv")
y_train_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/y_train_BC-I-100.csv")
y_val_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/y_val_BC-I-100.csv")
y_test_csv = pd.read_csv("FOFdata/Classification/AGORA/100/BC-I/csv/y_test_BC-I-100.csv")

X_train = X_train_csv.to_numpy()
X_val = X_val_csv.to_numpy()
X_test = X_test_csv.to_numpy()
y_train = y_train_csv.to_numpy()
y_val = y_val_csv.to_numpy()
y_test = y_test_csv.to_numpy()


y_train = y_train.reshape(-1)
y_val = y_val.reshape(-1,)
y_test = y_test.reshape(-1)

def create_confusion_matrix(y_true, y_pred):
    '''
    Description: Create a confusion matrix.
    Arguments: y_true (array-like): Ground truth labels;
               y_pred (array-like): Predicted labels.
    Outputs:
        pd.DataFrame: A confusion matrix as a pandas DataFrame.
    '''
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["True Negative", "True Positive"],
                             columns=["Predicted Negative", "Predicted Positive"])
    return cm_df


def score_metrics(y_true, y_pred, y_prob):
    '''
    Description: Calculate various metrics for binary classification.
    Arguments: y_true (array-like): Ground truth labels;
               y_pred (array-like): Predicted labels;
               y_prob (array-like): Predicted probabilities for the positive class.
    Outputs:
        dict
    '''
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["PR AUC"] = auc(recall, precision)
    return metrics


def train_and_evaluate_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                                output_dir="catboost_results", seed=4221):
    os.makedirs(output_dir, exist_ok=True)

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    test_pool = Pool(X_test, y_test)

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.01,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=seed,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = score_metrics(y_test, y_pred, y_proba)

    with open(os.path.join(output_dir, "catboost_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    model.save_model(os.path.join(output_dir, "catboost_model.pkl"))

    print(f"\nTest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

def train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                               output_dir="xgboost_results", seed=4221):
    os.makedirs(output_dir, exist_ok=True)

    model = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=seed
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = score_metrics(y_test, y_pred, y_proba)

    with open(os.path.join(output_dir, "xgboost_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(model, os.path.join(output_dir, "xgboost_model.pkl"))

    print(f"\nXGBoost Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def train_and_evaluate_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                                output_dir="lightgbm_results", seed=4221):
    os.makedirs(output_dir, exist_ok=True)

    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        objective='binary',
        random_state=seed,
        boosting_type='gbdt',
        metric='auc'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        verbose=100
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = score_metrics(y_test, y_pred, y_proba)

    with open(os.path.join(output_dir, "lightgbm_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(model, os.path.join(output_dir, "lightgbm_model.pkl"))

    print(f"\nLightGBM Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

catboost_metrics = train_and_evaluate_catboost(X_train, y_train, X_val, y_val, X_test, y_test)
xgb_metrics = train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
lgbm_metrics = train_and_evaluate_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
