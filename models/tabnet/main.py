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

import torch
#time management
from tqdm import tqdm
import time

#stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from itertools import combinations


from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import EarlyStopping

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def train_and_evaluate_tabnet(X_train, y_train, X_val, y_val, X_test, y_test,
                             output_dir="tabnet_results", seed=4221,
                             max_epochs=100, patience=10):
    '''
    Description: Train and evaluate TabNet on binclass.

    Arguments:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        output_dir: Directory to save results
        seed: Random seed

    Outputs:
        json
    '''

    os.makedirs(output_dir, exist_ok=True)

    # Init
    clf = TabNetClassifier(
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=1e-4, weight_decay=0.02),
        scheduler_params={"step_size":50, "gamma":0.99},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        n_d=64,
        n_a=64,
        seed=seed,
        device_name=device
    )

    # Train
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['accuracy'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=1024
    )

    # Test
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    with open(os.path.join(output_dir, "tabnet_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    clf.save_model(os.path.join(output_dir, "tabnet_model.zip"))

    print(f"\nTest Metrics:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"MCC: {metrics['MCC']:.4f}")

    return metrics

train_and_evaluate_tabnet(X_train, y_train, X_val, y_val, X_test, y_test)