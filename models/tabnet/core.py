import argparse
import numpy as np
import pandas as pd
import os
import torch
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TabNet")
    parser.add_argument("--col", type=str, required=True, help="Collection name, e.g. AGORA or CARVEME")
    parser.add_argument("--gr", type=str, required=True, help="Group/size, e.g. 100 or 50")
    parser.add_argument("--ds", type=str, required=True, help="Dataset split, e.g. MC-I, MC-II, MC-III")
    parser.add_argument("--seed", type=int, default=4221, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def train_tabnet(X_train, y_train, X_val, y_val,
                 output_dir="tabnet_results", model_name="TabNet-model",
                 seed=4221, max_epochs=100, patience=10):

    os.makedirs(output_dir, exist_ok=True)

    clf = TabNetClassifier(
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=1e-4, weight_decay=0.02),
        scheduler_params={"step_size": 50, "gamma": 0.99},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        n_d=64,
        n_a=64,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        seed=seed,
        device_name=device
    )

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['accuracy'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=8192
    )

    model_path = os.path.join(output_dir, model_name)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    clf.save_model(model_path)
    print(f"Model saved to {model_path}.zip")


def main():
    args = parse_args()
    col = args.col
    gr = args.gr
    ds = args.ds
    REPO_ID = "powidla/Friend-Or-Foe"
    base_path = f"Transfer Learning/{col}/{gr}/{ds}"
    
    X_train = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=f"{base_path}/X_train_{ds}.csv", repo_type="dataset"))
    X_val = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=f"{base_path}/X_val_{ds}.csv",   repo_type="dataset"))
    y_train = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=f"{base_path}/y_train_{ds}.csv", repo_type="dataset"))
    y_val = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=f"{base_path}/y_val_{ds}.csv",   repo_type="dataset"))
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy().reshape(-1)
    y_val = y_val.to_numpy().reshape(-1)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val]))
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    output_dir = os.path.join("tabnet_results", col, gr, ds)
    model_name = f"TabNet-{ds}-{col}-{gr}"

    train_tabnet(
        X_train, y_train, X_val, y_val,
        output_dir=output_dir,
        model_name=model_name,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
