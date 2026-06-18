# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal
from sklearn.preprocessing import LabelEncoder
import delu
import numpy as np
import scipy.special
import sklearn.metrics
import torch
import json
import torch.nn.functional as F
from torch import Tensor
from tqdm.std import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import argparse

#ignore warns
warnings.resetwarnings()
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--col", default="AGORA",  choices=["AGORA","CARVEME"])
parser.add_argument("--gr",  default="100",    choices=["100", "50"])
parser.add_argument("--ds",  default="TL-I",   choices=["TL-I","TL-II"]) # for Transfer Learning
args = parser.parse_args()
# files from HF are structured as follows Task/[COL]/[GR]/[DS]
REPO_ID ="powidla/Friend-Or-Foe"
COL = args.col
GR = args.gr
DS = args.ds

N_CONT_FEATURES = 153 # for regression and classification either 424 or 499 
N_EPOCHS = 2
PATIENCE = 16
BATCH_SIZE = 256

_base   = f"Transfer Learning/{COL}/{GR}/{DS}" # Classification, Regression
_suffix = f"{DS}"

MODEL_SAVE_PATH   = f"FT_{COL}-{GR}-{DS}.pt"
METRICS_SAVE_PATH = f"FT_{COL}-{GR}-{DS}_metrics.json"


def _dl(filename: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")


X_train = pd.read_csv(_dl(f"{_base}/X_train_{_suffix}.csv"))
X_val   = pd.read_csv(_dl(f"{_base}/X_val_{_suffix}.csv"))
X_test  = pd.read_csv(_dl(f"{_base}/X_test_{_suffix}.csv"))
y_train = pd.read_csv(_dl(f"{_base}/y_train_{_suffix}.csv"))
y_val   = pd.read_csv(_dl(f"{_base}/y_val_{_suffix}.csv"))
y_test  = pd.read_csv(_dl(f"{_base}/y_test_{_suffix}.csv"))

TaskType = Literal["regression", "binclass", "multiclass"]
task_type: TaskType = "multiclass" 
cat_cardinalities = []   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delu.random.seed(0)

le = LabelEncoder()
le.fit(np.concatenate([y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()]))
y_train_enc = le.transform(y_train.values.ravel())
y_val_enc   = le.transform(y_val.values.ravel())
y_test_enc  = le.transform(y_test.values.ravel())
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

n_classes = len(le.classes_)

data_numpy = {
    "train": {"x_cont": X_train.values.astype(np.float32), "y": y_train_enc},
    "val":   {"x_cont": X_val.values.astype(np.float32),   "y": y_val_enc},
    "test":  {"x_cont": X_test.values.astype(np.float32),  "y": y_test_enc},
}

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

for part in data:
    if task_type == "multiclass":
        data[part]["y"] = data[part]["y"].long()
    else:
        data[part]["y"] = data[part]["y"].float()

d_out = n_classes if task_type == "multiclass" else 1

model = FTTransformer(
    n_cont_features=N_CONT_FEATURES,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **FTTransformer.get_default_kwargs(),
).to(device)
optimizer = model.make_default_optimizer()


def apply_model(batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, (MLP, ResNet)):
        x_cat_ohe = (
            [
                F.one_hot(column, cardinality)
                for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)
            ]
            if "x_cat" in batch
            else []
        )
        return model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)
    elif isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")


loss_fn = (
    F.binary_cross_entropy_with_logits if task_type == "binclass"
    else F.cross_entropy if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()
    y_pred = (
        torch.cat([apply_model(b) for b in delu.iter_batches(data[part], 256)])
        .cpu().numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        return sklearn.metrics.accuracy_score(y_true, np.round(scipy.special.expit(y_pred)))
    elif task_type == "multiclass":
        return sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    else:
        return -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)


def calculate_and_save_metrics(part: str, save_path: str = METRICS_SAVE_PATH):
    model.eval()
    y_pred_list, y_true_list = [], []

    with torch.no_grad():
        for batch in delu.iter_batches(data[part], 256):
            y_pred_list.append(apply_model(batch).detach().cpu().numpy())
            y_true_list.append(batch["y"].detach().cpu().numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    metrics = {}

    if task_type == "binclass":
        probs = scipy.special.expit(y_pred)
        preds = np.round(probs)
        metrics["accuracy"] = float(sklearn.metrics.accuracy_score(y_true, preds))
        metrics["precision"] = float(sklearn.metrics.precision_score(y_true, preds))
        metrics["recall"] = float(sklearn.metrics.recall_score(y_true, preds))
        metrics["f1"] = float(sklearn.metrics.f1_score(y_true, preds))
        metrics["mcc"] = float(sklearn.metrics.matthews_corrcoef(y_true, preds))
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y_true, probs))

    elif task_type == "multiclass":
        preds = y_pred.argmax(1)
        metrics["accuracy"] = float(sklearn.metrics.accuracy_score(y_true, preds))
        metrics["precision"] = float(sklearn.metrics.precision_score(y_true, preds, average="macro"))
        metrics["recall"] = float(sklearn.metrics.recall_score(y_true, preds, average="macro"))
        metrics["f1"] = float(sklearn.metrics.f1_score(y_true, preds, average="macro"))
        metrics["mcc"] = float(sklearn.metrics.matthews_corrcoef(y_true, preds))
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y_true, y_pred, multi_class="ovr"))

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

# train
epoch_size = math.ceil(len(X_train) / BATCH_SIZE)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(PATIENCE, mode="max")
best = {"val": -math.inf, "test": -math.inf, "epoch": -1}

print(f"Run: COL={COL}  GR={GR}  DS={DS}")
print(f"Test score before training: {evaluate('test'):.4f}")
print("-" * 88 + "\n")

timer.run()
for epoch in range(N_EPOCHS):
    for batch in tqdm(
        delu.iter_batches(data["train"], BATCH_SIZE, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size
    ):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model(batch), batch["y"].squeeze(-1))
        loss.backward()
        optimizer.step()

    val_score  = evaluate("val")
    test_score = evaluate("test")
    print(f"(val) {val_score:.4f}  (test) {test_score:.4f}  [time] {timer}")

    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break

    if val_score > best["val"]:
        print("New best epoch!")
        best = {"val": val_score, "test": test_score, "epoch": epoch}
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\n\nResult:")
print(best)

