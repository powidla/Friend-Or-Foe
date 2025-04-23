# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import pandas as pd

warnings.resetwarnings()

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
delu.random.seed(0)

# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]

task_type: TaskType = "regression"
n_classes = None
dataset = sklearn.datasets.fetch_california_housing()
X_cont: np.ndarray = dataset["data"]
Y: np.ndarray = dataset["target"]

# NOTE: uncomment to solve a classification task.
n_classes = 3
assert n_classes >= 2
task_type: TaskType = 'binclass' if n_classes == 2 else 'multiclass'
X_cont, Y = sklearn.datasets.make_classification(
    n_samples=20000,
    n_features=8,
    n_classes=n_classes,
    n_informative=4,
    n_redundant=2,
)

# >>> Continuous features.
X_cont: np.ndarray = X_cont.astype(np.float32)
n_cont_features = X_cont.shape[1]

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, but,
# for the demonstration purposes, it is possible to generate them.
cat_cardinalities = [
    # NOTE: uncomment the two lines below to add two categorical features.
    # 4,  # Allowed values: [0, 1, 2, 3].
    # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
]
X_cat = (
    np.column_stack(
        [np.random.randint(0, c, (len(X_cont),)) for c in cat_cardinalities]
    )
    if cat_cardinalities
    else None
)

# >>> Labels.
# Regression labels must be represented by float32.
if task_type == "regression":
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

# >>> Split the dataset.
all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8
)
data_numpy = {
    "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
    "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
    "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
}

if X_cat is not None:
    data_numpy["train"]["x_cat"] = X_cat[train_idx]
    data_numpy["val"]["x_cat"] = X_cat[val_idx]
    data_numpy["test"]["x_cat"] = X_cat[test_idx]

# >>> Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# (A) Simple preprocessing strategy.
# preprocessing = sklearn.preprocessing.StandardScaler().fit(
#     data_numpy['train']['x_cont']
# )

# (B) Fancy preprocessing strategy.
# The noise is added to improve the output of QuantileTransformer in some cases.
X_cont_train_numpy = data_numpy["train"]["x_cont"]
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution="normal",
    subsample=10**9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

for part in data_numpy:
    data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

# >>> Label preprocessing.
if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
X_train = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/X_train_TL-II.csv")
X_val = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/X_val_TL-II.csv")
X_test = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/X_test_TL-II.csv")
y_train = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/y_train_TL-II.csv")
y_val = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/y_val_TL-II.csv")
y_test = pd.read_csv("FOFdata/TL/CARVEME/100/TL-II/csv/y_test_TL-II.csv")

data_numpy = {
    'train': {'x_cont': X_train.values, 'y': y_train.values},
    'val': {'x_cont': X_val.values, 'y': y_val.values},
    'test': {'x_cont': X_test.values, 'y': y_test.values},
}

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()

# The output size.
d_out = n_classes if task_type == "multiclass" else 1

# # NOTE: uncomment to train MLP
# model = MLP(
#     d_in=n_cont_features + sum(cat_cardinalities),
#     d_out=d_out,
#     n_blocks=2,
#     d_block=384,
#     dropout=0.1,
# ).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# # NOTE: uncomment to train ResNet
# model = ResNet(
#     d_in=n_cont_features + sum(cat_cardinalities),
#     d_out=d_out,
#     n_blocks=2,
#     d_block=192,
#     d_hidden=None,
#     d_hidden_multiplier=2.0,
#     dropout1=0.3,
#     dropout2=0.0,
# ).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

model = FTTransformer(
    n_cont_features=X_train.shape[1],
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
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()

    eval_batch_size = 4
    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in delu.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * Y_std)
    return score  # The higher -- the better.


print(f'Test score before training: {evaluate("test"):.4f}')

import json

def calculate_and_save_metrics(part: str, save_path: str = "FT-CM-100-TL-II_metrics.json"):
    model.eval()
    eval_batch_size = 256  # Much more memory-efficient

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch in delu.iter_batches(data[part], eval_batch_size):
            preds = apply_model(batch).detach().cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(batch["y"].detach().cpu().numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    metrics = {}

    if task_type == "regression":
        y_pred_rescaled = y_pred * Y_std
        y_true_rescaled = y_true * Y_std
        metrics["rmse"] = float(np.sqrt(sklearn.metrics.mean_squared_error(y_true_rescaled, y_pred_rescaled)))
        metrics["mae"] = float(sklearn.metrics.mean_absolute_error(y_true_rescaled, y_pred_rescaled))
        metrics["r2"] = float(sklearn.metrics.r2_score(y_true_rescaled, y_pred_rescaled))

    elif task_type == "binclass":
        probs = scipy.special.expit(y_pred)
        preds = np.round(probs)
        metrics["accuracy"] = float(sklearn.metrics.accuracy_score(y_true, preds))
        metrics["precision"] = float(sklearn.metrics.precision_score(y_true, preds))
        metrics["recall"] = float(sklearn.metrics.recall_score(y_true, preds))
        metrics["f1"] = float(sklearn.metrics.f1_score(y_true, preds))
        metrics["mcc"] = float(sklearn.metrics.matthews_corrcoef(y_true, preds))
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y_true, probs))

    elif task_type == "multiclass":
        probs = y_pred
        preds = probs.argmax(1)
        metrics["accuracy"] = float(sklearn.metrics.accuracy_score(y_true, preds))
        metrics["precision"] = float(sklearn.metrics.precision_score(y_true, preds, average="macro"))
        metrics["recall"] = float(sklearn.metrics.recall_score(y_true, preds, average="macro"))
        metrics["f1"] = float(sklearn.metrics.f1_score(y_true, preds, average="macro"))
        metrics["mcc"] = float(sklearn.metrics.matthews_corrcoef(y_true, preds))
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y_true, probs, multi_class="ovr"))

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics

# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
# n_epochs = 20
# patience = 2
n_epochs = 10
patience = 20

batch_size = 256
epoch_size = math.ceil(len(X_train) / batch_size)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}

print(f"Device: {device.type.upper()}")
print("-" * 88 + "\n")
timer.run()
for epoch in range(n_epochs):
    for batch in tqdm(
        delu.iter_batches(data["train"], batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model(batch), batch["y"].squeeze(-1))
        loss.backward()
        optimizer.step()

    val_score = evaluate("val")
    test_score = evaluate("test")
    print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break

    if val_score > best["val"]:
        print("New best epoch!")
        best = {"val": val_score, "test": test_score, "epoch": epoch}

        torch.save(model.state_dict(), "FT_CM-100-TL-II.pt")
        # print("Saved model checkpoint to 'best_model.pt'")


print("\n\nResult:")
print(best)

# Restore best model
model.load_state_dict(torch.load("FT_CM-100-TL-II.pt"))

# # Calculate and save final metrics
final_metrics = calculate_and_save_metrics("test", save_path="FT-CM-100-TL-II_metrics.json")
print("Final Test Metrics:", final_metrics)
