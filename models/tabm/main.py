import math
import random
import warnings
from typing import Literal, NamedTuple

import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
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
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, mean_squared_error
)
from model import Model, make_parameter_groups
warnings.simplefilter('ignore')

warnings.resetwarnings()
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
pass

# >>> Dataset.
TaskType = Literal['regression', 'binclass', 'multiclass']

# Regression.
task_type: TaskType = 'regression'
n_classes = None
dataset = sklearn.datasets.fetch_california_housing()
X_cont: np.ndarray = dataset['data']
Y: np.ndarray = dataset['target']

# # Classification.
# n_classes = 3
# assert n_classes >= 2
# task_type: TaskType = 'binclass' if n_classes == 2 else 'multiclass'
# X_cont, Y = sklearn.datasets.make_classification(
#     n_samples=20000,
#     n_features=8,
#     n_classes=n_classes,
#     n_informative=4,
#     n_redundant=2,
# )

task_is_regression = task_type == 'regression'

# >>> Continuous features.
# X_cont: np.ndarray = X_cont.astype(np.float32)
# n_cont_features = X_cont.shape[1]

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, however,
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
if task_type == 'regression':
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), 'Classification labels must form the range [0, 1, ..., n_classes - 1]'

# >>> Split the dataset.
all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8
)
data_numpy = {
    'train': {'x_cont': X_cont[train_idx], 'y': Y[train_idx]},
    'val': {'x_cont': X_cont[val_idx], 'y': Y[val_idx]},
    'test': {'x_cont': X_cont[test_idx], 'y': Y[test_idx]},
}
if X_cat is not None:
    data_numpy['train']['x_cat'] = X_cat[train_idx]
    data_numpy['val']['x_cat'] = X_cat[val_idx]
    data_numpy['test']['x_cat'] = X_cat[test_idx]

# Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# Simple preprocessing strategy.
# preprocessing = sklearn.preprocessing.StandardScaler().fit(
#     data_numpy['train']['x_cont']
# )

# Advanced preprocessing strategy.
# The noise is added to improve the output of QuantileTransformer in some cases.
X_cont_train_numpy = data_numpy['train']['x_cont']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

# Apply the preprocessing.
for part in data_numpy:
    data_numpy[part]['x_cont'] = preprocessing.transform(data_numpy[part]['x_cont'])


# Label preprocessing.
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


Y_train = data_numpy['train']['y'].copy()
if task_type == 'regression':
    # For regression tasks, it is highly recommended to standardize the training labels.
    regression_label_stats = RegressionLabelStats(
        Y_train.mean().item(), Y_train.std().item()
    )
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
else:
    regression_label_stats = None


X_train = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/X_train_GR-III.csv")
X_val = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/X_val_GR-III.csv")
X_test = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/X_test_GR-III.csv")
y_train = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/y_train_GR-III.csv")
y_val = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/y_val_GR-III.csv")
y_test = pd.read_csv("FOFdata/Regression/CARVEME/50/GR-III/csv/y_test_GR-III.csv")



data_numpy = {
    'train': {'x_cont': X_train.values, 'y': y_train.values},
    'val': {'x_cont': X_val.values, 'y': y_val.values},
    'test': {'x_cont': X_test.values, 'y': y_test.values},
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data to tensors
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(y_train.values, device=device)
if task_type == 'regression':
    for part in data:
        data[part]['y'] = data[part]['y'].float()
    Y_train = Y_train.float()

# Automatic mixed precision (AMP)
# torch.float16 is implemented for completeness,
# but it was not tested in the project,
# so torch.bfloat16 is used by default.
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
# Changing False to True will result in faster training on compatible hardware.
amp_enabled = False and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

# torch.compile
compile_model = False

# fmt: off
print(
    f'Device:        {device.type.upper()}'
    f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
    f'\ntorch.compile: {compile_model}'
)
# fmt: on

# Choose one of the two configurations below.

# TabM
arch_type = 'tabm'
bins = None

# TabM-mini with the piecewise-linear embeddings.
# arch_type = 'tabm-mini'
# bins = rtdl_num_embeddings.compute_bins(data['train']['x_cont'])

model = Model(
    n_num_features=X_train.shape[1],
    cat_cardinalities=[],
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3 if bins is None else 2,
        'd_block': 512,
        'dropout': 0.1,
    },
    bins=bins,
    num_embeddings=(
        None
        if bins is None
        else {
            'type': 'PiecewiseLinearEmbeddings',
            'd_embedding': 16,
            'activation': False,
            'version': 'B',
        }
    ),
    arch_type=arch_type,
    k=32,
).to(device)
optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)

if compile_model:
    # NOTE
    # `torch.compile` is intentionally called without the `mode` argument
    # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]['x_cont'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )


base_loss_fn = F.mse_loss if task_type == 'regression' else F.cross_entropy


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions per object. Each of them must be trained separately.
    # (regression)     y_pred.shape == (batch_size, k)
    # (classification) y_pred.shape == (batch_size, k, n_classes)
    k = y_pred.shape[-1 if task_type == 'regression' else -2]
    return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))


@evaluation_mode()
def evaluate(part: str) -> float:
    model.eval()

    # When using torch.compile, you may need to reduce the evaluation batch size.
    eval_batch_size = 16
    y_pred: np.ndarray = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]['y']), device=device).split(
                    eval_batch_size
                )
            ]
        )
        .cpu()
        .numpy()
    )
    if task_type == 'regression':
        # Transform the predictions back to the original label space.
        assert regression_label_stats is not None
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

    # Compute the mean of the k predictions.
    if task_type != 'regression':
        # For classification, the mean must be computed in the probabily space.
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    y_true = data[part]['y'].cpu().numpy()
    score = (
        -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    return float(score)  # The higher -- the better.


print(f'Test score before training: {evaluate("test"):.4f}')

def evaluate_and_save_metrics(part: str, output_file="tabm_cm_III-50-results.json") -> float:
    model.eval()

    eval_batch_size = 128
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size):
            y_batch = data[part]['y'][idx].cpu().numpy()
            y_pred_batch = apply_model(part, idx).cpu().numpy()

            y_true_list.append(y_batch)
            y_pred_list.append(y_pred_batch)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    results = {}

    if task_type == 'regression':
        assert regression_label_stats is not None

        # y_pred.shape: (N, k) → average across ensemble predictions
        y_pred = y_pred.mean(axis=1)

        # Rescale back to original label space
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
        y_true = y_true * regression_label_stats.std + regression_label_stats.mean

        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results["RMSE"] = rmse
        results["MAE"] = mae
        results["R2"] = r2

        score = -rmse

    else:
        # y_pred.shape: (N, k, C) → softmax → average probabilities
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(axis=1)  # Average over k predictions in probability space
        y_pred_classes = y_pred.argmax(axis=1)

        results["Test Accuracy"] = accuracy_score(y_true, y_pred_classes)
        results["Test AUC"] = roc_auc_score(y_true, y_pred, multi_class='ovr')
        results["Test Precision"] = precision_score(y_true, y_pred_classes, average=None, zero_division=0).tolist()
        results["Test Recall"] = recall_score(y_true, y_pred_classes, average=None, zero_division=0).tolist()
        results["Test F1"] = f1_score(y_true, y_pred_classes, average=None, zero_division=0).tolist()
        results["Test MCC"] = [
            matthews_corrcoef((y_true == i).astype(int), (y_pred_classes == i).astype(int))
            for i in range(y_pred.shape[1])
        ]
        score = results["Test Accuracy"]

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

    return score
# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
n_epochs = 2000
patience = 200
# n_epochs = 1_000_000_000
# patience = 16

batch_size = 256
epoch_size = math.ceil(len(train_idx) / batch_size)
best = {
    'val': -math.inf,
    'test': -math.inf,
    'epoch': -1,
}
remaining_patience = patience
print(y_test.values)
print('-' * 88 + '\n')
for epoch in range(n_epochs):
    for batch_idx in tqdm(
        torch.randperm(len(data['train']['y']), device=device).split(batch_size),
        desc=f'Epoch {epoch}',
        total=epoch_size,
    ):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore
            grad_scaler.step(optimizer)
            grad_scaler.update()

    val_score = evaluate('val')
    test_score = evaluate('test')
    print(f'(val) {val_score:.4f} (test) {test_score:.4f}')

    if val_score > best['val']:
        print('New best epoch!')
        best = {'val': val_score, 'test': test_score, 'epoch': epoch}
        remaining_patience = patience
        torch.save(model.state_dict(), 'TabM-CM-50-GR-III.pth')
    else:
        remaining_patience -= 1
    if remaining_patience < 0:
        break

    print() 

print('\n\nResult:')
print(best)
print(f"Test score before training: {evaluate_and_save_metrics('test'):.4f}")
