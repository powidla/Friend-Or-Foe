# ruff: noqa: E402
import math
import random
import warnings
from typing import Literal
import argparse
import numpy as np
import scipy.special
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download
from model import Model, make_parameter_groups

# ignore warns
warnings.resetwarnings()
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--col", default="AGORA",  choices=["AGORA", "CARVEME"])
parser.add_argument("--gr",  default="100",    choices=["100", "50"])
parser.add_argument("--ds",  default="TL-I",   choices=["TL-I","TL-II"])
args = parser.parse_args()

REPO_ID = "powidla/Friend-Or-Foe"
COL = args.col
GR = args.gr
DS = args.ds

PATIENCE   = 200
BATCH_SIZE = 256
# consturction
MODEL_SAVE_PATH   = f"TabM_{COL}-{GR}-{DS}.pth"
METRICS_SAVE_PATH = f"TabM_{COL}-{GR}-{DS}_metrics.json"

_base   = f"Transfer Learning/{COL}/{GR}/{DS}"
_suffix = f"{DS}"

def _dl(filename: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")

X_train = pd.read_csv(_dl(f"{_base}/X_train_{_suffix}.csv"))
X_val   = pd.read_csv(_dl(f"{_base}/X_val_{_suffix}.csv"))
X_test  = pd.read_csv(_dl(f"{_base}/X_test_{_suffix}.csv"))
y_train = pd.read_csv(_dl(f"{_base}/y_train_{_suffix}.csv"))
y_val   = pd.read_csv(_dl(f"{_base}/y_val_{_suffix}.csv"))
y_test  = pd.read_csv(_dl(f"{_base}/y_test_{_suffix}.csv"))

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.values.ravel())  # fit only on train
y_val_enc   = le.transform(y_val.values.ravel())
y_test_enc  = le.transform(y_test.values.ravel())

N_CLASSES = len(le.classes_)  
N_EPOCHS  = 10

task_type: Literal['binclass', 'multiclass'] = 'binclass' if N_CLASSES == 2 else 'multiclass'
n_classes = N_CLASSES

seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_numpy = {
    'train': {'x_cont': X_train.values.astype(np.float32), 'y': y_train_enc},
    'val':   {'x_cont': X_val.values.astype(np.float32),   'y': y_val_enc},
    'test':  {'x_cont': X_test.values.astype(np.float32),  'y': y_test_enc},
}

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

for part in data:
    data[part]['y'] = data[part]['y'].long()

Y_train = torch.as_tensor(y_train_enc, device=device).long()

amp_dtype = (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16 if torch.cuda.is_available()
    else None
)
amp_enabled  = False and amp_dtype is not None
grad_scaler  = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None
compile_model = False

print(
    f"Run: COL={COL}  GR={GR}  DS={DS}"
    f"\nDevice:        {device.type.upper()}"
    f"\nAMP:           {amp_enabled} (dtype: {amp_dtype})"
    f"\ntorch.compile: {compile_model}"
    f"\nModel save:    {MODEL_SAVE_PATH}"
    f"\nMetrics save:  {METRICS_SAVE_PATH}"
)

model = Model(
    n_num_features=X_train.shape[1],
    cat_cardinalities=[],
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3,
        'd_block': 512,
        'dropout': 0.1,
    },
    bins=None,
    num_embeddings=None,
    arch_type='tabm',
    k=32,
).to(device)
optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)

if compile_model:
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode


@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]['x_cont'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        .squeeze(-1)
        .float()
    )


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions per object — each trained separately.
    # (classification) y_pred.shape == (batch_size, k, n_classes)
    k = y_pred.shape[-2]
    return F.cross_entropy(y_pred.flatten(0, 1), y_true.repeat_interleave(k))


@evaluation_mode()
def evaluate(part: str) -> float:
    model.eval()
    eval_batch_size = 16
    y_pred: np.ndarray = (
        torch.cat([
            apply_model(part, idx)
            for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
        ])
        .cpu().numpy()
    )
    # Softmax → average over k predictions
    y_pred = scipy.special.softmax(y_pred, axis=-1).mean(1)
    y_true = data[part]['y'].cpu().numpy()
    return float(sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1)))


def calculate_and_save_metrics(part: str, save_path: str = METRICS_SAVE_PATH):
    model.eval()
    y_pred_list, y_true_list = [], []

    with torch.no_grad():
        for idx in torch.arange(len(data[part]['y']), device=device).split(128):
            y_pred_list.append(apply_model(part, idx).cpu().numpy())
            y_true_list.append(data[part]['y'][idx].cpu().numpy())

    y_pred = np.concatenate(y_pred_list)   # (N, k, C)
    y_true = np.concatenate(y_true_list)   # (N,)

    probs  = scipy.special.softmax(y_pred, axis=-1).mean(1)  # (N, C)
    preds  = probs.argmax(1)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, preds)),
        "auc":       float(roc_auc_score(y_true, probs[:, 1] if N_CLASSES == 2 else probs,
                                         multi_class="ovr")),
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

#train
epoch_size = math.ceil(len(X_train) / BATCH_SIZE)
best = {'val': -math.inf, 'test': -math.inf, 'epoch': -1}
remaining_patience = PATIENCE

print(f'\nTest score before training: {evaluate("test"):.4f}')
print('-' * 88 + '\n')

for epoch in range(N_EPOCHS):
    for batch_idx in tqdm(
        torch.randperm(len(data['train']['y']), device=device).split(BATCH_SIZE),
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
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

    val_score  = evaluate('val')
    test_score = evaluate('test')
    print(f'(val) {val_score:.4f}  (test) {test_score:.4f}')

    if val_score > best['val']:
        print('New best epoch!')
        best = {'val': val_score, 'test': test_score, 'epoch': epoch}
        remaining_patience = PATIENCE
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        remaining_patience -= 1

    if remaining_patience < 0:
        break

    print()

print('\n\nResult:')
print(best)
