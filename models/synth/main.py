# stdlib
import sys
import argparse
import warnings

# third party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics import eval_statistical

print("Test")
warnings.filterwarnings("ignore")

def load_data(train_path, test_path):
    data_train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return data_train, test

def evaluate_quality(real, synthetic):
    quality_eval = eval_statistical.AlphaPrecision()
    qual_res = quality_eval.evaluate(GenericDataLoader(real), synthetic)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(
        qual_res['delta_precision_alpha_naive'],
        qual_res['delta_coverage_beta_naive']
    ))
    return qual_res

def run_ddpm(train, test):
    plugin_params = dict(
        is_classification=False,
        n_iter=1000,
        lr=1e-6,
        weight_decay=1e-5,
        batch_size=100,
        model_type="mlp",
        model_params=dict(
            n_layers_hidden=3,
            n_units_hidden=256,
            dropout=0.3,
        ),
        num_timesteps=500,
        dim_embed=128,
        log_interval=10,
    )

    loader = GenericDataLoader(train, target_column="label")
    plugin = Plugins().get("ddpm", **plugin_params)
    plugin.fit(loader)
    return plugin.generate(test.shape[0])

def run_ctgan(train, test):
    plugin_params = dict(
        n_iter=1000,
        lr=0.0002,
        weight_decay=1e-4,
        batch_size=1000,
    )

    loader = GenericDataLoader(train, target_column="label")
    plugin = Plugins().get("ctgan", **plugin_params)
    plugin.fit(loader)
    return plugin.generate(test.shape[0])

def run_tvae(train, test):
    plugin_params = dict(
        n_iter=1000,
        lr=0.0002,
        weight_decay=1e-4,
        batch_size=1000,
    )

    loader = GenericDataLoader(train, target_column="label")
    plugin = Plugins().get("tvae", **plugin_params)
    plugin.fit(loader)
    return plugin.generate(test.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Synthcity model")
    parser.add_argument("--ddpm", action="store_true", help="Run DDPM model")
    parser.add_argument("--ctgan", action="store_true", help="Run CTGAN model")
    parser.add_argument("--tvae", action="store_true", help="Run TVAE model")
    parser.add_argument("--train_path", type=str, default="/FOFdata/df_trainAG-50.csv")
    parser.add_argument("--test_path", type=str, default="/FOFdata/df_testAG-50.csv")
    args = parser.parse_args()

    train, test = load_data(args.train_path, args.test_path)

    if args.ddpm:
        print("Running DDPM...")
        syn = run_ddpm(train, test)
    elif args.ctgan:
        print("Running CTGAN...")
        syn = run_ctgan(train, test)
    elif args.tvae:
        print("Running TVAE...")
        syn = run_tvae(train, test)
    else:
        raise ValueError("Specify at least one model: --ddpm, --ctgan, or --tvae")

    evaluate_quality(test, syn)
