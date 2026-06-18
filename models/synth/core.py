import argparse
import warnings
import os
import pickle
import json

# third party
import pandas as pd
# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics import eval_statistical
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Synthcity model")
    parser.add_argument("--ddpm", action="store_true", help="Run DDPM model")
    parser.add_argument("--ctgan", action="store_true", help="Run CTGAN model")
    parser.add_argument("--tvae", action="store_true", help="Run TVAE model")
    parser.add_argument("--col", default="AGORA", choices=["AGORA", "CARVEME"], help="Column type")
    parser.add_argument("--gr", default="100", choices=["100", "50"], help="Group type")
    parser.add_argument("--model_dir", default="./models", help="Directory to save models")
    return parser.parse_args()


REPO_ID = "powidla/Friend-Or-Foe"

def load_data_hf(col, gr):
    '''
    Load data from HuggingFace Hub using df_train_[COL]-[GR].csv and df_test_[COL]-[GR].csv
    '''
    def _dl(filename: str) -> str:
        return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    

    train_path = f"Generative/{col}/{gr}/GEN/df_train_AG-{gr}.csv"
    test_path = f"Generative/{col}/{gr}/GEN/df_test_AG-{gr}.csv"
    
    data_train = pd.read_csv(_dl(train_path))
    test = pd.read_csv(_dl(test_path))
    
    return data_train, test

def save_model(plugin, col, gr, model_name, model_dir):
    '''
    Save pkl
    '''
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"model_{col}_{gr}_{model_name}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(plugin, f)
    
    print(f"Saved model to: {model_path}")
    return model_path

def run_ddpm(train, test):
    plugin_params = dict(
        is_classification=False,
        n_iter=1000,
        lr=1e-6,
        weight_decay=1e-5,
        batch_size=2048,
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
    return plugin

def run_ctgan(train, test):
    plugin_params = dict(
        n_iter=1000,
        lr=0.0002,
        weight_decay=1e-4,
        batch_size=4096,
    )

    loader = GenericDataLoader(train, target_column="label")
    plugin = Plugins().get("ctgan", **plugin_params)
    plugin.fit(loader)
    return plugin

def run_tvae(train, test):
    plugin_params = dict(
        n_iter=1000,
        lr=0.0002,
        weight_decay=1e-4,
        batch_size=4096,
    )

    loader = GenericDataLoader(train, target_column="label")
    plugin = Plugins().get("tvae", **plugin_params)
    plugin.fit(loader)
    return plugin

if __name__ == "__main__":
    args = parse_args()

    train, test = load_data_hf(args.col, args.gr)
    
    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    if args.ddpm:
        print("Running DDPM...")
        plugin = run_ddpm(train, test)
        save_model(plugin, args.col, args.gr, "ddpm", args.model_dir)
    elif args.ctgan:
        print("Running CTGAN...")
        plugin = run_ctgan(train, test)
        save_model(plugin, args.col, args.gr, "ctgan", args.model_dir)
    elif args.tvae:
        print("Running TVAE...")
        plugin = run_tvae(train, test)
        save_model(plugin, args.col, args.gr, "tvae", args.model_dir)
    else:
        raise ValueError("Specify one model: --ddpm, --ctgan, or --tvae")
        
