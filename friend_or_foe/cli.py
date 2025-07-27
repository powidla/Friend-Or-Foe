"""
CLI Friend-Or-Foe.
Currently supports training from base.py and change different parameters.
Additionally, cli.py supports SHAP.
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd

from .data.loader import FriendOrFoeDataLoader
from .model.base import TabNetModel, XGBoostModel, LightGBMModel, CatBoostModel, FTTransformerModel, TabMModel


def main():
    '''
    Main CLI entry point for Friend-Or-Foe package.
    '''
    parser = argparse.ArgumentParser(
        description="Friend-Or-Foe: Microbial Interaction Dataset Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples to use:
  # List available datasets
  friend-or-foe list-datasets
  
  # Download a specific dataset
  friend-or-foe download --task Classification --collection AGORA --group 100 --dataset BC-I
  
  # Download all datasets
  friend-or-foe download-all --output-dir ./FOFdata
  
  # Run experiments with custom parameters
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model xgboost --xgb-n-estimators 500 --xgb-learning-rate 0.05
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model lightgbm --lgb-num-leaves 50 --lgb-learning-rate 0.08
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model catboost --cb-iterations 300 --cb-depth 8
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model tabnet --tabnet-n-d 64 --tabnet-n-steps 5
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model ft_transformer --ft-d-token 256 --ft-n-blocks 4
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model tabm --tabm-lr 0.001 --tabm-max-epochs 200
  
  # Use custom parameters from JSON file
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model xgboost --params ./xgb_params.json
  
  # Use custom parameters from JSON string
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model xgboost --params '{"n_estimators": 300, "max_depth": 8}'
  
  # Perform SHAP analysis on trained models
  friend-or-foe shap --model-path ./model_xgboost.pkl --model-type xgboost --task Classification --collection AGORA --group 100 --dataset BC-I
  friend-or-foe shap --model-path ./model_lightgbm.pkl --model-type lightgbm --task Classification --collection AGORA --group 100 --dataset BC-I --plot-type waterfall
  
  # Get dataset information
  friend-or-foe info --task Classification --collection AGORA --group 100 --dataset BC-I
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List datasets command
    list_parser = subparsers.add_parser('list-datasets', help='List all available datasets')
    list_parser.add_argument('--task', choices=['Classification', 'Regression'], help='Filter by task')
    list_parser.add_argument('--collection', choices=['AGORA', 'CARVEME'], help='Filter by collection')
    list_parser.add_argument('--group', choices=['50', '100'], help='Filter by group')
    
    # Download dataset command
    download_parser = subparsers.add_parser('download', help='Download a specific dataset')
    download_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    download_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    download_parser.add_argument('--group', required=True, choices=['50', '100'])
    download_parser.add_argument('--dataset', required=True, help='Dataset identifier (e.g., BC-I)')
    download_parser.add_argument('--output-dir', default='./data', help='Output directory')
    
    # Download all datasets command
    download_all_parser = subparsers.add_parser('download-all', help='Download all datasets')
    download_all_parser.add_argument('--output-dir', default='./FOFdata', help='Output directory')
    
    # Dataset info command
    info_parser = subparsers.add_parser('info', help='Get information about a dataset')
    info_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    info_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    info_parser.add_argument('--group', required=True, choices=['50', '100'])
    info_parser.add_argument('--dataset', required=True, help='Dataset identifier')
    
    # SHAP analysis command
    shap_parser = subparsers.add_parser('shap', help='Perform SHAP analysis on a trained model')
    shap_parser.add_argument('--model-path', required=True, help='Path to saved model file')
    shap_parser.add_argument('--model-type', required=True, 
                           choices=['xgboost', 'lightgbm', 'catboost'], 
                           help='Type of model')
    shap_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    shap_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    shap_parser.add_argument('--group', required=True, choices=['50', '100'])
    shap_parser.add_argument('--dataset', required=True, help='Dataset identifier')
    shap_parser.add_argument('--plot-type', default='summary', 
                           choices=['summary', 'waterfall', 'force'], 
                           help='Type of SHAP plot to generate')
    shap_parser.add_argument('--max-display', type=int, default=20, 
                           help='Maximum number of features to display')
    shap_parser.add_argument('--save-path', help='Path to save SHAP plots')
    shap_parser.add_argument('--sample-size', type=int, default=100, 
                           help='Number of samples to use for SHAP analysis')
    
    # Experiment command with all models and custom parameters
    exp_parser = subparsers.add_parser('experiment', help='Run a quick experiment')
    exp_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    exp_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    exp_parser.add_argument('--group', required=True, choices=['50', '100'])
    exp_parser.add_argument('--dataset', required=True, help='Dataset identifier')
    exp_parser.add_argument('--model', default='xgboost', 
                           choices=['tabnet', 'xgboost', 'lightgbm', 'catboost', 'ft_transformer', 'tabm'], 
                           help='Model to use')
    exp_parser.add_argument('--output-file', help='Save results to JSON file')
    exp_parser.add_argument('--params', help='JSON string or file path with custom model parameters')
    
    # Common parameters for all models
    exp_parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    exp_parser.add_argument('--verbose', action='store_true', help='Enable verbose training output')
    
    # XGBoost specific parameters
    exp_parser.add_argument('--xgb-n-estimators', type=int, help='XGBoost: Number of estimators')
    exp_parser.add_argument('--xgb-max-depth', type=int, help='XGBoost: Maximum tree depth')
    exp_parser.add_argument('--xgb-learning-rate', type=float, help='XGBoost: Learning rate')
    exp_parser.add_argument('--xgb-subsample', type=float, help='XGBoost: Subsample ratio')
    exp_parser.add_argument('--xgb-colsample-bytree', type=float, help='XGBoost: Column subsample ratio')
    
    # LightGBM specific parameters
    exp_parser.add_argument('--lgb-n-estimators', type=int, help='LightGBM: Number of estimators')
    exp_parser.add_argument('--lgb-num-leaves', type=int, help='LightGBM: Number of leaves')
    exp_parser.add_argument('--lgb-learning-rate', type=float, help='LightGBM: Learning rate')
    exp_parser.add_argument('--lgb-feature-fraction', type=float, help='LightGBM: Feature fraction')
    exp_parser.add_argument('--lgb-bagging-fraction', type=float, help='LightGBM: Bagging fraction')
    exp_parser.add_argument('--lgb-min-child-samples', type=int, help='LightGBM: Min child samples')
    
    # CatBoost specific parameters
    exp_parser.add_argument('--cb-iterations', type=int, help='CatBoost: Number of iterations')
    exp_parser.add_argument('--cb-depth', type=int, help='CatBoost: Tree depth')
    exp_parser.add_argument('--cb-learning-rate', type=float, help='CatBoost: Learning rate')
    exp_parser.add_argument('--cb-l2-leaf-reg', type=float, help='CatBoost: L2 regularization')
    
    # TabNet specific parameters
    exp_parser.add_argument('--tabnet-n-d', type=int, help='TabNet: Width of decision prediction layer')
    exp_parser.add_argument('--tabnet-n-a', type=int, help='TabNet: Width of attention embedding')
    exp_parser.add_argument('--tabnet-n-steps', type=int, help='TabNet: Number of steps in architecture')
    exp_parser.add_argument('--tabnet-gamma', type=float, help='TabNet: Coefficient for feature reusage')
    exp_parser.add_argument('--tabnet-lambda-sparse', type=float, help='TabNet: Sparsity regularization')
    exp_parser.add_argument('--tabnet-lr', type=float, help='TabNet: Learning rate')
    exp_parser.add_argument('--tabnet-max-epochs', type=int, help='TabNet: Maximum training epochs')
    exp_parser.add_argument('--tabnet-patience', type=int, help='TabNet: Early stopping patience')
    
    # FT-Transformer specific parameters
    exp_parser.add_argument('--ft-max-epochs', type=int, help='FT-Transformer: Maximum training epochs')
    exp_parser.add_argument('--ft-patience', type=int, help='FT-Transformer: Early stopping patience')
    exp_parser.add_argument('--ft-batch-size', type=int, help='FT-Transformer: Batch size')
    exp_parser.add_argument('--ft-eval-batch-size', type=int, help='FT-Transformer: Evaluation batch size')

    
    # # TabM specific parameters
    exp_parser.add_argument('--tabm-arch-type', choices=['tabm', 'tabm-mini'], default='tabm', help='TabM: Architecture type')
    exp_parser.add_argument('--tabm-k', type=int, help='TabM: Number of ensemble members')
    exp_parser.add_argument('--tabm-n-blocks', type=int, help='TabM: Number of MLP blocks')
    exp_parser.add_argument('--tabm-d-block', type=int, help='TabM: MLP block dimension')
    exp_parser.add_argument('--tabm-dropout', type=float, help='TabM: Dropout rate')
    exp_parser.add_argument('--tabm-lr', type=float, help='TabM: Learning rate')
    exp_parser.add_argument('--tabm-weight-decay', type=float, help='TabM: Weight decay')
    exp_parser.add_argument('--tabm-max-epochs', type=int, help='TabM: Maximum training epochs')
    exp_parser.add_argument('--tabm-patience', type=int, help='TabM: Early stopping patience')
    exp_parser.add_argument('--tabm-batch-size', type=int, help='TabM: Batch size')
    exp_parser.add_argument('--tabm-eval-batch-size', type=int, help='TabM: Evaluation batch size')
    # For test
    test_parser = subparsers.add_parser('test', help='Run comprehensive test suite')
    test_parser.add_argument('--models', nargs='+', 
                            choices=['xgboost', 'lightgbm', 'catboost', 'tabnet', 'ft_transformer', 'tabm'],
                            help='Test specific models only')
    test_parser.add_argument('--tasks', nargs='+', 
                            choices=['classification', 'regression'],
                            help='Test specific tasks only')
    test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')

    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize data loader
    loader = FriendOrFoeDataLoader()
    
    try:
        if args.command == 'list-datasets':
            handle_list_datasets(loader, args)
        elif args.command == 'download':
            handle_download(loader, args)
        elif args.command == 'download-all':
            handle_download_all(loader, args)
        elif args.command == 'info':
            handle_info(loader, args)
        elif args.command == 'experiment':
            handle_experiment(loader, args)
        elif args.command == 'shap':
            handle_shap_analysis(loader, args)
        elif args.command == 'test':
            handle_test_suite(args)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def parse_model_parameters(args):
    """Parse model-specific parameters from CLI arguments."""
    import json
    
    # Start with custom params from JSON if provided
    custom_params = {}
    if args.params:
        try:
            if Path(args.params).exists():
                # Load from file
                with open(args.params, 'r') as f:
                    custom_params = json.load(f)
            else:
                custom_params = json.loads(args.params)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not parse custom parameters: {e}")
    
    # Add common parameters
    if args.random_state is not None:
        custom_params['random_state'] = args.random_state
    
    # Model-specific parameters 
    if args.model == 'xgboost':
        xgb_params = {}
        if args.xgb_n_estimators is not None:
            xgb_params['n_estimators'] = args.xgb_n_estimators
        if args.xgb_max_depth is not None:
            xgb_params['max_depth'] = args.xgb_max_depth
        if args.xgb_learning_rate is not None:
            xgb_params['learning_rate'] = args.xgb_learning_rate
        if args.xgb_subsample is not None:
            xgb_params['subsample'] = args.xgb_subsample
        if args.xgb_colsample_bytree is not None:
            xgb_params['colsample_bytree'] = args.xgb_colsample_bytree
        
        # Set defaults if not specified
        default_xgb = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': args.random_state
        }
        for key, value in default_xgb.items():
            if key not in custom_params and key not in xgb_params:
                xgb_params[key] = value
        
        custom_params.update(xgb_params)
    
    elif args.model == 'lightgbm':
        lgb_params = {}
        if args.lgb_n_estimators is not None:
            lgb_params['n_estimators'] = args.lgb_n_estimators
        if args.lgb_num_leaves is not None:
            lgb_params['num_leaves'] = args.lgb_num_leaves
        if args.lgb_learning_rate is not None:
            lgb_params['learning_rate'] = args.lgb_learning_rate
        if args.lgb_feature_fraction is not None:
            lgb_params['feature_fraction'] = args.lgb_feature_fraction
        if args.lgb_bagging_fraction is not None:
            lgb_params['bagging_fraction'] = args.lgb_bagging_fraction
        if args.lgb_min_child_samples is not None:
            lgb_params['min_child_samples'] = args.lgb_min_child_samples
        
        # Set defaults
        default_lgb = {
            'n_estimators': 200,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': args.random_state,
            'verbose': -1
        }
        for key, value in default_lgb.items():
            if key not in custom_params and key not in lgb_params:
                lgb_params[key] = value
        
        custom_params.update(lgb_params)
    
    elif args.model == 'catboost':
        cb_params = {}
        if args.cb_iterations is not None:
            cb_params['iterations'] = args.cb_iterations
        if args.cb_depth is not None:
            cb_params['depth'] = args.cb_depth
        if args.cb_learning_rate is not None:
            cb_params['learning_rate'] = args.cb_learning_rate
        if args.cb_l2_leaf_reg is not None:
            cb_params['l2_leaf_reg'] = args.cb_l2_leaf_reg
        
        # Set defaults
        default_cb = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'verbose': args.verbose
        }
        for key, value in default_cb.items():
            if key not in custom_params and key not in cb_params:
                cb_params[key] = value
        
        custom_params.update(cb_params)
    
    elif args.model == 'tabnet':
        tabnet_params = {}
        if args.tabnet_n_d is not None:
            tabnet_params['n_d'] = args.tabnet_n_d
        if args.tabnet_n_a is not None:
            tabnet_params['n_a'] = args.tabnet_n_a
        if args.tabnet_n_steps is not None:
            tabnet_params['n_steps'] = args.tabnet_n_steps
        if args.tabnet_gamma is not None:
            tabnet_params['gamma'] = args.tabnet_gamma
        if args.tabnet_lambda_sparse is not None:
            tabnet_params['lambda_sparse'] = args.tabnet_lambda_sparse
        
        # Set defaults
        default_tabnet = {
            'n_d': 32,
            'n_a': 32, 
            'n_steps': 3,
            'seed': args.random_state
        }
        for key, value in default_tabnet.items():
            if key not in custom_params and key not in tabnet_params:
                tabnet_params[key] = value
        
        custom_params.update(tabnet_params)
    
    elif args.model == 'ft_transformer':
        ft_params = {}
        if args.ft_max_epochs is not None:
            ft_params['max_epochs'] = args.ft_max_epochs
        if args.ft_patience is not None:
            ft_params['patience'] = args.ft_patience
        if args.ft_batch_size is not None:
            ft_params['batch_size'] = args.ft_batch_size
        if args.ft_eval_batch_size is not None:
            ft_params['eval_batch_size'] = args.ft_eval_batch_size
        
    # Set defaults
        default_ft = {
            'max_epochs': 100,
            'patience': 16,
            'batch_size': 256,
            'eval_batch_size': 4096,
            'random_state': args.random_state
        }
        for key, value in default_ft.items():
            if key not in custom_params and key not in ft_params:
                ft_params[key] = value
        
        custom_params.update(ft_params)
        
    elif args.model == 'tabm':
        tabm_params = {}
        if args.tabm_arch_type is not None:
            tabm_params['arch_type'] = args.tabm_arch_type
        if args.tabm_k is not None:
            tabm_params['k'] = args.tabm_k
        if args.tabm_n_blocks is not None:
            tabm_params['n_blocks'] = args.tabm_n_blocks
        if args.tabm_d_block is not None:
            tabm_params['d_block'] = args.tabm_d_block
        if args.tabm_dropout is not None:
            tabm_params['dropout'] = args.tabm_dropout
        if args.tabm_lr is not None:
            tabm_params['lr'] = args.tabm_lr
        if args.tabm_weight_decay is not None:
            tabm_params['weight_decay'] = args.tabm_weight_decay
        if args.tabm_max_epochs is not None:
            tabm_params['max_epochs'] = args.tabm_max_epochs
        if args.tabm_patience is not None:
            tabm_params['patience'] = args.tabm_patience
        if args.tabm_batch_size is not None:
            tabm_params['batch_size'] = args.tabm_batch_size
        if args.tabm_eval_batch_size is not None:
            tabm_params['eval_batch_size'] = args.tabm_eval_batch_size
        
        # Set defaults
        default_tabm = {
            'arch_type': 'tabm',
            'k': 32,
            'n_blocks': 3,
            'd_block': 512,
            'dropout': 0.1,
            'lr': 2e-3,
            'weight_decay': 3e-4,
            'max_epochs': 2000,
            'patience': 200,
            'batch_size': 256,
            'eval_batch_size': 128,
            'random_state': args.random_state
        }
        for key, value in default_tabm.items():
            if key not in custom_params and key not in tabm_params:
                tabm_params[key] = value
        
        custom_params.update(tabm_params)

    return custom_params


def handle_list_datasets(loader: FriendOrFoeDataLoader, args):
    """Handle list-datasets command."""
    datasets = loader.list_available_datasets(
        task=args.task,
        collection=args.collection, 
        group=args.group
    )
    
    print(f"Found {len(datasets)} datasets:")
    print("-" * 50)
    
    for dataset_key in sorted(datasets.keys()):
        task, collection, group, dataset = dataset_key.split('/')
        print(f"Task: {task}, Collection: {collection}, Group: {group}, Dataset: {dataset}")
        
    if not datasets:
        print("No datasets found matching the criteria.")


def handle_download(loader: FriendOrFoeDataLoader, args):
    """Handle download command."""
    print(f"Downloading dataset: {args.task}/{args.collection}/{args.group}/{args.dataset}")
    
    data = loader.load_dataset(args.task, args.collection, args.group, args.dataset)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.task / args.collection / args.group / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    for key, df in data.items():
        filename = f"{key}_{args.dataset}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    
    print(f"Dataset saved to: {output_dir}")


def handle_download_all(loader: FriendOrFoeDataLoader, args):
    """Handle download-all command."""
    print(f"Downloading all datasets to: {args.output_dir}")
    loader.download_all_datasets(args.output_dir)
    print("Download complete!")


def handle_info(loader: FriendOrFoeDataLoader, args):
    '''
    Handle info command.
    '''
    info = loader.get_dataset_info(args.task, args.collection, args.group, args.dataset)
    
    if 'error' in info:
        print(f"Error getting dataset info: {info['error']}")
        return
    
    print(f"Dataset Information:")
    print("-" * 30)
    print(f"Task: {info['task']}")
    print(f"Collection: {info['collection']}")
    print(f"Group: {info['group']}")
    print(f"Dataset: {info['dataset']}")
    print(f"Number of features: {info['n_features']}")
    print(f"Sample shape: {info['sample_shape']}")
    print(f"Feature types: {len(set(info['dtypes'].values()))} unique types")
    
    print(f"\nFirst 10 features:")
    for i, feature in enumerate(info['feature_names'][:10]):
        print(f"  {i+1}. {feature} ({info['dtypes'][feature]})")
    
    if len(info['feature_names']) > 10:
        print(f"  ... and {len(info['feature_names']) - 10} more features")


def handle_experiment(loader: FriendOrFoeDataLoader, args):
    '''
    Handle experiment command with custom model parameters.
    '''
    print(f"Running experiment with {args.model} on {args.task}/{args.collection}/{args.group}/{args.dataset}")
    
    # Parse model parameters
    model_params = parse_model_parameters(args)
    print(f"Model parameters: {model_params}")
    
    # Load data
    print("Loading dataset...")
    data = loader.load_dataset(args.task, args.collection, args.group, args.dataset)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data.get('X_val')
    y_val = data.get('y_val')
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Initialize model with custom parameters
    print(f"Initializing {args.model} model...")
    if args.model == 'tabnet':
        model = TabNetModel(**model_params)
    elif args.model == 'xgboost':
        model = XGBoostModel(**model_params)
    elif args.model == 'lightgbm':
        model = LightGBMModel(**model_params)
    elif args.model == 'catboost':
        model = CatBoostModel(**model_params)
    elif args.model == 'ft_transformer':
        model = FTTransformerModel(**model_params)
    elif args.model == 'tabm':
        model = TabMModel(**model_params)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Handle TabNet specific training parameters
    fit_params = {}
    if args.model == 'tabnet':
        if args.tabnet_max_epochs is not None:
            fit_params['max_epochs'] = args.tabnet_max_epochs
        if args.tabnet_patience is not None:
            fit_params['patience'] = args.tabnet_patience
        if args.tabnet_lr is not None:
            fit_params['lr'] = args.tabnet_lr
    
    # Train mode
    print("Training model...")
    import time
    start_time = time.time()
    
    if args.model == 'tabnet' and fit_params:
        # For TabNet, we need to pass training params differently
        model.fit(X_train, y_train, X_val, y_val, task_type=args.task.lower(), **fit_params)
    else:
        model.fit(X_train, y_train, X_val, y_val, task_type=args.task.lower())
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, task_type=args.task.lower())
    
    # Display results
    print(f"\nResults:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:>15}: {value:.6f}")
    print(f"{'training_time':>15}: {training_time:.2f}s")
    
    # Show feature importance if available (only for tree-based models)
    try:
        if hasattr(model, 'get_feature_importance') and args.model in ['xgboost', 'lightgbm', 'catboost']:
            importance = model.get_feature_importance()
            print(f"\n Top 5 Most Important Features:")
            print("-" * 30)
            for idx, row in importance.head(5).iterrows():
                print(f"{row['feature']:>20}: {row['importance']:.6f}")
    except Exception as e:
        print(f"Could not get feature importance: {e}")
    
    # Save model
    model_save_path = f"model_{args.model}_{args.task}_{args.collection}_{args.group}_{args.dataset}.pkl"
    model.save_model(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save results if requested
    if args.output_file:
        results = {
            'dataset': f"{args.task}/{args.collection}/{args.group}/{args.dataset}",
            'model': args.model,
            'model_parameters': model_params,
            'metrics': metrics,
            'training_time': training_time,
            'data_info': {
                'train_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0] if X_val is not None else 0,
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            },
            'model_path': model_save_path
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    print(f"\n Experiment completed successfully!")
    return model, metrics

def handle_test_suite(args):
    '''
    Handle test suite command.
    '''
    try:
        from .test import run_all_tests
        
        success = run_all_tests()
        
        if success:
            print("All tests passed!")
            sys.exit(0)
        else:
            print("Some tests failed!")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Test dependencies missing: {e}")
        print("Install with: pip install friend-or-foe[dev]")
        sys.exit(1)

def handle_shap_analysis(loader: FriendOrFoeDataLoader, args):
    '''
    Handle SHAP analysis command.
    '''
    print(f"Performing SHAP analysis on {args.model_type} model")
    print(f"Model path: {args.model_path}")
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Load the dataset
    print("Loading dataset...")
    data = loader.load_dataset(args.task, args.collection, args.group, args.dataset)
    
    # Combine train and validation data for background
    X_background = pd.concat([data['X_train'], data['X_val']], ignore_index=True)
    
    # Use test data
    X_explain = data['X_test']
    if len(X_explain) > args.sample_size:
        X_explain = X_explain.sample(n=args.sample_size, random_state=42)
    
    print(f"Background data: {X_background.shape}")
    print(f"Explanation data: {X_explain.shape}")
    
    # Initialize and load the model
    print("Loading trained model...")
    if args.model_type == 'xgboost':
        model = XGBoostModel()
    elif args.model_type == 'lightgbm':
        model = LightGBMModel()
    elif args.model_type == 'catboost':
        model = CatBoostModel()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load the trained model
    try:
        model.load_model(args.model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Perform SHAP analysis
    print("Performing SHAP analysis...")
    try:
        shap_results = model.shap_analysis(
            X_background=X_background,
            X_explain=X_explain,
            plot_type=args.plot_type,
            max_display=args.max_display,
            save_path=args.save_path
        )
        
        print("SHAP analysis completed successfully!")
        
    except ImportError:
        print("Error: SHAP library not found. Please install with: pip install shap")
        sys.exit(1)
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        sys.exit(1)
    
    # Display feature importance results
    print(f"\n Top {min(10, len(shap_results['feature_importance']))} Most Important Features (by SHAP):")
    print("-" * 60)
    top_features = shap_results['feature_importance'].head(10)
    for idx, row in top_features.iterrows():
        print(f"{idx+1:2d}. {row['feature']:<25} | Importance: {row['shap_importance']:.6f}")
    
    # Compare with native feature importance if available
    try:
        native_importance = model.get_feature_importance()
        print(f"\n Top 10 Most Important Features (Native Model Importance):")
        print("-" * 60)
        top_native = native_importance.head(10)
        for idx, row in top_native.iterrows():
            print(f"{idx+1:2d}. {row['feature']:<25} | Importance: {row['importance']:.6f}")
    except Exception as e:
        print(f"Could not get native feature importance: {e}")
    
    # Save 
    if args.save_path:
        print(f"\n Saving results...")
        
        # Save SHAP feature importance
        shap_importance_file = f"{args.save_path}_shap_importance.csv"
        shap_results['feature_importance'].to_csv(shap_importance_file, index=False)
        print(f"SHAP importance saved to: {shap_importance_file}")
        
        # Save native feature importance
        try:
            native_importance = model.get_feature_importance()
            native_importance_file = f"{args.save_path}_native_importance.csv"
            native_importance.to_csv(native_importance_file, index=False)
            print(f"Native importance saved to: {native_importance_file}")
        except:
            pass
        
        # Save metadata
        metadata = {
            'model_type': args.model_type,
            'model_path': args.model_path,
            'dataset': f"{args.task}/{args.collection}/{args.group}/{args.dataset}",
            'plot_type': args.plot_type,
            'max_display': args.max_display,
            'sample_size': args.sample_size,
            'background_samples': len(X_background),
            'explanation_samples': len(X_explain),
            'expected_value': float(shap_results['expected_value']) if hasattr(shap_results['expected_value'], '__float__') else str(shap_results['expected_value'])
        }
        
        metadata_file = f"{args.save_path}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Analysis metadata saved to: {metadata_file}")
    
    print(f"\nSHAP analysis completed for {args.model_type} model!")
    if args.save_path:
        print(f"All results saved with prefix: {args.save_path}")


if __name__ == '__main__':
    main()
