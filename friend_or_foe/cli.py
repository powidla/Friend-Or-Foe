# friend_or_foe/cli.py
import argparse
import sys
from pathlib import Path
import json
from typing import Dict, Any

from .data.loader import FriendOrFoeDataLoader
from .models.base import TabNetModel


def main():
    """Main CLI entry point for Friend-Or-Foe package."""
    parser = argparse.ArgumentParser(
        description="Friend-Or-Foe: Microbial Interaction Dataset Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  friend-or-foe list-datasets
  
  # Download a specific dataset
  friend-or-foe download --task Classification --collection AGORA --group 100 --dataset BC-I
  
  # Download all datasets
  friend-or-foe download-all --output-dir ./FOFdata
  
  # Run a quick experiment
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model tabnet
  
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
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run a quick experiment')
    exp_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    exp_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    exp_parser.add_argument('--group', required=True, choices=['50', '100'])
    exp_parser.add_argument('--dataset', required=True, help='Dataset identifier')
    exp_parser.add_argument('--model', default='tabnet', choices=['tabnet'], help='Model to use')
    exp_parser.add_argument('--output-file', help='Save results to JSON file')
    
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
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


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
    """Handle info command."""
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
    """Handle experiment command with all model types."""
    print(f"Running experiment with {args.model} on {args.task}/{args.collection}/{args.group}/{args.dataset}")
    
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
    
    # Initialize model based on selection
    if args.model == 'tabnet':
        model = TabNetModel(n_d=32, n_a=32, n_steps=3)
    elif args.model == 'xgboost':
        model = XGBoostModel(n_estimators=200, max_depth=6, learning_rate=0.1)
    elif args.model == 'lightgbm':
        model = LightGBMModel(n_estimators=200, num_leaves=31, learning_rate=0.1)
    elif args.model == 'catboost':
        model = CatBoostModel(iterations=200, depth=6, learning_rate=0.1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train, X_val, y_val, task_type=args.task.lower())
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, task_type=args.task.lower())
    
    # Display results
    print("\nResults:")
    print("-" * 20)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model if requested
    model_save_path = f"model_{args.model}_{args.task}_{args.collection}_{args.group}_{args.dataset}.pkl"
    model.save_model(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save results if requested
    if args.output_file:
        results = {
            'dataset': f"{args.task}/{args.collection}/{args.group}/{args.dataset}",
            'model': args.model,
            'metrics': metrics,
            'data_info': {
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            },
            'model_path': model_save_path
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
