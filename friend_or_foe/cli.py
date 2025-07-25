# friend_or_foe/cli.py
import argparse
import sys
from pathlib import Path
import json
import pandas as pd

from .data.loader import FriendOrFoeDataLoader
from .model.base import TabNetModel, XGBoostModel, LightGBMModel, CatBoostModel


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
  
  # Run experiments with different models
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model xgboost
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model lightgbm
  friend-or-foe experiment --task Classification --collection AGORA --group 100 --dataset BC-I --model catboost
  
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
    
    # Experiment command with all models
    exp_parser = subparsers.add_parser('experiment', help='Run a quick experiment')
    exp_parser.add_argument('--task', required=True, choices=['Classification', 'Regression'])
    exp_parser.add_argument('--collection', required=True, choices=['AGORA', 'CARVEME'])
    exp_parser.add_argument('--group', required=True, choices=['50', '100'])
    exp_parser.add_argument('--dataset', required=True, help='Dataset identifier')
    exp_parser.add_argument('--model', default='xgboost', 
                           choices=['tabnet', 'xgboost', 'lightgbm', 'catboost'], 
                           help='Model to use')
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
        elif args.command == 'shap':
            handle_shap_analysis(loader, args)
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


def handle_shap_analysis(loader: FriendOrFoeDataLoader, args):
    """Handle SHAP analysis command."""
    print(f"🔍 Performing SHAP analysis on {args.model_type} model")
    print(f"Model path: {args.model_path}")
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Load the dataset
    print("📊 Loading dataset...")
    data = loader.load_dataset(args.task, args.collection, args.group, args.dataset)
    
    # Combine train and validation data for background
    X_background = pd.concat([data['X_train'], data['X_val']], ignore_index=True)
    
    # Use test data for explanation (sample if too large)
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
    print("Casting SHAP analysis...")
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
        print(f"\n📋 Top 10 Most Important Features (Native Model Importance):")
        print("-" * 60)
        top_native = native_importance.head(10)
        for idx, row in top_native.iterrows():
            print(f"{idx+1:2d}. {row['feature']:<25} | Importance: {row['importance']:.6f}")
    except Exception as e:
        print(f"⚠️ Could not get native feature importance: {e}")
    
    # Save results if save path is provided
    if args.save_path:
        print(f"\n💾 Saving results...")
        
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
        
        # Save analysis metadata
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
    
    print(f"\n SHAP analysis completed for {args.model_type} model!")
    if args.save_path:
        print(f"All results saved with prefix: {args.save_path}")


if __name__ == '__main__':
    main()
