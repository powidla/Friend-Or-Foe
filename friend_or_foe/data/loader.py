"""
Friend-Or-Foe Data Loader Module

This module provides utilities for loading and managing the Friend-Or-Foe datasets
from the offiical Hugging Face repo: https://huggingface.co/datasets/powidla/Friend-Or-Foe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download, list_repo_files
import requests
from tqdm import tqdm
import warnings


class FriendOrFoeDataLoader:
    '''
    Load datasets from the Friend-Or-Foe collection hosted on HuggingFace.
    
    This class provides convenient methods to download and load microbial interaction
    datasets for machine learning research.
    '''

    # the repo id
    REPO_ID = "powidla/Friend-Or-Foe"
    
    # Available configurations
    TASKS = ["Classification", "Regression"]
    COLLECTIONS = ["AGORA", "CARVEME"] 
    GROUPS = ["50", "100"]
    
    # Common dataset identifiers
    CLASSIFICATION_DATASETS = [
        "BC-I", "BC-II", "BC-III", "BC-IV",
        "GR-I", "GR-II", "GR-III", "GR-IV",
        "CC-I", "CC-II", "CC-III", "CC-IV",
        "AM-I", "AM-II", "AM-III", "AM-IV"
    ]
    
    REGRESSION_DATASETS = [
        "GR-I", "GR-II", "GR-III", "GR-IV"
    ]
    
    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        '''
        Initialize the Friend-Or-Foe data loader.
        
        Args:
            cache_dir: Directory to cache downloaded files. If None, uses HuggingFace default.
            verbose: Whether to print progress information.
        '''
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._repo_files = None
        
    def _get_repo_files(self) -> List[str]:
        '''
        Get list of all files in the repository.
        '''
        if self._repo_files is None:
            if self.verbose:
                print("Fetching repository file list...")
            try:
                self._repo_files = list_repo_files(
                    repo_id=self.REPO_ID, 
                    repo_type="dataset"
                )
            except Exception as e:
                warnings.warn(f"Could not fetch repo files: {e}")
                self._repo_files = []
        return self._repo_files
    
    def list_available_datasets(self, task: Optional[str] = None, 
                              collection: Optional[str] = None,
                              group: Optional[str] = None) -> Dict[str, List[str]]:
        '''
        List all available datasets with optional filtering.
        
        Args:
            task: Filter by task type ('Classification' or 'Regression')
            collection: Filter by collection ('AGORA' or 'CARVEME')
            group: Filter by group ('50' or '100')
            
        Outputs:
            Dictionary mapping dataset identifiers to their file paths
        '''
        files = self._get_repo_files()
        datasets = {}
        
        for file_path in files:
            if not file_path.endswith('.csv'):
                continue
                
            parts = file_path.split('/')
            if len(parts) < 4:
                continue
                
            file_task, file_collection, file_group = parts[0], parts[1], parts[2]
            
            # Apply filters
            if task and file_task != task:
                continue
            if collection and file_collection != collection:
                continue  
            if group and file_group != group:
                continue
                
            # Extract dataset identifier from filename
            filename = parts[-1]
            if '_' in filename:
                dataset_id = filename.split('_')[-1].replace('.csv', '').split('-')[0:2]
                if len(dataset_id) >= 2:
                    dataset_key = f"{file_task}/{file_collection}/{file_group}/{'-'.join(dataset_id)}"
                    if dataset_key not in datasets:
                        datasets[dataset_key] = []
                    datasets[dataset_key].append(file_path)
        
        return datasets
    
    def load_dataset(self, task: str, collection: str, group: str, 
                    dataset: str, splits: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        '''
        Load a specific dataset with all its splits.
        
        Args:
            task: Task type ('Classification' or 'Regression')
            collection: Collection type ('AGORA' or 'CARVEME')
            group: Group identifier ('50' or '100') 
            dataset: Dataset identifier (e.g., 'BC-I', 'GR-III')
            splits: List of splits to load. Default: ['train', 'val', 'test']
            
        Outputs:
            Dictionary containing DataFrames for each split and data type
            
        Example:
            >>> loader = FriendOrFoeDataLoader()
            >>> data = loader.load_dataset('Classification', 'AGORA', '100', 'BC-I')
            >>> X_train = data['X_train']
            >>> y_train = data['y_train']
        '''
        # Validate inputs
        if task not in self.TASKS:
            raise ValueError(f"Task must be one of {self.TASKS}")
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection must be one of {self.COLLECTIONS}")
        if group not in self.GROUPS:
            raise ValueError(f"Group must be one of {self.GROUPS}")
            
        if splits is None:
            splits = ['train', 'val', 'test']
            
        # Construct file paths
        base_path = f"{task}/{collection}/{group}/{dataset}"
        
        file_mapping = {}
        for split in splits:
            for data_type in ['X', 'y']:
                key = f"{data_type}_{split}"
                filename = f"{key}_{dataset}-{group}.csv"
                file_mapping[key] = f"{base_path}/{filename}"
        
        # Download and load files
        data = {}
        
        if self.verbose:
            print(f"Loading dataset: {task}/{collection}/{group}/{dataset}")
            
        for key, file_path in tqdm(file_mapping.items(), 
                                 desc="Downloading files", 
                                 disable=not self.verbose):
            try:
                local_path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    cache_dir=self.cache_dir
                )
                data[key] = pd.read_csv(local_path)
                
                if self.verbose and key == f"X_{splits[0]}":
                    print(f"  Features shape: {data[key].shape}")
                    print(f"  Feature columns: {list(data[key].columns[:5])}{'...' if len(data[key].columns) > 5 else ''}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load {key} from {file_path}: {e}")
                
        return data
    
    def load_multiple_datasets(self, configurations: List[Tuple[str, str, str, str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        '''
        Load multiple datasets at once.
        
        Args:
            configurations: List of (task, collection, group, dataset) tuples
            
        Outputs:
            Dictionary mapping configuration strings to dataset dictionaries
        '''
        all_data = {}
        
        for config in tqdm(configurations, desc="Loading datasets", disable=not self.verbose):
            task, collection, group, dataset = config
            config_key = f"{task}/{collection}/{group}/{dataset}"
            
            try:
                all_data[config_key] = self.load_dataset(task, collection, group, dataset)
            except Exception as e:
                warnings.warn(f"Failed to load {config_key}: {e}")
                
        return all_data
    
    def get_dataset_info(self, task: str, collection: str, group: str, dataset: str) -> Dict:
        '''
        Get information about a specific dataset without loading it.
        
        Args:
            task: Task type
            collection: Collection type  
            group: Group identifier
            dataset: Dataset identifier
            
        Outputs:
            Dictionary with dataset metadata
        '''
        try:
            # Load just a small sample to get info
            base_path = f"{task}/{collection}/{group}/{dataset}"
            sample_file = f"{base_path}/X_train_{dataset}-{group}.csv"
            
            local_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=sample_file, 
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            
            df = pd.read_csv(local_path, nrows=5)  # Just read header + few rows
            
            return {
                "task": task,
                "collection": collection, 
                "group": group,
                "dataset": dataset,
                "n_features": len(df.columns),
                "feature_names": list(df.columns),
                "sample_shape": df.shape,
                "dtypes": df.dtypes.to_dict()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_train_test_split(self, data: Dict[str, pd.DataFrame], 
                              test_size: float = 0.2, 
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Create a simple train-test split from loaded data.
        
        Args:
            data: Dictionary containing the loaded dataset
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Outputs:
            Tuple of (X_train, X_test, y_train, y_test)
        '''
        from sklearn.model_selection import train_test_split
        
        # Combine train and val data if available
        X_combined = []
        y_combined = []
        
        for split in ['train', 'val']:
            if f'X_{split}' in data and f'y_{split}' in data:
                X_combined.append(data[f'X_{split}'])
                y_combined.append(data[f'y_{split}'])
        
        if not X_combined:
            raise ValueError("No training data found in the dataset")
            
        X = pd.concat(X_combined, ignore_index=True)
        y = pd.concat(y_combined, ignore_index=True)
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def download_all_datasets(self, output_dir: str = "FOFdata"):
        '''
        Download all datasets and organize them in the expected directory structure.
        
        Args:
            output_dir: Directory to save all datasets
        '''
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        datasets = self.list_available_datasets()
        
        if self.verbose:
            print(f"Downloading {len(datasets)} datasets to {output_dir}")
        
        for dataset_key in tqdm(datasets.keys(), desc="Downloading datasets"):
            task, collection, group, dataset = dataset_key.split('/')
            
            # Create directory structure
            dataset_dir = output_path / task / collection / group / dataset / "csv"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                data = self.load_dataset(task, collection, group, dataset)
                
                # Save each split
                for key, df in data.items():
                    filename = f"{key}_{dataset}.csv"
                    df.to_csv(dataset_dir / filename, index=False)
                    
            except Exception as e:
                warnings.warn(f"Failed to download {dataset_key}: {e}")


# Utility functions
def quick_load(task: str = "Classification", collection: str = "AGORA", 
               group: str = "100", dataset: str = "BC-I") -> Dict[str, pd.DataFrame]:
    '''
    Quick utility function to load a dataset with default parameters.
    
    Args:
        task: Task type (default: 'Classification')
        collection: Collection type (default: 'AGORA') 
        group: Group identifier (default: '100')
        dataset: Dataset identifier (default: 'BC-I')
        
    Outputs:
        Dictionary containing the loaded dataset
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.load_dataset(task, collection, group, dataset)


def list_all_datasets() -> Dict[str, List[str]]:
    '''
    Quick utility to list all available datasets.
    
    Outputs:
        Dictionary mapping dataset identifiers to file paths
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.list_available_datasets()
