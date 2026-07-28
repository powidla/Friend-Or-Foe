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
from sklearn.model_selection import train_test_split
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
    TASKS = ["Classification", "Regression", "Transfer Learning"]
    COLLECTIONS = ["AGORA", "CARVEME"] 
    GROUPS = ["50", "100"]

    METADATA_TASKS = {"Classification", "Regression", "Transfer Learning"}
    METADATA_DIR = "utils/raw"

    # Collection abbreviations used in filenames
    COLLECTION_ABBREV = {
        "AGORA": "AG",
        "CARVEME": "CM",
    }

    # Tasks whose filenames include the group suffix (e.g. BC-I-100)
    # Tasks NOT in this set use bare dataset names (e.g. GR-I, TL-I)
    TASKS_WITH_GROUP_SUFFIX = {"Classification"}
    
    # Common dataset identifiers
    CLASSIFICATION_DATASETS = [
        "BC-I", "BC-II", "BC-III", "BC-IV",
        "GR-I", "GR-II", "GR-III", "GR-IV",
        "CC-I", "CC-II", "CC-III", "CC-IV",
        "AM-I", "AM-II", "AM-III", "AM-IV"
    ]
    
    REGRESSION_DATASETS = [
        "GR-I", "GR-II", "GR-III"
    ]

    TRANSFER_DATASETS = [
        "TL-I", "TL-II"
    ]
    
    CLUSTERING_DATASETS = [
        "US-I", "US-II"
    ]
    
    GENERATIVE_DATASETS = [
        "GEN"
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_repo_files(self) -> List[str]:
        '''Get list of all files in the repository.'''
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

    def _collection_abbrev(self, collection: str) -> str:
        '''Return the two-letter abbreviation for a collection name.'''
        return self.COLLECTION_ABBREV.get(collection, collection[:2].upper())

    def _dataset_suffix(self, task: str, dataset: str, group: str) -> str:
        '''
        Return the suffix used in split filenames for a given task.
        '''
        if task in self.TASKS_WITH_GROUP_SUFFIX:
            return f"{dataset}-{group}"
        return dataset

    def _task_dir_name(self, task: str) -> str:
        '''Return the top-level directory name used in the repo for a task.'''
        mapping = {
            "Transfer Learning": "Transfer Learning",
            "Classification": "Classification",
            "Regression": "Regression",
        }
        return mapping.get(task, task)
        
    def _metadata_filename(self, dataset: str, group: str, collection: str) -> str:
        '''Return filename: {dataset}-{group}-{collection_abbrev}-MiMj.csv'''
        abbrev = self._collection_abbrev(collection)
        return f"{dataset}-{group}-{abbrev}-MiMj.csv"

    def _metadata_path(self, dataset: str, group: str, collection: str) -> str:
        '''Full repo-relative path to the MiMj metadata file.'''
        return f"{self.METADATA_DIR}/{self._metadata_filename(dataset, group, collection)}"
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available_datasets(self, task: Optional[str] = None, collection: Optional[str] = None, group: Optional[str] = None) -> Dict[str, List[str]:
        '''
        List all available datasets with optional filtering.
        
        Args:
            task: Filter by task type ('Classification', 'Regression', 'Transfer Learning',
                  'Clustering', or 'Generative')
            collection: Filter by collection ('AGORA' or 'CARVEME')
            group: Filter by group ('50' or '100')
            
        Returns:
            Dictionary mapping dataset identifiers to their file paths.
        '''
        files = self._get_repo_files()
        datasets = {}
        
        for file_path in files:
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

            # Only index known file types
            if not (file_path.endswith('.csv') or file_path.endswith('.npy')):
                continue
                
            filename = parts[-1]
            if '_' in filename:
                dataset_id = filename.split('_')[-1].replace('.csv', '').replace('.npy', '').split('-')[0:2]
                if len(dataset_id) >= 2:
                    dataset_key = f"{file_task}/{file_collection}/{file_group}/{'-'.join(dataset_id)}"
                    if dataset_key not in datasets:
                        datasets[dataset_key] = []
                    datasets[dataset_key].append(file_path)
        
        return datasets

    def load_metadata(self, collection: str, group: str, dataset: str, task: Optional[str] = None, rename_columns: bool = True) -> pd.DataFrame:
        '''
        Load the MiMj microbe-pair metadata file, e.g.
        utils/raw/BC-I-100-AG-MiMj.csv for BC-I/AGORA/100.
        Columns: index of Mi, index of Mj, split (train/val/test).
        '''
        if task is not None and task not in self.METADATA_TASKS:
            raise ValueError(
                f"Metadata (MiMj) files are only available for {sorted(self.METADATA_TASKS)}, "
                f"not '{task}'."
            )
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection must be one of {self.COLLECTIONS}")
        if group not in self.GROUPS:
            raise ValueError(f"Group must be one of {self.GROUPS}")

        file_path = self._metadata_path(dataset, group, collection)

        if self.verbose:
            print(f"Loading metadata: {file_path}")

        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=file_path,
            repo_type="dataset",
            cache_dir=self.cache_dir
        )
        meta = pd.read_csv(local_path)

        if rename_columns and len(meta.columns) >= 3:
            renamed = dict(zip(meta.columns[:2], ["Mi", "Mj"]))
            third_col = meta.columns[2]
            if third_col.lower() != "split":
                renamed[third_col] = "split"
            meta = meta.rename(columns=renamed)

        if self.verbose:
            print(f"  Metadata shape: {meta.shape}")

        return meta
    
    def load_dataset(self, task: str, collection: str, group: str, dataset: str, splits: Optional[List[str]] = None, download_metadata: bool = False) -> Dict[str, pd.DataFrame]:
        '''
        Load a Classification, Regression, or Transfer dataset with all its splits.

        Args:
            task: Task type ('Classification', 'Regression', or 'Transfer Learning')
            collection: Collection type ('AGORA' or 'CARVEME')
            group: Group identifier ('50' or '100') 
            dataset: Dataset identifier (e.g., 'BC-I', 'GR-III', 'TL-I')
            splits: List of splits to load. Default: ['train', 'val', 'test']
            
        Returns:
            Dictionary with keys like 'X_train', 'y_train', 'X_val', etc.
            
        Example:
            >>> loader = FriendOrFoeDataLoader()
            >>> # Classification
            >>> data = loader.load_dataset('Classification', 'AGORA', '100', 'BC-I')
            >>> # Regression
            >>> data = loader.load_dataset('Regression', 'AGORA', '100', 'GR-I')
            >>> # Transfer Learning
            >>> data = loader.load_dataset('Transfer Learning', 'AGORA', '100', 'TL-I')
            >>> X_train = data['X_train']
            >>> y_train = data['y_train']
        '''
        valid_tasks = ("Classification", "Regression", "Transfer Learning")
        if task not in valid_tasks:
            raise ValueError(
                f"Task must be one of {valid_tasks}. "
                "For other tasks use load_generative_dataset or load_clustering_dataset."
            )
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection must be one of {self.COLLECTIONS}")
        if group not in self.GROUPS:
            raise ValueError(f"Group must be one of {self.GROUPS}")
            
        if splits is None:
            splits = ['train', 'val', 'test']

        suffix = self._dataset_suffix(task, dataset, group)
        task_dir = self._task_dir_name(task)
        base_path = f"{task_dir}/{collection}/{group}/{dataset}"
        
        file_mapping = {}
        for split in splits:
            for data_type in ['X', 'y']:
                key = f"{data_type}_{split}"
                filename = f"{key}_{suffix}.csv"
                file_mapping[key] = f"{base_path}/{filename}"
        
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
                    print(f"  Feature columns: {list(data[key].columns[:5])}"
                          f"{'...' if len(data[key].columns) > 5 else ''}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load {key} from {file_path}: {e}")
                
        if download_metadata:                                                    
            try:
                data['metadata'] = self.load_metadata(collection, group, dataset, task=task)
            except Exception as e:
                warnings.warn(f"Failed to load metadata for {task}/{collection}/{group}/{dataset}: {e}")
                
        return data

    # ------------------------------------------------------------------
    # load_generative_dataset
    # ------------------------------------------------------------------

    def load_generative_dataset(self, collection: str, group: str, splits: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        '''
        Load a Generative dataset.

        File naming convention:
            df_train_{coll_abbrev}-{group}.csv   e.g. df_train_AG-100.csv
            df_test_{coll_abbrev}-{group}.csv    e.g. df_test_CM-50.csv

        Args:
            collection: Collection type ('AGORA' or 'CARVEME')
            group: Group identifier ('50' or '100')
            splits: Splits to load. Default: ['train', 'test']

        Returns:
            Dictionary with keys 'df_train' and/or 'df_test' mapping to DataFrames.

        Example:
            >>> loader = FriendOrFoeDataLoader()
            >>> data = loader.load_generative_dataset('AGORA', '100')
            >>> df_train = data['df_train']
        '''
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection must be one of {self.COLLECTIONS}")
        if group not in self.GROUPS:
            raise ValueError(f"Group must be one of {self.GROUPS}")

        if splits is None:
            splits = ['train', 'test']

        abbrev = self._collection_abbrev(collection)
        base_path = f"Generative/{collection}/{group}/GEN"

        file_mapping = {}
        for split in splits:
            key = f"df_{split}"
            filename = f"df_{split}_{abbrev}-{group}.csv"
            file_mapping[key] = f"{base_path}/{filename}"

        data = {}

        if self.verbose:
            print(f"Loading Generative dataset: {collection}/{group}")

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

                if self.verbose:
                    print(f"  {key} shape: {data[key].shape}")

            except Exception as e:
                warnings.warn(f"Failed to load {key} from {file_path}: {e}")

        return data

    # ------------------------------------------------------------------
    # load_clustering_dataset
    # ------------------------------------------------------------------

    def load_clustering_dataset(self, collection: str, group: str, dataset: str) -> Dict[str, np.ndarray]:
        '''
        Load a Clustering dataset (stored as .npy files).

        File naming convention:
            {coll_abbrev}_{dataset}-{group}.npy   e.g. AG_US-I-100.npy

        Args:
            collection: Collection type ('AGORA' or 'CARVEME')
            group: Group identifier ('50' or '100')
            dataset: Dataset identifier (e.g., 'US-I', 'US-II')

        Returns:
            Dictionary with key 'data' mapping to a numpy array.

        Example:
            >>> loader = FriendOrFoeDataLoader()
            >>> data = loader.load_clustering_dataset('AGORA', '50', 'US-II')
            >>> X = data['data']
        '''
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection must be one of {self.COLLECTIONS}")
        if group not in self.GROUPS:
            raise ValueError(f"Group must be one of {self.GROUPS}")

        abbrev = self._collection_abbrev(collection)
        base_path = f"Clustering/{collection}/{group}/{dataset}"
        filename = f"{abbrev}_{dataset}-{group}.npy"
        file_path = f"{base_path}/{filename}"

        if self.verbose:
            print(f"Loading Clustering dataset: {collection}/{group}/{dataset}")
            print(f"  File: {filename}")

        try:
            local_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=file_path,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            array = np.load(local_path, allow_pickle=True)

            if self.verbose:
                print(f"  Array shape: {array.shape}, dtype: {array.dtype}")

            return {"data": array}

        except Exception as e:
            warnings.warn(f"Failed to load clustering dataset from {file_path}: {e}")
            return {}
    
    def load_multiple_datasets(self, configurations: List[Tuple[str, str, str, str]]) -> Dict[str, Dict]:
        '''
        Load multiple datasets at once.

        Args:
            configurations: List of (task, collection, group, dataset) tuples.
                            task may be 'Classification', 'Regression', 'Transfer Learning',
                            'Generative', or 'Clustering'.
        Returns:
            Dictionary mapping "{task}/{collection}/{group}/{dataset}" to dataset dicts.
        '''
        all_data = {}
        
        for config in tqdm(configurations, desc="Loading datasets", disable=not self.verbose):
            task, collection, group, dataset = config
            config_key = f"{task}/{collection}/{group}/{dataset}"
            
            try:
                if task in ("Classification", "Regression", "Transfer Learning"):
                    all_data[config_key] = self.load_dataset(task, collection, group, dataset)
                elif task == "Generative":
                    all_data[config_key] = self.load_generative_dataset(collection, group)
                elif task == "Clustering":
                    all_data[config_key] = self.load_clustering_dataset(collection, group, dataset)
                else:
                    warnings.warn(f"Unsupported task '{task}' for {config_key}; skipping.")
            except Exception as e:
                warnings.warn(f"Failed to load {config_key}: {e}")
                
        return all_data

    def get_dataset_info(self, task: str, collection: str, group: str, dataset: str) -> Dict:
        '''
        Dictionary with dataset metadata.
        '''
        try:
            abbrev = self._collection_abbrev(collection)

            if task in ("Classification", "Regression", "Transfer Learning"):
                suffix = self._dataset_suffix(task, dataset, group)
                task_dir = self._task_dir_name(task)
                base_path = f"{task_dir}/{collection}/{group}/{dataset}"
                sample_file = f"{base_path}/X_train_{suffix}.csv"
                local_path = hf_hub_download(
                    repo_id=self.REPO_ID, filename=sample_file,
                    repo_type="dataset", cache_dir=self.cache_dir
                )
                df = pd.read_csv(local_path, nrows=5)
                return {
                    "task": task, "collection": collection,
                    "group": group, "dataset": dataset,
                    "n_features": len(df.columns),
                    "feature_names": list(df.columns),
                    "sample_shape": df.shape,
                    "dtypes": df.dtypes.to_dict(),
                }

            elif task == "Generative":
                base_path = f"Generative/{collection}/{group}/GEN"
                sample_file = f"{base_path}/df_train_{abbrev}-{group}.csv"
                local_path = hf_hub_download(
                    repo_id=self.REPO_ID, filename=sample_file,
                    repo_type="dataset", cache_dir=self.cache_dir
                )
                df = pd.read_csv(local_path, nrows=5)
                return {
                    "task": task, "collection": collection,
                    "group": group, "dataset": "GEN",
                    "n_features": len(df.columns),
                    "feature_names": list(df.columns),
                    "sample_shape": df.shape,
                    "dtypes": df.dtypes.to_dict(),
                }

            elif task == "Clustering":
                base_path = f"Clustering/{collection}/{group}/{dataset}"
                sample_file = f"{base_path}/{abbrev}_{dataset}-{group}.npy"
                local_path = hf_hub_download(
                    repo_id=self.REPO_ID, filename=sample_file,
                    repo_type="dataset", cache_dir=self.cache_dir
                )
                array = np.load(local_path, allow_pickle=True)
                return {
                    "task": task, "collection": collection,
                    "group": group, "dataset": dataset,
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                }

            else:
                return {"error": f"get_dataset_info not supported for task '{task}'"}

        except Exception as e:
            return {"error": str(e)}
    
    def create_train_test_split(self, data: Dict[str, pd.DataFrame], test_size: float = 0.2, random_state: int = 4221) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    
    # ------------------------------------------------------------------
    # download_all_datasets
    # ------------------------------------------------------------------

    def download_all_datasets(self, output_dir: str = "FOFdata", download_metadata: bool = False):
        '''
        Download all datasets and organise them in the expected directory structure.
        
        Args:
            output_dir: Root directory to save all datasets.
        '''
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        datasets = self.list_available_datasets()
        
        if self.verbose:
            print(f"Downloading {len(datasets)} datasets to {output_dir}")
        
        for dataset_key in tqdm(datasets.keys(), desc="Downloading datasets"):
            task, collection, group, dataset = dataset_key.split('/', 3)
            dataset_dir = output_path / task / collection / group / dataset / "csv"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if task in ("Classification", "Regression", "Transfer Learning"):
                    data = self.load_dataset(task, collection, group, dataset)
                    suffix = self._dataset_suffix(task, dataset, group)
                    for key, df in data.items():
                        df.to_csv(dataset_dir / f"{key}_{suffix}.csv", index=False)
                    if download_metadata:                               
                        try:
                            meta = self.load_metadata(collection, group, dataset, task=task)
                            meta.to_csv(dataset_dir / self._metadata_filename(dataset, group, collection), index=False)
                        except Exception as e:
                            warnings.warn(f"Failed to download metadata for {dataset_key}: {e}")

                elif task == "Generative":
                    data = self.load_generative_dataset(collection, group)
                    for key, df in data.items():
                        df.to_csv(dataset_dir / f"{key}.csv", index=False)

                elif task == "Clustering":
                    data = self.load_clustering_dataset(collection, group, dataset)
                    npy_dir = output_path / task / collection / group / dataset
                    npy_dir.mkdir(parents=True, exist_ok=True)
                    if "data" in data:
                        abbrev = self._collection_abbrev(collection)
                        np.save(npy_dir / f"{abbrev}_{dataset}-{group}.npy", data["data"])

            except Exception as e:
                warnings.warn(f"Failed to download {dataset_key}: {e}")


# Utility functions
def quick_load(task: str = "Classification", collection: str = "AGORA", group: str = "100", dataset: str = "BC-I") -> Dict:
    '''
    Quick utility to load a Classification, Regression, or Transfer dataset.

    Args:
        task: Task type ('Classification', 'Regression', or 'Transfer')
        collection: Collection type (default: 'AGORA') 
        group: Group identifier (default: '100')
        dataset: Dataset identifier (default: 'BC-I')
        
    Returns:
        Dictionary containing the loaded dataset splits.

    Examples:
        >>> quick_load('Regression', 'AGORA', '100', 'GR-I')
        >>> quick_load('Transfer', 'AGORA', '100', 'TL-I')
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.load_dataset(task, collection, group, dataset)


def quick_load_generative(collection: str = "AGORA", group: str = "100") -> Dict:
    '''
    Quick utility to load a Generative dataset.

    Args:
        collection: Collection type (default: 'AGORA')
        group: Group identifier (default: '100')

    Returns:
        Dictionary with 'df_train' and 'df_test' DataFrames.
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.load_generative_dataset(collection, group)


def quick_load_clustering(collection: str = "AGORA", group: str = "50", dataset: str = "US-II") -> Dict:
    '''
    Quick utility to load a Clustering dataset.

    Args:
        collection: Collection type (default: 'AGORA')
        group: Group identifier (default: '50')
        dataset: Dataset identifier (default: 'US-II')

    Returns:
        Dictionary with key 'data' containing a numpy array.
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.load_clustering_dataset(collection, group, dataset)


def list_all_datasets() -> Dict[str, List[str]]:
    '''
    Quick utility to list all available datasets.
    
    Returns:
        Dictionary mapping dataset identifiers to file paths.
    '''
    loader = FriendOrFoeDataLoader(verbose=False)
    return loader.list_available_datasets()
