import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Tuple, Dict

class FriendOrFoeDataLoader:
    """Load datasets from the Friend-Or-Foe collection."""
    
    REPO_ID = "powidla/Friend-Or-Foe"
    
    def __init__(self):
        self.available_datasets = self._get_available_datasets()
    
    def load_dataset(self, task: str, collection: str, group: str, dataset: str) -> Dict[str, pd.DataFrame]:
        """
        Load a specific dataset.
        
        Args:
            task: 'Classification' or 'Regression'
            collection: 'AGORA' or 'CARVEME'  
            group: '50' or '100'
            dataset: Dataset identifier (e.g., 'BC-I')
            
        Returns:
            Dictionary containing train, validation, and test splits
        """
        base_path = f"{task}/{collection}/{group}/{dataset}"
        
        files = {
            'X_train': f"{base_path}/X_train_{dataset}-{group}.csv",
            'X_val': f"{base_path}/X_val_{dataset}-{group}.csv", 
            'X_test': f"{base_path}/X_test_{dataset}-{group}.csv",
            'y_train': f"{base_path}/y_train_{dataset}-{group}.csv",
            'y_val': f"{base_path}/y_val_{dataset}-{group}.csv",
            'y_test': f"{base_path}/y_test_{dataset}-{group}.csv"
        }
        
        data = {}
        for key, path in files.items():
            try:
                data[key] = pd.read_csv(hf_hub_download(
                    repo_id=self.REPO_ID, 
                    filename=path, 
                    repo_type="dataset"
                ))
            except Exception as e:
                print(f"Failed to load {key}: {e}")
                
        return data
