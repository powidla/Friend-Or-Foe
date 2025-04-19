# Friend-Or-Foe
Welcome to Friend or Foe repository! 
![Logo](https://github.com/powidla/Friend-Or-Foe/blob/main/assets/cartoon_v2.png?raw=true) 
FriendOrFoe is a collection of environmental datasets obtained from metabolic modeling of microbial communities [AGORA](https://www.nature.com/articles/nbt.3703) and [CARVEME](https://academic.oup.com/nar/article/46/15/7542/5042022).  FriendOrFoe gathers 64 tabular datasets (16 for AGORA with 100 additional compounds, 16 for AGORA with 50 additional compounds, 16 for CARVEME with 100 additional compounds, 16 for CARVEME with 50 additional compounds), which were constructed by analysing more than 10 000 pairs of microbes. Our collection is suitable for four machine learning frameworks.
![Logo](https://github.com/powidla/Friend-Or-Foe/blob/main/assets/forgit.png?raw=true) 
# Getting started
Download the data: https://huggingface.co/datasets/powidla/FriendOrFoe
`````python
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "powidla/Friend-Or-Foe"

# File paths within the repo
X_train_ID = "Classification/AGORA/100/BC-I/X_train_BC-I-100.csv"
X_val_ID = "Classification/AGORA/100/BC-I/X_val_BC-I-100.csv"
X_test_ID = "Classification/AGORA/100/BC-I/X_test_BC-I-100.csv"

y_train_ID = "Classification/AGORA/100/BC-I/y_train_BC-I-100.csv"
y_val_ID = "Classification/AGORA/100/BC-I/y_val_BC-I-100.csv"
y_test_ID = "Classification/AGORA/100/BC-I/y_test_BC-I-100.csv"

# Download and load CSVs as pandas DataFrames
X_train = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=X_train_ID, repo_type="dataset"))
X_val = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=X_val_ID, repo_type="dataset"))
X_test = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=X_test_ID, repo_type="dataset"))

y_train = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=y_train_ID, repo_type="dataset"))
y_val = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=y_val_ID, repo_type="dataset"))
y_test = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=y_test_ID, repo_type="dataset"))
`````
# Baseline Models

- Supervised models

- Unsupervised models

- Generative models

# Demo Notebooks
The notebooks contain a simple example of using baseline models for predicting microbial interactions.
# License
FriendOrFoe is under the Apache 2.0 license for code found on the associated GitHub repo and the Creative Commons Attribution 4.0 license for data hosted on HuggingFace. The LICENSE file for the repo can be found in the top-level directory.
