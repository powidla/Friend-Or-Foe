# Welcome to the Friend or Foe repository! 



<div align="center">
  <img src="https://github.com/powidla/Friend-Or-Foe/blob/main/assets/cartoon_v2.png?raw=true" alt="Logo" width="500"/>
</div>


FriendOrFoe is a collection of environmental datasets obtained from [metabolic modeling](https://www.biorxiv.org/content/10.1101/2024.07.03.601864v1.abstract) of microbial communities [AGORA](https://www.nature.com/articles/nbt.3703) and [CARVEME](https://academic.oup.com/nar/article/46/15/7542/5042022).  FriendOrFoe gathers 64 tabular datasets (16 for AGORA with 100 additional compounds, 16 for AGORA with 50 additional compounds, 16 for CARVEME with 100 additional compounds, 16 for CARVEME with 50 additional compounds), which were constructed by analysing more than 10 000 pairs of microbes. Our collection is suitable for four machine learning frameworks.
![Logo](https://github.com/powidla/Friend-Or-Foe/blob/main/assets/forgit.png?raw=true) 
# Repository structure

- examples: provides notebooks with examples on various tasks
- exp: stores $\texttt{.json}$ files with final metrics
- models: contains codes, environments and $\texttt{.json}$ files for the experiments that were not executed in a notebook format

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
# Baseline Demo Notebooks
#### Quickstart notebook
We provide an [end-to-end example](https://github.com/powidla/Friend-Or-Foe/blob/main/EndtoEnd_example.ipynb) on how to predict competitive and cooperative interactions with TabNet.

#### Examples

The notebooks contain a simple example of using baseline models for predicting microbial interactions.

- [Supervised models](https://github.com/powidla/Friend-Or-Foe/tree/main/examples/Supervised)

- [Unsupervised models](https://github.com/powidla/Friend-Or-Foe/tree/main/examples/Supervised)

- [Generative models](https://github.com/powidla/Friend-Or-Foe/tree/main/examples/Generative)

# Reproducing the results

### Supervised models

#### TabM
To train and test TabM we followed an [example](https://github.com/yandex-research/tabm/blob/main/example.ipynb). 
`````bash
mamba env create -f tabm.yaml
mkdir FOFdata
python main.py 

`````

#### FT-Transformer
To train and test FT-Transformer we followed an [example](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/example.ipynb). 
`````bash
mamba env create -f ft.yaml
mkdir FOFdata
python main.py 

`````
#### TabNet
To train and test TabNet we followed instructions from the [package](https://dreamquark-ai.github.io/tabnet/). 
`````bash
mamba env create -f tabnet.yaml
mkdir FOFdata
python main.py 

`````
#### GBDTs
`````bash
mamba env create -f gbdts.yaml
mkdir FOFdata
python main.py 

`````
### Supervised models

### Generative models

#### TVAE, CTGAN and TabDDPM

To test TVAE, CTGAN and TabDDPM we used [synthcity](https://github.com/vanderschaarlab/synthcity) package and adapted officially provided [examples](https://github.com/vanderschaarlab/synthcity/tree/main/tutorials/plugins/generic). We calculated $\alpha$-Precision and $\beta$-Recall by using $\texttt{eval statistical}$ from $\texttt{synthcity.metrics}$.

#### TabDiff

To train and test TabDiff we followed the [guidelines](https://github.com/MinkaiXu/TabDiff). The example we used for the AGORA50 dataset is below
`````bash
git clone https://github.com/MinkaiXu/TabDiff
mamba env create -f tabdiff.yaml
cd data
mkdir GenAGORA50
python process_dataset.py --dataname GenAGORA50
python main.py --dataname GenAGORA50 --mode train --no_wandb --non_learnable_schedule --exp_name GenAGORA50

`````
To evaluate and calc metrics 
`````bash
mamba env create -f synthcity.yaml
cd Info
cp info.json
python main.py --dataname GenAGORA50 --mode test --report --no_wandb

`````
# License
FriendOrFoe is under the Apache 2.0 license for code found on the associated GitHub repo and the Creative Commons Attribution 4.0 license for data hosted on HuggingFace. The LICENSE file for the repo can be found in the top-level directory.
