## Python at least 3.10 is required

### To update your python version is a virtual env :
conda activate "env_name"
conda install python=3.10

### Add the conda-forge channel
conda config --add channels conda-forge # Set conda-forge channel to the higher priority

### Download torch for cuda 11.8 (the cuda version on the CC Lyon)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
(For clustering, like knn_graph : )
conda install pytorch-cluster -c pyg   

### Dowload pyg
conda install pyg -c pyg
conda install tabulate # for torch_geometric.nn.summary
__dont do it : conda install -c pytorch torch-scatter # fast computation with global_max_pool__

### Dowload uproot
pip install uproot


### Additional librairies
conda install seaborn -c conda-forge


## Avec cuda 12.1
conda config --add channels conda-forge
conda install python=3.10


#### pytorch et nvidia
conda install pytorch==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia



#### Cuda 12.1 et dernier torch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

#### Pyg channel
conda install pyg -c pyg
conda install pytorch-cluster -c pyg 


#### Pip
pip install uproot

####  Affichage, infos..
conda install tabulate
conda install seaborn


### Pour watchmal : 
conda install hydra-core omegaconf h5py