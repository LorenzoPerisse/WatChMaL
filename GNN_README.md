
## Avec cuda 12.1
conda config --add channels conda-forge
conda install python=3.10

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