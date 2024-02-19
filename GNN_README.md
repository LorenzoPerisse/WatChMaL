# Python at least 3.10 is required

# To update your python version is a virtual env :
conda activate "env_name"
conda install python=3.10

# Add the conda-froge channel
conda config --add channels conda-forge # Set conda-forge channel to the higher priority

# Download torch for cuda 11.8 (the cuda version on the CC Lyon)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Dowload pyg
conda install pyg -c pyg
conda install pytorch-cluster -c pyg # for clustering, like knn_graph
conda install tabulate # for torch_geometric.nn.summary

# Dowload uproot
pip install uproot


# Additional librairies
conda install seaborn -c conda-forge