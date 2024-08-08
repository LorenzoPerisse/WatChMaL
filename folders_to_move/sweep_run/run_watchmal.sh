#!/bin/bash

# ----- Settings ------

env_name=gputears
miniconda_path=/sps/t2k/cehrhardt/miniconda3

working_dir=/sps/t2k/cehrhardt/Caverns/WatChMaL


export HYDRA_FULL_ERROR=1
# ---- Executed code -----

source ${miniconda_path}/bin/activate $env_name

cd $working_dir
python \
    main.py \
    --config-path=/sps/t2k/cehrhardt/Caverns/WatChMaL/config \
    --config-name=vtx_gnn \
    'gpu_list=[0,1]'