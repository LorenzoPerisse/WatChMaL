#!/bin/bash

export HYDRA_FULL_ERROR=1

# cd to save files "outputs" in this directory
cd /sps/t2k/cehrhardt/Caverns/WatChMaL

python \
    main.py \
    --config-path=/sps/t2k/cehrhardt/Caverns/WatChMaL/config \
    --config-name=vtx_gnn \
    'gpu_list=[0,1]'