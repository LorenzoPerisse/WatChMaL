#!/bin/bash


export HYDRA_FULL_ERROR=1

# cd to save files "outputs" in this directory
cd /sps/t2k/lperisse/Soft/WatChMaL/watchmal_caverns/WatChMaL

python \
    main.py \
    --config-path=/sps/t2k/lperisse/Soft/WatChMaL/watchmal_caverns/WatChMaL/config \
    --config-name=vtx_gnn \
    'gpu_list=[0]'

