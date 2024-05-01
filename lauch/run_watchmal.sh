#!/bin/bash


# ----- Settings ------

env_name=grant_cuda_12_1
miniconda_path=/sps/t2k/eleblevec/miniconda3


working_dir=/sps/t2k/eleblevec/updated_watchmal/WatChMaL_Sweep


export HYDRA_FULL_ERROR=1
# ---- Executed code -----

source ${miniconda_path}/bin/activate $env_name

cd $working_dir


python \
    main_sweep.py \
    --config-name=TEST_hk_gnn \
    'kind=gnn' \
    'gpu_list=[0, 1]'