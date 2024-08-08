#!/bin/bash

#SBATCH --mail-user=clems.ehrhardt@gmail.com            # Where to send mail
#SBATCH --mail-type=ALL                             # Mail events (NONE, BEGIN, END, FAIL, ALL)

# SLURM options:

#SBATCH --job-name=train_val_test_1.5M_noise_1000epoch                       # Job name
#SBATCH --output=/sps/t2k/cehrhardt/Caverns/logs/train_val_test_1.5M_noise_1000epoch_r30_%j.log             # Standard output and error log

#SBATCH --partition=gpu                              # Partition choice
#SBATCH --ntasks=1                                   # Maximum number of parallel processes

#SBATCH --gres=gpu:v100:2                         # Number and type of gpu
#SBATCH --mem=200G                                     # Amount of memory required

#SBATCH --time=3-00:00:00                              # 7 days by default on htc partition


## Set the paths
## Modify it according to your settings
export path_to_miniconda=/sps/t2k/cehrhardt/miniconda3
export conda_env_name=gputears

export path_to_watchmal=/sps/t2k/cehrhardt/Caverns/WatChMaL



## Executed code
source $path_to_miniconda/bin/activate $conda_env_name
conda activate $conda_env_name

export HYDRA_FULL_ERROR=1
cd $path_to_watchmal

python \
    main.py \
    --config-path=/sps/t2k/cehrhardt/Caverns/WatChMaL/config \
    --config-name=vtx_gnn \
    'gpu_list=[0,1]'
    