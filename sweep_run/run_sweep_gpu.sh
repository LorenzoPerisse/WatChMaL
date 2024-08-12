#!/bin/bash

#SBATCH --mail-user=clems.ehrhardt@gmail.com            # Where to send mail
#SBATCH --mail-type=ALL                             # Mail events (NONE, BEGIN, END, FAIL, ALL)

# SLURM options:

#SBATCH --job-name=train_val_test_1.5M_noise_1000epoch                       # Job name
#SBATCH --output=/sps/t2k/cehrhardt/Caverns/logs/train_val_test_500k_noise_500epoch_%j.log             # Standard output and error log

#SBATCH --partition=gpu                              # Partition choice
#SBATCH --ntasks=1                                   # Maximum number of parallel processes

#SBATCH --gres=gpu:v100:1                       # Number and type of gpu
#SBATCH --mem=150G                                     # Amount of memory required

#SBATCH --time=1-12:00:00                              # 7 days by default on htc partition

# For now
SWEEP_ID="dnyqyqt9" # "" works for no sweep
ENTITY=cehrhardt
PROJECT=WatChMaL
COUNT=3

sweep_config=/sps/t2k/cehrhardt/Caverns/sweep_run/sweep_config.yaml


# -- Code part --- #
source /sps/t2k/cehrhardt/miniconda3/bin/activate gputears

if [ -z "$SWEEP_ID" ]; then
    echo "No sweep id provided. Creating a new sweep"

    # Create the sweep
    wandb sweep \
    -e $ENTITY \
    -p $PROJECT \
    $sweep_config
    
    echo "Provide the sweep id :"
    read SWEEP_ID
fi
  
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to recognize the sweep id : ${SWEEP_ID}, exiting the code." 
    exit 1
fi 

export HYDRA_FULL_ERROR=1

echo "Sweep ID: $SWEEP_ID"
wandb agent \
    -e $ENTITY \
    -p $PROJECT \
    --count $COUNT \
    $SWEEP_ID


