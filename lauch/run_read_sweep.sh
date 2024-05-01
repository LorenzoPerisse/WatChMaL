#!/bin/bash

source $work/miniconda3/bin/activate grant_cuda_12_1

# For now
SWEEP_ID=${SWEEP_ID:-""} # "" works for no sweep
ENTITY=erwanlbv
PROJECT=sweep-playground
COUNT=3


# Define the sweep configuration in a temporary YAML file

# Sweep part
if [ -z "$SWEEP_ID" ]; then
    echo "No sweep id provided. Creating a new sweep"

    # Create the sweep
    wandb sweep \
    -e $ENTITY \
    -p $PROJECT \
    sweep_config.yaml
    
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


