#!/bin/bash

source $work/miniconda3/bin/activate grant_cuda_12_1

# For now
SWEEP_ID=${SWEEP_ID:-""} # "" works for no sweep
ENTITY=erwanlbv
PROJECT=sweep-playground
COUNT=3

  
#  wandb sweep \
#     -e $ENTITY \
#     -p $PROJECT \
#     sweep_config.yaml
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to recognize the sweep id : ${SWEEP_ID}, exiting the code." 
    exit 1
fi 

echo "Sweep ID: $SWEEP_ID"
echo "Project : $PROJECT"
echo "Entity : $ENTITY"


wandb agent \
    -e $ENTITY \
    -p $PROJECT \
    --count $COUNT \
    $SWEEP_ID


