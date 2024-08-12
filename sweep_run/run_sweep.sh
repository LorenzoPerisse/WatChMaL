# For now
SWEEP_ID="dnyqyqt9" # "" works for no sweep
ENTITY=cehrhardt
PROJECT=WatChMaL
COUNT=10

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


