#!/bin/bash

# SLURM options:
#SBATCH --mail-user=er.leblevec@gmail.com           # Where to send mail
#SBATCH --mail-type=ALL                             # Mail events (NONE, BEGIN, END, FAIL, ALL)


#SBATCH --job-name=sweep_gpus                         # Job name
#SBATCH --output=/sps/t2k/eleblevec/updated_watchmal/WatChMaL_Sweep/logs/sweep_gpus%j.log             # Standard output and error log

#SBATCH --partition=gpu                              # Partition choice
#SBATCH --mem=10G                                     # Amount of memory required
#SBATCH --gres=gpu:v100:2                          # Number and type of gpu

#SBATCH --ntasks=1                                   # Maximum number of parallel processes
#SBATCH --time=0:10:00                              # 7 days by default on htc partition

export HYDRA_FULL_ERROR=1
export SWEEP_ID="ja4dnx1o"

bash run_agent.sh