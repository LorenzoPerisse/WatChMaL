#!/bin/bash

# SLURM options:

#SBATCH --job-name=watchmal_training                          # Job name
#SBATCH --output=watchmal_resnet%j.log             # Standard output and error log

#SBATCH --partition=gpu                              # Partition choice
#SBATCH --ntasks=1                                   # Maximum number of parallel processes
#SBATCH --mem=20G                                     # Amount of memory required
#SBATCH --gres=gpu:v100:1                            # Number and type of gpu
#SBATCH --time=00:10:00                              # 7 days by default on htc partition

#SBATCH --mail-user=er.leblevec@gmail.com           # Where to send mail
#SBATCH --mail-type=ALL                             # Mail events (NONE, BEGIN, END, FAIL, ALL)


#conda init
conda activate grant
python /sps/t2k/eleblevec/updated_watchmal/WatChMaL/main.py --config-name=run_gnn