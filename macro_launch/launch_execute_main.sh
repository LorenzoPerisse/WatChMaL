#!/bin/bash

folder=`dirname $0`
cd $folder
echo "Current directory: $(pwd)"

# Submit script
sbatch -t 0-01:00 -n 1 --gres=gpu:v100:1 --mem 10G  $folder/execute_main.sh   # Submit on GPU
# sbatch -t 0-01:00 -n 1 --mem 40G  $folder/execute_main.sh                   # Submit on CPU
# bash  $folder/execute_main.sh                                               # Submit through interactive job

