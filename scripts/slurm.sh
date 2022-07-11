#!/bin/bash

#SBATCH --mem=140G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-19
#SBATCH --partition=sherlock,cpu2021,cpu2019,cpu2022-bf24,cpu2019-bf05,cpu2017-bf05
#SBATCH --time=05:00:00
#SBATCH --job-name=cnmfsns
#SBATCH --output=%x_%A_%a.out

# source activate cnmf  # uncomment this if cnmfsns requires a particular conda environment
echo $1 "/" $2 "/" $3   # cnmfsns uses $1 as the working directory, $2 as the output_directory relative to the working directory, and $3 as the cnmf run name
cd "$1"
cnmfsns factorize --output_dir $2 --name $3 --total_workers 20 --worker_index $SLURM_ARRAY_TASK_ID
