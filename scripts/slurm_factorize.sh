#!/bin/bash

#SBATCH --mem=140G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-39
#SBATCH --time=7-00:00:00
#SBATCH --job-name=mosaicmpi
#SBATCH --output=%x_%A_%a.out

# source activate environment_name  # adapt this if mosaicmpi requires a particular conda environment

# mosaicmpi uses $1 as the working directory, $2 as the output_directory relative to the working directory, and $3 as the cnmf run name
echo $1 "/" $2 "/" $3   
cd "$1"
mosaicmpi factorize --output_dir $2 --name $3 --total_workers 40 --worker_index $SLURM_ARRAY_TASK_ID
