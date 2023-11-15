#!/bin/bash

#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=7-00:00:00
#SBATCH --job-name=mosaicmpi
#SBATCH --output=%x_%A_%a.out

# source activate environment_name  # adapt this if mosaicmpi requires a particular conda environment

# mosaicmpi uses $1 as the working directory, $2 as the output_directory relative to the working directory, and $3 as the cnmf run name
# other arguments: #4 is number of CPUs, #5 is local density threshold, #6 is local neighborhood size
# variable #7 is usually reserved for flags (eg., --skip-missing-iterations and --force-h5ad-update)

echo $1 "/" $2 "/" $3   
cd "$1"
mosaicmpi postprocess --output_dir $2 --name $3 --cpus $4 --local_density_threshold $5 --local_neighborhood_size 