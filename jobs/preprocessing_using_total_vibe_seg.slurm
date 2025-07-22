#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --output=results/logs/jobID-%A.out
#SBATCH --error=results/logs/jobID-%A.err

#SBATCH --mail-user=abdelrahman.jaber999@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave

#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G


# Set PYTHONPATH to include current directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Use conda run to execute the script in the correct environment
conda run -p /meta/projects/practical_sose25/.conda/envs/registration python /vol/miltank/projects/practical_sose25/atlas_based_registeration/utils/crop_knee_patella_using_total_vibe_seg.py
