#!/bin/bash
#SBATCH --job-name=make_split
#SBATCH --output=results/logs/make_split_%j.out
#SBATCH --error=results/logs/make_split_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# Set PYTHONPATH to include current directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Use conda run to execute the script in the correct environment
conda run -p /meta/projects/practical_sose25/.conda/envs/registration python utils/make_split.py 


# # Activate conda environment
# source ~/.bashrc
# conda init
# conda deactivate
# conda activate /meta/projects/practical_sose25/.conda/envs/registration

# # Run the split creation script
# python utils/make_split.py 

# # Use conda run instead of conda activate
# # conda run -p /meta/projects/practical_sose25/.conda/envs/registration python jobs/make_split.py 