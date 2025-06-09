#!/bin/bash

#SBATCH --job-name=create_npy
#SBATCH --output=results/logs/jobID-%A.out
#SBATCH --error=results/logs/jobID-%A.err

#SBATCH --mail-user=kristin.fratzke@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave

#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G


# Load python module
ml python/anaconda3


# Properly initialize Conda
# source /meta/projects/practical_sose25/.conda/etc/profile.d/conda.sh


# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# alwavs run this script from a clean terminal 

conda init
conda deactivate


# If the following does not work, try 'source activate <env-name>
# conda activate registration

conda init
conda activate /meta/projects/practical_sose25/.conda/envs/atlas-kristin

# Run the program
python /vol/miltank/users/fratzke/Documents/atlas-based-registration/utils/create_npy.py
