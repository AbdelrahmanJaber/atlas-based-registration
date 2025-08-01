#!/bin/bash

#SBATCH --job-name=train_transmorph
#SBATCH --output=results/logs/jobID-%A.out
#SBATCH --error=results/logs/jobID-%A.err

#SBATCH --mail-user=kristin.fratzke@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave

#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G


# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you
# alwavs run this script from a clean terminal 

conda init
conda deactivate


# If the following does not work, try 'source activate <env-name>

conda activate atlas-kristin

nvidia-smi

# Run the program
CUDA_VISIBLE_DEVICES=0 python -u /vol/miltank/projects/practical_sose25/atlas_based_registeration/models/TransMorph_Transformer_for_Medical_Image_Registration/TransMorph/train_TransMorph.py