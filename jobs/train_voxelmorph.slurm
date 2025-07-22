#!/bin/bash
#SBATCH --job-name=voxelmorph_train
#SBATCH --output=logs/voxelmorph_train_%j.out
#SBATCH --error=logs/voxelmorph_train_%j.err
#SBATCH --sq=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
conda deactivate
conda activate /meta/projects/practical_sose25/.conda/envs/registration

python registration/run_registration.py --config configs/baseline_voxelmorph.yaml
