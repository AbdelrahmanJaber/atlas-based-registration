#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=logs/training.out
#SBATCH --error=logs/training.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

ml python/anaconda3
conda activate registration

python registration/run_registration.py --config configs/baseline_voxelmorph.yaml
