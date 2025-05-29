## Atlas Based Registration Project – Inter-patient POI Transfer

This project focuses on registering inter-patient full-body CT scans to transfer Points of Interest (POIs) from 
known subjects to new ones. These POIs can be used to plan surgical guides or to define forces in biomechanical simulations.

We will work with a dataset of ~300 CT scans and develop baseline registration methods using PyTorch, DeepAli, VoxelMorph, and TransMorph. The project also integrates preprocessing using TPTBox.

---

## Project Structure

```
atlas_based_registeration/
├── data/
│   ├── dataset-myelom/         # Original dataset
│   │   ├── rawdata/
│   │   └── derivatives/
│   ├── raw/                    # Flattened volumes for quick access
│   └── preprocessed/           # Cropped & normalized volumes
|
├── configs/              # YAML configuration files for experiments
│   ├── baseline_voxelmorph.yaml
│   └── baseline_transmorph.yaml
│
├── data/                 # Dataset folders
│   ├── raw/              # Original full-body CT scans
│   └── preprocessed/     # Cropped & normalized data (e.g. legs or torso)
│
├── models/               # Model definitions or wrappers (VoxelMorph, TransMorph)
│   ├── voxelmorph.py
│   └── transmorph.py
│
├── registration/         # Main training and inference scripts
│   └── run_registration.py
│
├── utils/                # Utility functions (e.g. preprocessing, metrics)
│   └── preprocessing.py
│
├── results/              # Output folder
│   ├── logs/             # TensorBoard or training logs
│   ├── checkpoints/      # Saved model checkpoints
│   └── visualizations/   # Plots or visual outputs
|
├── jobs/                 # SLURM submission scripts
│
├── main.py               # Entry point for experiments (uses configs)
├── requirements.txt      # Python dependencies
└── README.md             # Project overview (this file)
```