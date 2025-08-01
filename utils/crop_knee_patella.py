import os
from glob import glob
from pathlib import Path
from TPTBox import NII

# Define input and output base directories
DERIVATIVES_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives")
PREPROCESSED_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/preprocessed")

# Find all bone segmentation files
bone_seg_paths = glob(str(DERIVATIVES_DIR / "*" / "ses-*" / "sub-*_seg-bone_msk.nii.gz"))

for bone_path in bone_seg_paths:
    bone_path = Path(bone_path)
    print(f"Processing: {bone_path}")
    # Load segmentation
    seg = NII.load(str(bone_path), seg=True)
    arr = seg.get_array()
    z_dim = arr.shape[2]

    # Crop middle third in z-axis (knee region)
    z_start = z_dim // 3
    z_end = 2 * z_dim // 3
    knee_arr = arr[:, :, z_start:z_end]

    # Create new NII object
    knee_seg = NII((knee_arr, seg.affine, seg.header), seg=True)

    # Prepare output path
    rel_path = bone_path.relative_to(DERIVATIVES_DIR)
    out_dir = PREPROCESSED_DIR / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = bone_path.name.replace("seg-bone", "seg-knee")
    out_path = out_dir / out_name

    # Save
    knee_seg.save(str(out_path))
    print(f"Saved knee segmentation to: {out_path}") 
    