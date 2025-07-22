import os
import numpy as np
from glob import glob
from pathlib import Path
from TPTBox import NII

# User-configurable: how many slices to include on each side of the patella
PATELLA_MARGIN_SLICES = 100

# Set your patella label here (if known, e.g., from TotalVibeSegmentator)
PATELLA_LABELS = [123, 124]  # Example: left and right patella; replace with your actual label(s)

DERIVATIVES_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives")
PREPROCESSED_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/preprocessed_v2")

bone_seg_paths = glob(str(DERIVATIVES_DIR / "*" / "ses-*" / "sub-*_seg-bone_msk.nii.gz"))

def find_patella_center(arr):
    # Try to find patella label(s)
    patella_mask = np.isin(arr, PATELLA_LABELS)
    if np.any(patella_mask):
        # Use center of mass of patella label(s)
        coords = np.argwhere(patella_mask)
        z_patella = int(np.round(np.mean(coords[:, 2])))
        print(f"Patella label found, center at z={z_patella}")
        return z_patella
    else:
        # Fallback: use density peak in lower half
        z_dim = arr.shape[2]
        lower_half = arr[:, :, z_dim//2:]
        bone_density_per_slice = np.sum(lower_half > 0, axis=(0, 1))
        if bone_density_per_slice.size > 0:
            max_density_slice = np.argmax(bone_density_per_slice) + z_dim//2
            print(f"Patella label not found, using density peak at z={max_density_slice}")
            return max_density_slice
        else:
            # Final fallback: use middle slice
            print("No bone found, using middle slice")
            return z_dim // 2

for bone_path in bone_seg_paths:
    bone_path = Path(bone_path)
    print(f"Processing: {bone_path}")
    seg = NII.load(str(bone_path), seg=True)
    arr = seg.get_array()
    z_dim = arr.shape[2]

    z_patella = find_patella_center(arr)
    z_start = max(0, z_patella - PATELLA_MARGIN_SLICES)
    z_end = min(z_dim, z_patella + PATELLA_MARGIN_SLICES)

    cropped_arr = arr[:, :, z_start:z_end]
    new_affine = seg.affine.copy()
    new_affine[2, 3] += z_start * seg.affine[2, 2]  # Adjust z origin

    cropped_seg = NII((cropped_arr, new_affine, seg.header), seg=True)

    rel_path = bone_path.relative_to(DERIVATIVES_DIR)
    out_dir = PREPROCESSED_DIR / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = bone_path.name.replace("seg-bone", "seg-patella-crop")
    out_path = out_dir / out_name

    cropped_seg.save(str(out_path))
    print(f"Saved patella-centered crop to: {out_path}")