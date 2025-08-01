import json
import os
import numpy as np
from glob import glob
from pathlib import Path
from TPTBox import NII

# User-configurable: how many slices to include on each side of the patella
PATELLA_MARGIN_SLICES = 100

# Set your patella label here (if known, e.g., from TotalVibeSegmentator)
PATELLA_LABELS = [14]  # Example: left and right patella; replace with your actual label(s)

# DERIVATIVES_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives")
# PREPROCESSED_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/preprocessed_v2")

# bone_seg_paths = glob(str(DERIVATIVES_DIR / "*" / "ses-*" / "sub-*_seg-bone_msk.nii.gz"))
data = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/results/bone_paired_raw_seg.json"

with open(data, "r") as f:
    data = list(json.load(f).values())

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

for pairs in data:
    print(f"Processing: {pairs[0], pairs[1]}", flush = True)

    raw = NII.load(pairs[0], seg=False)
    seg = NII.load(pairs[1], seg=True)

    raw = raw.reorient(seg.orientation)
    seg.rescale((0.8, 0.8, 1))

    raw = raw.resample_from_to(seg) # Force into another spacing

    print(f"shape raw {raw.shape}, shape seg {seg.shape}", flush=True)

    seg_arr = seg.get_array()
    raw_arr = raw.get_array()

    z_dim = seg_arr.shape[2]

    z_patella = find_patella_center(seg_arr)
    z_start = max(0, z_patella - PATELLA_MARGIN_SLICES)
    z_end = min(z_dim, z_patella + PATELLA_MARGIN_SLICES)

    cropped_arr = seg_arr[:, :, z_start:z_end]
    raw_cropped_arr = raw_arr[:, :, z_start:z_end]

    new_affine = seg.affine.copy()
    new_affine[2, 3] += z_start * seg.affine[2, 2]  # Adjust z origin

    raw_new_affine = raw.affine.copy()
    raw_new_affine[2, 3] += z_start * raw.affine[2, 2]  # Adjust z origin

    cropped_seg = NII((cropped_arr, new_affine, seg.header), seg=True)
    cropped_raw = NII((raw_cropped_arr, raw_new_affine, raw.header), seg=False)

    seg_path = pairs[1].replace("_seg-bone-msk", "_seg-patella")
    raw_path = pairs[0].replace("_ct", "_ct-patella")

    seg_path = seg_path.replace("derivatives", "kristin/cropped_seg")
    raw_path = raw_path.replace("rawdata", "kristin/cropped_ct")

    cropped_seg.save(seg_path)
    cropped_raw.save(raw_path)

    print(f"Saved patella-centered crop to: {seg_path}, {raw_path}")