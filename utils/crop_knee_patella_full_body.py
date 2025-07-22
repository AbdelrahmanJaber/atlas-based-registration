import json
import os
import numpy as np
from glob import glob
from pathlib import Path
from TPTBox import NII

# User-configurable: how many slices to include on each side of the patella
PATELLA_MARGIN_SLICES = 100

# Directories
RAW_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata")
OUTPUT_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/cropped_full_body")

def find_patella_center_from_ct(ct_arr):
    """Find patella center using density analysis of CT data"""
    z_dim = ct_arr.shape[2]
    
    # Focus on the lower half of the body where knees are typically located
    lower_half = ct_arr[:, :, z_dim//2:]
    
    # Calculate bone density per slice (using threshold for bone-like HU values)
    # Bone typically has HU values > 200
    bone_threshold = 200
    bone_density_per_slice = np.sum(lower_half > bone_threshold, axis=(0, 1))
    
    if bone_density_per_slice.size > 0:
        # Find the slice with maximum bone density
        max_density_slice = np.argmax(bone_density_per_slice) + z_dim//2
        print(f"Patella center found at z={max_density_slice} using density analysis")
        return max_density_slice
    else:
        # Fallback: use middle slice
        print("No bone density found, using middle slice")
        return z_dim // 2

def crop_full_body_ct(ct_path):
    """Crop full body CT around the patella"""
    print(f"Loading full body CT: {ct_path}")
    ct_nii = NII.load(str(ct_path), seg=False)
    ct_arr = ct_nii.get_array()
    
    # Find patella center
    z_patella = find_patella_center_from_ct(ct_arr)
    z_dim = ct_arr.shape[2]
    
    # Calculate crop boundaries
    z_start = max(0, z_patella - PATELLA_MARGIN_SLICES)
    z_end = min(z_dim, z_patella + PATELLA_MARGIN_SLICES)
    
    print(f"Crop coordinates: z_start={z_start}, z_end={z_end}")
    
    # Crop the CT array
    ct_cropped_arr = ct_arr[:, :, z_start:z_end]
    
    # Adjust affine transformation to account for cropping
    new_affine = ct_nii.affine.copy()
    new_affine[2, 3] += z_start * ct_nii.affine[2, 2]  # Adjust z origin
    
    cropped_ct = NII((ct_cropped_arr, new_affine, ct_nii.header), seg=False)
    return cropped_ct

def main():
    # Find all CT files in rawdata
    ct_paths = glob(str(RAW_DIR / "*" / "ses-*" / "*_ct.nii.gz"))
    
    for ct_path in ct_paths:
        ct_path = Path(ct_path)
        print(f"\nProcessing CT: {ct_path}")
        
        # Skip preprocessed files
        if "preprocessed" in str(ct_path):
            print(f"Skipping preprocessed file: {ct_path}")
            continue
        
        # Extract subject and session info from CT path
        # Path format: rawdata/CTFU00065/ses-20100514/sub-CTFU00065_ses-20100514_sequ-4_ct.nii.gz
        parts = ct_path.parts
        subject = parts[-3]  # CTFU00065
        session = parts[-2]  # ses-20100514
        filename = parts[-1]  # sub-CTFU00065_ses-20100514_sequ-4_ct.nii.gz
        
        # Crop the full body CT
        cropped_ct = crop_full_body_ct(ct_path)
        
        # Create output directory
        output_subject_dir = OUTPUT_DIR / subject / session
        output_subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cropped CT
        output_filename = filename.replace("_ct.nii.gz", "_ct-patella-crop.nii.gz")
        output_path = output_subject_dir / output_filename
        
        cropped_ct.save(str(output_path))
        print(f"Saved cropped full body CT to: {output_path}")

if __name__ == "__main__":
    main()
