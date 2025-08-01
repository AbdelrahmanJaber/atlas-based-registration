import os
import numpy as np
from glob import glob
from pathlib import Path
from TPTBox import NII, BIDS_FILE
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_total_seg

# Define input and output base directories
DATA_ROOT = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata")
DERIVATIVES_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives")
PREPROCESSED_DIR = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/preprocessed_using_total_vibe_seg")

def generate_fullbody_segmentation(ct_path, force_regenerate=False):
    """
    Generate full-body segmentation using TotalVibeSegmentator.
    
    Args:
        ct_path: Path to CT file
        force_regenerate: Whether to regenerate existing segmentation
        
    Returns:
        Path to generated full-body segmentation file
    """
    # Create BIDS file object
    img_bids = BIDS_FILE(ct_path, str(DATA_ROOT))
    
    # Define output path for full-body segmentation
    out_file = img_bids.get_changed_path(
        "nii.gz", "msk",
        parent="derivative",
        info={"seg": "fullbody"}
    )
    
    # Check if segmentation already exists
    if out_file.exists() and not force_regenerate:
        print(f"  Full-body segmentation already exists: {out_file}")
        return str(out_file)
    
    try:
        print(f"  Generating full-body segmentation...")
        run_total_seg(
            img_bids.get_nii_file(),
            out_file,
            dataset_id=98,  # Full-body segmentation ID
            padd=5,
            override=force_regenerate
        )
        print(f"  Full-body segmentation saved to: {out_file}")
        return str(out_file)
    except Exception as e:
        print(f"  Error generating full-body segmentation: {e}")
        return None

def extract_patella_region(fullbody_seg_path):
    """
    Extract patella regions from full-body segmentation.
    
    Args:
        fullbody_seg_path: Path to full-body segmentation file
        
    Returns:
        NII object containing only patella regions
    """
    print(f"  Loading full-body segmentation: {fullbody_seg_path}")
    
    # Load full-body segmentation
    fullbody_seg = NII.load(fullbody_seg_path, seg=True)
    arr = fullbody_seg.get_array()
    
    print(f"  Full-body segmentation shape: {arr.shape}")
    print(f"  Unique labels: {np.unique(arr)}")
    
    # Create patella mask (combine left and right patella)
    # Note: The exact label values depend on TotalVibeSegmentator's labeling scheme
    # We'll need to identify the patella labels from the unique values
    patella_mask = np.zeros_like(arr, dtype=bool)
    
    # Look for patella labels in the segmentation
    # Common patella label values might be around 100-200 range
    # We'll create a mask for potential patella regions
    for label in np.unique(arr):
        if label > 0:  # Skip background
            # Check if this label appears in the lower half of the body (where knees are)
            lower_half = arr[:, :, arr.shape[2]//2:]
            if np.any(lower_half == label):
                # This could be a patella label
                patella_mask |= (arr == label)
                print(f"    Found potential patella label: {label}")
    
    # If no specific patella labels found, fall back to anatomical approach
    if not np.any(patella_mask):
        print("    No specific patella labels found, using anatomical fallback...")
        return extract_patella_anatomical(fullbody_seg)
    
    # Create NII object for patella regions
    patella_arr = arr.copy()
    patella_arr[~patella_mask] = 0
    
    print(f"  Patella voxels found: {np.sum(patella_mask)}")
    
    return NII((patella_arr, fullbody_seg.affine, fullbody_seg.header), seg=True)

def extract_patella_anatomical(fullbody_seg):
    """
    Fallback method: Extract patella region using anatomical knowledge.
    
    Args:
        fullbody_seg: Full-body segmentation NII object
        
    Returns:
        NII object containing knee region
    """
    arr = fullbody_seg.get_array()
    z_dim = arr.shape[2]
    
    # Look for the region with highest bone density in lower half
    # (knees typically have high bone density due to patella + femur/tibia junction)
    lower_half = arr[:, :, z_dim//2:]
    bone_density_per_slice = np.sum(lower_half > 0, axis=(0, 1))
    
    if len(bone_density_per_slice) > 0:
        # Find the slice with maximum bone density in lower half
        max_density_slice = np.argmax(bone_density_per_slice) + z_dim//2
        
        # Define knee region around this slice (±15% of total height)
        knee_region_size = int(z_dim * 0.15)  # 15% of total height
        z_start = max(0, max_density_slice - knee_region_size//2)
        z_end = min(z_dim, max_density_slice + knee_region_size//2)
        
        print(f"    Anatomical knee region: slices {z_start}-{z_end} (density peak at slice {max_density_slice})")
        
        # Crop the knee region
        knee_arr = arr[:, :, z_start:z_end]
        
        # Create new NII object with updated affine matrix
        new_affine = fullbody_seg.affine.copy()
        new_affine[2, 3] += z_start * fullbody_seg.affine[2, 2]  # Adjust z origin
        
        return NII((knee_arr, new_affine, fullbody_seg.header), seg=True)
    
    # Final fallback: use middle third
    z_start = z_dim // 3
    z_end = 2 * z_dim // 3
    print(f"    Final fallback: slices {z_start}-{z_end}")
    
    knee_arr = arr[:, :, z_start:z_end]
    new_affine = fullbody_seg.affine.copy()
    new_affine[2, 3] += z_start * fullbody_seg.affine[2, 2]
    
    return NII((knee_arr, new_affine, fullbody_seg.header), seg=True)

def process_ct_file(ct_path):
    """
    Process a single CT file to extract knee/patella region.
    
    Args:
        ct_path: Path to CT file
        
    Returns:
        Path to saved knee segmentation file
    """
    ct_path = Path(ct_path)
    print(f"Processing: {ct_path}")
    
    # Generate full-body segmentation if it doesn't exist
    fullbody_seg_path = generate_fullbody_segmentation(str(ct_path))
    
    if not fullbody_seg_path:
        print(f"  Failed to generate full-body segmentation for {ct_path}")
        return None
    
    # Extract patella region from full-body segmentation
    patella_seg = extract_patella_region(fullbody_seg_path)
    
    # Prepare output path
    rel_path = ct_path.relative_to(DATA_ROOT)
    out_dir = PREPROCESSED_DIR / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    ct_name = ct_path.stem.replace('_ct', '')
    out_name = f"{ct_name}_seg-knee_msk.nii.gz"
    out_path = out_dir / out_name
    
    # Save knee segmentation
    patella_seg.save(str(out_path))
    print(f"  Saved knee segmentation to: {out_path}")
    
    return out_path

# Find all CT files
ct_paths = glob(str(DATA_ROOT / "*" / "ses-*" / "*_ct.nii.gz"))

print(f"Found {len(ct_paths)} CT files")

for ct_path in ct_paths:
    try:
        process_ct_file(ct_path)
    except Exception as e:
        print(f"Error processing {ct_path}: {e}")

print("Knee/patella extraction completed!") 