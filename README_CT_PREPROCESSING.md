# CT Preprocessing for Atlas-based Registration

This document provides a comprehensive guide for preprocessing full-body CT scans for atlas-based registration tasks. The utilities handle Hounsfield unit clipping, segmentation generation, and point registration setup.

## Overview

Full-body CT scans present unique challenges for registration due to their large size and complex anatomy. This preprocessing pipeline addresses these challenges by:

1. **Hounsfield Unit Clipping**: CT data is in Hounsfield units (HU), which have predictable ranges. We clip to -1024 to 1024 HU and normalize to [0,1].
2. **Segmentation Generation**: Using TotalVibeSegmentator to generate bone and full-body segmentations.
3. **Point Registration**: Computing centroids from segmentations for reliable point-based registration.
4. **BIDS Compliance**: Working with BIDS-standard data organization.

## Installation

The preprocessing utilities require the following packages (already included in requirements.txt):

```bash
# Core dependencies
TPTBox  # For medical image processing
nibabel  # For NIfTI file handling
numpy  # For numerical operations
scipy  # For image processing

# Optional dependencies
monai  # For persistent dataset caching
```

## Quick Start

### 1. Basic CT Preprocessing

```python
from utils.ct_preprocessing import CTPreprocessor

# Initialize preprocessor
data_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata"
derivatives_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives"

preprocessor = CTPreprocessor(data_root, derivatives_root)

# Process a single CT scan
ct_path = "path/to/your/ct.nii.gz"
results = preprocessor.preprocess_ct_pipeline(
    ct_path=ct_path,
    subject_id="SUBJECT_ID",
    session_id="SESSION_ID", 
    sequence_id="SEQUENCE_ID",
    generate_segmentations=True
)
```

### 2. Using Enhanced Dataset

```python
from models.voxelmorph.enhanced_dataset import EnhancedNiftiPairDataset

# Create enhanced dataset
dataset = EnhancedNiftiPairDataset(
    data_root=data_root,
    derivatives_root=derivatives_root,
    split_file="subject_splits.json",
    split='train',
    use_preprocessing=True,
    hu_range=(-1024, 1024),
    use_segmentation=True,
    segmentation_type="bone",
    cache_dir="./cache"
)

# Get a sample
sample = dataset[0]
print("Sample keys:", sample.keys())
print("Moving CT shape:", sample['moving_ct'].shape)
```

## Detailed Usage

### Hounsfield Unit Clipping

CT data is stored in Hounsfield units with predictable ranges:
- Air: -1000 HU
- Water: 0 HU  
- Bone: 400-1000 HU
- Metal: >1000 HU

```python
from TPTBox import NII

# Load CT data
ct_nii = NII.load("ct.nii.gz", seg=False)

# Clip to standard range
clipped_ct = preprocessor.clip_hounsfield_units(ct_nii, hu_range=(-1024, 1024))

# Or use bone window for registration
bone_window_ct = preprocessor.clip_hounsfield_units(ct_nii, hu_range=(-200, 400))
```

### Segmentation Generation

Generate bone and full-body segmentations using TotalVibeSegmentator:

```python
# Generate bone segmentation (dataset_id=520)
bone_seg_path = preprocessor.generate_bone_segmentation(
    ct_path, subject_id, session_id, sequence_id
)

# Generate full-body segmentation (dataset_id=98)
fullbody_seg_path = preprocessor.generate_fullbody_segmentation(
    ct_path, subject_id, session_id, sequence_id
)
```

### Point Registration Setup

Set up point registration using centroids from segmentations:

```python
from TPTBox import NII

# Load segmentations
target_seg = NII.load(target_bone_path, seg=True)
atlas_seg = NII.load(atlas_bone_path, seg=True)

# Compute centroids
poi_target = preprocessor.compute_centroids(target_seg, second_stage=40)
poi_atlas = preprocessor.compute_centroids(atlas_seg, second_stage=40)

# Save centroids for later use
poi_target.save("target_centroids.json")
poi_atlas.save("atlas_centroids.json")

# Set up point registration
reg_point = preprocessor.setup_point_registration(
    target_seg, atlas_seg,
    save_centroids=True,
    centroids_path="registration_centroids.json"
)
```

### Center of Mass Alignment

Align images by moving their centers of mass to the same position:

```python
# Load and preprocess CT data
target_ct = NII.load(target_path, seg=False)
atlas_ct = NII.load(atlas_path, seg=False)

target_processed = preprocessor.clip_hounsfield_units(target_ct)
atlas_processed = preprocessor.clip_hounsfield_units(atlas_ct)

# Align centers of mass
target_aligned, atlas_aligned = preprocessor.align_centers_of_mass(
    target_processed, atlas_processed
)
```

### Leg Segmentation Split

Split bone segmentation into left and right leg components:

```python
# Load bone segmentation
bone_seg = NII.load(bone_seg_path, seg=True)

# Split into left and right legs
left_leg, right_leg = preprocessor.split_left_right_leg(
    bone_seg, c=2, min_volume=50
)

# Save split segmentations
left_leg.save("left_leg_segmentation.nii.gz")
right_leg.save("right_leg_segmentation.nii.gz")
```

## Dataset Classes

### EnhancedNiftiPairDataset

Enhanced dataset class that integrates preprocessing utilities:

```python
dataset = EnhancedNiftiPairDataset(
    data_root=data_root,
    derivatives_root=derivatives_root,
    split_file=split_file,
    split='train',
    use_preprocessing=True,
    hu_range=(-1024, 1024),
    use_segmentation=True,
    segmentation_type="bone",  # "bone", "fullbody", or "both"
    atlas_path=atlas_path,  # Optional: for atlas-based registration
    cache_dir="./cache"
)
```

### AtlasBasedDataset

Dataset specifically for atlas-based registration:

```python
dataset = AtlasBasedDataset(
    data_root=data_root,
    derivatives_root=derivatives_root,
    atlas_path=atlas_path,  # Atlas used as fixed image
    split_file=split_file,
    split='train',
    use_preprocessing=True,
    hu_range=(-1024, 1024),
    use_segmentation=True,
    segmentation_type="bone"
)
```

## BIDS Data Structure

The utilities work with BIDS-standard data organization:

```
dataset-myelom/
├── rawdata/
│   ├── CTFU00065/
│   │   └── ses-20100514/
│   │       ├── sub-CTFU00065_ses-20100514_sequ-4_ct.nii.gz
│   │       └── sub-CTFU00065_ses-20100514_sequ-4_ct.json
│   └── ...
└── derivatives/
    ├── CTFU00065/
    │   └── ses-20100514/
    │       ├── sub-CTFU00065_ses-20100514_sequ-4_seg-bone_msk.nii.gz
    │       └── sub-CTFU00065_ses-20100514_sequ-4_seg-fullbody_msk.nii.gz
    └── ...
```

### Metadata Access

Access patient metadata from BIDS JSON files:

```python
# Load metadata
metadata = preprocessor.load_metadata(json_path)

# Access common fields
age = metadata.get('PatientAge', 'N/A')
sex = metadata.get('PatientSex', 'N/A')
slice_thickness = metadata.get('SliceThickness', 'N/A')
pixel_spacing = metadata.get('PixelSpacing', 'N/A')
```

## Performance Optimization

### Caching

Use caching to speed up repeated operations:

```python
# Enable caching in dataset
dataset = EnhancedNiftiPairDataset(
    cache_dir="./cache",
    # ... other parameters
)
```

### MONAI PersistentDataset

For even faster loading, use MONAI's PersistentDataset:

```python
from models.voxelmorph.enhanced_dataset import create_persistent_enhanced_dataset

# Create persistent dataset
persistent_dataset = create_persistent_enhanced_dataset(
    data_root=data_root,
    derivatives_root=derivatives_root,
    split_file=split_file,
    split='train',
    cache_dir="./cache",
    use_preprocessing=True,
    hu_range=(-1024, 1024)
)
```

## Examples

Run the example script to see all utilities in action:

```bash
python utils/point_registration_example.py
```

This script demonstrates:
- Point registration setup
- Center of mass alignment
- Leg segmentation splitting
- Complete preprocessing pipeline
- Metadata analysis

## Tips for Full-body CT Registration

1. **Hounsfield Unit Clipping**: Always clip to a reasonable range (-1024 to 1024) to handle outliers and normalize the data.

2. **Segmentation-based Registration**: Use bone segmentations for more reliable registration, especially for full-body scans.

3. **Point Registration**: Compute centroids from segmentations for robust point-based registration.

4. **Center of Mass Alignment**: Pre-align images by their centers of mass to improve convergence.

5. **Caching**: Use caching for preprocessed data to speed up training.

6. **BIDS Compliance**: Maintain BIDS organization for reproducibility and data sharing.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Full-body CTs are large. Consider:
   - Using smaller HU windows
   - Cropping to regions of interest
   - Using MONAI PersistentDataset for caching

2. **Segmentation Failures**: If TotalVibeSegmentator fails:
   - Check if the model is properly installed
   - Verify the dataset_id (520 for bone, 98 for full-body)
   - Ensure sufficient memory for segmentation

3. **Point Registration Issues**: If centroids are poor:
   - Adjust the `second_stage` parameter
   - Use different segmentation types
   - Check segmentation quality

### Debugging

Enable verbose output and check intermediate files:

```python
# Check if files exist
print(f"CT exists: {os.path.exists(ct_path)}")
print(f"Segmentation exists: {os.path.exists(seg_path)}")

# Load and inspect data
ct_nii = NII.load(ct_path, seg=False)
print(f"CT shape: {ct_nii.shape}")
print(f"CT range: {ct_nii.get_array().min()} to {ct_nii.get_array().max()}")
```

## References

- [Hounsfield Scale](https://flexikon.doccheck.com/de/Hounsfield-Skala)
- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [TotalVibeSegmentator](https://github.com/robert-graf/TotalVibeSegmentator)
- [TPTBox](https://github.com/Hendrik-code/TPTBox) 