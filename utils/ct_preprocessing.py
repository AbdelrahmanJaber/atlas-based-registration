"""
CT Preprocessing Utilities for Atlas-based Registration

This module provides utilities for preprocessing full-body CT scans for registration tasks.
It includes:
- Hounsfield unit clipping and normalization
- Segmentation generation using TotalVibeSegmentator
- Point registration setup using centroids
- BIDS-compliant data handling
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import nibabel as nib

# TPTBox imports
from TPTBox import BIDS_FILE, NII, POI, POI_Global
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_total_seg
from TPTBox.registration.ridged_points import Point_Registration
from TPTBox import calc_centroids, calc_poi_from_subreg_vert, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance

# MONAI for persistent dataset
try:
    from monai.data import PersistentDataset
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Using regular dataset loading.")


class CTPreprocessor:
    """
    Preprocessor for full-body CT scans with Hounsfield unit handling and segmentation.
    """
    
    def __init__(self, data_root: str, derivatives_root: str):
        """
        Initialize the CT preprocessor.
        
        Args:
            data_root: Path to raw BIDS data
            derivatives_root: Path to derivatives directory
        """
        self.data_root = Path(data_root)
        self.derivatives_root = Path(derivatives_root)
        
        # Hounsfield unit clipping ranges
        self.hu_range = (-1024, 1024)  # Standard CT range
        self.hu_window = (-200, 400)   # Bone window for registration
        
    def clip_hounsfield_units(self, nii: NII, hu_range: Optional[Tuple[int, int]] = None) -> NII:
        """
        Clip CT data to Hounsfield unit range.
        
        Args:
            nii: NII object containing CT data
            hu_range: Optional custom HU range (min, max)
            
        Returns:
            Clipped NII object
        """
        if hu_range is None:
            hu_range = self.hu_range
            
        data = nii.get_array()
        data = np.clip(data, hu_range[0], hu_range[1])
        
        # Normalize to [0, 1] range
        data = (data - hu_range[0]) / (hu_range[1] - hu_range[0])
        
        return NII((data, nii.affine, nii.header), seg=False)
    
    def generate_bone_segmentation(self, ct_path: str, subject_id: str, session_id: str, 
                                 sequence_id: str, force_regenerate: bool = False) -> Optional[str]:
        """
        Generate bone segmentation using TotalVibeSegmentator.
        
        Args:
            ct_path: Path to CT file
            subject_id: Subject ID
            session_id: Session ID  
            sequence_id: Sequence ID
            force_regenerate: Whether to regenerate existing segmentation
            
        Returns:
            Path to generated bone segmentation file
        """
        # Create BIDS file object
        img_bids = BIDS_FILE(ct_path, str(self.data_root))
        
        # Define output path for bone segmentation
        out_file = img_bids.get_changed_path(
            "nii.gz", "msk", 
            parent="derivative", 
            info={"seg": "bone"}
        )
        
        # Check if segmentation already exists
        if out_file.exists() and not force_regenerate:
            print(f"Bone segmentation already exists: {out_file}")
            return str(out_file)
        
        try:
            print(f"Generating bone segmentation for {subject_id}...")
            run_total_seg(
                img_bids.get_nii_file(), 
                out_file, 
                dataset_id=520,  # Bone segmentation ID
                padd=5, 
                override=force_regenerate
            )
            print(f"Bone segmentation saved to: {out_file}")
            return str(out_file)
        except Exception as e:
            print(f"Error generating bone segmentation: {e}")
            return None
    
    def generate_fullbody_segmentation(self, ct_path: str, subject_id: str, session_id: str,
                                     sequence_id: str, force_regenerate: bool = False) -> Optional[str]:
        """
        Generate full-body segmentation using TotalVibeSegmentator.
        
        Args:
            ct_path: Path to CT file
            subject_id: Subject ID
            session_id: Session ID
            sequence_id: Sequence ID
            force_regenerate: Whether to regenerate existing segmentation
            
        Returns:
            Path to generated full-body segmentation file
        """
        img_bids = BIDS_FILE(ct_path, str(self.data_root))
        
        out_file = img_bids.get_changed_path(
            "nii.gz", "msk",
            parent="derivative",
            info={"seg": "fullbody"}
        )
        
        if out_file.exists() and not force_regenerate:
            print(f"Full-body segmentation already exists: {out_file}")
            return str(out_file)
        
        try:
            print(f"Generating full-body segmentation for {subject_id}...")
            run_total_seg(
                img_bids.get_nii_file(),
                out_file,
                dataset_id=98,  # Full-body segmentation ID
                padd=5,
                override=force_regenerate
            )
            print(f"Full-body segmentation saved to: {out_file}")
            return str(out_file)
        except Exception as e:
            print(f"Error generating full-body segmentation: {e}")
            return None
    
    def split_left_right_leg(self, seg_nii: NII, c: int = 2, min_volume: int = 50) -> Tuple[NII, NII]:
        """
        Split bone segmentation into left and right leg components.
        
        Args:
            seg_nii: Bone segmentation NII
            c: Center coordinate for splitting
            min_volume: Minimum volume threshold
            
        Returns:
            Tuple of (left_leg, right_leg) NII objects
        """
        data = seg_nii.get_array()
        
        # Split based on center coordinate
        left_leg = data[:, :, :c].copy()
        right_leg = data[:, :, c:].copy()
        
        # Apply minimum volume threshold
        left_leg = self._apply_volume_threshold(left_leg, min_volume)
        right_leg = self._apply_volume_threshold(right_leg, min_volume)
        
        # Create NII objects
        left_nii = NII(left_leg, seg_nii.affine, seg_nii.header)
        right_nii = NII(right_leg, seg_nii.affine, seg_nii.header)
        
        return left_nii, right_nii
    
    def _apply_volume_threshold(self, data: np.ndarray, min_volume: int) -> np.ndarray:
        """Apply minimum volume threshold to segmentation."""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(data)
        
        # Remove small components
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled == i)
            if component_size < min_volume:
                data[labeled == i] = 0
                
        return data
    
    def compute_centroids(self, seg_nii: NII, second_stage: int = 40) -> POI:
        """
        Compute centroids from segmentation for point registration.
        
        Args:
            seg_nii: Segmentation NII object
            second_stage: Second stage parameter for centroid calculation
            
        Returns:
            POI object containing centroids
        """
        return calc_centroids(seg_nii, second_stage=second_stage)
    
    def align_centers_of_mass(self, target_nii: NII, atlas_nii: NII) -> Tuple[NII, NII]:
        """
        Align two images by moving their centers of mass to the same position.
        
        Args:
            target_nii: Target image
            atlas_nii: Atlas image
            
        Returns:
            Tuple of aligned (target, atlas) NII objects
        """
        # Compute centers of mass
        target_com = self._compute_center_of_mass(target_nii)
        atlas_com = self._compute_center_of_mass(atlas_nii)
        
        # Calculate translation
        translation = target_com - atlas_com
        
        # Apply translation to atlas
        atlas_aligned = self._translate_nii(atlas_nii, translation)
        
        return target_nii, atlas_aligned
    
    def _compute_center_of_mass(self, nii: NII) -> np.ndarray:
        """Compute center of mass in world coordinates."""
        data = nii.get_array()
        
        # Get non-zero coordinates
        coords = np.where(data > 0)
        if len(coords[0]) == 0:
            return np.array([0, 0, 0])
        
        # Compute center of mass in voxel coordinates
        com_voxel = np.array([
            np.mean(coords[0]),
            np.mean(coords[1]), 
            np.mean(coords[2])
        ])
        
        # Convert to world coordinates
        com_world = nii.affine[:3, :3] @ com_voxel + nii.affine[:3, 3]
        
        return com_world
    
    def _translate_nii(self, nii: NII, translation: np.ndarray) -> NII:
        """Apply translation to NII object."""
        new_affine = nii.affine.copy()
        new_affine[:3, 3] += translation
        
        return NII(nii.get_array(), new_affine, nii.header)
    
    def setup_point_registration(self, target_seg: NII, atlas_seg: NII, 
                               save_centroids: bool = True, 
                               centroids_path: Optional[str] = None) -> Point_Registration:
        """
        Set up point registration between target and atlas segmentations.
        
        Args:
            target_seg: Target segmentation
            atlas_seg: Atlas segmentation
            save_centroids: Whether to save computed centroids
            centroids_path: Path to save centroids (optional)
            
        Returns:
            Point_Registration object
        """
        # Compute centroids
        poi_target = self.compute_centroids(target_seg, second_stage=40)
        poi_atlas = self.compute_centroids(atlas_seg, second_stage=40)
        
        # Save centroids if requested
        if save_centroids:
            if centroids_path is None:
                centroids_path = "centroids.json"
            poi_atlas.save(centroids_path)
            print(f"Centroids saved to: {centroids_path}")
        
        # Create point registration
        reg_point = Point_Registration(poi_target, poi_atlas)
        
        return reg_point
    
    def load_metadata(self, json_path: str) -> Dict[str, Any]:
        """
        Load metadata from BIDS JSON file.
        
        Args:
            json_path: Path to JSON metadata file
            
        Returns:
            Dictionary containing metadata
        """
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def preprocess_ct_pipeline(self, ct_path: str, subject_id: str, session_id: str,
                             sequence_id: str, generate_segmentations: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a CT scan.
        
        Args:
            ct_path: Path to CT file
            subject_id: Subject ID
            session_id: Session ID
            sequence_id: Sequence ID
            generate_segmentations: Whether to generate segmentations
            
        Returns:
            Dictionary containing paths to preprocessed files
        """
        results = {
            'ct_path': ct_path,
            'subject_id': subject_id,
            'session_id': session_id,
            'sequence_id': sequence_id
        }
        
        # Load CT data
        print(f"Loading CT data from: {ct_path}")
        ct_nii = NII.load(ct_path, seg=False)
        
        # Clip Hounsfield units
        print("Clipping Hounsfield units...")
        ct_clipped = self.clip_hounsfield_units(ct_nii)
        
        # Save preprocessed CT
        preprocessed_ct_path = ct_path.replace('.nii.gz', '_preprocessed.nii.gz')
        ct_clipped.save(preprocessed_ct_path)
        results['preprocessed_ct_path'] = preprocessed_ct_path
        
        # Generate segmentations if requested
        if generate_segmentations:
            bone_seg_path = self.generate_bone_segmentation(
                ct_path, subject_id, session_id, sequence_id
            )
            if bone_seg_path:
                results['bone_seg_path'] = bone_seg_path
            
            fullbody_seg_path = self.generate_fullbody_segmentation(
                ct_path, subject_id, session_id, sequence_id
            )
            if fullbody_seg_path:
                results['fullbody_seg_path'] = fullbody_seg_path
        
        # Load metadata
        json_path = ct_path.replace('.nii.gz', '.json')
        if os.path.exists(json_path):
            metadata = self.load_metadata(json_path)
            results['metadata'] = metadata
        
        return results


def create_persistent_dataset(data_paths: list, cache_dir: str = "./cache"):
    """
    Create a MONAI PersistentDataset for faster loading.
    
    Args:
        data_paths: List of data file paths
        cache_dir: Directory to store cached data
        
    Returns:
        PersistentDataset object if MONAI is available, None otherwise
    """
    if not MONAI_AVAILABLE:
        print("MONAI not available. Returning None.")
        return None
    
    # Import here to avoid issues when MONAI is not available
    try:
        from monai.data import PersistentDataset
    except ImportError:
        print("MONAI not available. Returning None.")
        return None
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create dataset
    dataset = PersistentDataset(
        data=data_paths,
        transform=None,  # Add your transforms here
        cache_dir=cache_dir
    )
    
    return dataset


def main():
    """Example usage of the CT preprocessor."""
    # Initialize preprocessor
    data_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata"
    derivatives_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/derivatives"
    
    preprocessor = CTPreprocessor(data_root, derivatives_root)
    print("test", flush=True)
    # Example: Process one subject
    ct_path = f"{data_root}/CTFU00065/ses-20100514/sub-CTFU00065_ses-20100514_sequ-4_ct.nii.gz"
    
    if os.path.exists(ct_path):
        results = preprocessor.preprocess_ct_pipeline(
            ct_path=ct_path,
            subject_id="CTFU00065",
            session_id="ses-20100514",
            sequence_id="sequ-4",
            generate_segmentations=True
        )
        
        print("Preprocessing results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print(f"CT file not found: {ct_path}")


if __name__ == "__main__":
    main() 