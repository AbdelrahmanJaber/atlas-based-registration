"""
Compare Original vs Preprocessed CT Values

This script shows the differences between original CT data and preprocessed data,
including Hounsfield unit ranges and normalization effects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TPTBox imports
from TPTBox import NII

def analyze_ct_values(original_path, preprocessed_path):
    """
    Analyze and compare CT values between original and preprocessed images.
    
    Args:
        original_path: Path to original CT file
        preprocessed_path: Path to preprocessed CT file
    """
    print("=== CT Value Analysis ===")
    
    # Load images
    print(f"Loading original CT: {original_path}")
    original_nii = NII.load(original_path, seg=False)
    original_data = original_nii.get_array()
    
    print(f"Loading preprocessed CT: {preprocessed_path}")
    preprocessed_nii = NII.load(preprocessed_path, seg=False)
    preprocessed_data = preprocessed_nii.get_array()
    
    # Analyze original data
    print("\n--- Original CT Analysis ---")
    print(f"Data type: {original_data.dtype}")
    print(f"Shape: {original_data.shape}")
    print(f"Min value: {original_data.min():.2f} HU")
    print(f"Max value: {original_data.max():.2f} HU")
    print(f"Mean value: {original_data.mean():.2f} HU")
    print(f"Standard deviation: {original_data.std():.2f} HU")
    
    # Analyze preprocessed data
    print("\n--- Preprocessed CT Analysis ---")
    print(f"Data type: {preprocessed_data.dtype}")
    print(f"Shape: {preprocessed_data.shape}")
    print(f"Min value: {preprocessed_data.min():.6f} (normalized)")
    print(f"Max value: {preprocessed_data.max():.6f} (normalized)")
    print(f"Mean value: {preprocessed_data.mean():.6f} (normalized)")
    print(f"Standard deviation: {preprocessed_data.std():.6f} (normalized)")
    
    # Calculate what the normalized values represent in HU
    hu_range = (-1024, 1024)
    min_hu_equivalent = preprocessed_data.min() * (hu_range[1] - hu_range[0]) + hu_range[0]
    max_hu_equivalent = preprocessed_data.max() * (hu_range[1] - hu_range[0]) + hu_range[0]
    mean_hu_equivalent = preprocessed_data.mean() * (hu_range[1] - hu_range[0]) + hu_range[0]
    
    print(f"\n--- Normalization Analysis ---")
    print(f"Normalized min {preprocessed_data.min():.6f} = {min_hu_equivalent:.2f} HU")
    print(f"Normalized max {preprocessed_data.max():.6f} = {max_hu_equivalent:.2f} HU")
    print(f"Normalized mean {preprocessed_data.mean():.6f} = {mean_hu_equivalent:.2f} HU")
    
    # Show tissue ranges in original data
    print(f"\n--- Tissue Analysis in Original Data ---")
    air_pixels = np.sum(original_data < -900)
    fat_pixels = np.sum((original_data >= -100) & (original_data < -50))
    water_pixels = np.sum((original_data >= -10) & (original_data < 10))
    muscle_pixels = np.sum((original_data >= 10) & (original_data < 40))
    bone_pixels = np.sum((original_data >= 400) & (original_data < 1000))
    metal_pixels = np.sum(original_data >= 1000)
    
    total_pixels = original_data.size
    
    print(f"Air pixels (< -900 HU): {air_pixels:,} ({air_pixels/total_pixels*100:.2f}%)")
    print(f"Fat pixels (-100 to -50 HU): {fat_pixels:,} ({fat_pixels/total_pixels*100:.2f}%)")
    print(f"Water pixels (-10 to 10 HU): {water_pixels:,} ({water_pixels/total_pixels*100:.2f}%)")
    print(f"Muscle pixels (10 to 40 HU): {muscle_pixels:,} ({muscle_pixels/total_pixels*100:.2f}%)")
    print(f"Bone pixels (400 to 1000 HU): {bone_pixels:,} ({bone_pixels/total_pixels*100:.2f}%)")
    print(f"Metal pixels (>= 1000 HU): {metal_pixels:,} ({metal_pixels/total_pixels*100:.2f}%)")
    
    return original_data, preprocessed_data

def create_histograms(original_data, preprocessed_data, save_path="ct_comparison.png"):
    """
    Create histograms comparing original and preprocessed CT values.
    
    Args:
        original_data: Original CT data
        preprocessed_data: Preprocessed CT data
        save_path: Path to save the comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data histogram
    ax1.hist(original_data.flatten(), bins=100, alpha=0.7, color='blue', density=True)
    ax1.set_xlabel('Hounsfield Units (HU)')
    ax1.set_ylabel('Density')
    ax1.set_title('Original CT Values Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add tissue range markers
    ax1.axvline(x=-1000, color='red', linestyle='--', alpha=0.7, label='Air (-1000 HU)')
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Water (0 HU)')
    ax1.axvline(x=400, color='orange', linestyle='--', alpha=0.7, label='Bone (400 HU)')
    ax1.axvline(x=1000, color='purple', linestyle='--', alpha=0.7, label='Metal (1000 HU)')
    ax1.legend()
    
    # Preprocessed data histogram
    ax2.hist(preprocessed_data.flatten(), bins=100, alpha=0.7, color='red', density=True)
    ax2.set_xlabel('Normalized Values [0, 1]')
    ax2.set_ylabel('Density')
    ax2.set_title('Preprocessed CT Values Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add normalized tissue markers
    hu_range = (-1024, 1024)
    air_norm = (-1000 - hu_range[0]) / (hu_range[1] - hu_range[0])
    water_norm = (0 - hu_range[0]) / (hu_range[1] - hu_range[0])
    bone_norm = (400 - hu_range[0]) / (hu_range[1] - hu_range[0])
    metal_norm = (1000 - hu_range[0]) / (hu_range[1] - hu_range[0])
    
    ax2.axvline(x=air_norm, color='red', linestyle='--', alpha=0.7, label=f'Air ({air_norm:.3f})')
    ax2.axvline(x=water_norm, color='green', linestyle='--', alpha=0.7, label=f'Water ({water_norm:.3f})')
    ax2.axvline(x=bone_norm, color='orange', linestyle='--', alpha=0.7, label=f'Bone ({bone_norm:.3f})')
    ax2.axvline(x=metal_norm, color='purple', linestyle='--', alpha=0.7, label=f'Metal ({metal_norm:.3f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {save_path}")
    plt.show()

def show_slice_comparison(original_data, preprocessed_data, slice_idx=None):
    """
    Show a side-by-side comparison of a single slice.
    
    Args:
        original_data: Original CT data
        preprocessed_data: Preprocessed CT data
        slice_idx: Slice index to show (if None, shows middle slice)
    """
    if slice_idx is None:
        slice_idx = original_data.shape[0] // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original slice
    im1 = ax1.imshow(original_data[slice_idx], cmap='gray', vmin=-1000, vmax=1000)
    ax1.set_title(f'Original CT - Slice {slice_idx}\nRange: {original_data[slice_idx].min():.0f} to {original_data[slice_idx].max():.0f} HU')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Hounsfield Units')
    
    # Preprocessed slice
    im2 = ax2.imshow(preprocessed_data[slice_idx], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Preprocessed CT - Slice {slice_idx}\nRange: {preprocessed_data[slice_idx].min():.3f} to {preprocessed_data[slice_idx].max():.3f}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Normalized Values')
    
    plt.tight_layout()
    plt.savefig(f'slice_comparison_{slice_idx}.png', dpi=300, bbox_inches='tight')
    print(f"Slice comparison saved to: slice_comparison_{slice_idx}.png")
    plt.show()

def main():
    """Main function to run the comparison."""
    # Example paths (adjust these to your actual paths)
    data_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata"
    
    original_path = f"{data_root}/CTFU00065/ses-20100514/sub-CTFU00065_ses-20100514_sequ-4_ct.nii.gz"
    preprocessed_path = f"{data_root}/CTFU00065/ses-20100514/sub-CTFU00065_ses-20100514_sequ-4_ct_preprocessed.nii.gz"
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"Original CT not found: {original_path}")
        return
    
    if not os.path.exists(preprocessed_path):
        print(f"Preprocessed CT not found: {preprocessed_path}")
        print("Run the preprocessing script first to generate the preprocessed file.")
        return
    
    # Analyze the data
    original_data, preprocessed_data = analyze_ct_values(original_path, preprocessed_path)
    
    # Create visualizations
    try:
        create_histograms(original_data, preprocessed_data)
        show_slice_comparison(original_data, preprocessed_data)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
        print("This might be due to missing matplotlib or display issues.")

if __name__ == "__main__":
    main() 