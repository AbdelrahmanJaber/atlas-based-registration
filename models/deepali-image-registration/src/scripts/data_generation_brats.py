import os
import nibabel as nib
import numpy as np

def process_nifti_files(base_folder, file_ending, output_file):
    all_middle_slices = []

    subject_folders = sorted(os.listdir(base_folder))
    
    # Iterate through subject folders
    for subject_folder in subject_folders:
        subject_path = os.path.join(base_folder, subject_folder)
        if os.path.isdir(subject_path):
            # Find files ending with the specified ending
            matching_files = [f for f in os.listdir(subject_path) if f.endswith(file_ending)]
            
            for file_name in matching_files:
                file_path = os.path.join(subject_path, file_name)
                try:
                    # Load the NIfTI file
                    nifti_img = nib.load(file_path)
                    
                    # Get the image data
                    nifti_data = nifti_img.get_fdata()
                    
                    # Extract the middle slice along the last dimension
                    middle_slice_index = nifti_data.shape[-1] // 2
                    middle_slice = nifti_data[..., middle_slice_index]
                    
                    all_middle_slices.append(middle_slice)
                    
                    print(f"Processed: {subject_folder}/{file_name}")
                    
                    # To save memory
                    if len(all_middle_slices) >= 500: 
                        save_to_npy(all_middle_slices, output_file)
                        all_middle_slices = []
                        
                except Exception as e:
                    print(f"Error processing {subject_folder}/{file_name}: {str(e)}")
            
            if not matching_files:
                print(f"No matching files found in: {subject_folder}")
    
    # Save any remaining slices
    if all_middle_slices:
        save_to_npy(all_middle_slices, output_file)

def save_to_npy(slices, output_file):
    if os.path.exists(output_file):
        # If the file exists, load it, append new data, and save
        existing_data = np.load(output_file)
        combined_data = np.concatenate([existing_data, slices])
        np.save(output_file, combined_data)
    else:
        # If the file doesn't exist, create it
        np.save(output_file, slices)
    print(f"Saved batch to {output_file}")

# Usage
base_folder = '/media/datasets/dataset-BraTS2021'
file_ending = '_flair.nii.gz'  # Example: files ending with '_t1.nii.gz'
output_file = 'brats_flair_slices.npy'

process_nifti_files(base_folder, file_ending, output_file)