import os
import numpy as np
import random
import nibabel as nib
from skimage import exposure

def load_and_process_data(root_dir):
    subject_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    all_data = []
    all_data_seg = []
    for idx in range(len(subject_dirs)):
        # Select a random second index for pairs
        idx2 = random.randint(0, len(subject_dirs) - 1)
        
        subject_dir1 = subject_dirs[idx]
        subject_dir2 = subject_dirs[idx2]
        
        slice_norm_path1 = os.path.join(root_dir, subject_dir1, 'slice_norm.nii.gz')
        slice_norm_path2 = os.path.join(root_dir, subject_dir2, 'slice_norm.nii.gz')
        slice_seg_path1 = os.path.join(root_dir, subject_dir1, 'slice_seg4.nii.gz')
        slice_seg_path2 = os.path.join(root_dir, subject_dir2, 'slice_seg4.nii.gz')
        
        source = nib.load(slice_norm_path1).get_fdata()
        target = nib.load(slice_norm_path2).get_fdata()

        source_seg = nib.load(slice_seg_path1).get_fdata()
        target_seg = nib.load(slice_seg_path2).get_fdata()
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        source_seg = source_seg.astype(np.float32)
        target_seg = target_seg.astype(np.float32)

        matched_source_image = exposure.match_histograms(source, target)
        
        all_data.append((matched_source_image, target))
        all_data_seg.append((source_seg, target_seg))
    
    return np.array(all_data), np.array(all_data_seg)

def save_as_npy(data, save_path):
    np.save(save_path, data)
    print(f'Dataset saved as {save_path}')

if __name__ == "__main__":
    root_dir = 'data/brain/' # Root directory containing the data
    save_path = 'data/oasis_dataset_1.npy' # Path to save the dataset
    save_path_seg = 'data/oasis_dataset_seg.npy' # Path to save the dataset
    
    # Load and process the data
    data, seg = load_and_process_data(root_dir)
    
    # Save the entire dataset as a single .npy file
    save_as_npy(data, save_path)
    save_as_npy(seg, save_path_seg)
