import numpy as np
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    '''
    Custom dataset class for loading 2D MRI volumes with optional segmentations
    '''
    def __init__(self, data, seg_data=None, transform=None):
        '''
        Constructor for the CustomDataset class
        :param data: Numpy array of shape (N, 2, H, W, D)
        :param seg_data: Optional numpy array of shape (N, 2, H, W, D)
        :param transform: Transform to apply to the images
        '''
        self.data = data
        self.seg_data = seg_data
        self.transform = transform

    def __len__(self):
        '''
        Get the length of the dataset
        '''
        return self.data.shape[0]

    def __getitem__(self, idx):
        '''
        Get an item from the dataset
        '''
        # Get the moving and fixed images
        moving_img = np.transpose(self.data[idx, 0, :, :, :], (2, 0, 1))  # Transpose to (D, H, W)
        fixed_img = np.transpose(self.data[idx, 1, :, :, :], (2, 0, 1))  # Transpose to (D, H, W)

        # Apply transformations (if any) to images
        if self.transform:
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)

        # Convert to PyTorch tensors
        moving_img = torch.from_numpy(moving_img).float()
        fixed_img = torch.from_numpy(fixed_img).float()

        # If segmentation data is provided, process it
        if self.seg_data is not None:
            moving_seg = np.transpose(self.seg_data[idx, 0, :, :, :], (2, 0, 1))  # Transpose to (D, H, W)
            fixed_seg = np.transpose(self.seg_data[idx, 1, :, :, :], (2, 0, 1))  # Transpose to (D, H, W)

            # Apply transformations (if any) to segmentations
            if self.transform:
                moving_seg = self.transform(moving_seg)
                fixed_seg = self.transform(fixed_seg)

            # Convert segmentations to PyTorch tensors
            moving_seg = torch.from_numpy(moving_seg).float()
            fixed_seg = torch.from_numpy(fixed_seg).float()

            return moving_img, fixed_img, moving_seg, fixed_seg
        else:
            return moving_img, fixed_img