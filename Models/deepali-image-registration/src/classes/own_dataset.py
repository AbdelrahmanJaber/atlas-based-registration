import numpy as np
from torch.utils.data import Dataset
import torch
from TPTBox import NII

class OwnDataset(Dataset):
    def __init__(self, data, seg_data=None, transform=None):
        """
        Custom dataset class for loading 2D MRI volumes with optional segmentations.
        
        :param data: Numpy array of shape (N, 2, H, W, D)
        :param seg_data: Optional numpy array of shape (N, 2, H, W, D)
        :param transform: Transform to apply to the images
        """
        self.data = data
        self.seg_data = seg_data
        self.transform = transform

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        :param idx: Index of the item to retrieve
        :return: Tuple of moving image, fixed image, and optionally segmentations
        """

        fixed_nii = NII.load(self.data[idx], seg=False)

        ran = None
        while ran == None or ran == idx:
            ran = np.random.randint(0, len(self.data.shape[0]))
        moving_nii = NII.load(self.data[ran], seg=False)

        # Get the moving and fixed images
        fixed_img = np.transpose(fixed_nii.get_array(), (2, 0, 1))  # Transpose to (D, H, W)
        moving_img = np.transpose(moving_nii.get_array(), (2, 0, 1))  # Transpose to (D, H, W)

        if self.transform:
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)

        moving_img = torch.from_numpy(moving_img).float()
        fixed_img = torch.from_numpy(fixed_img).float()

        if self.seg_data is not None:
            fixed_seg_nii = NII.laod(self.seg_data[idx], seg=True)
            moving_seg_nii = NII.load(self.seg_data[ran], seg=True)

            fixed_seg_resampled = seg.resample_from_to(fixed_seg_nii)
            moving_seg_resampled = seg.resample_from_to(moving_seg_nii)

            fixed_seg = np.transpose(fixed_seg_resampled.get_array(), (2, 0, 1))  # Transpose to (D, H, W)
            moving_seg = np.transpose(moving_seg_resampled.get_array(), (2, 0, 1))

            if self.transform:
                moving_seg = self.transform(moving_seg)
                fixed_seg = self.transform(fixed_seg)

            moving_seg = torch.from_numpy(moving_seg).float()
            fixed_seg = torch.from_numpy(fixed_seg).float()

            return moving_img, fixed_img, moving_seg, fixed_seg

        else:
            return moving_img, fixed_img
