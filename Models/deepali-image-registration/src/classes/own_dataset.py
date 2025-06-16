import numpy as np
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    '''
    Custom dataset for random moving/fixed image pairing from a list of 3D volumes.
    Expects data shape: (N, H, W, D) → N volumes.
    Returns tensors in shape (1, D, H, W)
    '''
    def __init__(self, data, seg_data=None, transform=None):
        self.data = data            # shape: (N, H, W, D)
        self.seg_data = seg_data    # optional: (N, H, W, D)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Wähle zwei zufällige, verschiedene Indizes
        moving_idx = idx
        fixed_idx = np.random.randint(0, len(self.data))
        while fixed_idx == moving_idx:
            fixed_idx = np.random.randint(0, len(self.data))

        # Hole die Bilder
        moving_img = self.data[moving_idx]  # (H, W, D)
        fixed_img  = self.data[fixed_idx]   # (H, W, D)

        print(f"Moving index: {moving_idx}, Fixed index: {fixed_idx}")
        print(f"Moving image shape: {moving_img.shape}, Fixed image shape: {fixed_img.shape}")

        # Transponieren → (D, H, W)
        moving_img = np.transpose(moving_img, (2, 0, 1))
        fixed_img  = np.transpose(fixed_img, (2, 0, 1))


        # Transformation anwenden (optional)
        if self.transform:
            moving_img = self.transform(moving_img)
            fixed_img  = self.transform(fixed_img)

        # In Torch-Tensor umwandeln
        moving_img = torch.from_numpy(moving_img).float()
        fixed_img  = torch.from_numpy(fixed_img).float()

        # Optional: Segmentierungen
        if self.seg_data is not None:
            moving_seg = self.seg_data[moving_idx]
            fixed_seg  = self.seg_data[fixed_idx]

            moving_seg = np.transpose(moving_seg, (2, 0, 1))
            fixed_seg  = np.transpose(fixed_seg, (2, 0, 1))

            moving_seg = torch.from_numpy(moving_seg).float()
            fixed_seg  = torch.from_numpy(fixed_seg).float()

            print(f"Moving image shape: {moving_img.shape}, Fixed image shape: {fixed_img.shape}")
            print(f"Moving segmentation shape: {moving_seg.shape}, Fixed segmentation shape: {fixed_seg.shape}")

            return moving_img, fixed_img, moving_seg, fixed_seg

        print(f"Moving image shape: {moving_img.shape}, Fixed image shape: {fixed_img.shape}")
        return moving_img, fixed_img
