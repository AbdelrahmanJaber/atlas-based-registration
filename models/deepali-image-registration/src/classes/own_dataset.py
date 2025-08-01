import numpy as np
from torch.utils.data import Dataset
import torch

import torch.nn.functional as F

def pad_to_multiple(img, multiple=16):
    """
    Pads a 2D or 3D tensor (C, H, W) or (H, W) to next multiple of `multiple`
    """
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        _, h, w = img.shape
    else:
        raise ValueError("Image must be (H, W) or (C, H, W)")

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = [pad_left, pad_right, pad_top, pad_bottom]  # left, right, top, bottom
    img = F.pad(img, [1, 1, 2, 2])

    return img


class CustomDataset(Dataset):
    '''
    Dataset that returns a pair of 2D slices (same z) from two randomly selected volumes
    '''
    def __init__(self, data, seg_data=None, transform=None):
        '''
        :param data: Numpy array of shape (N, H, W, D)
        :param seg_data: Optional numpy array of shape (N, H, W, D)
        :param transform: Optional transform to apply to slices
        '''
        assert data.ndim == 4, "Expected data shape (N, H, W, D)"
        self.data = data
        self.seg_data = seg_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Zufällige Volumenindizes
        idx1 = np.random.randint(0, len(self.data))
        idx2 = np.random.randint(0, len(self.data))

        # Zufälliger Slice entlang D
        depth = self.data.shape[3]
        z = np.random.randint(0, depth)

        # Slice extrahieren
        moving_img = self.data[idx1, :, :, z]  # (H, W)
        fixed_img  = self.data[idx2, :, :, z]  # (H, W)

        if self.transform:
            moving_img = self.transform(moving_img)
            fixed_img  = self.transform(fixed_img)

        moving_img = (moving_img - np.min(moving_img)) / (np.max(moving_img) - np.min(moving_img))
        fixed_img = (fixed_img - np.min(fixed_img)) / (np.max(fixed_img) - np.min(fixed_img))

        # → [1, H, W] (für Channels-first Kompatibilität)
        moving_img = torch.from_numpy(moving_img).unsqueeze(0).float()
        fixed_img  = torch.from_numpy(fixed_img).unsqueeze(0).float()

        print(f"BEFORE:     Moving image shape: {moving_img.shape}, Fixed image shape: {fixed_img.shape}")

        moving_img = pad_to_multiple(moving_img, multiple=16)
        fixed_img = pad_to_multiple(fixed_img, multiple=16)

        print(f"AFTER:      Moving image shape: {moving_img.shape}, Fixed image shape: {fixed_img.shape}")

        if self.seg_data is not None:
            moving_seg = self.seg_data[idx1, :, :, z]
            fixed_seg  = self.seg_data[idx2, :, :, z]

            if self.transform:
                moving_seg = self.transform(moving_seg)
                fixed_seg  = self.transform(fixed_seg)

            moving_seg = torch.from_numpy(moving_seg).unsqueeze(0).float()
            fixed_seg  = torch.from_numpy(fixed_seg).unsqueeze(0).float()

            moving_seg = pad_to_multiple(moving_seg, multiple=16)
            fixed_seg = pad_to_multiple(fixed_seg, multiple=16)


            return moving_img, fixed_img, moving_seg, fixed_seg
        else:
            return moving_img, fixed_img