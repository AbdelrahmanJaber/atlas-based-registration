import numpy as np
from torch.utils.data import Dataset
import torch
import json
from TPTBox import NII

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
    img = F.pad(img, padding)

    return img

def pad_to(img, target_shape):
    """
    Pads a 2D or 3D tensor (C, H, W) or (H, W) to the target shape
    """
    if img.ndim == 2:
        h, w = img.shape
        target_h, target_w = target_shape
    elif img.ndim == 3:
        _, h, w = img.shape
        target_h, target_w = target_shape
    else:
        raise ValueError("Image must be (H, W) or (C, H, W)")

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = [pad_left, pad_right, pad_top, pad_bottom]  # left, right, top, bottom
    img = F.pad(img, padding)

    # Also want to check if it is to big and then crop it in the center
    if img.ndim == 2:
        h, w = img.shape
        if h > target_h or w > target_w:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            img = img[start_h:start_h + target_h, start_w:start_w + target_w]
    elif img.ndim == 3:
        _, h, w = img.shape
        if h > target_h or w > target_w:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            img = img[:, start_h:start_h + target_h, start_w:start_w + target_w]

    return img


class CustomDataset(Dataset):
    '''
    Dataset that returns a pair of 2D slices (same z) from two randomly selected volumes
    '''
    def __init__(self, data_list, size, with_depth=False, transform=None):
        '''
        :param data: Numpy array of shape (N, H, W, D)
        :param seg_data: Optional numpy array of shape (N, H, W, D)
        :param transform: Optional transform to apply to slices
        '''
        self.data = self.read_data(data_list)
        self.transform = transform
        self.size = size
        self.with_depth = with_depth

    def read_data(self, data_list):
        """
        Reads data from a list of numpy arrays.
        """
        with open(data_list, 'rb') as f:
            data = json.load(f)

        pairs = [v for v in data.values()]  # Liste aller Paare
        return np.array(pairs)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Zufällige Volumenindizes
        moving_idx = np.random.randint(0, len(self.data))
        while moving_idx == idx:
            moving_idx = np.random.randint(0, len(self.data))

        atlas_img_path, atlas_seg_path = self.data[idx]
        moving_img_path, moving_seg_path = self.data[moving_idx]

        atlas_img = NII.load(atlas_img_path, seg=False).copy()
        atlas_seg = NII.load(atlas_seg_path, seg=True).copy()
        moving_img = NII.load(moving_img_path, seg=False).copy()
        moving_seg = NII.load(moving_seg_path, seg=True).copy()

        moving_img = (moving_img - NII.min(moving_img)) / (NII.max(moving_img) - NII.min(moving_img))
        atlas_img = (atlas_img - NII.min(atlas_img)) / (NII.max(atlas_img) - NII.min(atlas_img))


        moving_img = np.array(moving_img)
        atlas_img  = np.array(atlas_img)
        moving_seg = np.array(moving_seg)
        atlas_seg  = np.array(atlas_seg)

        if not self.with_depth:
            # Zufälliger Slice entlang D
            depth = min(atlas_img.shape[2], moving_img.shape[2])
            z = np.random.randint(0, depth)

            # Slice extrahieren
            atlas_img = atlas_img[:, :, z]  # (H, W)
            moving_img = moving_img[:, :, z]  # (H, W)

            atlas_seg = atlas_seg[:, :, z]  # (H, W)
            moving_seg = moving_seg[:, :, z]  # (H, W)

        if self.transform:
            moving_img = self.transform(moving_img)
            atlas_img  = self.transform(atlas_img)
            moving_seg = self.transform(moving_seg)
            atlas_seg  = self.transform(atlas_seg)

        moving_img = (moving_img - np.min(moving_img)) / (np.max(moving_img) - np.min(moving_img))
        atlas_img = (atlas_img - np.min(atlas_img)) / (np.max(atlas_img) - np.min(atlas_img))

        if not self.with_depth:
            # → [1, H, W] (für Channels-first Kompatibilität)
            moving_img = torch.from_numpy(moving_img).unsqueeze(0).float().cpu().contiguous()
            atlas_img  = torch.from_numpy(atlas_img).unsqueeze(0).float().cpu().contiguous()
            moving_seg = torch.from_numpy(moving_seg).unsqueeze(0).float().cpu().contiguous()
            atlas_seg  = torch.from_numpy(atlas_seg).unsqueeze(0).float().cpu().contiguous()
        else:
            moving_img = torch.from_numpy(moving_img).permute(2, 0, 1).float().cpu().contiguous()
            atlas_img  = torch.from_numpy(atlas_img).permute(2, 0, 1).float().cpu().contiguous()
            moving_seg = torch.from_numpy(moving_seg).permute(2, 0, 1).float().cpu().contiguous()
            atlas_seg  = torch.from_numpy(atlas_seg).permute(2, 0, 1).float().cpu().contiguous()

        moving_img = pad_to(moving_img, target_shape=self.size)
        atlas_img = pad_to(atlas_img, target_shape=self.size)
        moving_seg = pad_to(moving_seg, target_shape=self.size)
        atlas_seg = pad_to(atlas_seg, target_shape=self.size)

        return atlas_img, moving_img, atlas_seg, moving_seg