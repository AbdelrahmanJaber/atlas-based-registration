from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
from getDataLoader import BaseDataProvider
import scipy.io as sio
import torch.nn.functional as F
import torch

def pad_to_multiple(img, multiple=16):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()

    d, h, w = img.shape  # [D, H, W]

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad paddet die letzten beiden Dimensionen → wir müssen auf [*, H, W] enden
    img = img  # [D, H, W] – keine Permute nötig

    # pad H, W → letzteres zuerst
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))  # → paddet W, H

    return img  # bleibt [D, H, W]

def select_d_slices(data, num_slices=5, start=0, stop=None):
    """
    data: Tensor oder Array mit shape [D, H, W]
    """
    if stop is None:
        stop = data.shape[0]  # D liegt an dim=0

    indices = np.linspace(start, stop - 1, num_slices).astype(int)
    return data[indices, :, :]  # subset über die D-Achse

class DataProvider(BaseDataProvider):
    def __init__(self, inputSize, fineSize, path, labelPath, mode=None):
        super(DataProvider, self).__init__()
        self.inputSize  = inputSize
        self.fineSize   = fineSize
        self.path       = path
        self.labelPath  = labelPath
        self.mode       = mode
        self.data_idx   = -1
        self.n_data     = self._load_data()

    def _load_data(self):
        self.imageNum = []

        datapath = os.path.join(self.path)
        dataFiles  = sorted(os.listdir(datapath))
        for isub, dataName in enumerate(dataFiles):
            self.imageNum.append(os.path.join(datapath, dataName))
        label = np.load(os.path.join(self.labelPath, 'atlas_norm.npz'))
        self.label = label['vol'] #['seg']
        self.label = np.transpose(self.label, (2, 1, 0))  # aus (W, H, D) → (D, H, W)
        self.label = select_d_slices(self.label, num_slices=10)
        self.label = pad_to_multiple(self.label, 16)

        if self.mode == "train":
            np.random.shuffle(self.imageNum)
        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        dataPath = self.imageNum[self.data_idx]
        data_ = sio.loadmat(dataPath)
        data = data_['data_affine']
        data = np.transpose(data, (0, 1, 2))  # aus (W, H, D) → (D, H, W)
        # Gleichmäßig 5 Indizes aus 0..199
        data = select_d_slices(data, num_slices=10)
        data = pad_to_multiple(data, 16)
        return data, self.label, dataPath

    def _augment_data(self, data, label):
        if self.mode == "train":
            # Rotation 90
            op = np.random.randint(0, 4)
            data, label = np.rot90(data, op), np.rot90(label, op)

            # Flip horizon / vertical
            op = np.random.randint(0, 3)
            if op < 2:
                data, label = np.flip(data, op), np.flip(label, op)

        return data, label


