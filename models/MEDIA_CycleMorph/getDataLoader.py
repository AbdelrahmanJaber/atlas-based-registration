from __future__ import print_function, division, absolute_import, unicode_literals
import torch
import numpy as np
import os

class BaseDataProvider(object):

    def _load_data_and_label(self):
        data, labels, path = self._next_data()
        #data, label = self._augment_data(data, label)

        print(f"🔍 Lade Daten: {path} (Index: {self.data_idx}) shape: {data.shape}, label shape: {labels.shape}")

        #data = data.permute(1, 2, 0).float()
        #labels = labels.permute(1, 2, 0).float()

        data = data.numpy()
        labels = labels.numpy()

        nd = data.shape[0]
        nw = data.shape[1]
        nh = data.shape[2]
        return path, data.reshape(1, 1, nd, nw, nh), labels.reshape(1, 1, nd, nw, nh)

    def _toTorchFloatTensor(self, img):
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        path, data, labels = self._load_data_and_label()
        P = []

        X = self._toTorchFloatTensor(data)
        Y = self._toTorchFloatTensor(labels)
        P.append(path)

        return X, Y, P