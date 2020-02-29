import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class DIV2K(Dataset):

    def __init__(self, paths, transform=None):
        self._lr_paths = self._get_filenames(paths["lr"])
        self._hr_paths = self._get_filenames(paths["hr"])
        assert len(self._lr_paths) == len(self._hr_paths)
        self.transform = transform

    def __len__(self):
        return len(self._lr_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lr_paths, hr_paths = self._lr_paths[idx], self._hr_paths[idx]
        lr_data = self.read_images_as_array(lr_paths)
        hr_data = self.read_images_as_array(hr_paths)

        sample = (lr_data, hr_data)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_filenames(self, directory):
        return np.array(sorted(glob.glob(os.path.join(directory, "*.png"))))[2:3]

    def read_images_as_array(self, paths):
        def read(path):
            return np.array(Image.open(path)).astype("float")

        return read(paths) if isinstance(paths, str) else \
                np.array([read(p) for p in paths])
