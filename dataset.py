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
        return np.array(sorted(glob.glob(os.path.join(directory, "*.png"))))

    def read_images_as_array(self, paths):
        def read(path):
            return np.array(Image.open(path)).astype("float")

        if isinstance(paths, str):
            return read(paths)
        else:
            return np.array([read(p) for p in paths])


class RandomCrop:

    def __init__(self, hr_crop_size, scale):
        self.hr_crop_size = hr_crop_size
        self.scale = scale

    def __call__(self, sample):
        lr, hr = sample
        lr_h, lr_w, lr_c = lr.shape

        self.lr_crop_size = self.hr_crop_size // self.scale
        lr_x = np.random.randint(0, lr_h - self.lr_crop_size + 1)
        lr_y = np.random.randint(0, lr_w - self.lr_crop_size + 1)

        hr_x, hr_y = lr_x * self.scale, lr_y * self.scale

        lr_cropped = lr[lr_x:lr_x + self.lr_crop_size, 
                        lr_y:lr_y + self.lr_crop_size]
        hr_cropped = hr[hr_x:hr_x + self.hr_crop_size, 
                        hr_y:hr_y + self.hr_crop_size]
        return lr_cropped, hr_cropped


class RandomFlip:

    def __init__(self, vp=0.5, hp=0.5):
        self.vp, self.hp = vp, hp
        
    def __call__(self, sample):
        lr, hr = sample
        if np.random.uniform() < self.vp:
            lr = np.flip(lr, axis=0)
            hr = np.flip(hr, axis=0)
        if np.random.uniform() < self.hp:
            lr = np.flip(lr, axis=1)
            hr = np.flip(hr, axis=1)
        return lr, hr


class RandomRotate:

    def __call__(self, sample):
        lr, hr = sample
        k = np.random.randint(4)
        lr, hr = np.rot90(lr, k), np.rot90(hr, k)
        return lr, hr


class ToTensor:

    def __call__(self, sample):
        lr, hr = sample
        lr = (lr / 255.0).transpose((2, 0, 1)).copy()
        hr = (hr / 255.0).transpose((2, 0, 1)).copy()
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return lr, hr
