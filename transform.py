import numpy as np
import torch


class RandomCrop:

    def __init__(self, hr_crop_size, scale):
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = self.hr_crop_size // scale
        self.scale = scale

    def __call__(self, sample):
        lr, hr = sample
        lr_h, lr_w, lr_c = lr.shape

        np.random.seed(np.random.randint(100000))
        lr_x = np.random.randint(0, lr_h - self.lr_crop_size + 1)
        lr_y = np.random.randint(0, lr_w - self.lr_crop_size + 1)
        print(lr_x, (0, lr_h - self.lr_crop_size + 1))
        print(lr_y, (0, lr_w - self.lr_crop_size + 1))

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
