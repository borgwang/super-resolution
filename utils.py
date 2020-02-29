import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def read_images_as_array(paths):
    def read(path):
        return np.array(Image.open(path)).astype("float")

    return read(paths) if isinstance(paths, str) else \
            np.array([read(p) for p in paths])


def img2tensor(path):
    data = read_images_as_array(path)
    data = data[np.newaxis, :]
    data = (data / 255.0).transpose((0, 3, 1, 2)).copy()
    return torch.from_numpy(data).float()


def visualize_samples(samples, name, save=False, directory=None, size=3):
    cols = len(samples)
    rows = len(list(samples.values())[0])
    fig = plt.figure(figsize=(cols * size, rows * size))
    for i, (k, v) in enumerate(samples.items()):
        v = v.numpy().transpose((0, 2, 3, 1)).clip(0, 1)
        for j in range(rows):
            plt.subplot(rows, cols, j * cols + i + 1)
            plt.title(k)
            plt.imshow(v[j])
            plt.xticks([])
            plt.yticks([])
    if save and directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, name)
        plt.savefig(path)
        print(f"sampls saved to {path}")
    return fig


def compute_psnr(samples):
    hr, sr = samples["hr"], samples["sr"]
    hr = hr.numpy().transpose((0, 2, 3, 1)).clip(0, 1)
    sr = sr.numpy().transpose((0, 2, 3, 1)).clip(0, 1)
    psnrs = []
    for i in range(len(hr)):
        psnrs.append(10 * np.log10(1.0 / np.mean((hr[i] - sr[i]) ** 2)))
    return np.mean(psnrs)
