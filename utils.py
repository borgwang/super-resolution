import os

import matplotlib.pyplot as plt
import numpy as np

from config import cfg


def visualize_samples(samples, name, save=False):
    cols = len(samples)
    rows = len(list(samples.values())[0])
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i, (k, v) in enumerate(samples.items()):
        v = v.numpy().transpose((0, 2, 3, 1))
        for j in range(rows):
            plt.subplot(rows, cols, j * cols + i + 1)
            plt.title(k)
            plt.imshow(v[j])
            plt.xticks([])
            plt.yticks([])
    if save:
        diretory = cfg["sample_dir"]
        if not os.path.exists(diretory):
            os.makedirs(diretory)
        plt.savefig(os.path.join(diretory, name))
    return fig


def compute_psnr(samples):
    hr, sr = samples["hr"], samples["sr"]
    hr = hr.numpy().transpose((0, 2, 3, 1))
    sr = sr.numpy().transpose((0, 2, 3, 1))
    psnrs = []
    for i in range(len(hr)):
        psnrs.append(10 * np.log10(1.0 / np.mean((hr[i] - sr[i]) ** 2)))
    return np.mean(psnrs)

