import os
import matplotlib.pyplot as plt

from config import cfg


def visualize_samples(samples, name, show=True, save=False):
    cols = len(samples)
    rows = len(list(samples.values())[0])
    plt.figure(figsize=(rows * 5, cols * 5))
    for i, (k, v) in enumerate(samples.items()):
        for j in range(rows):
            sample = v[j].numpy().transpose((1, 2, 0))
            print(f"{k}-shape {sample.shape}")
            plt.subplot(rows, cols, j * cols + i + 1)
            plt.title(k)
            plt.imshow(sample)
            plt.xticks([])
            plt.yticks([])
    if save:
        diretory = cfg["sample_dir"]
        if not os.path.exists(diretory):
            os.makedirs(diretory)
        plt.savefig(os.path.join(diretory, name))
    if show:
        plt.show()
