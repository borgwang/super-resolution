import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RGB_MEAN = np.array([0.4488, 0.4371, 0.4040])


def conv(in_channel, out_channel, kernel_size=3, bias=True, groups=None):
    groups = 1 if groups is None else groups
    return nn.Conv2d(in_channel, out_channel, kernel_size,
                     padding=(kernel_size // 2), bias=bias, groups=groups)


class ResBlock(nn.Module):

    def __init__(self, n_feats, rescale=1.0):
        super().__init__()
        self.conv1 = conv(n_feats, n_feats)
        self.conv2 = conv(n_feats, n_feats)
        self.rescale = torch.tensor(rescale, requires_grad=False)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out *= self.rescale
        out += x
        return out 


class UpsampleBlock(nn.Module):

    def __init__(self, n_feats, scale):
        super().__init__()
        layers = []
        if scale in (2, 4, 8):
            for _ in range(int(np.log2(scale))):
                layers.append(conv(n_feats, 4 * n_feats))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(conv(n_feats, 9 * n_feats))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Invalid scale={scale}")
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EDSR(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        n_colors = 3  # RGB
        n_feats = cfg["n_feats"]
        self.criterion = torch.nn.L1Loss()

        layers = []
        head = [conv(n_colors, n_feats)]
        body = [ResBlock(n_feats, cfg["rescale"]) 
                for _ in range(cfg["n_residual_blocks"])]
        body += [conv(n_feats, n_feats)]
        tail = [UpsampleBlock(n_feats, cfg["scale"])] 
        tail += [conv(n_feats, n_colors)]

        self.head = nn.Sequential(*head)

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_mean = torch.Tensor(RGB_MEAN).reshape(1, -1, 1, 1).to(device)

    def forward(self, x):
        x -= self.rgb_mean  # normalize
        x = self.head(x)
        out = self.body(x)
        x = out + x
        x = self.tail(x)
        x += self.rgb_mean  # denormalize
        return x

    def loss(self, sr, hr):
        l1_loss = self.criterion(sr, hr)
        return l1_loss


if __name__ == "__main__":
    net = EDSR()
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())
