import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RGB_MEAN = np.array([0.4488, 0.4371, 0.4040])


def conv(in_c, out_c, kernel_size=3, bias=True):
    return nn.Conv2d(in_c, out_c, kernel_size, padding=(kernel_size // 2), bias=bias)


class BasicBlock(nn.Module):

    def __init__(self, n_feats):
        super().__init__()
        stride = 1
        self.conv1 = conv(n_feats, n_feats, 3)
        self.conv2 = conv(n_feats, n_feats, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out 


class UpsampleBlock(nn.Module):

    def __init__(self, n_feats, scale):
        super().__init__()
        layers = []
        for _ in range(int(np.log2(scale))):
            layers.append(conv(n_feats, 4 * n_feats))
            layers.append(nn.PixelShuffle(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EDSR(nn.Module):

    def __init__(self):
        super().__init__()
        n_colors = 3
        n_feats = 64
        n_residual_blocks = 16
        scale = 4

        layers = []
        head = [conv(n_colors, n_feats)]
        body = [BasicBlock(n_feats) for _ in range(n_residual_blocks)]
        tail = [UpsampleBlock(n_feats, scale), conv(n_feats, n_colors)]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_mean = torch.Tensor(RGB_MEAN).reshape(1, -1, 1, 1).to(device)

    def forward(self, x):
        # normalize
        x -= self.rgb_mean
        x = self.head(x)
        out = self.body(x)
        out += x
        out = self.tail(out)
        # denormalize
        out += self.rgb_mean
        return out


if __name__ == "__main__":
    # test
    net = EDSR()
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())
