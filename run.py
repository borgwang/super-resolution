import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import cfg
from dataset import *
from model import EDSR
from utils import visualize_samples


def main(args):
    # data preparation
    transform = transforms.Compose([
        RandomCrop(256, scale=4), 
        RandomFlip(vp=0, hp=0.5), 
        RandomRotate(), 
        ToTensor()])

    train_set = DIV2K(cfg["data"]["train"], transform=transform)
    # valid_set = DIV2K(cfg["data"]["valid"], transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
    model = EDSR().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7.5e-4)

    model.train()
    for epoch in range(300):
        for i, batch in enumerate(train_loader):
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr), hr)
            print(f"epoch-{epoch} batch-{i} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            lr, hr = batch["lr"][:3], batch["hr"][:3]
            sr = model(lr.to(device)).cpu()
            samples = {"lr": lr, "hr": hr, "sr": sr}
            visualize_samples(samples, f"epoch-{epoch}", show=False, save=True)
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
