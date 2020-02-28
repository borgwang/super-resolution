import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import cfg
from dataset import DIV2K
from model import EDSR
from transform import RandomCrop
from transform import RandomFlip
from transform import RandomRotate
from transform import ToTensor
from utils import visualize_samples


def get_data_loader():
    scale = cfg["scale"]
    transform = transforms.Compose([
        RandomCrop(256, scale=scale), 
        RandomFlip(vp=0, hp=0.5), 
        RandomRotate(), 
        ToTensor()])

    train_set = DIV2K(cfg["train_dir"], transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], 
                              shuffle=True, num_workers=8)
    return train_loader


def main():
    train_loader = get_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSR().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"])

    model.train()
    for epoch in range(cfg["n_epoch"]):
        for i, batch in enumerate(train_loader):
            lr, hr = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr), hr)
            print(f"epoch-{epoch} batch-{i} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        # eval
        model.eval()
        with torch.no_grad():
            n_samples = 3
            lr, hr = batch[0][:n_samples], batch[1][:n_samples]
            sr = model(lr.to(device)).cpu()
            samples = {"low-resolution": lr, "high-resolution": hr, "EDSR": sr}
            visualize_samples(samples, f"epoch-{epoch}", show=False, save=True)
        model.train()


if __name__ == "__main__":
    main()
