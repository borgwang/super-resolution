import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import cfg_dict
from dataset import DIV2K
from model import EDSR
from transform import RandomCrop
from transform import RandomFlip
from transform import RandomRotate
from transform import ToTensor
from utils import compute_psnr
from utils import img2tensor
from utils import visualize_samples


def get_data_loader(cfg, data_dir, batch_size=None):
    batch_size = cfg["batch_size"] if batch_size is None else batch_size
    transform = transforms.Compose([
        RandomCrop(cfg["hr_crop_size"], cfg["scale"]), 
        ToTensor()])
    dataset = DIV2K(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


def main(args):
    cfg = cfg_dict[args.cfg_name]
    writer = SummaryWriter(os.path.join("runs", args.cfg_name))
    train_loader = get_data_loader(cfg, cfg["train_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSR(cfg).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"],
                                 betas=(0.9, 0.999), eps=1e-8)

    global_batches = 0
    if args.train:
        for epoch in range(cfg["n_epoch"]):
            model.train()
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                lr, hr = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                sr = model(lr)
                loss = model.loss(sr, hr)
                # loss = criterion(model(lr), hr)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                global_batches += 1
                if global_batches % cfg["lr_decay_every"] == 0:
                    for param_group in optimizer.param_groups:
                        print(f"decay lr to {param_group['lr'] / 10}")
                        param_group["lr"] /= 10

            if epoch % args.log_every == 0:
                model.eval()
                with torch.no_grad():
                    batch_samples = {"lr": batch[0], "hr": batch[1], 
                                     "sr": sr.cpu()}
                    writer.add_scalar("training-loss", 
                                      running_loss / len(train_loader),
                                      global_step=global_batches)
                    writer.add_scalar("PSNR", compute_psnr(batch_samples), 
                                      global_step=global_batches)
                    samples = {k: v[:3] for k, v in batch_samples.items()}
                    fig = visualize_samples(samples, f"epoch-{epoch}")
                    writer.add_figure("sample-visualization", fig, 
                                      global_step=global_batches)

            if epoch % args.save_every == 0:
                state = {"net": model.state_dict(), 
                         "optim": optimizer.state_dict()}
                checkpoint_dir = args.checkpoint_dir
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                path = os.path.join(checkpoint_dir, args.cfg_name)
                torch.save(state, path)
    
    # eval
    if args.eval:
        assert args.model_path and args.lr_img_path
        print(f"evaluating {args.lr_img_path}")
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optim"])

        with torch.no_grad():
            lr = img2tensor(args.lr_img_path)
            sr = model(lr.clone().to(device)).cpu()
            samples = {"lr": lr, "sr": sr}
            if args.hr_img_path:
                samples["hr"] = img2tensor(args.hr_img_path)
                print(f"PSNR: {compute_psnr(samples)}")
            directory = os.path.dirname(args.lr_img_path)
            name = f"eval-{args.cfg_name}-{args.lr_img_path.split('/')[-1]}"
            visualize_samples(samples, name, save=True, 
                              directory=directory, size=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--cfg_name", type=str, 
                        default="scale4-feat64-block16")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lr_img_path", type=str, default=None)
    parser.add_argument("--hr_img_path", type=str, default=None)

    parser.add_argument("--save_every", type=int, default=100, 
                        help="save model every n epochs")
    parser.add_argument("--log_every", type=int, default=50, 
                        help="log every n epochs")

    main(parser.parse_args())
