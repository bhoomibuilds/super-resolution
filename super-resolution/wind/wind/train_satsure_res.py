import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import xarray as xr
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

############################################
# CONFIG
############################################
RESUME = True
RESUME_CKPT = "wind_uv_ckpt/satsure_epoch_3500.pth"

START_EPOCH = 3501
TOTAL_EPOCHS = 6000

############################################
# MODEL BLOCKS
############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class SatSuRE(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_res_blocks=16, scale_factor=4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        upsample_layers = []
        for _ in range(int(scale_factor / 2)):
            upsample_layers.append(UpsampleBlock(64, 2))

        self.upsample = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.conv2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.conv3(x)

############################################
# SSIM
############################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D @ _1D.t()
    return _2D.expand(channel,1,window_size,window_size).contiguous()

def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - (mu1*mu2)

    C1, C2 = 0.01**2, 0.03**2

    return (((2*mu1*mu2 + C1)*(2*sigma12 + C2)) /
            ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()

############################################
# DATASET
############################################
class WindDataset(Dataset):
    def __init__(self, lr_u_nc, lr_v_nc, hr_u_nc, hr_v_nc):
        self.lr_u = xr.open_dataset(lr_u_nc)["u10"].values
        self.lr_v = xr.open_dataset(lr_v_nc)["v10"].values
        self.hr_u = xr.open_dataset(hr_u_nc)["u10"].values
        self.hr_v = xr.open_dataset(hr_v_nc)["v10"].values

    def __len__(self):
        return self.lr_u.shape[0]

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def __getitem__(self, idx):
        lr = np.stack([self.lr_u[idx], self.lr_v[idx]])
        hr = np.stack([self.hr_u[idx], self.hr_v[idx]])

        lr = self.normalize(lr)
        hr = self.normalize(hr)

        return torch.tensor(lr).float(), torch.tensor(hr).float()

############################################
# DDP SETUP
############################################
def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

############################################
# MAIN TRAINING
############################################
def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    train_dataset = WindDataset(
        "data_wind/wind_u/train/LR/2015.nc",
        "data_wind/wind_v/train/LR/2015.nc",
        "data_wind/wind_u/train/HR/2015.nc",
        "data_wind/wind_v/train/HR/2015.nc"
    )

    val_dataset = WindDataset(
        "data_wind/wind_u/val/LR/2020.nc",
        "data_wind/wind_v/val/LR/2020.nc",
        "data_wind/wind_u/val/HR/2020.nc",
        "data_wind/wind_v/val/HR/2020.nc"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=DistributedSampler(train_dataset),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=1)

    model = SatSuRE().to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    ############################################
    # LOSS HISTORY
    ############################################
    train_losses = []
    val_losses = []

    start_epoch = 1

    ############################################
    # RESUME
    ############################################
    if RESUME:
        ckpt = torch.load(RESUME_CKPT, map_location=device)

        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

        # 🔥 LOAD LOSS HISTORY
        if "train_losses" in ckpt:
            train_losses = ckpt["train_losses"]
            val_losses = ckpt["val_losses"]

        print(f"✅ Resumed from epoch {start_epoch-1}")

    ############################################
    # TRAIN LOOP
    ############################################
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        model.train()
        train_loss = 0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            loss = l1_loss(sr, hr) + 0.1*(1 - ssim(sr, hr))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        ################################
        # VALIDATION
        ################################
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                val_loss += (l1_loss(sr, hr) + 0.1*(1 - ssim(sr, hr))).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        ################################
        # LOGGING
        ################################
        if rank == 0:
            print(f"Epoch {epoch} Train:{train_loss:.6f} Val:{val_loss:.6f}")

            os.makedirs("wind_uv_ckpt", exist_ok=True)

            # SAVE EVERY 200 EPOCHS
            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses
                }, f"wind_uv_ckpt/satsure_epoch_{epoch}.pth")

                # ALSO SAVE JSON (optional)
                with open("wind_uv_ckpt/loss_history.json", "w") as f:
                    json.dump({
                        "train_losses": train_losses,
                        "val_losses": val_losses
                    }, f)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()