import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import xarray as xr

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

############################################
# PATHS
############################################
BASE = "data_wind"

TRAIN_U_LR = f"{BASE}/wind_u/train/LR/2015.nc"
TRAIN_V_LR = f"{BASE}/wind_v/train/LR/2015.nc"
TRAIN_U_HR = f"{BASE}/wind_u/train/HR/2015.nc"
TRAIN_V_HR = f"{BASE}/wind_v/train/HR/2015.nc"

VAL_U_LR = f"{BASE}/wind_u/val/LR/2020.nc"
VAL_V_LR = f"{BASE}/wind_v/val/LR/2020.nc"
VAL_U_HR = f"{BASE}/wind_u/val/HR/2020.nc"
VAL_V_HR = f"{BASE}/wind_v/val/HR/2020.nc"

CKPT_DIR = "wind_realesrgan_ckpt"
CSV_LOG = "wind_loss_log.csv"

############################################
# RESUME SETTINGS
############################################
RESUME = True
RESUME_EPOCH = 4500
CKPT_PATH = f"{CKPT_DIR}/epoch_{RESUME_EPOCH}.pth"

############################################
# DATASET
############################################
class WindDataset(Dataset):
    def __init__(self, lr_u, lr_v, hr_u, hr_v, mean=None, std=None):

        self.lr_u = xr.open_dataset(lr_u)["u10"].values
        self.lr_v = xr.open_dataset(lr_v)["v10"].values
        self.hr_u = xr.open_dataset(hr_u)["u10"].values
        self.hr_v = xr.open_dataset(hr_v)["v10"].values

        if mean is None:
            all_data = np.concatenate([
                self.lr_u.flatten(),
                self.lr_v.flatten(),
                self.hr_u.flatten(),
                self.hr_v.flatten()
            ])
            self.mean = all_data.mean()
            self.std = all_data.std()
        else:
            self.mean = mean
            self.std = std

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return self.lr_u.shape[0]

    def __getitem__(self, idx):
        lr = np.stack([self.lr_u[idx], self.lr_v[idx]])
        hr = np.stack([self.hr_u[idx], self.hr_v[idx]])

        return torch.tensor(self.normalize(lr)).float(), \
               torch.tensor(self.normalize(hr)).float()

############################################
# MODEL
############################################
class RRDB(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(64) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.body(x1))
        feat = x1 + x2

        feat = self.lrelu(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.lrelu(self.up1(feat))

        feat = self.lrelu(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.lrelu(self.up2(feat))

        return self.conv3(feat)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1),
                nn.InstanceNorm2d(o),
                nn.LeakyReLU(0.2)
            )

        self.net = nn.Sequential(
            block(2, 64, 1),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            nn.Conv2d(256, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

############################################
# DDP SETUP
############################################
def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

############################################
# MAIN
############################################
def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    ############################################
    # DATA
    ############################################
    train_dataset = WindDataset(TRAIN_U_LR, TRAIN_V_LR, TRAIN_U_HR, TRAIN_V_HR)
    mean, std = train_dataset.mean, train_dataset.std

    val_dataset = WindDataset(
        VAL_U_LR, VAL_V_LR, VAL_U_HR, VAL_V_HR, mean, std
    )

    sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=8,
                              sampler=sampler, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=8) if rank == 0 else None

    ############################################
    # MODELS
    ############################################
    G = DDP(Generator().to(device), device_ids=[rank])
    D = DDP(Discriminator().to(device), device_ids=[rank])

    ############################################
    # LOSS + OPTIM
    ############################################
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    g_opt = optim.Adam(G.parameters(), lr=1e-4)
    d_opt = optim.Adam(D.parameters(), lr=1e-4)

    ############################################
    # RESUME (FIXED)
    ############################################
    start_epoch = 1

    if RESUME and os.path.exists(CKPT_PATH):

        if rank == 0:
            print(f"🔁 Resuming from {CKPT_PATH}")

        ckpt = torch.load(CKPT_PATH, map_location=device)

        # OLD FORMAT (only generator)
        if isinstance(ckpt, dict) and "G" not in ckpt:
            if rank == 0:
                print("⚠️ Old checkpoint detected (Generator only)")

            G.module.load_state_dict(ckpt)
            start_epoch = RESUME_EPOCH + 1

        # NEW FORMAT (full)
        else:
            if rank == 0:
                print("✅ Full checkpoint detected")

            G.module.load_state_dict(ckpt["G"])
            D.module.load_state_dict(ckpt["D"])
            g_opt.load_state_dict(ckpt["g_opt"])
            d_opt.load_state_dict(ckpt["d_opt"])

            start_epoch = ckpt["epoch"] + 1

    ############################################
    # LOGGING
    ############################################
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

        if not RESUME:
            with open(CSV_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "g_loss", "d_loss", "val_loss"])

    ############################################
    # TRAIN LOOP
    ############################################
    for epoch in range(start_epoch, 6001):

        sampler.set_epoch(epoch)
        G.train(); D.train()

        g_total, d_total = 0, 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            # ---- D ----
            fake = G(lr).detach()

            loss_D = (
                bce(D(hr), torch.ones_like(D(hr))) +
                bce(D(fake), torch.zeros_like(D(fake)))
            ) * 0.5

            d_opt.zero_grad()
            loss_D.backward()
            d_opt.step()

            # ---- G ----
            fake = G(lr)
            loss_G = l1(fake, hr) + 0.01 * bce(D(fake), torch.ones_like(D(fake)))

            g_opt.zero_grad()
            loss_G.backward()
            g_opt.step()

            g_total += loss_G.item()
            d_total += loss_D.item()

        g_total /= len(train_loader)
        d_total /= len(train_loader)

        ############################################
        # VALIDATION
        ############################################
        if rank == 0:

            G.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    val_loss += l1(G(lr), hr).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | G:{g_total:.6f} | D:{d_total:.6f} | Val:{val_loss:.6f}")

            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow([epoch, g_total, d_total, val_loss])

            ############################################
            # SAVE (NEW FORMAT)
            ############################################
            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "G": G.module.state_dict(),
                    "D": D.module.state_dict(),
                    "g_opt": g_opt.state_dict(),
                    "d_opt": d_opt.state_dict()
                }, f"{CKPT_DIR}/epoch_{epoch}.pth")

    dist.destroy_process_group()

############################################

if __name__ == "__main__":
    main()