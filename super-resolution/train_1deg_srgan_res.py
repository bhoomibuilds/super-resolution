import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

TRAIN_LR = f"{BASE}/train/LR"
TRAIN_HR = f"{BASE}/train/HR"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_DIR = "sst_ckpt_srgan_final"
CSV_LOG = "loss_log_srgan_final.csv"

EPOCHS = 6000
RESUME_CKPT = f"{CKPT_DIR}/srgan_epoch_2500.pth"

############################################
# GENERATOR
############################################

class ResidualBlock(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.BatchNorm2d(n_feats)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU()
        )

        self.tail = nn.Conv2d(64, 1, 9, padding=4)

    def forward(self, x):
        x1 = self.head(x)
        x = self.res_blocks(x1)
        x = self.mid(x) + x1
        x = self.upsample(x)
        return self.tail(x)

############################################
# DISCRIMINATOR
############################################

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride):
            return [
                nn.Conv2d(in_c, out_c, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=False)
            ]

        layers = []
        layers += block(1, 64, 1)
        layers += block(64, 64, 2)
        layers += block(64, 128, 1)
        layers += block(128, 128, 2)
        layers += block(128, 256, 1)
        layers += block(256, 256, 2)

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

############################################
# DATASET
############################################

class SSTDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, vmin=25, vmax=35):

        def get_date(fname):
            return fname.split("_")[-1].replace(".npy", "")

        lr_dict = {get_date(f): os.path.join(lr_dir, f)
                   for f in os.listdir(lr_dir) if f.endswith(".npy")}

        hr_dict = {get_date(f): os.path.join(hr_dir, f)
                   for f in os.listdir(hr_dir) if f.endswith(".npy")}

        dates = sorted(set(lr_dict) & set(hr_dict))

        self.lr_files = [lr_dict[d] for d in dates]
        self.hr_files = [hr_dict[d] for d in dates]

        self.vmin = vmin
        self.vmax = vmax

    def extract(self, data):
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, np.ndarray):
                    return v

        if isinstance(data, np.ndarray) and data.dtype == object:
            return self.extract(data.item())

        if isinstance(data, np.ndarray):
            return data

        raise ValueError("Unsupported format")

    def normalize(self, x):
        return (x - self.vmin) / (self.vmax - self.vmin)

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):

        lr = self.extract(np.load(self.lr_files[idx], allow_pickle=True))
        hr = self.extract(np.load(self.hr_files[idx], allow_pickle=True))

        lr = np.expand_dims(self.normalize(np.squeeze(lr)), 0)
        hr = np.expand_dims(self.normalize(np.squeeze(hr)), 0)

        return torch.tensor(lr, dtype=torch.float32), torch.tensor(hr, dtype=torch.float32)

############################################
# DDP
############################################

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

############################################
# TRAIN
############################################

def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    train_dataset = SSTDataset(TRAIN_LR, TRAIN_HR)
    val_dataset = SSTDataset(VAL_LR, VAL_HR)

    sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=sampler, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=16)

    G = DDP(Generator().to(device), device_ids=[rank])
    D = DDP(Discriminator().to(device), device_ids=[rank])

    g_opt = optim.Adam(G.parameters(), lr=1e-4)
    d_opt = optim.Adam(D.parameters(), lr=1e-4)

    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    ############################################
    # RESUME
    ############################################

    start_epoch = 1

    if RESUME_CKPT and os.path.exists(RESUME_CKPT):

        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        ckpt = torch.load(RESUME_CKPT, map_location=map_location)

        G.module.load_state_dict(ckpt["G"])
        D.module.load_state_dict(ckpt["D"])

        g_opt.load_state_dict(ckpt["g_opt"])
        d_opt.load_state_dict(ckpt["d_opt"])

        start_epoch = ckpt["epoch"] + 1

        if rank == 0:
            print(f"✅ Resumed from epoch {ckpt['epoch']}")

    ############################################
    # LOG FILE (NO OVERWRITE)
    ############################################

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

        if not os.path.exists(CSV_LOG):
            with open(CSV_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_g", "train_d", "val_loss"])

    ############################################
    # TRAIN LOOP
    ############################################

    for epoch in range(start_epoch, EPOCHS + 1):

        sampler.set_epoch(epoch)

        G.train()
        D.train()

        g_loss_total = 0
        d_loss_total = 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            # ---- D ----
            fake = G(lr).detach()

            d_opt.zero_grad()

            real_out = D(hr)
            fake_out = D(fake)

            d_loss = (
                bce(real_out, torch.ones_like(real_out)) +
                bce(fake_out, torch.zeros_like(fake_out))
            )

            d_loss.backward()
            d_opt.step()

            # ---- G ----
            g_opt.zero_grad()

            fake = G(lr)
            fake_out = D(fake)

            g_loss = (
                l1(fake, hr) +
                1e-3 * bce(fake_out, torch.ones_like(fake_out))
            )

            g_loss.backward()
            g_opt.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

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

            print(f"Epoch {epoch} | G {g_loss_total:.6f} | D {d_loss_total:.6f} | Val {val_loss:.6f}")

            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow([epoch, g_loss_total, d_loss_total, val_loss])

            ############################################
            # SAVE CKPT EVERY 500
            ############################################

            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "G": G.module.state_dict(),
                    "D": D.module.state_dict(),
                    "g_opt": g_opt.state_dict(),
                    "d_opt": d_opt.state_dict()
                }, f"{CKPT_DIR}/srgan_epoch_{epoch}.pth")

    dist.destroy_process_group()

############################################

if __name__ == "__main__":
    main()