import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import math

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

CKPT_DIR = "sst_ckpt_sinsr_improved"
CSV_LOG = "loss_log_sinsr_improved.csv"

EPOCHS = 6000
BATCH_SIZE = 16
LR_RATE = 1e-4

VMIN = 25.0
VMAX = 35.0

############################################
# NORMALIZATION
############################################

def normalize(x):
    return 2 * (x - VMIN) / (VMAX - VMIN) - 1

############################################
# GRADIENT LOSS (🔥 KEY FIX)
############################################

def gradient_loss(pred, target):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]

    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

############################################
# SSIM LOSS
############################################

def ssim_loss(pred, target):
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim.mean()

############################################
# TIME EMBEDDING
############################################

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

############################################
# BLOCKS
############################################

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t).view(t.size(0), -1, 1, 1)
        h = self.act(self.conv2(h))
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.softmax(q.transpose(1, 2) @ k / (C ** 0.5), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2)

        return self.proj(out.reshape(B, C, H, W)) + x

############################################
# MODEL
############################################

class SinSR_UNet(nn.Module):
    def __init__(self, t_dim=256):
        super().__init__()

        self.time_embed = TimeEmbedding(t_dim)

        self.input = nn.Conv2d(1, 64, 3, padding=1)

        self.down1 = ResBlock(64, 128, t_dim)
        self.attn1 = Attention(128)

        self.down2 = ResBlock(128, 256, t_dim)
        self.mid = ResBlock(256, 256, t_dim)

        self.up1 = ResBlock(256, 128, t_dim)
        self.attn2 = Attention(128)

        self.up2 = ResBlock(128, 64, t_dim)

        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        t = torch.zeros(x.size(0), device=x.device)
        t_emb = self.time_embed(t)

        x1 = self.input(x)
        d1 = self.attn1(self.down1(x1, t_emb))
        d2 = self.down2(d1, t_emb)

        m = self.mid(d2, t_emb)

        u1 = self.attn2(self.up1(m, t_emb))
        u2 = self.up2(u1, t_emb)

        return x + self.out(u2)

############################################
# DATASET (same)
############################################

def extract_array(data):
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, np.ndarray):
                return v
    if isinstance(data, np.ndarray) and data.dtype == object:
        return extract_array(data.item())
    return data

class SSTDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):

        def get_date(f):
            return f.split("_")[-1].replace(".npy", "")

        lr_dict = {get_date(f): os.path.join(lr_dir, f) for f in os.listdir(lr_dir)}
        hr_dict = {get_date(f): os.path.join(hr_dir, f) for f in os.listdir(hr_dir)}

        dates = sorted(set(lr_dict) & set(hr_dict))

        self.lr = [lr_dict[d] for d in dates]
        self.hr = [hr_dict[d] for d in dates]

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, i):
        lr = extract_array(np.load(self.lr[i], allow_pickle=True))
        hr = extract_array(np.load(self.hr[i], allow_pickle=True))

        lr = normalize(torch.from_numpy(np.squeeze(lr).astype(np.float32))).unsqueeze(0)
        hr = normalize(torch.from_numpy(np.squeeze(hr).astype(np.float32))).unsqueeze(0)

        lr = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode="bicubic", align_corners=False).squeeze(0)

        return lr, hr

############################################
# DDP
############################################

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

############################################
# TRAIN
############################################

def compute_loss(pred, hr):
    mse = F.mse_loss(pred, hr)
    l1 = F.l1_loss(pred, hr)
    grad = gradient_loss(pred, hr)
    ssim = ssim_loss(pred, hr)

    return 0.6*mse + 0.2*l1 + 0.1*grad + 0.1*ssim

def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    train_dataset = SSTDataset(TRAIN_LR, TRAIN_HR)
    val_dataset = SSTDataset(VAL_LR, VAL_HR)

    sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DDP(SinSR_UNet().to(device), device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        with open(CSV_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, EPOCHS + 1):

        sampler.set_epoch(epoch)
        model.train()
        train_loss = 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            pred = model(lr)
            loss = compute_loss(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if rank == 0:

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    pred = model(lr)
                    val_loss += compute_loss(pred, hr).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss])

            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, f"{CKPT_DIR}/sinsr_epoch_{epoch}.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()