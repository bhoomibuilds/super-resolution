import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
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

CKPT_DIR = "sst_ckpt_realesrgan"
CSV_LOG = "loss_log_realesrgan.csv"

RESUME = True
RESUME_PATH = "sst_ckpt_realesrgan/realesrgan_epoch_2500.pth"
TOTAL_EPOCHS = 6000

############################################
# RRDB
############################################

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3*growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4*growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, nf=64, nb=10):
        super().__init__()

        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.last_conv = nn.Conv2d(nf, out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.upconv1(fea))

        fea = self.lrelu(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.upconv2(fea))

        return self.last_conv(self.lrelu(self.hr_conv(fea)))

############################################
# DISCRIMINATOR
############################################

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False)
            )

        self.model = nn.Sequential(
            block(1, 64, 1),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            nn.Conv2d(256, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

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
        return data

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

    train_loader = DataLoader(train_dataset, batch_size=8,
                              sampler=sampler, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=8) if rank == 0 else None

    generator = DDP(RRDBNet().to(device), device_ids=[rank])
    discriminator = DDP(Discriminator().to(device), device_ids=[rank])

    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    g_opt = optim.Adam(generator.parameters(), lr=1e-4)
    d_opt = optim.Adam(discriminator.parameters(), lr=1e-4)

    ##########################################
    # RESUME FIX
    ##########################################
    start_epoch = 1

    if RESUME:
        checkpoint = torch.load(RESUME_PATH, map_location=device)

        if isinstance(checkpoint, dict) and "generator" in checkpoint:
            generator.module.load_state_dict(checkpoint["generator"])
            discriminator.module.load_state_dict(checkpoint["discriminator"])
            g_opt.load_state_dict(checkpoint["g_opt"])
            d_opt.load_state_dict(checkpoint["d_opt"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            generator.module.load_state_dict(checkpoint)
            start_epoch = 2001   # 🔥 FORCE START

        if rank == 0:
            print(f"✅ Starting from epoch {start_epoch}")

    ##########################################
    # LOG
    ##########################################
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

    ##########################################
    # TRAIN LOOP
    ##########################################
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()

        g_loss_total = 0
        d_loss_total = 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            with torch.no_grad():
                fake_hr = generator(lr)

            pred_real = discriminator(hr)
            pred_fake = discriminator(fake_hr.detach())

            d_loss = (
                bce_loss(pred_real, torch.ones_like(pred_real)) +
                bce_loss(pred_fake, torch.zeros_like(pred_fake))
            ) * 0.5

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_hr = generator(lr)
            pred_fake = discriminator(fake_hr)

            g_loss = l1_loss(fake_hr, hr) + 0.01 * bce_loss(pred_fake, torch.ones_like(pred_fake))

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        g_loss_total /= len(train_loader)
        d_loss_total /= len(train_loader)

        if rank == 0:

            generator.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = generator(lr)
                    val_loss += l1_loss(sr, hr).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | G {g_loss_total:.6f} | D {d_loss_total:.6f} | Val {val_loss:.6f}")

            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "generator": generator.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "g_opt": g_opt.state_dict(),
                    "d_opt": d_opt.state_dict()
                }, f"{CKPT_DIR}/realesrgan_epoch_{epoch}.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()