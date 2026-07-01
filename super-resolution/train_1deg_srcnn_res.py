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

CKPT_DIR = "sst_ckpt_srcnn"
CSV_LOG = "loss_log_srcnn.csv"

START_EPOCH = 6001
END_EPOCH = 15000
RESUME_CKPT = f"{CKPT_DIR}/srcnn_epoch_6000.pth"

############################################
# MODEL (SRCNN)
############################################

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

############################################
# SSIM
############################################

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2/(2*sigma**2))
        for x in range(window_size)
    ])
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
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    return (((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
           ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()

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

        lr = np.squeeze(lr)
        hr = np.squeeze(hr)

        lr = self.normalize(lr)
        hr = self.normalize(hr)

        lr = torch.tensor(lr, dtype=torch.float32).unsqueeze(0)
        hr = torch.tensor(hr, dtype=torch.float32).unsqueeze(0)

        # Bicubic upscale LR → HR
        lr = F.interpolate(
            lr.unsqueeze(0),
            size=hr.shape[-2:],
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        return lr, hr

############################################
# DDP SETUP
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=16) if rank == 0 else None

    model = DDP(SRCNN().to(device), device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    ############################################
    # RESUME
    ############################################

    if os.path.exists(RESUME_CKPT):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.module.load_state_dict(
            torch.load(RESUME_CKPT, map_location=map_location)
        )
        if rank == 0:
            print(f"✅ Loaded checkpoint: {RESUME_CKPT}")
    else:
        if rank == 0:
            print("⚠️ No checkpoint found, starting fresh")

    ############################################
    # LOG SETUP
    ############################################

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

        if not os.path.exists(CSV_LOG):
            with open(CSV_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    ############################################
    # TRAIN LOOP
    ############################################

    for epoch in range(START_EPOCH, END_EPOCH + 1):

        sampler.set_epoch(epoch)

        model.train()
        train_loss = 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            loss = l1_loss(sr, hr) + 0.1 * (1 - ssim(sr, hr))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ############################################
        # VALIDATION
        ############################################

        if rank == 0:

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)

                    val_loss += (
                        l1_loss(sr, hr) + 0.1 * (1 - ssim(sr, hr))
                    ).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss])

            ############################################
            # SAVE CHECKPOINT
            ############################################

            if epoch % 500 == 0:
                torch.save(
                    model.module.state_dict(),
                    f"{CKPT_DIR}/srcnn_epoch_{epoch}.pth"
                )

    dist.destroy_process_group()

############################################

if __name__ == "__main__":
    main()
