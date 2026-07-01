import os
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

CKPT_DIR = "sst_ckpt_mdsr"


############################################
# MODEL
############################################

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale


class UpsamplerX4(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.conv(x)


class MDSR(nn.Module):
    def __init__(self):
        super().__init__()

        n_feats = 64

        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(16)])
        self.upsample = UpsamplerX4(n_feats)
        self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upsample(res)
        x = self.tail(x)
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
# DATASET (ULTIMATE FIX)
############################################

class SSTDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, vmin=25, vmax=35):

        def get_date(fname):
            return fname.split("_")[-1].replace(".npy", "")

        lr_dict = {
            get_date(f): os.path.join(lr_dir, f)
            for f in os.listdir(lr_dir) if f.endswith(".npy")
        }

        hr_dict = {
            get_date(f): os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir) if f.endswith(".npy")
        }

        dates = sorted(set(lr_dict) & set(hr_dict))
        assert len(dates) > 0, "No matching dates!"

        self.lr_files = [lr_dict[d] for d in dates]
        self.hr_files = [hr_dict[d] for d in dates]

        print(f"Matched samples: {len(self.lr_files)}")

        self.vmin = vmin
        self.vmax = vmax

    def normalize(self, x):
        return (x - self.vmin) / (self.vmax - self.vmin)

    def extract(self, data):

        # Case 1: dict
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    return v

        # Case 2: object array
        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                data = data.item()
                return self.extract(data)
            except:
                pass

        # Case 3: already ndarray
        if isinstance(data, np.ndarray):
            return data

        raise ValueError("Unknown data format")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):

        lr = np.load(self.lr_files[idx], allow_pickle=True)
        hr = np.load(self.hr_files[idx], allow_pickle=True)

        lr = self.extract(lr)
        hr = self.extract(hr)

        lr = np.array(lr, dtype=np.float32)
        hr = np.array(hr, dtype=np.float32)

        lr = np.squeeze(lr)
        hr = np.squeeze(hr)

        lr = self.normalize(lr)
        hr = self.normalize(hr)

        lr = np.expand_dims(lr, 0)
        hr = np.expand_dims(hr, 0)

        return torch.tensor(lr), torch.tensor(hr)


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=DistributedSampler(train_dataset),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=16) if rank == 0 else None

    model = DDP(MDSR().to(device), device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

    for epoch in range(1, 6001):

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

        if rank == 0:

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:

                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)

                    val_loss += (l1_loss(sr, hr) + 0.1 * (1 - ssim(sr, hr))).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            if epoch % 500 == 0:
                torch.save(model.module.state_dict(), f"{CKPT_DIR}/epoch_{epoch}.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()