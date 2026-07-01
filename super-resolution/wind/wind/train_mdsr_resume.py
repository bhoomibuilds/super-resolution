import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import xarray as xr

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

############################################
# MODEL
############################################
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)

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
    def __init__(self, n_resblocks=16, n_feats=64):
        super().__init__()
        self.head = nn.Conv2d(2, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.upsample = UpsamplerX4(n_feats)
        self.tail = nn.Conv2d(n_feats, 2, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upsample(res)
        return self.tail(x)

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

    sigma1 = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1**2
    sigma2 = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2**2
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1*mu2

    C1, C2 = 0.01**2, 0.03**2

    return (((2*mu1*mu2 + C1)*(2*sigma12 + C2)) /
            ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))).mean()

############################################
# DATASET
############################################
class WindDataset(Dataset):
    def __init__(self, lr_u, lr_v, hr_u, hr_v):
        self.lr_u = xr.open_dataset(lr_u)["u10"].values
        self.lr_v = xr.open_dataset(lr_v)["v10"].values
        self.hr_u = xr.open_dataset(hr_u)["u10"].values
        self.hr_v = xr.open_dataset(hr_v)["v10"].values

    def __len__(self):
        return self.lr_u.shape[0]

    def __getitem__(self, idx):
        lr = torch.tensor(np.stack([self.lr_u[idx], self.lr_v[idx]])).float()
        hr = torch.tensor(np.stack([self.hr_u[idx], self.hr_v[idx]])).float()
        return lr, hr

############################################
# DDP SETUP
############################################
def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


############################################
# MAIN
############################################
def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    ################################
    # DATA
    ################################
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

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=train_sampler, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=1)

    ################################
    # MODEL
    ################################
    model = MDSR().to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    ################################
    # RESUME
    ################################
    resume_path = "wind_uv_ckpt_mdsr/mdsr_epoch_6000.pth"
    start_epoch = 1
    total_epochs = 10000   # 🔥 YOUR TARGET

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path,
                       
                                map_location={'cuda:0': f'cuda:{rank}'})

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            model.module.load_state_dict(checkpoint)
            start_epoch = 6001

        dist.barrier()

        if rank == 0:
            print("✅ FULL checkpoint loaded")
            print(f"🚀 Resuming from epoch {start_epoch} → {total_epochs}")

    ################################
    # LOGGING
    ################################
    if rank == 0:
        os.makedirs("wind_uv_ckpt_mdsr", exist_ok=True)

    loss_log = []
    if os.path.exists("wind_uv_mdsr_loss.csv") and rank == 0:
        loss_log = np.loadtxt("wind_uv_mdsr_loss.csv",
                             delimiter=",", skiprows=1).tolist()

    ################################
    # TRAIN LOOP
    ################################
    for epoch in range(start_epoch, total_epochs + 1):

        train_sampler.set_epoch(epoch)
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

        ################################
        # VALIDATION (ONLY RANK 0)
        ################################
        if rank == 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    val_loss += (l1_loss(sr, hr) + 0.1 * (1 - ssim(sr, hr))).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            loss_log.append([epoch, train_loss, val_loss])

            np.savetxt("wind_uv_mdsr_loss.csv", loss_log,
                       delimiter=",", header="epoch,train,val", comments="")

            if epoch % 500 == 0:
                torch.save({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, f"wind_uv_ckpt_mdsr/mdsr_epoch_{epoch}.pth")

    dist.destroy_process_group()


############################################
if __name__ == "__main__":
    main()