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
# MODEL (SinSR-style)
############################################
class SinSR_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, x, noise):
        x = x + noise
        return self.net(x)


############################################
# DATASET
############################################
class WindDataset(Dataset):
    def __init__(self, lr_u_nc, lr_v_nc, hr_u_nc, hr_v_nc, mean=None, std=None):

        self.lr_u = xr.open_dataset(lr_u_nc)["u10"].values
        self.lr_v = xr.open_dataset(lr_v_nc)["v10"].values
        self.hr_u = xr.open_dataset(hr_u_nc)["u10"].values
        self.hr_v = xr.open_dataset(hr_v_nc)["v10"].values

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
# DDP
############################################
def setup():
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


############################################
# VALIDATION (LOSS + RMSE)
############################################
def validate(model, val_loader, device, mean, std):

    model.eval()
    mse = nn.MSELoss()

    total_loss = 0
    total_rmse = 0

    with torch.no_grad():
        for lr, hr in val_loader:

            lr, hr = lr.to(device), hr.to(device)

            lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear')

            noise = torch.randn_like(lr) * 0.1
            sr = model(lr, noise)

            loss = mse(sr, hr)
            total_loss += loss.item()

            # denormalized RMSE
            sr_dn = sr * std + mean
            hr_dn = hr * std + mean

            rmse = torch.sqrt(torch.mean((sr_dn - hr_dn) ** 2))
            total_rmse += rmse.item()

    return total_loss / len(val_loader), total_rmse / len(val_loader)


############################################
# TRAIN
############################################
def main():

    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    ############################################
    # DATA
    ############################################
    train_dataset = WindDataset(
        "data_wind/wind_u/train/LR/2015.nc",
        "data_wind/wind_v/train/LR/2015.nc",
        "data_wind/wind_u/train/HR/2015.nc",
        "data_wind/wind_v/train/HR/2015.nc"
    )

    mean, std = train_dataset.mean, train_dataset.std

    val_dataset = WindDataset(
        "data_wind/wind_u/val/LR/2020.nc",
        "data_wind/wind_v/val/LR/2020.nc",
        "data_wind/wind_u/val/HR/2020.nc",
        "data_wind/wind_v/val/HR/2020.nc",
        mean, std
    )

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=train_sampler, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False)

    ############################################
    # MODEL
    ############################################
    model = SinSR_Model().to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    ############################################
    # SETTINGS
    ############################################
    epochs = 6000
    best_rmse = float("inf")

    if rank == 0:
        os.makedirs("sinsr_ckpt_wind", exist_ok=True)

    # 🔥 STORE HISTORY
    train_losses = []
    val_losses = []
    val_rmses = []

    ############################################
    # TRAIN LOOP
    ############################################
    for epoch in range(1, epochs + 1):

        train_sampler.set_epoch(epoch)
        model.train()

        total_loss = 0

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear')

            noise = torch.randn_like(lr) * 0.1

            sr = model(lr, noise)

            loss = mse(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        ############################################
        # VALIDATION + LOGGING
        ############################################
        if rank == 0:

            train_loss = total_loss / len(train_loader)

            val_loss, val_rmse = validate(
                model.module, val_loader, device, mean, std
            )

            # 🔥 SAVE HISTORY
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_rmses.append(val_rmse)

            print(f"Epoch {epoch} | Train:{train_loss:.6f} "
                  f"| Val:{val_loss:.6f} | RMSE:{val_rmse:.6f}")

            ############################################
            # BEST MODEL
            ############################################
            if val_rmse < best_rmse:
                best_rmse = val_rmse

                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "mean": mean,
                    "std": std,
                    "best_rmse": best_rmse,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_rmses": val_rmses
                }, "sinsr_ckpt_wind/best_model.pth")

                print("✅ Saved BEST model")

            ############################################
            # CHECKPOINT EVERY 500
            ############################################
            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "mean": mean,
                    "std": std,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_rmses": val_rmses
                }, f"sinsr_ckpt_wind/ckpt_{epoch}.pth")

                print(f"💾 Saved checkpoint at epoch {epoch}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()