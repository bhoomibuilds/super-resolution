import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import xarray as xr

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


############################################
# GENERATOR
############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.prelu(res)
        res = self.conv2(res)
        return x + res


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ps(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.upsample = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64)
        )

        self.conv3 = nn.Conv2d(64, out_channels, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.conv2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.conv3(x)


############################################
# DISCRIMINATOR
############################################
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=False)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),

            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        return self.model(x).view(x.size(0))


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

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    ############################################
    # MODELS
    ############################################
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    ############################################
    # LOSS + OPTIMIZER
    ############################################
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    ############################################
    # LOSS HISTORY
    ############################################
    losses_G = []
    losses_D = []

    ############################################
    # RESUME
    ############################################
    start_epoch = 3001
    ckpt_path = "srgan_ckpt/checkpoint_3000.pth"

    if os.path.exists(ckpt_path):
        if rank == 0:
            print(f"Loading checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)

        generator.module.load_state_dict(checkpoint["generator"])
        discriminator.module.load_state_dict(checkpoint["discriminator"])

        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        losses_G = checkpoint.get("losses_G", [])
        losses_D = checkpoint.get("losses_D", [])

        start_epoch = checkpoint["epoch"] + 1

        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")

    ############################################
    # TRAIN LOOP
    ############################################
    epochs = 6000

    if rank == 0:
        os.makedirs("srgan_ckpt", exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):

        train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()

        for lr, hr in train_loader:

            lr, hr = lr.to(device), hr.to(device)

            ################################
            # GENERATOR
            ################################
            optimizer_G.zero_grad()

            sr = generator(lr)
            pred_fake = discriminator(sr)

            loss_content = content_loss(sr, hr)
            loss_adv = adversarial_loss(pred_fake, torch.ones_like(pred_fake))

            loss_G = loss_content + 1e-3 * loss_adv

            loss_G.backward()
            optimizer_G.step()

            ################################
            # DISCRIMINATOR
            ################################
            optimizer_D.zero_grad()

            pred_real = discriminator(hr)
            pred_fake = discriminator(sr.detach())

            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

        ############################################
        # LOG + SAVE
        ############################################
        if rank == 0:
            losses_G.append(loss_G.item())
            losses_D.append(loss_D.item())

            print(f"Epoch {epoch} | G: {loss_G.item():.6f} D: {loss_D.item():.6f}")

            if epoch % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "generator": generator.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "losses_G": losses_G,
                    "losses_D": losses_D
                }, f"srgan_ckpt/checkpoint_{epoch}.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()