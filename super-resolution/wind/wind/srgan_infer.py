import os
import torch
import torch.nn as nn
import numpy as np
import xarray as xr

############################################
# MODEL (SAME AS TRAINING)
############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.prelu(self.conv1(x)))


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ps(self.conv(x)))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64)
        )
        self.conv3 = nn.Conv2d(64, 2, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.conv2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.conv3(x)

############################################
# CONFIG
############################################
CKPT_PATH = "sr_l1_ckpt/best_model.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"   # optional
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_PATH = "srgan_wind_sr.nc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# LOAD MODEL
############################################
def load_model():
    model = Generator().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["generator"])

    mean = ckpt["mean"]
    std = ckpt["std"]

    model.eval()
    return model, mean, std

############################################
# LOAD DATA
############################################
def load_data():
    lr_u = xr.open_dataset(LR_U)["u10"].values
    lr_v = xr.open_dataset(LR_V)["v10"].values

    hr_u = xr.open_dataset(HR_U)["u10"].values if os.path.exists(HR_U) else None
    hr_v = xr.open_dataset(HR_V)["v10"].values if os.path.exists(HR_V) else None

    return lr_u, lr_v, hr_u, hr_v

############################################
# NORMALIZATION
############################################
def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def denormalize(x, mean, std):
    return x * std + mean

############################################
# INFERENCE
############################################
def run_inference():

    model, mean, std = load_model()
    lr_u, lr_v, hr_u, hr_v = load_data()

    sr_u_list, sr_v_list = [], []
    rmse_list = []

    for i in range(lr_u.shape[0]):

        ################################
        # PREPARE INPUT
        ################################
        lr = np.stack([lr_u[i], lr_v[i]])
        lr = normalize(lr, mean, std)

        lr = torch.tensor(lr).unsqueeze(0).float().to(DEVICE)

        ################################
        # MODEL
        ################################
        with torch.no_grad():
            sr = model(lr)

        sr = sr.squeeze(0).cpu().numpy()

        ################################
        # DENORMALIZE
        ################################
        sr = denormalize(sr, mean, std)

        sr_u, sr_v = sr[0], sr[1]

        sr_u_list.append(sr_u)
        sr_v_list.append(sr_v)

        ################################
        # METRICS
        ################################
        if hr_u is not None:
            hr = np.stack([hr_u[i], hr_v[i]])
            rmse = np.sqrt(np.mean((sr - hr) ** 2))
            rmse_list.append(rmse)

    ################################
    # SAVE NETCDF
    ################################
    sr_u_arr = np.array(sr_u_list)
    sr_v_arr = np.array(sr_v_list)

    ds = xr.Dataset({
        "u10": (["time", "lat", "lon"], sr_u_arr),
        "v10": (["time", "lat", "lon"], sr_v_arr),
    })

    ds.to_netcdf(SAVE_PATH)

    print(f"✅ Saved SR → {SAVE_PATH}")

    ################################
    # PRINT METRICS
    ################################
    if rmse_list:
        print("\n===== RESULTS =====")
        print(f"RMSE : {np.mean(rmse_list):.4f}")

############################################
if __name__ == "__main__":
    run_inference()
