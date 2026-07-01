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
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class SatSuRE(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_res_blocks=16, scale_factor=4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        upsample_layers = []
        for _ in range(int(scale_factor / 2)):
            upsample_layers.append(UpsampleBlock(64, 2))

        self.upsample = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

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
CKPT_PATH = "wind_uv_ckpt/satsure_epoch_6000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_PATH = "satsure_wind_sr.nc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# NORMALIZATION (SAME AS TRAINING)
############################################
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

############################################
# LOAD MODEL
############################################
def load_model():
    model = SatSuRE().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    model.eval()
    return model

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
# INFERENCE
############################################
def run_inference():

    model = load_model()
    lr_u, lr_v, hr_u, hr_v = load_data()

    sr_u_list, sr_v_list = [], []
    rmse_list = []

    for i in range(lr_u.shape[0]):

        ################################
        # PREP INPUT
        ################################
        lr = np.stack([lr_u[i], lr_v[i]])
        lr_norm = normalize(lr)

        lr_tensor = torch.tensor(lr_norm).unsqueeze(0).float().to(DEVICE)

        ################################
        # MODEL
        ################################
        with torch.no_grad():
            sr = model(lr_tensor)

        sr = sr.squeeze(0).cpu().numpy()

        ################################
        # IMPORTANT: RESCALE BACK
        ################################
        # since min-max used per sample
        lr_min = lr.min()
        lr_max = lr.max()

        sr = sr * (lr_max - lr_min) + lr_min

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
    # SAVE OUTPUT
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
    # RESULTS
    ################################
    if rmse_list:
        print("\n===== RESULTS =====")
        print(f"RMSE : {np.mean(rmse_list):.4f}")

############################################
if __name__ == "__main__":
    run_inference()
