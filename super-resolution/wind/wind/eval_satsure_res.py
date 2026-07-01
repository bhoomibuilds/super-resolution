import os
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


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
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2)
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
# METRICS
############################################
def compute_metrics(pred, gt):
    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        data_range = 1.0

    psnr_val = psnr(gt, pred, data_range=data_range)
    ssim_val = ssim(gt, pred, data_range=data_range)
    rmse_val = np.sqrt(np.mean((pred - gt) ** 2))

    if np.std(pred) > 0 and np.std(gt) > 0:
        corr_val = np.corrcoef(pred.flatten(), gt.flatten())[0, 1]
    else:
        corr_val = 0.0

    return psnr_val, ssim_val, rmse_val, corr_val


############################################
# WIND SPEED
############################################
def wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


############################################
# NORMALIZATION (MATCH TRAINING)
############################################
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


############################################
# PATHS (EDIT HERE)
############################################
CKPT = "wind_uv_ckpt/satsure_epoch_6000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_DIR = "satsure_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_TIME = "2020-06-06T16:00:00"


############################################
# LOAD MODEL
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SatSuRE().to(device)

ckpt = torch.load(CKPT, map_location=device)

# 🔥 HANDLE DDP CHECKPOINT
if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

# remove "module." if exists
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

print("✅ Loaded checkpoint:", CKPT)


############################################
# LOAD DATA
############################################
lr_u = xr.open_dataset(LR_U)["u10"].sel(time=TARGET_TIME).values
lr_v = xr.open_dataset(LR_V)["v10"].sel(time=TARGET_TIME).values

hr_u = xr.open_dataset(HR_U)["u10"].sel(time=TARGET_TIME).values
hr_v = xr.open_dataset(HR_V)["v10"].sel(time=TARGET_TIME).values


############################################
# PREPARE INPUT
############################################
lr = normalize(np.stack([lr_u, lr_v]))
hr = normalize(np.stack([hr_u, hr_v]))

lr_tensor = torch.tensor(lr).float().unsqueeze(0).to(device)


############################################
# INFERENCE
############################################
with torch.no_grad():
    sr = model(lr_tensor).squeeze().cpu().numpy()

sr_u, sr_v = sr


############################################
# ERROR
############################################
error_u = np.abs(sr_u - hr[0])
error_v = np.abs(sr_v - hr[1])


############################################
# METRICS
############################################
print("\n===== U COMPONENT =====")
print("PSNR, SSIM, RMSE, CORR:", compute_metrics(sr_u, hr[0]))

print("\n===== V COMPONENT =====")
print("PSNR, SSIM, RMSE, CORR:", compute_metrics(sr_v, hr[1]))

speed_sr = wind_speed(sr_u, sr_v)
speed_hr = wind_speed(hr[0], hr[1])

print("\n===== WIND SPEED =====")
print("PSNR, SSIM, RMSE, CORR:", compute_metrics(speed_sr, speed_hr))


############################################
# VISUALIZATION
############################################
LAT_MIN, LAT_MAX = 5, 15
LON_MIN, LON_MAX = 80, 90

vmin, vmax = 0, 1
err_min, err_max = 0, 0.5

plt.figure(figsize=(18,10))

# ---- U ----
titles = ["LR", "SR", "HR", "Error"]

data_u = [lr[0], sr_u, hr[0], error_u]
data_v = [lr[1], sr_v, hr[1], error_v]

for i in range(4):
    plt.subplot(2,4,i+1)
    plt.title(f"U {titles[i]}")
    plt.imshow(data_u[i],
               cmap="jet" if i<3 else "gray_r",
               vmin=vmin if i<3 else err_min,
               vmax=vmax if i<3 else err_max,
               interpolation="bilinear",
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    plt.colorbar()

for i in range(4):
    plt.subplot(2,4,i+5)
    plt.title(f"V {titles[i]}")
    plt.imshow(data_v[i],
               cmap="jet" if i<3 else "gray_r",
               vmin=vmin if i<3 else err_min,
               vmax=vmax if i<3 else err_max,
               interpolation="bilinear",
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    plt.colorbar()

plt.tight_layout()

out_file = f"{SAVE_DIR}/satsure_eval_{TARGET_TIME}_3000.png"
plt.savefig(out_file, dpi=300)
plt.close()

print("\n✅ Saved:", out_file)
