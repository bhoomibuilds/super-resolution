import os
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


############################################
# GENERATOR (MATCH TRAINING)
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

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

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
# CONFIG
############################################
CKPT = "srgan_ckpt/checkpoint_4000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_DIR = "srgan_eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_TIME = "2020-06-06T16:00:00"


############################################
# LOAD MODEL
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Generator().to(device)
checkpoint = torch.load(CKPT, map_location=device)
model.load_state_dict(checkpoint["generator"])
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
# NORMALIZATION (MATCH TRAINING)
############################################
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

lr = normalize(np.stack([lr_u, lr_v]))
hr = normalize(np.stack([hr_u, hr_v]))


############################################
# INFERENCE
############################################
lr_tensor = torch.tensor(lr).float().unsqueeze(0).to(device)

with torch.no_grad():
    sr = model(lr_tensor).squeeze().cpu().numpy()

sr_u, sr_v = sr


############################################
# ERROR MAPS
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
# VISUALIZATION (WITH ERROR MAPS)
############################################
LAT_MIN, LAT_MAX = 5, 15
LON_MIN, LON_MAX = 80, 90

vmin, vmax = 0, 1
err_vmin, err_vmax = 0.5, 0.0

plt.figure(figsize=(20,10))

# ===== U =====
plt.subplot(2,4,1)
plt.title("U - LR")
plt.imshow(lr[0], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,2)
plt.title("U - SR")
plt.imshow(sr_u, cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,3)
plt.title("U - HR")
plt.imshow(hr[0], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,4)
plt.title("U Error |SR-HR|")
plt.imshow(error_u, cmap="gray_r", vmin=err_vmin, vmax=err_vmax,
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

# ===== V =====
plt.subplot(2,4,5)
plt.title("V - LR")
plt.imshow(lr[1], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,6)
plt.title("V - SR")
plt.imshow(sr_v, cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,7)
plt.title("V - HR")
plt.imshow(hr[1], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,4,8)
plt.title("V Error |SR-HR|")
plt.imshow(error_v, cmap="gray_r", vmin=err_vmin, vmax=err_vmax,
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()

outfile = f"{SAVE_DIR}/srgan_eval_final_{TARGET_TIME}_4000.png"
plt.savefig(outfile, dpi=300)
plt.close()

print("\n✅ Saved visualization:", outfile)
