import os
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


############################################
# MODEL
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
        return self.net(x + noise)


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

    corr_val = np.corrcoef(pred.flatten(), gt.flatten())[0, 1]

    return psnr_val, ssim_val, rmse_val, corr_val


############################################
# WIND SPEED
############################################
def wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


############################################
# PATHS
############################################
CKPT = "sinsr_ckpt_wind/ckpt_6000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_DIR = "sinsr_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_TIME = "2020-06-06T16:00:00"


############################################
# LOAD MODEL
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT, map_location=device)

model = SinSR_Model().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

mean = ckpt["mean"]
std = ckpt["std"]

print("Loaded checkpoint:", CKPT)


############################################
# LOAD DATA
############################################
lr_u = xr.open_dataset(LR_U)["u10"].sel(time=TARGET_TIME).values
lr_v = xr.open_dataset(LR_V)["v10"].sel(time=TARGET_TIME).values

hr_u = xr.open_dataset(HR_U)["u10"].sel(time=TARGET_TIME).values
hr_v = xr.open_dataset(HR_V)["v10"].sel(time=TARGET_TIME).values


############################################
# NORMALIZATION
############################################
def normalize(x):
    return (x - mean) / (std + 1e-8)

def denormalize(x):
    return x * std + mean


lr_stack = np.stack([lr_u, lr_v])
hr_stack = np.stack([hr_u, hr_v])

lr_norm = normalize(lr_stack)
hr_norm = normalize(hr_stack)


############################################
# INFERENCE
############################################
lr_tensor = torch.tensor(lr_norm).float().unsqueeze(0).to(device)

lr_tensor = F.interpolate(lr_tensor, size=hr_norm.shape[-2:], mode='bilinear')

# ✅ FIX: smaller noise
noise = torch.randn_like(lr_tensor) * 0.01

with torch.no_grad():
    sr_norm = model(lr_tensor, noise).squeeze().cpu().numpy()

# denormalize properly
sr = denormalize(sr_norm)
hr = hr_stack  # already original scale
lr = lr_stack  # original scale

sr_u, sr_v = sr


############################################
# DEBUG (IMPORTANT)
############################################
print("\n==== DEBUG ====")
print("SR min/max:", sr.min(), sr.max())
print("HR min/max:", hr.min(), hr.max())


############################################
# ERROR
############################################
error_u = np.abs(sr_u - hr[0])
error_v = np.abs(sr_v - hr[1])


############################################
# METRICS
############################################
print("\n===== U COMPONENT =====")
print(compute_metrics(sr_u, hr[0]))

print("\n===== V COMPONENT =====")
print(compute_metrics(sr_v, hr[1]))

speed_sr = wind_speed(sr_u, sr_v)
speed_hr = wind_speed(hr[0], hr[1])

print("\n===== WIND SPEED =====")
print(compute_metrics(speed_sr, speed_hr))


############################################
# VISUALIZATION (FIXED)
############################################
LAT_MIN, LAT_MAX = 5, 15
LON_MIN, LON_MAX = 80, 90

# ✅ dynamic scaling (CRITICAL FIX)
vmin = min(hr.min(), sr.min())
vmax = max(hr.max(), sr.max())

plt.figure(figsize=(18,10))

# ---- U ----
plt.subplot(2,4,1)
plt.title("U LR")
plt.imshow(lr[0], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,2)
plt.title("U SR ")
plt.imshow(sr_u, cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,3)
plt.title("U HR")
plt.imshow(hr[0], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,4)
plt.title("U Error")
plt.imshow(error_u, cmap="gray_r", vmin=0, vmax=np.max(error_u),
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

# ---- V ----
plt.subplot(2,4,5)
plt.title("V LR")
plt.imshow(lr[1], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,6)
plt.title("V SR ")
plt.imshow(sr_v, cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,7)
plt.title("V HR")
plt.imshow(hr[1], cmap="jet", vmin=vmin, vmax=vmax,interpolation="bilinear",
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.subplot(2,4,8)
plt.title("V Error")
plt.imshow(error_v, cmap="gray_r", vmin=0, vmax=np.max(error_v),
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.tight_layout()

outfile = f"{SAVE_DIR}/sinsr_eval_6000.png"
plt.savefig(outfile, dpi=300)
plt.close()

print("\nSaved:", outfile)