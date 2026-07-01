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
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


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
# PATHS
############################################
CKPT = "srcnn_ckpt/ckpt_6000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_DIR = "srcnn_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_TIME = "2020-06-06T16:00:00"


############################################
# LOAD MODEL
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT, map_location=device)

model = SRCNN().to(device)
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
# NORMALIZATION (FIXED)
############################################
def normalize(x):
    return (x - mean) / (std + 1e-8)

# ✅ ONLY normalize LR
lr = normalize(np.stack([lr_u, lr_v]))

# ❌ DO NOT normalize HR
hr = np.stack([hr_u, hr_v])


############################################
# INFERENCE
############################################
lr_tensor = torch.tensor(lr).float().unsqueeze(0).to(device)

# 🔥 REQUIRED for SRCNN
lr_tensor = F.interpolate(lr_tensor, size=hr.shape[-2:], mode='bilinear', align_corners=False)

with torch.no_grad():
    sr = model(lr_tensor).squeeze().cpu().numpy()

# ✅ denormalize ONLY SR
sr = sr * std + mean

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
print(compute_metrics(sr_u, hr[0]))

print("\n===== V COMPONENT =====")
print(compute_metrics(sr_v, hr[1]))

speed_sr = wind_speed(sr_u, sr_v)
speed_hr = wind_speed(hr[0], hr[1])

print("\n===== WIND SPEED =====")
print(compute_metrics(speed_sr, speed_hr))


############################################
# VISUALIZATION (FIXED SCALE)
############################################
LAT_MIN, LAT_MAX = 5, 15
LON_MIN, LON_MAX = 80, 90

# ✅ dynamic range (IMPORTANT)
vmin = min(hr.min(), sr.min())
vmax = max(hr.max(), sr.max())

error_vmin, error_vmax = 0, np.max([error_u.max(), error_v.max()])


plt.figure(figsize=(18,10))

# ---- U ----
plt.subplot(2,4,1)
plt.title("U LR")
plt.imshow(lr[0], cmap="jet",interpolation="bilinear",
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
plt.imshow(error_u, cmap="gray_r", vmin=error_vmin, vmax=error_vmax,
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

# ---- V ----
plt.subplot(2,4,5)
plt.title("V LR")
plt.imshow(lr[1], cmap="jet",interpolation="bilinear",
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
plt.imshow(error_v, cmap="gray_r", vmin=error_vmin, vmax=error_vmax,
           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
plt.colorbar()

plt.tight_layout()

outfile = f"{SAVE_DIR}/srcnn_eval_{TARGET_TIME}_6000.png"
plt.savefig(outfile, dpi=300)
plt.close()

print("\nSaved:", outfile)