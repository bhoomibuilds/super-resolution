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
class RRDB(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(64) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.body(x1))
        feat = x1 + x2
        feat = self.lrelu(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.lrelu(self.up1(feat))
        feat = self.lrelu(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.lrelu(self.up2(feat))
        return self.conv3(feat)

############################################
# 🔥 FIXED CHECKPOINT LOADER
############################################
def load_generator_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    print("Checkpoint keys:", ckpt.keys())

    # Case 1: Full GAN checkpoint
    if "G" in ckpt:
        print("✅ Loading Generator weights from 'G'")
        model.load_state_dict(ckpt["G"], strict=True)

    # Case 2: DataParallel saved weights
    elif "state_dict" in ckpt:
        print("✅ Loading from 'state_dict'")
        state_dict = ckpt["state_dict"]

        # remove "module." if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict, strict=True)

    # Case 3: Direct weights
    else:
        print("✅ Loading direct state_dict")
        model.load_state_dict(ckpt, strict=True)

############################################
# UTILS
############################################
def compute_metrics(pred, gt):
    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        data_range = 1.0

    return (
        psnr(gt, pred, data_range=data_range),
        ssim(gt, pred, data_range=data_range),
        np.sqrt(np.mean((pred - gt) ** 2)),
        np.corrcoef(pred.flatten(), gt.flatten())[0, 1],
    )


def ensure_2d(x):
    x = np.squeeze(x)
    if x.ndim == 3:
        x = x[0]
    return x

############################################
# PATHS
############################################
CKPTS = [5000, 5500, 6000]

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"
HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_DIR = "realesrgan_results_multi"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_TIME = "2020-06-06"

############################################
# LOAD DATA
############################################
lr_u = ensure_2d(xr.open_dataset(LR_U)["u10"].sel(time=TARGET_TIME).values)
lr_v = ensure_2d(xr.open_dataset(LR_V)["v10"].sel(time=TARGET_TIME).values)

hr_u = ensure_2d(xr.open_dataset(HR_U)["u10"].sel(time=TARGET_TIME).values)
hr_v = ensure_2d(xr.open_dataset(HR_V)["v10"].sel(time=TARGET_TIME).values)

lr_stack = np.stack([lr_u, lr_v])
hr_stack = np.stack([hr_u, hr_v])

############################################
# ORIENTATION FIX
############################################
lr_stack = np.flip(lr_stack, axis=1)
hr_stack = np.flip(hr_stack, axis=1)

############################################
# NORMALIZATION
############################################
all_data = np.concatenate([lr_stack.flatten(), hr_stack.flatten()])
mean, std = all_data.mean(), all_data.std()

def normalize(x): return (x - mean) / (std + 1e-8)
def denormalize(x): return x * std + mean

lr_norm = normalize(lr_stack)

############################################
# DEVICE
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# LOOP
############################################
for epoch in CKPTS:

    print(f"\n🔥 Evaluating Epoch {epoch}")

    ckpt_path = f"wind_realesrgan_ckpt/epoch_{epoch}.pth"

    model = Generator().to(device)

    # ✅ FIXED LOAD
    load_generator_weights(model, ckpt_path, device)

    model.eval()

    lr_tensor = torch.tensor(lr_norm).float().unsqueeze(0).to(device)

    with torch.no_grad():
        sr_norm = model(lr_tensor).squeeze().cpu().numpy()

    sr = denormalize(sr_norm)
    sr_u, sr_v = sr

    ############################################
    # ERROR
    ############################################
    error_u = np.abs(sr_u - hr_stack[0])
    error_v = np.abs(sr_v - hr_stack[1])

    ############################################
    # METRICS
    ############################################
    print("U:", compute_metrics(sr_u, hr_stack[0]))
    print("V:", compute_metrics(sr_v, hr_stack[1]))

    ############################################
    # VISUALIZATION
    ############################################
    LAT_MIN, LAT_MAX = 5, 15
    LON_MIN, LON_MAX = 80, 90

    vmin = min(hr_stack.min(), sr.min())
    vmax = max(hr_stack.max(), sr.max())

    plt.figure(figsize=(18, 10))

    plots = [
        (lr_stack[0], "U LR"),
        (sr_u, "U SR"),
        (hr_stack[0], "U HR"),
        (error_u, "U Error"),
        (lr_stack[1], "V LR"),
        (sr_v, "V SR"),
        (hr_stack[1], "V HR"),
        (error_v, "V Error"),
    ]

    for i, (data, title) in enumerate(plots):
        plt.subplot(2, 4, i + 1)

        cmap = "gray_r" if "Error" in title else "jet"

        plt.imshow(
            data,
            cmap=cmap,
            interpolation="bilinear",
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower",
            vmin=vmin if "Error" not in title else None,
            vmax=vmax if "Error" not in title else None
        )

        plt.title(title)
        plt.colorbar()

    plt.tight_layout()

    outfile = f"{SAVE_DIR}/eval_epoch_{epoch}.png"
    plt.savefig(outfile, dpi=300)
    plt.close()

    print("✅ Saved:", outfile)