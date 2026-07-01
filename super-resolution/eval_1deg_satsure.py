import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

############################################
# CONFIG
############################################

BASE = "D:\SR\SR\OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_PATH = "sst_ckpt_satsure/satsure_epoch_6000.pth"

TARGET_DATE = "2017-01-01"

OUT_DIR = "satsure_sst"

VMIN = 25.0
VMAX = 35.0

############################################
# SATSURE MODEL
############################################

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SatSuRE(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv3 = nn.Conv2d(64, 1, 9, padding=4)

    def forward(self, x):

        x1 = self.conv1(x)

        x = self.residual_blocks(x1)

        x = self.conv2(x) + x1

        x = self.upsample(x)

        x = self.conv3(x)

        return x


############################################
# SSIM
############################################

def gaussian(window_size, sigma):

    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2*sigma**2))
        for x in range(window_size)
    ])

    return gauss / gauss.sum()


def create_window(window_size, channel):

    _1D = gaussian(window_size, 1.5).unsqueeze(1)

    _2D = _1D @ _1D.t()

    window = _2D.expand(
        channel,
        1,
        window_size,
        window_size
    ).contiguous()

    return window


def ssim(img1, img2, window_size=11):

    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(
        img1,
        window,
        padding=window_size//2,
        groups=channel
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=window_size//2,
        groups=channel
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=window_size//2,
            groups=channel
        ) - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=window_size//2,
            groups=channel
        ) - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=window_size//2,
            groups=channel
        ) - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = (
        (2 * mu1_mu2 + C1) *
        (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) *
        (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


############################################
# METRICS
############################################

def compute_rmse(pred, target):

    return torch.sqrt(
        torch.mean((pred - target) ** 2)
    ).item()


def compute_psnr(pred, target):

    mse = torch.mean((pred - target) ** 2)

    return 20 * torch.log10(
        35.0 / torch.sqrt(mse + 1e-8)
    ).item()


def compute_correlation(pred, target):

    pred = pred.flatten()
    target = target.flatten()

    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    num = torch.sum(
        (pred - pred_mean) *
        (target - target_mean)
    )

    den = torch.sqrt(
        torch.sum((pred - pred_mean)**2) *
        torch.sum((target - target_mean)**2)
    )

    return (num / (den + 1e-8)).item()


############################################
# HELPERS
############################################

def find_file(folder, date):

    for f in os.listdir(folder):

        if date in f:
            return os.path.join(folder, f)

    raise FileNotFoundError(date)


def extract(data):

    if isinstance(data, dict):

        for v in data.values():

            if isinstance(v, np.ndarray):
                return v

    if isinstance(data, np.ndarray) and data.dtype == object:
        return extract(data.item())

    return data


############################################
# VISUALIZATION
############################################

def visualize(
    lr,
    bicubic,
    sr,
    hr,
    save_path
):

    error_bicubic = hr - bicubic
    error_sr = hr - sr

    vmin, vmax = 25.3, 28.15

    err_min, err_max = -0.5, 0.5

    lat_min, lat_max = 5, 20
    lon_min, lon_max = 60, 72

    extent = [
        lon_min,
        lon_max,
        lat_min,
        lat_max
    ]

    plt.figure(figsize=(15, 10))

    ##################################
    # LR
    ##################################

    plt.subplot(2, 3, 1)

    plt.title("LR-OISST (1°)")

    plt.imshow(
        lr,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        interpolation="bilinear"
    )

    plt.colorbar()

    ##################################
    # Bicubic
    ##################################

    plt.subplot(2, 3, 2)

    plt.title("Bicubic")

    plt.imshow(
        bicubic,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        interpolation="bilinear"
    )

    plt.colorbar()

    ##################################
    # SatSuRE
    ##################################

    plt.subplot(2, 3, 3)

    plt.title("SatSuRE")

    plt.imshow(
        sr,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        interpolation="bilinear"
    )

    plt.colorbar()

    ##################################
    # HR
    ##################################

    plt.subplot(2, 3, 4)

    plt.title("HR-OISST (0.25°)")

    plt.imshow(
        hr,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        interpolation="bilinear"
    )

    plt.colorbar()

    ##################################
    # Bicubic Error
    ##################################

    plt.subplot(2, 3, 5)

    plt.title("Error (HR - Bicubic)")

    plt.imshow(
        error_bicubic,
        cmap='bwr',
        vmin=err_min,
        vmax=err_max,
        extent=extent,
        origin="lower"
    )

    plt.colorbar()

    ##################################
    # SatSuRE Error
    ##################################

    plt.subplot(2, 3, 6)

    plt.title("Error (HR - SatSuRE)")

    plt.imshow(
        error_sr,
        cmap='bwr',
        vmin=err_min,
        vmax=err_max,
        extent=extent,
        origin="lower"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


############################################
# MAIN
############################################

def evaluate():

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ##################################
    # LOAD MODEL
    ##################################

    model = SatSuRE().to(device)

    ckpt = torch.load(
        CKPT_PATH,
        map_location=device
    )

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    ##################################
    # LOAD DATA
    ##################################

    lr = extract(
        np.load(
            find_file(VAL_LR, TARGET_DATE),
            allow_pickle=True
        )
    )

    hr = extract(
        np.load(
            find_file(VAL_HR, TARGET_DATE),
            allow_pickle=True
        )
    )

    lr = np.squeeze(lr).astype(np.float32)

    hr = np.squeeze(hr).astype(np.float32)

    ##################################
    # NORMALIZE
    ##################################

    lr_n = (lr - VMIN) / (VMAX - VMIN)

    hr_n = (hr - VMIN) / (VMAX - VMIN)

    lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)

    hr_t = torch.tensor(hr_n).unsqueeze(0).unsqueeze(0).to(device)

    ##################################
    # SATSURE INFERENCE
    ##################################

    with torch.no_grad():

        sr = model(lr_t)

    ##################################
    # BICUBIC BASELINE
    # HR -> LR -> HR
    ##################################

    with torch.no_grad():

        # Downsample HR to LR resolution
        bicubic_lr = F.interpolate(
            hr_t,
            size=lr.shape,
            mode="bicubic",
            align_corners=False
        )

        # Upsample LR back to HR
        bicubic = F.interpolate(
            bicubic_lr,
            size=hr.shape,
            mode="bicubic",
            align_corners=False
        )

    ##################################
    # DENORMALIZE
    ##################################

    sr_c = sr * (VMAX - VMIN) + VMIN

    bicubic_c = bicubic * (VMAX - VMIN) + VMIN

    hr_c = hr_t * (VMAX - VMIN) + VMIN

    ##################################
    # METRICS
    ##################################

    rmse = compute_rmse(sr_c, hr_c)

    psnr = compute_psnr(sr_c, hr_c)

    ssim_val = ssim(sr, hr_t).item()

    corr = compute_correlation(sr_c, hr_c)

    ##################################
    # BICUBIC METRICS
    ##################################

    bicubic_rmse = compute_rmse(
        bicubic_c,
        hr_c
    )

    bicubic_psnr = compute_psnr(
        bicubic_c,
        hr_c
    )

    bicubic_ssim = ssim(
        bicubic,
        hr_t
    ).item()

    bicubic_corr = compute_correlation(
        bicubic_c,
        hr_c
    )

    ##################################
    # PRINT RESULTS
    ##################################

    print(f"\n===== SATSURE RESULTS ({TARGET_DATE}) =====")

    print("\n----- Bicubic -----")

    print(f"RMSE : {bicubic_rmse:.6f}")
    print(f"PSNR : {bicubic_psnr:.6f}")
    print(f"SSIM : {bicubic_ssim:.6f}")
    print(f"CORR : {bicubic_corr:.6f}")

    print("\n----- SatSuRE -----")

    print(f"RMSE : {rmse:.6f}")
    print(f"PSNR : {psnr:.6f}")
    print(f"SSIM : {ssim_val:.6f}")
    print(f"CORR : {corr:.6f}")

    ##################################
    # VISUALIZE
    ##################################

    save_path = (
        f"{OUT_DIR}/"
        f"{TARGET_DATE}_satsure_6000.png"
    )

    visualize(
        lr,
        bicubic_c.squeeze().cpu().numpy(),
        sr_c.squeeze().cpu().numpy(),
        hr,
        save_path
    )

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":

    evaluate()