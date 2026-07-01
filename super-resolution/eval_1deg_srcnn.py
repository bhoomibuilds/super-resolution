import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_PATH = "sst_ckpt_srcnn/srcnn_epoch_6000.pth"

TARGET_DATE = "2017-01-01"

OUT_DIR = "srcnn_sst"

VMIN = 25.0
VMAX = 35.0

############################################
# SRCNN MODEL
############################################

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

############################################
# SSIM (same)
############################################

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2/(2*sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D @ _1D.t()
    return _2D.expand(channel,1,window_size,window_size).contiguous()


def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    return (((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
           ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()

############################################
# METRICS (same)
############################################

def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(35.0 / torch.sqrt(mse + 1e-8)).item()


def compute_correlation(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    num = torch.sum((pred - pred_mean) * (target - target_mean))
    den = torch.sqrt(torch.sum((pred - pred_mean)**2) *
                     torch.sum((target - target_mean)**2))

    return (num / (den + 1e-8)).item()

############################################
# HELPERS (same)
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
# VISUALIZATION (updated title)
############################################

def visualize(lr, sr, hr, save_path):

    error = sr - hr

    vmin, vmax = 25.3, 28.15
    err_min, err_max = -0.5, 0.5

    lat_min, lat_max = 5, 20
    lon_min, lon_max = 60, 72

    extent = [lon_min, lon_max, lat_min, lat_max]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("LR-OISST (1°)")
    plt.imshow(lr, cmap='jet', vmin=vmin, vmax=vmax,
               extent=extent, origin="lower", interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("SRCNN")  # ✅ changed
    plt.imshow(sr, cmap='jet', vmin=vmin, vmax=vmax,
               extent=extent, origin="lower", interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("HR-OISST (0.25°)")
    plt.imshow(hr, cmap='jet', vmin=vmin, vmax=vmax,
               extent=extent, origin="lower", interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Error (HR - SR)")
    plt.imshow(error, cmap='bwr',
               vmin=err_min, vmax=err_max,
               extent=extent, origin="lower")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

############################################
# MAIN
############################################

def evaluate():

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN().to(device)  # ✅ replaced

    ckpt = torch.load(CKPT_PATH, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    ##################################
    # LOAD DATA
    ##################################
    lr = extract(np.load(find_file(VAL_LR, TARGET_DATE), allow_pickle=True))
    hr = extract(np.load(find_file(VAL_HR, TARGET_DATE), allow_pickle=True))

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
    # 🔥 CRITICAL: UPSCALE LR → HR SIZE
    ##################################
    lr_t = F.interpolate(
        lr_t,
        size=hr_t.shape[-2:],
        mode='bicubic',
        align_corners=False
    )

    ##################################
    # INFERENCE
    ##################################
    with torch.no_grad():
        sr = model(lr_t)

    ##################################
    # DENORMALIZE
    ##################################
    sr_c = sr * (VMAX - VMIN) + VMIN
    hr_c = hr_t * (VMAX - VMIN) + VMIN

    ##################################
    # METRICS
    ##################################
    rmse = compute_rmse(sr_c, hr_c)
    psnr = compute_psnr(sr_c, hr_c)
    ssim_val = ssim(sr, hr_t).item()
    corr = compute_correlation(sr_c, hr_c)

    print(f"\n===== SRCNN RESULTS ({TARGET_DATE}) =====")
    print(f"RMSE : {rmse:.6f}")
    print(f"PSNR : {psnr:.6f}")
    print(f"SSIM : {ssim_val:.6f}")
    print(f"CORR : {corr:.6f}")

    ##################################
    # VISUALIZE
    ##################################
    save_path = f"{OUT_DIR}/{TARGET_DATE}_srcnn_6000.png"
    visualize(lr, sr_c.squeeze().cpu().numpy(), hr, save_path)

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    evaluate()
