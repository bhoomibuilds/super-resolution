import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_DIR = "sst_ckpt_mdsr"   # folder containing all checkpoints

CKPTS = list(range(1000, 6001, 500)) 

TARGET_DATE = "2017-01-01"

CSV_OUT = "mdsr_ckpt_metrics.csv"

VMIN = 25.0
VMAX = 35.0

############################################
# MODEL
############################################

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale


class UpsamplerX4(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.conv(x)


class MDSR(nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 64
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(16)])
        self.upsample = UpsamplerX4(n_feats)
        self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upsample(res)
        x = self.tail(x)
        return x

############################################
# SSIM
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
# METRICS
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
# MAIN LOOP
############################################

def evaluate_all():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##################################
    # LOAD DATA (ONCE)
    ##################################
    lr = extract(np.load(find_file(VAL_LR, TARGET_DATE), allow_pickle=True))
    hr = extract(np.load(find_file(VAL_HR, TARGET_DATE), allow_pickle=True))

    lr = np.squeeze(lr).astype(np.float32)
    hr = np.squeeze(hr).astype(np.float32)

    lr_n = (lr - VMIN) / (VMAX - VMIN)
    hr_n = (hr - VMIN) / (VMAX - VMIN)

    lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)
    hr_t = torch.tensor(hr_n).unsqueeze(0).unsqueeze(0).to(device)

    ##################################
    # CSV
    ##################################
    with open(CSV_OUT, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ckpt", "rmse", "psnr", "ssim", "corr"])

        ##################################
        # LOOP CKPTS
        ##################################
        for epoch in CKPTS:

            ckpt_path = f"{CKPT_DIR}/epoch_{epoch}.pth"

            if not os.path.exists(ckpt_path):
                print(f" Missing {ckpt_path}")
                continue

            print(f"\nEvaluating MDSR CKPT: {epoch}")

            model = MDSR().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

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

            print(f"RMSE={rmse:.4f}, PSNR={psnr:.4f}, SSIM={ssim_val:.4f}, CORR={corr:.4f}")

            ##################################
            # WRITE CSV
            ##################################
            writer.writerow([epoch, rmse, psnr, ssim_val, corr])

    print(f"\n CSV saved → {CSV_OUT}")

############################################

if __name__ == "__main__":
    evaluate_all()
