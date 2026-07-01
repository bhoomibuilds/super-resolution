import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_PATH = "sst_ckpt_sinsr_improved/sinsr_epoch_6000.pth"

TARGET_DATE = "2017-01-01"

OUT_DIR = "sinsr_sst"

VMIN = 25.0
VMAX = 35.0

############################################
# NORMALIZATION
############################################

def normalize(x):
    return 2 * (x - VMIN) / (VMAX - VMIN) - 1

def denormalize(x):
    return (x + 1) / 2 * (VMAX - VMIN) + VMIN

############################################
# SinSR MODEL
############################################

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t).view(t.size(0), -1, 1, 1)
        h = self.act(self.conv2(h))
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.softmax(q.transpose(1, 2) @ k / (C ** 0.5), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2)

        return self.proj(out.reshape(B, C, H, W)) + x

class SinSR_UNet(nn.Module):
    def __init__(self, t_dim=256):
        super().__init__()

        self.time_embed = TimeEmbedding(t_dim)

        self.input = nn.Conv2d(1, 64, 3, padding=1)

        self.down1 = ResBlock(64, 128, t_dim)
        self.attn1 = Attention(128)

        self.down2 = ResBlock(128, 256, t_dim)
        self.mid = ResBlock(256, 256, t_dim)

        self.up1 = ResBlock(256, 128, t_dim)
        self.attn2 = Attention(128)

        self.up2 = ResBlock(128, 64, t_dim)

        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        t = torch.zeros(x.size(0), device=x.device)
        t_emb = self.time_embed(t)

        x1 = self.input(x)
        d1 = self.attn1(self.down1(x1, t_emb))
        d2 = self.down2(d1, t_emb)

        m = self.mid(d2, t_emb)

        u1 = self.attn2(self.up1(m, t_emb))
        u2 = self.up2(u1, t_emb)

        return x + self.out(u2)

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
# VISUALIZATION
############################################

def visualize(lr, sr, hr, save_path):

    error = sr - hr

    vmin, vmax = 25.3, 28.15
    err_min, err_max = -0.5, 0.5

    extent = [60, 72, 5, 20]

    plt.figure(figsize=(10, 8))

    plt.subplot(2,2,1)
    plt.title("LR-OISST")
    plt.imshow(lr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin='lower',interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.title("SinSR")
    plt.imshow(sr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin='lower',interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title("HR-OISST")
    plt.imshow(hr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin='lower',interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.title("Error")
    plt.imshow(error, cmap='bwr', vmin=err_min, vmax=err_max, extent=extent, origin='lower')
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

    model = SinSR_UNet().to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])   # 🔥 important
    model.eval()

    lr = extract(np.load(find_file(VAL_LR, TARGET_DATE), allow_pickle=True))
    hr = extract(np.load(find_file(VAL_HR, TARGET_DATE), allow_pickle=True))

    lr = np.squeeze(lr).astype(np.float32)
    hr = np.squeeze(hr).astype(np.float32)

    lr_t = torch.tensor(normalize(lr)).unsqueeze(0).unsqueeze(0).to(device)
    hr_t = torch.tensor(normalize(hr)).unsqueeze(0).unsqueeze(0).to(device)

# 🔥 CRITICAL FIX: Upsample LR to HR size
    lr_t = F.interpolate(lr_t, size=hr_t.shape[-2:], mode="bicubic", align_corners=False)

    with torch.no_grad():
        sr = model(lr_t)

    sr_c = denormalize(sr)
    hr_c = denormalize(hr_t)

    rmse = compute_rmse(sr_c, hr_c)
    psnr = compute_psnr(sr_c, hr_c)
    ssim_val = ssim(sr, hr_t).item()
    corr = compute_correlation(sr_c, hr_c)

    print(f"\n===== SinSR RESULTS ({TARGET_DATE}) =====")
    print(f"RMSE : {rmse:.6f}")
    print(f"PSNR : {psnr:.6f}")
    print(f"SSIM : {ssim_val:.6f}")
    print(f"CORR : {corr:.6f}")

    save_path = f"{OUT_DIR}/{TARGET_DATE}_sinsr_6000.png"
    visualize(lr, sr_c.squeeze().cpu().numpy(), hr, save_path)

    print(f"\nSaved → {save_path}")

############################################

if __name__ == "__main__":
    evaluate()