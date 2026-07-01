import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_PATH = "sst_ckpt_sinsr_improved/sinsr_epoch_6000.pth"

SAVE_SR = "sinsr.npy"
SAVE_HR = "hr.npy"

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
# MODEL (SAME)
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
# HELPERS
############################################

def extract(data):
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, np.ndarray):
                return v

    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            return extract(data.item())
        except:
            pass

    return data


def match_files(lr_dir, hr_dir):
    def get_date(f):
        return f.split("_")[-1].replace(".npy", "")

    lr_dict = {get_date(f): f for f in os.listdir(lr_dir) if f.endswith(".npy")}
    hr_dict = {get_date(f): f for f in os.listdir(hr_dir) if f.endswith(".npy")}

    dates = sorted(set(lr_dict) & set(hr_dict))

    lr_files = [lr_dict[d] for d in dates]
    hr_files = [hr_dict[d] for d in dates]

    print(f"Matched samples: {len(dates)}")

    return lr_files, hr_files

############################################
# INFERENCE
############################################

def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ##################################
    # LOAD MODEL
    ##################################
    model = SinSR_UNet().to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])  # important

    model.eval()

    ##################################
    # FILES
    ##################################
    lr_files, hr_files = match_files(VAL_LR, VAL_HR)

    sr_all = []
    hr_all = []

    ##################################
    # LOOP
    ##################################
    with torch.no_grad():
        for lr_f, hr_f in tqdm(zip(lr_files, hr_files), total=len(lr_files)):

            lr = extract(np.load(os.path.join(VAL_LR, lr_f), allow_pickle=True))
            hr = extract(np.load(os.path.join(VAL_HR, hr_f), allow_pickle=True))

            lr = np.squeeze(lr).astype(np.float32)
            hr = np.squeeze(hr).astype(np.float32)

            ##################################
            # NORMALIZE [-1,1]
            ##################################
            lr_n = normalize(lr)

            lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)

            ##################################
            # UPSCALE FIRST (CRITICAL)
            ##################################
            lr_t = F.interpolate(
                lr_t,
                size=hr.shape,
                mode="bicubic",
                align_corners=False
            )

            ##################################
            # INFERENCE
            ##################################
            sr = model(lr_t)

            ##################################
            # DENORMALIZE
            ##################################
            sr = denormalize(sr)

            sr_np = sr.squeeze().cpu().numpy()

            ##################################
            # STORE
            ##################################
            sr_all.append(sr_np)
            hr_all.append(hr)

    ##################################
    # SAVE
    ##################################
    sr_all = np.array(sr_all)
    hr_all = np.array(hr_all)

    np.save(SAVE_SR, sr_all)
    np.save(SAVE_HR, hr_all)

    print("\n✅ DONE")
    print("SR shape:", sr_all.shape)
    print("HR shape:", hr_all.shape)


############################################

if __name__ == "__main__":
    run()
