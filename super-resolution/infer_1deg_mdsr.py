import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

############################################
# CONFIG
############################################

BASE = "/home/incois/tvsubhaskar/super_resolution/data/OISST"

VAL_LR = f"{BASE}/val/LR"
VAL_HR = f"{BASE}/val/HR"

CKPT_PATH = "sst_ckpt_mdsr/epoch_6000.pth"   # your checkpoint

SAVE_SR = "mdsr.npy"
SAVE_HR = "hr.npy"

VMIN = 25.0
VMAX = 35.0

############################################
# MODEL (IDENTICAL)
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
# DATA HELPERS (MATCH TRAINING EXACTLY)
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
    def get_date(fname):
        return fname.split("_")[-1].replace(".npy", "")

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
    model = MDSR().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    ##################################
    # LOAD FILES
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

            lr = np.squeeze(np.array(lr, dtype=np.float32))
            hr = np.squeeze(np.array(hr, dtype=np.float32))

            ##################################
            # NORMALIZE (same as training)
            ##################################
            lr_n = (lr - VMIN) / (VMAX - VMIN)

            ##################################
            # MODEL INPUT
            ##################################
            lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)

            ##################################
            # INFERENCE
            ##################################
            sr = model(lr_t)

            ##################################
            # DENORMALIZE
            ##################################
            sr = sr * (VMAX - VMIN) + VMIN

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