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

CKPT_PATH = "sst_ckpt_satsure/satsure_epoch_6000.pth"

SAVE_SR = "satsure.npy"
SAVE_HR = "hr.npy"

VMIN = 25.0
VMAX = 35.0

############################################
# MODEL (SatSuRE)
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
        return self.conv3(x)

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
    model = SatSuRE().to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)

    # handle both formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

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

            lr = np.squeeze(lr).astype(np.float32)
            hr = np.squeeze(hr).astype(np.float32)

            ##################################
            # NORMALIZATION
            ##################################
            lr_n = (lr - VMIN) / (VMAX - VMIN)

            ##################################
            # TO TENSOR
            ##################################
            lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)

            ##################################
            # INFERENCE (NO UPSCALE BEFORE)
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
