import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

############################################
# MODEL (SAME AS TRAINING)
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
        x = x + noise
        return self.net(x)

############################################
# CONFIG
############################################
CKPT_PATH = "sinsr_ckpt_wind/ckpt_6000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_PATH = "sinsr_wind_sr.nc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# NORMALIZATION
############################################
def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def denormalize(x, mean, std):
    return x * std + mean

############################################
# LOAD MODEL
############################################
def load_model():
    model = SinSR_Model().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])
    mean = ckpt["mean"]
    std = ckpt["std"]

    model.eval()
    return model, mean, std

############################################
# LOAD DATA
############################################
def load_data():
    lr_u = xr.open_dataset(LR_U)["u10"].values
    lr_v = xr.open_dataset(LR_V)["v10"].values

    hr_u = xr.open_dataset(HR_U)["u10"].values if os.path.exists(HR_U) else None
    hr_v = xr.open_dataset(HR_V)["v10"].values if os.path.exists(HR_V) else None

    return lr_u, lr_v, hr_u, hr_v

############################################
# INFERENCE
############################################
def run_inference():

    model, mean, std = load_model()
    lr_u, lr_v, hr_u, hr_v = load_data()

    sr_u_list, sr_v_list = [], []
    rmse_list = []

    for i in range(lr_u.shape[0]):

        ################################
        # PREP INPUT
        ################################
        lr = np.stack([lr_u[i], lr_v[i]])

        # normalize
        lr = normalize(lr, mean, std)

        lr = torch.tensor(lr).unsqueeze(0).float().to(DEVICE)

        ################################
        # 🔥 UPSAMPLE FIRST
        ################################
        if hr_u is not None:
            target_h, target_w = hr_u.shape[1], hr_u.shape[2]
            lr = F.interpolate(lr, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        else:
            lr = F.interpolate(lr, scale_factor=4,
                               mode='bilinear', align_corners=False)

        ################################
        # 🔥 NOISE (IMPORTANT)
        ################################
        noise = torch.randn_like(lr) * 0.1

        ################################
        # MODEL
        ################################
        with torch.no_grad():
            sr = model(lr, noise)

        sr = sr.squeeze(0).cpu().numpy()

        ################################
        # DENORMALIZE
        ################################
        sr = denormalize(sr, mean, std)

        sr_u, sr_v = sr[0], sr[1]

        sr_u_list.append(sr_u)
        sr_v_list.append(sr_v)

        ################################
        # METRICS
        ################################
        if hr_u is not None:
            hr = np.stack([hr_u[i], hr_v[i]])
            rmse = np.sqrt(np.mean((sr - hr) ** 2))
            rmse_list.append(rmse)

    ################################
    # SAVE NETCDF
    ################################
    sr_u_arr = np.array(sr_u_list)
    sr_v_arr = np.array(sr_v_list)

    ds = xr.Dataset({
        "u10": (["time", "lat", "lon"], sr_u_arr),
        "v10": (["time", "lat", "lon"], sr_v_arr),
    })

    ds.to_netcdf(SAVE_PATH)

    print(f"✅ Saved SR → {SAVE_PATH}")

    ################################
    # RESULTS
    ################################
    if rmse_list:
        print("\n===== RESULTS =====")
        print(f"RMSE : {np.mean(rmse_list):.4f}")

############################################
if __name__ == "__main__":
    run_inference()
