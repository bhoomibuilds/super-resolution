import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

############################################
# MODEL (SAME AS TRAINING)
############################################
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)

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
    def __init__(self, n_resblocks=16, n_feats=64):
        super().__init__()
        self.head = nn.Conv2d(2, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.upsample = UpsamplerX4(n_feats)
        self.tail = nn.Conv2d(n_feats, 2, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upsample(res)
        return self.tail(x)

############################################
# SSIM (same as training)
############################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
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

    sigma1 = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1**2
    sigma2 = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2**2
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1*mu2

    C1, C2 = 0.01**2, 0.03**2

    return (((2*mu1*mu2 + C1)*(2*sigma12 + C2)) /
            ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))).mean()

############################################
# CONFIG
############################################
CKPT_PATH = "wind_uv_ckpt_mdsr/mdsr_epoch_10000.pth"

LR_U = "data_wind/wind_u/val/LR/2020.nc"
LR_V = "data_wind/wind_v/val/LR/2020.nc"

HR_U = "data_wind/wind_u/val/HR/2020.nc"  # optional
HR_V = "data_wind/wind_v/val/HR/2020.nc"

SAVE_PATH = "mdsr_wind_sr.nc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# LOAD MODEL
############################################
def load_model():
    model = MDSR().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

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
    model = load_model()
    lr_u, lr_v, hr_u, hr_v = load_data()

    sr_u_list, sr_v_list = [], []

    rmse_list, ssim_list = [], []

    for i in range(lr_u.shape[0]):

        lr = np.stack([lr_u[i], lr_v[i]])
        lr = torch.tensor(lr).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            sr = model(lr)

        sr = sr.squeeze(0).cpu().numpy()
        sr_u, sr_v = sr[0], sr[1]

        sr_u_list.append(sr_u)
        sr_v_list.append(sr_v)

        ################################
        # METRICS (optional)
        ################################
        if hr_u is not None:
            hr = np.stack([hr_u[i], hr_v[i]])
            hr = torch.tensor(hr).unsqueeze(0).float().to(DEVICE)

            sr_t = torch.tensor(sr).unsqueeze(0).to(DEVICE)

            rmse = torch.sqrt(F.mse_loss(sr_t, hr)).item()
            ssim_val = ssim(sr_t, hr).item()

            rmse_list.append(rmse)
            ssim_list.append(ssim_val)

    ################################
    # SAVE OUTPUT
    ################################
    sr_u_arr = np.array(sr_u_list)
    sr_v_arr = np.array(sr_v_list)

    ds = xr.Dataset({
        "u10": (["time", "lat", "lon"], sr_u_arr),
        "v10": (["time", "lat", "lon"], sr_v_arr),
    })

    ds.to_netcdf(SAVE_PATH)

    print(f"✅ Saved SR file → {SAVE_PATH}")

    ################################
    # PRINT METRICS
    ################################
    if rmse_list:
        print("\n===== RESULTS =====")
        print(f"RMSE : {np.mean(rmse_list):.4f}")
        print(f"SSIM : {np.mean(ssim_list):.4f}")

############################################
if __name__ == "__main__":
    run_inference()
