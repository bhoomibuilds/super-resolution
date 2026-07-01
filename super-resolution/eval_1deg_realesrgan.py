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

CKPT_PATH = "sst_ckpt_realesrgan/realesrgan_epoch_6000.pth"

TARGET_DATE = "2017-01-01"

OUT_DIR = "realesrgan_sst"

VMIN = 25.0
VMAX = 35.0

############################################
# RRDBNet
############################################

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3*growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4*growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, nf=64, nb=10):
        super().__init__()

        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.last_conv = nn.Conv2d(nf, out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.upconv1(fea))

        fea = self.lrelu(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.upconv2(fea))

        out = self.last_conv(self.lrelu(self.hr_conv(fea)))
        return out

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



def load_model(model, ckpt_path, device):

    ckpt = torch.load(ckpt_path, map_location=device)

    print("Checkpoint keys:", ckpt.keys())

   
    if "generator" in ckpt:
        state_dict = ckpt["generator"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

   
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    print(" Model loaded successfully")

    return model

############################################
# VISUALIZATION
############################################

def visualize(lr, sr, hr, save_path):

    error = sr - hr

    vmin, vmax = 25.3, 28.15
    err_min, err_max = -0.5, 0.5

    extent = [60, 72, 5, 20]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("LR-OISST (1°)")
    plt.imshow(lr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin="lower",interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("RealESRGAN")  # fixed label
    plt.imshow(sr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin="lower",interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("HR-OISST (0.25°)")
    plt.imshow(hr, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin="lower",interpolation="bilinear")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Error (HR - SR)")
    plt.imshow(error, cmap='bwr', vmin=err_min, vmax=err_max,
               extent=extent, origin="lower")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def evaluate():

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RRDBNet().to(device)
    model = load_model(model, CKPT_PATH, device)
    model.eval()

   
    lr = extract(np.load(find_file(VAL_LR, TARGET_DATE), allow_pickle=True))
    hr = extract(np.load(find_file(VAL_HR, TARGET_DATE), allow_pickle=True))

    lr = np.squeeze(lr).astype(np.float32)
    hr = np.squeeze(hr).astype(np.float32)

    
    lr_n = (lr - VMIN) / (VMAX - VMIN)
    hr_n = (hr - VMIN) / (VMAX - VMIN)

    lr_t = torch.tensor(lr_n).unsqueeze(0).unsqueeze(0).to(device)
    hr_t = torch.tensor(hr_n).unsqueeze(0).unsqueeze(0).to(device)

    
    with torch.no_grad():
        sr = model(lr_t)

   
    sr_c = sr * (VMAX - VMIN) + VMIN
    hr_c = hr_t * (VMAX - VMIN) + VMIN

   
    rmse = compute_rmse(sr_c, hr_c)
    psnr = compute_psnr(sr_c, hr_c)
    corr = compute_correlation(sr_c, hr_c)

    print(f"\n===== RealESRGAN RESULTS ({TARGET_DATE}) =====")
    print(f"RMSE : {rmse:.6f}")
    print(f"PSNR : {psnr:.6f}")
    print(f"CORR : {corr:.6f}")

    ##################################
    # VISUALIZE
    ##################################
    save_path = f"{OUT_DIR}/{TARGET_DATE}_realesrgan_6000.png"
    visualize(lr, sr_c.squeeze().cpu().numpy(), hr, save_path)

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    evaluate()
