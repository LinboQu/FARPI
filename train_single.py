"""
train_single.py

AI-only baseline / pretrain:
- Train inverse model (seismic -> AI) on selected wells only
- Save full checkpoint with stats (to avoid train/test mismatch)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import join
from torch.utils.data import DataLoader

from setting import TCN1D_train_p
from utils.utils import standardize
from utils.datasets import SeismicDataset1D  # expects (seismic, model, traces)

from model.tcn import TCN_IV_1D_C
from model.CNN2Layer import VishalNet
from model.M2M_LSTM import GRU_MM
from model.Unet_1D import Unet_1D
from model.Transformer import TransformerModel


# -----------------------------
# helpers: read selected wells without pandas
# -----------------------------
def load_selected_wells_trace_indices(csv_path: str, IL: int, XL: int, no_wells: int = 20, seed: int = 2026):
    if (csv_path is None) or (not os.path.isfile(csv_path)):
        raise FileNotFoundError(f"[WELLS] selected_wells_csv not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")

    def _col_idx(name: str):
        if name not in header:
            raise ValueError(f"[WELLS] CSV must contain '{name}'. Got: {header}")
        return header.index(name)

    il_col = _col_idx("INLINE")
    xl_col = _col_idx("XLINE")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    il = data[:, il_col].astype(int)
    xl = data[:, xl_col].astype(int)
    il = np.clip(il, 0, IL - 1)
    xl = np.clip(xl, 0, XL - 1)

    traces = (il * XL + xl).astype(np.int64)
    traces = np.unique(traces)

    if no_wells is not None and len(traces) != int(no_wells):
        rng = np.random.default_rng(int(seed))
        if len(traces) > int(no_wells):
            traces = rng.choice(traces, size=int(no_wells), replace=False).astype(np.int64)
        else:
            print(f"[WELLS][WARN] CSV wells={len(traces)} < requested={no_wells}. Using {len(traces)}.")

    return traces


def get_data_raw_stanford_vi():
    seismic3d = np.load(join("data", "Stanford_VI", "synth_40HZ.npy"))  # (H,IL,XL)
    ai3d = np.load(join("data", "Stanford_VI", "AI.npy"))              # (H,IL,XL)
    H, IL, XL = seismic3d.shape

    seismic = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))  # (N,H)
    model = np.transpose(ai3d.reshape(H, IL * XL), (1, 0))         # (N,H)
    meta = {"H": H, "inline": IL, "xline": XL}
    return seismic, model, meta


def build_inverse_model(model_name: str, in_ch: int):
    if model_name == "tcnc":
        choice = TCN_IV_1D_C
    elif model_name == "VishalNet":
        choice = VishalNet
    elif model_name == "GRU_MM":
        choice = GRU_MM
    elif model_name == "Unet_1D":
        choice = Unet_1D
    elif model_name == "Transformer":
        choice = TransformerModel
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # robust constructor
    try:
        return choice(input_dim=in_ch)
    except TypeError:
        try:
            return choice(in_ch)
        except TypeError:
            return choice()


def train_single(train_p: dict):
    os.makedirs("save_train_model", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data_flag = train_p.get("data_flag", "Stanford_VI")
    if data_flag != "Stanford_VI":
        raise NotImplementedError("This clean single-train script currently supports Stanford_VI only.")

    # 1) raw data
    seismic_raw, model_raw, meta = get_data_raw_stanford_vi()
    H, IL, XL = meta["H"], meta["inline"], meta["xline"]

    # 2) stats (global_model) + save
    no_wells = int(train_p.get("no_wells", 20))
    seismic_std, model_std, stats = standardize(seismic_raw, model_raw, no_wells=no_wells, mode="global_model")
    run_id = f"{train_p['model_name']}_AIonly"
    np.save(join("save_train_model", f"norm_stats_{run_id}_{data_flag}.npy"), stats)

    # 3) crop to 8-multiple
    s_L = seismic_std.shape[-1]
    n = int((s_L // 8) * 8)
    seismic_std = seismic_std[:, :n]
    model_std = model_std[:, :n]

    # 4) add channel dim
    seismic = seismic_std[:, None, :].astype(np.float32)  # (N,1,H)
    model = model_std[:, None, :].astype(np.float32)

    # 5) selected wells traces
    seed = int(train_p.get("seed", 2026))
    csv_path = train_p.get("selected_wells_csv", join("data", "Stanford_VI", "selected_wells_20_seed2026.csv"))
    traces_train = load_selected_wells_trace_indices(csv_path, IL=IL, XL=XL, no_wells=no_wells, seed=seed)
    np.save(join("results", f"{run_id}_{data_flag}_well_trace_indices.npy"), traces_train)

    # 6) dataset/loader
    train_ds = SeismicDataset1D(seismic, model, traces_train)
    num_workers = int(train_p.get("num_workers", 0))
    pin_memory = bool(train_p.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(train_p.get("persistent_workers", True)) and (num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inv = build_inverse_model(train_p["model_name"], in_ch=seismic.shape[1]).to(device)

    optim = torch.optim.Adam(model_inv.parameters(), lr=float(train_p.get("lr", 1e-4)), weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    losses = []
    epochs = int(train_p.get("epochs", 200))

    for ep in range(1, epochs + 1):
        model_inv.train()
        ep_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model_inv(x)
            loss = criterion(y_pred, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_inv.parameters(), float(train_p.get("grad_clip", 1.0)))
            optim.step()
            ep_loss += loss.item()

        ep_loss /= max(1, len(train_loader))
        losses.append(ep_loss)
        if ep % 10 == 0 or ep == 1:
            print(f"[AI-only] Epoch {ep:04d}/{epochs} | loss={ep_loss:.6f}")

    # save full ckpt with stats
    ckpt_path = join("save_train_model", f"{run_id}_full_ckpt_{data_flag}.pth")
    torch.save(
        {
            "inverse_state_dict": model_inv.state_dict(),
            "epoch": epochs,
            "stats": stats,
            "train_p": train_p,
        },
        ckpt_path,
    )
    print(f"[CKPT] saved: {ckpt_path}")

    plt.figure()
    plt.plot(losses)
    plt.title("AI-only train loss")
    plt.tight_layout()
    plt.savefig(join("results", f"{run_id}_{data_flag}_loss.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    train_single(TCN1D_train_p)