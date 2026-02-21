"""
test_3D.py (FINAL, drop-in replacement)

鍔熻兘锛?
1) 璇诲彇 Stanford VI / Fanny 鏁版嵁
2) 鍔犺浇璁粌濂界殑妯″瀷鏉冮噸骞跺仛鍏ㄤ綋绱犳帹鐞?
3) 杈撳嚭瀹氶噺鎸囨爣锛歊2 / PCC / SSIM / PSNR / MSE / MAE / MedAE
4) 淇濆瓨鍙鍖栫粨鏋滃埌 results/锛?
   - Pred/True xline=50 鍓栭潰
   - Pred/True inline=100 鍓栭潰
   - Pred/True depth slice = 40/100/160
   - Seismic xline=50 鍓栭潰
   - 鍗曠偣 trace 瀵规瘮
5) 鑻ュ紑鍚?use_aniso_conditioning锛氭瀯寤哄苟淇濆瓨 R(x) 鐨勫垏鐗囧浘 + R3D.npy

鐢ㄦ硶锛圴SCode / F5锛夛細
- 鍦?setting.py 閲岄厤缃?TCN1D_test_p:
  - data_flag='Stanford_VI'
  - model_name='VishalNet_cov_para_Facies_s_uns'
  - no_wells=20
  - use_aniso_conditioning=True/False
"""

import os
import errno
import csv
import json
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import ndimage

import torch
from os.path import join
from torch.utils.data import DataLoader

from setting import *
from utils.utils import standardize
from utils.datasets import SeismicDataset1D
from utils.config_cast import get_bool, get_float, get_int
from src.eval_fingerprint import sha1_of_array, sha1_of_file, stats_of_array
from model.CNN2Layer import VishalNet
from model.tcn import TCN_IV_1D_C
from model.M2M_LSTM import GRU_MM
from model.Unet_1D import Unet_1D
from model.Transformer import TransformerModel

from scipy.stats import pearsonr
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None
import cv2  # 淇PSNR瀵煎叆鏂瑰紡锛堝師浠ｇ爜鐩存帴from cv2 import PSNR鍙兘鎶ラ敊锛?

# anisotropic reliability (FARP)
from utils.reliability_aniso import build_R_and_prior_from_cube


def _infer_run_id_for_full_ckpt(model_name: str, data_flag: str | None = None) -> str:
    """
    Your repo saves full_ckpt as:
      {run_id}_full_ckpt_{data_flag}.pth
    while inference ckpt may be:
      {run_id}_s_uns_{data_flag}.pth
      {model_name}.pth
      {model_name}_{data_flag}.pth

    We therefore try to map 'model_name' -> 'run_id' robustly.
    """
    run_id = str(model_name)

    # If model_name ends with _{data_flag}, strip it (e.g., noise_none_s_uns_Stanford_VI)
    if data_flag:
        df = str(data_flag)
        if run_id.endswith("_" + df):
            run_id = run_id[: -(len(df) + 1)]

    # common suffixes in this repo
    for suf in ["_s_uns", "_uns", "_s", "_best", "_final"]:
        if run_id.endswith(suf):
            run_id = run_id[: -len(suf)]

    # strip data_flag again after suffix removal (handles xxx_s_uns_Stanford_VI)
    if data_flag:
        df = str(data_flag)
        if run_id.endswith("_" + df):
            run_id = run_id[: -(len(df) + 1)]

    return run_id


def load_stats_strict(model_name: str, data_flag: str, out_dir: str = ".", run_id_override: str | None = None) -> dict:
    """
    Strict stats loading:
      1) full_ckpt: save_train_model/{run_id}_full_ckpt_{data_flag}.pth

    No fallback is allowed to avoid cross-run mismatch.
    """
    run_id = run_id_override if (run_id_override is not None and str(run_id_override) != "") else _infer_run_id_for_full_ckpt(model_name, data_flag)
    full_ckpt_path = join(out_dir, "save_train_model", f"{run_id}_full_ckpt_{data_flag}.pth")

    # 1) from full_ckpt
    if os.path.isfile(full_ckpt_path):
        ckpt = torch.load(full_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and ("stats" in ckpt) and (ckpt["stats"] is not None):
            print(f"[NORM] stats loaded from full_ckpt: {full_ckpt_path}")
            return ckpt["stats"]
        raise RuntimeError(f"[NORM][ERROR] full_ckpt exists but stats missing: {full_ckpt_path}")

    raise FileNotFoundError(
        f"Cannot find stats for model={model_name}, data_flag={data_flag}\n"
        f"Tried:\n  {full_ckpt_path}"
    )

# -----------------------------
# 宸ュ叿鍑芥暟
# -----------------------------
def load_selected_wells_trace_indices(
    csv_path: str,
    IL: int,
    XL: int,
    no_wells: int = 20,
    seed: int = 2026,
):
    """
    Read selected wells CSV (expects columns INLINE, XLINE) and convert to trace indices.

    Trace index convention MUST match your flatten order:
      idx = inline * XL + xline
    which is consistent with your get_data_raw flatten:
      seismic3d (H,IL,XL) -> reshape(H, IL*XL) with XL fastest.

    Returns:
      traces_well: np.ndarray shape (no_wells,) dtype int64
    """
    import csv

    if (csv_path is None) or (not os.path.isfile(csv_path)):
        raise FileNotFoundError(f"selected_wells_csv not found: {csv_path}")

    ils = []
    xls = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if ("INLINE" not in reader.fieldnames) or ("XLINE" not in reader.fieldnames):
            raise ValueError(
                f"CSV must contain INLINE and XLINE columns. Got columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                ils.append(int(float(row["INLINE"])))
                xls.append(int(float(row["XLINE"])))
            except Exception:
                continue

    if len(ils) == 0:
        raise ValueError(f"No valid wells parsed from: {csv_path}")

    il = np.clip(np.asarray(ils, dtype=np.int64), 0, IL - 1)
    xl = np.clip(np.asarray(xls, dtype=np.int64), 0, XL - 1)

    traces = (il * XL + xl).astype(np.int64)
    traces = np.unique(traces)

    if no_wells is not None and len(traces) != int(no_wells):
        rng = np.random.default_rng(int(seed))
        if len(traces) > int(no_wells):
            traces = rng.choice(traces, size=int(no_wells), replace=False).astype(np.int64)
        else:
            print(f"[WELLS][WARN] CSV has {len(traces)} wells < requested {no_wells}. Using {len(traces)}.")

    return traces

def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _np_stats(name: str, arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float64)
    return (
        f"{name}: shape={a.shape} min={float(np.nanmin(a)):.6g} max={float(np.nanmax(a)):.6g} "
        f"mean={float(np.nanmean(a)):.6g} std={float(np.nanstd(a)):.6g}"
    )


def _assert_finite_np(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        bad = int(np.size(a) - int(np.isfinite(a).sum()))
        raise RuntimeError(f"[FINITE][ERROR] {name} contains non-finite values: bad={bad}")


def _assert_finite_torch(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int(x.numel() - int(torch.isfinite(x).sum().item()))
        raise RuntimeError(f"[FINITE][ERROR] {name} contains non-finite values: bad={bad}")


def _build_inverse_model(arch_name: str, input_dim: int, device: torch.device) -> torch.nn.Module:
    arch = str(arch_name).strip()
    if arch == "tcnc":
        cls = TCN_IV_1D_C
    elif arch == "VishalNet":
        cls = VishalNet
    elif arch == "GRU_MM":
        cls = GRU_MM
    elif arch == "Unet_1D":
        cls = Unet_1D
    elif arch == "Transformer":
        cls = TransformerModel
    else:
        raise ValueError(f"Unknown inverse model architecture: {arch}")
    try:
        model = cls(input_dim=input_dim).to(device)
    except TypeError:
        try:
            model = cls(input_dim).to(device)
        except TypeError:
            model = cls().to(device)
    return model


def _data_range_true(y_true: np.ndarray, eps: float = 1e-12) -> float:
    dr = float(np.nanmax(y_true) - np.nanmin(y_true))
    return float(dr if dr > eps else 1.0)


def _compute_main_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> dict:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    _assert_finite_np("y_true(metric)", yt)
    _assert_finite_np("y_pred(metric)", yp)

    err = yp - yt
    sse = float(np.sum(err * err))
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    medae = float(np.median(np.abs(err)))

    mu = float(np.mean(yt))
    sst = float(np.sum((yt - mu) ** 2))
    if sst <= eps:
        r2 = np.nan
    else:
        r2 = float(1.0 - (sse / (sst + eps)))

    pcc = _safe_pearson(yt, yp)
    dr_true = _data_range_true(yt, eps=eps)
    if mse <= eps:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((dr_true * dr_true) / (mse + eps)))

    if ssim is None:
        warnings.warn("skimage.metrics.structural_similarity unavailable; SSIM set to NaN.", RuntimeWarning)
        ssim_val = np.nan
    else:
        # Keep existing shape convention: (N,H) -> transpose to (H,N)
        yt2 = np.asarray(y_true, dtype=np.float64).T
        yp2 = np.asarray(y_pred, dtype=np.float64).T
        try:
            ssim_val = float(ssim(yt2, yp2, data_range=dr_true))
        except Exception:
            ssim_val = np.nan

    return {
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "PCC": float(pcc) if np.isfinite(pcc) else np.nan,
        "SSIM": float(ssim_val),
        "PSNR": float(psnr),
        "MSE": float(mse),
        "MAE": float(mae),
        "MedAE": float(medae),
        "SSE": float(sse),
        "SST": float(sst),
    }


def _deterministic_eval_trace_ids(
    traces_test: np.ndarray,
    n_total: int,
    max_eval_traces: int,
    seed: int,
) -> np.ndarray:
    base = np.asarray(traces_test, dtype=np.int64) if traces_test is not None else np.arange(int(n_total), dtype=np.int64)
    if base.ndim != 1:
        raise ValueError(f"traces_test must be 1D, got {base.shape}")
    if (max_eval_traces is None) or (int(max_eval_traces) <= 0) or (int(max_eval_traces) >= int(base.size)):
        return base.copy()
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(base, size=int(max_eval_traces), replace=False)
    picked = np.asarray(np.sort(picked), dtype=np.int64)
    return picked


def diagnose_alignment(
    y_true_eval: np.ndarray,
    y_pred_eval: np.ndarray,
    *,
    allow_apply: bool = False,
) -> tuple[np.ndarray, dict]:
    yt = np.asarray(y_true_eval, dtype=np.float64)
    yp = np.asarray(y_pred_eval, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"diagnose_alignment shape mismatch: true={yt.shape}, pred={yp.shape}")
    if yt.ndim != 2:
        report = {
            "allow_apply": bool(allow_apply),
            "auto_applied": False,
            "reason": f"ndim={yt.ndim} != 2, skip alignment diagnosis",
            "best_transform": "identity",
            "best_metrics": {},
            "identity_metrics": {},
            "candidates": [],
        }
        return y_pred_eval, report

    candidates: list[tuple[str, np.ndarray]] = [("identity", yp)]
    if yp.T.shape == yt.shape:
        candidates.append(("transpose", yp.T))
    candidates.append(("flip_trace", yp[::-1, :]))
    candidates.append(("flip_depth", yp[:, ::-1]))
    candidates.append(("negate", -yp))
    candidates.append(("negate_flip_trace", (-yp)[::-1, :]))
    candidates.append(("negate_flip_depth", (-yp)[:, ::-1]))

    cand_reports = []
    best = None
    for name, cand in candidates:
        m = _compute_main_metrics(yt, cand)
        cand_reports.append(
            {
                "name": name,
                "shape": list(cand.shape),
                "R2": float(m.get("R2", np.nan)),
                "PCC": float(m.get("PCC", np.nan)),
                "MSE": float(m.get("MSE", np.nan)),
                "SSE": float(m.get("SSE", np.nan)),
                "SST": float(m.get("SST", np.nan)),
            }
        )
        r2_s = float(m.get("R2", -np.inf))
        pcc_s = float(m.get("PCC", -np.inf))
        if not np.isfinite(r2_s):
            r2_s = -np.inf
        if not np.isfinite(pcc_s):
            pcc_s = -np.inf
        score = (r2_s, pcc_s)
        if (best is None) or (score > best["score"]):
            best = {"name": name, "arr": cand, "score": score, "metrics": m}

    orig = next((r for r in cand_reports if r["name"] == "identity"), cand_reports[0])
    best_r = next((r for r in cand_reports if r["name"] == best["name"]), cand_reports[0])
    pcc_gain = float(best_r["PCC"] - orig["PCC"]) if (np.isfinite(best_r["PCC"]) and np.isfinite(orig["PCC"])) else -np.inf
    r2_gain = float(best_r["R2"] - orig["R2"]) if (np.isfinite(best_r["R2"]) and np.isfinite(orig["R2"])) else -np.inf
    auto_apply = bool(allow_apply) and (best["name"] != "identity") and (pcc_gain >= 0.15) and (r2_gain >= 0.5)
    chosen = best if auto_apply else {"name": "identity", "arr": yp, "metrics": _compute_main_metrics(yt, yp)}
    report = {
        "allow_apply": bool(allow_apply),
        "auto_applied": bool(auto_apply),
        "chosen_transform": str(chosen["name"]),
        "orig_transform": "identity",
        "best_transform": str(best["name"]),
        "best_metrics": {
            "R2": float(best_r.get("R2", np.nan)),
            "PCC": float(best_r.get("PCC", np.nan)),
            "MSE": float(best_r.get("MSE", np.nan)),
            "SSE": float(best_r.get("SSE", np.nan)),
            "SST": float(best_r.get("SST", np.nan)),
        },
        "identity_metrics": {
            "R2": float(orig.get("R2", np.nan)),
            "PCC": float(orig.get("PCC", np.nan)),
            "MSE": float(orig.get("MSE", np.nan)),
            "SSE": float(orig.get("SSE", np.nan)),
            "SST": float(orig.get("SST", np.nan)),
        },
        "pcc_gain": float(pcc_gain) if np.isfinite(pcc_gain) else np.nan,
        "r2_gain": float(r2_gain) if np.isfinite(r2_gain) else np.nan,
        "thresholds": {"min_pcc_gain": 0.15, "min_r2_gain": 0.5},
        "candidates": cand_reports,
    }
    return chosen["arr"].astype(y_pred_eval.dtype, copy=False), report


def _depth_axis_mismatch_check(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    *,
    sample_traces: int = 128,
    seed: int = 2026,
    min_gain: float = 0.30,
    min_abs_pcc: float = 0.10,
) -> dict:
    yt = np.asarray(y_true_2d, dtype=np.float64)
    yp = np.asarray(y_pred_2d, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"_depth_axis_mismatch_check shape mismatch: true={yt.shape}, pred={yp.shape}")
    if yt.ndim != 2:
        raise ValueError(f"_depth_axis_mismatch_check expects 2D arrays, got ndim={yt.ndim}")

    n_tr = int(yt.shape[0])
    if n_tr <= 0:
        return {
            "n_total_traces": 0,
            "n_sampled": 0,
            "n_valid": 0,
            "mean_pcc_identity": np.nan,
            "mean_pcc_flipdepth": np.nan,
            "gain": np.nan,
            "min_gain": float(min_gain),
            "min_abs_pcc": float(min_abs_pcc),
            "suspicious": False,
        }

    n_pick = int(max(1, min(int(sample_traces), n_tr)))
    rs = np.random.RandomState(int(seed))
    if n_pick < n_tr:
        idx = rs.choice(np.arange(n_tr, dtype=np.int64), size=n_pick, replace=False).astype(np.int64)
    else:
        idx = np.arange(n_tr, dtype=np.int64)

    p0_list = []
    p1_list = []
    for i in idx.tolist():
        a = yt[i, :]
        b = yp[i, :]
        if (a.size < 2) or (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            continue
        sa = float(np.std(a))
        sb = float(np.std(b))
        if (sa < 1e-12) or (sb < 1e-12):
            continue
        p0 = _safe_pearson(a, b)
        p1 = _safe_pearson(a, b[::-1])
        if np.isfinite(p0):
            p0_list.append(float(p0))
        if np.isfinite(p1):
            p1_list.append(float(p1))

    n_valid = int(min(len(p0_list), len(p1_list)))
    if n_valid <= 0:
        mean_p0 = np.nan
        mean_p1 = np.nan
        gain = np.nan
    else:
        mean_p0 = float(np.mean(np.asarray(p0_list[:n_valid], dtype=np.float64)))
        mean_p1 = float(np.mean(np.asarray(p1_list[:n_valid], dtype=np.float64)))
        gain = float(mean_p1 - mean_p0)

    suspicious = bool(
        np.isfinite(gain)
        and np.isfinite(mean_p1)
        and (gain >= float(min_gain))
        and (abs(mean_p1) >= float(min_abs_pcc))
    )
    return {
        "n_total_traces": int(n_tr),
        "n_sampled": int(len(idx)),
        "n_valid": int(n_valid),
        "mean_pcc_identity": float(mean_p0) if np.isfinite(mean_p0) else np.nan,
        "mean_pcc_flipdepth": float(mean_p1) if np.isfinite(mean_p1) else np.nan,
        "gain": float(gain) if np.isfinite(gain) else np.nan,
        "min_gain": float(min_gain),
        "min_abs_pcc": float(min_abs_pcc),
        "sample_traces": int(sample_traces),
        "seed": int(seed),
        "suspicious": bool(suspicious),
    }


def _apply_alignment_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(arr)
    if transform == "identity":
        return x
    if transform == "transpose":
        return x.T
    if transform == "flip_trace":
        return x[::-1, :]
    if transform == "flip_depth":
        return x[:, ::-1]
    if transform == "negate":
        return -x
    if transform == "negate_flip_trace":
        return (-x)[::-1, :]
    if transform == "negate_flip_depth":
        return (-x)[:, ::-1]
    raise ValueError(f"Unknown alignment transform: {transform}")


def _load_inverse_model_strict(
    out_dir: str,
    run_id: str,
    inferred_run_id: str,
    data_flag: str,
    input_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict]:
    used_run_id = run_id if run_id else inferred_run_id
    if not used_run_id:
        raise RuntimeError("[CKPT][ERROR] empty run_id and inferred_run_id, cannot resolve unique checkpoint path.")
    ckpt_path = join(out_dir, "save_train_model", f"{used_run_id}_full_ckpt_{data_flag}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CKPT][ERROR] full_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or ("inverse_state_dict" not in ckpt):
        raise RuntimeError(f"[CKPT][ERROR] invalid full_ckpt format: {ckpt_path}")

    train_p = ckpt.get("train_p", {}) if isinstance(ckpt.get("train_p", {}), dict) else {}
    arch_name = str(train_p.get("model_name", "VishalNet"))
    model = _build_inverse_model(arch_name=arch_name, input_dim=input_dim, device=device)

    incompat = model.load_state_dict(ckpt["inverse_state_dict"], strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    print(
        f"[CKPT] strict-check path={ckpt_path} | run_id={used_run_id} | arch={arch_name} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"[CKPT][ERROR] state_dict mismatch: missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    model.to(device)
    return model, ckpt_path, train_p


def _percentile_vminmax(arr: np.ndarray, p_low=1, p_high=99):
    """Compute robust low/high percentiles for visualization range."""
    vmin = np.percentile(arr, p_low)
    vmax = np.percentile(arr, p_high)
    return float(vmin), float(vmax)


def _safe_pearson(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2:
        return np.nan
    if (float(np.std(x)) < 1e-12) or (float(np.std(y)) < 1e-12):
        return np.nan
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return np.nan


def _safe_r2(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    sst = float(np.sum((x - mu) ** 2))
    if sst < 1e-12:
        return np.nan
    sse = float(np.sum((y - x) ** 2))
    return float(1.0 - (sse / (sst + 1e-12)))


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    medae = float(np.median(np.abs(y_pred - y_true)))
    dr_true = _data_range_true(y_true, eps=1e-12)
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10((dr_true * dr_true) / (mse + 1e-12)))
    return {
        "R2": _safe_r2(y_true, y_pred),
        "PCC": _safe_pearson(y_true, y_pred),
        "MSE": mse,
        "MAE": mae,
        "MedAE": medae,
        "PSNR": psnr,
    }


def save_facies_metrics(
    ai_true: np.ndarray,
    ai_pred: np.ndarray,
    facies_flat: np.ndarray,
    out_dir: str,
    out_prefix: str,
    facies_name_map: dict | None = None,
):
    """
    Save class-wise metrics for facies regions.
    facies_flat shape should be (N,H), aligned with ai_true/ai_pred.
    """
    _ensure_dir(join(out_dir, "results"))
    y_true = np.asarray(ai_true, dtype=np.float32)
    y_pred = np.asarray(ai_pred, dtype=np.float32)
    fac = np.asarray(facies_flat).astype(np.int64)
    if fac.shape != y_true.shape:
        raise ValueError(f"facies shape mismatch: fac={fac.shape}, ai={y_true.shape}")

    uniq = np.unique(fac)
    if facies_name_map is None:
        facies_name_map = {0: "boundary", 1: "interbay", 2: "channel", 3: "pointbar"}

    rows = []
    for fid in uniq:
        mask = (fac == int(fid))
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        m = _compute_basic_metrics(yt, yp)

        # Approximate SSIM in ROI by filling outside ROI with facies-mean to keep shape.
        yt_img = y_true.copy()
        yp_img = y_pred.copy()
        yt_mean = float(np.mean(yt))
        yp_mean = float(np.mean(yp))
        yt_img[~mask] = yt_mean
        yp_img[~mask] = yp_mean
        dr = float(yt_img.max() - yt_img.min() + 1e-12)
        if ssim is None:
            warnings.warn("skimage.metrics.structural_similarity unavailable; facies SSIM set to NaN.", RuntimeWarning)
            ssim_val = np.nan
        else:
            try:
                ssim_val = float(ssim(yt_img.T, yp_img.T, data_range=dr))
            except Exception:
                ssim_val = np.nan

        row = {
            "facies_id": int(fid),
            "facies_name": str(facies_name_map.get(int(fid), f"facies_{int(fid)}")),
            "count": cnt,
            "ratio": float(cnt / fac.size),
            "R2": m["R2"],
            "PCC": m["PCC"],
            "SSIM": ssim_val,
            "PSNR": m["PSNR"],
            "MSE": m["MSE"],
            "MAE": m["MAE"],
            "MedAE": m["MedAE"],
        }
        rows.append(row)

    headers = ["facies_id", "facies_name", "count", "ratio", "R2", "PCC", "SSIM", "PSNR", "MSE", "MAE", "MedAE"]
    csv_path = join(out_dir, "results", f"{out_prefix}_facies_metrics.csv")
    md_path = join(out_dir, "results", f"{out_prefix}_facies_metrics.md")
    json_path = join(out_dir, "results", f"{out_prefix}_facies_metrics.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r["facies_id"]),
                        str(r["facies_name"]),
                        str(r["count"]),
                        f"{float(r['ratio']):.6f}",
                        "nan" if np.isnan(r["R2"]) else f"{float(r['R2']):.6f}",
                        "nan" if np.isnan(r["PCC"]) else f"{float(r['PCC']):.6f}",
                        "nan" if np.isnan(r["SSIM"]) else f"{float(r['SSIM']):.6f}",
                        f"{float(r['PSNR']):.6f}",
                        f"{float(r['MSE']):.6f}",
                        f"{float(r['MAE']):.6f}",
                        f"{float(r['MedAE']):.6f}",
                    ]
                )
                + " |\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return rows, csv_path, md_path, json_path


def save_residual_maps(
    ai_true_flat: np.ndarray,
    ai_pred_flat: np.ndarray,
    meta: dict,
    out_prefix: str,
    out_dir: str = ".",
    xline_pick: int = 50,
    inline_pick: int = 100,
    depth_slices=(40, 100, 160),
):
    """Save residual cube and slices: residual = pred - true."""
    _ensure_dir(join(out_dir, "results"))
    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    reshape_order = meta.get("reshape_order", "C")

    pred = np.asarray(ai_pred_flat).reshape(IL, XL, H, order=reshape_order)
    true = np.asarray(ai_true_flat).reshape(IL, XL, H, order=reshape_order)
    res = pred - true
    np.save(join(out_dir, "results", f"{out_prefix}_residual_AI_ILXLH.npy"), res.astype(np.float32))
    np.save(join(out_dir, "results", f"{out_prefix}_residual_AI_HILXL.npy"), np.transpose(res, (2, 0, 1)).astype(np.float32))

    vmax = float(np.percentile(np.abs(res), 99))
    vmax = max(vmax, 1e-6)

    xline_pick = int(np.clip(xline_pick, 0, XL - 1))
    inline_pick = int(np.clip(inline_pick, 0, IL - 1))
    imshow_kws = dict(cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest", origin="upper")

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.imshow(res[:, xline_pick, :].T, **imshow_kws)
    ax.set_title(f"Residual (Pred-True) | xline={xline_pick}")
    ax.set_xlabel("Inline index")
    ax.set_ylabel("Depth")
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Residual_xline_{xline_pick}.png"), bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.imshow(res[inline_pick, :, :].T, **imshow_kws)
    ax.set_title(f"Residual (Pred-True) | inline={inline_pick}")
    ax.set_xlabel("Xline index")
    ax.set_ylabel("Depth")
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Residual_inline_{inline_pick}.png"), bbox_inches="tight")
    plt.close()

    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.imshow(res[:, :, z], **imshow_kws)
        ax.set_title(f"Residual (Pred-True) | depth={z}")
        ax.set_xlabel("Xline index")
        ax.set_ylabel("Inline index")
        plt.savefig(join(out_dir, "results", f"{out_prefix}_Residual_depth_{z}.png"), bbox_inches="tight")
        plt.close()


def get_data_raw(data_flag='Stanford_VI', seismic_override_path: str | None = None):
    """
    浠呰鍙栧師濮嬫暟鎹紝涓嶅仛浠讳綍鏍囧噯鍖?瑁佸壀鎿嶄綔
    杈撳嚭锛?
      seismic_raw: (N, H)  鍘熷鍦伴渿鏁版嵁锛堟湭鏍囧噯鍖栵級
      model_raw  : (N, H)  鍘熷妯″瀷鏁版嵁锛堟湭鏍囧噯鍖栵級
      facies_raw : (N, H)  鍘熷鍦伴渿鐩告暟鎹?
      meta       : dict(H, inline, xline, seismic3d, model3d, facies3d)
    """
    meta = {}

    if data_flag == 'Stanford_VI':
        # 璇诲彇鍘熷3D鏁版嵁 (H, IL, XL)
        seismic_path = seismic_override_path if seismic_override_path else join('data', data_flag, 'synth_40HZ.npy')
        seismic3d = np.load(seismic_path)
        model3d = np.load(join('data', data_flag, 'AI.npy'))
        facies3d = np.load(join('data', data_flag, 'Facies.npy'))

        H, IL, XL = seismic3d.shape
        meta = {
            'H': H, 'inline': IL, 'xline': XL,
            'seismic3d': seismic3d, 'model3d': model3d, 'facies3d': facies3d
        }

        # 灞曞钩涓簍race缁村害锛?H, IL*XL) -> (IL*XL, H)
        seismic_raw = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))
        model_raw = np.transpose(model3d.reshape(H, IL * XL), (1, 0))
        facies_raw = np.transpose(facies3d.reshape(H, IL * XL), (1, 0))

        print(f"[{data_flag}] seismic source: {seismic_path}")
        print(f"[{data_flag}] 鍘熷鏁版嵁缁村害: model={model_raw.shape}, seismic={seismic_raw.shape}, facies={facies_raw.shape}")
        print(f"[{data_flag}] 鍘熷鏁版嵁鍧囧€? model={float(model_raw.mean()):.4f}, seismic={float(seismic_raw.mean()):.4f}")

    elif data_flag == 'Fanny':
        # 鍏煎Fanny鏁版嵁闆嗭紙淇濇寔鍜岃缁冧竴鑷寸殑鍘熷璇诲彇閫昏緫锛?
        seismic_raw = np.load(join('data', data_flag, 'seismic.npy'))
        GR_raw = np.load(join('data', data_flag, 'GR.npy'))
        model_raw = np.load(join('data', data_flag, 'Impedance.npy'))
        facies_raw = np.load(join('data', data_flag, 'facies.npy'))
        # 鐩爣鍙互鏄疓R锛堝拰璁粌涓€鑷达級
        model_raw = GR_raw
        # 鑷姩璁＄畻Fanny鐨刬nline/xline锛堝亣璁炬槸姝ｆ柟褰級
        n_traces = model_raw.shape[0]
        IL = XL = int(np.sqrt(n_traces))
        meta = {
            'H': model_raw.shape[-1], 'inline': IL, 'xline': XL,
            'seismic3d': seismic_raw.reshape(IL, XL, -1).transpose(2,0,1),  # 閫傞厤3D鏍煎紡
            'model3d': model_raw.reshape(IL, XL, -1).transpose(2,0,1)
        }
        print(f"[{data_flag}] 鍘熷鏁版嵁缁村害: model={model_raw.shape}, seismic={seismic_raw.shape}")

    else:
        raise ValueError(f"涓嶆敮鎸佺殑鏁版嵁闆? {data_flag}")

    return seismic_raw, model_raw, facies_raw, meta


def show_stanford_vi(
    AI_act_flat: np.ndarray,
    AI_pred_flat: np.ndarray,
    seismic_flat: np.ndarray,
    meta: dict,
    out_prefix: str,
    xline_pick: int = 50,
    inline_pick: int = 100,
    depth_slices=(42, 100, 160),
    out_dir: str = ".",
):
    """
    鍙鍖朣tanford VI鐨勯娴?鐪熷疄鍓栭潰鍜屽垏鐗?
    閫傞厤鑷鍚庣殑reshape order锛岃В鍐崇珫鐩存潯甯﹂棶棰?
    """
    _ensure_dir(join(out_dir, "results"))

    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    # 浠巑eta涓幏鍙栬嚜妫€鍚庣殑reshape椤哄簭锛圕/F锛?
    reshape_order = meta.get("reshape_order", "C")

    # 鍏抽敭锛氫娇鐢ㄦ纭殑reshape order閲嶅涓?D
    AI_act = AI_act_flat.reshape(IL, XL, H, order=reshape_order)
    AI_pred = AI_pred_flat.reshape(IL, XL, H, order=reshape_order)

    # 澶勭悊鍦伴渿鏁版嵁鐨剅eshape
    if seismic_flat.ndim == 3:
        seis_amp = seismic_flat[:, 0, :].reshape(IL, XL, H, order=reshape_order)
    else:
        seis_amp = seismic_flat.reshape(IL, XL, H, order=reshape_order)

    # 娣诲姞杞诲井鍣０鎻愬崌鍙鍖栨晥鏋滐紙鍜屽師濮嬫暟鎹鏍间竴鑷达級
    blurred = ndimage.gaussian_filter(seis_amp, sigma=1.1)
    seis_plot = blurred + 0.5 * blurred.std() * np.random.random(blurred.shape)

    # 鍧愭爣杞崲锛堢储寮曗啋绫筹紝IL/XL姣忔牸25绫筹級
    il_dist = IL * 25
    xl_dist = XL * 25

    # 鍩轰簬鐪熷疄鍊艰绠梤obust鐨剉min/vmax
    vmin, vmax = _percentile_vminmax(AI_act, 1, 99)
    # Avoid matplotlib resampling artifacts (thin stripe lines) on saved PNGs.
    imshow_kws = dict(interpolation="nearest", origin="upper")

    # -------- Pred xline 鍓栭潰 --------
    xline_pick = int(np.clip(xline_pick, 0, XL - 1))
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(
        AI_pred[:, xline_pick, :].T,
        vmin=vmin,
        vmax=vmax,
        extent=(0, il_dist, H, 0),
        **imshow_kws,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Inversion Profile (Pred) | xline={xline_pick}", fontsize=20)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Pred_xline_{xline_pick}.png"), bbox_inches='tight')
    plt.close()

    # -------- Pred inline 鍓栭潰 --------
    inline_pick = int(np.clip(inline_pick, 0, IL - 1))
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(
        AI_pred[inline_pick, :, :].T,
        vmin=vmin,
        vmax=vmax,
        extent=(0, xl_dist, H, 0),
        **imshow_kws,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(45 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Inversion Profile (Pred) | inline={inline_pick}", fontsize=20)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Pred_inline_{inline_pick}.png"), bbox_inches='tight')
    plt.close()

    # -------- Pred depth 鍒囩墖 --------
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(
            AI_pred[:, :, z],
            vmin=vmin,
            vmax=vmax,
            extent=(0, xl_dist, 0, il_dist),
            **imshow_kws,
        )
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("x(m)", fontsize=20)
        ax.set_ylabel("y(m)", fontsize=20)
        ax.set_title(f"Inversion Slice (Pred) | depth={z}", fontsize=20)
        plt.savefig(join(out_dir, "results", f"{out_prefix}_Pred_depth_{z}.png"), bbox_inches='tight')
        plt.close()

    # -------- True xline 鍓栭潰 --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(
        AI_act[:, xline_pick, :].T,
        vmin=vmin,
        vmax=vmax,
        extent=(0, il_dist, H, 0),
        **imshow_kws,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Ground Truth | xline={xline_pick}", fontsize=20)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_True_xline_{xline_pick}.png"), bbox_inches='tight')
    plt.close()

    # -------- True inline 鍓栭潰 --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(
        AI_act[inline_pick, :, :].T,
        vmin=vmin,
        vmax=vmax,
        extent=(0, xl_dist, H, 0),
        **imshow_kws,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(45 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Ground Truth | inline={inline_pick}", fontsize=20)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_True_inline_{inline_pick}.png"), bbox_inches='tight')
    plt.close()

    # -------- True depth 鍒囩墖 --------
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(
            AI_act[:, :, z],
            vmin=vmin,
            vmax=vmax,
            extent=(0, xl_dist, 0, il_dist),
            **imshow_kws,
        )
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("x(m)", fontsize=20)
        ax.set_ylabel("y(m)", fontsize=20)
        ax.set_title(f"Ground Truth | depth={z}", fontsize=20)
        plt.savefig(join(out_dir, "results", f"{out_prefix}_True_depth_{z}.png"), bbox_inches='tight')
    plt.close()

    # -------- Seismic xline 鍓栭潰 --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(
        seis_plot[:, xline_pick, :].T,
        cmap="seismic",
        extent=(0, il_dist, H, 0),
        **imshow_kws,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Seismic Profile | xline={xline_pick}", fontsize=20)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Seismic_xline_{xline_pick}.png"), bbox_inches='tight')
    plt.close()

    # -------- 鍗曠偣 Trace 瀵规瘮 --------
    B_x, B_y = inline_pick, xline_pick
    depth_index = np.arange(H) * 1.0

    fig, ax = plt.subplots(figsize=(16, 6), dpi=400)
    ax.plot(depth_index, AI_pred[B_x, B_y, :], linestyle="--", label="Pred", linewidth=3.0, color='red')
    ax.plot(depth_index, AI_act[B_x, B_y, :], linestyle="-", label="True", linewidth=3.0, color='blue')
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("Depth(m)", fontsize=20)
    ax.set_ylabel("Impedance (standardized)", fontsize=20)
    ax.set_title(f"Trace inline={B_x}, xline={B_y}", fontsize=20)
    plt.legend(loc="upper left", fontsize=16)
    plt.savefig(join(out_dir, "results", f"{out_prefix}_Trace_inline_{B_x}_xline_{B_y}.png"), bbox_inches='tight')
    plt.close()


def save_R_visualization(R_flat_torch: torch.Tensor, meta: dict, out_prefix: str, depth_slices=(40, 100, 160), out_dir: str = "."):
    """
    鍙鍖栧苟淇濆瓨R(x)鐨?D鏁版嵁鍜屽垏鐗囷紝閫傞厤姝ｇ‘鐨剅eshape order
    """
    _ensure_dir(join(out_dir, "results"))
    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    reshape_order = meta.get("reshape_order", "C")

    # 鐢ㄦ纭殑order閲嶅R鏁版嵁
    R3d = R_flat_torch.detach().cpu().numpy().reshape(IL, XL, H, order=reshape_order)
    # 淇濆瓨涓ょ缁村害椤哄簭鐨凴3D鏁版嵁
    np.save(join(out_dir, "results", f"{out_prefix}_R3d_ILXLH.npy"), R3d)
    np.save(join(out_dir, "results", f"{out_prefix}_R3d_HILXL.npy"), np.transpose(R3d, (2, 0, 1)))

    # 淇濆瓨R(x)娣卞害鍒囩墖
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        plt.figure(figsize=(6, 5), dpi=250)
        plt.imshow(R3d[:, :, z], cmap="hot")
        plt.colorbar(label="Reliability R(x)")
        plt.title(f"Anisotropic reliability R | depth={z}")
        plt.tight_layout()
        plt.savefig(join(out_dir, "results", f"{out_prefix}_R_depth_{z}.png"))
        plt.close()


# -----------------------------
# 涓绘祴璇曞嚱鏁?
# -----------------------------
def test(test_p: dict):
    out_dir = test_p.get("out_dir", ".")
    _ensure_dir(out_dir)
    _ensure_dir(join(out_dir, "results"))
    _ensure_dir(join(out_dir, "save_train_model"))

    # 璇诲彇閰嶇疆
    model_name = test_p["model_name"]
    run_id = str(test_p.get("run_id", "")).strip()
    data_flag = test_p["data_flag"]
    no_wells = int(test_p.get("no_wells", 20))

    # 鍚堝苟閰嶇疆锛堟祴璇曢厤缃鐩栬缁冮厤缃級
    cfg = {**TCN1D_train_p, **test_p}
    debug_metrics = get_bool(cfg, "debug_metrics", True)
    max_eval_traces = get_int(cfg, "max_eval_traces", -1)
    skip_plots = get_bool(cfg, "skip_plots", False)
    save_artifacts = get_bool(cfg, "save_artifacts", True)
    enable_margin_gate_resolved = get_bool(cfg, "enable_margin_gate", True)
    print(
        "[CFG-RESOLVED-TEST] "
        f"save_artifacts={int(save_artifacts)} | max_eval_traces={int(max_eval_traces)} | "
        f"enable_margin_gate={int(enable_margin_gate_resolved)}"
    )
    print(
        f"[ABLATION] model_name={model_name} | out_dir={out_dir} | "
        f"run_id={run_id if run_id else _infer_run_id_for_full_ckpt(model_name, data_flag)} | "
        f"lambda_recon={float(cfg.get('lambda_recon', 1.0))} | "
        f"use_aniso_conditioning={int(get_bool(cfg, 'use_aniso_conditioning', False))} | "
        f"aniso_gamma={float(cfg.get('aniso_gamma', 8.0))} | "
        f"iterative_R={int(get_bool(cfg, 'iterative_R', False))}"
    )
    print(
        "[CFG] "
        + " | ".join(
            [
                f"use_aniso_conditioning={int(get_bool(cfg, 'use_aniso_conditioning', False))}",
                f"iterative_R={int(get_bool(cfg, 'iterative_R', False))}",
                f"adaptive_eta_enable={int(get_bool(cfg, 'adaptive_eta_enable', False))}",
                f"enable_margin_gate={int(get_bool(cfg, 'enable_margin_gate', True))}",
                f"alpha_update_mode={cfg.get('alpha_update_mode', '')}",
                f"eta_update_mode={cfg.get('eta_update_mode', '')}",
                f"snr_power={get_float(cfg, 'snr_power', 2.0)}",
                f"max_eval_traces={max_eval_traces}",
                f"debug_metrics={int(debug_metrics)}",
                f"skip_plots={int(skip_plots)}",
                f"save_artifacts={int(save_artifacts)}",
            ]
        )
    )
    inferred_run_id = _infer_run_id_for_full_ckpt(model_name, data_flag)
    if run_id and (run_id != inferred_run_id):
        print(
            f"[WARN] run_id ({run_id}) != inferred model run_id ({inferred_run_id}); "
            f"preferring run_id with strict no-fallback."
        )
    noise_tag = str(cfg.get("noise_tag", cfg.get("run_tag", ""))).strip()
    seismic_override_path = cfg.get("seismic_override_path", None)
    if noise_tag and (noise_tag != "noise_none"):
        if (seismic_override_path is None) or (str(seismic_override_path).strip() == ""):
            raise RuntimeError(
                f"[INPUT][ERROR] noise_tag={noise_tag} requires explicit seismic_override_path, got None/empty."
            )
        if not os.path.isfile(str(seismic_override_path)):
            raise RuntimeError(
                f"[INPUT][ERROR] noise_tag={noise_tag} but seismic_override_path does not exist: {seismic_override_path}"
            )

    ### 1. 璇诲彇鍘熷鏁版嵁锛堟棤鏍囧噯鍖栥€佹棤瑁佸壀锛?
    seismic_raw, model_raw, facies_raw, meta = get_data_raw(
        data_flag=data_flag,
        seismic_override_path=seismic_override_path,
    )

    ### 2. Trace椤哄簭鑷锛堟牳蹇冿紒瑙ｅ喅绔栫洿鏉″甫闂锛?
    IL, XL, H = meta["inline"], meta["xline"], meta["H"]
    # 楠岃瘉flatten/reshape鍙€嗘€?
    model_3d = model_raw.reshape(IL, XL, H)
    model_back = model_3d.reshape(IL * XL, H)
    reshape_err = np.abs(model_back - model_raw).max()
    print(f"[SANITY CHECK] flatten/reshape 鏈€澶ц宸? {reshape_err:.6f}")

    # 寮傚父澶勭悊锛氳嚜鍔ㄩ€傞厤C/F order
    if reshape_err > 1e-10:
        print("[WARNING] 灞曞钩/閲嶅涓嶅彲閫嗭紒灏濊瘯Fortran椤哄簭锛坥rder='F'锛変慨姝?..")
        model_3d_F = model_raw.reshape(IL, XL, H, order='F')
        model_back_F = model_3d_F.reshape(IL * XL, H, order='F')
        reshape_err_F = np.abs(model_back_F - model_raw).max()
        print(f"[SANITY CHECK] 淇鍚庢渶澶ц宸? {reshape_err_F:.6f}")
    
        if reshape_err_F < 1e-10:
            meta["reshape_order"] = "F"
            print("[SANITY CHECK] 淇鎴愬姛锛佷娇鐢‵ortran椤哄簭锛堝垪浼樺厛锛塺eshape")
        else:
            raise RuntimeError(
                f"灞曞钩/閲嶅濮嬬粓涓嶅彲閫嗭紒鍘熷璇樊={reshape_err:.6f}, 淇鍚?{reshape_err_F:.6f} \n"
                f"璇锋鏌ョ淮搴︼細IL={IL}, XL={XL}, H={H}, model_raw.shape={model_raw.shape}"
            )
    else:
        meta["reshape_order"] = "C"
        print("[SANITY CHECK] flatten/reshape reversible, dimension order looks correct.")

    ### 3. 鍔犺浇璁粌闆哠tats锛堜紭鍏堜粠full_ckpt璇诲彇锛屾潨缁濋敊閰嶏級
    stats = load_stats_strict(
        model_name=model_name,
        data_flag=data_flag,
        out_dir=out_dir,
        run_id_override=run_id if run_id else None,
    )
    print(f"[NORM] stats loaded | mode={stats.get('mode')} | keys={list(stats.keys())}")

    ### 4. 搴旂敤璁粌Stats鍋氭爣鍑嗗寲锛堝拰璁粌瀹屽叏涓€鑷达級
    seismic, model, _ = standardize(seismic_raw, model_raw, stats=stats)

    ### 5. 瑁佸壀鍒?鐨勫€嶆暟锛堥€傞厤UNet/TCN涓嬮噰鏍凤級
    s_L = seismic.shape[-1]
    n = int((s_L // 8) * 8)
    seismic = seismic[:, :n]
    model = model[:, :n]
    facies = facies_raw[:, :n]

    ### 6. 娣诲姞閫氶亾缁村害锛堝拰璁粌涓€鑷达級
    seismic = seismic[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
    model = model[:, np.newaxis, :].astype(np.float32)
    facies = facies[:, np.newaxis, :]

    ### 7. 鏋勫缓鍚勫悜寮傛€閫氶亾锛堣嫢寮€鍚級
    R_flat = None
    if get_bool(cfg, "use_aniso_conditioning", False) and data_flag == "Stanford_VI":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # 鉁?鐪熷疄浜曠偣浣滀负绉嶅瓙锛堜粠CSV璇诲彇 INLINE/XLINE锛?
        seed = int(cfg.get("seed", 2026))
        csv_path = cfg.get("selected_wells_csv", None)
    
        traces_well = load_selected_wells_trace_indices(
            csv_path=csv_path,
            IL=int(meta["inline"]),
            XL=int(meta["xline"]),
            no_wells=int(no_wells),
            seed=seed,
        )
        print(f"[WELLS] using selected wells from CSV: {csv_path} | count={len(traces_well)}")
        np.save(join(out_dir, "results", f"{model_name}_{data_flag}_well_trace_indices.npy"), traces_well)

        well_idx = torch.from_numpy(traces_well).to(device)
        # 鍔犺浇3D鏁版嵁鍒拌澶?
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        fac3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        # 鏋勫缓R(x)
        ch_id = int(cfg.get("channel_id", 2))
        p0_3d = (fac3d == ch_id).float().clamp(0.0, 1.0)
        conf0_3d = torch.ones_like(p0_3d)
        _assert_finite_torch("p0_3d", p0_3d)
        _assert_finite_torch("conf0_3d", conf0_3d)
        rho_aniso_map0_3d = None
        eta_update_map0_3d = None
        eta_map0_3d = None
        if get_bool(cfg, "adaptive_eta_enable", False):
            noise_cfg_path = join(out_dir, "noise_config.json")
            snr_db_real = None
            snr_db_target = None
            if os.path.isfile(noise_cfg_path):
                try:
                    with open(noise_cfg_path, "r", encoding="utf-8") as f:
                        ncfg = json.load(f)
                    if ncfg.get("snr_db_real", None) is not None:
                        snr_db_real = float(ncfg["snr_db_real"])
                    if ncfg.get("snr_db_target", None) is not None:
                        snr_db_target = float(ncfg["snr_db_target"])
                except Exception:
                    snr_db_real = None
                    snr_db_target = None
            snr_low = float(cfg.get("snr_low", 10.0))
            snr_high = float(cfg.get("snr_high", 30.0))
            snr_power = float(cfg.get("snr_power", 2.0))
            eta0 = float(cfg.get("aniso_eta", 0.6))
            adapt_eps = float(cfg.get("adaptive_eps", 1e-8))
            snr_freeze_cfg = cfg.get("snr_freeze", 12.0)
            snr_freeze = None if snr_freeze_cfg is None else float(snr_freeze_cfg)
            snr_eff = snr_db_real if snr_db_real is not None else snr_db_target
            if snr_eff is None:
                g_snr0 = 1.0
            else:
                g_snr0_lin = _clamp01((float(snr_eff) - snr_low) / (snr_high - snr_low + adapt_eps))
                g_snr0 = float(g_snr0_lin ** snr_power)
            rho0_eff = float(np.clip(g_snr0, 0.0, 1.0))
            if (snr_freeze is not None) and (snr_eff is not None) and (float(snr_eff) <= float(snr_freeze)):
                rho0_eff = 0.0
            rho_aniso_map0_3d = torch.full_like(p0_3d, float(rho0_eff)).clamp(0.0, 1.0)
            eta_update_map0_3d = torch.full_like(p0_3d, float(eta0))
            eta_map0_3d = eta_update_map0_3d
            _assert_finite_torch("rho_aniso_map0_3d", rho_aniso_map0_3d)
            _assert_finite_torch("eta_update_map0_3d", eta_update_map0_3d)

        R_flat, _ = build_R_and_prior_from_cube(
            seismic_3d=seis3d,
            ai_3d=ai3d,
            well_trace_indices=well_idx,
            p_channel_3d=p0_3d,
            conf_3d=conf0_3d,
            facies_prior_3d=fac3d,
            channel_id=ch_id,
            steps_R=int(cfg.get("aniso_steps_R", 25)),
            eta=float(cfg.get("aniso_eta", 0.6)),
            rho_aniso_map_3d=rho_aniso_map0_3d,
            eta_update_map_3d=eta_update_map0_3d,
            eta_map_3d=eta_map0_3d,
            rho_skip=float(cfg.get("rho_skip", 0.10)),
            gamma=float(cfg.get("aniso_gamma", 8.0)),
            tau=float(cfg.get("aniso_tau", 0.6)),
            kappa=float(cfg.get("aniso_kappa", 4.0)),
            sigma_st=float(cfg.get("aniso_sigma_st", 1.2)),
            curr_epoch=0,
            max_epoch=int(max(1, int(cfg.get("epochs", 1000)) - 1)),
            gamma_warmup_ratio=float(cfg.get("gamma_warmup_ratio", 0.30)),
            gamma_cap_ratio=float(cfg.get("gamma_cap_ratio", 0.80)),
            use_soft_prior=get_bool(cfg, "use_soft_prior", False),
            steps_prior=int(cfg.get("aniso_steps_prior", 35)),
        )

        # 鎷兼帴R閫氶亾鍒板湴闇囨暟鎹?
        _assert_finite_torch("R_flat(raw)", R_flat)
        R_flat = R_flat.clamp(0.0, 1.0)
        _assert_finite_torch("R_flat(clamped)", R_flat)
        if debug_metrics:
            print("[ANISO] " + _np_stats("p0_3d", p0_3d.detach().cpu().numpy()))
            if rho_aniso_map0_3d is not None:
                print("[ANISO] " + _np_stats("rho0_map", rho_aniso_map0_3d.detach().cpu().numpy()))
            print("[ANISO] " + _np_stats("R_flat", R_flat.detach().cpu().numpy()))
        if R_flat.shape[0] != seismic.shape[0]:
            raise RuntimeError(f"[ANISO][ERROR] R trace count mismatch: R={R_flat.shape}, seismic={seismic.shape}")
        R_np = R_flat.detach().cpu().numpy().astype(np.float32)
        if R_np.shape[1] < n:
            raise RuntimeError(f"[ANISO][ERROR] R depth too short: R={R_np.shape}, n={n}")
        R_np = R_np[:, :n][:, np.newaxis, :]
        _assert_finite_np("R_np", R_np)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)

        # 淇濆瓨R(x)鍙鍖?
        out_prefix = f"{model_name}_{data_flag}"
        if not skip_plots:
            save_R_visualization(R_flat, meta, out_prefix, depth_slices=(40, 100, 160), out_dir=out_dir)

    print(f"[TEST] 鏈€缁堣緭鍏ョ淮搴? seismic={seismic.shape}, model={model.shape}")

    ### 8. 鏋勫缓DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traces_test = np.arange(len(model), dtype=np.int64)
    eval_trace_ids = _deterministic_eval_trace_ids(
        traces_test=traces_test,
        n_total=len(model),
        max_eval_traces=max_eval_traces,
        seed=int(cfg.get("seed", 2026)),
    )
    if eval_trace_ids.size <= 0:
        raise RuntimeError("[EVAL][ERROR] eval_trace_ids is empty.")
    print(f"[TEST] eval traces: {len(eval_trace_ids)} / {len(traces_test)}")
    print(f"[TEST] traces_test[:10]={traces_test[:10].tolist()}")
    print(f"[TEST] eval_trace_ids[:10]={eval_trace_ids[:10].tolist()}")
    if save_artifacts:
        np.save(join(out_dir, "eval_trace_ids.npy"), eval_trace_ids.astype(np.int64))
    test_dataset = SeismicDataset1D(seismic, model, eval_trace_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ### 9. 鍔犺浇妯″瀷
    inferred_run_id = _infer_run_id_for_full_ckpt(model_name, data_flag)
    in_ch = int(seismic.shape[1])
    inver_model, ckpt_path, train_p_ckpt = _load_inverse_model_strict(
        out_dir=out_dir,
        run_id=run_id,
        inferred_run_id=inferred_run_id,
        data_flag=data_flag,
        input_dim=in_ch,
        device=device,
    )
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"[CKPT][ERROR] resolved ckpt_path does not exist: {ckpt_path}")
    print(f"[TEST] model loaded strictly from full_ckpt: {ckpt_path}")
    ### 10. 鍏ㄤ綋绱犳帹鐞?
    print("[TEST] 鎺ㄧ悊涓?...")
    x0, y0 = test_dataset[0]
    H = y0.shape[-1]

    AI_pred_eval = torch.zeros((len(test_dataset), H), dtype=torch.float32, device=device)
    AI_act_eval = torch.zeros((len(test_dataset), H), dtype=torch.float32, device=device)
    y_true_full_norm = model[:, 0, :].astype(np.float32, copy=True)
    y_pred_full_norm = np.full_like(y_true_full_norm, np.nan, dtype=np.float32)

    mem = 0
    with torch.no_grad():
        inver_model.eval()
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = inver_model(x)

            bs = x.shape[0]
            AI_pred_eval[mem:mem + bs] = y_pred.squeeze(1) if y_pred.ndim == 3 else y_pred
            AI_act_eval[mem:mem + bs] = y.squeeze(1) if y.ndim == 3 else y
            tr_ids = eval_trace_ids[mem:mem + bs]
            y_pred_full_norm[tr_ids, :] = AI_pred_eval[mem:mem + bs].detach().cpu().numpy()
            mem += bs

    if mem != len(eval_trace_ids):
        raise RuntimeError(f"[EVAL][ERROR] inference count mismatch: mem={mem}, eval={len(eval_trace_ids)}")

    AI_pred_np = AI_pred_eval.detach().cpu().numpy()
    AI_act_np = AI_act_eval.detach().cpu().numpy()
    y_true_eval_norm = y_true_full_norm[eval_trace_ids, :n]
    y_pred_eval_norm = y_pred_full_norm[eval_trace_ids, :n]
    _assert_finite_np("y_true_eval_norm", y_true_eval_norm)
    _assert_finite_np("y_pred_eval_norm", y_pred_eval_norm)
    if y_true_eval_norm.shape != y_pred_eval_norm.shape:
        raise RuntimeError(f"[EVAL][ERROR] shape mismatch: true={y_true_eval_norm.shape}, pred={y_pred_eval_norm.shape}")

    out_prefix = f"{model_name}_{data_flag}"

    print("\n[TEST] inference finished")
    # For metrics, evaluate in physical impedance scale to avoid scale artifacts.
    model_mean = float(stats.get("model_mean", 0.0))
    model_std = float(stats.get("model_std", 1.0))
    AI_pred_phys = y_pred_eval_norm * model_std + model_mean
    AI_act_phys = y_true_eval_norm * model_std + model_mean
    _assert_finite_np("AI_pred_phys(before_align)", AI_pred_phys)
    _assert_finite_np("AI_act_phys", AI_act_phys)
    allow_alignment_fix = get_bool(cfg, "allow_alignment_fix", True)
    depth_failfast = get_bool(cfg, "depth_mismatch_failfast", True)
    depth_sample_traces = get_int(cfg, "sample_traces", 128)
    depth_min_gain = get_float(cfg, "min_gain", 0.30)
    depth_min_abs_pcc = get_float(cfg, "min_abs_pcc", 0.10)
    depth_seed = int(cfg.get("seed", 2026))
    depth_rep = _depth_axis_mismatch_check(
        AI_act_phys,
        AI_pred_phys,
        sample_traces=int(depth_sample_traces),
        seed=int(depth_seed),
        min_gain=float(depth_min_gain),
        min_abs_pcc=float(depth_min_abs_pcc),
    )
    with open(join(out_dir, "debug_depth_mismatch.json"), "w", encoding="utf-8") as f:
        json.dump(depth_rep, f, indent=2, ensure_ascii=False)

    rs_dbg = np.random.RandomState(int(depth_seed))
    dbg_n = int(min(5, AI_act_phys.shape[0]))
    if dbg_n > 0:
        if dbg_n < AI_act_phys.shape[0]:
            dbg_ids = rs_dbg.choice(np.arange(AI_act_phys.shape[0], dtype=np.int64), size=dbg_n, replace=False).astype(np.int64)
        else:
            dbg_ids = np.arange(dbg_n, dtype=np.int64)
        np.savez(
            join(out_dir, "debug_depth_profiles.npz"),
            trace_ids=eval_trace_ids[dbg_ids].astype(np.int64),
            y_true=AI_act_phys[dbg_ids, :].astype(np.float32),
            y_pred=AI_pred_phys[dbg_ids, :].astype(np.float32),
            y_pred_flipdepth=AI_pred_phys[dbg_ids, ::-1].astype(np.float32),
        )

    allow_apply_effective = bool(allow_alignment_fix and (not depth_failfast))
    align_pred_phys, align_report = diagnose_alignment(
        AI_act_phys,
        AI_pred_phys,
        allow_apply=allow_apply_effective,
    )
    align_report["depth_mismatch_check"] = depth_rep
    with open(join(out_dir, "debug_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(align_report, f, indent=2, ensure_ascii=False)

    if bool(depth_failfast) and bool(depth_rep.get("suspicious", False)):
        raise RuntimeError(
            "Depth axis mismatch suspected: "
            f"flip_depth improves mean PCC by {float(depth_rep.get('gain', np.nan)):.6f} "
            f"(identity={float(depth_rep.get('mean_pcc_identity', np.nan)):.6f}, "
            f"flip_depth={float(depth_rep.get('mean_pcc_flipdepth', np.nan)):.6f})."
        )

    chosen_transform = str(align_report.get("chosen_transform", "identity"))
    metrics_pred_phys = AI_pred_phys
    metrics_pred_norm = y_pred_eval_norm
    if bool(align_report.get("auto_applied", False)):
        print(f"[ALIGN-FIX] auto-applied transform: {chosen_transform}")
        metrics_pred_phys = align_pred_phys
        metrics_pred_norm = _apply_alignment_transform(y_pred_eval_norm, chosen_transform)
    _assert_finite_np("AI_pred_phys(after_align)", metrics_pred_phys)

    if debug_metrics:
        print("[METRIC-DEBUG] " + _np_stats("y_true(norm)", y_true_eval_norm))
        print("[METRIC-DEBUG] " + _np_stats("y_pred(norm)", metrics_pred_norm))
        print("[METRIC-DEBUG] " + _np_stats("y_true(phys)", AI_act_phys))
        print("[METRIC-DEBUG] " + _np_stats("y_pred(phys)", metrics_pred_phys))

    print("\n[TEST] inference done, computing metrics")
    metrics = _compute_main_metrics(AI_act_phys, metrics_pred_phys)
    pcc_p = pearsonr(AI_act_phys.ravel(), metrics_pred_phys.ravel())[1] if np.isfinite(metrics["PCC"]) else np.nan

    print(f"  R2        : {float(metrics['R2']):.4f}")
    print(f"  PCC       : {float(metrics['PCC']):.4f} (p-value: {float(pcc_p):.2e})")
    print(f"  SSIM      : {float(metrics['SSIM']):.4f}")
    print(f"  MSE       : {float(metrics['MSE']):.6f}")
    print(f"  MAE       : {float(metrics['MAE']):.6f}")
    print(f"  MedAE     : {float(metrics['MedAE']):.6f}")
    print(f"  PSNR      : {float(metrics['PSNR']):.4f} dB")
    if np.isfinite(metrics["R2"]):
        print(f"  MSE vs (1-R2): {float(metrics['MSE']):.6f} vs {float(1.0-metrics['R2']):.6f}")
    if debug_metrics:
        print(f"  SSE       : {float(metrics.get('SSE', np.nan)):.6f}")
        print(f"  SST       : {float(metrics.get('SST', np.nan)):.6f}")

    align_report["eval_trace_count"] = int(len(eval_trace_ids))
    align_report["traces_test_head10"] = traces_test[:10].astype(int).tolist()
    align_report["eval_trace_ids_head10"] = eval_trace_ids[:10].astype(int).tolist()
    align_report["y_true_eval_stats"] = {
        "min": float(np.min(AI_act_phys)),
        "max": float(np.max(AI_act_phys)),
        "mean": float(np.mean(AI_act_phys)),
        "std": float(np.std(AI_act_phys)),
    }
    align_report["y_pred_eval_stats"] = {
        "min": float(np.min(metrics_pred_phys)),
        "max": float(np.max(metrics_pred_phys)),
        "mean": float(np.mean(metrics_pred_phys)),
        "std": float(np.std(metrics_pred_phys)),
    }
    align_report["metrics_after_alignment"] = {
        "R2": float(metrics.get("R2", np.nan)),
        "PCC": float(metrics.get("PCC", np.nan)),
        "MSE": float(metrics.get("MSE", np.nan)),
        "MAE": float(metrics.get("MAE", np.nan)),
        "SSE": float(metrics.get("SSE", np.nan)),
        "SST": float(metrics.get("SST", np.nan)),
    }

    seismic_eval_input = np.asarray(seismic[eval_trace_ids, :, :n], dtype=np.float32)
    y_true_eval = np.asarray(AI_act_phys, dtype=np.float32)
    y_pred_eval = np.asarray(metrics_pred_phys, dtype=np.float32)
    eval_meta = {
        "ckpt_path": str(ckpt_path),
        "ckpt_sha1": sha1_of_file(str(ckpt_path)),
        "seismic_path": str(seismic_override_path) if seismic_override_path else join("data", data_flag, "synth_40HZ.npy"),
        "seismic_sha1": sha1_of_array(seismic_eval_input),
        "y_true_sha1": sha1_of_array(y_true_eval),
        "y_pred_sha1": sha1_of_array(y_pred_eval),
        "eval_trace_ids_sha1": sha1_of_array(eval_trace_ids.astype(np.int64)),
        "n_samples": int(n),
        "scale_flag": "phys",
        "stats_true": stats_of_array(y_true_eval),
        "stats_pred": stats_of_array(y_pred_eval),
        "noise_tag": noise_tag if noise_tag else "unknown",
        "alignment_applied": bool(align_report.get("auto_applied", False)),
        "alignment_transform": str(align_report.get("chosen_transform", "identity")),
    }
    with open(join(out_dir, "eval_meta.json"), "w", encoding="utf-8") as f:
        json.dump(eval_meta, f, indent=2, ensure_ascii=False)

    metrics_main = {
        "MAE": float(metrics.get("MAE", np.nan)),
        "R2": float(metrics.get("R2", np.nan)),
        "PCC": float(metrics.get("PCC", np.nan)),
        "SSIM": float(metrics.get("SSIM", np.nan)),
        "PSNR": float(metrics.get("PSNR", np.nan)),
        "MSE": float(metrics.get("MSE", np.nan)),
        "MedAE": float(metrics.get("MedAE", np.nan)),
        "SSE": float(metrics.get("SSE", np.nan)),
        "SST": float(metrics.get("SST", np.nan)),
        "alignment_applied": bool(align_report.get("auto_applied", False)),
        "alignment_transform": str(align_report.get("chosen_transform", "identity")),
    }
    with open(join(out_dir, "metrics_main.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_main, f, indent=2, ensure_ascii=False)
    if save_artifacts:
        np.save(join(out_dir, "results", f"{out_prefix}_pred_AI.npy"), metrics_pred_norm.astype(np.float32))
        np.save(join(out_dir, "results", f"{out_prefix}_true_AI.npy"), y_true_eval_norm.astype(np.float32))
        np.save(join(out_dir, "results", f"{out_prefix}_eval_trace_ids.npy"), eval_trace_ids.astype(np.int64))
        np.save(join(out_dir, "results", f"{out_prefix}_metrics.npy"), metrics_main, allow_pickle=True)

    facies_map = cfg.get("facies_name_map", None)
    if isinstance(facies_map, dict):
        facies_name_map = {int(k): str(v) for k, v in facies_map.items()}
    else:
        facies_name_map = {0: "boundary", 1: "interbay", 2: "channel", 3: "pointbar"}
    facies_eval = facies_raw[eval_trace_ids, :n]
    if save_artifacts:
        facies_rows, fac_csv, fac_md, fac_json = save_facies_metrics(
            ai_true=AI_act_phys,
            ai_pred=metrics_pred_phys,
            facies_flat=facies_eval,
            out_dir=out_dir,
            out_prefix=out_prefix,
            facies_name_map=facies_name_map,
        )
        metrics["facies_metrics"] = facies_rows
        print(f"[TEST] facies metrics saved: {fac_csv}")
        print(f"[TEST] facies metrics saved: {fac_md}")
        print(f"[TEST] facies metrics saved: {fac_json}")

    can_plot_full = int(len(eval_trace_ids)) == int(meta["inline"] * meta["xline"])
    if (not can_plot_full) and (not skip_plots):
        print(
            f"[PLOT][WARN] skip cube plots because eval traces are subset: "
            f"{len(eval_trace_ids)} != {int(meta['inline'] * meta['xline'])}"
        )
    if data_flag == "Stanford_VI" and (not skip_plots) and can_plot_full:
        show_stanford_vi(
            AI_act_flat=AI_act_np,
            AI_pred_flat=AI_pred_np,
            seismic_flat=seismic,
            meta=meta,
            out_prefix=out_prefix,
            xline_pick=50,
            inline_pick=100,
            depth_slices=(40, 100, 160),
            out_dir=out_dir,
        )
        save_residual_maps(
            ai_true_flat=AI_act_np,
            ai_pred_flat=AI_pred_np,
            meta=meta,
            out_prefix=out_prefix,
            out_dir=out_dir,
            xline_pick=50,
            inline_pick=100,
            depth_slices=(40, 100, 160),
        )
        print(f"\n[TEST] 鍙鍖栫粨鏋滃凡淇濆瓨鍒?results/ 鐩綍锛屽墠缂€: {out_prefix}")

    return metrics



if __name__ == "__main__":
    test(TCN1D_test_p)


