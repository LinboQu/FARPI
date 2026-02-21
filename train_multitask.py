"""
train_multitask.py (UPDATED)

Key upgrades (no network changes)
--------------------------------
1) Robust stats/ckpt binding:
   - Save FULL checkpoint (state_dict + stats + train_p) every run.
2) Use real well locations as seeds:
   - Read selected_wells_20_seed2026.csv (INLINE/XLINE) and convert to trace indices.
3) Facies-adaptive anisotropic conditioning R(x) that can be UPDATED iteratively:
   - Initial R uses facies prior (for Stanford VI-E we use Facies.npy as prior).
   - Optionally refresh R every R_update_every epochs using predicted facies probabilities
     + physics residual damping, and EMA update to stabilize.

Notes:
- This file assumes utils/datasets.py returns CPU tensors. We move tensors to GPU in the loop.
- Default behavior remains compatible if you keep iterative_R=False.
"""

import os
import csv
import json
import errno
import argparse
import random
from os.path import join
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.CNN2Layer import VishalNet
from model.tcn import TCN_IV_1D_C
from model.M2M_LSTM import GRU_MM
from model.Unet_1D import Unet_1D
from model.Transformer import TransformerModel
from model.Forward import forward_model_0, forward_model_1, forward_model_2
from model.geomorphology_classification import Facies_model_class

from setting import *
from utils.utils import standardize
from utils.datasets import SeismicDataset1D, SeismicDataset1D_SPF, SeismicDataset1D_SPF_WS
from utils.reliability_aniso import build_R_and_prior_from_cube
from utils.config_cast import get_bool, get_float, get_int, get_str


# -----------------------------
# helpers
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _summary_stats_np(x: np.ndarray):
    return float(np.mean(x)), float(np.min(x)), float(np.max(x))


def _stats_dict_np(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _assert_finite_torch(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int(x.numel() - int(torch.isfinite(x).sum().item()))
        raise RuntimeError(f"[FINITE][ERROR] {name} contains non-finite values: bad={bad}")


def _set_reproducibility(seed: int, cfg: dict) -> None:
    deterministic = get_bool(cfg, "deterministic", True)
    use_det_algo = get_bool(cfg, "use_deterministic_algorithms", True)
    det_warn_only = get_bool(cfg, "deterministic_warn_only", True)
    cudnn_det = get_bool(cfg, "cudnn_deterministic", True)
    cudnn_bench = get_bool(cfg, "cudnn_benchmark", False)
    set_cublas = get_bool(cfg, "set_cublas_workspace_config", True)
    cublas_cfg = get_str(cfg, "cublas_workspace_config", ":4096:8")
    set_pyhash = get_bool(cfg, "set_pythonhashseed", True)
    disable_tf32 = get_bool(cfg, "disable_tf32", True)

    if set_pyhash:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    if set_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_cfg

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.deterministic = bool(cudnn_det)
    torch.backends.cudnn.benchmark = bool(cudnn_bench)
    if disable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if use_det_algo:
        try:
            torch.use_deterministic_algorithms(True, warn_only=det_warn_only)
        except Exception as e:
            if deterministic:
                raise RuntimeError(f"Failed to enable deterministic algorithms: {e}") from e
            print(f"[REPRO][WARN] torch.use_deterministic_algorithms(True, warn_only={int(det_warn_only)}) failed: {e}")
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def load_selected_wells_trace_indices(
    csv_path: str | None,
    IL: int,
    XL: int,
    no_wells: int,
    seed: int = 2026,
) -> np.ndarray:
    """
    Convert (INLINE, XLINE) in CSV to flattened trace indices (inline * XL + xline).
    If csv_path is missing, fallback to uniform linspace sampling (legacy).
    """
    if csv_path is None or (not os.path.isfile(csv_path)):
        # fallback: keep legacy behavior
        return np.linspace(0, IL * XL - 1, int(no_wells), dtype=np.int64)

    ils: list[int] = []
    xls: list[int] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or ("INLINE" not in reader.fieldnames) or ("XLINE" not in reader.fieldnames):
            raise ValueError(f"CSV must contain INLINE,XLINE columns. Got: {reader.fieldnames}")
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
    traces = np.unique((il * XL + xl).astype(np.int64))

    # enforce count
    if len(traces) != int(no_wells):
        rng = np.random.default_rng(int(seed))
        if len(traces) > int(no_wells):
            traces = rng.choice(traces, size=int(no_wells), replace=False).astype(np.int64)
        else:
            print(f"[WELLS][WARN] CSV has {len(traces)} wells < requested {no_wells}. Using {len(traces)}.")

    return traces


def get_data_SPF(no_wells=10, data_flag="Stanford_VI", get_F=0, seismic_override_path: str | None = None):
    """
    Read Stanford VI / Fanny raw cubes and standardize using global_model stats.
    Returns:
      seismic: (N,1,H) float32
      model  : (N,1,H) float32
      facies : (N,1,H) int64
      meta   : dict(H, inline, xline, seismic3d, model3d, facies3d)
      stats  : dict(mean/std etc) used for standardize
    """
    meta = {}

    if data_flag == "Stanford_VI":
        seismic_path = seismic_override_path if seismic_override_path else join("data", data_flag, "synth_40HZ.npy")
        seismic3d = np.load(seismic_path)  # (H,IL,XL)
        model3d = np.load(join("data", data_flag, "AI.npy"))
        facies3d = np.load(join("data", data_flag, "Facies.npy"))

        H, IL, XL = seismic3d.shape
        meta = {"H": H, "inline": IL, "xline": XL, "seismic3d": seismic3d, "model3d": model3d, "facies3d": facies3d}

        seismic = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))
        model = np.transpose(model3d.reshape(H, IL * XL), (1, 0))
        facies = np.transpose(facies3d.reshape(H, IL * XL), (1, 0))

        print(f"[{data_flag}] seismic source: {seismic_path}")
        print(f"[{data_flag}] raw shapes: model={model.shape}, seismic={seismic.shape}, facies={facies.shape}")
        print(f"[{data_flag}] raw means : model={float(model.mean()):.4f}, seismic={float(seismic.mean()):.4f}")

    elif data_flag == "Fanny":
        seismic = np.load(join("data", data_flag, "seismic.npy"))
        model = np.load(join("data", data_flag, "impedance.npy"))
        facies = np.load(join("data", data_flag, "facies.npy"))
        H = model.shape[-1]
        n_traces = model.shape[0]
        IL = XL = int(np.sqrt(n_traces))
        meta = {"H": H, "inline": IL, "xline": XL}

    else:
        raise ValueError(f"Unsupported data_flag: {data_flag}")

    # standardize (global_model) and return stats
    seismic, model, stats = standardize(seismic, model, no_wells=no_wells, mode="global_model")

    # crop to multiple of 8 (for UNet-like downsampling)
    s_L = seismic.shape[-1]
    n = int((s_L // 8) * 8)
    seismic = seismic[:, :n]
    model = model[:, :n]
    facies = facies[:, :n]

    return (
        seismic[:, np.newaxis, :].astype(np.float32),
        model[:, np.newaxis, :].astype(np.float32),
        facies[:, np.newaxis, :].astype(np.int64),
        meta,
        stats,
    )


# -----------------------------
# main train
# -----------------------------
def train(train_p: dict):
    # pick model classes
    model_name = train_p["model_name"]
    Forward_model = train_p["Forward_model"]
    Facies_model_C = train_p["Facies_model"]

    if model_name == "tcnc":
        choice_model = TCN_IV_1D_C
    elif model_name == "VishalNet":
        choice_model = VishalNet
    elif model_name == "GRU_MM":
        choice_model = GRU_MM
    elif model_name == "Unet_1D":
        choice_model = Unet_1D
    elif model_name == "Transformer":
        choice_model = TransformerModel
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if Forward_model == "cnn":
        forward = forward_model_0
    elif Forward_model == "convolution":
        forward = forward_model_1
    elif Forward_model == "cov_para":
        forward = forward_model_2
    else:
        raise ValueError(f"Unknown Forward_model: {Forward_model}")

    if Facies_model_C != "Facies":
        raise ValueError(f"Unknown Facies model: {Facies_model_C}")
    Facies_class = Facies_model_class

    data_flag = train_p["data_flag"]
    no_wells = int(train_p.get("no_wells", 20))
    seed = int(train_p.get("seed", 2026))
    selected_wells_csv = train_p.get("selected_wells_csv", None)
    out_dir = train_p.get("out_dir", ".")
    run_id = train_p.get("run_id", f"{model_name}_{Forward_model}_{Facies_model_C}")
    _set_reproducibility(seed=seed, cfg=train_p)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(out_dir)
    _ensure_dir(join(out_dir, "save_train_model"))
    _ensure_dir(join(out_dir, "results"))

    # data
    seismic, model, facies, meta, stats = get_data_SPF(
        no_wells=no_wells,
        data_flag=data_flag,
        get_F=train_p.get("get_F", 0),
        seismic_override_path=train_p.get("seismic_override_path"),
    )

    # save stats (both legacy + run-specific)
    np.save(join(out_dir, "save_train_model", f"norm_stats_{data_flag}.npy"), stats)  # legacy (may be overwritten)
    np.save(join(out_dir, "save_train_model", f"norm_stats_{run_id}_{data_flag}.npy"), stats)  # strong binding
    print(f"[NORM] saved stats: norm_stats_{run_id}_{data_flag}.npy (and legacy norm_stats_{data_flag}.npy)")
    print(
        f"[ABLATION] run_id={run_id} | out_dir={out_dir} | "
        f"lambda_recon={float(train_p.get('lambda_recon', 1.0))} | "
        f"lambda_recon_ws={float(train_p.get('lambda_recon_ws', train_p.get('lambda_recon', 1.0)))} | "
        f"ws_warmup_epochs={int(train_p.get('ws_warmup_epochs', 0))} | "
        f"use_aniso_conditioning={int(get_bool(train_p, 'use_aniso_conditioning', False))} | "
        f"aniso_gamma={float(train_p.get('aniso_gamma', 8.0))} | "
        f"iterative_R={int(get_bool(train_p, 'iterative_R', False))}"
    )
    if get_bool(train_p, "log_reproducibility", True):
        print(
            f"[REPRO] seed={seed} deterministic={int(get_bool(train_p, 'deterministic', True))} "
            f"use_det_alg={int(get_bool(train_p, 'use_deterministic_algorithms', True))} "
            f"cudnn_det={int(get_bool(train_p, 'cudnn_deterministic', True))} "
            f"cudnn_benchmark={int(get_bool(train_p, 'cudnn_benchmark', False))} "
            f"loader_num_workers={int(train_p.get('loader_num_workers', train_p.get('num_workers', 0)))} "
            f"persistent_workers={int(get_bool(train_p, 'persistent_workers', False))} "
            f"cublas_ws={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')} "
            f"pythonhashseed={os.environ.get('PYTHONHASHSEED', '')}"
        )

    # -------- noise- & uncertainty-aware adaptive eta (proposed_v2) --------
    adaptive_eta_enable = get_bool(train_p, "adaptive_eta_enable", False)
    eta0 = get_float(train_p, "aniso_eta", 0.6)
    eta_min = get_float(train_p, "eta_min", 0.0)
    alpha_floor_ratio_cfg = train_p.get("alpha_floor_ratio", None)
    eta_floor_ratio_cfg = train_p.get("eta_floor_ratio", None)
    eta_floor_ratio_used_as_alpha_floor_ratio = 0
    if alpha_floor_ratio_cfg is not None:
        alpha_floor_ratio = float(alpha_floor_ratio_cfg)
    elif eta_floor_ratio_cfg is not None:
        alpha_floor_ratio = float(eta_floor_ratio_cfg)
        eta_floor_ratio_used_as_alpha_floor_ratio = 1
    else:
        alpha_floor_ratio = 0.20
    snr_power = get_float(train_p, "snr_power", 2.0)
    ent_power = get_float(train_p, "ent_power", 2.0)
    ch_power = get_float(train_p, "ch_power", 2.0)
    rho_cut = get_float(train_p, "rho_cut", 0.30)
    phys_power = get_float(train_p, "phys_power", 0.5)
    tau = get_float(train_p, "tau", 0.60)
    s = max(get_float(train_p, "s", 0.08), 1e-8)
    k = get_float(train_p, "k", 1.0)
    enable_margin_gate = get_bool(train_p, "enable_margin_gate", True)
    margin0 = get_float(train_p, "margin0", 0.15)
    margin_s = get_float(train_p, "margin_s", 0.05)
    if margin_s <= 0.0:
        raise ValueError(f"margin_s must be > 0, got {margin_s}")
    rho_warmup_ratio = get_float(train_p, "rho_warmup_ratio", 0.35)
    gamma_warmup_ratio = get_float(train_p, "gamma_warmup_ratio", 0.30)
    gamma_cap_ratio = get_float(train_p, "gamma_cap_ratio", 0.50)
    w_ent = get_float(train_p, "w_ent", 0.5)
    w_phys = get_float(train_p, "w_phys", 0.5)
    w_sum = max(1e-8, w_ent + w_phys)
    w_ent = float(w_ent / w_sum)
    w_phys = float(w_phys / w_sum)
    alpha_update_mode = str(train_p.get("alpha_update_mode", "decoupled")).strip().lower()
    eta_update_mode = str(train_p.get("eta_update_mode", "decoupled")).strip().lower()
    if alpha_update_mode not in {"decoupled", "coupled"}:
        raise ValueError(f"alpha_update_mode must be decoupled/coupled, got {alpha_update_mode}")
    if eta_update_mode not in {"decoupled", "coupled"}:
        raise ValueError(f"eta_update_mode must be decoupled/coupled, got {eta_update_mode}")
    rho_skip = get_float(train_p, "rho_skip", 0.10)
    snr_freeze_cfg = train_p.get("snr_freeze", 12.0)
    snr_freeze = None if snr_freeze_cfg is None else float(snr_freeze_cfg)
    snr_cap_by_residual = get_bool(train_p, "snr_cap_by_residual", True)
    snr_low = get_float(train_p, "snr_low", 10.0)
    snr_high = get_float(train_p, "snr_high", 30.0)
    adapt_eps = get_float(train_p, "adaptive_eps", 1e-8)
    residual_ema_decay = get_float(train_p, "residual_ema_decay", 0.9)
    residual_pctl = get_float(train_p, "residual_pctl", 95.0)
    debug_sanity = get_bool(train_p, "debug_sanity", False)
    debug_sanity_every = get_int(train_p, "debug_sanity_every", 50)
    debug_aniso_stats = get_bool(train_p, "debug_aniso_stats", False)
    debug_aniso_every = get_int(train_p, "debug_aniso_every", 200)
    debug_aniso_out = train_p.get("debug_aniso_out", None)
    debug_aniso_path = str(debug_aniso_out).strip() if (debug_aniso_out is not None) else join(out_dir, "debug_aniso_stats.jsonl")
    eta_floor_ratio_update_cfg = train_p.get("eta_floor_ratio", None)
    eta_floor_ratio_update = 0.25 if eta_floor_ratio_update_cfg is None else float(eta_floor_ratio_update_cfg)
    eta_floor_ratio_update = float(np.clip(eta_floor_ratio_update, 0.0, 1.0))
    eta_log_every = get_int(train_p, "eta_log_every", 1)
    eta_warn_ratio_thr = get_float(train_p, "eta_warn_ratio_thr", 0.1)
    deterministic_flag = int(get_bool(train_p, "deterministic", True))
    loader_num_workers_cfg = get_int(train_p, "loader_num_workers", get_int(train_p, "num_workers", 0))
    cudnn_det_flag = int(get_bool(train_p, "cudnn_deterministic", True))
    cudnn_bench_flag = int(get_bool(train_p, "cudnn_benchmark", False))
    use_det_alg_flag = int(get_bool(train_p, "use_deterministic_algorithms", True))
    cublas_ws_cfg = str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", train_p.get("cublas_workspace_config", "")))
    pythonhashseed_cfg = str(os.environ.get("PYTHONHASHSEED", ""))
    resolved_cfg = {
        "enable_margin_gate": bool(enable_margin_gate),
        "rho_warmup_ratio": float(rho_warmup_ratio),
        "gamma_cap_ratio": float(gamma_cap_ratio),
        "eta_floor_ratio": float(eta_floor_ratio_update),
        "ent_power": float(ent_power),
        "ch_power": float(ch_power),
        "rho_cut": float(rho_cut),
        "snr_power": float(snr_power),
        "debug_aniso_stats": bool(debug_aniso_stats),
        "debug_aniso_every": int(debug_aniso_every),
    }
    print("[CFG-RESOLVED] " + " | ".join([f"{k}={v}" for k, v in resolved_cfg.items()]))
    try:
        if out_dir and os.path.isdir(out_dir):
            with open(join(out_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
                json.dump(resolved_cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CFG-RESOLVED][WARN] failed to write resolved_config.json: {e}")

    # best-effort SNR source from noise_config.json (noise ablation pipeline)
    noise_cfg_path = join(out_dir, "noise_config.json")
    snr_db_real = None
    snr_db_target = None
    if os.path.isfile(noise_cfg_path):
        try:
            with open(noise_cfg_path, "r", encoding="utf-8") as f:
                ncfg = json.load(f)
            v = ncfg.get("snr_db_real", None)
            if v is not None:
                snr_db_real = float(v)
            vt = ncfg.get("snr_db_target", None)
            if vt is not None:
                snr_db_target = float(vt)
        except Exception:
            snr_db_real = None
            snr_db_target = None

    eta_log_path = join(out_dir, "results", f"{run_id}_{data_flag}_eta_adapt_log.csv")
    if adaptive_eta_enable:
        print(
            f"[ETA-ADAPT] enabled=True | eta0={eta0:.4f} eta_min={eta_min:.4f} "
            f"alpha_floor_ratio={alpha_floor_ratio:.3f} eta_floor_alias_used={eta_floor_ratio_used_as_alpha_floor_ratio} "
            f"snr_low/high=({snr_low:.2f},{snr_high:.2f}) "
            f"snr_power={snr_power:.3f} ent_power={ent_power:.3f} ch_power={ch_power:.3f} phys_power={phys_power:.3f} "
            f"tau={tau:.3f} s={s:.3f} k={k:.3f} "
            f"enable_margin_gate={int(enable_margin_gate)} margin0={margin0:.3f} margin_s={margin_s:.3f} rho_cut={rho_cut:.3f} "
            f"rho_warmup_ratio={rho_warmup_ratio:.3f} "
            f"gamma_warmup_ratio={gamma_warmup_ratio:.3f} gamma_cap_ratio={gamma_cap_ratio:.3f} "
            f"eta_floor_ratio_update={eta_floor_ratio_update:.3f} "
            f"w_ent={w_ent:.3f} w_phys={w_phys:.3f} "
            f"alpha_update_mode={alpha_update_mode} eta_update_mode={eta_update_mode} "
            f"rho_skip={rho_skip:.3f} snr_freeze={snr_freeze} residual_pctl={residual_pctl:.1f} "
            f"residual_ema_decay={residual_ema_decay:.3f} noise_cfg_snr_real={snr_db_real} "
            f"noise_cfg_snr_target={snr_db_target} snr_cap_by_residual={int(snr_cap_by_residual)}"
        )
        with open(eta_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "snr_db_source",
                    "snr_db_value",
                    "freeze_triggered",
                    "g_snr_lin_mean",
                    "g_snr_lin_min",
                    "g_snr_lin_max",
                    "g_snr_mean",
                    "g_snr_min",
                    "g_snr_max",
                    "H_norm_mean",
                    "H_norm_min",
                    "H_norm_max",
                    "g_ent_mean",
                    "g_ent_min",
                    "g_ent_max",
                    "r_phys_norm_mean",
                    "r_phys_norm_min",
                    "r_phys_norm_max",
                    "g_phys_mean",
                    "g_phys_min",
                    "g_phys_max",
                    "rho_aniso_mean",
                    "rho_aniso_min",
                    "rho_aniso_max",
                    "alpha_update_mean",
                    "alpha_update_min",
                    "alpha_update_max",
                    "ratio_alpha_le_rho_skip",
                    "freeze_aniso_triggered",
                    "eta_floor",
                    "eta_floor_ratio",
                    "eta_floor_ratio_used_as_alpha_floor_ratio",
                    "eta_mean",
                    "eta_min",
                    "eta_max",
                    "rho_mean",
                    "rho_min",
                    "rho_max",
                    "ratio_rho_le_rho_skip",
                    "ratio_eta_lt_0p1eta0",
                    "p_channel_mean",
                    "p_channel_min",
                    "p_channel_max",
                    "g_ch_mean",
                    "g_ch_min",
                    "g_ch_max",
                    "rho_snr_mean",
                    "g_ent_mean_rho",
                    "rho_aniso_mean_gate",
                    "rho_aniso_min_gate",
                    "rho_aniso_max_gate",
                    "snr_db_config_value",
                    "snr_db_residual_value",
                    "snr_db_eff_value",
                    "snr_is_capped",
                    "p_other_max_mean",
                    "p_other_max_min",
                    "p_other_max_max",
                    "margin_mean",
                    "margin_min",
                    "margin_max",
                    "g_dom_mean",
                    "g_dom_min",
                    "g_dom_max",
                    "g_ch_final_mean",
                    "g_ch_final_min",
                    "g_ch_final_max",
                    "rho_warmup_ratio_used",
                    "rho_warmup_progress",
                    "rho_warmup_w",
                    "rho_aniso_mean_pre_warmup",
                    "rho_aniso_min_pre_warmup",
                    "rho_aniso_max_pre_warmup",
                    "alpha_update_mode",
                    "eta_update_mode",
                    "seed",
                    "deterministic",
                    "loader_num_workers",
                    "cudnn_deterministic",
                    "cudnn_benchmark",
                    "use_deterministic_algorithms",
                    "cublas_workspace_config",
                    "pythonhashseed",
                ]
            )

    # -----------------------------
    # Wells (keep identical across ablations)
    # -----------------------------
    if data_flag == "Stanford_VI":
        IL, XL = int(meta["inline"]), int(meta["xline"])
        traces_train = load_selected_wells_trace_indices(selected_wells_csv, IL, XL, no_wells=no_wells, seed=seed)
        print(f"[WELLS] using wells: {selected_wells_csv} | count={len(traces_train)}")
    else:
        traces_train = np.linspace(0, len(model) - 1, no_wells, dtype=int)
        print(f"[WELLS] fallback linspace wells | count={len(traces_train)}")
    np.save(join(out_dir, "results", f"{run_id}_{data_flag}_well_trace_indices.npy"), traces_train)

    # -----------------------------
    # Build / maintain anisotropic R(x)
    # -----------------------------
    R_prev_flat = None
    prior_np = None

    if get_bool(train_p, "use_aniso_conditioning", False) and data_flag == "Stanford_VI":

        # build initial R from facies PRIOR (Stanford VI: Facies.npy is available; in real: interpreter prior)
        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device)
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        # ---- NEW: provide p_channel_3d + conf_3d for init (do NOT rely on facies_3d truth) ----
        ch_id = int(train_p.get("channel_id", 2))
        p0_3d = (fac_prior3d == ch_id).float()          # [H,IL,XL] in {0,1}
        conf0_3d = torch.ones_like(p0_3d)               # [H,IL,XL] all confident
        _assert_finite_torch("p0_3d", p0_3d)
        _assert_finite_torch("conf0_3d", conf0_3d)
        
        eta_map0_3d = None
        rho_aniso_map0_3d = None
        eta_update_map0_3d = None
        if adaptive_eta_enable:
            if snr_db_real is not None:
                g_snr0_lin = _clamp01((float(snr_db_real) - snr_low) / (snr_high - snr_low + adapt_eps))
                g_snr0 = float(g_snr0_lin ** snr_power)
            else:
                g_snr0_lin = 1.0
                g_snr0 = 1.0
            rho0_eff = float(np.clip(g_snr0, 0.0, 1.0))
            if (snr_freeze is not None) and (snr_db_real is not None) and (float(snr_db_real) <= float(snr_freeze)):
                rho0_eff = 0.0
            rho_aniso_map0_3d = torch.full_like(p0_3d, float(rho0_eff)).clamp(0.0, 1.0)
            eta_update_map0_3d = torch.full_like(p0_3d, float(eta0))
            eta_map0_3d = eta_update_map0_3d
            _assert_finite_torch("rho_aniso_map0_3d", rho_aniso_map0_3d)
            _assert_finite_torch("eta_update_map0_3d", eta_update_map0_3d)

        R_prev_flat, prior_flat = build_R_and_prior_from_cube(
            seismic_3d=seis3d,
            ai_3d=ai3d,
            well_trace_indices=well_idx,
            
            # ✅ key: provide one of the required "channel-likeness" sources
            p_channel_3d=p0_3d,
            conf_3d=conf0_3d,
            
            # anchor prior (still used as your geological prior)
            facies_prior_3d=fac_prior3d,
            
            channel_id=ch_id,
            alpha_prior=1.0,      # initial: pure prior
            conf_thresh=0.0,
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            rho_aniso_map_3d=rho_aniso_map0_3d,
            eta_update_map_3d=eta_update_map0_3d,
            eta_map_3d=eta_map0_3d,
            rho_skip=float(rho_skip),
            gamma=float(train_p.get("aniso_gamma", 8.0)),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            curr_epoch=0,
            max_epoch=int(max(1, int(train_p.get("epochs", 1000)) - 1)),
            gamma_warmup_ratio=float(gamma_warmup_ratio),
            gamma_cap_ratio=float(gamma_cap_ratio),
            use_soft_prior=get_bool(train_p, "use_soft_prior", False),
            steps_prior=int(train_p.get("aniso_steps_prior", 35)),
        )

        # append R as extra channel
        R_np = R_prev_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)
        prior_np = prior_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32) if prior_flat is not None else None

        # log stats
        r_np = R_prev_flat.detach().cpu().numpy()
        print(f"[R0] mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")
    else:
        print("[R0] mean=0.0000 max=0.0000 ratio(R>0.5)=0.0000 (disabled)")

    # datasets / loaders
    train_dataset = SeismicDataset1D_SPF(seismic, model, facies, traces_train)

    num_workers = int(train_p.get("loader_num_workers", train_p.get("num_workers", 0)))
    pin_memory = get_bool(train_p, "pin_memory", torch.cuda.is_available())
    persistent_workers = get_bool(train_p, "persistent_workers", True) and (num_workers > 0)
    train_gen = torch.Generator()
    train_gen.manual_seed(int(seed))
    ws_gen = torch.Generator()
    ws_gen.manual_seed(int(seed) + 1)
    val_gen = torch.Generator()
    val_gen.manual_seed(int(seed) + 2)

    def _seed_worker(worker_id: int):
        worker_seed = (int(seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    worker_init = _seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init,
        generator=train_gen,
        drop_last=False,
    )

    # Weak supervision loader (scheme A scheduling)
    u = train_p.get("unsupervised_seismic", None)
    if u is None or int(u) <= 0:
        ws_traces = np.arange(len(model), dtype=int)
    else:
        ws_traces = np.linspace(0, len(model) - 1, int(u), dtype=int)

    Wsupervised_dataset = SeismicDataset1D_SPF_WS(seismic, facies, ws_traces, prior=prior_np)
    Wsupervised_loader = DataLoader(
        Wsupervised_dataset,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init,
        generator=ws_gen,
        drop_last=False,
    )

    # validation (small subset)
    traces_validation = np.linspace(0, len(model) - 1, 3, dtype=int)
    val_dataset = SeismicDataset1D(seismic, model, traces_validation)
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0,
        worker_init_fn=None,
        generator=val_gen,
    )

    # models
    in_ch = int(seismic.shape[1])
    try:
        inverse_model = choice_model(input_dim=in_ch).to(device)
    except TypeError:
        try:
            inverse_model = choice_model(in_ch).to(device)
        except TypeError:
            inverse_model = choice_model().to(device)

    forward_model = forward().to(device)
    Facies_model = Facies_class(facies_n=int(train_p.get("facies_n", 4))).to(device)

    # losses
    criterion_ai = torch.nn.MSELoss()
    criterion_rec = torch.nn.MSELoss()
    criterion_facies = nn.CrossEntropyLoss()

    lam_ai = float(train_p.get("lambda_ai", 5.0))
    lam_fac = float(train_p.get("lambda_facies", 0.2))
    lam_rec = float(train_p.get("lambda_recon", 1.0))
    # Weak-supervision reconstruction is usually much denser than well supervision.
    # Keep an explicit knob to avoid collapsing to reconstruction-only behavior.
    lam_rec_ws = float(train_p.get("lambda_recon_ws", lam_rec))
    ws_warmup_epochs = int(train_p.get("ws_warmup_epochs", 0))
    facies_detach_y = get_bool(train_p, "facies_detach_y", True)
    grad_clip = float(train_p.get("grad_clip", 0.0))

    optimizer = torch.optim.Adam(
        list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
        weight_decay=float(train_p.get("weight_decay", 1e-4)),
        lr=float(train_p.get("lr", 1e-4)),
    )

    # weak supervision scheduling
    ws_every = int(train_p.get("ws_every", 5))
    ws_max_batches = int(train_p.get("ws_max_batches", 50))

    # iterative R scheduling (recommended)
    iterative_R = get_bool(train_p, "iterative_R", False) and (R_prev_flat is not None)
    R_update_every = int(train_p.get("R_update_every", 50))
    R_ema_beta = float(train_p.get("R_ema_beta", 0.85))
    alpha_start = float(train_p.get("alpha_prior_start", 1.0))
    alpha_end = float(train_p.get("alpha_prior_end", 0.3))
    alpha_decay_epochs = int(train_p.get("alpha_prior_decay_epochs", max(1, train_p.get("epochs", 1000))))
    conf_thresh = float(train_p.get("conf_thresh", 0.75))
    lambda_phys_damp = float(train_p.get("lambda_phys_damp", 0.0))
    residual_scale_ema = None  # running robust scale for residual normalization

    def _snr_db_from_residual_proxy(sig_np: np.ndarray, pres_np: np.ndarray, eps: float) -> float:
        p_sig = float(np.mean(sig_np ** 2) + eps)
        p_noise = float(np.mean(pres_np ** 2) + eps)
        return float(10.0 * np.log10(p_sig / p_noise))

    def _alpha_prior(epoch: int) -> float:
        t = min(1.0, max(0.0, epoch / float(alpha_decay_epochs)))
        return alpha_start * (1 - t) + alpha_end * t

    @torch.no_grad()
    def update_R(epoch: int) -> None:
        """Recompute p_channel/conf/residual from current models, then rebuild R and EMA-update."""
        nonlocal seismic, R_prev_flat, residual_scale_ema

        if not iterative_R:
            return
        if R_update_every <= 0:
            return
        if epoch == 0 or (epoch % R_update_every) != 0:
            return

        print(f"[R-UPDATE] epoch={epoch} ...")

        inverse_model.eval()
        forward_model.eval()
        Facies_model.eval()

        # build predicted facies probabilities + physics residual on ALL traces
        N = seismic.shape[0]
        Hs = seismic.shape[-1]
        bs = int(train_p.get("R_update_bs", 16))
        # sequential loader over all traces (shuffle=False)
        all_traces = np.arange(N, dtype=int)
        all_ds = SeismicDataset1D_SPF(seismic, model, facies, all_traces)
        all_ld = DataLoader(all_ds, batch_size=bs, shuffle=False, num_workers=0)

        pch = np.zeros((N, Hs), dtype=np.float32)
        pother = np.zeros((N, Hs), dtype=np.float32)
        conf = np.zeros((N, Hs), dtype=np.float32)
        ent = np.zeros((N, Hs), dtype=np.float32)
        pres = np.zeros((N, Hs), dtype=np.float32)
        sig = np.zeros((N, Hs), dtype=np.float32)
        channel_id = int(train_p.get("channel_id", 2))

        mem = 0
        for x, y_gt, z_gt in all_ld:
            x = x.to(device, non_blocking=True)
            y_pred = inverse_model(x)
            x_rec = forward_model(y_pred)

            fac_in = y_pred.detach() if facies_detach_y else y_pred
            logits = Facies_model(fac_in)  # [B,K,H]
            probs = torch.softmax(logits, dim=1)
            pch_b = probs[:, int(channel_id), :].detach().cpu().numpy()
            p_other = probs.clone()
            p_other[:, int(channel_id), :] = torch.finfo(p_other.dtype).min
            pother_b = p_other.max(dim=1).values.detach().cpu().numpy()
            conf_b = probs.max(dim=1).values.detach().cpu().numpy()
            k_cls = int(probs.shape[1])
            h_b = -(probs * torch.log(probs + float(adapt_eps))).sum(dim=1) / np.log(max(2, k_cls))
            ent_b = h_b.detach().cpu().numpy()

            # physics residual (abs error in seismic reconstruction)
            # x_rec: [B,1,H], x[:,0:1,:] is amplitude channel
            res_b = torch.abs(x_rec - x[:, 0:1, :]).squeeze(1).detach().cpu().numpy()
            sig_b = torch.abs(x[:, 0:1, :]).squeeze(1).detach().cpu().numpy()

            bsz = x.shape[0]
            pch[mem:mem+bsz] = pch_b
            pother[mem:mem+bsz] = pother_b
            conf[mem:mem+bsz] = conf_b
            ent[mem:mem+bsz] = ent_b
            pres[mem:mem+bsz] = res_b
            sig[mem:mem+bsz] = sig_b
            mem += bsz

        # reshape to 3D (H,IL,XL)
        IL, XL = int(meta["inline"]), int(meta["xline"])
        H = int(meta["H"])
        assert H == Hs, f"Depth mismatch: meta.H={H}, Hs={Hs}"

        pch_3d = torch.from_numpy(pch.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        p_other_max_3d = torch.from_numpy(pother.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        conf_3d = torch.from_numpy(conf.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        pres_3d = torch.from_numpy(pres.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        ent_3d = torch.from_numpy(ent.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)

        # prior facies (anchor). For Stanford VI-E we use Facies.npy; for real data use interpreter prior.
        fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)

        # rebuild R_new from p_channel/conf, with prior mixing + confidence gating + physics damping
        alpha = _alpha_prior(epoch)
        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device=device)

        eta_map_3d = None
        rho_aniso_map_3d = None
        eta_update_map_3d = None
        g_snr_lin_3d = torch.ones_like(pch_3d)
        g_snr_3d = torch.ones_like(pch_3d)
        g_ent_3d = torch.ones_like(pch_3d)
        r_phys_norm_3d = torch.zeros_like(pch_3d)
        g_phys_3d = torch.ones_like(pch_3d)
        rho_3d = torch.ones_like(pch_3d)
        rho_aniso_3d = torch.ones_like(pch_3d)
        alpha_update_3d = torch.ones_like(pch_3d)
        p_other_max_3d_log = torch.zeros_like(pch_3d)
        margin_3d_log = torch.zeros_like(pch_3d)
        g_dom_3d = torch.ones_like(pch_3d)
        g_ch_final_3d = torch.ones_like(pch_3d)
        eta_floor_eff = float(eta_min)
        snr_source = "none"
        snr_used = np.nan
        snr_db_residual_proxy = np.nan
        snr_db_eff = np.nan
        snr_is_capped = 0
        rho_warmup_progress = np.nan
        rho_warmup_w = 1.0
        freeze_triggered = 0
        freeze_aniso_triggered = 0
        if adaptive_eta_enable:
            snr_db_residual_proxy = _snr_db_from_residual_proxy(sig, pres, adapt_eps)
            if snr_cap_by_residual and (snr_db_target is not None):
                snr_db_eff = float(min(float(snr_db_target), float(snr_db_residual_proxy)))
                snr_is_capped = int(snr_db_eff < float(snr_db_target))
                snr_source = "noise_config_cap_residual"
            elif snr_db_real is not None:
                snr_db_eff = float(snr_db_real)
                snr_source = "noise_config"
            elif snr_db_target is not None:
                snr_db_eff = float(snr_db_target)
                snr_source = "noise_config_target"
            else:
                # No explicit noise config (e.g. clean/noise_none run):
                # do not trust early-epoch residual proxy for freeze gating.
                # Treat as clean/high-SNR to avoid disabling anisotropic updates.
                snr_db_eff = float(snr_high)
                snr_source = "default_clean"
            snr_used = float(snr_db_eff)
            g_snr_lin = _clamp01((snr_db_eff - snr_low) / (snr_high - snr_low + adapt_eps))
            g_snr = float(g_snr_lin ** snr_power)
            g_snr_lin_3d = torch.full_like(pch_3d, float(g_snr_lin))
            g_snr_3d = torch.full_like(pch_3d, float(g_snr))
            rho_snr_3d = g_snr_3d.clamp(0.0, 1.0)
            if ent_3d is None:
                print("[ANISO][WARN] entropy source missing; fallback to zeros_like(p_channel).")
                entropy_3d = torch.zeros_like(pch_3d)
            else:
                entropy_3d = ent_3d.clamp(0.0, 1.0)
            p_channel_3d = pch_3d.clamp(0.0, 1.0)
            rho_conf_3d = (
                (1.0 - entropy_3d).clamp(0.0, 1.0).pow(float(ent_power))
                * p_channel_3d.pow(float(ch_power))
            ).clamp(0.0, 1.0)
            p_other_max_3d_log = p_other_max_3d.clamp(0.0, 1.0)
            margin_3d_log = (pch_3d - p_other_max_3d_log)
            g_ch_3d = torch.sigmoid((pch_3d - float(tau)) / float(s))
            g_dom_3d = torch.sigmoid((margin_3d_log - float(margin0)) / float(margin_s))
            if enable_margin_gate:
                g_ch_final_3d = (g_ch_3d * g_dom_3d).clamp(0.0, 1.0)
            else:
                g_ch_final_3d = torch.ones_like(g_ch_3d)
            gate_low_ratio = float((g_ch_final_3d < 0.5).float().mean().item())
            print(f"[GATE] enabled={int(enable_margin_gate)} ratio(g<0.5)={gate_low_ratio:.4f}")
            rho_aniso_map_3d = (rho_snr_3d * rho_conf_3d * g_ch_final_3d).clamp(0.0, 1.0)
            rho_aniso_map_3d = torch.where(
                rho_aniso_map_3d > float(rho_cut),
                rho_aniso_map_3d,
                torch.zeros_like(rho_aniso_map_3d),
            )
            _assert_finite_torch("g_ch_final_3d", g_ch_final_3d)
            _assert_finite_torch("rho_snr_3d", rho_snr_3d)
            _assert_finite_torch("rho_conf_3d", rho_conf_3d)
            _assert_finite_torch("rho_aniso_map_3d(pre_warmup)", rho_aniso_map_3d)

            # entropy gate
            h_norm = entropy_3d
            g_ent_raw_3d = (1.0 - h_norm).clamp(0.0, 1.0)
            g_ent_3d = g_ent_raw_3d.pow(float(ent_power)).clamp(0.0, 1.0)

            # physics residual gate with robust EMA scale
            cur_scale = float(np.percentile(pres, residual_pctl))
            if (residual_scale_ema is None) or (not np.isfinite(residual_scale_ema)):
                residual_scale_ema = cur_scale
            else:
                residual_scale_ema = (residual_ema_decay * residual_scale_ema) + ((1.0 - residual_ema_decay) * cur_scale)
            scale = max(float(residual_scale_ema), float(adapt_eps))
            r_phys_norm_3d = (pres_3d / scale).clamp(0.0, 1.0)
            g_phys_raw_3d = (1.0 - r_phys_norm_3d).clamp(0.0, 1.0)
            g_phys_3d = g_phys_raw_3d.pow(float(phys_power)).clamp(0.0, 1.0)

            eta_floor_eff = float(alpha_floor_ratio)
            alpha_raw_3d = (float(w_ent) * g_ent_3d) + (float(w_phys) * g_phys_3d)
            if alpha_update_mode == "coupled":
                alpha_raw_3d = alpha_raw_3d * rho_aniso_map_3d
            alpha_update_map_3d = alpha_raw_3d.clamp(min=float(eta_floor_eff), max=1.0)
            eta_t = float(eta0)
            eta_update_map_3d = (
                eta_t
                * (float(eta_floor_ratio_update) + (1.0 - float(eta_floor_ratio_update)) * rho_aniso_map_3d)
            ).clamp(min=float(eta_t * eta_floor_ratio_update), max=float(eta_t))
            if eta_update_mode == "coupled":
                eta_update_map_3d = (eta_update_map_3d * rho_aniso_map_3d).clamp(
                    min=float(eta_t * eta_floor_ratio_update), max=float(eta_t)
                )
            _assert_finite_torch("alpha_update_map_3d", alpha_update_map_3d)
            _assert_finite_torch("eta_update_map_3d", eta_update_map_3d)
            eta_map_3d = eta_update_map_3d
            # Freeze anisotropic update only when SNR comes from explicit noise config,
            # never from default-clean fallback.
            can_freeze = str(snr_source) in {"noise_config_cap_residual", "noise_config", "noise_config_target"}
            if can_freeze and (snr_freeze is not None) and np.isfinite(snr_db_eff) and (float(snr_db_eff) <= float(snr_freeze)):
                rho_aniso_map_3d = torch.zeros_like(rho_aniso_map_3d)
                freeze_triggered = 1
                freeze_aniso_triggered = 1
            rho_aniso_map_3d_pre_warmup = rho_aniso_map_3d
            max_epoch_rho = int(max(1, int(train_p.get("epochs", 1000)) - 1))
            rho_warmup_progress = float(epoch) / float(max_epoch_rho)
            if float(rho_warmup_ratio) <= 0.0:
                rho_warmup_w = 1.0
            else:
                if rho_warmup_progress < float(rho_warmup_ratio):
                    rho_warmup_w = 0.0
                else:
                    rho_warmup_w = (rho_warmup_progress - float(rho_warmup_ratio)) / max(
                        float(adapt_eps), (1.0 - float(rho_warmup_ratio))
                    )
                rho_warmup_w = _clamp01(rho_warmup_w)
            rho_aniso_map_3d = (rho_aniso_map_3d_pre_warmup * float(rho_warmup_w)).clamp(0.0, 1.0)
            rho_aniso_3d = rho_aniso_map_3d.clamp(0.0, 1.0)
            gamma_base = float(train_p.get("aniso_gamma", 8.0))
            max_epoch_gamma = int(max(1, int(train_p.get("epochs", 1000)) - 1))
            gamma_progress = float(epoch) / float(max_epoch_gamma)
            if float(gamma_warmup_ratio) <= 0.0:
                gamma_t = float(gamma_base)
            else:
                if gamma_progress < float(gamma_warmup_ratio):
                    gamma_t = 0.0
                else:
                    gamma_t = float(gamma_base) * (
                        (gamma_progress - float(gamma_warmup_ratio))
                        / max(float(adapt_eps), 1.0 - float(gamma_warmup_ratio))
                    )
            gamma_t = max(0.0, float(gamma_t))
            gamma_eff_3d = (float(gamma_t) * rho_aniso_3d).clamp(
                min=0.0, max=float(gamma_t) * float(gamma_cap_ratio)
            )
            if float(gamma_t) > float(adapt_eps):
                rho_aniso_for_gamma_3d = (gamma_eff_3d / float(gamma_t)).clamp(0.0, 1.0)
            else:
                rho_aniso_for_gamma_3d = torch.zeros_like(rho_aniso_3d)
            _assert_finite_torch("gamma_eff_3d", gamma_eff_3d)
            _assert_finite_torch("rho_aniso_for_gamma_3d", rho_aniso_for_gamma_3d)
            alpha_update_3d = (eta_update_map_3d / (eta0 + adapt_eps)).clamp(0.0, 1.0)
            rho_3d = alpha_update_3d
            _assert_finite_torch("rho_aniso_3d(post_warmup)", rho_aniso_3d)
            _assert_finite_torch("rho_3d", rho_3d)
        else:
            gamma_t = float(train_p.get("aniso_gamma", 8.0))
            gamma_eff_3d = torch.zeros_like(pch_3d)
            rho_aniso_for_gamma_3d = None
            eta_t = float(eta0)

        R_new_flat, _ = build_R_and_prior_from_cube(
            seismic_3d=torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32),
            ai_3d=torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32),
            well_trace_indices=well_idx,
            p_channel_3d=pch_3d,
            conf_3d=conf_3d,
            facies_prior_3d=fac_prior3d,
            channel_id=int(train_p.get("channel_id", 2)),
            alpha_prior=float(alpha),
            conf_thresh=float(conf_thresh),
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            rho_aniso_map_3d=rho_aniso_for_gamma_3d if (rho_aniso_for_gamma_3d is not None) else rho_aniso_map_3d,
            eta_update_map_3d=eta_update_map_3d,
            eta_map_3d=eta_map_3d,
            rho_skip=float(rho_skip),
            gamma=float(gamma_t),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            curr_epoch=int(epoch),
            max_epoch=int(max(1, int(train_p.get("epochs", 1000)) - 1)),
            gamma_warmup_ratio=float(gamma_warmup_ratio),
            gamma_cap_ratio=1.0,
            phys_residual_3d=pres_3d if lambda_phys_damp > 0 else None,
            lambda_phys=float(lambda_phys_damp),
            use_soft_prior=False,
        )

        # EMA update for stability
        R_upd = (float(R_ema_beta) * R_prev_flat + (1.0 - float(R_ema_beta)) * R_new_flat).clamp(0.0, 1.0)
        R_prev_flat = R_upd

        # write back to seismic second channel in-place
        R_np = R_upd.detach().cpu().numpy().astype(np.float32)[:, np.newaxis, :]
        seismic[:, 1:2, :] = R_np

        # log
        r_np = R_upd.detach().cpu().numpy()
        print(f"[R-UPDATE] alpha_prior={alpha:.3f} mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")
        if debug_sanity and ((int(epoch) % max(1, int(debug_sanity_every))) == 0):
            rho_np_dbg = rho_aniso_3d.detach().cpu().numpy()
            gch_np_dbg = g_ch_final_3d.detach().cpu().numpy()
            print(
                "[SANITY-TRAIN] "
                f"epoch={int(epoch)} | "
                f"R(mean/std/min/max)=({float(r_np.mean()):.6f}/{float(r_np.std()):.6f}/{float(r_np.min()):.6f}/{float(r_np.max()):.6f}) | "
                f"rho(mean/std/min/max)=({float(rho_np_dbg.mean()):.6f}/{float(rho_np_dbg.std()):.6f}/{float(rho_np_dbg.min()):.6f}/{float(rho_np_dbg.max()):.6f}) | "
                f"g_ch_final(mean/std/min/max)=({float(gch_np_dbg.mean()):.6f}/{float(gch_np_dbg.std()):.6f}/{float(gch_np_dbg.min()):.6f}/{float(gch_np_dbg.max()):.6f})"
            )
        if adaptive_eta_enable and debug_aniso_stats and ((int(epoch) % max(1, int(debug_aniso_every))) == 0):
            rho_snr_np = rho_snr_3d.detach().cpu().numpy()
            entropy_np = entropy_3d.detach().cpu().numpy()
            p_channel_np_dbg = p_channel_3d.detach().cpu().numpy()
            rho_aniso_np_dbg = rho_aniso_3d.detach().cpu().numpy()
            g_ch_final_np_dbg = g_ch_final_3d.detach().cpu().numpy()
            gamma_eff_np = gamma_eff_3d.detach().cpu().numpy()
            eta_update_np_dbg = eta_update_map_3d.detach().cpu().numpy()
            rec = {
                "iter": int(epoch),
                "epoch": int(epoch),
                "step": 0,
                "gate_enabled": bool(enable_margin_gate),
                "rho_cut": float(rho_cut),
                "rho_snr": {
                    **_stats_dict_np(rho_snr_np),
                    "ratio_gt_0.3": float((rho_snr_np > 0.3).mean()),
                    "ratio_gt_0.5": float((rho_snr_np > 0.5).mean()),
                },
                "entropy": {
                    **_stats_dict_np(entropy_np),
                    "ratio_lt_0.3": float((entropy_np < 0.3).mean()),
                    "ratio_lt_0.5": float((entropy_np < 0.5).mean()),
                },
                "p_channel": {
                    **_stats_dict_np(p_channel_np_dbg),
                    "ratio_gt_0.5": float((p_channel_np_dbg > 0.5).mean()),
                },
                "rho_aniso": {
                    **_stats_dict_np(rho_aniso_np_dbg),
                    "ratio_gt_rho_cut": float((rho_aniso_np_dbg > float(rho_cut)).mean()),
                    "ratio_gt_0.5": float((rho_aniso_np_dbg > 0.5).mean()),
                },
                "g_ch_final": {
                    **_stats_dict_np(g_ch_final_np_dbg),
                    "ratio_lt_0.5": float((g_ch_final_np_dbg < 0.5).mean()),
                },
                "gamma_t": float(gamma_t),
                "gamma_eff": {
                    "mean": float(np.mean(gamma_eff_np)),
                    "max": float(np.max(gamma_eff_np)),
                },
                "eta_t": float(eta_t),
                "eta_update_map": {
                    "mean": float(np.mean(eta_update_np_dbg)),
                    "min": float(np.min(eta_update_np_dbg)),
                    "max": float(np.max(eta_update_np_dbg)),
                },
            }
            print(
                "[ANISO-DBG] "
                f"ep={int(epoch)} rho_snr_mean={rec['rho_snr']['mean']:.4f} "
                f"entropy_mean={rec['entropy']['mean']:.4f} rho_aniso_mean={rec['rho_aniso']['mean']:.4f} "
                f"gate_lt0.5={rec['g_ch_final']['ratio_lt_0.5']:.4f} gamma_t={rec['gamma_t']:.4f} "
                f"gamma_eff_mean={rec['gamma_eff']['mean']:.4f} eta_mean={rec['eta_update_map']['mean']:.4f}"
            )
            try:
                with open(debug_aniso_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[ANISO-DBG][WARN] failed to write {debug_aniso_path}: {e}")
        if adaptive_eta_enable and ((epoch % max(1, eta_log_every)) == 0):
            g_snr_lin_np = g_snr_lin_3d.detach().cpu().numpy()
            g_snr_np = g_snr_3d.detach().cpu().numpy()
            h_np = ent_3d.detach().cpu().numpy().clip(0.0, 1.0)
            g_ent_np = g_ent_3d.detach().cpu().numpy()
            r_phys_np = r_phys_norm_3d.detach().cpu().numpy()
            g_phys_np = g_phys_3d.detach().cpu().numpy()
            p_channel_np = pch_3d.detach().cpu().numpy().clip(0.0, 1.0)
            p_other_max_np = p_other_max_3d_log.detach().cpu().numpy().clip(0.0, 1.0)
            margin_np = margin_3d_log.detach().cpu().numpy()
            g_ch_np = torch.sigmoid((pch_3d - float(tau)) / float(s)).detach().cpu().numpy()
            g_dom_np = g_dom_3d.detach().cpu().numpy().clip(0.0, 1.0)
            g_ch_final_np = g_ch_final_3d.detach().cpu().numpy().clip(0.0, 1.0)
            rho_snr_np = g_snr_3d.detach().cpu().numpy()
            g_ent_rho_np = rho_conf_3d.detach().cpu().numpy()
            rho_aniso_pre_np = rho_aniso_map_3d_pre_warmup.detach().cpu().numpy()
            rho_aniso_np = rho_aniso_3d.detach().cpu().numpy()
            alpha_update_np = alpha_update_3d.detach().cpu().numpy()
            rho_np = rho_3d.detach().cpu().numpy()
            eta_np = eta_map_3d.detach().cpu().numpy() if eta_map_3d is not None else np.full_like(g_snr_np, eta0)

            with open(eta_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        int(epoch),
                        str(snr_source),
                        float(snr_used) if np.isfinite(snr_used) else np.nan,
                        int(freeze_triggered),
                        *_summary_stats_np(g_snr_lin_np),
                        *_summary_stats_np(g_snr_np),
                        *_summary_stats_np(h_np),
                        *_summary_stats_np(g_ent_np),
                        *_summary_stats_np(r_phys_np),
                        *_summary_stats_np(g_phys_np),
                        *_summary_stats_np(rho_aniso_np),
                        *_summary_stats_np(alpha_update_np),
                        float((alpha_update_np <= float(rho_skip)).mean()),
                        int(freeze_aniso_triggered),
                        float(eta_floor_eff),
                        float(alpha_floor_ratio),
                        int(eta_floor_ratio_used_as_alpha_floor_ratio),
                        *_summary_stats_np(eta_np),
                        *_summary_stats_np(rho_np),
                        float((rho_np <= float(rho_skip)).mean()),
                        float((eta_np < (eta_warn_ratio_thr * eta0)).mean()),
                        *_summary_stats_np(p_channel_np),
                        *_summary_stats_np(g_ch_np),
                        float(np.mean(rho_snr_np)),
                        float(np.mean(g_ent_rho_np)),
                        *_summary_stats_np(rho_aniso_np),
                        float(snr_db_target) if snr_db_target is not None else np.nan,
                        float(snr_db_residual_proxy) if np.isfinite(snr_db_residual_proxy) else np.nan,
                        float(snr_db_eff) if np.isfinite(snr_db_eff) else np.nan,
                        int(snr_is_capped),
                        *_summary_stats_np(p_other_max_np),
                        *_summary_stats_np(margin_np),
                        *_summary_stats_np(g_dom_np),
                        *_summary_stats_np(g_ch_final_np),
                        float(rho_warmup_ratio),
                        float(rho_warmup_progress),
                        float(rho_warmup_w),
                        *_summary_stats_np(rho_aniso_pre_np),
                        str(alpha_update_mode),
                        str(eta_update_mode),
                        int(seed),
                        int(deterministic_flag),
                        int(loader_num_workers_cfg),
                        int(cudnn_det_flag),
                        int(cudnn_bench_flag),
                        int(use_det_alg_flag),
                        str(cublas_ws_cfg),
                        str(pythonhashseed_cfg),
                    ]
                )

        # optional save
        if int(train_p.get("save_R_every", 0)) > 0 and (epoch % int(train_p["save_R_every"])) == 0:
            np.save(join(out_dir, "results", f"{run_id}_{data_flag}_R_flat_epoch{epoch:04d}.npy"), r_np)

        inverse_model.train()
        forward_model.train()
        Facies_model.train()

    # training loop
    train_loss = []
    val_loss = []
    ws_loss_list = []

    for epoch in range(int(train_p.get("epochs", 1000))):
        update_R(epoch)

        inverse_model.train()
        forward_model.train()
        Facies_model.train()

        # weak supervision (Scheme A)
        if (ws_every > 0) and ((epoch % ws_every) == 0):
            ws_running = 0.0
            ws_batches = 0
            for bi, batch in enumerate(Wsupervised_loader):
                if ws_max_batches > 0 and bi >= ws_max_batches:
                    break
                optimizer.zero_grad()

                if len(batch) == 2:
                    x, z = batch
                    prior_b = None
                else:
                    x, z, prior_b = batch

                x = x.to(device, non_blocking=True)
                z = z.to(device, non_blocking=True)
                if prior_b is not None:
                    prior_b = prior_b.to(device, non_blocking=True)

                y_pred = inverse_model(x)
                x_rec = forward_model(y_pred)
                fac_in = y_pred.detach() if facies_detach_y else y_pred
                fac_logits = Facies_model(fac_in)

                rec_w = lam_rec_ws if epoch >= ws_warmup_epochs else 0.0
                loss_ws = (lam_fac * criterion_facies(fac_logits, z)) + (rec_w * criterion_rec(x_rec, x[:, 0:1, :]))

                if prior_b is not None and get_bool(train_p, "use_soft_prior", False):
                    # weight by current R channel if present
                    Rch = x[:, 1:2, :] if x.shape[1] > 1 else 1.0
                    w = 0.1 + 0.9 * Rch
                    l_prior = ((y_pred - prior_b) ** 2 * w).mean() * float(train_p.get("lambda_prior", 0.20))
                    loss_ws = loss_ws + l_prior

                loss_ws.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
                        max_norm=grad_clip,
                    )
                optimizer.step()
                ws_running += float(loss_ws.item())
                ws_batches += 1

            if ws_batches > 0:
                ws_loss_list.append(ws_running / ws_batches)

        # supervised on wells
        running = 0.0
        nb = 0
        for x, y, z in train_loader:
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            y_pred = inverse_model(x)
            x_rec = forward_model(y_pred)
            fac_in = y_pred.detach() if facies_detach_y else y_pred
            fac_logits = Facies_model(fac_in)

            l_ai = criterion_ai(y_pred, y)
            l_fac = criterion_facies(fac_logits, z)
            l_rec = criterion_rec(x_rec, x[:, 0:1, :])

            loss = (lam_ai * l_ai) + (lam_fac * l_fac) + (lam_rec * l_rec)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
                    max_norm=grad_clip,
                )
            optimizer.step()

            running += float(loss.item())
            nb += 1

        if nb > 0:
            train_loss.append(running / nb)

        # val (AI only)
        inverse_model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = inverse_model(x)
                loss_val = criterion_ai(y_pred, y)
                val_loss.append(float(loss_val.item()))

        print(f"Epoch {epoch:04d} | Train {train_loss[-1]:.4f} | Val {val_loss[-1]:.4f}")

        # save best-like checkpoints every some epochs (optional)
        if int(train_p.get("save_every", 0)) > 0 and (epoch % int(train_p["save_every"])) == 0:
            ckpt_path = join(out_dir, "save_train_model", f"{run_id}_{data_flag}_epoch{epoch:04d}.pth")
            torch.save(inverse_model, ckpt_path)

    # -----------------------------
    # save FULL checkpoint with stats (STRICT BINDING)
    # -----------------------------
    full_ckpt_path = join(out_dir, "save_train_model", f"{run_id}_full_ckpt_{data_flag}.pth")
    torch.save(
        {
            "inverse_state_dict": inverse_model.state_dict(),
            "forward_state_dict": forward_model.state_dict(),
            "facies_state_dict": Facies_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(train_p.get("epochs", 1000)) - 1,
            "stats": stats,
            "train_p": train_p,
        },
        full_ckpt_path,
    )
    print(f"[CKPT] full checkpoint saved: {full_ckpt_path}")

    # also keep legacy "model object" save for test_3D.py compatibility
    legacy_path = join(out_dir, "save_train_model", f"{run_id}_s_uns_{data_flag}.pth")
    torch.save(inverse_model, legacy_path)
    print(f"[CKPT] legacy model saved: {legacy_path}")

    # curves
    plt.figure()
    plt.plot(train_loss, "r", label="train")
    plt.plot(val_loss, "k", label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(out_dir, "results", f"{run_id}_s_uns_{data_flag}.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    train(train_p=TCN1D_train_p)
