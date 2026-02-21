import argparse
import copy
import csv
import glob
import json
import os
import random
import subprocess
import sys
from os.path import join

import numpy as np
try:
    import torch
except Exception:
    torch = None

from setting import TCN1D_test_p, TCN1D_train_p
from utils.config_cast import get_bool, get_float, get_int
from utils.noise_augment import add_noise_by_snr_db, add_noise_by_std_ratio


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def next_available_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    idx = 1
    while True:
        cand = f"{stem}_rerun{idx:02d}{ext}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def next_available_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        cand = f"{path}_rerun{idx:02d}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def _resolve_run_dir_with_reruns(base_dir: str) -> str | None:
    """
    Resolve actual run directory under non-overwrite mode.
    Priority:
      1) base_dir (if it has config.json)
      2) latest *_rerunXX dir with config.json
    """
    candidates = []
    if os.path.isfile(join(base_dir, "config.json")):
        candidates.append(base_dir)
    for d in sorted(glob.glob(base_dir + "_rerun*")):
        if os.path.isfile(join(d, "config.json")):
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(join(p, "config.json")))
    return candidates[-1]


def _pick_metrics_file(run_dir: str, run_tag: str, data_flag: str = "Stanford_VI") -> str | None:
    """
    Pick the most specific metrics file first, then fallback to newest *_metrics.npy.
    """
    expected = join(run_dir, "results", f"{run_tag}_s_uns_{data_flag}_metrics.npy")
    if os.path.isfile(expected):
        return expected
    res_dir = join(run_dir, "results")
    if not os.path.isdir(res_dir):
        return None
    metrics = [join(res_dir, fn) for fn in os.listdir(res_dir) if fn.endswith("_metrics.npy")]
    if not metrics:
        return None
    metrics.sort(key=lambda p: os.path.getmtime(p))
    return metrics[-1]


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_bool_fields(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    bool_keys = [
        "use_aniso_conditioning",
        "iterative_R",
        "adaptive_eta_enable",
        "enable_margin_gate",
        "snr_cap_by_residual",
        "deterministic",
        "use_deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "set_cublas_workspace_config",
        "set_pythonhashseed",
        "disable_tf32",
        "persistent_workers",
        "pin_memory",
        "log_reproducibility",
        "debug_aniso_stats",
        "save_artifacts",
    ]
    for k in bool_keys:
        if k in out:
            out[k] = bool(get_bool(out, k, bool(out[k])))
    return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_noise_experiments(scheme: str):
    if scheme == "snr":
        return [
            {"name": "No noise", "tag": "noise_none", "snr_db": None, "std_ratio": None},
            {"name": "Light noise", "tag": "noise_light", "snr_db": 30.0, "std_ratio": None},
            {"name": "Medium noise", "tag": "noise_medium", "snr_db": 20.0, "std_ratio": None},
            {"name": "Heavy noise", "tag": "noise_heavy", "snr_db": 10.0, "std_ratio": None},
        ]
    return [
        {"name": "No noise", "tag": "noise_none", "snr_db": None, "std_ratio": 0.0},
        {"name": "Light noise", "tag": "noise_light", "snr_db": None, "std_ratio": 0.05},
        {"name": "Medium noise", "tag": "noise_medium", "snr_db": None, "std_ratio": 0.10},
        {"name": "Heavy noise", "tag": "noise_heavy", "snr_db": None, "std_ratio": 0.20},
    ]


def prepare_noisy_cube(
    src_path: str,
    out_path: str,
    seed: int,
    noise_mode: str,
    scheme: str,
    snr_db: float | None,
    std_ratio: float | None,
):
    seismic3d = np.load(src_path).astype(np.float32)

    if (snr_db is None) and (std_ratio is None):
        noisy = seismic3d.copy()
        info = {
            "mode": "none",
            "snr_db_target": None,
            "snr_db_real": None,
            "noise_ratio": 0.0,
        }
    elif scheme == "snr":
        noisy, info = add_noise_by_snr_db(
            seismic_3d=seismic3d,
            snr_db=float(snr_db),
            mode=noise_mode,
            seed=seed,
        )
    else:
        noisy, info = add_noise_by_std_ratio(
            seismic_3d=seismic3d,
            noise_ratio=float(std_ratio),
            mode=noise_mode,
            seed=seed,
        )

    ensure_dir(os.path.dirname(out_path))
    np.save(out_path, noisy.astype(np.float32))
    return info


def parse_noise_tags(noise_levels: str):
    mapping = {
        "none": "noise_none",
        "light": "noise_light",
        "medium": "noise_medium",
        "heavy": "noise_heavy",
        "noise_none": "noise_none",
        "noise_light": "noise_light",
        "noise_medium": "noise_medium",
        "noise_heavy": "noise_heavy",
    }
    out = []
    for tok in str(noise_levels).split(","):
        t = tok.strip().lower()
        if t == "":
            continue
        if t not in mapping:
            raise ValueError(f"Unsupported noise level token: {tok}")
        v = mapping[t]
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("noise-levels resolved empty set.")
    return out


def parse_seed_list(seed: int, seeds_text: str):
    if seeds_text is None or str(seeds_text).strip() == "":
        return [int(seed)]
    out = []
    for tok in str(seeds_text).split(","):
        t = tok.strip()
        if t == "":
            continue
        out.append(int(t))
    if len(out) == 0:
        out = [int(seed)]
    return out


def write_final5_summary(out_root: str, rows: list[dict], overwrite: bool):
    headers = [
        "method", "noise_tag", "run_dir",
        "seed", "repeat_id",
        "MAE", "R2", "PCC", "SSIM", "PSNR", "MSE", "MedAE",
        "iterative_R", "enable_margin_gate", "alpha_update_mode", "eta_update_mode",
        "lambda_recon", "rho_warmup_ratio", "gamma_cap_ratio",
        "tau", "s", "k", "margin0", "margin_s", "snr_low", "snr_high", "snr_power",
        "noise_mode", "scheme",
    ]
    csv_path = join(out_root, "final5_summary.csv")
    md_path = join(out_root, "final5_summary.md")
    if not overwrite:
        csv_path = next_available_path(csv_path)
        md_path = next_available_path(md_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    by_noise = {}
    for r in rows:
        by_noise.setdefault(str(r["noise_tag"]), []).append(r)
    method_order = ["baseline", "noniter", "proposed", "proposed_noC", "coupled"]
    with open(md_path, "w", encoding="utf-8") as f:
        for noise_tag in ["noise_none", "noise_light", "noise_medium", "noise_heavy"]:
            group = by_noise.get(noise_tag, [])
            if not group:
                continue
            f.write(f"### {noise_tag}\n\n")
            f.write("| method | MAE | R2 | SSIM | PCC |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            best_mae = min(float(g["MAE"]) for g in group)
            best_r2 = max(float(g["R2"]) for g in group)
            best_ssim = max(float(g["SSIM"]) for g in group)
            best_pcc = max(float(g["PCC"]) for g in group)
            gm = {g["method"]: g for g in group}
            for m in method_order:
                if m not in gm:
                    continue
                g = gm[m]
                mae = f"{float(g['MAE']):.6f}"
                r2 = f"{float(g['R2']):.6f}"
                ssim = f"{float(g['SSIM']):.6f}"
                pcc = f"{float(g['PCC']):.6f}"
                if abs(float(g["MAE"]) - best_mae) < 1e-12:
                    mae = f"**{mae}**"
                if abs(float(g["R2"]) - best_r2) < 1e-12:
                    r2 = f"**{r2}**"
                if abs(float(g["SSIM"]) - best_ssim) < 1e-12:
                    ssim = f"**{ssim}**"
                if abs(float(g["PCC"]) - best_pcc) < 1e-12:
                    pcc = f"**{pcc}**"
                f.write(f"| {m} | {mae} | {r2} | {ssim} | {pcc} |\n")
            f.write("\n")

    print(f"[FINAL5] summary saved: {csv_path}")
    print(f"[FINAL5] summary saved: {md_path}")

    # mean/std across seeds (and repeats if provided)
    agg_map = {}
    for r in rows:
        key = (str(r.get("method", "")), str(r.get("noise_tag", "")))
        if key not in agg_map:
            agg_map[key] = {
                "method": key[0],
                "noise_tag": key[1],
                "n_seeds": set(),
                "n_runs": 0,
                "vals": {k: [] for k in ["MAE", "R2", "PCC", "SSIM", "PSNR", "MSE", "MedAE"]},
                "cfg": {
                    "iterative_R": r.get("iterative_R", ""),
                    "enable_margin_gate": r.get("enable_margin_gate", ""),
                    "alpha_update_mode": r.get("alpha_update_mode", ""),
                    "eta_update_mode": r.get("eta_update_mode", ""),
                    "lambda_recon": r.get("lambda_recon", ""),
                    "rho_warmup_ratio": r.get("rho_warmup_ratio", ""),
                    "gamma_cap_ratio": r.get("gamma_cap_ratio", ""),
                    "tau": r.get("tau", ""),
                    "s": r.get("s", ""),
                    "k": r.get("k", ""),
                    "margin0": r.get("margin0", ""),
                    "margin_s": r.get("margin_s", ""),
                    "snr_low": r.get("snr_low", ""),
                    "snr_high": r.get("snr_high", ""),
                    "snr_power": r.get("snr_power", ""),
                    "noise_mode": r.get("noise_mode", ""),
                    "scheme": r.get("scheme", ""),
                },
            }
        ag = agg_map[key]
        ag["n_runs"] += 1
        ag["n_seeds"].add(str(r.get("seed", "")))
        for mk in ["MAE", "R2", "PCC", "SSIM", "PSNR", "MSE", "MedAE"]:
            try:
                ag["vals"][mk].append(float(r.get(mk, np.nan)))
            except Exception:
                pass

    ms_headers = [
        "method", "noise_tag", "n_seeds", "n_runs",
        "MAE_mean", "MAE_std", "R2_mean", "R2_std", "PCC_mean", "PCC_std",
        "SSIM_mean", "SSIM_std", "PSNR_mean", "PSNR_std", "MSE_mean", "MSE_std", "MedAE_mean", "MedAE_std",
        "iterative_R", "enable_margin_gate", "alpha_update_mode", "eta_update_mode",
        "lambda_recon", "rho_warmup_ratio", "gamma_cap_ratio",
        "tau", "s", "k", "margin0", "margin_s", "snr_low", "snr_high", "snr_power", "noise_mode", "scheme",
    ]
    ms_rows = []
    for (_, _), ag in sorted(agg_map.items()):
        row = {
            "method": ag["method"],
            "noise_tag": ag["noise_tag"],
            "n_seeds": len([s for s in ag["n_seeds"] if s != ""]),
            "n_runs": ag["n_runs"],
        }
        for mk in ["MAE", "R2", "PCC", "SSIM", "PSNR", "MSE", "MedAE"]:
            arr = np.asarray(ag["vals"][mk], dtype=float)
            if arr.size == 0:
                row[f"{mk}_mean"] = np.nan
                row[f"{mk}_std"] = np.nan
            else:
                row[f"{mk}_mean"] = float(np.nanmean(arr))
                row[f"{mk}_std"] = float(np.nanstd(arr))
        row.update(ag["cfg"])
        ms_rows.append(row)

    ms_csv = join(out_root, "final5_summary_meanstd.csv")
    ms_md = join(out_root, "final5_summary_meanstd.md")
    if not overwrite:
        ms_csv = next_available_path(ms_csv)
        ms_md = next_available_path(ms_md)
    with open(ms_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ms_headers)
        writer.writeheader()
        writer.writerows(ms_rows)
    with open(ms_md, "w", encoding="utf-8") as f:
        f.write("| method | noise_tag | n_seeds | MAE(mean±std) | R2(mean±std) | SSIM(mean±std) | PCC(mean±std) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in ms_rows:
            f.write(
                f"| {r['method']} | {r['noise_tag']} | {int(r['n_seeds'])} | "
                f"{float(r['MAE_mean']):.6f}±{float(r['MAE_std']):.6f} | "
                f"{float(r['R2_mean']):.6f}±{float(r['R2_std']):.6f} | "
                f"{float(r['SSIM_mean']):.6f}±{float(r['SSIM_std']):.6f} | "
                f"{float(r['PCC_mean']):.6f}±{float(r['PCC_std']):.6f} |\n"
            )
    print(f"[FINAL5] summary saved: {ms_csv}")
    print(f"[FINAL5] summary saved: {ms_md}")


def main():
    def _nullable_float(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"none", "null", ""}:
            return None
        return float(v)

    parser = argparse.ArgumentParser(description="Noise-level controlled proposed_v2 ablation (noise- & uncertainty-aware adaptive eta).")
    parser.add_argument("--suite", choices=["none", "final5"], default="none")
    parser.add_argument("--out-root", type=str, default=join("runs", "final5_ablation"))
    parser.add_argument("--noise-levels", type=str, default="none,light,medium,heavy")
    parser.add_argument("--scheme", choices=["snr", "std"], default="snr", help="Noise control by SNR(dB) or std ratio.")
    parser.add_argument(
        "--noise-mode",
        choices=["correlated", "white"],
        default="correlated",
        help="Use correlated noise (recommended for seismic) or white noise.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 2024,2025,2026")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat times per seed.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--stable-loader",
        action="store_true",
        help="Force deterministic DataLoader settings (num_workers=0, persistent_workers=False, pin_memory=False).",
    )
    parser.add_argument("--runs-root", type=str, default=join("runs", "noise_ablation_proposed_v2"))
    parser.add_argument("--seed-subdir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--base-seismic", type=str, default=join("data", "Stanford_VI", "synth_40HZ.npy"))
    parser.add_argument("--eta0", type=float, default=0.6)
    parser.add_argument("--eta-min", type=float, default=0.0, help="Legacy absolute eta floor (kept for compatibility).")
    parser.add_argument("--eta-floor-ratio", type=float, default=None, help="Legacy alias of alpha_floor_ratio.")
    parser.add_argument("--alpha-floor-ratio", type=float, default=None)
    parser.add_argument("--alpha-update-mode", choices=["decoupled", "coupled"], default="decoupled")
    parser.add_argument("--eta-update-mode", choices=["decoupled", "coupled"], default="decoupled")
    parser.add_argument("--snr-low", type=float, default=10.0)
    parser.add_argument("--snr-high", type=float, default=30.0)
    parser.add_argument("--snr-power", type=float, default=2.0)
    parser.add_argument("--ent-power", type=float, default=0.5)
    parser.add_argument("--phys-power", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.60)
    parser.add_argument("--s", type=float, default=0.08)
    parser.add_argument("--k", type=float, default=1.0)
    mg_group = parser.add_mutually_exclusive_group()
    mg_group.add_argument("--enable-margin-gate", dest="enable_margin_gate", action="store_true")
    mg_group.add_argument("--disable-margin-gate", dest="enable_margin_gate", action="store_false")
    parser.set_defaults(enable_margin_gate=True)
    parser.add_argument("--margin0", type=float, default=0.15)
    parser.add_argument("--margin-s", type=float, default=0.05)
    parser.add_argument("--w-ent", type=float, default=0.5)
    parser.add_argument("--w-phys", type=float, default=0.5)
    parser.add_argument("--rho-skip", type=float, default=0.10)
    parser.add_argument("--snr-freeze", type=str, default="12.0", help="Set None/null to disable freeze.")
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument("--snr-cap-by-residual", dest="snr_cap_by_residual", action="store_true")
    cap_group.add_argument("--no-snr-cap-by-residual", dest="snr_cap_by_residual", action="store_false")
    parser.set_defaults(snr_cap_by_residual=True)
    parser.add_argument("--rho-warmup-ratio", type=float, default=0.35)
    parser.add_argument("--gamma-warmup-ratio", type=float, default=0.30)
    parser.add_argument("--gamma-cap-ratio", type=float, default=0.50)
    parser.add_argument("--residual-ema-decay", type=float, default=0.9)
    parser.add_argument("--residual-pctl", type=float, default=95.0)
    det_group = parser.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", dest="deterministic", action="store_true")
    det_group.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--loader-num-workers", type=int, default=None)
    parser.add_argument("--lambda-recon", type=float, default=None)
    parser.add_argument("--lambda-recon-ws", type=float, default=None)
    parser.add_argument("--lambda-phys-damp", type=float, default=None)
    parser.add_argument("--aniso-gamma", type=float, default=None)
    parser.add_argument("--r-update-every", type=int, default=None)
    ir_group = parser.add_mutually_exclusive_group()
    ir_group.add_argument("--iterative-r", dest="iterative_R", action="store_true")
    ir_group.add_argument("--no-iterative-r", dest="iterative_R", action="store_false")
    parser.set_defaults(iterative_R=None)
    aa_group = parser.add_mutually_exclusive_group()
    aa_group.add_argument("--adaptive-eta-enable", dest="adaptive_eta_enable", action="store_true")
    aa_group.add_argument("--disable-adaptive-eta", dest="adaptive_eta_enable", action="store_false")
    parser.set_defaults(adaptive_eta_enable=None)
    ua_group = parser.add_mutually_exclusive_group()
    ua_group.add_argument("--use-aniso-conditioning", dest="use_aniso_conditioning", action="store_true")
    ua_group.add_argument("--no-use-aniso-conditioning", dest="use_aniso_conditioning", action="store_false")
    parser.set_defaults(use_aniso_conditioning=None)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate noisy cubes and configs.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing experiment directories/summary files. Default is keep-old and create *_rerunXX.",
    )
    args = parser.parse_args()
    seed_list = parse_seed_list(args.seed, args.seeds)
    repeat_n = max(1, int(args.repeat))
    # Keep folder layout consistent with runs/final5_ablation for single-seed runs.
    use_seed_subdir = (len(seed_list) > 1) or (repeat_n > 1)
    alpha_floor_ratio = 0.20
    eta_floor_alias_used = 0
    if args.alpha_floor_ratio is not None:
        alpha_floor_ratio = float(args.alpha_floor_ratio)
    elif args.eta_floor_ratio is not None:
        alpha_floor_ratio = float(args.eta_floor_ratio)
        eta_floor_alias_used = 1
    if (not args.prepare_only) and (torch is None):
        raise RuntimeError("PyTorch is required for training/testing. Use --prepare-only to only generate noisy data.")
    train = None
    test = None
    if not args.prepare_only:
        from train_multitask import train as train_fn
        from test_3D import test as test_fn
        train = train_fn
        test = test_fn

    runs_root = args.runs_root
    ensure_dir(runs_root)
    set_global_seed(int(seed_list[0]))

    base_train = copy.deepcopy(TCN1D_train_p)
    base_train.update(
        {
            "use_aniso_conditioning": True,
            "iterative_R": True,
            # Conservative/stable default for proposed.
            "lambda_ai": 10.0,
            "lambda_facies": 0.05,
            "lambda_recon": 0.0,
            "lambda_recon_ws": 0.05,
            "ws_every": 4,
            "ws_max_batches": 40,
            "ws_warmup_epochs": 60,
            "seed": int(seed_list[0]),
            "aniso_eta": float(args.eta0),
            "adaptive_eta_enable": True,
            "eta_min": float(args.eta_min),
            "alpha_floor_ratio": float(alpha_floor_ratio),
            "eta_floor_ratio": float(alpha_floor_ratio),  # alias for backward compatibility
            "eta_floor_ratio_used_as_alpha_floor_ratio": int(eta_floor_alias_used),
            "alpha_update_mode": str(args.alpha_update_mode),
            "eta_update_mode": str(args.eta_update_mode),
            "snr_low": float(args.snr_low),
            "snr_high": float(args.snr_high),
            "snr_power": float(args.snr_power),
            "ent_power": float(args.ent_power),
            "phys_power": float(args.phys_power),
            "tau": float(args.tau),
            "s": float(args.s),
            "k": float(args.k),
            "enable_margin_gate": bool(args.enable_margin_gate),
            "margin0": float(args.margin0),
            "margin_s": float(args.margin_s),
            "w_ent": float(args.w_ent),
            "w_phys": float(args.w_phys),
            "rho_skip": float(args.rho_skip),
            "snr_freeze": _nullable_float(args.snr_freeze),
            "snr_cap_by_residual": bool(args.snr_cap_by_residual),
            "rho_warmup_ratio": float(args.rho_warmup_ratio),
            "gamma_warmup_ratio": float(args.gamma_warmup_ratio),
            "gamma_cap_ratio": float(args.gamma_cap_ratio),
            "residual_ema_decay": float(args.residual_ema_decay),
            "residual_pctl": float(args.residual_pctl),
            "enable_suite_final5": bool(args.suite == "final5"),
            "deterministic": bool(args.deterministic),
            "use_deterministic_algorithms": bool(args.deterministic),
            "deterministic_warn_only": True,
            "cudnn_deterministic": bool(args.deterministic),
            "cudnn_benchmark": bool(not args.deterministic),
            "set_cublas_workspace_config": True,
            "cublas_workspace_config": str(TCN1D_train_p.get("cublas_workspace_config", ":4096:8")),
            "set_pythonhashseed": True,
            "disable_tf32": True,
            "loader_num_workers": int(TCN1D_train_p.get("loader_num_workers", 0)),
            "log_reproducibility": True,
            "seeds": str(args.seeds if args.suite == "final5" else ",".join(str(int(s)) for s in seed_list)),
        }
    )
    if args.epochs is not None:
        base_train["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        base_train["batch_size"] = int(args.batch_size)
    if args.lambda_recon is not None:
        base_train["lambda_recon"] = float(args.lambda_recon)
    if args.lambda_recon_ws is not None:
        base_train["lambda_recon_ws"] = float(args.lambda_recon_ws)
    if args.lambda_phys_damp is not None:
        base_train["lambda_phys_damp"] = float(args.lambda_phys_damp)
    else:
        base_train["lambda_phys_damp"] = float(base_train.get("lambda_phys_damp", 0.20))
    if args.aniso_gamma is not None:
        base_train["aniso_gamma"] = float(args.aniso_gamma)
    else:
        base_train["aniso_gamma"] = 4.0
    if args.r_update_every is not None:
        base_train["R_update_every"] = int(args.r_update_every)
    if args.iterative_R is not None:
        base_train["iterative_R"] = bool(args.iterative_R)
    if args.adaptive_eta_enable is not None:
        base_train["adaptive_eta_enable"] = bool(args.adaptive_eta_enable)
    if args.use_aniso_conditioning is not None:
        base_train["use_aniso_conditioning"] = bool(args.use_aniso_conditioning)
    if args.loader_num_workers is not None:
        base_train["loader_num_workers"] = int(args.loader_num_workers)
    if bool(args.stable_loader):
        base_train.update({"num_workers": 0, "loader_num_workers": 0, "persistent_workers": False, "pin_memory": False})

    noise_tags = parse_noise_tags(args.noise_levels)
    experiments = [e for e in build_noise_experiments(args.scheme) if e["tag"] in set(noise_tags)]
    summary_rows = []
    facies_summary_rows = []
    if args.suite == "final5":
        out_root = args.out_root
        ensure_dir(out_root)
        method_cmds = [
            ("baseline", [
                "--no-iterative-r",
                "--no-use-aniso-conditioning",
                "--disable-adaptive-eta",
                "--lambda-recon", "0.0",
                "--lambda-recon-ws", "0.0",
                "--lambda-phys-damp", "0.0",
                "--aniso-gamma", "0.0",
                "--disable-margin-gate",
                "--alpha-update-mode", "decoupled",
                "--eta-update-mode", "decoupled",
            ]),
            ("noniter", [
                "--no-iterative-r",
                "--use-aniso-conditioning",
                "--adaptive-eta-enable",
                "--alpha-update-mode", "decoupled",
                "--eta-update-mode", "decoupled",
            ]),
            ("proposed", [
                "--iterative-r",
                "--use-aniso-conditioning",
                "--adaptive-eta-enable",
                "--enable-margin-gate",
                "--alpha-update-mode", "decoupled",
                "--eta-update-mode", "decoupled",
            ]),
            ("proposed_noC", [
                "--iterative-r",
                "--use-aniso-conditioning",
                "--adaptive-eta-enable",
                "--disable-margin-gate",
                "--alpha-update-mode", "decoupled",
                "--eta-update-mode", "decoupled",
            ]),
            ("coupled", [
                "--iterative-r",
                "--use-aniso-conditioning",
                "--adaptive-eta-enable",
                "--enable-margin-gate",
                "--alpha-update-mode", "coupled",
                "--eta-update-mode", "coupled",
            ]),
        ]

        common_cmd = [
            sys.executable, __file__,
            "--suite", "none",
            "--scheme", str(args.scheme),
            "--noise-mode", str(args.noise_mode),
            "--noise-levels", str(args.noise_levels),
            "--base-seismic", str(args.base_seismic),
            "--eta0", str(float(args.eta0)),
            "--eta-min", str(float(args.eta_min)),
            "--alpha-floor-ratio", str(float(alpha_floor_ratio)),
            "--snr-low", str(float(args.snr_low)),
            "--snr-high", str(float(args.snr_high)),
            "--snr-power", str(float(args.snr_power)),
            "--ent-power", str(float(args.ent_power)),
            "--phys-power", str(float(args.phys_power)),
            "--tau", str(float(args.tau)),
            "--s", str(float(args.s)),
            "--k", str(float(args.k)),
            "--margin0", str(float(args.margin0)),
            "--margin-s", str(float(args.margin_s)),
            "--w-ent", str(float(args.w_ent)),
            "--w-phys", str(float(args.w_phys)),
            "--rho-skip", str(float(args.rho_skip)),
            "--snr-freeze", str(args.snr_freeze),
            "--rho-warmup-ratio", str(float(args.rho_warmup_ratio)),
            "--gamma-warmup-ratio", str(float(args.gamma_warmup_ratio)),
            "--gamma-cap-ratio", str(float(args.gamma_cap_ratio)),
            "--residual-ema-decay", str(float(args.residual_ema_decay)),
            "--residual-pctl", str(float(args.residual_pctl)),
            "--deterministic" if args.deterministic else "--no-deterministic",
            "--loader-num-workers", str(int(base_train.get("loader_num_workers", 0))),
            "--repeat", "1",
        ]
        if args.epochs is not None:
            common_cmd.extend(["--epochs", str(int(args.epochs))])
        if args.batch_size is not None:
            common_cmd.extend(["--batch-size", str(int(args.batch_size))])
        if args.snr_cap_by_residual:
            common_cmd.append("--snr-cap-by-residual")
        else:
            common_cmd.append("--no-snr-cap-by-residual")
        if args.stable_loader:
            common_cmd.append("--stable-loader")
        if args.prepare_only:
            common_cmd.append("--prepare-only")
        if args.overwrite:
            common_cmd.append("--overwrite")

        for method, m_args in method_cmds:
            method_root = join(out_root, method)
            for sd in seed_list:
                for rep_i in range(repeat_n):
                    seed_subdir = ""
                    if use_seed_subdir:
                        seed_subdir = f"seed{int(sd)}" if repeat_n <= 1 else f"seed{int(sd)}_rep{rep_i+1:02d}"
                    cmd = common_cmd + [
                        "--runs-root", method_root,
                        "--seed", str(int(sd)),
                        "--seeds", str(int(sd)),
                    ] + m_args
                    if seed_subdir:
                        cmd += ["--seed-subdir", seed_subdir]
                    print("[FINAL5] running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)

        if args.prepare_only:
            print("[FINAL5] prepare-only finished. No training/testing executed.")
            return

        final_rows = []
        for method, _ in method_cmds:
            for sd in seed_list:
                for rep_i in range(repeat_n):
                    seed_subdir = ""
                    if use_seed_subdir:
                        seed_subdir = f"seed{int(sd)}" if repeat_n <= 1 else f"seed{int(sd)}_rep{rep_i+1:02d}"
                    for exp in experiments:
                        tag = exp["tag"]
                        run_dir_base = join(out_root, method, tag)
                        if seed_subdir:
                            run_dir_base = join(run_dir_base, seed_subdir)
                        run_dir = _resolve_run_dir_with_reruns(run_dir_base)
                        if run_dir is None:
                            continue
                        cfg_path = join(run_dir, "config.json")
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        metrics_path = _pick_metrics_file(run_dir, tag, data_flag=str(cfg.get("data_flag", "Stanford_VI")))
                        if metrics_path is None:
                            continue
                        mobj = np.load(metrics_path, allow_pickle=True).item()
                        final_rows.append(
                            {
                                "method": method,
                                "noise_tag": tag,
                                "run_dir": run_dir,
                                "seed": cfg.get("seed", int(sd)),
                                "repeat_id": int(rep_i + 1),
                                "MAE": mobj.get("MAE", np.nan),
                                "R2": mobj.get("R2", np.nan),
                                "PCC": mobj.get("PCC", np.nan),
                                "SSIM": mobj.get("SSIM", np.nan),
                                "PSNR": mobj.get("PSNR", np.nan),
                                "MSE": mobj.get("MSE", np.nan),
                                "MedAE": mobj.get("MedAE", np.nan),
                                "iterative_R": cfg.get("iterative_R", ""),
                                "enable_margin_gate": cfg.get("enable_margin_gate", ""),
                                "alpha_update_mode": cfg.get("alpha_update_mode", ""),
                                "eta_update_mode": cfg.get("eta_update_mode", ""),
                                "lambda_recon": cfg.get("lambda_recon", ""),
                                "rho_warmup_ratio": cfg.get("rho_warmup_ratio", ""),
                                "gamma_cap_ratio": cfg.get("gamma_cap_ratio", ""),
                                "tau": cfg.get("tau", ""),
                                "s": cfg.get("s", ""),
                                "k": cfg.get("k", ""),
                                "margin0": cfg.get("margin0", ""),
                                "margin_s": cfg.get("margin_s", ""),
                                "snr_low": cfg.get("snr_low", ""),
                                "snr_high": cfg.get("snr_high", ""),
                                "snr_power": cfg.get("snr_power", ""),
                                "noise_mode": args.noise_mode,
                                "scheme": args.scheme,
                            }
                        )
        write_final5_summary(out_root, final_rows, overwrite=args.overwrite)
        return

    for exp in experiments:
        name = exp["name"]
        tag = exp["tag"]
        base_out_dir = join(runs_root, tag)
        if str(args.seed_subdir).strip() != "":
            base_out_dir = join(base_out_dir, str(args.seed_subdir).strip())
        out_dir = base_out_dir if args.overwrite else next_available_dir(base_out_dir)
        data_dir = join(out_dir, "data")
        noisy_path = join(data_dir, "synth_40HZ_noisy.npy")

        ensure_dir(out_dir)
        ensure_dir(join(out_dir, "save_train_model"))
        ensure_dir(join(out_dir, "results"))
        ensure_dir(data_dir)

        noise_info = prepare_noisy_cube(
            src_path=args.base_seismic,
            out_path=noisy_path,
            seed=int(base_train.get("seed", seed_list[0])),
            noise_mode=args.noise_mode,
            scheme=args.scheme,
            snr_db=exp["snr_db"],
            std_ratio=exp["std_ratio"],
        )

        train_cfg = copy.deepcopy(base_train)
        train_cfg.update(
            {
                "out_dir": out_dir,
                "run_tag": tag,
                "run_id": tag,
                "seismic_override_path": noisy_path,
                "seed": int(base_train.get("seed", seed_list[0])),
            }
        )

        train_cfg = normalize_bool_fields(train_cfg)
        write_json(join(out_dir, "config.json"), train_cfg)
        write_json(
            join(out_dir, "noise_config.json"),
            {
                "scheme": args.scheme,
                "noise_mode": args.noise_mode,
                "base_seismic": args.base_seismic,
                "noisy_seismic": noisy_path,
                **noise_info,
            },
        )

        print("=" * 88)
        print(f"[NOISE-ABLATION] {name} ({tag})")
        print(
            f"[NOISE-ABLATION] source={noisy_path} | scheme={args.scheme} | "
            f"mode={args.noise_mode} | noise_info={noise_info}"
        )
        # Critical for fair ablation: keep initialization and sampling seed identical across noise levels.
        set_global_seed(int(train_cfg["seed"]))

        if not args.prepare_only:
            train(train_p=train_cfg)

            test_cfg = copy.deepcopy(TCN1D_test_p)
            test_cfg.update(copy.deepcopy(train_cfg))
            test_cfg.update(
                {
                    "out_dir": out_dir,
                    "data_flag": train_cfg["data_flag"],
                    "no_wells": train_cfg["no_wells"],
                    "seed": get_int(train_cfg, "seed", args.seed),
                    "selected_wells_csv": train_cfg.get("selected_wells_csv"),
                    "use_aniso_conditioning": get_bool(train_cfg, "use_aniso_conditioning", False),
                    "aniso_gamma": get_float(train_cfg, "aniso_gamma", TCN1D_train_p.get("aniso_gamma", 8.0)),
                    "iterative_R": get_bool(train_cfg, "iterative_R", False),
                    "lambda_recon": get_float(train_cfg, "lambda_recon", 1.0),
                    "enable_margin_gate": get_bool(train_cfg, "enable_margin_gate", True),
                    "model_name": f"{tag}_s_uns",
                    "run_id": tag,
                    "seismic_override_path": noisy_path,
                    "allow_alignment_fix": True,
                    "depth_mismatch_failfast": False,
                }
            )

            metrics = test(test_cfg)
            summary_rows.append(
                {
                    "Experiment": name,
                    "run_tag": tag,
                    "scheme": args.scheme,
                    "noise_mode": args.noise_mode,
                    "snr_db_target": noise_info.get("snr_db_target"),
                    "snr_db_real": noise_info.get("snr_db_real"),
                    "noise_ratio": noise_info.get("noise_ratio"),
                    "R2": metrics["R2"],
                    "PCC": metrics["PCC"],
                    "SSIM": metrics["SSIM"],
                    "PSNR": metrics["PSNR"],
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                    "MedAE": metrics["MedAE"],
                }
            )
            for fr in metrics.get("facies_metrics", []):
                facies_summary_rows.append(
                    {
                        "Experiment": name,
                        "run_tag": tag,
                        "actual_run_dir": out_dir,
                        "scheme": args.scheme,
                        "noise_mode": args.noise_mode,
                        "snr_db_target": noise_info.get("snr_db_target"),
                        "snr_db_real": noise_info.get("snr_db_real"),
                        "noise_ratio": noise_info.get("noise_ratio"),
                        "facies_id": fr.get("facies_id"),
                        "facies_name": fr.get("facies_name"),
                        "count": fr.get("count"),
                        "ratio": fr.get("ratio"),
                        "R2": fr.get("R2"),
                        "PCC": fr.get("PCC"),
                        "SSIM": fr.get("SSIM"),
                        "PSNR": fr.get("PSNR"),
                        "MSE": fr.get("MSE"),
                        "MAE": fr.get("MAE"),
                        "MedAE": fr.get("MedAE"),
                    }
                )

    if args.prepare_only:
        print("[NOISE-ABLATION] prepare-only finished. No training/testing executed.")
        return

    headers = [
        "Experiment",
        "run_tag",
        "scheme",
        "noise_mode",
        "snr_db_target",
        "snr_db_real",
        "noise_ratio",
        "R2",
        "PCC",
        "SSIM",
        "PSNR",
        "MSE",
        "MAE",
        "MedAE",
    ]
    csv_path = join(runs_root, "noise_ablation_summary.csv")
    md_path = join(runs_root, "noise_ablation_summary.md")
    if not args.overwrite:
        csv_path = next_available_path(csv_path)
        md_path = next_available_path(md_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in summary_rows:
            row = [
                str(r["Experiment"]),
                str(r["run_tag"]),
                str(r["scheme"]),
                str(r["noise_mode"]),
                "None" if r["snr_db_target"] is None else f"{float(r['snr_db_target']):.3f}",
                "None" if r["snr_db_real"] is None else f"{float(r['snr_db_real']):.3f}",
                "None" if r["noise_ratio"] is None else f"{float(r['noise_ratio']):.6f}",
                f"{r['R2']:.6f}",
                f"{r['PCC']:.6f}",
                f"{r['SSIM']:.6f}",
                f"{r['PSNR']:.6f}",
                f"{r['MSE']:.6f}",
                f"{r['MAE']:.6f}",
                f"{r['MedAE']:.6f}",
            ]
            f.write("| " + " | ".join(row) + " |\n")

    print("=" * 88)
    print(f"[NOISE-ABLATION] summary saved: {csv_path}")
    print(f"[NOISE-ABLATION] summary saved: {md_path}")

    if facies_summary_rows:
        f_headers = [
            "Experiment",
            "run_tag",
            "actual_run_dir",
            "scheme",
            "noise_mode",
            "snr_db_target",
            "snr_db_real",
            "noise_ratio",
            "facies_id",
            "facies_name",
            "count",
            "ratio",
            "R2",
            "PCC",
            "SSIM",
            "PSNR",
            "MSE",
            "MAE",
            "MedAE",
        ]
        f_csv = join(runs_root, "noise_ablation_facies_summary.csv")
        f_md = join(runs_root, "noise_ablation_facies_summary.md")
        if not args.overwrite:
            f_csv = next_available_path(f_csv)
            f_md = next_available_path(f_md)
        with open(f_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=f_headers)
            writer.writeheader()
            writer.writerows(facies_summary_rows)
        with open(f_md, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(f_headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(f_headers)) + "|\n")
            for r in facies_summary_rows:
                vals = [r.get(h) for h in f_headers]
                line = []
                for v in vals:
                    if v is None:
                        line.append("None")
                    elif isinstance(v, float):
                        line.append(f"{v:.6f}")
                    else:
                        line.append(str(v))
                f.write("| " + " | ".join(line) + " |\n")
        print(f"[NOISE-ABLATION] facies summary saved: {f_csv}")
        print(f"[NOISE-ABLATION] facies summary saved: {f_md}")


if __name__ == "__main__":
    main()
