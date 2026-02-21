import argparse
import json
import os
import sys
from os.path import join

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import test_3D


def synthetic_metric_regression_check() -> None:
    # Non-unit variance to avoid accidental MSE ~= 1-R2 identity.
    y_true = np.linspace(-10.0, 20.0, 4000, dtype=np.float64).reshape(100, 40)
    y_pred_same = y_true.copy()
    y_pred_noise = y_true + 0.8 * np.sin(np.linspace(0, 40.0, y_true.size)).reshape(y_true.shape)

    m_same = test_3D._compute_main_metrics(y_true, y_pred_same)
    m_noise = test_3D._compute_main_metrics(y_true, y_pred_noise)

    assert m_same["MSE"] < 1e-12, f"MSE(identity) expected ~0, got {m_same['MSE']}"
    assert m_same["R2"] > 0.999999, f"R2(identity) expected ~1, got {m_same['R2']}"
    assert np.isinf(m_same["PSNR"]) or (m_same["PSNR"] > 120.0), f"PSNR(identity) too small: {m_same['PSNR']}"

    assert m_noise["MSE"] > 0.0, "MSE(noisy) must be > 0"
    assert m_noise["R2"] < 1.0, "R2(noisy) must be < 1"
    assert m_noise["PSNR"] < m_same["PSNR"] or np.isinf(m_same["PSNR"]), "PSNR(noisy) should drop"

    # Regression guard: avoid accidental column mixup where MSE is written as (1-R2).
    diff = abs(float(m_noise["MSE"]) - float(1.0 - m_noise["R2"]))
    assert diff > 1e-3, f"Suspicious metric coupling: MSE ~= 1-R2 (diff={diff})"
    print("[SANITY] synthetic metric regression check passed.")


def run_one_eval(run_dir: str, max_eval_traces: int) -> dict:
    cfg_path = join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json missing: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tag = str(cfg.get("run_tag", os.path.basename(run_dir)))
    test_cfg = dict(cfg)
    test_cfg.update({
        "out_dir": run_dir,
        "data_flag": cfg.get("data_flag", "Stanford_VI"),
        "no_wells": int(cfg.get("no_wells", 20)),
        "seed": int(cfg.get("seed", 2026)),
        "selected_wells_csv": cfg.get("selected_wells_csv"),
        "use_aniso_conditioning": bool(cfg.get("use_aniso_conditioning", False)),
        "aniso_gamma": float(cfg.get("aniso_gamma", 8.0)),
        "iterative_R": bool(cfg.get("iterative_R", False)),
        "lambda_recon": float(cfg.get("lambda_recon", 1.0)),
        "model_name": f"{tag}_s_uns",
        "run_id": tag,
        "seismic_override_path": cfg.get("seismic_override_path"),
        "debug_metrics": True,
        "skip_plots": True,
        "save_artifacts": False,
        "max_eval_traces": int(max_eval_traces),
    })
    print(f"[SANITY] evaluating {run_dir}")
    m = test_3D.test(test_cfg)
    if np.isfinite(m["R2"]):
        diff = abs(float(m["MSE"]) - float(1.0 - m["R2"]))
        print(f"[SANITY] MSE-(1-R2)={diff:.6f}")
    return m


def main():
    parser = argparse.ArgumentParser(description="Sanity-check ckpt loading, anisotropic tensors and metrics.")
    parser.add_argument("--baseline-run", required=True, help="e.g. runs/final5_ablation_2/baseline/noise_none")
    parser.add_argument("--anisotropic-run", required=True, help="e.g. runs/final5_ablation_2/proposed/noise_none")
    parser.add_argument("--max-eval-traces", type=int, default=4096)
    args = parser.parse_args()

    synthetic_metric_regression_check()
    mb = run_one_eval(args.baseline_run, max_eval_traces=args.max_eval_traces)
    ma = run_one_eval(args.anisotropic_run, max_eval_traces=args.max_eval_traces)
    print("[SANITY] baseline metrics:", mb)
    print("[SANITY] anisotropic metrics:", ma)


if __name__ == "__main__":
    main()
