import argparse
import json
import os
from os.path import join

import numpy as np

from run_noise_ablation_proposed_v2 import (
    _resolve_run_dir_with_reruns,
    write_final5_summary,
)


METHODS = ["baseline", "noniter", "proposed", "proposed_noC", "coupled"]
NOISE_TAGS = ["noise_none", "noise_light", "noise_medium", "noise_heavy"]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_noise_group_consistency(rows: list[dict]) -> None:
    for tag in NOISE_TAGS:
        g = [r for r in rows if str(r.get("noise_tag", "")) == tag]
        if len(g) <= 1:
            continue
        for key in ["seismic_sha1", "y_true_sha1", "eval_trace_ids_sha1"]:
            vals = {str(r.get(key, "")) for r in g}
            if len(vals) != 1:
                lines = [
                    f"  method={r['method']} | run_dir={r['run_dir']} | {key}={r.get(key, '')}"
                    for r in g
                ]
                raise RuntimeError(
                    f"[CONSISTENCY][ERROR] noise_tag={tag} has inconsistent {key} across methods.\n"
                    + "\n".join(lines)
                )


def rebuild(out_root: str, overwrite: bool) -> int:
    rows = []
    for method in METHODS:
        for tag in NOISE_TAGS:
            run_base = join(out_root, method, tag)
            run_dir = _resolve_run_dir_with_reruns(run_base)
            if run_dir is None:
                continue
            cfg_path = join(run_dir, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            cfg = _load_json(cfg_path)
            metrics_path = join(run_dir, "metrics_main.json")
            meta_path = join(run_dir, "eval_meta.json")
            if (not os.path.isfile(metrics_path)) or (not os.path.isfile(meta_path)):
                raise FileNotFoundError(
                    f"[REBUILD][ERROR] missing metrics_main.json/eval_meta.json in run_dir={run_dir}"
                )
            metrics = _load_json(metrics_path)
            eval_meta = _load_json(meta_path)
            rows.append(
                {
                    "method": method,
                    "noise_tag": tag,
                    "run_dir": run_dir,
                    "seed": cfg.get("seed", ""),
                    "repeat_id": 1,
                    "MAE": metrics.get("MAE", np.nan),
                    "R2": metrics.get("R2", np.nan),
                    "PCC": metrics.get("PCC", np.nan),
                    "SSIM": metrics.get("SSIM", np.nan),
                    "PSNR": metrics.get("PSNR", np.nan),
                    "MSE": metrics.get("MSE", np.nan),
                    "MedAE": metrics.get("MedAE", np.nan),
                    "seismic_sha1": eval_meta.get("seismic_sha1", ""),
                    "y_true_sha1": eval_meta.get("y_true_sha1", ""),
                    "eval_trace_ids_sha1": eval_meta.get("eval_trace_ids_sha1", ""),
                    "ckpt_sha1": eval_meta.get("ckpt_sha1", ""),
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
                    "noise_mode": cfg.get("noise_mode", "correlated"),
                    "scheme": cfg.get("scheme", "snr"),
                }
            )
    _assert_noise_group_consistency(rows)
    write_final5_summary(out_root, rows, overwrite=overwrite)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Rebuild final5 summary from existing run folders.")
    parser.add_argument("--out-root", default=join("runs", "final5_ablation_2"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    n = rebuild(args.out_root, overwrite=args.overwrite)
    print(f"[REBUILD] rows={n} | out_root={args.out_root}")


if __name__ == "__main__":
    main()
