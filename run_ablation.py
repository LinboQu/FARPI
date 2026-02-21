import copy
import csv
import json
import os
import random
from os.path import join

import numpy as np
import torch

from setting import TCN1D_test_p, TCN1D_train_p
from test_3D import test
from train_multitask import train


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_experiments():
    return [
        {
            "name": "Baseline",
            "tag": "ablation_baseline",
            "overrides": {
                "lambda_recon": 0.0,
                "use_aniso_conditioning": False,
                "iterative_R": False,
            },
        },
        {
            "name": "Isotropic",
            "tag": "ablation_isotropic",
            "overrides": {
                "use_aniso_conditioning": True,
                "aniso_gamma": 0.0,
                "iterative_R": True,
                "lambda_recon_ws": 0.2,
                "ws_warmup_epochs": 100,
            },
        },
        {
            "name": "Non-iterative",
            "tag": "ablation_noniter",
            "overrides": {
                "use_aniso_conditioning": True,
                "iterative_R": False,
                "lambda_recon_ws": 0.2,
                "ws_warmup_epochs": 100,
            },
        },
        {
            "name": "Proposed",
            "tag": "ablation_proposed",
            "overrides": {
                "use_aniso_conditioning": True,
                "iterative_R": True,
                "lambda_recon_ws": 0.2,
                "ws_warmup_epochs": 100,
            },
        },
    ]


def main() -> None:
    runs_root = "runs"
    ensure_dir(runs_root)

    base_train = copy.deepcopy(TCN1D_train_p)
    experiments = build_experiments()
    summary_rows = []

    for exp in experiments:
        name = exp["name"]
        tag = exp["tag"]
        out_dir = join(runs_root, tag)

        ensure_dir(out_dir)
        ensure_dir(join(out_dir, "save_train_model"))
        ensure_dir(join(out_dir, "results"))

        train_cfg = copy.deepcopy(base_train)
        train_cfg.update(exp["overrides"])
        train_cfg["out_dir"] = out_dir
        train_cfg["run_tag"] = tag
        train_cfg["run_id"] = tag

        write_json(join(out_dir, "config.json"), train_cfg)
        set_global_seed(int(train_cfg.get("seed", 2026)))

        print("=" * 88)
        print(f"[ABLATION] {name} ({tag})")
        print(
            "[ABLATION] switches: "
            f"lambda_recon={train_cfg.get('lambda_recon')} | "
            f"use_aniso_conditioning={train_cfg.get('use_aniso_conditioning')} | "
            f"aniso_gamma={train_cfg.get('aniso_gamma')} | "
            f"iterative_R={train_cfg.get('iterative_R')}"
        )

        train(train_p=train_cfg)

        test_cfg = copy.deepcopy(TCN1D_test_p)
        test_cfg.update(
            {
                "out_dir": out_dir,
                "data_flag": train_cfg["data_flag"],
                "no_wells": train_cfg["no_wells"],
                "seed": train_cfg.get("seed", 2026),
                "selected_wells_csv": train_cfg.get("selected_wells_csv"),
                "use_aniso_conditioning": train_cfg.get("use_aniso_conditioning", False),
                "aniso_gamma": train_cfg.get("aniso_gamma", TCN1D_train_p.get("aniso_gamma", 8.0)),
                "iterative_R": train_cfg.get("iterative_R", False),
                "lambda_recon": train_cfg.get("lambda_recon", 1.0),
                "model_name": f"{tag}_s_uns",
                "run_id": tag,
            }
        )

        metrics = test(test_cfg)
        summary_rows.append(
            {
                "Experiment": name,
                "run_tag": tag,
                "R2": metrics["R2"],
                "PCC": metrics["PCC"],
                "SSIM": metrics["SSIM"],
                "PSNR": metrics["PSNR"],
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "MedAE": metrics["MedAE"],
            }
        )

    headers = ["Experiment", "run_tag", "R2", "PCC", "SSIM", "PSNR", "MSE", "MAE", "MedAE"]
    csv_path = join(runs_root, "ablation_summary.csv")
    md_path = join(runs_root, "ablation_summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in summary_rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r["Experiment"]),
                        str(r["run_tag"]),
                        f"{r['R2']:.6f}",
                        f"{r['PCC']:.6f}",
                        f"{r['SSIM']:.6f}",
                        f"{r['PSNR']:.6f}",
                        f"{r['MSE']:.6f}",
                        f"{r['MAE']:.6f}",
                        f"{r['MedAE']:.6f}",
                    ]
                )
                + " |\n"
            )

    print("=" * 88)
    print(f"[ABLATION] summary saved: {csv_path}")
    print(f"[ABLATION] summary saved: {md_path}")


if __name__ == "__main__":
    main()
