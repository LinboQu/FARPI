import argparse
import copy
import csv
import json
import os
import random
from os.path import join

import numpy as np
try:
    import torch
except Exception:
    torch = None

from setting import TCN1D_test_p, TCN1D_train_p
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


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def main():
    parser = argparse.ArgumentParser(description="Noise-level controlled ablation on Stanford_VI seismic cube.")
    parser.add_argument("--scheme", choices=["snr", "std"], default="snr", help="Noise control by SNR(dB) or std ratio.")
    parser.add_argument(
        "--noise-mode",
        choices=["correlated", "white"],
        default="correlated",
        help="Use correlated noise (recommended for seismic) or white noise.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--stable-loader",
        action="store_true",
        help="Force deterministic DataLoader settings (num_workers=0, persistent_workers=False, pin_memory=False).",
    )
    parser.add_argument("--runs-root", type=str, default=join("runs", "noise_ablation"))
    parser.add_argument("--base-seismic", type=str, default=join("data", "Stanford_VI", "synth_40HZ.npy"))
    parser.add_argument("--prepare-only", action="store_true", help="Only generate noisy cubes and configs.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing experiment directories/summary files. Default is keep-old and create *_rerunXX.",
    )
    args = parser.parse_args()
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
    set_global_seed(args.seed)

    base_train = copy.deepcopy(TCN1D_train_p)
    base_train.update(
        {
            "use_aniso_conditioning": True,
            "iterative_R": True,
            "lambda_recon_ws": 0.2,
            "ws_warmup_epochs": 100,
            "seed": args.seed,
        }
    )
    if args.stable_loader:
        base_train.update({"num_workers": 0, "persistent_workers": False, "pin_memory": False})

    experiments = build_noise_experiments(args.scheme)
    summary_rows = []
    facies_summary_rows = []

    for exp in experiments:
        name = exp["name"]
        tag = exp["tag"]
        base_out_dir = join(runs_root, tag)
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
            seed=args.seed,
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
                "seed": args.seed,
            }
        )

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
            test_cfg.update(
                {
                    "out_dir": out_dir,
                    "data_flag": train_cfg["data_flag"],
                    "no_wells": train_cfg["no_wells"],
                    "seed": train_cfg.get("seed", args.seed),
                    "selected_wells_csv": train_cfg.get("selected_wells_csv"),
                    "use_aniso_conditioning": train_cfg.get("use_aniso_conditioning", False),
                    "aniso_gamma": train_cfg.get("aniso_gamma", TCN1D_train_p.get("aniso_gamma", 8.0)),
                    "iterative_R": train_cfg.get("iterative_R", False),
                    "lambda_recon": train_cfg.get("lambda_recon", 1.0),
                    "model_name": f"{tag}_s_uns",
                    "run_id": tag,
                    "seismic_override_path": noisy_path,
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
