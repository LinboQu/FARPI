import argparse
import subprocess
import sys
from os.path import join


def main():
    parser = argparse.ArgumentParser(
        description="Final proposed runner (fixed rho_warmup=0.20, gamma_cap=0.80) for full noise ablation."
    )
    parser.add_argument("--scheme", choices=["snr", "std"], default="snr")
    parser.add_argument("--noise-mode", choices=["correlated", "white"], default="correlated")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runs-root", type=str, default=join("runs", "noise_ablation_proposed_final"))
    parser.add_argument("--base-seismic", type=str, default=join("data", "Stanford_VI", "synth_40HZ.npy"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "run_noise_ablation_proposed_v2.py",
        "--scheme",
        str(args.scheme),
        "--noise-mode",
        str(args.noise_mode),
        "--seed",
        str(int(args.seed)),
        "--runs-root",
        str(args.runs_root),
        "--base-seismic",
        str(args.base_seismic),
        "--stable-loader",
        "--rho-warmup-ratio",
        "0.20",
        "--gamma-cap-ratio",
        "0.80",
    ]

    if args.epochs is not None:
        cmd.extend(["--epochs", str(int(args.epochs))])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(int(args.batch_size))])
    if args.prepare_only:
        cmd.append("--prepare-only")
    if args.overwrite:
        cmd.append("--overwrite")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
