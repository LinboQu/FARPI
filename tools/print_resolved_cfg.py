import argparse
import json
import os
import sys
from os.path import join

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.config_cast import get_bool, get_float, get_int


def main():
    ap = argparse.ArgumentParser(description="Print resolved config values with strict bool parsing.")
    ap.add_argument("--run-dir", required=True, help="Run directory containing config.json")
    args = ap.parse_args()

    cfg_path = join(args.run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    resolved = {
        "enable_margin_gate": get_bool(cfg, "enable_margin_gate", True),
        "debug_aniso_stats": get_bool(cfg, "debug_aniso_stats", False),
        "save_artifacts": get_bool(cfg, "save_artifacts", True),
        "use_aniso_conditioning": get_bool(cfg, "use_aniso_conditioning", False),
        "iterative_R": get_bool(cfg, "iterative_R", False),
        "rho_warmup_ratio": get_float(cfg, "rho_warmup_ratio", 0.35),
        "gamma_cap_ratio": get_float(cfg, "gamma_cap_ratio", 0.50),
        "eta_floor_ratio": get_float(cfg, "eta_floor_ratio", 0.25),
        "ent_power": get_float(cfg, "ent_power", 2.0),
        "ch_power": get_float(cfg, "ch_power", 2.0),
        "rho_cut": get_float(cfg, "rho_cut", 0.30),
        "snr_power": get_float(cfg, "snr_power", 2.0),
        "debug_aniso_every": get_int(cfg, "debug_aniso_every", 200),
    }
    print(json.dumps(resolved, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
