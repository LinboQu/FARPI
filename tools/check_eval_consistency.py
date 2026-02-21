import argparse
import json
import os
from os.path import join


METHODS = ["baseline", "noniter", "proposed", "proposed_noC", "coupled"]
NOISE_TAGS = ["noise_none", "noise_light", "noise_medium", "noise_heavy"]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check eval fingerprint consistency across methods.")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    bad = False
    for tag in NOISE_TAGS:
        group = []
        for method in METHODS:
            run_dir = join(args.out_root, method, tag)
            meta_path = join(run_dir, "eval_meta.json")
            if not os.path.isfile(meta_path):
                continue
            m = _load_json(meta_path)
            group.append(
                {
                    "method": method,
                    "run_dir": run_dir,
                    "seismic_sha1": str(m.get("seismic_sha1", "")),
                    "y_true_sha1": str(m.get("y_true_sha1", "")),
                    "eval_trace_ids_sha1": str(m.get("eval_trace_ids_sha1", "")),
                }
            )
        if not group:
            continue
        print(f"[CHECK] noise_tag={tag} | runs={len(group)}")
        for key in ["seismic_sha1", "y_true_sha1", "eval_trace_ids_sha1"]:
            vals = {x[key] for x in group}
            ok = len(vals) == 1
            print(f"  - {key}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad = True
                for x in group:
                    print(f"    method={x['method']} | {key}={x[key]} | run_dir={x['run_dir']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
