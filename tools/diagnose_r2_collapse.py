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


def _find_single_file(run_dir: str, suffix: str) -> str | None:
    res_dir = join(run_dir, "results")
    if not os.path.isdir(res_dir):
        return None
    cands = [join(res_dir, fn) for fn in os.listdir(res_dir) if fn.endswith(suffix)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _load_bundle(run_dir: str) -> dict:
    pred_p = _find_single_file(run_dir, "_pred_AI.npy")
    true_p = _find_single_file(run_dir, "_true_AI.npy")
    ids_p = _find_single_file(run_dir, "_eval_trace_ids.npy")
    if pred_p is None or true_p is None:
        raise FileNotFoundError(f"pred/true npy missing under {join(run_dir, 'results')}")
    pred = np.load(pred_p)
    true = np.load(true_p)
    if pred.shape != true.shape:
        raise RuntimeError(f"shape mismatch in {run_dir}: pred={pred.shape}, true={true.shape}")
    ids = None
    if ids_p is not None:
        ids = np.load(ids_p).astype(np.int64)
    return {
        "run_dir": run_dir,
        "pred_path": pred_p,
        "true_path": true_p,
        "ids_path": ids_p,
        "pred": pred,
        "true": true,
        "ids": ids,
    }


def _pick_common_eval_ids(a: dict, b: dict, n: int) -> tuple[np.ndarray, str]:
    a_ids = a.get("ids", None)
    b_ids = b.get("ids", None)
    if (a_ids is not None) and (b_ids is not None):
        if a_ids.shape == b_ids.shape and np.array_equal(a_ids, b_ids):
            ids = a_ids.copy()
            reason = "both eval_trace_ids are identical"
        else:
            ids = np.intersect1d(a_ids, b_ids, assume_unique=False)
            reason = "used intersection of eval_trace_ids"
    else:
        m = min(len(a["true"]), len(b["true"]))
        ids = np.arange(m, dtype=np.int64)
        reason = "eval_trace_ids missing in at least one run; fallback to first min(N) rows"
    if (n is not None) and (n > 0) and (ids.size > int(n)):
        ids = ids[: int(n)]
        reason += f", truncated to n={int(n)}"
    return ids.astype(np.int64), reason


def _subset_by_ids(bundle: dict, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(bundle["pred"], dtype=np.float64)
    true = np.asarray(bundle["true"], dtype=np.float64)
    bid = bundle.get("ids", None)
    if bid is None:
        if ids.max() >= len(true):
            raise RuntimeError(f"ids out of range for run {bundle['run_dir']}: max_id={ids.max()} >= {len(true)}")
        return true[ids], pred[ids]
    pos = {int(t): i for i, t in enumerate(bid.tolist())}
    rows = []
    for t in ids.tolist():
        if int(t) not in pos:
            continue
        rows.append(pos[int(t)])
    rows = np.asarray(rows, dtype=np.int64)
    return true[rows], pred[rows]


def _stats(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _conclusion(aln_report: dict, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    if bool(aln_report.get("auto_applied", False)):
        return "对齐错位高度可疑：transpose/flip 显著提升，已触发自动修复"
    dr = float(np.max(y_true) - np.min(y_true))
    std_t = float(np.std(y_true))
    std_p = float(np.std(y_pred))
    if dr < 1e-6 or std_t < 1e-6:
        return "数据尺度/分布异常：y_true 动态范围过小"
    if (std_p > 5.0 * max(std_t, 1e-8)) or (std_p < 0.2 * max(std_t, 1e-8)):
        return "数据尺度/分布异常：y_pred 方差相对 y_true 极端"
    return "对齐正常：R2 仍崩，优先排查训练侧/各向异性传播链路"


def main():
    ap = argparse.ArgumentParser(description="Offline diagnose for R2 collapse (alignment/index/scale).")
    ap.add_argument("--run-a", required=True, help="baseline-like run dir")
    ap.add_argument("--run-b", required=True, help="anisotropic run dir")
    ap.add_argument("--n", type=int, default=256, help="max number of common eval traces")
    args = ap.parse_args()

    a = _load_bundle(args.run_a)
    b = _load_bundle(args.run_b)
    common_ids, reason = _pick_common_eval_ids(a, b, args.n)
    if common_ids.size <= 0:
        raise RuntimeError("no common eval_trace_ids to compare")

    ya_true, ya_pred = _subset_by_ids(a, common_ids)
    yb_true, yb_pred = _subset_by_ids(b, common_ids)
    if ya_true.shape != ya_pred.shape or yb_true.shape != yb_pred.shape:
        raise RuntimeError("true/pred shape mismatch after subsetting")

    ma = test_3D._compute_main_metrics(ya_true, ya_pred)
    mb_raw = test_3D._compute_main_metrics(yb_true, yb_pred)
    yb_pred_fix, align_report = test_3D.diagnose_alignment(yb_true, yb_pred)
    mb_fix = test_3D._compute_main_metrics(yb_true, yb_pred_fix)

    report = {
        "run_a": {"dir": args.run_a, "pred_path": a["pred_path"], "true_path": a["true_path"], "ids_path": a["ids_path"]},
        "run_b": {"dir": args.run_b, "pred_path": b["pred_path"], "true_path": b["true_path"], "ids_path": b["ids_path"]},
        "common_eval_ids_reason": reason,
        "common_eval_ids_count": int(common_ids.size),
        "common_eval_ids_head10": common_ids[:10].astype(int).tolist(),
        "run_a_stats_true": _stats(ya_true),
        "run_a_stats_pred": _stats(ya_pred),
        "run_b_stats_true": _stats(yb_true),
        "run_b_stats_pred_raw": _stats(yb_pred),
        "run_b_stats_pred_fixed": _stats(yb_pred_fix),
        "metrics_run_a": ma,
        "metrics_run_b_raw": mb_raw,
        "metrics_run_b_fixed": mb_fix,
        "alignment_report_run_b": align_report,
        "conclusion": _conclusion(align_report, mb_fix, yb_true, yb_pred_fix),
    }

    out_path = join(os.getcwd(), "diagnose_r2_collapse.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[DIAGNOSE] saved: {out_path}")


if __name__ == "__main__":
    main()
