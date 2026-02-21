import argparse
import os
from os.path import join
import csv
import re
from collections import defaultdict


def _read_csv(path: str):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _write_csv(path: str, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})


def _to_float(x, default=float("nan")):
    try:
        if x is None or x == "" or str(x).lower() == "none":
            return default
        return float(x)
    except Exception:
        return default


def _csv_versions(root: str, base_filename: str):
    """
    Return available versioned files:
      key ''        -> base.csv
      key 'rerun01' -> base_rerun01.csv
    """
    stem, ext = os.path.splitext(base_filename)
    versions = {}
    if not os.path.isdir(root):
        return versions
    base_path = join(root, base_filename)
    if os.path.isfile(base_path):
        versions[""] = base_path
    pat = re.compile(rf"^{re.escape(stem)}_rerun(\d+){re.escape(ext)}$")
    for name in os.listdir(root):
        m = pat.match(name)
        if m is None:
            continue
        key = f"rerun{int(m.group(1)):02d}"
        versions[key] = join(root, name)
    return versions


def _resolve_consistent_csv_pair(root: str, overall_name: str, facies_name: str):
    """
    Select overall/facies CSV from the SAME version key to avoid cross-run mixing.
    Choose the newest common key by pair mtime.
    """
    ov = _csv_versions(root, overall_name)
    fa = _csv_versions(root, facies_name)
    common = sorted(set(ov.keys()) & set(fa.keys()))
    if common:
        def _pair_score(k):
            return min(os.path.getmtime(ov[k]), os.path.getmtime(fa[k]))
        best_key = max(common, key=_pair_score)
        return ov[best_key], fa[best_key], best_key

    # fallback: independently newest-by-mtime (best effort)
    def _latest(vmap, default_name):
        if not vmap:
            return join(root, default_name), ""
        best_key = max(vmap.keys(), key=lambda k: os.path.getmtime(vmap[k]))
        return vmap[best_key], best_key
    ov_path, ov_key = _latest(ov, overall_name)
    fa_path, fa_key = _latest(fa, facies_name)
    return ov_path, fa_path, f"{ov_key}|{fa_key}"


def _resolve_method_root(primary: str | None, fallback_list: list[str]) -> str:
    if primary is not None:
        if os.path.isdir(primary):
            return primary
        print(f"[SUMMARY][WARN] requested root not found: {primary}. Falling back to defaults.")
    for p in fallback_list:
        if os.path.isdir(p):
            return p
    return fallback_list[0]


def _collect_headers(rows: list[dict], prefix_headers: list[str]) -> list[str]:
    seen = set(prefix_headers)
    extra = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return prefix_headers + sorted(extra)


def _is_number_like(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"", "none", "nan"}:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge noise ablation results across methods into Sheet1/Sheet2 CSVs.")
    parser.add_argument("--proposed-root", default=None, help="Primary proposed root. If omitted, auto-pick v3 then v2.")
    parser.add_argument("--proposed-v2-root", default=join("runs", "noise_ablation_proposed_v2"), help="Legacy alias.")
    parser.add_argument("--baseline-root", default=join("runs", "noise_ablation_baseline"))
    parser.add_argument("--isotropic-root", default=join("runs", "noise_ablation_isotropic"))
    parser.add_argument("--noniter-root", default=join("runs", "noise_ablation_noniter"))
    parser.add_argument("--out-dir", default=join("runs", "noise_ablation_compare"))
    args = parser.parse_args()
    proposed_root = _resolve_method_root(
        args.proposed_root,
        [join("runs", "noise_ablation_proposed_v3"), args.proposed_v2_root],
    )

    method_roots = [
        ("proposed", proposed_root),
        ("baseline", args.baseline_root),
        ("isotropic", args.isotropic_root),
        ("noniter", args.noniter_root),
    ]

    overall_rows = []
    facies_rows = []
    for method, root in method_roots:
        sum_csv, fac_csv, ver_key = _resolve_consistent_csv_pair(
            root,
            "noise_ablation_summary.csv",
            "noise_ablation_facies_summary.csv",
        )
        print(f"[SUMMARY][{method}] version key  : {ver_key}")
        print(f"[SUMMARY][{method}] overall source: {sum_csv}")
        print(f"[SUMMARY][{method}] facies  source: {fac_csv}")
        sum_rows = _read_csv(sum_csv)
        fac_rows = _read_csv(fac_csv)
        if len(sum_rows) == 0:
            raise FileNotFoundError(
                f"[SUMMARY][{method}] missing/empty overall summary CSV: {sum_csv}. "
                f"Check method root: {root}"
            )
        if len(fac_rows) == 0:
            raise FileNotFoundError(
                f"[SUMMARY][{method}] missing/empty facies summary CSV: {fac_csv}. "
                f"Check method root: {root}"
            )
        for r in sum_rows:
            rr = dict(r)
            rr["method"] = method
            overall_rows.append(rr)
        for r in fac_rows:
            rr = dict(r)
            rr["method"] = method
            facies_rows.append(rr)

    overall_headers = _collect_headers(
        overall_rows,
        [
            "method", "Experiment", "run_tag", "scheme", "noise_mode",
            "snr_db_target", "snr_db_real", "noise_ratio",
        ],
    )
    facies_headers = _collect_headers(
        facies_rows,
        [
            "method", "Experiment", "run_tag", "actual_run_dir", "scheme", "noise_mode",
            "snr_db_target", "snr_db_real", "noise_ratio",
            "facies_id", "facies_name", "count", "ratio",
        ],
    )

    _write_csv(join(args.out_dir, "sheet1_overall.csv"), overall_headers, overall_rows)
    _write_csv(join(args.out_dir, "sheet2_facies.csv"), facies_headers, facies_rows)

    # Facies MAE comparison pivot-like table (kept for backward compatibility)
    key_order = ["No noise", "Light noise", "Medium noise", "Heavy noise"]
    by_key = {}
    for r in facies_rows:
        key = (r.get("Experiment", ""), r.get("facies_name", ""), r.get("method", ""))
        by_key[key] = _to_float(r.get("MAE"))

    mae_rows = []
    methods = ["proposed", "isotropic", "noniter", "baseline"]
    facies_names = sorted({r.get("facies_name", "") for r in facies_rows})
    for exp in key_order:
        for fn in facies_names:
            row = {"Experiment": exp, "facies_name": fn}
            for m in methods:
                row[f"MAE_{m}"] = by_key.get((exp, fn, m), float("nan"))
            mae_rows.append(row)
    mae_headers = ["Experiment", "facies_name"] + [f"MAE_{m}" for m in methods]
    _write_csv(join(args.out_dir, "facies_mae_compare.csv"), mae_headers, mae_rows)

    # Wide comparison over ALL numeric overall metrics
    overall_meta = {"method", "Experiment", "run_tag", "scheme", "noise_mode", "snr_db_target", "snr_db_real", "noise_ratio"}
    overall_metric_keys = sorted(
        {
            k
            for r in overall_rows
            for k, v in r.items()
            if (k not in overall_meta) and _is_number_like(v)
        }
    )
    methods_present = sorted({r.get("method", "") for r in overall_rows})
    overall_wide_map = defaultdict(dict)
    for r in overall_rows:
        key = (
            r.get("Experiment", ""),
            r.get("run_tag", ""),
            r.get("scheme", ""),
            r.get("noise_mode", ""),
            r.get("snr_db_target", ""),
            r.get("snr_db_real", ""),
            r.get("noise_ratio", ""),
        )
        overall_wide_map[key]["Experiment"] = r.get("Experiment", "")
        overall_wide_map[key]["run_tag"] = r.get("run_tag", "")
        overall_wide_map[key]["scheme"] = r.get("scheme", "")
        overall_wide_map[key]["noise_mode"] = r.get("noise_mode", "")
        overall_wide_map[key]["snr_db_target"] = r.get("snr_db_target", "")
        overall_wide_map[key]["snr_db_real"] = r.get("snr_db_real", "")
        overall_wide_map[key]["noise_ratio"] = r.get("noise_ratio", "")
        for mk in overall_metric_keys:
            overall_wide_map[key][f"{mk}_{r.get('method', '')}"] = r.get(mk, "")
    overall_wide_rows = list(overall_wide_map.values())
    overall_wide_headers = [
        "Experiment", "run_tag", "scheme", "noise_mode", "snr_db_target", "snr_db_real", "noise_ratio"
    ]
    for mk in overall_metric_keys:
        for m in methods_present:
            overall_wide_headers.append(f"{mk}_{m}")
    _write_csv(join(args.out_dir, "sheet3_overall_wide.csv"), overall_wide_headers, overall_wide_rows)

    # Wide comparison over ALL numeric facies metrics
    facies_meta = {
        "method", "Experiment", "run_tag", "actual_run_dir", "scheme", "noise_mode",
        "snr_db_target", "snr_db_real", "noise_ratio", "facies_id", "facies_name", "count", "ratio",
    }
    facies_metric_keys = sorted(
        {
            k
            for r in facies_rows
            for k, v in r.items()
            if (k not in facies_meta) and _is_number_like(v)
        }
    )
    facies_wide_map = defaultdict(dict)
    for r in facies_rows:
        key = (
            r.get("Experiment", ""),
            r.get("run_tag", ""),
            r.get("scheme", ""),
            r.get("noise_mode", ""),
            r.get("snr_db_target", ""),
            r.get("snr_db_real", ""),
            r.get("noise_ratio", ""),
            r.get("facies_id", ""),
            r.get("facies_name", ""),
        )
        facies_wide_map[key]["Experiment"] = r.get("Experiment", "")
        facies_wide_map[key]["run_tag"] = r.get("run_tag", "")
        facies_wide_map[key]["scheme"] = r.get("scheme", "")
        facies_wide_map[key]["noise_mode"] = r.get("noise_mode", "")
        facies_wide_map[key]["snr_db_target"] = r.get("snr_db_target", "")
        facies_wide_map[key]["snr_db_real"] = r.get("snr_db_real", "")
        facies_wide_map[key]["noise_ratio"] = r.get("noise_ratio", "")
        facies_wide_map[key]["facies_id"] = r.get("facies_id", "")
        facies_wide_map[key]["facies_name"] = r.get("facies_name", "")
        for mk in facies_metric_keys:
            facies_wide_map[key][f"{mk}_{r.get('method', '')}"] = r.get(mk, "")
    facies_wide_rows = list(facies_wide_map.values())
    facies_wide_headers = [
        "Experiment", "run_tag", "scheme", "noise_mode", "snr_db_target", "snr_db_real", "noise_ratio", "facies_id", "facies_name"
    ]
    for mk in facies_metric_keys:
        for m in methods_present:
            facies_wide_headers.append(f"{mk}_{m}")
    _write_csv(join(args.out_dir, "sheet4_facies_wide.csv"), facies_wide_headers, facies_wide_rows)

    # Consistency check: overall MAE vs facies-weighted MAE for each method/experiment/run_tag
    fac_group = defaultdict(list)
    for r in facies_rows:
        k = (r.get("method", ""), r.get("Experiment", ""), r.get("run_tag", ""))
        fac_group[k].append(r)
    overall_group = {}
    for r in overall_rows:
        k = (r.get("method", ""), r.get("Experiment", ""), r.get("run_tag", ""))
        overall_group[k] = r

    check_rows = []
    for k in sorted(set(fac_group.keys()) | set(overall_group.keys())):
        method, exp, run_tag = k
        ov = overall_group.get(k, {})
        frows = fac_group.get(k, [])
        w_sum = 0.0
        mae_w = 0.0
        for fr in frows:
            w = _to_float(fr.get("ratio"), 0.0)
            if not (w == w):  # nan
                w = 0.0
            mae = _to_float(fr.get("MAE"), float("nan"))
            if mae == mae:
                mae_w += w * mae
                w_sum += w
        facies_weighted_mae = (mae_w / w_sum) if w_sum > 0 else float("nan")
        overall_mae = _to_float(ov.get("MAE"), float("nan"))
        delta = overall_mae - facies_weighted_mae if (overall_mae == overall_mae and facies_weighted_mae == facies_weighted_mae) else float("nan")
        check_rows.append(
            {
                "method": method,
                "Experiment": exp,
                "run_tag": run_tag,
                "overall_MAE": overall_mae,
                "facies_weighted_MAE": facies_weighted_mae,
                "delta_overall_minus_facies_weighted": delta,
                "facies_weight_sum": w_sum,
                "facies_row_count": len(frows),
            }
        )
    check_headers = [
        "method", "Experiment", "run_tag",
        "overall_MAE", "facies_weighted_MAE", "delta_overall_minus_facies_weighted",
        "facies_weight_sum", "facies_row_count",
    ]
    _write_csv(join(args.out_dir, "sheet5_consistency_check.csv"), check_headers, check_rows)

    print(f"[SUMMARY] saved: {join(args.out_dir, 'sheet1_overall.csv')}")
    print(f"[SUMMARY] saved: {join(args.out_dir, 'sheet2_facies.csv')}")
    print(f"[SUMMARY] saved: {join(args.out_dir, 'facies_mae_compare.csv')}")
    print(f"[SUMMARY] saved: {join(args.out_dir, 'sheet3_overall_wide.csv')}")
    print(f"[SUMMARY] saved: {join(args.out_dir, 'sheet4_facies_wide.csv')}")
    print(f"[SUMMARY] saved: {join(args.out_dir, 'sheet5_consistency_check.csv')}")


if __name__ == "__main__":
    main()
