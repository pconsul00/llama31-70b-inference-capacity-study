#!/usr/bin/env python3
"""
Repairs the three DCGM peak columns in manifest.csv (peak_sm_active,
peak_tensor_active, peak_dram_active) using the raw dcgm_*.log files already
saved for every load point -- no benchmarks are rerun.

Background: earlier versions of run_sweep.sh read these columns one field off
(they actually stored GR-engine, SM-occupancy and tensor peaks respectively,
rounded to 2 decimals). This recomputes them from the raw logs as:
    peak_sm_active     <- peak SMACT
    peak_tensor_active <- peak TENSO
    peak_dram_active   <- peak DRAMA
and adds avg_sm_active / avg_tensor_active / avg_dram_active (run-window averages).

The original manifest is backed up first (manifest.csv.bak). Nothing is deleted.

Usage:
    python fix_manifest.py --sweep-dir /path/to/sweep_results
Run it from the same folder as analyze_sweep.py (it reuses that file's log parser).
"""
import argparse
import os
import shutil

import pandas as pd

from analyze_sweep import parse_dcgm_log_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', required=True)
    args = ap.parse_args()

    manifest_path = os.path.join(args.sweep_dir, 'manifest.csv')
    backup_path = manifest_path + '.bak'
    if os.path.exists(backup_path):
        print(f"Backup already exists ({backup_path}); leaving it untouched.")
    else:
        shutil.copy2(manifest_path, backup_path)
        print(f"Backed up original manifest to {backup_path}")

    df = pd.read_csv(manifest_path)
    mapping = {
        'peak_sm_active': 'dcgm_peak_SMACT',
        'peak_tensor_active': 'dcgm_peak_TENSO',
        'peak_dram_active': 'dcgm_peak_DRAMA',
    }

    fixed, missing = 0, []
    new_vals = {col: [] for col in mapping}
    for _, row in df.iterrows():
        base = f"in{int(row['input_len'])}_out{int(row['output_len'])}_rate{int(row['rate'])}"
        parsed = parse_dcgm_log_full(args.sweep_dir, base)
        ok = all(src in parsed for src in mapping.values())
        for col, src in mapping.items():
            new_vals[col].append(round(parsed[src], 3) if ok else None)
        if ok:
            fixed += 1
        else:
            missing.append(base)

    for col in mapping:
        df[col] = new_vals[col]

    # Also add AVERAGE columns (time-average over the whole run window, all 4 GPUs,
    # numeric samples only), inserted just before start_time to match new sweeps.
    avg_mapping = {
        'avg_sm_active': 'dcgm_avg_SMACT',
        'avg_tensor_active': 'dcgm_avg_TENSO',
        'avg_dram_active': 'dcgm_avg_DRAMA',
    }
    avg_vals = {col: [] for col in avg_mapping}
    for _, row in df.iterrows():
        base = f"in{int(row['input_len'])}_out{int(row['output_len'])}_rate{int(row['rate'])}"
        parsed = parse_dcgm_log_full(args.sweep_dir, base)
        for col, src in avg_mapping.items():
            avg_vals[col].append(round(parsed[src], 3) if src in parsed else None)
    insert_at = list(df.columns).index('start_time') if 'start_time' in df.columns else len(df.columns)
    for k, col in enumerate(avg_mapping):
        if col in df.columns:
            df[col] = avg_vals[col]
        else:
            df.insert(insert_at + k, col, avg_vals[col])
    df.to_csv(manifest_path, index=False)

    print(f"Recomputed {fixed} of {len(df)} rows from raw DCGM logs.")
    if missing:
        print(f"{len(missing)} rows had no usable DCGM log; their three columns were left blank:")
        for m in missing[:20]:
            print("   ", m)
    print("Done. Rerun analyze_sweep.py to regenerate full_summary.csv.")


if __name__ == '__main__':
    main()
