#!/usr/bin/env python3
"""
Analysis pipeline for the Llama 3.1 70B inference sweep.

Reads the sweep output (manifest.csv, bench_*.json, gpu_*.csv, dcgm_*.log) and, optionally, the
vLLM server logs, and writes:

  full_summary.csv              one row per load point, all measured and derived metrics
  latency_percentile_table.csv  p50/p90/p99 time to first token and time per output token
  vram_flatness_summary.txt     range of reserved VRAM across the sweep
  context_plot_rps50.png                  run time, power, GR engine and SM activity vs output length at 50 req/s
  throughput_tpot_by_output_len.png       throughput and time per output token vs output length
  gpu_activity_by_output_len.png          GR engine, SM, DRAM, tensor activity and SM occupancy vs output length
  kv_cache_fill_by_output_len.png         peak KV cache fill vs output length, one line per input length   (needs --logs-dir)
  vram_composition_stack_<mean|peak>.png  VRAM split into weights, idle KV pool and used KV cache         (needs --logs-dir or --weights-gib)

Usage:
    python analyze_sweep_clean.py --sweep-dir /path/to/sweep_results --out-dir ./analysis --logs-dir /path/to/logs
"""
import argparse
import json
import glob
import os
import re
import csv as csvmod
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless server, no display
import matplotlib.pyplot as plt

# Colors used by the VRAM composition chart (bottom to top: weights, idle KV pool, used KV cache)
COLOR_WEIGHTS, COLOR_IDLE, COLOR_USED = '#00274C', '#00B2A9', '#D86018'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-dir', required=True, help='Directory with manifest.csv, bench_*.json, gpu_*.csv, dcgm_*.log')
    p.add_argument('--out-dir', default='./analysis', help='Where to write tables and plots')
    p.add_argument('--logs-dir', default=None,
                   help='Directory containing vllm_server_*.log / vllm_*.log files. Used to parse the live KV cache '
                        'usage percentage (vLLM logs it directly) and the model weights size. Optional.')
    p.add_argument('--weights-gib', type=float, default=None,
                   help='Model weights per GPU in GiB, for the VRAM composition chart. If omitted it is read from the '
                        'server log ("Model loading took X GiB").')
    p.add_argument('--vram-agg', choices=['mean', 'peak', 'both'], default='mean',
                   help='Which VRAM composition chart to draw: average or peak KV cache use (default: mean).')
    return p.parse_args()


KV_LOG_LINE_RE = re.compile(
    r'(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2}).*?GPU KV cache usage:\s*([\d.]+)%'
)


def parse_server_logs_for_kv_usage(logs_dir, reference_year):
    """Parse every vllm server log file for vLLM's own 'GPU KV cache usage: X%'
    lines -- the live fill level of the pre-allocated KV cache pool, logged by
    vLLM's scheduler itself. This is the metric that actually varies with load,
    unlike nvidia-smi/DCGM's static reservation size. Log lines have no year,
    so reference_year is used to reconstruct full timestamps."""
    import datetime as dt
    entries = []
    log_files = glob.glob(os.path.join(logs_dir, 'vllm_server_*.log')) + \
                glob.glob(os.path.join(logs_dir, 'vllm_*.log'))
    log_files = sorted(set(log_files))
    for path in log_files:
        try:
            with open(path, errors='ignore') as f:
                for line in f:
                    m = KV_LOG_LINE_RE.search(line)
                    if m:
                        mo, da, hh, mi, se, pct = m.groups()
                        try:
                            ts = dt.datetime(reference_year, int(mo), int(da), int(hh), int(mi), int(se))
                            entries.append((ts, float(pct)))
                        except ValueError:
                            continue
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    return entries


def join_kv_cache_usage(df, kv_entries):
    """For each load point, find the server-log KV-usage samples that fall
    within [start_time, start_time + elapsed_s] and compute avg/peak."""
    if not kv_entries or 'start_time' not in df.columns:
        df['avg_kv_cache_pct'] = None
        df['peak_kv_cache_pct'] = None
        return df
    import datetime as dt
    timestamps = [e[0] for e in kv_entries]
    pcts = [e[1] for e in kv_entries]

    avg_vals, peak_vals = [], []
    for _, row in df.iterrows():
        try:
            start = dt.datetime.fromisoformat(str(row['start_time']))
            start = start.replace(tzinfo=None)
            end = start + dt.timedelta(seconds=float(row['elapsed_s']))
        except Exception:
            avg_vals.append(None)
            peak_vals.append(None)
            continue
        window = [p for t, p in zip(timestamps, pcts) if start <= t <= end]
        if window:
            avg_vals.append(sum(window) / len(window))
            peak_vals.append(max(window))
        else:
            avg_vals.append(None)
            peak_vals.append(None)
    df['avg_kv_cache_pct'] = avg_vals
    df['peak_kv_cache_pct'] = peak_vals
    return df


WEIGHTS_LOG_LINE_RE = re.compile(r'Model loading took\s+([\d.]+)\s*GiB')


def parse_model_weights_gib(logs_dir):
    """Pull the actual per-GPU model weights footprint directly from the
    server's own startup log line ('Model loading took X.XX GiB memory'),
    rather than assuming or hardcoding a figure."""
    log_files = glob.glob(os.path.join(logs_dir, 'vllm_server_*.log')) + \
                glob.glob(os.path.join(logs_dir, 'vllm_*.log'))
    for path in sorted(set(log_files)):
        try:
            with open(path, errors='ignore') as f:
                for line in f:
                    m = WEIGHTS_LOG_LINE_RE.search(line)
                    if m:
                        return float(m.group(1))
        except Exception:
            continue
    return None


def load_manifest(sweep_dir):
    df = pd.read_csv(os.path.join(sweep_dir, 'manifest.csv'))
    df = df[df['status'] == 'completed'].copy()
    df['expected_s'] = df['num_prompts'] / df['rate']
    df['overhead_ratio'] = df['elapsed_s'] / df['expected_s']  # kept for reference; not used for the boundary anymore
    # Achieved throughput: what the system actually delivered, in req/s.
    df['achieved_rps'] = df['num_prompts'] / df['elapsed_s']
    # Keep-up ratio: achieved vs configured. Below capacity this stays roughly
    # flat as rate increases (the system absorbs whatever you send). At/above
    # capacity, achieved throughput plateaus even as configured rate keeps
    # climbing, so this ratio falls.
    df['keep_up_ratio'] = df['achieved_rps'] / df['rate']
    return df


def enrich_with_json_percentiles(df, sweep_dir):
    """Pull p50/p90/p99 TTFT and TPOT out of each cell's bench JSON, if present."""
    records = []
    for _, row in df.iterrows():
        fpath = os.path.join(sweep_dir, row['filename'])
        rec = row.to_dict()
        try:
            with open(fpath) as f:
                d = json.load(f)
            # Field names vary slightly by vLLM version -- try common variants
            for key_variants, out_name in [
                (['ttfts', 'ttft_ms', 'time_to_first_token_ms'], 'ttft'),
                (['itls', 'tpots', 'tpot_ms', 'inter_token_latencies_ms'], 'tpot'),
            ]:
                arr = None
                for k in key_variants:
                    if k in d and isinstance(d[k], list):
                        arr = np.array(d[k])
                        break
                if arr is not None and len(arr) > 0:
                    rec[f'{out_name}_p50'] = float(np.percentile(arr, 50))
                    rec[f'{out_name}_p90'] = float(np.percentile(arr, 90))
                    rec[f'{out_name}_p99'] = float(np.percentile(arr, 99))
            # Fallback to summary means already in the JSON if arrays weren't found
            rec.setdefault('mean_ttft_ms', d.get('mean_ttft_ms'))
            rec.setdefault('mean_tpot_ms', d.get('mean_tpot_ms'))
            rec['output_throughput'] = d.get('output_throughput', d.get('output_token_throughput'))
            rec['peak_output_throughput'] = d.get('peak_output_throughput', d.get('peak_output_token_throughput'))
        except Exception as e:
            rec['parse_error'] = str(e)
        records.append(rec)
    return pd.DataFrame(records)


def parse_dcgm_log_full(sweep_dir, filename_base):
    """Parse a raw dcgm_*.log fully -- every field DCGM actually sampled,
    not just the three (sm_active, tensor_active, dram_active) that make it
    into the manifest. Returns avg/peak for each column found, skipping N/A rows."""
    path = os.path.join(sweep_dir, f"dcgm_{filename_base}.log")
    if not os.path.exists(path):
        return {}
    try:
        # dcgmi dmon output: header row starts with #Entity, second row is "ID",
        # then repeating data rows per GPU per sample. Whitespace-delimited.
        rows = []
        with open(path) as f:
            lines = f.readlines()
        header = None
        for line in lines:
            if line.startswith('#Entity'):
                header = line.replace('#', '').split()
                continue
            if line.strip().startswith('ID') or not line.strip():
                continue
            parts = line.split()
            if header and len(parts) >= len(header):
                # parts[0:2] = "GPU", "N" -- entity id is two tokens
                num_fields = len(header) - 1  # header includes "Entity", which isn't a data column
                vals = parts[2:2 + num_fields]
                rows.append(vals)
        if not header or not rows:
            return {}
        # The header row is "#Entity GRACT SMACT SMOCC ..." -- only "Entity" needs
        # dropping. "ID" is a separate line entirely, not part of this row.
        field_names = header[1:]
        result = {}
        for i, fname in enumerate(field_names):
            col_vals = []
            for row in rows:
                if i < len(row):
                    try:
                        col_vals.append(float(row[i]))
                    except ValueError:
                        continue  # N/A or non-numeric
            if col_vals:
                result[f'dcgm_avg_{fname}'] = sum(col_vals) / len(col_vals)
                result[f'dcgm_peak_{fname}'] = max(col_vals)
        return result
    except Exception:
        return {}


def join_full_dcgm_data(df, sweep_dir):
    """Join every DCGM field (not just the manifest's three) into the summary."""
    records = []
    for _, row in df.iterrows():
        base = f"in{row['input_len']}_out{row['output_len']}_rate{row['rate']}"
        dcgm_full = parse_dcgm_log_full(sweep_dir, base)
        rec = row.to_dict()
        rec.update(dcgm_full)
        records.append(rec)
    return pd.DataFrame(records)


def compute_gpu_averages_from_csv(df, sweep_dir):
    """Recompute avg/peak from the raw GPU CSVs -- redundant if manifest already
    has these columns, but useful as a cross-check / backfill for older cells."""
    avg_utils, peak_utils = [], []
    for _, row in df.iterrows():
        gpulog = os.path.join(sweep_dir, f"gpu_in{row['input_len']}_out{row['output_len']}_rate{row['rate']}.csv")
        if not os.path.exists(gpulog):
            avg_utils.append(np.nan)
            peak_utils.append(np.nan)
            continue
        try:
            g = pd.read_csv(gpulog)
            util_col = [c for c in g.columns if 'utilization.gpu' in c][0]
            vals = g[util_col].astype(str).str.extract(r'(\d+)').astype(float)[0]
            avg_utils.append(vals.mean())
            peak_utils.append(vals.max())
        except Exception:
            avg_utils.append(np.nan)
            peak_utils.append(np.nan)
    df['avg_util_recomputed'] = avg_utils
    df['peak_util_recomputed'] = peak_utils
    return df


def compute_derived_efficiency_metrics(df):
    """Metrics that only make sense once throughput, power, and DCGM data are joined."""
    if 'output_throughput' in df.columns and 'avg_power_draw_w' in df.columns:
        # tokens generated per joule of energy consumed (per-GPU average power x duration)
        total_energy_j = df['avg_power_draw_w'] * df['elapsed_s'] * 4  # x4 GPUs
        total_tokens = df['output_throughput'] * df['elapsed_s']
        df['tokens_per_joule'] = (total_tokens / total_energy_j).round(3)

    if 'output_throughput' in df.columns and 'dcgm_avg_SMACT' in df.columns:
        # throughput achieved per unit of SM activity -- a rough "useful work per busy-cycle" proxy
        df['tokens_per_sm_active'] = (df['output_throughput'] / df['dcgm_avg_SMACT']).round(2)

    if 'dcgm_avg_SMOCC' in df.columns and 'dcgm_avg_SMACT' in df.columns:
        # how "full" the GPU is when it IS busy -- occupancy relative to activity
        df['occupancy_when_active'] = (df['dcgm_avg_SMOCC'] / df['dcgm_avg_SMACT']).round(4)

    return df


def make_percentile_table(df, out_dir):
    cols = ['input_len', 'output_len', 'rate']
    for c in ['ttft_p50', 'ttft_p90', 'ttft_p99', 'tpot_p50', 'tpot_p90', 'tpot_p99',
              'mean_ttft_ms', 'mean_tpot_ms', 'output_throughput']:
        if c in df.columns:
            cols.append(c)
    df[cols].sort_values(['input_len', 'output_len', 'rate']).to_csv(
        os.path.join(out_dir, 'latency_percentile_table.csv'), index=False)


def make_vram_flatness_table(df, out_dir):
    cols = [c for c in ['input_len', 'output_len', 'rate', 'avg_mem_used_mib', 'peak_mem_used_mib'] if c in df.columns]
    if not cols:
        return None
    t = df[cols]
    summary = {
        'min_avg_mem': t['avg_mem_used_mib'].min() if 'avg_mem_used_mib' in t else np.nan,
        'max_avg_mem': t['avg_mem_used_mib'].max() if 'avg_mem_used_mib' in t else np.nan,
        'range_mib': (t['avg_mem_used_mib'].max() - t['avg_mem_used_mib'].min()) if 'avg_mem_used_mib' in t else np.nan,
    }
    with open(os.path.join(out_dir, 'vram_flatness_summary.txt'), 'w') as f:
        f.write(f"VRAM avg range across all cells: {summary['range_mib']:.0f} MiB "
                f"(min {summary['min_avg_mem']:.0f}, max {summary['max_avg_mem']:.0f})\n")
        f.write("Small range relative to total VRAM confirms static reservation, not per-cell scaling.\n")
    return summary


def plot_context_plot(df, out_dir, rate=50):
    """Consolidated view at a single fixed RPS: x-axis is output length
    (the dominant driver), one line per input length, across four metrics
    in one figure -- elapsed time, avg power, GR engine active, SM active."""
    subset = df[df['rate'] == rate]
    if len(subset) == 0:
        return

    metrics = [('elapsed_s', 'Elapsed time (s)'),
               ('avg_power_draw_w', 'Avg power (W)'),
               ('dcgm_avg_GRACT', 'GR engine active'),
               ('dcgm_avg_SMACT', 'SM active')]
    available = [(c, l) for c, l in metrics if c in subset.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax, (col, label) in zip(axes_flat, available):
        for il in sorted(subset['input_len'].unique()):
            d = subset[subset['input_len'] == il].sort_values('output_len').dropna(subset=[col])
            if len(d):
                ax.plot(d['output_len'], d[col], 'o-', label=f'input_len={il}', markersize=4)
        ax.set_xlabel('Output length (tokens)')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    fig.suptitle(f'Context Plot — Load Point RPS = {rate}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'context_plot_rps{rate}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_vram_composition(df, out_dir, weights_gib, agg='mean'):
    """Stacked bars, one per (output length, input length): model weights, idle reserved KV pool and used KV cache,
    in GiB per GPU. The total is the reserved VRAM (nearly constant); the split shows how much of the pool is used."""
    if 'avg_kv_cache_pct' not in df.columns or df['avg_kv_cache_pct'].isna().all():
        print("  (no KV cache usage data available -- skipping the VRAM composition chart)")
        return
    if weights_gib is None:
        print("  WARNING: model weights size unknown (no --weights-gib and no 'Model loading took' line in the logs); "
              "skipping the VRAM composition chart.")
        return
    weights_mib = weights_gib * 1024
    pct_col = 'peak_kv_cache_pct' if agg == 'peak' else 'avg_kv_cache_pct'
    mem_col = 'peak_mem_used_mib' if agg == 'peak' else 'avg_mem_used_mib'
    fn = 'max' if agg == 'peak' else 'mean'
    word = 'peak' if agg == 'peak' else 'average'

    outs = sorted(df['output_len'].unique()); ins = sorted(df['input_len'].unique())
    g = df.groupby(['output_len', 'input_len']).agg(total=(mem_col, fn), kv=(pct_col, fn)).reset_index()
    g['pool_gib'] = (g['total'] - weights_mib) / 1024
    g['used_gib'] = g['pool_gib'] * g['kv'] / 100.0
    g['idle_gib'] = g['pool_gib'] - g['used_gib']

    n = len(ins); bw = 0.8 / n
    fig, ax = plt.subplots(figsize=(13, 7.2))
    for i, il in enumerate(ins):
        sub = g[g['input_len'] == il].set_index('output_len').reindex(outs)
        x = [j + i * bw for j in range(len(outs))]
        ax.bar(x, [weights_gib] * len(outs), width=bw, color=COLOR_WEIGHTS, edgecolor='white', linewidth=0.4,
               label='Model weights' if i == 0 else None)
        ax.bar(x, sub['idle_gib'], width=bw, bottom=weights_gib, color=COLOR_IDLE, edgecolor='white', linewidth=0.4,
               label='Idle reserved KV pool' if i == 0 else None)
        ax.bar(x, sub['used_gib'], width=bw, bottom=weights_gib + sub['idle_gib'], color=COLOR_USED, edgecolor='white',
               linewidth=0.4, label=f'Used KV cache ({word})' if i == 0 else None)
        for xi in x:
            ax.annotate(f'in={il:,}', xy=(xi, 0), xytext=(0, -38), textcoords='offset points', ha='center', va='top',
                        fontsize=8.5, rotation=90, color='#444')
    ax.set_xticks([j + (n - 1) * bw / 2 for j in range(len(outs))]); ax.set_xticklabels([f'{o:,}' for o in outs], fontsize=11)
    ax.set_xlabel('Output length (tokens)', labelpad=68, fontsize=11); ax.set_ylabel('VRAM per GPU (GiB)', fontsize=11)
    ax.set_ylim(0, 100)  # headroom so the legend sits above the bars
    ax.set_title(f'VRAM Composition by Output Length x Input Length ({word} KV usage)\n'
                 'Each cluster = one output length; bars within a cluster = input lengths 128-1,024', fontsize=12)
    ax.legend(loc='upper center', ncol=3, fontsize=10, framealpha=0.95)
    fig.subplots_adjust(bottom=0.30, top=0.89, left=0.07, right=0.98)  # no tight_layout: it would override these margins
    plt.savefig(os.path.join(out_dir, f'vram_composition_stack_{agg}.png'), dpi=200); plt.close(fig)
    g.round(2).to_csv(os.path.join(out_dir, f'vram_composition_{agg}.csv'), index=False)


def plot_throughput_tpot(df, out_dir):
    """Mean output throughput (bars) and mean time per output token (line) by output length."""
    outs = sorted(df['output_len'].unique())
    g = df.groupby('output_len').agg(thr=('output_throughput', 'mean'), tpot=('mean_tpot_ms', 'mean')).reindex(outs)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(range(len(g)), g.thr.values, color='#2E4057', alpha=0.85, label='Output throughput (tokens/s)')
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f'{o:,}' for o in g.index]); ax.set_ylim(0, 2300)
    ax.set_xlabel('Output length (tokens)'); ax.set_ylabel('Output throughput (tokens/s)')
    ax2 = ax.twinx()
    ax2.plot(range(len(g)), g.tpot.values, color='#E8793C', marker='o', label='Time per output token (ms)')
    ax2.set_ylabel('Mean time per output token (ms)'); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_title('Throughput and per-token time by output length (mean of 25 load points each)', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'throughput_tpot_by_output_len.png'), dpi=200); plt.close(fig)


def plot_gpu_activity(df, out_dir):
    """GPU activity metrics (DCGM) by output length, averaged over all load points with that output length."""
    cols = [('dcgm_avg_GRACT', 'GR engine active', '#2E4057'), ('dcgm_avg_SMACT', 'SM active', '#E8793C'),
            ('dcgm_avg_DRAMA', 'DRAM active', '#6A994E'), ('dcgm_avg_TENSO', 'Tensor active', '#9B2226'),
            ('dcgm_avg_SMOCC', 'SM occupancy', '#8D99AE')]
    g = df.groupby('output_len')[[c for c, _, _ in cols]].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for c, lab, color in cols:
        ax.plot(range(len(g)), g[c].values, marker='o', ms=5, label=lab, color=color)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f'{o:,}' for o in g.index])
    ax.set_xlabel('Output length (tokens)'); ax.set_ylabel('Average fraction of sampled time (0-1)')
    ax.set_title('GPU activity metrics by output length (mean of all load points per output length)', fontsize=10)
    ax.legend(fontsize=8, loc='center right'); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'gpu_activity_by_output_len.png'), dpi=200); plt.close(fig)


def plot_kv_fill(df, out_dir):
    """Peak live KV cache fill (percent of the pool) by output length, one line per input length."""
    if 'peak_kv_cache_pct' not in df.columns or df['peak_kv_cache_pct'].isna().all():
        print("  (no KV cache usage data available -- skipping the KV fill chart)")
        return
    outs = sorted(df['output_len'].unique()); ins = sorted(df['input_len'].unique())
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, il in enumerate(ins):
        gg = df[df['input_len'] == il].groupby('output_len')['peak_kv_cache_pct'].max().reindex(outs)
        ax.plot(range(len(gg)), gg.values, marker='o', ms=4, color=cmap(k), label=f'input {il:,}')
    ax.axhline(100, color='k', ls='--', lw=0.8); ax.text(0.05, 101.5, 'KV pool full', fontsize=8)
    ax.set_xticks(range(len(outs))); ax.set_xticklabels([f'{o:,}' for o in outs]); ax.set_ylim(0, 108)
    ax.set_xlabel('Output length (tokens)'); ax.set_ylabel('Peak KV cache in use (% of pool)')
    ax.set_title('Live KV cache fill from vLLM server logs (maximum over the five rates)', fontsize=10)
    ax.legend(fontsize=8, title='Input length (tokens)', title_fontsize=8, loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'kv_cache_fill_by_output_len.png'), dpi=200); plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading manifest...")
    df = load_manifest(args.sweep_dir)
    print(f"  {len(df)} completed load points")

    print("Enriching with per-load-point JSON results...")
    df = enrich_with_json_percentiles(df, args.sweep_dir)

    print("Cross-checking GPU utilization from raw CSVs...")
    df = compute_gpu_averages_from_csv(df, args.sweep_dir)

    print("Joining full DCGM field data...")
    df = join_full_dcgm_data(df, args.sweep_dir)

    weights_gib = args.weights_gib
    if args.logs_dir:
        print(f"Parsing vLLM server logs in {args.logs_dir} for live KV cache usage...")
        ref_year = pd.to_datetime(df['start_time'].iloc[0]).year if len(df) else datetime.datetime.now().year
        kv_entries = parse_server_logs_for_kv_usage(args.logs_dir, ref_year)
        print(f"  found {len(kv_entries)} KV cache usage log lines")
        df = join_kv_cache_usage(df, kv_entries)
        if weights_gib is None:
            weights_gib = parse_model_weights_gib(args.logs_dir)
    else:
        print("No --logs-dir given -- skipping KV cache usage parsing.")

    print("Computing derived metrics (tokens per joule, etc.)...")
    df = compute_derived_efficiency_metrics(df)

    print("Writing tables...")
    df.to_csv(os.path.join(args.out_dir, 'full_summary.csv'), index=False)
    make_percentile_table(df, args.out_dir)
    make_vram_flatness_table(df, args.out_dir)

    print("Generating figures...")
    plot_context_plot(df, args.out_dir, rate=50)
    plot_throughput_tpot(df, args.out_dir)
    plot_gpu_activity(df, args.out_dir)
    plot_kv_fill(df, args.out_dir)
    for agg in (['mean', 'peak'] if args.vram_agg == 'both' else [args.vram_agg]):
        plot_vram_composition(df, args.out_dir, weights_gib, agg)

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
