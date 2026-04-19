#!/usr/bin/env python3
"""
Analyze per-frame timing and GPU metrics for VINS systems.

Usage — named model presets (recommended):
    python3 analyze_timing.py --model supervins
    python3 analyze_timing.py --model vinsfusion
    python3 analyze_timing.py --model supervins --model vinsfusion
    python3 analyze_timing.py --model supervins --model vinsfusion --model adaptivevins

Usage — explicit paths (for custom runs):
    python3 analyze_timing.py --timing path/to/timing_log.csv --label MyRun
    python3 analyze_timing.py --timing a.csv --timing b.csv --label SuperVINS --label AdaptiveVINS

Options:
    --save          Archive current CSVs to results/ before loading (preserves previous runs)
    --no-plot       Skip matplotlib plots
    --no-gpu        Skip GPU summary even if gpu_log.csv exists for a model
"""

import argparse
import sys
import os
import shutil
import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# Model presets 

MODELS = {
    "supervins": {
        "label":   "SuperVINS",
        "timing":  "SuperVINS/time_consumption/timing_log.csv",
        "gpu":     "SuperVINS/time_consumption/gpu_log.csv",
    },
    "vinsfusion": {
        "label":   "VINS-Fusion",
        "timing":  "VINS-Fusion/time_consumption/timing_log.csv",
        "gpu":     "VINS-Fusion/time_consumption/gpu_log.csv",  # system baseline (CPU-only algo)
    },
    "adaptivevins": {
        "label":   "AdaptiveVINS",
        "timing":  "AdaptiveVINS/time_consumption/timing_log.csv",
        "gpu":     "AdaptiveVINS/time_consumption/gpu_log.csv",
    },
}

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found: pip install pandas")

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found — skipping plots (pip install matplotlib)")



# Helpers


def resolve_path(rel: str) -> str:
    """Resolve a path relative to the script directory."""
    return os.path.join(SCRIPT_DIR, rel)


def load_timing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "frontend_ms", "extraction_ms", "matching_ms", "estimation_ms"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"timing CSV missing columns: {missing}  (file: {path})")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_ms"] = df["frontend_ms"] + df["estimation_ms"]
    return df


def load_gpu(path: str) -> pd.DataFrame:
    """
    Parses nvidia-smi output:
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1
    Lines look like:  75 %, 2048 MiB
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                util = float(parts[0].replace("%", "").strip())
                mem  = float(parts[1].replace("MiB", "").strip())
                rows.append({"gpu_util_pct": util, "gpu_mem_mib": mem})
            except ValueError:
                continue
    if not rows:
        sys.exit(f"Could not parse any rows from GPU log: {path}")
    return pd.DataFrame(rows)


def load_aug_log(path: str) -> pd.DataFrame:
    """Load aug_log.csv produced by AdaptiveVINS hybrid frontend."""
    df = pd.read_csv(path)
    for col in ["difficulty", "n_classical", "augmented", "n_aug_features"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mean_track_len" in df.columns:
        df["mean_track_len"] = pd.to_numeric(df["mean_track_len"], errors="coerce")
    return df


def print_aug_summary(df: pd.DataFrame, label: str):
    n_frames    = len(df)
    n_aug       = int(df["augmented"].sum())
    aug_rate    = 100.0 * n_aug / n_frames if n_frames > 0 else 0.0
    aug_frames  = df[df["augmented"] == 1]
    mean_added  = aug_frames["n_aug_features"].mean() if len(aug_frames) > 0 else 0.0
    total_added = int(df["n_aug_features"].sum())
    mean_diff   = df["difficulty"].mean()
    mean_cls    = df["n_classical"].mean()

    print(f"\n{'='*60}")
    print(f"  Augmentation summary — {label}  ({n_frames} frames)")
    print(f"{'='*60}")
    print(f"  Frames augmented     : {n_aug} / {n_frames}  ({aug_rate:.1f}%)")
    print(f"  Mean features added  : {mean_added:.1f}  (on frames that fired)")
    print(f"  Total deep features  : {total_added}")
    print(f"  Mean difficulty score: {mean_diff:.3f}")
    print(f"  Mean classical feats : {mean_cls:.1f}")
    if "mean_track_len" in df.columns:
        print(f"  Mean track length    : {df['mean_track_len'].mean():.2f} frames")


def save_archive(model_key: str, preset: dict):
    """Copy current timing/GPU CSVs to results/ with a timestamp suffix."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for kind, rel in [("timing", preset["timing"]), ("gpu", preset.get("gpu"))]:
        if rel is None:
            continue
        src = resolve_path(rel)
        if os.path.exists(src):
            ext = os.path.splitext(rel)[1]
            dst = os.path.join(RESULTS_DIR, f"{model_key}_{kind}_{stamp}{ext}")
            shutil.copy2(src, dst)
            print(f"[save] archived {src} -> {dst}")
        else:
            print(f"[save] nothing to archive for {model_key} {kind} (file not found)")


def print_timing_summary(df: pd.DataFrame, label: str):
    cols = ["total_ms", "frontend_ms", "extraction_ms", "matching_ms", "estimation_ms"]
    stats = df[cols].agg(["mean", "std", "min", "median", "max"])

    width = 20
    print(f"\n{'='*60}")
    print(f"  Timing summary — {label}  ({len(df)} frames)")
    print(f"{'='*60}")
    header = f"{'':>{width}}" + "".join(f"{'mean':>10}{'std':>8}{'min':>8}{'med':>8}{'max':>8}")
    print(header)
    print("-" * (width + 42))
    for col in cols:
        row = stats[col]
        print(f"{col:>{width}}"
              f"{row['mean']:>10.2f}"
              f"{row['std']:>8.2f}"
              f"{row['min']:>8.2f}"
              f"{row['median']:>8.2f}"
              f"{row['max']:>8.2f}")
    print(f"{'(all values in ms)':>{width}}")


def print_gpu_summary(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  GPU summary — {label}  ({len(df)} samples)")
    print(f"{'='*60}")
    for col, unit in [("gpu_util_pct", "%"), ("gpu_mem_mib", "MiB")]:
        mean = df[col].mean()
        std  = df[col].std()
        mn   = df[col].min()
        mx   = df[col].max()
        print(f"  {col:<20}  mean={mean:6.1f}{unit}  std={std:5.1f}  "
              f"min={mn:6.1f}  max={mx:6.1f}")


def print_comparison_table(timing_dfs: list, labels: list):
    """Print a compact side-by-side comparison when multiple models are loaded."""
    if len(timing_dfs) < 2:
        return
    cols = ["total_ms", "frontend_ms", "extraction_ms", "matching_ms", "estimation_ms"]
    col_w = 14
    label_w = 16
    print(f"\n{'='*60}")
    print(f"  Side-by-side comparison (mean ± std, ms)")
    print(f"{'='*60}")
    header = f"{'metric':<{label_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * (label_w + col_w * len(labels)))
    for col in cols:
        row = f"{col:<{label_w}}"
        for df in timing_dfs:
            cell = f"{df[col].mean():.1f}±{df[col].std():.1f}"
            row += f"{cell:>{col_w}}"
        print(row)


def plot_timing(dfs: list, labels: list, out_dir: str = None):
    if out_dir is None:
        out_dir = os.path.join(SCRIPT_DIR, "results")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    #fig.suptitle("Per-frame timing breakdown", fontsize=14)

    cols = [
        ("total_ms",      "Total latency (frontend + backend)"),
        ("extraction_ms", "Feature extraction"),
        ("matching_ms",   "Feature matching / tracking"),
        ("estimation_ms", "Backend (Ceres optimization)"),
    ]

    for ax, (col, title) in zip(axes.flat, cols):
        for df, label in zip(dfs, labels):
            t = (df["timestamp"] - df["timestamp"].iloc[0]).to_numpy()
            ax.plot(t, df[col].to_numpy(), alpha=0.6, linewidth=0.8, label=label)
            ax.axhline(df[col].mean(), linestyle="--", linewidth=1.2,
                       label=f"{label} mean={df[col].mean():.1f}ms")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "timing_plot.png")
    plt.savefig(out, dpi=150)
    print(f"\n[info] plot saved to {out}")
    plt.show()


def plot_gpu(gpu_dfs: list, gpu_labels: list, out_dir: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    #fig.suptitle("GPU metrics", fontsize=14)

    for df, label in zip(gpu_dfs, gpu_labels):
        t = np.arange(len(df))
        ax1.plot(t, df["gpu_util_pct"].to_numpy(), alpha=0.7, linewidth=0.8, label=label)
        ax1.axhline(df["gpu_util_pct"].mean(), linestyle="--", linewidth=1.2)
        ax2.plot(t, df["gpu_mem_mib"].to_numpy(),  alpha=0.7, linewidth=0.8, label=label)
        ax2.axhline(df["gpu_mem_mib"].mean(),  linestyle="--", linewidth=1.2)

    ax1.set_title("GPU utilization (%)")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("GPU memory (MiB)")
    ax2.set_xlabel("sample")
    ax2.set_ylabel("MiB")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(SCRIPT_DIR, "results")
    out = os.path.join(out_dir, "gpu_plot.png")
    plt.savefig(out, dpi=150)
    print(f"[info] plot saved to {out}")
    plt.show()



# Main


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VINS timing + GPU metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Model presets available:",
            *[f"  {k:<14} → {v['timing']}" for k, v in MODELS.items()],
        ])
    )
    parser.add_argument("--model",   action="append", default=[],
                        choices=list(MODELS.keys()),
                        metavar="NAME",
                        help=f"named model preset: {{{', '.join(MODELS.keys())}}}")
    parser.add_argument("--timing",  action="append", default=[],
                        metavar="CSV",  help="explicit timing CSV path")
    parser.add_argument("--gpu",     action="append", default=[],
                        metavar="CSV",  help="explicit GPU log path (matches --timing order)")
    parser.add_argument("--label",   action="append", default=[],
                        metavar="NAME", help="display label for each --timing entry")
    parser.add_argument("--save",    action="store_true",
                        help="archive current CSVs to results/ before loading")
    parser.add_argument("--no-plot", action="store_true", help="skip plots")
    parser.add_argument("--no-gpu",  action="store_true", help="skip GPU summaries")
    parser.add_argument("--aug",     default=None, metavar="CSV",
                        help="aug_log.csv from AdaptiveVINS hybrid frontend")
    parser.add_argument("--output-dir", default=None,
                        help="directory to save plots (default: results/ next to this script)")
    args = parser.parse_args()

    # If nothing specified, default to all models that have data
    if not args.model and not args.timing:
        found = [k for k, v in MODELS.items() if os.path.exists(resolve_path(v["timing"]))]
        if not found:
            sys.exit("No timing CSVs found. Run a bag first, then re-run this script.\n"
                     "Expected locations:\n" +
                     "\n".join(f"  {resolve_path(v['timing'])}" for v in MODELS.values()))
        args.model = found
        print(f"[info] auto-detected models: {', '.join(found)}")

    # --- Build run list from --model presets ---
    timing_paths, gpu_paths, labels = [], [], []

    for key in args.model:
        preset = MODELS[key]
        t_path = resolve_path(preset["timing"])
        if not os.path.exists(t_path):
            sys.exit(f"[error] timing CSV not found for --model {key}:\n  {t_path}\n"
                     f"  Run the bag first (cd to the {key} package dir, then rosrun).")
        if args.save:
            save_archive(key, preset)
        timing_paths.append(t_path)
        labels.append(preset["label"])
        # GPU: use preset path unless --no-gpu
        g_rel = preset.get("gpu")
        if g_rel and not args.no_gpu:
            g_path = resolve_path(g_rel)
            gpu_paths.append(g_path if os.path.exists(g_path) else None)
        else:
            gpu_paths.append(None)

    # --- Append explicit --timing entries ---
    for i, t_path in enumerate(args.timing):
        if not os.path.exists(t_path):
            sys.exit(f"[error] timing CSV not found: {t_path}")
        timing_paths.append(t_path)
        labels.append(args.label[i] if i < len(args.label) else f"run{i+1}")
        g_path = args.gpu[i] if i < len(args.gpu) else None
        gpu_paths.append(g_path)

    # --- Load and summarise timing ---
    timing_dfs = []
    for path, label in zip(timing_paths, labels):
        print(f"[info] loading timing ({label}): {path}")
        timing_dfs.append(load_timing(path))

    for df, label in zip(timing_dfs, labels):
        print_timing_summary(df, label)

    print_comparison_table(timing_dfs, labels)

    # --- Load and summarise GPU ---
    gpu_dfs, gpu_labels = [], []
    for i, g_path in enumerate(gpu_paths):
        if g_path is None:
            continue
        print(f"\n[info] loading GPU log ({labels[i]}): {g_path}")
        gpu_dfs.append(load_gpu(g_path))
        gpu_labels.append(labels[i])
        print_gpu_summary(gpu_dfs[-1], gpu_labels[-1])

    # --- Augmentation summary (AdaptiveVINS hybrid only) ---
    if args.aug and os.path.exists(args.aug):
        print(f"\n[info] loading augmentation log: {args.aug}")
        aug_df = load_aug_log(args.aug)
        print_aug_summary(aug_df, labels[0] if labels else "model")

    # --- Plots ---
    out_dir = args.output_dir if args.output_dir else os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    if HAS_PLOT and not args.no_plot:
        plot_timing(timing_dfs, labels, out_dir)
        if gpu_dfs:
            plot_gpu(gpu_dfs, gpu_labels, out_dir)


if __name__ == "__main__":
    main()
