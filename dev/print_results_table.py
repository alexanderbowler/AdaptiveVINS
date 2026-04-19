#!/usr/bin/env python3
"""
Print ATE and timing summary tables from ~/results/.

Usage:
    python3 print_results_table.py
    python3 print_results_table.py --ate-metric mean
    python3 print_results_table.py --ate-metric median   (default)
"""

import os
import re
import argparse

RESULTS_DIR = os.path.expanduser("~/results")

BAGS = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]

MODELS = ["supervins", "adaptivevinsV1", "adaptivevinsV2", "adaptivevinsV3", "adaptivevinsV4", "vinsfusion"]
MODEL_LABELS = {"supervins": "SuperVINS", "adaptivevinsV1": "AdaptiveVINS-V1", "adaptivevinsV2": "AdaptiveVINS-V2", "adaptivevinsV3": "AdaptiveVINS-V3", "adaptivevinsV4": "AdaptiveVINS-V4", "vinsfusion": "VINS-Fusion"}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Print ATE and timing tables")
parser.add_argument("--ate-metric", choices=["mean", "median"], default="median",
                    help="Statistic to show in the ATE table (default: median)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_summary(path):
    """Return (mean, median, valid, attempts) from summary.txt, or (None, None, None, None)."""
    if not os.path.exists(path):
        return None, None, None, None
    text = open(path).read()
    mean_m    = re.search(r"mean\s*:\s*([\d.]+)\s*m", text)
    median_m  = re.search(r"median\s*:\s*([\d.]+)\s*m", text)
    valid_m   = re.search(r"Valid runs:\s*(\d+)", text)
    attempts_m = re.search(r"Attempts:\s*(\d+)", text)
    mean     = float(mean_m.group(1))    if mean_m     else None
    median   = float(median_m.group(1))  if median_m   else None
    valid    = int(valid_m.group(1))     if valid_m    else None
    attempts = int(attempts_m.group(1))  if attempts_m else None
    return mean, median, valid, attempts


def parse_timing(path):
    """Return dict of {metric: mean_ms} from timing_summary.txt, or {}."""
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    metrics = {}
    for name in ("total_ms", "frontend_ms", "extraction_ms", "matching_ms", "estimation_ms"):
        m = re.search(rf"{re.escape(name)}\s+([\d.]+)", text)
        if m:
            metrics[name] = float(m.group(1))
    return metrics


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(title, headers, rows, col_width=16, label_width=20):
    total_width = label_width + col_width * len(headers)
    print()
    print("=" * total_width)
    print(f"  {title}")
    print("=" * total_width)
    print(f"  {'Bag':<{label_width}}" + "".join(f"{h:>{col_width}}" for h in headers))
    print("  " + "-" * (total_width - 2))
    prev_prefix = None
    for bag, cells in rows:
        # Blank line between MH / V1 / V2 groups
        prefix = bag[:2]
        if prev_prefix and prefix != prev_prefix:
            print()
        prev_prefix = prefix
        print(f"  {bag:<{label_width}}" + "".join(f"{c:>{col_width}}" for c in cells))
    print("=" * total_width)


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

ate_rows = []
timing_rows = []

for bag in BAGS:
    ate_cells = []
    timing_cells = []

    for model in MODELS:
        base = os.path.join(RESULTS_DIR, f"{bag}-{model}-rmse")
        mean, median, valid, attempts = parse_summary(os.path.join(base, "summary.txt"))
        timing = parse_timing(os.path.join(base, "timing_summary.txt"))

        val = median if args.ate_metric == "median" else mean
        if val is not None:
            failed = (attempts or 0) - (valid or 0)
            ate_cells.append(f"{val:.4f} ({valid}ok/{failed}f)")
        else:
            ate_cells.append("—")

        total = timing.get("total_ms")
        if total is not None:
            timing_cells.append(f"{total:.1f} ms")
        else:
            timing_cells.append("—")

    ate_rows.append((bag, ate_cells))
    timing_rows.append((bag, timing_cells))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

model_headers = [MODEL_LABELS[m] for m in MODELS]
metric_label = args.ate_metric.capitalize()

print_table(
    f"{metric_label} ATE (m)   format: {args.ate_metric} (Nok/Nfailed)",
    model_headers,
    ate_rows,
    col_width=22,
    label_width=20,
)

print_table(
    "Mean total latency per frame (frontend + backend)",
    model_headers,
    timing_rows,
    col_width=18,
    label_width=20,
)

# ---------------------------------------------------------------------------
# Detailed timing breakdown
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("  Detailed timing breakdown (mean ms per frame)")
print("=" * 80)

timing_cols = ["total_ms", "frontend_ms", "extraction_ms", "matching_ms", "estimation_ms"]
col_w = 12
label_w = 20

for model in MODELS:
    print(f"\n  {MODEL_LABELS[model]}")
    print(f"  {'Bag':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in ["total", "frontend", "extract", "match", "backend"]))
    print("  " + "-" * (label_w + col_w * len(timing_cols)))
    prev_prefix = None
    for bag in BAGS:
        prefix = bag[:2]
        if prev_prefix and prefix != prev_prefix:
            print()
        prev_prefix = prefix
        base = os.path.join(RESULTS_DIR, f"{bag}-{model}-rmse")
        timing = parse_timing(os.path.join(base, "timing_summary.txt"))
        if timing:
            cells = [f"{timing.get(c, 0):.1f}" for c in timing_cols]
        else:
            cells = ["—"] * len(timing_cols)
        print(f"  {bag:<{label_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

print()
