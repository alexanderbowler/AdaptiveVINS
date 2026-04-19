#!/bin/bash
# run_benchmark.sh — launch a VINS benchmark run and save metric outputs
#
# Usage:
#   ./run_benchmark.sh --model <model> --bag <bag> [--visualize]
#
# Arguments:
#   --model       supervins | adaptivevins | vinsfusion
#   --bag         bag name (e.g. V1_01_easy) or full path to .bag file
#   --visualize   launch rviz; if omitted just runs roscore (headless)
#
# Example:
#   ./run_benchmark.sh --model supervins --bag V1_01_easy --visualize
#   ./run_benchmark.sh --model vinsfusion --bag V2_02_medium

CATKIN_WS=~/catkin_ws
SRC_DIR=$CATKIN_WS/src
RESULTS_DIR=~/results

MODEL=""
BAG_ARG=""
VISUALIZE=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2";    shift 2 ;;
        --bag)        BAG_ARG="$2";  shift 2 ;;
        --visualize)  VISUALIZE=true; shift   ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required args
# ---------------------------------------------------------------------------
if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required (supervins | adaptivevins | vinsfusion)"
    exit 1
fi
if [[ -z "$BAG_ARG" ]]; then
    echo "Error: --bag is required"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve bag path
# ---------------------------------------------------------------------------
resolve_bag() {
    local arg="$1"
    # Expand ~ manually in case it's in the string
    arg="${arg/#\~/$HOME}"

    # Full path given and exists
    if [[ -f "$arg" ]]; then
        echo "$arg"; return
    fi

    # Try name-only search under ~/data/euroc/
    local found
    found=$(find ~/data/euroc -name "${arg}.bag" 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        echo "$found"; return
    fi

    # Try with .bag already appended
    found=$(find ~/data/euroc -name "${arg}" 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        echo "$found"; return
    fi

    echo ""
}

BAG=$(resolve_bag "$BAG_ARG")
if [[ -z "$BAG" ]]; then
    echo "Error: could not find bag '$BAG_ARG'"
    echo "Searched: full path and ~/data/euroc/**/${BAG_ARG}.bag"
    exit 1
fi

BAG_NAME=$(basename "$BAG" .bag)

# ---------------------------------------------------------------------------
# Model-specific configuration
# ---------------------------------------------------------------------------
case $MODEL in
    supervins)
        PKG=supervins
        NODE=supervins_node
        RVIZ_LAUNCH="supervins supervins_rviz.launch"
        SRC=$SRC_DIR/SuperVINS
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/supervins_estimator/vio_output/vio.csv
        ;;
    adaptivevins)
        PKG=adaptivevins
        NODE=adaptivevins_node
        RVIZ_LAUNCH="adaptivevins adaptivevins_rviz.launch"
        SRC=$SRC_DIR/AdaptiveVINS
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/adaptivevins_estimator/vio_output/vio.csv
        ;;
    vinsfusion)
        PKG=vins
        NODE=vins_node
        RVIZ_LAUNCH="vins vins_rviz.launch"
        SRC=$SRC_DIR/VINS-Fusion
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/vins_estimator/vio_output/vio.csv
        ;;
    *)
        echo "Error: unknown model '$MODEL' (use supervins | adaptivevins | vinsfusion)"
        exit 1
        ;;
esac

RESULTS_SUBDIR=$RESULTS_DIR/${BAG_NAME}-${MODEL}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "======================================================="
echo "  Model      : $MODEL"
echo "  Bag        : $BAG_NAME"
echo "  Visualize  : $VISUALIZE"
echo "  Results    : $RESULTS_SUBDIR"
echo "======================================================="

# ---------------------------------------------------------------------------
# Source workspace
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source $CATKIN_WS/devel/setup.bash

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
ROS_PID=""
NODE_PID=""

cleanup() {
    echo ""
    echo "=== Shutting down... ==="
    [[ -n "$NODE_PID" ]] && kill "$NODE_PID" 2>/dev/null
    [[ -n "$ROS_PID"  ]] && kill "$ROS_PID"  2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Start ROS infrastructure
# ---------------------------------------------------------------------------
if $VISUALIZE; then
    echo "Launching rviz ($RVIZ_LAUNCH)..."
    roslaunch $RVIZ_LAUNCH &
    ROS_PID=$!
else
    echo "Starting roscore (headless)..."
    roscore &
    ROS_PID=$!
fi

echo "Waiting for rosmaster..."
until rostopic list &>/dev/null 2>&1; do sleep 0.5; done
echo "rosmaster ready."
sleep 2

# ---------------------------------------------------------------------------
# Launch estimator node
# (must cd into the repo root so timing_log.csv / gpu_log.csv land there)
# ---------------------------------------------------------------------------
echo "Starting $NODE..."
cd "$SRC"
rosrun $PKG $NODE "$CONFIG" &
NODE_PID=$!

echo "Waiting for node to initialize (8s)..."
sleep 8

# ---------------------------------------------------------------------------
# Play rosbag (blocks until bag is finished)
# ---------------------------------------------------------------------------
echo "Playing bag: $BAG"
rosbag play "$BAG"
echo "Bag finished."

# Wait for the estimator to finish writing vio.csv by monitoring its file size.
# Polls every 3 seconds; declares done when the file is non-empty and has not
# grown for 5 consecutive checks (~15s of stability). Gives up after MAX_WAIT.
VIO_SRC="${VIO_CSV/#\~/$HOME}"
if [[ "$MODEL" == "vinsfusion" ]]; then
    MAX_WAIT=30
else
    MAX_WAIT=300   # deep models can lag well behind real-time
fi

echo "Waiting for estimator queue to drain (max ${MAX_WAIT}s)..."
elapsed=0
prev_size=-1
stable_count=0
while [[ $elapsed -lt $MAX_WAIT ]]; do
    cur_size=$(stat -c%s "$VIO_SRC" 2>/dev/null || echo 0)
    if [[ "$cur_size" -gt 0 && "$cur_size" -eq "$prev_size" ]]; then
        (( stable_count++ ))
        if [[ $stable_count -ge 5 ]]; then
            echo "Queue drained (vio.csv stable at ${cur_size} bytes after ${elapsed}s)."
            break
        fi
    else
        stable_count=0
    fi
    prev_size=$cur_size
    sleep 3
    (( elapsed += 3 ))
done

if [[ $elapsed -ge $MAX_WAIT ]]; then
    echo "WARNING: hit ${MAX_WAIT}s timeout — vio.csv may be incomplete (${prev_size} bytes)."
fi

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
echo ""
echo "=== Saving results to $RESULTS_SUBDIR ==="
mkdir -p "$RESULTS_SUBDIR"

save_file() {
    local src="$1" dst_name="$2"
    # Expand ~ in src path
    src="${src/#\~/$HOME}"
    if [[ -f "$src" ]]; then
        cp "$src" "$RESULTS_SUBDIR/$dst_name"
        echo "  [OK] $dst_name"
    else
        echo "  [MISSING] $dst_name (expected at $src)"
    fi
}

save_file "$TIMING_CSV" "timing_log.csv"
save_file "$GPU_CSV"    "gpu_log.csv"
save_file "$VIO_CSV"    "vio.csv"

# ---------------------------------------------------------------------------
# Run analysis — plots and text summary go into the results subfolder
# ---------------------------------------------------------------------------
echo ""
echo "=== Running analysis ==="

ANALYSIS_TXT="$RESULTS_SUBDIR/analysis.txt"

# MPLBACKEND=Agg renders plots to file without needing a display
MPLBACKEND=Agg python3 "$SRC_DIR/analyze_timing.py" \
    --timing "$RESULTS_SUBDIR/timing_log.csv" \
    --gpu    "$RESULTS_SUBDIR/gpu_log.csv" \
    --label  "$MODEL" \
    --output-dir "$RESULTS_SUBDIR" \
    2>&1 | tee "$ANALYSIS_TXT"

echo ""
echo "  [OK] analysis.txt"
echo "  [OK] timing_plot.png"
echo "  [OK] gpu_plot.png"


# ---------------------------------------------------------------------------
# Run evo_ape — ATE against ground truth
# ---------------------------------------------------------------------------
echo ""
echo "=== Running evo_ape ==="

GROUND_TRUTH=$(find ~/data/euroc -name "data.tum" -path "*${BAG_NAME}*" 2>/dev/null | head -1)

if [[ -z "$GROUND_TRUTH" ]]; then
    echo "  [SKIPPED] Could not find data.tum for bag '$BAG_NAME'"
    echo "            Expected under ~/data/euroc/**/${BAG_NAME}/mav0/state_groundtruth_estimate0/"
else
    echo "  Ground truth: $GROUND_TRUTH"
    evo_ape tum "$GROUND_TRUTH" "$RESULTS_SUBDIR/vio.csv" \
        --align \
        --save_plot "$RESULTS_SUBDIR/ape_plot.pdf" \
        2>&1 | tee "$RESULTS_SUBDIR/evo_ape.txt"
    echo "  [OK] evo_ape.txt"
    echo "  [OK] ape_plot.pdf"
fi

echo ""
echo "=== Benchmark complete: ${BAG_NAME}-${MODEL} ==="
echo "    Results: $RESULTS_SUBDIR"
