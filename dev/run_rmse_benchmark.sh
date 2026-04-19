#!/bin/bash
# run_rmse_benchmark.sh — run a model on a bag until N good runs are collected,
# capped at MAX_ATTEMPTS total. Reports per-attempt results and statistics.
#
# Usage:
#   ./run_rmse_benchmark.sh --model <model> --bag <bag> [options]
#
# Arguments:
#   --model          supervins | adaptivevins | vinsfusion
#   --bag            bag name (e.g. V1_01_easy) or full path
#   --target         number of successful runs to collect (default: 5)
#   --max-attempts   maximum total attempts before giving up (default: 10)
#   --max-rmse       RMSE threshold above which a run is marked failed (default: 1.0 m)
#   --min-poses      minimum vio.csv lines to count as valid (default: 200)
#
# Example:
#   ./run_rmse_benchmark.sh --model supervins --bag V1_01_easy
#   ./run_rmse_benchmark.sh --model vinsfusion --bag V2_01_easy --target 3 --max-attempts 6

CATKIN_WS=~/catkin_ws
SRC_DIR=$CATKIN_WS/src
RESULTS_DIR=~/results

MODEL=""
BAG_ARG=""
TARGET_GOOD=5
MAX_ATTEMPTS=10
MAX_RMSE=1.0
MIN_POSES=200

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2";        shift 2 ;;
        --bag)          BAG_ARG="$2";      shift 2 ;;
        --target)       TARGET_GOOD="$2";  shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --max-rmse)     MAX_RMSE="$2";     shift 2 ;;
        --min-poses)    MIN_POSES="$2";    shift 2 ;;
        -h|--help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" || -z "$BAG_ARG" ]]; then
    echo "Error: --model and --bag are required"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve bag path
# ---------------------------------------------------------------------------
resolve_bag() {
    local arg="${1/#\~/$HOME}"
    [[ -f "$arg" ]] && { echo "$arg"; return; }
    local found
    found=$(find ~/data/euroc -name "${arg}.bag" 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    found=$(find ~/data/euroc -name "${arg}" 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    echo ""
}

BAG=$(resolve_bag "$BAG_ARG")
if [[ -z "$BAG" ]]; then
    echo "Error: could not find bag '$BAG_ARG'"
    exit 1
fi
BAG_NAME=$(basename "$BAG" .bag)

# ---------------------------------------------------------------------------
# Model-specific config
# ---------------------------------------------------------------------------
case $MODEL in
    supervins)
        PKG=supervins; NODE=supervins_node
        SRC=$SRC_DIR/SuperVINS
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/supervins_estimator/vio_output/vio.csv
        MAX_DRAIN=300
        ;;
    adaptivevins|adaptivevinsV1|adaptivevinsV2|adaptivevinsV3|adaptivevinsV4)
        PKG=adaptivevins; NODE=adaptivevins_node
        SRC=$SRC_DIR/AdaptiveVINS
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/adaptivevins_estimator/vio_output/vio.csv
        MAX_DRAIN=300
        ;;
    vinsfusion)
        PKG=vins; NODE=vins_node
        SRC=$SRC_DIR/VINS-Fusion
        CONFIG=$SRC/config/euroc/euroc_mono_imu_config.yaml
        TIMING_CSV=$SRC/time_consumption/timing_log.csv
        GPU_CSV=$SRC/time_consumption/gpu_log.csv
        VIO_CSV=$SRC/vins_estimator/vio_output/vio.csv
        MAX_DRAIN=30
        ;;
    *)
        echo "Error: unknown model '$MODEL'"
        exit 1
        ;;
esac

# Find ground truth
GROUND_TRUTH=$(find ~/data/euroc -name "data.tum" -path "*${BAG_NAME}*" 2>/dev/null | head -1)
if [[ -z "$GROUND_TRUTH" ]]; then
    echo "Error: could not find ground truth data.tum for '$BAG_NAME'"
    exit 1
fi

RESULTS_SUBDIR=$RESULTS_DIR/${BAG_NAME}-${MODEL}-rmse
mkdir -p "$RESULTS_SUBDIR"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "======================================================="
echo "  Model        : $MODEL"
echo "  Bag          : $BAG_NAME"
echo "  Target good  : $TARGET_GOOD"
echo "  Max attempts : $MAX_ATTEMPTS"
echo "  Max RMSE     : ${MAX_RMSE} m  (above = failed)"
echo "  Min poses    : ${MIN_POSES} lines (below = failed)"
echo "  Results      : $RESULTS_SUBDIR"
echo "======================================================="
echo ""

source $CATKIN_WS/devel/setup.bash

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
declare -a RMSE_VALUES    # RMSE values for successful attempts
declare -a ALL_RMSE       # RMSE or "FAILED" for every attempt (in order)
declare -a ALL_REASONS    # failure reason or "OK" for every attempt
ATTEMPT=0
ROS_PID=""
NODE_PID=""

cleanup() {
    [[ -n "$NODE_PID" ]] && kill "$NODE_PID" 2>/dev/null
    [[ -n "$ROS_PID"  ]] && kill "$ROS_PID"  2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: parse RMSE from evo_ape output
# ---------------------------------------------------------------------------
parse_rmse() {
    grep -E "^\s+rmse\s+" "$1" | awk '{print $2}'
}

# ---------------------------------------------------------------------------
# Start roscore once — kept alive across all attempts
# ---------------------------------------------------------------------------
VIO_SRC="${VIO_CSV/#\~/$HOME}"

roscore &
ROS_PID=$!
until rostopic list &>/dev/null 2>&1; do sleep 0.5; done
sleep 1
echo "  roscore started (PID $ROS_PID)"
echo ""

ensure_roscore() {
    # Restart roscore if it died unexpectedly
    if ! kill -0 "$ROS_PID" 2>/dev/null; then
        echo "  [WARN] roscore died — restarting..."
        roscore &
        ROS_PID=$!
        until rostopic list &>/dev/null 2>&1; do sleep 0.5; done
        sleep 1
    fi
}

# ---------------------------------------------------------------------------
# Run loop — keep going until TARGET_GOOD successes or MAX_ATTEMPTS reached
# ---------------------------------------------------------------------------
while [[ ${#RMSE_VALUES[@]} -lt $TARGET_GOOD && $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
    (( ATTEMPT++ ))
    N_VALID=${#RMSE_VALUES[@]}
    N_FAILED=$(( ATTEMPT - 1 - N_VALID ))

    ATTEMPT_DIR="$RESULTS_SUBDIR/attempt${ATTEMPT}"
    mkdir -p "$ATTEMPT_DIR"

    echo "---------------------------------------------------"
    echo "  Attempt ${ATTEMPT} / ${MAX_ATTEMPTS}   (good: ${N_VALID}/${TARGET_GOOD}  failed so far: ${N_FAILED})"
    echo "---------------------------------------------------"

    ensure_roscore

    # Start estimator
    cd "$SRC"
    rosrun $PKG $NODE "$CONFIG" &
    NODE_PID=$!
    echo "  Waiting for node to initialize (8s)..."
    sleep 8

    # Check node is still alive after init
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        echo "  [FAILED] node crashed during initialization"
        ALL_RMSE+=("FAILED")
        ALL_REASONS+=("node crashed during initialization")
        echo "FAILED: node crashed during initialization" > "$ATTEMPT_DIR/result.txt"
        NODE_PID=""
        sleep 1; continue
    fi

    # Play bag in background so we can monitor the node simultaneously
    echo "  Playing bag..."
    rosbag play "$BAG" -q &
    BAG_PID=$!

    NODE_DIED=0
    while kill -0 "$BAG_PID" 2>/dev/null; do
        if ! kill -0 "$NODE_PID" 2>/dev/null; then
            echo "  [WARN] node crashed mid-run — aborting bag early"
            kill "$BAG_PID" 2>/dev/null
            NODE_DIED=1
            break
        fi
        sleep 2
    done
    wait "$BAG_PID" 2>/dev/null
    echo "  Bag finished."

    if [[ $NODE_DIED -eq 1 ]]; then
        ALL_RMSE+=("FAILED")
        ALL_REASONS+=("node crashed during bag playback")
        echo "FAILED: node crashed during bag playback" > "$ATTEMPT_DIR/result.txt"
        NODE_PID=""
        sleep 1; continue
    fi

    # Wait for queue to drain
    elapsed=0; prev_size=-1; stable_count=0
    while [[ $elapsed -lt $MAX_DRAIN ]]; do
        cur_size=$(stat -c%s "$VIO_SRC" 2>/dev/null || echo 0)
        if [[ "$cur_size" -gt 0 && "$cur_size" -eq "$prev_size" ]]; then
            (( stable_count++ ))
            [[ $stable_count -ge 5 ]] && break
        else
            stable_count=0
        fi
        prev_size=$cur_size
        sleep 3; (( elapsed += 3 ))
    done

    # Kill node only — roscore stays alive for the next attempt
    kill "$NODE_PID" 2>/dev/null; wait "$NODE_PID" 2>/dev/null; NODE_PID=""

    # Save CSVs for this attempt
    cp "$VIO_SRC"    "$ATTEMPT_DIR/vio.csv"         2>/dev/null
    cp "$TIMING_CSV" "$ATTEMPT_DIR/timing_log.csv"  2>/dev/null
    cp "$GPU_CSV"    "$ATTEMPT_DIR/gpu_log.csv"     2>/dev/null
    cp "${TIMING_CSV%timing_log.csv}aug_log.csv" "$ATTEMPT_DIR/aug_log.csv" 2>/dev/null

    # ---------------------------------------------------------------------------
    # Evaluate this attempt
    # ---------------------------------------------------------------------------
    POSE_COUNT=$(wc -l < "$ATTEMPT_DIR/vio.csv" 2>/dev/null || echo 0)
    FAIL_REASON=""

    if [[ "$POSE_COUNT" -lt "$MIN_POSES" ]]; then
        FAIL_REASON="only ${POSE_COUNT} poses (< ${MIN_POSES} minimum)"
    else
        APE_OUT="$ATTEMPT_DIR/evo_ape.txt"
        evo_ape tum "$GROUND_TRUTH" "$ATTEMPT_DIR/vio.csv" \
            --align 2>&1 > "$APE_OUT" || true

        RMSE=$(parse_rmse "$APE_OUT")

        if [[ -z "$RMSE" ]]; then
            FAIL_REASON="evo_ape failed to produce output"
        else
            IS_FAIL=$(python3 -c "print('yes' if float('$RMSE') > float('$MAX_RMSE') else 'no')")
            if [[ "$IS_FAIL" == "yes" ]]; then
                FAIL_REASON="RMSE ${RMSE} m > threshold ${MAX_RMSE} m (tracking diverged)"
            fi
        fi
    fi

    if [[ -n "$FAIL_REASON" ]]; then
        echo "  [FAILED] ${FAIL_REASON}"
        ALL_RMSE+=("FAILED")
        ALL_REASONS+=("$FAIL_REASON")
        echo "FAILED: $FAIL_REASON" > "$ATTEMPT_DIR/result.txt"
    else
        echo "  [OK] RMSE = ${RMSE} m  (${POSE_COUNT} poses)"
        RMSE_VALUES+=("$RMSE")
        ALL_RMSE+=("$RMSE")
        ALL_REASONS+=("OK")
        echo "RMSE: $RMSE" > "$ATTEMPT_DIR/result.txt"
    fi

    echo ""
    sleep 1
done

# Shut down roscore now that all attempts are done
kill "$ROS_PID" 2>/dev/null; wait "$ROS_PID" 2>/dev/null; ROS_PID=""

TOTAL_ATTEMPTS=$ATTEMPT
N_VALID=${#RMSE_VALUES[@]}
N_FAILED=$(( TOTAL_ATTEMPTS - N_VALID ))

# ---------------------------------------------------------------------------
# Aggregate timing analysis across all attempts
# ---------------------------------------------------------------------------
COMBINED_TIMING="$RESULTS_SUBDIR/timing_combined.csv"
COMBINED_GPU="$RESULTS_SUBDIR/gpu_combined.csv"
COMBINED_AUG="$RESULTS_SUBDIR/aug_combined.csv"
rm -f "$COMBINED_GPU"

first_timing=1
first_aug=1
for (( i=1; i<=TOTAL_ATTEMPTS; i++ )); do
    t="$RESULTS_SUBDIR/attempt${i}/timing_log.csv"
    g="$RESULTS_SUBDIR/attempt${i}/gpu_log.csv"
    a="$RESULTS_SUBDIR/attempt${i}/aug_log.csv"
    if [[ -f "$t" && $(wc -l < "$t") -gt 1 ]]; then
        if [[ $first_timing -eq 1 ]]; then
            cat "$t" > "$COMBINED_TIMING"
            first_timing=0
        else
            tail -n +2 "$t" >> "$COMBINED_TIMING"
        fi
    fi
    [[ -f "$g" ]] && cat "$g" >> "$COMBINED_GPU"
    if [[ -f "$a" && $(wc -l < "$a") -gt 1 ]]; then
        if [[ $first_aug -eq 1 ]]; then
            cat "$a" > "$COMBINED_AUG"
            first_aug=0
        else
            tail -n +2 "$a" >> "$COMBINED_AUG"
        fi
    fi
done

if [[ $first_timing -eq 0 ]]; then
    TIMING_CMD=(python3 "$SRC_DIR/analyze_timing.py"
        --timing "$COMBINED_TIMING"
        --label "$MODEL"
        --output-dir "$RESULTS_SUBDIR")
    [[ -s "$COMBINED_GPU" ]] && TIMING_CMD+=(--gpu "$COMBINED_GPU")
    [[ -s "$COMBINED_AUG" ]] && TIMING_CMD+=(--aug "$COMBINED_AUG")
    echo "  Running timing analysis..."
    MPLBACKEND=Agg "${TIMING_CMD[@]}" 2>&1 | tee "$RESULTS_SUBDIR/timing_summary.txt"
fi

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
echo "======================================================="
echo "  RESULTS: ${BAG_NAME} — ${MODEL}"
echo "======================================================="
echo ""

# Per-attempt table
printf "  %-9s %-12s %s\n" "Attempt" "RMSE (m)" "Reason"
printf "  %-9s %-12s %s\n" "-------" "--------" "------"
for (( i=0; i<TOTAL_ATTEMPTS; i++ )); do
    printf "  %-9s %-12s %s\n" "$((i+1))" "${ALL_RMSE[$i]}" "${ALL_REASONS[$i]}"
done

echo ""
echo "  Total attempts : ${TOTAL_ATTEMPTS} / ${MAX_ATTEMPTS}"
echo "  Valid runs     : ${N_VALID} / ${TARGET_GOOD} (target)"
echo "  Failed runs    : ${N_FAILED} / ${TOTAL_ATTEMPTS} attempts"

if [[ $N_VALID -lt $TARGET_GOOD ]]; then
    echo "  NOTE: target of ${TARGET_GOOD} good runs not reached (hit attempt cap)"
fi

if [[ $N_VALID -gt 0 ]]; then
    VALS=$(IFS=,; echo "${RMSE_VALUES[*]}")
    python3 - <<EOF
vals = [float(x) for x in "$VALS".split(",")]
vals.sort()
n = len(vals)
mean = sum(vals) / n
median = vals[n//2] if n % 2 == 1 else (vals[n//2-1] + vals[n//2]) / 2
variance = sum((x - mean)**2 for x in vals) / n
std = variance ** 0.5

print(f"\n  --- Statistics over {n} good run(s) ---")
print(f"  mean   : {mean:.4f} m")
print(f"  median : {median:.4f} m")
print(f"  std    : {std:.4f} m")
print(f"  min    : {min(vals):.4f} m")
print(f"  max    : {max(vals):.4f} m")
EOF
else
    echo ""
    echo "  WARNING: all attempts failed — check initialization or lower --max-rmse threshold"
fi

echo ""
echo "  Per-attempt results: $RESULTS_SUBDIR/attempt*/"
echo "======================================================="

# ---------------------------------------------------------------------------
# Save summary to file
# ---------------------------------------------------------------------------
{
    echo "Model:        $MODEL"
    echo "Bag:          $BAG_NAME"
    echo "Attempts:     $TOTAL_ATTEMPTS / $MAX_ATTEMPTS"
    echo "Valid runs:   $N_VALID / $TARGET_GOOD (target)"
    echo "Failed runs:  $N_FAILED / $TOTAL_ATTEMPTS attempts"
    echo ""
    printf "%-9s %-12s %s\n" "Attempt" "RMSE (m)" "Reason"
    printf "%-9s %-12s %s\n" "-------" "--------" "------"
    for (( i=0; i<TOTAL_ATTEMPTS; i++ )); do
        printf "%-9s %-12s %s\n" "$((i+1))" "${ALL_RMSE[$i]}" "${ALL_REASONS[$i]}"
    done
    if [[ $N_VALID -gt 0 ]]; then
        VALS=$(IFS=,; echo "${RMSE_VALUES[*]}")
        python3 - <<EOF
vals = [float(x) for x in "$VALS".split(",")]
vals.sort()
n = len(vals)
mean = sum(vals) / n
median = vals[n//2] if n % 2 == 1 else (vals[n//2-1] + vals[n//2]) / 2
variance = sum((x - mean)**2 for x in vals) / n
std = variance ** 0.5
print(f"\nStatistics over {n} good run(s):")
print(f"  mean   : {mean:.4f} m")
print(f"  median : {median:.4f} m")
print(f"  std    : {std:.4f} m")
print(f"  min    : {min(vals):.4f} m")
print(f"  max    : {max(vals):.4f} m")
EOF
    fi
} > "$RESULTS_SUBDIR/summary.txt"

echo "  Summary written to: $RESULTS_SUBDIR/summary.txt"
