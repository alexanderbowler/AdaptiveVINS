#!/bin/bash
# run_full_sweep.sh — run all models against all EuRoC vicon_room1/2 bags
# and print a final RMSE comparison table.
#
# Usage:
#   ./run_full_sweep.sh [--target 5] [--max-attempts 10] [--max-rmse 1.0]
#                       [--models "supervins adaptivevins vinsfusion"]
#                       [--bags "V1_01_easy V1_02_medium ..."]
#
# Defaults:
#   --target 5  --max-attempts 10
#   All three models, all 11 EuRoC bags (MH 1-5, V1 1-3, V2 1-3)

SRC_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR=~/results
RMSE_SCRIPT="$SRC_DIR/run_rmse_benchmark.sh"

TARGET_GOOD=5
MAX_ATTEMPTS=10
MAX_RMSE=1.0
MIN_POSES=200
MODELS_ARG=""
BAGS_ARG=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)       TARGET_GOOD="$2";  shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --max-rmse)     MAX_RMSE="$2";     shift 2 ;;
        --min-poses)    MIN_POSES="$2";    shift 2 ;;
        --models)       MODELS_ARG="$2";   shift 2 ;;
        --bags)         BAGS_ARG="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODELS_ARG" ]]; then
    MODELS_ARG="supervins adaptivevinsV1 adaptivevinsV2 adaptivevinsV3 adaptivevinsV4 vinsfusion"
fi
if [[ -z "$BAGS_ARG" ]]; then
    BAGS_ARG="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
fi

read -ra MODELS <<< "$MODELS_ARG"
read -ra BAGS   <<< "$BAGS_ARG"

N_MODELS=${#MODELS[@]}
N_BAGS=${#BAGS[@]}
COMBOS=$(( N_MODELS * N_BAGS ))

echo "======================================================="
echo "  FULL SWEEP"
echo "  Models       : ${MODELS[*]}"
echo "  Bags         : ${BAGS[*]}"
echo "  Target good  : $TARGET_GOOD per combination"
echo "  Max attempts : $MAX_ATTEMPTS per combination"
echo "  Combinations : $COMBOS"
echo "======================================================="
echo ""

COMBO=0
SKIPPED=0
for BAG in "${BAGS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        (( COMBO++ ))
        SUMMARY="${RESULTS_DIR/#\~/$HOME}/${BAG}-${MODEL}-rmse/summary.txt"
        if [[ -f "$SUMMARY" ]]; then
            echo "  [$COMBO/$COMBOS]  $MODEL  ×  $BAG  — SKIPPING (results exist)"
            (( SKIPPED++ ))
            continue
        fi
        echo ""
        echo "##################################################"
        echo "  [$COMBO/$COMBOS]  $MODEL  ×  $BAG"
        echo "##################################################"
        bash "$RMSE_SCRIPT" \
            --model        "$MODEL" \
            --bag          "$BAG" \
            --target       "$TARGET_GOOD" \
            --max-attempts "$MAX_ATTEMPTS" \
            --max-rmse     "$MAX_RMSE" \
            --min-poses    "$MIN_POSES"
    done
done

echo ""
echo "  Skipped $SKIPPED / $COMBOS combinations (results already existed)"

# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------
echo ""
echo "======================================================="
echo "  SWEEP COMPLETE — median RMSE (m)"
echo "  Target: $TARGET_GOOD good runs, cap: $MAX_ATTEMPTS attempts"
echo "======================================================="

BAG_W=22
COL_W=22

printf "\n  %-${BAG_W}s" "Bag"
for MODEL in "${MODELS[@]}"; do
    printf "%-${COL_W}s" "$MODEL"
done
echo ""

printf "  %-${BAG_W}s" "---"
for MODEL in "${MODELS[@]}"; do
    printf "%-${COL_W}s" "----------------------"
done
echo ""

for BAG in "${BAGS[@]}"; do
    printf "  %-${BAG_W}s" "$BAG"
    for MODEL in "${MODELS[@]}"; do
        SUMMARY="$RESULTS_DIR/${BAG}-${MODEL}-rmse/summary.txt"
        if [[ -f "$SUMMARY" ]]; then
            MEDIAN=$(grep "median" "$SUMMARY" | awk '{print $3}' | head -1)
            VALID=$(grep "^Valid runs:" "$SUMMARY" | awk '{print $3}')
            ATTEMPTS=$(grep "^Attempts:" "$SUMMARY" | awk '{print $2}')
            FAILED=$(grep "^Failed runs:" "$SUMMARY" | awk '{print $3}')
            if [[ -n "$MEDIAN" ]]; then
                # e.g.  0.0972 (3ok/1fail/4att)
                printf "%-${COL_W}s" "${MEDIAN} ${VALID}ok/${FAILED}fail"
            else
                printf "%-${COL_W}s" "ALL FAILED (${ATTEMPTS}att)"
            fi
        else
            printf "%-${COL_W}s" "NO DATA"
        fi
    done
    echo ""
done

echo ""
echo "  Format: median (Nok/Nfail)"
echo "  Full results: $RESULTS_DIR/<bag>-<model>-rmse/"
echo "======================================================="
