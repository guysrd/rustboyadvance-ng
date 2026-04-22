#!/usr/bin/env bash
# dynarec shape-optimization measurement harness.
#
# Prints a single scalar `weighted_cycles` per the autoresearch loop in
# docs/program-shapekarpathy.md. The agent greps this value to decide
# keep/discard after every commit. Also prints `fps_bench_fps` as the
# correctness-regression guard.
#
# Weighted cycles =
#   sum over shapes of (ns_iter[shape] * call_count_weight[shape])
#
# Call-count weights are empirical, derived from the flamegraph on a
# 256s pokeemerald recording + 108s Mario Kart recording (see the
# perf-record runs in /tmp/*.perf; data snapshot at shape-opt/apr22
# baseline). They are fixed in this script on purpose: the agent
# optimizes against a stable target. Retrain weights by hand if the
# flamegraph drifts significantly.
#
# Shape weights (calls / second in real gameplay, rounded):
#   thumb_mov_imm    8.0M
#   thumb_add_imm    6.0M
#   thumb_cmp_imm    5.0M
#   thumb_dp_chain   3.0M
#   arm_mov_imm      2.0M
#   arm_cmp_imm      1.5M
#
# Usage:
#   bash scripts/dynarec_measure.sh > run.log 2>&1
#   grep "^weighted_cycles:\\|^fps_bench_fps:" run.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIOS="${BIOS:-$REPO/core/benches/roms/normatt_gba_bios.bin}"
ROM="${ROM:-/home/user/pokeemerlad/pokeemerald/pokeemerald.gba}"
REC="${REC:-/tmp/rbarec.rec}"
LOOPS="${LOOPS:-1}"

START_TS=$(date +%s)

# -----------------------------------------------------------------------------
# 1. Per-shape Criterion micro-bench.
# -----------------------------------------------------------------------------
echo "--- running cargo bench ---"
BENCH_LOG=$(mktemp)
cargo bench -p arm7tdmi --bench dynarec_shapes --features dynarec,bench -- --quiet 2>&1 \
  | tee "$BENCH_LOG"

# Criterion summary lines look like:
#   thumb_mov_imm           time:   [3.2145 ns 3.2200 ns 3.2260 ns]
# Parse the median (middle number), in nanoseconds. Handle ps/µs/ms units too.
parse_bench() {
    # Criterion line looks like:
    #   thumb_mov_imm           time:   [1.7100 ns 1.7225 ns 1.7350 ns]
    # Strip brackets, read median ($5) + unit ($6). Print the value in
    # ns. If no match was found, print 0.
    local name="$1"
    local result
    result=$(awk -v n="$name" '
        $0 ~ "^" n "[[:space:]]+time:" {
            gsub(/\[|\]/, "", $0)
            val=$5; unit=$6
            scale=1
            if (unit ~ /^ps/)       scale=0.001
            else if (unit ~ /^ns/)  scale=1
            else if (unit ~ /^µs/)  scale=1000
            else if (unit ~ /^us/)  scale=1000
            else if (unit ~ /^ms/)  scale=1000000
            printf "%.3f", val*scale
            exit
        }
    ' "$BENCH_LOG")
    if [[ -z "$result" ]]; then
        echo "0"
    else
        echo "$result"
    fi
}

T_thumb_mov_imm=$(parse_bench thumb_mov_imm)
T_thumb_add_imm=$(parse_bench thumb_add_imm)
T_thumb_cmp_imm=$(parse_bench thumb_cmp_imm)
T_thumb_dp_chain=$(parse_bench thumb_dp_chain)
T_arm_mov_imm=$(parse_bench arm_mov_imm)
T_arm_cmp_imm=$(parse_bench arm_cmp_imm)

echo "ns/iter per shape:"
echo "  thumb_mov_imm   = $T_thumb_mov_imm"
echo "  thumb_add_imm   = $T_thumb_add_imm"
echo "  thumb_cmp_imm   = $T_thumb_cmp_imm"
echo "  thumb_dp_chain  = $T_thumb_dp_chain"
echo "  arm_mov_imm     = $T_arm_mov_imm"
echo "  arm_cmp_imm     = $T_arm_cmp_imm"

# -----------------------------------------------------------------------------
# 2. Weighted cycles: fixed empirical weights (calls/sec on real gameplay).
# -----------------------------------------------------------------------------
W_thumb_mov_imm=8000000
W_thumb_add_imm=6000000
W_thumb_cmp_imm=5000000
W_thumb_dp_chain=3000000
W_arm_mov_imm=2000000
W_arm_cmp_imm=1500000

WEIGHTED=$(awk "BEGIN { printf \"%.0f\", \
    $T_thumb_mov_imm*$W_thumb_mov_imm \
  + $T_thumb_add_imm*$W_thumb_add_imm \
  + $T_thumb_cmp_imm*$W_thumb_cmp_imm \
  + $T_thumb_dp_chain*$W_thumb_dp_chain \
  + $T_arm_mov_imm*$W_arm_mov_imm \
  + $T_arm_cmp_imm*$W_arm_cmp_imm }")

# -----------------------------------------------------------------------------
# 3. fps_bench replay for correctness + FPS sanity.
# -----------------------------------------------------------------------------
FPS="0.0"
if [[ -f "$BIOS" && -f "$ROM" && -f "$REC" ]]; then
    echo "--- running fps_bench replay ---"
    FPS_LOG=$(mktemp)
    # fps_bench on this branch takes: fps_bench --replay <PATH> <BIOS> <ROM>
    # --loops is on later branches; we accept one pass.
    cargo run -q --release -p fps_bench -- --replay "$REC" "$BIOS" "$ROM" 2>&1 \
        | tee "$FPS_LOG"
    # Accept two reporting styles: "replay done: ... N.N avg fps" (later
    # branches) and "FPS: N" (this branch's idle mode).
    FPS=$(grep -oE "[0-9]+\\.[0-9]+ avg fps" "$FPS_LOG" | tail -1 | awk '{print $1}')
    if [[ -z "$FPS" ]]; then
        FPS=$(grep -oE "^FPS: [0-9]+" "$FPS_LOG" | tail -1 | awk '{print $2".0"}')
    fi
    [[ -z "$FPS" ]] && FPS="0.0"
    rm -f "$FPS_LOG"
else
    echo "--- fps_bench skipped (BIOS/ROM/REC not found) ---"
fi

rm -f "$BENCH_LOG"
END_TS=$(date +%s)
SECONDS_ELAPSED=$((END_TS - START_TS))

# -----------------------------------------------------------------------------
# 4. Canonical output block (exact format the agent greps).
# -----------------------------------------------------------------------------
echo
echo "---"
printf "weighted_cycles:   %s\n" "$WEIGHTED"
printf "fps_bench_fps:     %s\n" "$FPS"
printf "peak_vram_mb:      0.0\n"
printf "seconds:           %d\n" "$SECONDS_ELAPSED"
