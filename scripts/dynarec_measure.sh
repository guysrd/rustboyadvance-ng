#!/usr/bin/env bash
# dynarec shape-optimization measurement harness.
#
# Prints a single scalar `weighted_cycles` per the autoresearch loop in
# docs/program-shapekarpathy.md. The agent greps this value to decide
# keep/discard after every commit. Also prints a per-ROM FPS sanity pair
# so regressions that only show up on one game's code paths still fail
# the keep/discard gate.
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
# FPS sanity runs the real SDL frontend end-to-end (video render + all
# GPU/CPU subsystems, `--no-audio` to silence the audio path noise).
# Two ROMs, both must stay within their regression budget — this is the
# "each game is a separate measure" half of the gate. fps_bench was
# fast but headless; the SDL binary catches GPU-side regressions that
# never show up in a core-only replay.
#
# Shape weights (calls / second in real gameplay, rounded):
#   thumb_mov_imm    8.0M
#   thumb_add_imm    6.0M
#   thumb_cmp_imm    5.0M
#   thumb_dp_chain   3.0M
#   arm_mov_imm      2.0M
#   arm_cmp_imm      1.5M
#   shift_pair_sxtb  2.5M
#
# Usage:
#   bash scripts/dynarec_measure.sh > run.log 2>&1
#   grep "^weighted_cycles:\\|^pokeemerald_fps:\\|^mario_kart_fps:" run.log
#
# Env overrides:
#   BIOS, POKEEMERALD_ROM, MARIO_KART_ROM, POKEEMERALD_REC, MARIO_KART_REC
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIOS="${BIOS:-$REPO/core/benches/roms/normatt_gba_bios.bin}"
POKEEMERALD_ROM="${POKEEMERALD_ROM:-/home/user/pokeemerlad/pokeemerald/pokeemerald.gba}"
MARIO_KART_ROM="${MARIO_KART_ROM:-/tmp/mks/Mario Kart - Super Circuit (USA).gba}"
POKEEMERALD_REC="${POKEEMERALD_REC:-/tmp/rbarec.rec}"
MARIO_KART_REC="${MARIO_KART_REC:-/tmp/mks.rec}"

START_TS=$(date +%s)

# -----------------------------------------------------------------------------
# 1. Per-shape Criterion micro-bench (unchanged — this is the primary scalar).
# -----------------------------------------------------------------------------
echo "--- running cargo bench ---"
BENCH_LOG=$(mktemp)
cargo bench -p arm7tdmi --bench dynarec_shapes --features dynarec,bench -- --quiet 2>&1 \
  | tee "$BENCH_LOG"

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
T_shift_pair_sxtb=$(parse_bench shift_pair_sxtb)

echo "ns/iter per shape:"
echo "  thumb_mov_imm   = $T_thumb_mov_imm"
echo "  thumb_add_imm   = $T_thumb_add_imm"
echo "  thumb_cmp_imm   = $T_thumb_cmp_imm"
echo "  thumb_dp_chain  = $T_thumb_dp_chain"
echo "  arm_mov_imm     = $T_arm_mov_imm"
echo "  arm_cmp_imm     = $T_arm_cmp_imm"
echo "  shift_pair_sxtb = $T_shift_pair_sxtb"

# -----------------------------------------------------------------------------
# 2. Weighted cycles: fixed empirical weights (calls/sec on real gameplay).
# -----------------------------------------------------------------------------
W_thumb_mov_imm=8000000
W_thumb_add_imm=6000000
W_thumb_cmp_imm=5000000
W_thumb_dp_chain=3000000
W_arm_mov_imm=2000000
W_arm_cmp_imm=1500000
W_shift_pair_sxtb=2500000

WEIGHTED=$(awk "BEGIN { printf \"%.0f\", \
    $T_thumb_mov_imm*$W_thumb_mov_imm \
  + $T_thumb_add_imm*$W_thumb_add_imm \
  + $T_thumb_cmp_imm*$W_thumb_cmp_imm \
  + $T_thumb_dp_chain*$W_thumb_dp_chain \
  + $T_arm_mov_imm*$W_arm_mov_imm \
  + $T_arm_cmp_imm*$W_arm_cmp_imm \
  + $T_shift_pair_sxtb*$W_shift_pair_sxtb }")

# -----------------------------------------------------------------------------
# 3. Per-ROM SDL replay: full frontend, video on, --no-audio.
# -----------------------------------------------------------------------------
# Build the SDL binary once (release) so the per-ROM runs share a binary.
echo "--- building rustboyadvance-sdl2 (release, features dynarec) ---"
cargo build --release -p rustboyadvance-sdl2 --features dynarec 2>&1 | tail -3

SDL_BIN="$REPO/target/release/rustboyadvance-sdl2"

# Run SDL replay on one (ROM, REC) pair. Prints the avg fps number
# parsed from the "replay done" summary line, or 0.0 on any failure.
# Stderr of the binary is captured but not shown on success to keep the
# log short — interesting signals (error backtraces) still go to the
# end of the log on failure.
run_sdl_replay() {
    local label="$1"
    local rom="$2"
    local rec="$3"

    if [[ ! -f "$BIOS" || ! -f "$rom" || ! -f "$rec" ]]; then
        echo "--- $label skipped (BIOS/ROM/REC not found) ---" >&2
        echo "0.0"
        return
    fi

    echo "--- $label: SDL replay ---" >&2
    local log
    log=$(mktemp)
    # --skip-bios so we're not burning wall time on the BIOS animation.
    # --no-audio so no SDL audio device noise.
    # --replay drives the keypad and exits when the last edge is past.
    if ! "$SDL_BIN" \
            --bios "$BIOS" \
            --skip-bios \
            --no-audio \
            --replay "$rec" \
            "$rom" > "$log" 2>&1; then
        echo "--- $label FAILED, tail of log: ---" >&2
        tail -n 30 "$log" >&2
        rm -f "$log"
        echo "0.0"
        return
    fi

    # "replay done: N frames in T.TTs wall, F.F avg fps (...)" is printed
    # to stdout (not stderr) by the SDL frontend's replay exit path.
    local fps
    fps=$(grep -oE "[0-9]+\\.[0-9]+ avg fps" "$log" | tail -1 | awk '{print $1}')
    [[ -z "$fps" ]] && fps="0.0"
    echo "$label: $fps FPS" >&2
    rm -f "$log"
    echo "$fps"
}

POKEEMERALD_FPS=$(run_sdl_replay pokeemerald "$POKEEMERALD_ROM" "$POKEEMERALD_REC")
MARIO_KART_FPS=$(run_sdl_replay mario_kart  "$MARIO_KART_ROM"  "$MARIO_KART_REC")

rm -f "$BENCH_LOG"
END_TS=$(date +%s)
SECONDS_ELAPSED=$((END_TS - START_TS))

# -----------------------------------------------------------------------------
# 4. Canonical output block (exact format the agent greps).
# -----------------------------------------------------------------------------
echo
echo "---"
printf "weighted_cycles:    %s\n" "$WEIGHTED"
printf "pokeemerald_fps:    %s\n" "$POKEEMERALD_FPS"
printf "mario_kart_fps:     %s\n" "$MARIO_KART_FPS"
printf "peak_vram_mb:       0.0\n"
printf "seconds:            %d\n" "$SECONDS_ELAPSED"
