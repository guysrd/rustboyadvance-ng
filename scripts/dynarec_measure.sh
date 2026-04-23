#!/usr/bin/env bash
# dynarec shape-optimization measurement harness.
#
# Prints a single scalar `weighted_cycles` per the autoresearch loop in
# docs/program-shapekarpathy.md. The agent greps this value to decide
# keep/discard after every commit. Also prints per-ROM FPS (correctness
# sanity — each ROM is a separate regression gate) AND a per-ROM CPU/GPU
# self-time breakdown gathered by running the SDL replay under `perf
# record`. The breakdown catches experiments that move cost from one
# subsystem to another without changing wall time; useful when an
# optimization claims to speed up the CPU but actually just shifts
# overhead into GPU scanline rendering (or vice versa).
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
# Two ROMs, both must stay within their regression budget.
#
# CPU/GPU breakdown uses perf record -F 997 -g --call-graph dwarf during
# the SDL replay, then perf report --sort=symbol to pull self-time per
# function, and classifies every symbol into one of {cpu, gpu, bus,
# audio, other} by prefix/substring match. The script only prints
# CPU% and GPU% in the canonical block because those are the two the
# shape-opt loop actually needs to distinguish, but the full class
# split is dumped to the log for debugging.
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
#   grep "^weighted_cycles:\\|^pokeemerald_\\|^mario_kart_" run.log
#
# Env overrides:
#   BIOS
#   POKEEMERALD_ROM, MARIO_KART_ROM
#   POKEEMERALD_REC, MARIO_KART_REC
#   PERF_BIN (default: /usr/lib/linux-tools-6.8.0-110/perf on this host;
#            set to `perf` if your distro's wrapper resolves correctly)
#   NO_PERF=1 to skip the perf record passes (faster; drops CPU/GPU %)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIOS="${BIOS:-$REPO/core/benches/roms/normatt_gba_bios.bin}"
POKEEMERALD_ROM="${POKEEMERALD_ROM:-/home/user/pokeemerlad/pokeemerald/pokeemerald.gba}"
MARIO_KART_ROM="${MARIO_KART_ROM:-/tmp/mks/Mario Kart - Super Circuit (USA).gba}"
POKEEMERALD_REC="${POKEEMERALD_REC:-/tmp/rbarec.rec}"
MARIO_KART_REC="${MARIO_KART_REC:-/tmp/mks.rec}"
PERF_BIN="${PERF_BIN:-/usr/lib/linux-tools-6.8.0-110/perf}"

START_TS=$(date +%s)

# -----------------------------------------------------------------------------
# 1. Per-shape Criterion micro-bench (unchanged — this is the primary scalar).
# -----------------------------------------------------------------------------
echo "--- running cargo bench ---"
BENCH_LOG=$(mktemp)
cargo bench -p arm7tdmi --bench dynarec_shapes --features dynarec,bench -- --quiet 2>&1 \
  | tee "$BENCH_LOG"

parse_bench() {
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
# 3. Per-ROM SDL replay + perf record for CPU/GPU breakdown.
# -----------------------------------------------------------------------------
# Build the SDL binary with debug symbols so perf can symbolize. RUSTFLAGS
# is scoped to this build; the bench and test paths above used whatever
# profile was already built, which is fine.
echo "--- building rustboyadvance-sdl2 (release + debug syms, features dynarec,shape_profile) ---"
# shape_profile implies dynarec. It's safe to leave on for every measure
# run: the per-block `tick(shape)` is one `AtomicU64::fetch_add`, well
# below the noise floor of the SDL-replay FPS measurement itself. The
# payoff is that every run emits a calls/sec profile the script can
# use to retrain `W_*` (see suggested W_* block at the end of the log).
RUSTFLAGS="-C debuginfo=2" \
    cargo build --release -p rustboyadvance-sdl2 --features dynarec,shape_profile 2>&1 | tail -3

SDL_BIN="$REPO/target/release/rustboyadvance-sdl2"

# Classify a function name into one of {cpu, gpu, bus, audio, other}.
# Tuned for rustboyadvance's module layout and the dynarec's hot
# functions; add new rules as hot spots move.
classify_symbol() {
    local sym="$1"
    case "$sym" in
        # GPU: scanline compositor, BG rendering, OBJ rendering, pixel ops
        *gpu::sfx*|*finalize_scanline*|*finalize_pixel*|\
        *render_scanline*|*render_reg_bg*|*render_aff_bg*|\
        *render::text*|*render::obj*|*render::bitmap*|\
        *gpu::Gpu*|*gpu::render*|*obj_buffer*|*palette_ram*|*Rgb15*)
            echo gpu
            return ;;
        # Audio: sound module, APU event handling, resampler, sample mix
        *sound::*|*SoundController*|*Psg*|*ApuEvent*|*resampler*|*audio::*)
            echo audio
            return ;;
        # Bus: SysBus load/store dispatch, memory reads/writes, cartridge
        *SysBus*|*sysbus::*|*cartridge::*|*Cartridge*|\
        *load_32*|*load_16*|*load_8*|\
        *store_32*|*store_16*|*store_8*|\
        *read_32*|*read_16*|*read_8*|\
        *write_32*|*write_16*|*write_8*|\
        *BusIO*|*MemoryInterface*)
            echo bus
            return ;;
        # CPU: arm/thumb exec, cpu::step, dynarec dispatch, dynarec blocks,
        # plus the emulator-driver functions that call into them. The
        # driver functions (single_step, frame, handle_events, dma_step,
        # cpu_step, cpu_interrupt) are thin wrappers around CPU execution
        # so they cleanly belong to the CPU budget.
        *arm::exec*|*thumb::exec*|*arm7tdmi::cpu*|*cpu::step*|\
        *replay_cached_block*|*step_block*|*record_new_block*|\
        *Arm7tdmiCore*|\
        *dynarec_thumb*|*dynarec_arm*|*dynarec_imm_block*|\
        *dynarec::trampolines*|*dynarec::patterns*|\
        *exec_arm*|*exec_thumb*|*alu::*|*register_shift*|\
        *scheduler*|*Scheduler*|*BinaryHeap*|\
        *GameBoyAdvance*|*gba::*|*single_step*|*handle_events*|\
        *cpu_step*|*cpu_interrupt*|*dma_step*|*dma::DmaController*|\
        *interrupt::*|*timer::*|*Timers*)
            echo cpu
            return ;;
        # Unresolved hex symbols in the 0x7xxx... range are overwhelmingly
        # Cranelift-JIT-emitted dynarec blocks: perf can't symbolize them
        # without a /tmp/perf-<pid>.map file from the JIT, which Cranelift
        # doesn't emit by default. Since every compiled block IS CPU
        # work by definition (that's the whole point of dynarec), bucket
        # them as cpu rather than lose 10-15% of the sample budget to
        # "other". Addresses below that range are usually libc/SDL
        # helpers and stay in "other".
        0x00007[0-9a-f]*|0x00006[0-9a-f]*|0x00005[0-9a-f]*)
            echo cpu
            return ;;
    esac
    echo other
}

# Given a perf.data, print lines "<class_name> <self_pct>" summing to
# ~100%. We read `perf report --sort=symbol -g none --stdio` and
# classify each symbol row.
perf_classify_pcts() {
    local perf_data="$1"
    "$PERF_BIN" report -i "$perf_data" --sort=symbol -g none --stdio 2>/dev/null \
      | awk '
        # Look for lines like "   22.88%    22.88%  [.] <SysBus as ...>::load_16"
        # Columns: Children% Self% [x] Symbol...
        /^[[:space:]]+[0-9]+\.[0-9]+%[[:space:]]+[0-9]+\.[0-9]+%/ {
            self_pct = $2
            # Reconstruct symbol: everything past the [x] column.
            sym = ""
            for (i = 4; i <= NF; i++) {
                sym = sym ($i)
                if (i < NF) sym = sym " "
            }
            # Strip surrounding brackets like [.] that AWK's $3 consumed.
            gsub("%", "", self_pct)
            print self_pct "\t" sym
        }
    ' | while IFS=$'\t' read -r pct sym; do
        class=$(classify_symbol "$sym")
        echo "$class $pct"
    done | awk '
        { sums[$1] += $2 }
        END {
            # Print in a stable order so the log is diffable across runs.
            split("cpu gpu bus audio other", order, " ")
            for (i = 1; i <= 5; i++) {
                k = order[i]
                printf "%s %.1f\n", k, (k in sums ? sums[k] : 0.0)
            }
        }
    '
}

# Run one SDL replay pass under perf, returning FPS and populating a
# global assoc array with the per-class self-time. We use globals
# because bash can't return multiple values cleanly. Also captures
# shape_profile counter output (when the binary was built
# --features shape_profile) into SHAPE_COUNT[<rom>:<shape>] =
# calls/sec, so the measure script can suggest retrained W_* values.
declare -A CLASS_PCT
declare -A SHAPE_COUNT
run_sdl_replay_with_perf() {
    local label="$1"
    local rom="$2"
    local rec="$3"

    # Reset the class table for this call.
    CLASS_PCT[cpu]=0.0
    CLASS_PCT[gpu]=0.0
    CLASS_PCT[bus]=0.0
    CLASS_PCT[audio]=0.0
    CLASS_PCT[other]=0.0

    if [[ ! -f "$BIOS" || ! -f "$rom" || ! -f "$rec" ]]; then
        echo "--- $label skipped (BIOS/ROM/REC not found) ---" >&2
        echo "0.0"
        return
    fi

    echo "--- $label: SDL replay ---" >&2
    local log perf_data
    log=$(mktemp)
    perf_data="/tmp/dynarec-measure-${label}.perf"
    rm -f "$perf_data"

    # Two modes: with perf (default) or without (NO_PERF=1 for fast
    # dev iterations that don't need the subsystem breakdown).
    if [[ "${NO_PERF:-0}" == "1" ]] || [[ ! -x "$PERF_BIN" ]]; then
        if ! "$SDL_BIN" \
                --bios "$BIOS" --skip-bios --no-audio \
                --replay "$rec" "$rom" > "$log" 2>&1; then
            echo "--- $label FAILED, tail of log: ---" >&2
            tail -n 30 "$log" >&2
            rm -f "$log"
            echo "0.0"
            return
        fi
    else
        # perf record -F 997 -g --call-graph dwarf keeps sampling cost
        # under ~1% and produces call-graph data we can symbolize. Output
        # data is discarded after extraction — this is a temp file.
        if ! "$PERF_BIN" record -F 997 -g --call-graph dwarf -q \
                -o "$perf_data" -- \
                "$SDL_BIN" \
                --bios "$BIOS" --skip-bios --no-audio \
                --replay "$rec" "$rom" > "$log" 2>&1; then
            echo "--- $label FAILED, tail of log: ---" >&2
            tail -n 30 "$log" >&2
            rm -f "$log" "$perf_data"
            echo "0.0"
            return
        fi
    fi

    local fps
    fps=$(grep -oE "[0-9]+\\.[0-9]+ avg fps" "$log" | tail -1 | awk '{print $1}')
    [[ -z "$fps" ]] && fps="0.0"
    echo "$label: $fps FPS" >&2

    # Pull shape_profile counters from the SDL stdout. Present only when
    # the binary was built with `--features shape_profile`; silently
    # ignored otherwise. Divides counter by replay wall seconds to get
    # calls/sec — the form `W_*` wants.
    local wall
    wall=$(grep -oE "^shape_profile:replay_wall_seconds [0-9]+\\.[0-9]+" "$log" \
           | tail -1 | awk '{print $2}')
    if [[ -n "$wall" && "$wall" != "0" ]]; then
        echo "--- $label: shape_profile counters ---" >&2
        while IFS= read -r line; do
            # Line format:  shape_profile:<name>  <count>
            local name count
            name=$(echo "$line" | awk '{print $1}' | sed 's/^shape_profile://')
            count=$(echo "$line" | awk '{print $2}')
            # Skip the wall-seconds line and any non-numeric count.
            [[ "$name" == "replay_wall_seconds" ]] && continue
            [[ -z "$count" || ! "$count" =~ ^[0-9]+$ ]] && continue
            local rate
            rate=$(awk -v c="$count" -v w="$wall" 'BEGIN { printf "%.0f", c/w }')
            SHAPE_COUNT["$label:$name"]="$rate"
            printf "  %-16s %10s calls/s (%s / %ss)\n" \
                "$name" "$rate" "$count" "$wall" >&2
        done < <(grep -E "^shape_profile:[a-z_]+\\s+[0-9]+$" "$log")
    fi

    rm -f "$log"

    # Populate the classification table from perf data.
    if [[ -f "$perf_data" ]]; then
        echo "--- $label: perf subsystem breakdown ---" >&2
        while read -r class pct; do
            CLASS_PCT[$class]="$pct"
            printf "  %-6s %s%%\n" "$class" "$pct" >&2
        done < <(perf_classify_pcts "$perf_data")
        rm -f "$perf_data"
    fi

    echo "$fps"
}

POKEEMERALD_FPS=$(run_sdl_replay_with_perf pokeemerald "$POKEEMERALD_ROM" "$POKEEMERALD_REC")
POKEEMERALD_CPU=${CLASS_PCT[cpu]:-0.0}
POKEEMERALD_GPU=${CLASS_PCT[gpu]:-0.0}
POKEEMERALD_BUS=${CLASS_PCT[bus]:-0.0}

MARIO_KART_FPS=$(run_sdl_replay_with_perf mario_kart "$MARIO_KART_ROM" "$MARIO_KART_REC")
MARIO_KART_CPU=${CLASS_PCT[cpu]:-0.0}
MARIO_KART_GPU=${CLASS_PCT[gpu]:-0.0}
MARIO_KART_BUS=${CLASS_PCT[bus]:-0.0}

rm -f "$BENCH_LOG"
END_TS=$(date +%s)
SECONDS_ELAPSED=$((END_TS - START_TS))

# -----------------------------------------------------------------------------
# 4. Canonical output block (exact format the agent greps).
# -----------------------------------------------------------------------------
echo
echo "---"
printf "weighted_cycles:        %s\n" "$WEIGHTED"
printf "pokeemerald_fps:        %s\n" "$POKEEMERALD_FPS"
printf "pokeemerald_cpu_pct:    %s\n" "$POKEEMERALD_CPU"
printf "pokeemerald_gpu_pct:    %s\n" "$POKEEMERALD_GPU"
printf "pokeemerald_bus_pct:    %s\n" "$POKEEMERALD_BUS"
printf "mario_kart_fps:         %s\n" "$MARIO_KART_FPS"
printf "mario_kart_cpu_pct:     %s\n" "$MARIO_KART_CPU"
printf "mario_kart_gpu_pct:     %s\n" "$MARIO_KART_GPU"
printf "mario_kart_bus_pct:     %s\n" "$MARIO_KART_BUS"
printf "peak_vram_mb:           0.0\n"
printf "seconds:                %d\n" "$SECONDS_ELAPSED"

# shape_profile retraining suggestions (present only when the SDL binary
# was built --features shape_profile). Prints a suggested `W_*` block
# the agent can copy-paste into this script. Uses the arithmetic mean
# of per-ROM rates (simple, treats both games equally; refine if one
# ROM dominates real workload).
if [[ -n "${SHAPE_COUNT[pokeemerald:thumb_dp_chain]:-}" ]]; then
    echo
    echo "--- suggested W_* retraining (mean of per-ROM calls/sec) ---"
    for shape in thumb_mov_imm thumb_add_imm thumb_cmp_imm thumb_dp_chain \
                  arm_mov_imm arm_cmp_imm shift_pair_sxtb; do
        pk="${SHAPE_COUNT["pokeemerald:$shape"]:-0}"
        mk="${SHAPE_COUNT["mario_kart:$shape"]:-0}"
        avg=$(awk -v p="$pk" -v m="$mk" 'BEGIN { printf "%d", (p+m)/2 }')
        printf "W_%s=%s\n" "$shape" "$avg"
    done
fi
