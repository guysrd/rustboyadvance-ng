#!/usr/bin/env bash
# AOT-LLVM measurement harness (per docs/aot-llvm-program.md V1+V2+V3+V4).
#
# Builds with `--features aot`, runs SDL replay AOT + scalar
# back-to-back × 3 each, takes median fps, diffs fb-hashes,
# computes drift, runs differential tests, prints the canonical
# output block.
#
# Hard-fail conditions per the program doc:
#   - pe_divs > 0 or mk_divs > 0       (V1)
#   - |pe_drift| > 1000 or |mk_drift| > 1000 cycles (V2)
#   - diff_failures > 0                (V3)
#   - replay non-determinism           (V4)
#
# Usage:
#   bash scripts/aot_measure.sh [AOT_SWEEP_CAP_KB]

set -e
cd /home/user/pokeemerlad/rustboyadvance-ng

SWEEP_KB=${1:-0}
PE_ROM=/home/user/pokeemerlad/pokeemerald/pokeemerald.gba
MK_ROM=/home/user/pokeemerlad/recordings/roms/mks.gba
PE_REC=/home/user/pokeemerlad/recordings/pokeemerald_run.rec
MK_REC=/home/user/pokeemerlad/recordings/mks.rec
BIOS=core/benches/roms/normatt_gba_bios.bin
PE_REF=/home/user/pokeemerlad/recordings/pe_scalar_ref.txt
MK_REF=/home/user/pokeemerlad/recordings/mk_scalar_ref.txt

echo "AOT_SWEEP_CAP_KB=$SWEEP_KB" >&2

# Build.
LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 \
    cargo build --release --workspace --features aot 2>&1 | tail -3

BIN=target/release/rustboyadvance-sdl2

run_one() {
    # $1=label  $2=rom  $3=rec  $4=outfile
    AOT_SWEEP_CAP_KB=$SWEEP_KB timeout 90 "$BIN" --aot --no-audio \
        --replay "$3" --frame-hash-every 60 --bios "$BIOS" "$2" \
        2>/dev/null > "$4"
}

run_scalar() {
    # baseline: no --aot. Used for fps reference.
    timeout 90 "$BIN" --no-audio \
        --replay "$3" --frame-hash-every 60 --bios "$BIOS" "$2" \
        2>/dev/null > "$4"
}

extract_fps()  { grep "avg fps" "$1" | awk '{ for (i=1;i<=NF;i++) if ($i=="avg") print $(i-1) }'; }
extract_cyc()  { grep "fb_hash" "$1" | tail -1 | awk -F'cycle=|hash=' '{print $2}' | awk '{print $1}'; }
median3()      { printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n '2p'; }
hash_lines()   { grep "^fb_hash:" "$1" | awk '{print $4}'; }

count_divs() {
    # $1=aot_out $2=ref_file
    diff <(hash_lines "$1") <(hash_lines "$2") | grep -c '^>' || true
}

measure_rom() {
    # $1=label  $2=rom  $3=rec  $4=ref_hashes
    local label="$1" rom="$2" rec="$3" ref="$4"
    local f1=/tmp/aot_${label}_run1.txt
    local f2=/tmp/aot_${label}_run2.txt
    local f3=/tmp/aot_${label}_run3.txt
    local fs=/tmp/scalar_${label}.txt
    run_one "$label" "$rom" "$rec" "$f1"
    run_one "$label" "$rom" "$rec" "$f2"
    run_one "$label" "$rom" "$rec" "$f3"
    run_scalar "$label" "$rom" "$rec" "$fs"
    local f1_fps=$(extract_fps "$f1")
    local f2_fps=$(extract_fps "$f2")
    local f3_fps=$(extract_fps "$f3")
    local s_fps=$(extract_fps "$fs")
    local aot_fps=$(median3 "$f1_fps" "$f2_fps" "$f3_fps")
    local divs=$(count_divs "$f1" "$ref")
    local drift=$(( $(extract_cyc "$f1") - $(extract_cyc "$ref") ))
    local h1=$(md5sum < <(hash_lines "$f1") | awk '{print $1}')
    local h2=$(md5sum < <(hash_lines "$f2") | awk '{print $1}')
    local determinism_ok=0
    if [ "$h1" = "$h2" ]; then determinism_ok=1; fi
    local cov=$(grep "aot dispatch:" "$f1" | awk '{print $7}' | sed 's/%//')
    [ -z "$cov" ] && cov=0
    echo "$aot_fps $s_fps $divs $drift $determinism_ok $cov"
}

echo "running PE..." >&2
PE=$(measure_rom pe "$PE_ROM" "$PE_REC" "$PE_REF")
echo "running MK..." >&2
MK=$(measure_rom mk "$MK_ROM" "$MK_REC" "$MK_REF")

PE_FPS_AOT=$(echo "$PE" | awk '{print $1}')
PE_FPS_SCA=$(echo "$PE" | awk '{print $2}')
PE_DIVS=$(echo "$PE" | awk '{print $3}')
PE_DRIFT=$(echo "$PE" | awk '{print $4}')
PE_DET=$(echo "$PE" | awk '{print $5}')
PE_COV=$(echo "$PE" | awk '{print $6}')
MK_FPS_AOT=$(echo "$MK" | awk '{print $1}')
MK_FPS_SCA=$(echo "$MK" | awk '{print $2}')
MK_DIVS=$(echo "$MK" | awk '{print $3}')
MK_DRIFT=$(echo "$MK" | awk '{print $4}')
MK_DET=$(echo "$MK" | awk '{print $5}')
MK_COV=$(echo "$MK" | awk '{print $6}')

# Differential tests (V3).
echo "running differential tests..." >&2
DIFF_FAIL=$(LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 \
    cargo test --workspace --features aot aot_diff_ 2>&1 | grep -c '^test .* FAILED' || true)

# AOT score.
SCORE=$(echo "$PE_FPS_AOT $PE_FPS_SCA $MK_FPS_AOT $MK_FPS_SCA" | \
    awk '{ printf "%.0f", ($1 - $2) + ($3 - $4) }')

cat <<OUT
---
aot_score:           $SCORE
pe_fps_aot:          $PE_FPS_AOT
pe_fps_scalar:       $PE_FPS_SCA
pe_drift:            $PE_DRIFT
pe_divs:             $PE_DIVS
pe_determinism_ok:   $PE_DET
pe_coverage_pct:     $PE_COV
mk_fps_aot:          $MK_FPS_AOT
mk_fps_scalar:       $MK_FPS_SCA
mk_drift:            $MK_DRIFT
mk_divs:             $MK_DIVS
mk_determinism_ok:   $MK_DET
mk_coverage_pct:     $MK_COV
diff_failures:       $DIFF_FAIL
---
OUT
