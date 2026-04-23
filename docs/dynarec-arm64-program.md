# docs/dynarec-arm64-program.md

Autoresearch program (Karpathy format) for optimizing the shapes the
`arm7tdmi` dynarec emits on arm64 (and x86_64, same code path). One
scalar to minimize, one command that prints it, one TSV log of every
experiment, loop forever.

This is not a hand-written optimization narrative. It's the runbook an
agent follows, commit by commit. The plan that spawned it lives at
`docs/program-shapekarpathy.md`.

---

## Setup

1. Agree on a run tag based on today's date, e.g. `apr22`. Branch
   `shape-opt/<tag>` must not already exist in this clone.
2. `git checkout -b shape-opt/<tag>` from the head of the branch that
   carries the dynarec (`dynarec-feature` if upstream master hasn't
   picked it up yet).
3. Read every in-scope file end-to-end:
   - `arm7tdmi/src/dynarec/mod.rs` — the whole JIT, all `try_compile_*`
     paths, all `emit_*` functions, the `trampolines` module.
   - `arm7tdmi/src/dynarec/patterns.rs` — block-level pattern matcher
     (stub today; you'll add synthesized shapes here).
   - `arm7tdmi/src/dynarec/dump.rs` — capstone-based host-asm dumper.
     You don't modify this; you use the output when picking the next
     experiment.
   - `arm7tdmi/benches/dynarec_shapes.rs` — Criterion per-shape micro
     bench. You don't modify this; you add one fn per new synthesized
     shape when it lands.
   - Skim `scripts/dynarec_measure.sh`. Do not modify it.
4. Environment checks (each one-liner, all needed):

        # Desktop/WSL2 needs an X display for the SDL window. Headless?
        # Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99
        echo "DISPLAY=$DISPLAY"

        # PERF_BIN defaults to /usr/lib/linux-tools-6.8.0-110/perf (the
        # Ubuntu/WSL2 path on this host). On other distros:
        #   export PERF_BIN=$(which perf)
        "$PERF_BIN" --version || echo "set PERF_BIN to your perf binary"

        # Fast iteration without perf (drops the CPU/GPU breakdown):
        #   export NO_PERF=1
        # Leave unset for the real loop.

5. Verify the measurement harness runs end-to-end:

        bash scripts/dynarec_measure.sh > run.log 2>&1
        grep "^weighted_cycles:\|^pokeemerald_\|^mario_kart_" run.log

   You should see one `weighted_cycles` line plus four lines per ROM
   (`_fps`, `_cpu_pct`, `_gpu_pct`, `_bus_pct`). If the SDL replay
   can't find the BIOS/ROMs/recordings, export `BIOS=`,
   `POKEEMERALD_ROM=`, `MARIO_KART_ROM=`, `POKEEMERALD_REC=`,
   `MARIO_KART_REC=` env vars and rerun.

6. Characterize the noise floor on THIS host before trusting the gate.
   Run the same commit (baseline) five times back-to-back and note the
   min/max spread on each metric. Rule of thumb on this harness:
   weighted_cycles jitter is typically ~2%, per-ROM FPS is ~3–5%,
   per-class % is ~1–2pp. If your host is noisier (busy laptop, WSL2
   under thermal throttle) the gate thresholds below need scaling up
   proportionally — don't fight noise.

7. Initialize `results.tsv` (untracked, never committed). Ten columns:

        printf 'commit\tweighted_cycles\tpokeemerald_fps\tpokeemerald_cpu_pct\tpokeemerald_gpu_pct\tmario_kart_fps\tmario_kart_cpu_pct\tmario_kart_gpu_pct\tstatus\tdescription\n' > results.tsv

   (bus_pct, audio_pct, other_pct are still printed in run.log for
   debugging but left out of results.tsv to keep it diffable — they're
   recoverable from the per-commit run.log if needed.)

8. Run the baseline and record it:

        bash scripts/dynarec_measure.sh > run.log 2>&1
        WC=$(awk '/^weighted_cycles:/       {print $2}' run.log)
        PFP=$(awk '/^pokeemerald_fps:/      {print $2}' run.log)
        PCPU=$(awk '/^pokeemerald_cpu_pct:/ {print $2}' run.log)
        PGPU=$(awk '/^pokeemerald_gpu_pct:/ {print $2}' run.log)
        MFP=$(awk '/^mario_kart_fps:/       {print $2}' run.log)
        MCPU=$(awk '/^mario_kart_cpu_pct:/  {print $2}' run.log)
        MGPU=$(awk '/^mario_kart_gpu_pct:/  {print $2}' run.log)
        SHA=$(git rev-parse --short=7 HEAD)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tkeep\tbaseline\n' \
            "$SHA" "$WC" "$PFP" "$PCPU" "$PGPU" "$MFP" "$MCPU" "$MGPU" \
            >> results.tsv

   Confirm the row looks sensible. This baseline is what every
   subsequent experiment measures against. Two separate FPS numbers
   because each game catches different regressions — a gain that
   only helps pokeemerald but costs mario_kart is not a keeper.

9. Recording length note. The default `POKEEMERALD_REC` (`/tmp/rbarec.rec`)
   is roughly **32 minutes of emulated gameplay** because it was captured
   while the host was in turbo mode — a single SDL-replay pass of it
   takes ~10 min wall time. That makes each experiment iteration
   ~12–15 min end-to-end (cargo bench ~2 min + build ~1 min + two
   replays ~11 min). If that cadence is too slow, re-record a shorter
   (~60–90s real-time gameplay) pokeemerald rec and set
   `POKEEMERALD_REC` to it; mario_kart is already short (~21s at
   replay speed). Don't shorten by truncating the existing rec — the
   last recorded edge's cycle stamp drives the exit condition.

---

## Experimentation

What you CAN modify:

- `arm7tdmi/src/dynarec/mod.rs`
- `arm7tdmi/src/dynarec/patterns.rs`

Everything in those two files is fair game. Rewrite `emit_flag_update`,
rewrite `emit_cond_check`, add a new synthesized shape to `patterns.rs`,
change the Cranelift IR a given emit function produces, introduce a
new block-level classifier. If it's in one of those two files, you may
touch it.

What you CANNOT modify:

- `scripts/dynarec_measure.sh`
- `arm7tdmi/benches/dynarec_shapes.rs`
- `arm7tdmi/src/dynarec/dump.rs`
- `arm7tdmi/tests/dynarec_asm_baseline.rs`
- `arm7tdmi/tests/dynarec_pattern_differential.rs`
- `platform/rustboyadvance-sdl2/` (the SDL frontend the measure script wraps) and anything else under `platform/`
- `fps_bench/` (kept for perf-record runs, not part of the primary loop)
- `arm7tdmi/Cargo.toml` (no new crate deps, no new feature gates)

These are the evaluation. Modifying them is cheating on the metric.

Adding a new benchmark for a synthesized shape is the one exception: a
new `fn bench_<shape>(c: &mut Criterion)` plus the call at the bottom
of `criterion_group!` in `dynarec_shapes.rs`, AND a corresponding
call-count weight added to `scripts/dynarec_measure.sh`. Those two
edits together are allowed only when a new synthesized shape lands.
Nothing else.

Correctness gate:

        cargo test -p arm7tdmi --features dynarec
        cargo test -p arm7tdmi --features dynarec --test dynarec_pattern_differential

Both must be green after every change. If either fails, the experiment
is a crash: log it, revert.

Goal: minimize `weighted_cycles` from `scripts/dynarec_measure.sh`.
BOTH `pokeemerald_fps` AND `mario_kart_fps` must not regress more than
**1%** vs the **global best so far** (not the previous keep — using the
previous keep lets 10 sequential 0.9% regressions sneak through the
gate and silently drift the branch worse than baseline). Each game is
its own regression check — gains in one ROM that come at the cost of
the other are rejected.

The per-class percentages (`cpu_pct`, `gpu_pct`, `bus_pct`) are
**diagnostic, not gating**. Use them to interpret a keep/discard
decision, not to drive it. Rules of thumb:

- You modified CPU codegen (mod.rs emit_* / patterns.rs) → expect
  `cpu_pct` to move. `gpu_pct` and `bus_pct` should stay flat
  within the per-class noise floor (~1–2 pp). If they don't, the
  change has a second-order effect you didn't intend — look at
  what else shifted before keeping.
- You modified a sysbus fast path (not allowed in this loop, but
  same logic applies) → `bus_pct` moves, others flat.
- `weighted_cycles` drops but `cpu_pct` didn't shift → probably
  noise or a cold-path win that doesn't show in the hot samples.
  If it repeats on two runs, keep; otherwise treat as coin flip.
- `weighted_cycles` unchanged but `cpu_pct` shifts ≥3pp at flat
  FPS → you've moved work between subsystems without changing
  total cost. Usually not a keeper by itself; it's the setup for
  a follow-up that exploits the shift.

Simplicity criterion (Karpathy's): tiny gains that add ugly code lose.
Neutral-or-better diffs that delete code always win. If you find yourself
writing a 200-line PR to shave 0.2% off weighted_cycles, step back and
try a different shape.

---

## Output format

`scripts/dynarec_measure.sh` prints exactly this block at the end
(anything before it is diagnostic noise the agent can ignore):

        ---
        weighted_cycles:        12345678
        pokeemerald_fps:        312.4
        pokeemerald_cpu_pct:    61.8
        pokeemerald_gpu_pct:    11.5
        pokeemerald_bus_pct:    8.6
        mario_kart_fps:         297.1
        mario_kart_cpu_pct:     62.3
        mario_kart_gpu_pct:     10.9
        mario_kart_bus_pct:     9.1
        peak_vram_mb:           0.0
        seconds:                295.4

Target metric:

        grep "^weighted_cycles:" run.log | awk '{print $2}'

Gating sanity metrics (each game is its own check):

        grep "^pokeemerald_fps:" run.log | awk '{print $2}'
        grep "^mario_kart_fps:"  run.log | awk '{print $2}'

Diagnostic sanity metrics (interpret a result, don't gate on them):

        grep "^pokeemerald_cpu_pct:\|^pokeemerald_gpu_pct:\|^pokeemerald_bus_pct:" run.log
        grep "^mario_kart_cpu_pct:\|^mario_kart_gpu_pct:\|^mario_kart_bus_pct:" run.log

`peak_vram_mb` is N/A for this dynarec (always 0.0) but kept for
Karpathy-template parity.

---

## Logging results

`results.tsv`, tab-separated, **10 columns**, header row mandatory:

        commit    weighted_cycles    pokeemerald_fps    pokeemerald_cpu_pct    pokeemerald_gpu_pct    mario_kart_fps    mario_kart_cpu_pct    mario_kart_gpu_pct    status    description

- `commit` — 7-char git hash of the experiment commit (HEAD after your
  edit). `HEAD^` on discard/crash.
- `weighted_cycles` — integer, `0` on crash.
- `pokeemerald_fps` / `mario_kart_fps` — one decimal, `0.0` on crash.
- `pokeemerald_cpu_pct` / `mario_kart_cpu_pct` — one decimal, `0.0`
  on crash. Diagnostic only.
- `pokeemerald_gpu_pct` / `mario_kart_gpu_pct` — one decimal, `0.0`
  on crash. Diagnostic only.
- `status` — `keep` | `discard` | `crash`.
- `description` — one line, human-readable. No tabs.

(`bus_pct`, `audio_pct`, `other_pct` are printed in run.log but left
out of results.tsv to keep the table diffable. If a commit's row looks
off, the full breakdown lives in run.log next to the commit.)

Example after four rows:

        commit    weighted_cycles    pokeemerald_fps    pokeemerald_cpu_pct    pokeemerald_gpu_pct    mario_kart_fps    mario_kart_cpu_pct    mario_kart_gpu_pct    status    description
        abc1234   12345678   312.4   61.8   11.5   297.1   62.3   10.9   keep     baseline
        def5678   11800000   318.8   63.7   11.3   299.3   64.0   10.7   keep     NZCV held in host I8 vars across block
        012cafe   12400000   313.0   61.9   11.5   289.1   62.4   10.8   discard  fused cond_check — mario_kart FPS regressed >1%
        cafe001   0          0.0    0.0    0.0    0.0    0.0    0.0    crash    patterns.rs UDIV magic miscompile vs interp

`results.tsv` is untracked. Never commit it.

---

## The experiment loop

        LOOP FOREVER:
            1. Note current branch/commit. Load the global best row:
               BEST_WC=$(awk -F'\t' 'NR>1 && $9=="keep" {print $2}' results.tsv | sort -n | head -1)
               BEST_PFP=$(awk -F'\t' 'NR>1 && $9=="keep" {if($3>m)m=$3} END{print m}' results.tsv)
               BEST_MFP=$(awk -F'\t' 'NR>1 && $9=="keep" {if($6>m)m=$6} END{print m}' results.tsv)
            2. Pick an experiment from the idea menu below.
               Edit mod.rs / patterns.rs only.
            3. cargo test -p arm7tdmi --features dynarec
               - must be green. If not, fix or revert and goto 1.
            4. git commit -am "<short description>"
            5. bash scripts/dynarec_measure.sh > run.log 2>&1
            6. grep "^weighted_cycles:\|^pokeemerald_\|^mario_kart_" run.log
               - empty output on weighted_cycles => crash. tail -n 50
                 run.log. Decide fix-in-place vs discard.
               - EITHER pokeemerald_fps OR mario_kart_fps regressed
                 > 1% vs its BEST_*FP => discard. Each game is its own
                 gate.
               - weighted_cycles > BEST_WC => discard.
               - weighted_cycles < BEST_WC AND BOTH fps numbers within
                 1% of BEST_*FP => keep.
               - weighted_cycles between BEST_WC and BEST_WC*1.005 (i.e.
                 within noise floor) AND cpu_pct / gpu_pct unchanged
                 within 1pp each ROM => "coin flip": re-run ONCE. If
                 the second run is also inside the noise floor, discard
                 (the change is not doing anything signal-bearing).
            7. Record the row in results.tsv with all 10 columns.
            8. keep     => branch advances, continue to 1.
               discard  => git reset --hard HEAD~1, continue to 1.
               crash    => git reset --hard HEAD~1, continue to 1.

Typical wall time per iteration on this host:
  - cargo bench (dynarec_shapes):  ~2 min
  - cargo build (SDL + debuginfo): ~1 min (cached after first build)
  - SDL replay + perf (pokeemerald): ~10 min (long rec, see Setup §9)
  - SDL replay + perf (mario_kart):  ~0.5 min
  Total: ~13–15 min per experiment. >25 min => kill and treat as crash.

If you want a faster inner loop, set `NO_PERF=1` to skip the perf
record passes — drops the `_cpu_pct` / `_gpu_pct` / `_bus_pct` fields
from the output (they'll show 0.0), but FPS + weighted_cycles still
work and each replay pass becomes ~1 min faster. Use this mode when
you're iterating on a single change and don't need the subsystem
breakdown; unset it for the keep/discard gate decision.

---

## Findings from the apr22 run (what's already in HEAD)

This branch landed 10 kept experiments from 16 attempts. Numbers below
are apples-to-apples on the 6-shape bench at commit `ca11be7`; a
`shift_pair_sxtb` shape joins the metric at `772d381` so WC bumps
~+3M from that point on without being a regression.

**Top wins measured:**
- `thumb_dp_chain` 4.19 → 1.78 ns  (−58%) via exp 10: Thumb-block
  dead-flag-write pre-pass (items whose NZCV write is covered by the
  next unconditional flag-setter get their flag update skipped
  entirely).
- `thumb_mov_imm`  ~1.70 → 1.05 ns (−38%) via exp 5: Thumb3 MOV #imm8
  inline fast path — skips rd load + N compute; since imm8 ∈ [0,255]
  and N is always 0, the flag update collapses to a single
  `cpsr_masked | (imm8==0 ? 0x4000_0000 : 0)`.
- `arm_mov_imm`    1.20 → 1.04 ns (−13%) via exp 8: skip cpsr load +
  store when no instr in block reads or writes flags.

**Don't-bother list (I tried these, they regress or are noise on x86_64):**
- `select(cond, mask_imm, 0)` in place of `icmp + uextend + ishl_imm`
  — lowers WORSE on x86_64 than the three-op sequence (exp 4, +8.8%
  regression).
- `opt_level = speed` on Cranelift — runtime-level regression on short
  JITted blocks (exp 6, +16% regression). Alias-analysis needed it
  enabled but the compile-time cost wasn't offset by execution gain
  on our shapes.
- Lazy-extract each flag in `emit_cond_check` (exp 2, within noise).
  Cranelift already DCEs unused flag extracts when only some are
  needed.
- Explicit skip of the `rd_val` load in `emit_thumb_format3` MOV (exp
  3, within noise). Cranelift already DCEs the unused value.

**Structural scaffolding already in place (don't re-invent):**
- `patterns::ShiftPairSignExtendByte/Half` and
  `ShiftPairZeroExtendByte/Half` — first real block-level pattern
  stencil. Matches two-instruction LSL+ASR / LSL+LSR Thumb blocks,
  emits `ireduce + sextend/uextend` which Cranelift lowers to a single
  `sxtb/sxth/uxtb/uxth` on arm64 (`movsx` on x86_64). Proven
  bit-exact against interpreter via 4 differential tests.
- Dead-flag-write pre-pass in `try_compile_thumb_block` AND
  `try_compile_imm_block`. Both analyse flag-write coverage and skip
  the cpsr pack when observably dead.
- `arm7tdmi::dynarec::dump` (capstone-based host-asm dumper, gated by
  `dynarec_asm_dump` feature). Use it to inspect what Cranelift emits
  for any existing or new shape; per-shape tests in
  `tests/dynarec_asm_baseline.rs`.
- `scripts/dynarec_measure.sh` + `benches/dynarec_shapes.rs` — all 7
  shape benches wired; `weighted_cycles` is a single scalar the loop
  minimises; per-ROM SDL-replay FPS + CPU/GPU/bus classification
  (via `perf record`) is the correctness-drift + subsystem-shift
  guard, each game checked separately.

## Signal-to-noise caveat

Noise floor on this host (re-measure yours per Setup §6 before
trusting the gate):

- weighted_cycles: ≈2% run-to-run (pure Criterion micro-bench;
  relatively quiet because no other subsystems are running).
- SDL-replay FPS: ≈3–5% per ROM. Full video pipeline is exercised,
  `perf record` adds ~1%, and the OS scheduler adds the rest.
- cpu_pct / gpu_pct / bus_pct: ≈1–2 pp per ROM. Anything smaller
  than 1 pp is almost certainly sampling noise; anything >2 pp is
  probably a real subsystem shift.

Experiments whose true gain is smaller than ~3% are
indistinguishable from WC noise and show up alternately as keep or
discard. The coin-flip rule in the experiment loop handles these:
re-run once, if both runs stay inside the noise floor AND the
cpu/gpu/bus breakdown didn't shift > noise either, discard.

## When to retrain weights (W_* constants in dynarec_measure.sh)

The shape call-counts baked into `scripts/dynarec_measure.sh`
(`W_thumb_mov_imm=8M`, etc.) are from the apr22 baseline
flamegraph. After you land several experiments that change how
often each shape compiles, those numbers start lying — a shape
the agent now rarely hits still has 8M weight, distorting WC.

Retrain when **any** of these fires:

- 10 kept commits have landed since the last retrain.
- A new synthesized shape gets added to `patterns.rs` (its weight
  needs to be measured, not guessed).
- `cpu_pct` shifts > 5 pp on either ROM across consecutive keeps
  without an obvious cause — the hot distribution moved.

How to retrain: rebuild with `--features shape_profile` (once
wired, see backlog §1), run one pass per ROM, dump the counter
table, scale to calls/sec based on the replay length, update
`W_*` in the script. Label the commit `measure: retrain shape
weights against <ROM> <date>` and record it in `results.tsv`
with `status=keep` and `description="retrain weights (not a
code experiment)"`. It shifts WC because the weights moved,
not because a shape got faster — that's expected and fine.

## What to do next (honest backlog, ordered by payoff / complexity)

1. **Wire the shape_profile counter.** The `shape_profile` feature
   flag is in `arm7tdmi/Cargo.toml` but not used anywhere. Plumb it
   through `BlockCache::finish_record` to count actual execution of
   each compile-path / format. Dump at SDL-replay exit (one pass per
   ROM). Feed the
   empirical distribution back into `scripts/dynarec_measure.sh`'s
   `W_*` constants. Replaces my hand-picked {8M, 6M, 5M, 3M, 2M,
   1.5M, 2.5M} guesses with real-gameplay weights. ~1 hour. Every
   subsequent experiment's WC signal becomes meaningful.

2. **Pattern: MUL-by-constant.** Matches `MOV Rtmp, #imm ;
   MUL Rd, Rs` → `Rd = imm * Rs`. Requires first adding
   `Thumb4Op::Mul` as a decoded shape (currently Thumb MUL falls
   through so the 2-instr block doesn't compile). ~2 hours.

3. **Pattern: branchless abs** `(x ^ (x>>31)) - (x>>31)`.
   Three-instruction Thumb sequence; lowers to arm64 `cmp + cneg`.
   ~1 hour.

4. **Part A1 lazy NZCV.** Hold N/Z/C/V in four Cranelift Variables
   across a block; unpack cpsr once at entry, pack once at exit.
   Large refactor touching every `emit_*`. Only pays off when blocks
   have ≥2 flag-setters AND have a conditional mid-block. Guard
   with shape_profile data before committing to this; otherwise
   it's a coin-flip against the host-level noise.

5. **CLZ polyfill / popcount SWAR patterns.** Real but rarer shapes;
   each is a 10+-instruction recognizer + a 1-instruction stencil.
   ~3 hours each. Skip unless shape_profile shows them hot.

Blockers/discovered caveats:
- Thumb MUL (format 4, op=1101) is not currently a dynarec-supported
  shape; blocks containing it fall through to the interpreter. Any
  pattern involving MUL needs decoder work first.
- SWI (BIOS call intercept, e.g. CpuSet) is deliberately scalar-only
  per the base dynarec design. BIOS CpuSet-fold patterns require
  lifting that restriction first.

NEVER STOP: loop indefinitely. If ideas run out, re-read `mod.rs` cold,
re-read the idea menu, combine near-misses, try a more radical Cranelift
IR rewrite, or try a hand-written stencil via `patterns.rs`.

---

## Experiment idea menu

### Part A — common-sink helpers (wins multiply)

Every compiled block goes through `emit_flag_update` (writes NZCV back
to CPSR) and `emit_cond_check` (reads NZCV out of CPSR). Any saving
there scales by (# blocks with a flag-setting op) × (# conditional
ops). That's most of pokeemerald.

- **A1. Hold NZCV in four host I8 Cranelift Variables across the
  whole block.** Materialize back to the packed CPSR word only at
  block exit. `emit_cond_check` reads the Variables directly, skipping
  the per-instr `ushr / band / ishl / bor` dance.
- **A2. Fuse `emit_flag_update` and `emit_cond_check` on the common
  case where instruction K+1 reads the flags K just set** (CMP
  followed by Bcond). Keep NZCV in host regs, skip the CPSR round
  trip.
- **A3. Replace the repeated `ushr_imm / band 1 / ishl_imm / bor`
  packing sequence with a single `iconcat` or `bitselect` where
  Cranelift lowers it to fewer arm64 instructions.** Inspect via
  `dynarec_asm_dump`.

### Part B — per-shape emitters (in mod.rs file order)

These are the per-instruction emit functions. For each: look at the
current arm64 via `cargo test --features dynarec_asm_dump --test
dynarec_asm_baseline -- --nocapture`, and identify patterns like:

- Repeated `load gpr[Rd]` / `store gpr[Rd]` when the same guest register
  is used twice in a compiled block. Hoist the load into a Variable
  that persists across the block.
- Redundant zero/sign-extension steps when the input is provably
  already zero/sign-extended.
- The "preserve C" dance on shifts: replace the `ushr_imm + band 1 +
  ishl_imm + bor` sequence with a persisted Variable.

Functions, by source order in `mod.rs`:

- `emit_conditional_instr`
- `emit_thumb_format14` (PUSH/POP)
- `emit_thumb_format11` (SP-relative LDR/STR)
- `emit_thumb_format9` (imm-offset LDR/STR)
- `emit_thumb_format5_non_branch`
- `emit_thumb_format1` (shifted move)
- `emit_thumb_format4_logical`
- `emit_thumb_format2` (add/sub reg)
- `emit_thumb_format3` (imm8 arithmetic)
- `emit_conditional_mem`
- `emit_mem_body`
- `emit_data_processing_imm`

### Part C — synthesized shapes (magic-number lowering in patterns.rs)

GBA game code and the GBA BIOS contain many hand-rolled bit tricks.
Each is dozens of ARM/Thumb instructions but lowers to one or two
native instructions on arm64. The block-level matcher in `patterns.rs`
recognizes the trigger sequence and emits a tight stencil.

Each entry tagged with **[pokeemerald]** / **[mario_kart]** /
**[both]** = which replay ROM surfaces this shape in the perf
flamegraph, so the gate doesn't silently veto a real win because
the change didn't move the ROM it wasn't targeting. Unmarked =
compiler-generated idiom, both games hit it.

1. **MUL-by-constant** → shift-add tree or `madd`. Trigger: `MOV Rtmp,
   #imm ; MUL Rd, Rs, Rtmp` where `imm` is small/power-of-2. **[both]**
   Thumb MUL not yet a decoded shape; first add `Thumb4Op::Mul`.
2. **UDIV/SDIV-by-constant via Granlund–Montgomery.** Magic
   `M = ceil(2^(32+k)/d)` → arm64 `umulh ; lsr`. Trigger: the standard
   magic-multiply sequence, or BIOS `SWI 0x06` preceded by `MOV r1, #imm`.
   **[pokeemerald]** — emerald computes a lot of 1/60 and 1/240 via
   this idiom; Mario Kart is largely 2D/3D fixed-point and rarely
   divides by constants.
3. **Branchless abs**: `(x ^ (x>>31)) - (x>>31)` → `cmp ; cneg`.
   **[both]**
4. **CLZ polyfill** (de Bruijn 0x077CB531 table, Stanford binary search)
   → arm64 `clz`. **[pokeemerald]** — RNG code.
5. **popcount SWAR** (0x55555555 / 0x33333333 / 0x0f0f0f0f / 0x01010101)
   → arm64 NEON `cnt ; addv`. **[pokeemerald]** — Pokemon-count /
   flag-check paths.
6. **Branchless MIN/MAX**: `b + ((a-b) & -(a<b))` → `cmp ; csel`.
   **[both]**
7. **Shift-pair sign/zero extend** `LSL #24 ; ASR #24` (and 16-bit) →
   `sxtb` / `sxth` / `uxtb` / `uxth`. **[both]** — already landed.
8. **Byte-swap idiom** → `rev`. **[pokeemerald]** — save serialization.
9. **GBA BIOS CpuSet / CpuFastSet fold** (`SWI 0x0B` / `0x0C` + known
   `r2` mode bits) → inlined memcpy/memset stencil. **[mario_kart]** —
   3D track geometry upload uses CpuFastSet heavily per frame.
10. **Dead DP chain DCE**: drop register-write chains whose results
    are all overwritten before block exit without being observed.
    **[both]**

When an experiment targets a ROM-specific pattern, expect:

- The non-targeted ROM's FPS and per-class % to stay within noise
  (if they don't, you've hit an unrelated hot path — investigate).
- The targeted ROM's cpu_pct to drop; its FPS to rise; weighted_cycles
  to drop only if the pattern fires often enough. If WC stays flat,
  the shape isn't hot in the bench-weighted sense yet — wait for
  shape_profile to confirm before ripping the pattern out.

Each new synthesized shape lands with:

- an entry in `arm7tdmi::dynarec::patterns::Pattern` (enum variant)
  plus the matcher arm in `try_match_*`;
- a new `#[test]` in `arm7tdmi/tests/dynarec_pattern_differential.rs`
  covering the trigger sequence + boundary cases;
- a new `bench_<shape>` in `arm7tdmi/benches/dynarec_shapes.rs` so
  `scripts/dynarec_measure.sh` can pick it up;
- a new weight entry in `scripts/dynarec_measure.sh` (`W_<shape>=...`)
  reflecting roughly how many times per real-gameplay second the
  trigger fires. Use 0 when unsure until frequency data comes in.
