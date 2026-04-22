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
4. Verify the measurement harness runs end-to-end:

        bash scripts/dynarec_measure.sh > run.log 2>&1
        grep "^weighted_cycles:\|^fps_bench_fps:" run.log

   You should see both keys. If fps_bench can't find the BIOS/ROM/replay,
   export `BIOS=`, `ROM=`, `REC=` env vars and rerun.
5. Initialize `results.tsv` (untracked, never committed):

        printf 'commit\tweighted_cycles\tfps\tstatus\tdescription\n' > results.tsv

6. Run the baseline and record it:

        bash scripts/dynarec_measure.sh > run.log 2>&1
        WC=$(awk '/^weighted_cycles:/  {print $2}' run.log)
        FP=$(awk '/^fps_bench_fps:/    {print $2}' run.log)
        SHA=$(git rev-parse --short=7 HEAD)
        printf '%s\t%s\t%s\tkeep\tbaseline\n' "$SHA" "$WC" "$FP" >> results.tsv

   Confirm the row looks sensible. This baseline is what every
   subsequent experiment measures against.

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
- `fps_bench/` and anything under `platform/`
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
`fps_bench_fps` must not regress more than **1%** vs the previous best.

Simplicity criterion (Karpathy's): tiny gains that add ugly code lose.
Neutral-or-better diffs that delete code always win. If you find yourself
writing a 200-line PR to shave 0.2% off weighted_cycles, step back and
try a different shape.

---

## Output format

`scripts/dynarec_measure.sh` prints exactly this block at the end
(anything before it is diagnostic noise the agent can ignore):

        ---
        weighted_cycles:   12345678
        fps_bench_fps:     60.0
        peak_vram_mb:      0.0
        seconds:           295.4

Target metric:

        grep "^weighted_cycles:" run.log | awk '{print $2}'

Sanity metric:

        grep "^fps_bench_fps:" run.log | awk '{print $2}'

`peak_vram_mb` is N/A for this dynarec (always 0.0) but kept for
Karpathy-template parity.

---

## Logging results

`results.tsv`, tab-separated, 5 columns, header row mandatory:

        commit    weighted_cycles    fps    status    description

- `commit` — 7-char git hash of the experiment commit (HEAD after your
  edit). `HEAD^` on discard/crash.
- `weighted_cycles` — integer, `0` on crash.
- `fps` — one decimal, `0.0` on crash.
- `status` — `keep` | `discard` | `crash`.
- `description` — one line, human-readable. No tabs.

Example after four rows:

        commit    weighted_cycles    fps    status    description
        abc1234   12345678   60.0   keep     baseline
        def5678   11800000   60.0   keep     NZCV held in host I8 vars across block
        012cafe   12400000   59.9   discard  fused cond_check into each emit_ - icache bloat
        cafe001   0          0.0    crash    patterns.rs UDIV magic miscompile vs interp

`results.tsv` is untracked. Never commit it.

---

## The experiment loop

        LOOP FOREVER:
            1. Note current branch/commit.
            2. Pick an experiment from the idea menu below.
               Edit mod.rs / patterns.rs only.
            3. cargo test -p arm7tdmi --features dynarec
               - must be green. If not, fix or revert and goto 1.
            4. git commit -am "<short description>"
            5. bash scripts/dynarec_measure.sh > run.log 2>&1
            6. grep "^weighted_cycles:\|^fps_bench_fps:" run.log
               - empty output => crash. tail -n 50 run.log. Decide
                 fix-in-place vs discard.
               - fps regressed > 1% vs previous best => discard.
               - weighted_cycles went up => discard.
               - weighted_cycles went down AND fps within 1% => keep.
            7. Record the row in results.tsv.
            8. keep     => branch advances, continue to 1.
               discard  => git reset --hard HEAD~1, continue to 1.
               crash    => git reset --hard HEAD~1, continue to 1.

Timeout: each measurement run is ~5 minutes. >10 min => kill and
treat as crash.

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

1. **MUL-by-constant** → shift-add tree or `madd`. Trigger: `MOV Rtmp,
   #imm ; MUL Rd, Rs, Rtmp` where `imm` is small/power-of-2.
2. **UDIV/SDIV-by-constant via Granlund–Montgomery.** Magic
   `M = ceil(2^(32+k)/d)` → arm64 `umulh ; lsr`. Trigger: the standard
   magic-multiply sequence, or BIOS `SWI 0x06` preceded by `MOV r1, #imm`.
3. **Branchless abs**: `(x ^ (x>>31)) - (x>>31)` → `cmp ; cneg`.
4. **CLZ polyfill** (de Bruijn 0x077CB531 table, Stanford binary search)
   → arm64 `clz`.
5. **popcount SWAR** (0x55555555 / 0x33333333 / 0x0f0f0f0f / 0x01010101)
   → arm64 NEON `cnt ; addv`.
6. **Branchless MIN/MAX**: `b + ((a-b) & -(a<b))` → `cmp ; csel`.
7. **Shift-pair sign/zero extend** `LSL #24 ; ASR #24` (and 16-bit) →
   `sxtb` / `sxth` / `uxtb` / `uxth`.
8. **Byte-swap idiom** → `rev`.
9. **GBA BIOS CpuSet / CpuFastSet fold** (`SWI 0x0B` / `0x0C` + known
   `r2` mode bits) → inlined memcpy/memset stencil.
10. **Dead DP chain DCE**: drop register-write chains whose results
    are all overwritten before block exit without being observed.

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
