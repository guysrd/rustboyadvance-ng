Plan: docs/dynarec-arm64-program.md (autoresearch-style) + Cranelift-disasm dump helper + shape benchmark harness
Context

arm7tdmi/src/dynarec/mod.rs uses Cranelift to JIT ARM7TDMI blocks into host code. On Android arm64 (the PGO target in scripts/build_apk_pgo.sh + scripts/build_pgo_android.sh) the emitted code is arm64. We've never read what Cranelift produces per shape (DP imm/reg-no-shift, ARM mem/branch, Thumb formats 1/2/3/4-logical/5/9/11/14, BX/Bcc/B/POP-pc tails). emit_flag_update (NZCV repack, mod.rs:3029) and emit_cond_check (NZCV unpack, mod.rs:2285) wrap almost every compiled block, so wins there multiply. Beyond per-shape micro-opts, GBA game code and the GBA BIOS are full of hand-rolled bit tricks (divide-by-constant, CLZ, popcount, abs, min/max, shift-pair sign-extend, BIOS CpuSet) that a block-wide pattern matcher can recognize and lower to a single arm64 instruction with a magic constant.

The doc is written as an autoresearch program.md (Karpathy format): an autonomous research loop an agent runs end-to-end, not a hand-written optimization narrative. One target scalar, one run command, one TSV, LOOP FOREVER.
Shape of the change

┌──────────────────────────────────────────────────────────────────┐
│ arm7tdmi/src/dynarec/mod.rs   (existing behavior preserved)      │
│   DynarecCompiler ──► try_compile_* ──► CLIF ──► Cranelift ──► arm64
│                            ▲                                      │
│                            │ (new pre-pass, always-on)            │
│   opcodes ──► patterns::try_match ──► synthesized shape ──────────┘
│                     │                       │
│                     │                       └─► hand-written CLIF
│                     │                           stencil w/ magic
│                     ▼                           constants
│               (falls through to existing emit_* if no pattern)
└──────────┬────────────────────────────────────────────────────┬──┘
           │ feat="dynarec_asm_dump"                            │ always-on
           ▼                                                    ▼
   dynarec/dump.rs                                  dynarec/patterns.rs
   compile_and_dump(opcodes) ─► (bytes, disasm)     try_match / emit
           │                                                    │
           ▼                                                    ▼
   tests/dynarec_asm_baseline.rs                tests/dynarec_pattern_differential.rs
   (one test per existing shape + synthesized)  (gpr+cpsr parity vs interpreter)
           │                                                    │
           └────────────────────┬───────────────────────────────┘
                                ▼
                   scripts/dynarec_measure.sh
                   ─► prints ONE scalar: weighted_cycles
                                │
                                ▼
                   docs/dynarec-arm64-program.md
                   (autoresearch loop; agent reads+follows)
                                │
                                ▼
                        results.tsv (untracked)

Critical files
New

    docs/dynarec-arm64-program.md — the autoresearch program. Section layout mirrors Karpathy's template (## Setup, ## Experimentation, ## Output format, ## Logging results, ## The experiment loop). Full section contents below.
    arm7tdmi/src/dynarec/dump.rs — feature-gated (dynarec_asm_dump). One public fn per existing try_compile_* entry point, returning (Vec<u8>, String) = raw bytes + disasm listing. Bypasses the JIT define_function path and calls ctx.compile(&*isa, &mut ControlPlane::default()) directly so CompiledCode::code_buffer() is available. Uses capstone (new optional dev dep under the same feature) for arm64 disassembly.
    arm7tdmi/src/dynarec/patterns.rs — always-on under dynarec. pub(crate) fn try_match(opcodes: &[…]) -> Option<Pattern> plus one emitter per pattern. Called at the top of each full-block try_compile_* before the per-instruction classifier loop. Returning None = fall through unchanged (so the initial stub is a true no-op).
    arm7tdmi/benches/dynarec_shapes.rs — Criterion micro-bench, one bench per shape (existing + synthesized). Compiles the block once, then times black_box(func(gpr_ptr, cpsr_ptr, …)) in a tight loop. New bench feature gates the criterion dep so default builds are unmoved.
    arm7tdmi/tests/dynarec_asm_baseline.rs — one #[test] per shape, prints === SHAPE=<name> ===\n<disasm>\n=== END === to stdout. Gated behind dynarec_asm_dump. Not run in CI. Used by the agent to seed the "current arm64" field for each shape when it decides to tackle it.
    arm7tdmi/tests/dynarec_pattern_differential.rs — for every synthesized shape, asserts gpr+cpsr match the interpreter bit-for-bit on crafted opcode sequences. Reuses differential / differential_with_flags (currently in mod.rs's #[cfg(test)] mod tests; move to a pub(crate) helper module so integration tests can call them).
    scripts/dynarec_measure.sh — the measuring function. Deterministic. Runs:
        cargo bench -p arm7tdmi --bench dynarec_shapes --features dynarec,bench -- --noplot --quiet, collects median ns/iter per shape.
        cargo build --release -p rustboyadvance-sdl2 --features dynarec once, then for each of Pokemon Emerald and Mario Kart Super Circuit runs the built binary with --replay <fixed RBAREC01 recording> --no-audio --skip-bios --bios <fixed BIOS>. Window stays on — we want the whole GPU + audio-mixer + scheduler path exercised exactly as a real gameplay session would, with audio only silenced (NullAudio sink) so the SDL audio device doesn't add host-dependent noise. Replay exits as soon as the last recorded keypad edge is past, at which point the frontend prints a `replay done: N frames in T.TTs wall, F.F avg fps (...)` summary line.
        Computes a single scalar weighted_cycles: sum over shapes of ns_iter[shape] * call_count_in_SDL_replay[shape]. Call counts come from a one-shot profile generated by a lightweight counter installed in BlockCache::finish_record under a new shape_profile cfg flag (off by default, on for this script only).
        Each ROM is measured separately — an optimization that helps pokeemerald but regresses mario_kart must fail the keep gate, and vice versa. The ROMs' numbers are combined into the canonical output block as two distinct fps lines.
        Prints the canonical block:

        ---
        weighted_cycles:    12345678
        pokeemerald_fps:    312.4
        mario_kart_fps:     297.1
        peak_vram_mb:       0.0   # N/A for dynarec, kept for parity with Karpathy template
        seconds:            295.4

        Single number to minimize: weighted_cycles. Lower is better. Both `pokeemerald_fps` and `mario_kart_fps` are sanity checks — **each** must not regress more than 1% versus the previous best for the change to be kept.

Modified

    arm7tdmi/Cargo.toml — add features dynarec_asm_dump (enables dynarec + optional capstone) and bench (enables optional criterion). Both deps optional.
    arm7tdmi/src/dynarec/mod.rs — add #[cfg(feature = "dynarec_asm_dump")] pub mod dump; and mod patterns;. In each full-block try_compile_*, prepend:

    if let Some(p) = patterns::try_match(opcodes) { return Some(p.emit(self, …)); }

    No edits to any existing emit_*. Move differential / differential_with_flags from #[cfg(test)] mod tests to pub(crate) mod test_utils so integration tests can call them.

Reused (do NOT duplicate)

    DynarecCompiler::decode_* classifiers — patterns.rs calls them to recognize trigger sequences.
    emit_conditional_instr, emit_thumb_format*, emit_conditional_mem, emit_mem_body, emit_data_processing_imm, emit_flag_update, emit_cond_check — dump.rs calls them verbatim so disasm matches production.
    dynarec::trampolines — reused by dump.rs for shapes needing bus access.
    platform/rustboyadvance-sdl2/ — the full SDL frontend, including the new --replay + --no-audio flags. scripts/dynarec_measure.sh shells out to its release binary. Window stays on; audio goes to the NullAudio sink.
    scripts/measure.sh, scripts/measure_device.sh, the APK record/replay harness (platform/rustboyadvance-jni/) — scripts/dynarec_measure.sh does not wrap these directly, they're separate measurement paths kept around for perf-record and on-device runs.

The program.md itself (Karpathy format, section by section)

The doc lives at docs/dynarec-arm64-program.md. Contents:
## Setup

    Agree on a run tag based on today's date (e.g. apr22). Branch shape-opt/<tag> must not exist.
    git checkout -b shape-opt/<tag> from current master.
    Read the in-scope files: arm7tdmi/src/dynarec/mod.rs, arm7tdmi/src/dynarec/patterns.rs, arm7tdmi/src/dynarec/dump.rs, arm7tdmi/benches/dynarec_shapes.rs. Read scripts/dynarec_measure.sh but do NOT modify it.
    Verify scripts/dynarec_measure.sh runs end-to-end on this machine.
    Initialize results.tsv with just the header row (untracked, never committed).
    Run the baseline: bash scripts/dynarec_measure.sh > run.log 2>&1. Record the baseline row. Confirm setup looks good.

## Experimentation

What you CAN modify: arm7tdmi/src/dynarec/mod.rs, arm7tdmi/src/dynarec/patterns.rs. Everything in these files is fair game — rewrite emit_flag_update, add a new synthesized shape to patterns.rs, change the CLIF emitted for a Thumb format, introduce a new shape classifier, etc.

What you CANNOT modify: prepare.py-equivalents here are scripts/dynarec_measure.sh, arm7tdmi/benches/dynarec_shapes.rs, platform/rustboyadvance-sdl2/ (the full SDL frontend the measure script wraps), arm7tdmi/src/dynarec/dump.rs, and all the tests/dynarec_* harnesses. These are the evaluation. You also cannot add new crate dependencies or modify Cargo.toml feature gates.

Correctness gate: cargo test -p arm7tdmi --features dynarec must still pass after every change. If it fails, the experiment is a crash — log and revert.

Goal: minimize weighted_cycles from dynarec_measure.sh. Both pokeemerald_fps AND mario_kart_fps must not regress more than 1% vs the **global best so far** (the min-weighted_cycles across all kept commits, not the previous keep — prevents drift from 10 sequential 0.9% sub-gate regressions). Each game is a separate regression check — wins on one ROM that regress another are not kept. The per-class cpu_pct / gpu_pct / bus_pct fields are diagnostic (interpret the move), not gating.

Simplicity criterion: same as Karpathy's — tiny gains that add ugly code lose; neutral-or-better diffs that delete code win.
## Output format

scripts/dynarec_measure.sh prints (exact block the agent greps):

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

Extract:

grep "^weighted_cycles:" run.log        # target
grep "^pokeemerald_fps:" run.log        # gate (per-ROM)
grep "^mario_kart_fps:"  run.log        # gate (per-ROM)
grep "_cpu_pct:\|_gpu_pct:\|_bus_pct:" run.log   # diagnostic

## Logging results

results.tsv, tab-separated, **10 columns**:

commit	weighted_cycles	pokeemerald_fps	pokeemerald_cpu_pct	pokeemerald_gpu_pct	mario_kart_fps	mario_kart_cpu_pct	mario_kart_gpu_pct	status	description

    short git hash (7 chars)
    weighted_cycles — integer, 0 on crash
    pokeemerald_fps / mario_kart_fps — .1f, 0.0 on crash
    pokeemerald_cpu_pct / mario_kart_cpu_pct — .1f diagnostic, 0.0 on crash
    pokeemerald_gpu_pct / mario_kart_gpu_pct — .1f diagnostic, 0.0 on crash
    status: keep | discard | crash
    short description (no tabs)

(bus_pct / audio_pct / other_pct are printed in run.log for debugging
but left out of results.tsv to keep the table diffable.)

Example:

commit	weighted_cycles	pokeemerald_fps	pokeemerald_cpu_pct	pokeemerald_gpu_pct	mario_kart_fps	mario_kart_cpu_pct	mario_kart_gpu_pct	status	description
abc1234	12345678	312.4	61.8	11.5	297.1	62.3	10.9	keep	baseline
def5678	11800000	318.8	63.7	11.3	299.3	64.0	10.7	keep	NZCV held in host nzcv via icmp+flag-use
012cafe	12400000	313.0	61.9	11.5	289.1	62.4	10.8	discard	fused cond_check — mario_kart FPS regressed >1%
cafe001	0	0.0	0.0	0.0	0.0	0.0	0.0	crash	patterns.rs UDIV magic - miscompile vs interp

Never commit results.tsv.
## The experiment loop

LOOP FOREVER:
1. Load global best from results.tsv (min weighted_cycles, max each FPS, `status=keep` only).
2. Pick an experiment idea (menu below). Edit mod.rs / patterns.rs.
3. `cargo test -p arm7tdmi --features dynarec` - must be green.
4. `git commit -am "<short desc>"`
5. `bash scripts/dynarec_measure.sh > run.log 2>&1`
6. `grep "^weighted_cycles:\|^pokeemerald_\|^mario_kart_" run.log`
   - Empty output on weighted_cycles => crash. `tail -n 50 run.log`, decide fix-in-place vs discard.
   - EITHER pokeemerald_fps OR mario_kart_fps regressed >1% vs its global best => discard. Each game is a separate check.
   - weighted_cycles > global best => discard.
   - weighted_cycles < global best AND BOTH fps numbers within 1% of their global best => keep.
   - weighted_cycles inside noise (within 0.5% of global best) AND no class-pct moved > 1pp => re-run ONCE; if still inside noise on the second pass, discard as a coin flip.
7. Record row in results.tsv (all 10 columns).
8. keep => branch advances.
   discard/crash => `git reset --hard HEAD~1`.

Timeout: each run ~12–15 minutes (cargo bench ~2 min + SDL build ~1 min + two SDL replays with perf ~11 min, dominated by the ~32 min of emulated pokeemerald gameplay captured in the default recording). >25 min => kill and treat as crash.

If you need faster iterations, set `NO_PERF=1` to skip the `perf record` passes — drops the `_cpu_pct`/`_gpu_pct`/`_bus_pct` fields (they come back as 0.0) and saves ~1–2 min per run. Unset it before making a keep/discard decision; those fields help interpret subsystem shifts.

Retraining the per-shape W_* weights in dynarec_measure.sh is allowed every ~10 kept commits (or when shape_profile reveals the hot distribution has shifted). Commit it with `status=keep` and `description=retrain weights (not a code experiment)`. The resulting WC shift is expected.

NEVER STOP: loop indefinitely. If ideas run out, re-read mod.rs, re-read the menu, combine near-misses, try more radical CLIF rewrites or hand-written stencils.
## Experiment idea menu

Part A — common-sink helpers (wins multiply):

    emit_cond_check (mod.rs:2285): skip the per-instr CPSR unpack by holding N/Z/C/V in separate Cranelift I8 vars across a block; materialize back to the packed word only at block exit.
    emit_flag_update (mod.rs:3029): same — avoid the repeated ushr/band/ishl/bor dance. Consider a single iconcat/bitselect when Cranelift lowers it well on arm64.

Part B — per-shape emitters (in mod.rs file order):

    emit_conditional_instr (2355), emit_thumb_format14 PUSH/POP (2390), emit_thumb_format11 SP-rel LDR/STR (2457), emit_thumb_format9 imm-offset LDR/STR (2488), emit_thumb_format5_non_branch (2538), emit_thumb_format1 shifted move (2598), emit_thumb_format4_logical (2699), emit_thumb_format2 add/sub (2759), emit_thumb_format3 imm8 arithmetic (2801), emit_conditional_mem (2855), emit_mem_body (2891), emit_data_processing_imm (2969). For each: identify redundant gpr_ptr loads when the same ARM register is used twice in a block, hoist the load; replace ushr_imm + band 1 preserve-C patterns with a preserved Variable.

Part C — synthesized shapes (magic-number lowering in patterns.rs):

    MUL-by-constant → shift-add tree or single madd. Trigger: MOV Rtmp,#imm ; MUL Rd,Rs,Rtmp.
    UDIV/SDIV-by-constant via Granlund–Montgomery M = ceil(2^(32+k)/d) → arm64 umulh ; lsr. Trigger: the standard magic-multiply sequence or a SWI 0x06 preceded by MOV r1,#imm.
    Branchless abs (x ^ (x>>31)) - (x>>31) → cmp ; cneg.
    CLZ polyfill (de Bruijn 0x077CB531 table or Stanford binary search) → clz.
    popcount SWAR (0x55555555/0x33333333/0x0f0f0f0f/0x01010101) → NEON cnt ; addv.
    Branchless MIN/MAX b + ((a-b) & -(a<b)) → cmp ; csel.
    Shift-pair sign/zero extend LSL #24 ; ASR #24 (and 16-bit) → sxtb/sxth/uxtb/uxth.
    Byte-swap idiom → rev.
    GBA BIOS CpuSet/CpuFastSet fold (SWI 0x0B/0x0C + known r2 mode bits) → inlined memcpy/memset stencil.
    Dead DP chain DCE — drop chains whose results are all overwritten before block exit without being observed.

Each new synthesized shape lands with:

    An entry in patterns.rs.
    A new #[test] in dynarec_pattern_differential.rs covering the trigger sequence + boundary cases.
    Its dynarec_asm_baseline.rs test showing the emitted arm64.

Implementation order (what lands in the initial plan PR, before the agent loop starts)

    Add dynarec_asm_dump + bench features + optional capstone / criterion deps in arm7tdmi/Cargo.toml.
    Move differential / differential_with_flags out of mod.rs's test module into pub(crate) mod test_utils.
    Write dump.rs, one entry per existing try_compile_*.
    Write benches/dynarec_shapes.rs, one bench per existing shape (synthesized shapes added as they land).
    Stub patterns.rs with try_match returning None + the mod patterns wiring in mod.rs + the if let Some(p)… pre-pass on every full-block try_compile_*. Verify existing tests + benches are byte-identical.
    Write tests/dynarec_asm_baseline.rs (one #[test] per existing shape) and an empty tests/dynarec_pattern_differential.rs (real tests come with each synthesized shape).
    Write scripts/dynarec_measure.sh with the shape-profile counter installed in BlockCache::finish_record under cfg(shape_profile). Verify it produces the canonical output block and a single weighted_cycles number.
    Write docs/dynarec-arm64-program.md using the section contents above.

Steps 1–8 are what this plan produces. Step 9 onward is the agent running the experiment loop defined by the doc, one weighted_cycles-reducing commit at a time. results.tsv is the log of record for those follow-ups.
Verification

    cargo build -p arm7tdmi (no features) — unchanged.
    cargo build -p arm7tdmi --features dynarec — unchanged, byte-identical binary vs. pre-plan (because patterns::try_match is a stub returning None).
    cargo test -p arm7tdmi --features dynarec — existing differential tests pass; new pattern differential harness compiles (and has zero tests until the first synthesized shape).
    cargo test -p arm7tdmi --features dynarec_asm_dump -- --nocapture dynarec_asm_baseline — produces per-shape arm64 disasm the agent reads when picking an experiment.
    bash scripts/dynarec_measure.sh > run.log 2>&1 ; grep "^weighted_cycles:\|^pokeemerald_fps:\|^mario_kart_fps:" run.log — prints all three numbers. Baseline recorded in results.tsv row 1.
    Open docs/dynarec-arm64-program.md, confirm all five Karpathy sections (Setup, Experimentation, Output format, Logging results, The experiment loop) plus the idea menu are present and that the exact commands/grep patterns above are in the doc verbatim.

