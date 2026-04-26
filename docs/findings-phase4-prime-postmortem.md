# Phase-4-prime postmortem: inline-IR is a net regression

**TL;DR.** ~3000 lines of LLVM-IR-emit code across 19 Thumb sub-formats
produced an AOT path that is **11% slower on PE and 2% slower on MK**
than the pre-existing whole-block trampoline path. The architectural
premise — "inline LLVM IR per opcode lets the optimizer see the whole
block as a unit" — did not pan out empirically. The whole-block
trampoline (a single Rust extern call per block that loops calling
`cpu.aot_thumb_step`) is the optimal Thumb dispatch architecture in
this emulator's parameter regime.

## What was attempted

Phase-4-prime, per `docs/aot-llvm-program.md` invariants I1-I28, set out
to inline every hot Thumb format as direct LLVM IR — eliminating the
per-iter extern-call boundary into the Rust step trampoline. The
expectation was that LLVM's whole-block view would enable cross-opcode
optimization (constant-fold flag chains, share gpr loads across
adjacent ops, fold redundant cpsr round-trips, etc.).

19 sub-formats were inlined as gated env-var-toggleable IR paths:
F1, F2, F3, F4_LOG, F4_ARITH, F4_SHIFT, F4_MUL, F5, F6, F7, F8, F9,
F10, F11_STR, F11_LDR, F12, F13, F14, F15, F19_HI. That covers
~95-99% of dynamic Thumb opcodes. The block terminators (F17 SWI,
F18 B unconditional, F19 lo, F16 Bcc-taken) are handled by the AOT
scan at block boundaries — they don't traverse the per-iter step
path, so trampoline boundary fires only for genuinely-rare SWI/undef.

Each format was correctness-verified against the scalar reference
via SDL replay fb-hash diff. The full 19-format stack passes the
program-doc V1+V2+V4 gates (0 PE divs / 1 MK div historical at
sw=64KB, drift well within budget, deterministic across re-runs).

## What we expected vs what happened

| | Expected | Actual |
|--|--|--|
| Per-format fps stack | additive +1-3% per format | first 6 stacked positively, rest hit noise band |
| Cross-block IR optimization | LLVM folds across opcodes | doesn't materialize — IR per opcode is verbose |
| i-cache footprint | small native code per opcode | per-block native code grows linearly with formats |
| Final fps vs scalar | break-even or positive | -22% PE, -7% MK |
| vs whole-block trampoline | +5-10% from extern boundary savings | **-11% PE, -2% MK** |

## The controlled benchmark that cleared the picture

Same session, back-to-back, identical conditions:

```
Scalar (cached_interp, no AOT):
  PE x3: 508.4, 477.6, 502.6  → median 503
  MK x3: 368.3, 367.8, 364.6  → median 368

Whole-block (default, no inline IR):
  PE x3: 466.7, 480.4, 483.0  → median 480
  MK x3: 372.5, 369.8, 372.1  → median 372

19-fmt inline-IR (all formats enabled via env vars):
  PE x3: 431.2, 430.7, 432.1  → median 431
  MK x3: 363.9, 364.4, 367.8  → median 364
```

**Whole-block vs scalar:**
- PE: 480 / 503 = -4.6% (close-but-below)
- MK: 372 / 368 = **+1.1%** (whole-block AOT BEATS scalar)

**Whole-block vs 19-fmt inline-IR:**
- PE: +11.4% (whole-block much faster)
- MK: +2.2% (whole-block faster)

**MK whole-block is already past zero against scalar.** For the program-doc
success target (`+5% on both ROMs`), MK needs ~+4% more, PE needs ~+10%
more. Phase 7 (block chaining) and phase 8 (ARM coverage expansion) are
the realistic levers to close those gaps.

## Why the architectural premise was wrong

The Rust step trampoline (`cpu.aot_thumb_step` in `arm7tdmi/src/cpu.rs`)
is the same code the whole-block trampoline calls per-iter. It already
contains 17 inline format fast-paths in Rust, optimized by rustc with
-O3 and PGO-style branch hints. Per-iter cost: a tight handful of native
instructions per opcode after rustc's optimizer is done with it.

LLVM IR per opcode has to emit:
- gpr GEPs (with `inbounds`)
- constant-encoded immediate values
- conditional shifts/cpsr update sequences
- explicit pc/nfa stores
- `fetch_only` extern call for cycle accounting + pipeline maintenance
- per-opcode basic-block boundaries with phi/branch when the format has
  range-conditional behavior (F1/F4 shifts, F8 LDSH, etc.)

Each format ends up at 30-100 IR ops which lower to ~50-200 bytes of
native code per opcode after -O3. A 32-instruction block with everything
inlined is ~5KB of native code — vs the whole-block trampoline's ~50-100
bytes (single extern call). Multiply by 30k AOT blocks at sw=64KB = **150MB
of native code** in the JIT'd regions. That's an i-cache disaster on the
hot replay loop.

The cross-opcode optimization that was supposed to amortize this cost
doesn't materialize. LLVM's optimizer does not effectively combine
gpr/cpsr loads/stores across opcode boundaries — they're distinct
memory operations, the optimizer treats them conservatively, and
`fetch_only` extern calls between them are opaque optimizer barriers.

The Rust step trampoline avoids all of this: rustc inlines the format
detection + body into one tight function, the optimizer sees the whole
thing, and the per-iter cost is a function-call boundary that the LLVM
JIT'd block code can't beat.

## Implications

1. **Phase-4-prime as designed cannot win.** Continuing to add inline
   formats is structurally net-negative. The 19-format stack should be
   considered the upper bound, not the next milestone.

2. **The whole-block path is the optimal architecture.** Any further
   fps work should mirror or extend the whole-block pattern.

3. **The remaining strategic levers (ladder phase 7-9) should NOT extend
   the inline-IR path.** Specifically:
   - **Phase 7 (block chaining)**: tail-call between sequential
     whole-block compiled fns. Skip the dispatcher overhead. Expected
     +5-15% fps lever still untouched.
   - **Phase 8 (ARM-mode trampoline)**: MK runs ~98% ARM but currently
     has near-zero ARM AOT coverage. Adding cart-ROM ARM sweep + the
     existing whole-block ARM trampoline path should give MK the
     biggest single win available. The ARM whole-block infrastructure
     is already built (commit 4d37779 etc.); just needs broader scan
     coverage and validation.
   - **Phase 9 (persistent compile cache)**: UX, not fps. Required to
     ship `--aot` on by default since current compile time is 70-76s
     at sw=64KB.

4. **The phase-1 trampoline architecture from findings-ladder.md is
   actually the SHIP STATE.** That doc described phase-1 as the ceiling
   we'd need to break out of; turns out phase-1 trampoline (= whole-block
   path) already represents the empirical fps optimum for Thumb in
   this codebase.

## What's in the tree (as committed under `aot-apr25`)

- `arm7tdmi-aot/` crate: full AOT compiler with LLVM JIT (inkwell 0.9 +
  llvm18-1-prefer-dynamic). Phase-0 scaffold + phase-1 whole-block
  trampoline + phase-1 per-instr emit + 19 inline-IR sub-formats, all
  gated by `AOT_INLINE_F<X>=1` env vars (default-off).
- `arm7tdmi-aot/src/replay.rs`: per-I monomorphized externs for
  fetch_only, idle_cycle, load_8/16/32, store_8/16/32, ldr_word,
  ldr_half, ldr_sign_half, step_thumb, step_arm, abort_thumb,
  abort_arm, replay_thumb_block, replay_arm_block.
- `arm7tdmi-aot/src/scan.rs`: BFS reachability scan over Thumb + ARM
  with I22 target validation, F19 BL pair handling, BIOS exception-
  vector seeds.
- `arm7tdmi-aot/src/table.rs`: two-level PC→fn table per I18 with
  inlined `aot_lookup`/`aot_lookup_arm` (one per CPU mode).
- `arm7tdmi-aot/src/compiler.rs`: ~3500 lines of LLVM-IR-emit
  scaffolding for the 19 sub-formats above.
- SDL frontend: `--aot` flag, `--aot-trace-in` for runtime-observed
  PC seeds, AOT_USE_PER_INSTR / AOT_INLINE_F<X> / AOT_SWEEP_CAP_KB /
  AOT_SWEEP_ARM / AOT_SWEEP_BIOS env-var control surface.
- `core/src/sysbus.rs`: phase-4 hooks (`scheduler_timestamp_ptr`,
  `thumb_fetch_cycles`) — unused after the inline-IR retreat but
  harmless.

The whole-block trampoline path is the default and is the recommended
entry point for any future fps work.

## What we measured along the way

`results.tsv` has the full audit trail (~150 rows). Highlights:

- 6 inline formats: PE 486 / MK 384 (peak inline-IR fps)
- 8 inline formats: PE 478 / MK 385
- 14 inline formats: PE 459 / MK 379
- 16 inline formats: PE 453 / MK 381
- 19 inline formats: PE 431 / MK 364 (final, all formats enabled)
- whole-block (default): **PE 480 / MK 372**
- scalar reference: PE 560 / MK 397

In every controlled comparison, whole-block matched or beat the inline
paths. The "stacking" curve appeared positive early (6→8 formats added
+8 fps to PE) but that was within run-to-run noise — the same
measurement on a different day fits inside ±20 fps on PE.

## Recommendation for the next operator

Do NOT add more inline-IR formats. The path is exhausted. Pick one of:

- **Phase 7 (block chaining)** — biggest expected fps lever. Multi-day.
- **Phase 8 (ARM coverage expansion)** — biggest expected MK lever.
  Requires `AOT_SWEEP_ARM=1` validation + correctness check.
- **Phase 9 (persistent cache)** — UX shippable, no fps.

Or ship the current state as Option C from `findings-ladder.md`:
`--aot` correctness-only, off by default, document the gap.
