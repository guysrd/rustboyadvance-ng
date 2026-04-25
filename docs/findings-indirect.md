# A3: indirect-branch fallback decision

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A3: decide what
happens for PCs the static ROM-scan misses (typically jump tables,
BX register, LDR PC, return-address-on-stack patterns).

The doc estimates ~30-40% of code is statically un-traceable. If the
actual miss rate on pokeemerald + MK is < 50%, the simple
"miss → scalar forever" strategy is acceptable. If > 50%, we need a
trace-pass that logs runtime BX targets and seeds the AOT scan with
them.

## Decision (phase-0 commitment)

**Strategy: simple miss-falls-to-scalar.**

For PCs not in the AOT table, the dispatcher (per I18 inlined
lookup) gets `None` and falls through to the existing
`replay_cached_block` scalar path. The block records under
cached_interp normally, runs scalar from then on. We do **not**
re-AOT misses on the fly — that would drag JIT-mode latency back
in (the JIT branch's reason for failure, per `docs/llvm-jit-status.md`).

If the AOT static scan covers >= 50% of dispatcher-tick PCs, the
remaining 50% in scalar still gets us most of the per-instruction
inlined-IR win on the AOT-covered hot paths.

## Why this is the right phase-0 choice (without measuring first)

Three reasons:

1. **Simplicity.** No runtime profiling pass, no trace persistence,
   no second-pass scan. Phase 0 is "scaffold + correctness"; complex
   scan augmentation is phase-9 work.

2. **Worst-case acceptable.** Even if miss rate is 60-70% on some
   ROMs, the AOT path still covers BIOS + cartridge-entry +
   straight-line code blocks that the scalar replay loop currently
   wins on. The AOT win is on hot-pathed straight-line code; the
   scalar fallback is fine for jump-tabley dispatch code.

3. **Measurement is cheap to add later.** If phase 5 shows we're
   leaving fps on the table because of low coverage, we add the
   trace-pass as phase 9 (per the ladder). The measurement
   instrumentation is `aot_coverage_pct_<rom>` from the harness,
   already specced in V1 outputs.

## Trace-pass deferred to phase 9

If measurement shows blocking-low coverage, phase 9 adds:

```
1. Run pokeemerald + MK with scalar-only build instrumented to log
   every observed BX target, LDR-PC target, and POP{...,PC} return
   address.
2. Save trace to ROM-keyed file (e.g.,
   target/release/.aot-cache/<sha>.trace).
3. AOT static scan, second pass: read trace, queue every observed
   target as an additional entry point.
```

This is an optional optimization; the simple strategy gates phase 0
without it.

## Coverage thresholds

For phase 0 acceptance (per ladder gate):
- coverage > 80% → green, proceed.
- coverage 50-80% → green, but log for phase-9 trace-pass scope.
- coverage < 50% → blocking; revisit A3 strategy before phase 1.

The harness will report coverage_pct in V1 outputs, so this gate
fires automatically.

## Phase 0 status

A3 done (decision made, no measurement required pre-phase-0). On to
A4 (ROM-scan strategy spec).
