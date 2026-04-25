# LLVM JIT — work stopped here, pivoted to AOT

**Date:** 2026-04-25
**Last commit on this branch:** `2051619` (dynarec: drop redundant cpsr round-trip on block dispatch)
**Pivot branch:** `aot-llvm`

## Summary

The LLVM-via-inkwell JIT integration on this branch reaches a structural ceiling well below the cached-interpreter scalar baseline. After ripping out Cranelift and consolidating around LLVM-only dispatch with trampoline-mode block compilation, SDL replay perf landed at:

| build | pokeemerald | mario kart | divs |
|-------|-------------|------------|------|
| cached_interp scalar | 540 fps | 383 fps | 0 (ref) |
| LLVM JIT (--jit) | 170-200 fps | 85-100 fps | 0 |

The JIT is **2.7-4.4× slower** than scalar. Each iteration of optimization on this branch (separate vs merged abort trampolines, every-other vs every-fourth abort cadence, cpsr round-trip elision, two-trampoline split, block chaining) confirmed the same structural truth: **the per-instruction `extern C` trampoline boundary dominates. Scalar's tight inline loop with PGO'd direct handler calls is unbeatable through trampoline-mode dispatch alone.**

## What it would take to beat scalar

LLVM has the optimizer to match — but only if every per-instruction operation is **inlined into the LLVM IR**, eliminating the trampoline boundary. That requires:

- Per-format inline IR for all Thumb formats (F1-F19) and ARM
- Inline fetch (direct bus access from LLVM IR — bus is `Shared<I>` generic, tricky)
- Inline scheduler advance (`add_cycles` baked in)
- Inline abort check (read flag bytes via baked offsets)
- Lazy NZCV materialization
- Block chaining (only pays off after per-instr cost drops)

Each of those is non-trivial individually. Layering them is the real engineering project. Doing this incrementally on top of the existing JIT pipeline drags along the trampoline architecture. **A clean AOT-LLVM rewrite, designed inline-from-day-one, is the better shape.**

## Pivot

Work continues on branch `aot-llvm` as an autoresearch program (Karpathy-style — single scalar, single run command, one experiment loop). See `docs/aot-llvm-program.md` (on that branch) for the program spec.

The current branch `shape-opt/apr22` is preserved in case JIT learnings prove useful — particularly the trampoline definitions, the per-format compile fns, and the SDL replay validation harness. Nothing is being deleted from history.

## Receipts

Detailed measurements and rejected experiments:

- `feedback_*.md` and `dynarec_*.md` in `~/.claude/projects/.../memory/` — narrative notes per session.
- Last 3 commits on this branch — incremental JIT optimizations (split trampoline, cpsr elision).
- SDL test scripts: `/tmp/sdl_divs.sh`, `/tmp/sdl_divs_mk.sh` (regenerate scalar reference with `target/release/rustboyadvance-sdl2 --no-audio --replay …`).
