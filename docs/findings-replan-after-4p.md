# Re-plan after Phase 4-prime extended attempts

Per program-doc "Phase-N permanent-fail re-planning" rule: phase 4P
has now been attempted three times with different approaches. The
ladder needs a refresh.

## What was tried

| Attempt | What | Outcome |
|---------|------|---------|
| Phase 4P (original) | 19 Thumb formats inlined as IR via `fetch_only` extern per opcode | 0 PE divs / 1 MK div but PE 431 / MK 364 — net regression vs whole-block (PE 480 / MK 372). Postmortem in `findings-phase4-prime-postmortem.md`. |
| Phase 4P-A (foundation) | WAITCNT gen counter on SysBus + CpuOffsets + SDL plumbing. Data-only, no IR change. | Committed 1451e03. Behavioral no-op. |
| Phase 4P-B (baked + gen-check) | Replace per-opcode `fetch_only` extern with in-module IR helper that does inline cycle add. Lookup-side gen-check on `AotTable` to disable AOT post-WAITCNT. | Committed dcb0b55. 0 divs ✓ but PE writes WAITCNT within first ~50 emulated cycles, gen-check kills AOT after 7 dispatch hits, can't measure fps. |
| Phase 4P-C (live cycle_luts) | Helper IR loads cycle_luts from a `*const usize` baked at AOT setup; always-current values, gen-check unnecessary. | Reverted (a21de0d). Deadlocks PE in BIOS animation, Thumb coverage collapsed 21M→6k hits, ARM dispatched 2.6B times. Root cause not identified. |

Total: 4 phase-4P attempts. Original + three extensions. None
crossed the "score > 0 on at least one ROM at sweep=64KB" gate
on the `aot-apr25` recordings.

## Why phase 4P keeps failing

Two structural issues compound:

1. **i-cache pressure dominates the 19-fmt path.** Per postmortem:
   ~5KB native code per fully-inlined block × ~30k blocks = ~150MB
   of JIT'd code. Even at low coverage the hot inner loop's icache
   footprint is too large. Each format added past ~6 stacks linearly.

2. **WAITCNT-stale baking is a correctness mine field.** PE writes
   WAITCNT during cart-init. Anything that bakes cycle constants
   at AOT compile time goes stale immediately. The two safe paths
   (gen-check + scalar-fall-back, OR live cycle_luts read) each
   have their own bugs we couldn't fully resolve in the available
   debugging time.

The fundamental premise of phase 4P — that LLVM-inlining the per-
opcode body cross the block boundary buys cycles — does not
materialize empirically. Cross-opcode optimization within a block
doesn't fold the way the original hypothesis predicted.

## What's left in the postmortem ladder

Per `findings-phase4-prime-postmortem.md`:

- **Phase 7 (block chaining)** — tail-call between sequential AOT
  blocks via LLVM `set_tail_call(true)`. Skips dispatcher
  overhead between adjacent compiled fns. Estimate +5-15% PE.
  Multi-day. Includes I10 stack-safety mechanism.

- **Phase 8 (ARM-mode IR)** — mirror the 19-fmt Thumb inline IR
  work for ARM. MK is 98% ARM; current ARM AOT coverage is 1.4%
  with whole-block trampoline only. Whole-block ARM was tried
  earlier (phase8-* commits), got to 54% MK coverage but was 0.9%
  slower than scalar — meaning ARM whole-block trampoline overhead
  also dominates. Inline-IR ARM (like phase 4P for Thumb) would
  be needed instead. Multi-WEEK, not multi-day.

- **Phase 9 (persistent compile cache)** — UX, not fps. Required
  for `--aot` on by default since current compile time is ~140s
  on the recordings. Independent of correctness/fps work.

## New recommendation

Given the compounding evidence that LLVM-inlining-per-opcode is the
wrong optimization shape for this emulator's parameter regime, the
next investment of operator time should be **phase 7 (block
chaining) NOT phase 8 (more ARM IR)**.

Reasoning:
- Phase 7 is an architectural lever that helps regardless of how
  inline the per-opcode IR is. It attacks dispatcher overhead.
- Phase 8 ARM IR shares the same i-cache pressure problem as
  phase 4P (probably worse — ARM has more formats).
- Phase 7 has a clean fail-fast property: if `set_tail_call(true)`
  doesn't actually emit jumps in the JIT's asm, the chain fallback
  exits to dispatcher (no correctness cost, just no fps win).

If phase 7 ALSO fails, the program should genuinely ship `--aot`
correctness-only (Option C from `findings-ladder.md`) and call it
done. The implicit early-stop at "AOT matches scalar with 0 divs
+ scalar fallback for unsupported shapes" is a viable ship state
even without the +5% gate.

## Process learning

The cron loop discipline of "one commitable step per firing" works
for incremental work but breaks down on multi-day phases like 4P-C.
Each firing tries to make progress and instead burns ~5 min of
compile + bench time on a feature that needs hours of focused
debugging.

For phase 7: when work begins, plan it as ONE multi-firing block
under a single mental context, NOT a series of "small commits per
firing." Otherwise we'll get the same shallow-then-revert pattern.

## Status

Replan committed. The cron loop is in HOLDING state per program-doc
"per-firing action while paused": ONE measurement appended per fire,
no code changes, until the operator gives a phase-7 green-light.

Recordings are stable in `/home/user/pokeemerlad/recordings/`;
benches use those paths.
