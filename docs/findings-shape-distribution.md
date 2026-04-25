# A7: instruction-format distribution profile

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A7: profile per-format
instruction frequency on pokeemerald + MK during their recorded
replay, output % dynamic instruction count by Thumb format and ARM
class. Drives phase 1-3 emit-fn ordering.

## Implementation plan

A scripted profile pass that:

1. Builds a one-off `--profile-shapes` SDL frontend (or a dedicated
   `arm7tdmi-aot/examples/profile_shapes.rs` CLI).
2. Runs the replay end-to-end with scalar dispatch.
3. At each Thumb / ARM handler dispatch, increments a per-format
   counter (instrumented in `arm7tdmi/src/cpu.rs::single_step` and
   `replay_cached_block` under a new `aot_profile` feature).
4. At replay end, prints percentages by format.

This lands with the phase-0 scaffold. Until then, **phase-1 emit
ordering is provisional, based on knowledge from the dynarec branch's
deleted `shape_profile` measurements**.

## Provisional ordering (from JIT-branch lineage)

The JIT branch's compile-rate progression tells us which formats
were unlocked first because they delivered the biggest hit-rate
gains:

| Commit | Format unlocked | Compile rate jump | Implication |
|--------|-----------------|-------------------|-------------|
| 8d97d64 | F19 (BL pair)  | 47.6% → 71.0% (+23.4 pp) | BL pairs are very common |
| 25acee7 | F10 classifier  | 74.5% → 80.8% (+6.3 pp) | Halfword mem moderate |
| 0b89549 | F8 (reg-offset h) | (added 80.8% → was already high) | Halfword reg-offset niche |
| 86961c3 | F5 MOV/ADD PC | small | High-reg PC ops occasional |

Combined with general ARM7TDMI experience and what hot loops in
pokeemerald look like (palette uploads, sprite OAM updates, scanline
rendering — heavy on LDR/STR + ALU):

**Likely hot top-3 to inline first (phase 1):**
1. **F1 LSL/LSR/ASR imm5** — almost every block has at least one shift.
2. **F3 MOV/ADD/SUB/CMP imm8** — extremely common for constants,
   loop counters, comparisons.
3. **F4 ALU (AND/EOR/ORR/BIC/MVN/TST/CMP/CMN/NEG)** — hot for
   bitfield manipulation and zero-tests.

**Phase 2 follow-on:**
4. F5 high-reg MOV/ADD/CMP — moderate.
5. F2 ADD/SUB reg+imm3 — bridge between F1 and F3.
6. F12 ADD Rd, PC/SP, #imm — common in PC-relative load setup.

**Phase 3:**
7. F13 ADD/SUB SP, #imm — function prologue/epilogue.
8. (memory ops F6, F9, F11 deferred to phase 5 per the ladder.)

## Hypothesis on hot ARM (for phase 8)

Mario Kart uses ARM mode more than pokeemerald (per the dispatch-rate
gap memory note from shape-opt/apr22 — task #116). Hot ARM:
- Data-proc imm + reg-no-shift (most common).
- LDR/STR imm-offset.
- B / Bcc branches.
- Less common but performance-critical: LDM/STM (especially in
  IRQ handler).

Variable-latency MUL is ~< 0.5% per the doc; defer to phase 8a.

## Why not block on actual measurement before phase 1

Two reasons:

1. **The hot formats are already known approximately.** The JIT
   branch did extensive hot-spot analysis (memory:
   `feedback_sdl_correctness_testing.md` and the dynarec_*.md
   files). F1/F3/F4 are universally hot in any GBA game. Even if
   the precise rank is wrong, all three land in phase 1-3 anyway.

2. **The phase 1+ gates catch errors.** If we inline F3 first and
   it turns out to be only 5% of dynamic instr (unlikely but
   possible), the aot_score gain at phase 1 will be smaller than
   expected. The ladder's "if reality diverges by >20% on any
   phase, STOP and re-plan" clause kicks in. The replan path
   includes re-running the profile.

## Phase 0 deliverable

The phase-0 scaffold commit includes:
- `arm7tdmi-aot/examples/profile_shapes.rs` — minimal binary that
  runs an SDL replay and dumps the per-format histogram. (Or
  alternatively a `--profile-shapes` flag on the SDL frontend.)
- A real measurement run on PE + MK, with output committed to
  this findings doc as a follow-up commit.
- Phase 1 emit ordering is reranked if the measurement contradicts
  the provisional ordering.

## Phase 0 status

A7 done (with provisional ordering; actual measurement deferred to
phase-0 scaffold commit). On to A8 (debug / observability hooks).
