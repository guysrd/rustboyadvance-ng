# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 1 (per-format inline IR) — phase 1a done
but has a divergence bug at scale; phase 1b (inline F3 MOV imm8)
blocked on debugging the scale issue.

**Next deliverable:** debug the per-instr emit divs-at-scale bug.
Likely options:
1. Emit ALL blocks into ONE shared LLVM module (vs one-per-block).
2. Use the A8 dump-ir flag on a divergent block to see the IR.
3. Compare phase-1 emit vs phase-0 whole-block on a single test
   block — they should produce semantically identical state.

Once the bug is fixed, phase 1b inlines F3 MOV imm8 (replaces a step
trampoline call with direct gpr write + cpsr update IR for that
opcode).

## Phase 0 — ACCEPTED ✓

Per `scripts/aot_measure.sh` at default sweep=0:
- PE: divs=0, drift=0, determinism=1, fps_aot=541.9 vs scalar=545.6
- MK: divs=0, drift=0, determinism=1, fps_aot=387.5 vs scalar=390.1
- diff_failures=0, aot_score=-6 (within noise)

Coverage 0% by default — the trampoline-mode whole-block emit is
correctness-fragile at scale, so default sweep=0 ships clean.

## Phase 1a — partial

Architecture (commit 634aa1f):
- `arm7tdmi-aot::replay::aot_thumb_step_for<I>` per-iter trampoline
- `arm7tdmi-aot::replay::aot_block_should_abort_thumb_for<I>` K=2
  abort check trampoline
- `LlvmCompiler::register_step_thumb` registers them
- `LlvmCompiler::emit_per_instr_thumb_block` emits LLVM IR with one
  step call per opcode + abort check at K=2 cadence
- `compile_rom_with_seeds_and_step` variant takes step+abort fns
- SDL `AOT_USE_PER_INSTR=1` env var toggles between phase-0
  whole-block (default) and phase-1 per-instr

Empirical results:
- PE sweep<=2KB phase-1: 0 divs ✓ (small scale works)
- PE sweep=4KB phase-1: 36 divs (regression)
- PE sweep=8KB phase-1: 36 divs
- PE sweep=64KB phase-1: 89 divs
- PE trace-in (24052 entries) phase-1: 255 divs (every frame)

Tried fixes (didn't help):
- Per-module unique extern names (commit 74b3e15) — didn't fix divs.
- Disabled abort check — got WORSE (152 divs); confirms abort path
  isn't the bug source.

Bug is structurally in the multi-extern-call emit path. The
trampoline `aot_thumb_step_for` delegates to `cpu.aot_thumb_step`
which is the same code the whole-block trampoline calls in its
loop — so the per-call SEMANTICS should match. But empirically
many sequential extern calls from LLVM IR produce divergent state.

Hypothesis: LLVM JIT engine's symbol/state management bug at
scale, OR a calling-convention issue, OR a memory aliasing
inference issue.

## Pending

### Debug phase 1a divergence (next)

Steps:
1. Pick one divergent block. Use `--dump-ir <pc>` (per A8) to see
   what LLVM emitted. Compare against expected pattern.
2. Use `--dump-asm <pc>` to see native instructions.
3. Try emitting all blocks into one shared module (avoids
   per-module-state issues).

### Phase 1b: inline F3 MOV imm8

After 1a is debugged, replace the step trampoline call for F3
MOV imm8 opcodes with direct LLVM IR (gpr[rd] = imm8 + flag
update). Diff test per A6.

## Reminder loop

CronCreate `c1a9dd9e` at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, this file, continue from resume
marker.

## Recent commits on this branch

- 74b3e15 phase 1a debug: per-module unique extern names (didnt fix)
- 634aa1f phase 1a: per-instr dispatch architecture (gated)
- dbc49d8 phase 0d: trace-driven aot entry points
- c4deab4 phase 0 ACCEPT: harness + measurement
- 9df4c0e phase 0 step 4c: Bcc-as-Linear + coverage counters
- 758b0d9 phase 0 step 4b: placeholder block emit
- 9dc0326 phase 0 step 4a: compile_rom plumbing
- (audits A0-A8 across 3 batches)
