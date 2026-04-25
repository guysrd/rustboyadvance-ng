# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 1 (per-format inline IR). Phase 1a's LLVM
per-instr emit has a divs-at-scale bug (gated behind
AOT_USE_PER_INSTR=1, ignore for now). Phase 1b shipped F3 family
(MOV/CMP/ADD/SUB imm8) inline in `aot_thumb_step`. Phase 1c adds
F1 (LSL/LSR/ASR imm5) and F4 (ALU low-reg, all 16 ops including
MUL).

**Next deliverable:** phase 2 candidates per A7 — F5 high-reg ops,
F2 ADD/SUB reg+imm3, F12 ADD Rd, PC/SP, #imm. After that, phase 4
inlines fetch and abort check (saves the per-iter `load_16` +
indirect call), which is where actual fps gain over scalar starts.

Trampoline F19-orphan divs at trace seeds (24052) is documented in
`docs/findings-phase1-trampoline-divs.md`; option-c partial-accept
chosen for now (ship Rust-level inline fast paths at sweep=0).

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

## Phase 1b/1c/2 (rust-inline) — incremental accept

`aot_thumb_step` in arm7tdmi/src/cpu.rs now has Rust-level inline
fast paths for:
- F1: LSL/LSR/ASR Rd, Rs, #imm5 (MoveShiftedReg)
- F2: ADD/SUB Rd, Rs, Rn or #imm3 (AddSub)
- F3: MOV/CMP/ADD/SUB Rd, #imm8 (DataProcessImm)
- F4: AND/EOR/LSL/LSR/ASR/ADC/SBC/ROR/TST/NEG/CMP/CMN/ORR/MUL/BIC/MVN
  (AluOps; all 16 ops, MUL has variable-latency idle_cycle replication)
- F5: ADD/CMP/MOV/BX hi-reg (Rd=R15 → PipelineFlushed)
- F12: ADD Rd, [PC|SP], #imm8 (LoadAddress)
- F13: ADD/SUB SP, #imm7<<2 (AddSp)

These skip the THUMB_LUT indirect call when the block is dispatched
through AOT. Each is bit-exact with its scalar handler — same alu
helpers, same idle_cycle calls, same PipelineFlushed semantics.

Empirical:
- PE 100-seed trace-in: 0 divs, 0.77% coverage.
- sweep=0: 0 divs both ROMs, score=+3 (noise, gray zone).

Phase 1 full accept (and any meaningful fps gain) blocked by the
F19-orphan trampoline bug documented in
docs/findings-phase1-trampoline-divs.md. Coverage stays low until
that's fixed OR phase 4 inlines fetch into the LLVM IR (which
sidesteps the per-iter trampoline path entirely).

## Pending

### Phase 2: F5 / F2 / F12 inline (per A7)

Continue Rust-level inline fast paths in `aot_thumb_step`:
- F5: high-reg MOV/ADD/CMP (Rd = R8..R15 or Rs = R8..R15).
  CMP/MOV are simple; ADD with Rd=R15 reload_pipeline16 is the
  branchy variant.
- F2: ADD/SUB Rd, Rs, Rn (reg) or #imm3 (imm).
- F12: ADD Rd, [PC|SP], #imm8 — PC-rel address compute.

### Phase 4: inline fetch + abort

This is where wins start showing up. Currently the per-iter
`load_16` for fetch + cycle bookkeeping happens via real bus
calls; phase 4 inlines them as direct memory loads + cycle adds.

## Reminder loop

CronCreate `c1a9dd9e` at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, this file, continue from resume
marker.

## Recent commits on this branch

- a0bd4f2 phase 1b: F3 family expansion + trampoline divs investigation
- fa38437 phase 1b: F3 MOV imm8 fast-path
- 74b3e15 phase 1a debug: per-module unique extern names (didnt fix)
- 634aa1f phase 1a: per-instr dispatch architecture (gated)
- dbc49d8 phase 0d: trace-driven aot entry points
- c4deab4 phase 0 ACCEPT: harness + measurement
- 9df4c0e phase 0 step 4c: Bcc-as-Linear + coverage counters
- 758b0d9 phase 0 step 4b: placeholder block emit
- 9dc0326 phase 0 step 4a: compile_rom plumbing
- (audits A0-A8 across 3 batches)
