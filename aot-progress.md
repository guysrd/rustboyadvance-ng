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

**Next deliverable:** phase 4 (inline fetch + abort check) is where
actual fps gain starts. Two paths:

a) Rust-level: a NEW trampoline `aot_thumb_step_no_fetch` that
   skips `load_16` per iter; a once-per-block `add_cycles_const`
   call inside the AOT compiled fn pre-emits the Thumb fetch cost.
   Pipeline[] state is restored at block exit via a final load_16.
b) LLVM-level: emit_per_instr_thumb_block that bakes the cycle-add
   directly into IR. Currently has a divs-at-scale bug
   (AOT_USE_PER_INSTR=1) under investigation per phase 1a.

Plus phase 2 leftovers: F8 (LDSB/LDRH/LDSH reg-offset), F14 PUSH/POP,
F15 LDM/STM. F16/F18/F19 are block terminators handled by the AOT
emit, not by aot_thumb_step.

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

## Phase 1b/1c/2/5 (rust-inline) — incremental accept

`aot_thumb_step` in arm7tdmi/src/cpu.rs now has Rust-level inline
fast paths for:
- F1 MoveShiftedReg LSL/LSR/ASR Rd, Rs, #imm5
- F2 AddSub ADD/SUB Rd, Rs, Rn or #imm3
- F3 DataProcessImm MOV/CMP/ADD/SUB Rd, #imm8
- F4 AluOps all 16 ops (incl. MUL with cycle accuracy)
- F5 HiRegOpOrBranchExchange ADD/CMP/MOV/BX (Rd=R15 → PipelineFlushed)
- F6 LdrPc (literal pool load)
- F7 LdrStrRegOffset LDR/STR reg-offset, byte/word
- F9 LdrStrImmOffset LDR/STR imm5-offset, byte/word
- F10 LdrStrHalfWord LDRH/STRH imm5*2 offset
- F11 LdrStrSp LDR/STR sp-relative word
- F12 LoadAddress ADD Rd, [PC|SP], #imm8
- F13 AddSp ADD/SUB SP, #imm7<<2

Missing (deferred): F8 (LDSB/LDRH/LDSH reg-offset, 4 sub-cases),
F14 PUSH/POP, F15 LDM/STM, F16 Bcc tail (terminator), F18 B tail
(terminator), F19 BL pair (terminator).

Each is bit-exact with its scalar handler — same alu helpers, same
idle_cycle calls, same PipelineFlushed/AdvancePC semantics, same
Seq/NonSeq next-fetch-access encoding.

Trace seed pc fix: trace-out wrote pipeline-head pcs (= exec_addr
+ 4 in Thumb / +8 in ARM) but scan interprets entry_pc as exec_addr.
SDL frontend now subtracts 4/8 before passing seeds to scan. Fixed
in commit f0e5434.

Empirical:
- PE 100-seed trace-in: 0 divs, 0.52% coverage.
- sweep=0: 0 divs both ROMs, score in [-8, +3] noise band.

Phase 1 full accept (and any meaningful fps gain) blocked by the
F19-orphan trampoline bug documented in
docs/findings-phase1-trampoline-divs.md. The trace-pc fix exposed
~31 divs at 24052 trace seeds (was 32 before fix); the residual
bug is in trampoline cycle accounting on the F19 lo orphan path,
not in the inline fast paths. Coverage stays low at sweep=0 until
that's fixed OR phase 4 inlines fetch into LLVM IR (which sidesteps
the per-iter trampoline path entirely).

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

- 41412f4 phase 5: f7 ldr/str reg-offset + f10 halfword imm-offset inline
- 422928e phase 5: f9 ldr/str imm-offset + f11 sp-rel inline
- a529065 phase 5: f6 ldr pc-rel inline (literal pool)
- f0e5434 fix: trace seed pcs are pipeline-head, subtract 4/8 before scan
- 015159a phase 2: f2 add/sub, f5 hi-reg, f12 load-addr, f13 add-sp inline
- 56b3b26 phase 1c: f1 shifts + f4 alu inline
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
