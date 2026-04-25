# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 1 (per-format inline IR).
**Next deliverable:** start phase 1 codegen — refactor `compile_thumb_block`
to dispatch per-opcode (one inline-IR sequence OR one trampoline
call per opcode) instead of one trampoline call per block. Then
implement inline emit for F3 MOV imm8 as the first format. Diff
test (per A6) verifies bit-exact match with scalar handler.

## Phase 0 — ACCEPTED ✓

Per `scripts/aot_measure.sh` at default sweep=0:
- PE: divs=0, drift=0, determinism=1, fps_aot=541.9 vs scalar=545.6
- MK: divs=0, drift=0, determinism=1, fps_aot=387.5 vs scalar=390.1
- diff_failures=0
- aot_score=-6 (within noise)

Coverage 0% by default — AOT path is plumbed and dispatched on every
block boundary, just doesn't have any blocks to find (the placeholder
trampoline whole-block emit is correctness-fragile at scale; needs
per-instruction emit which is phase 1).

`results.tsv` row `cac3181` records the accept.

## Trace-driven entry points (commit dbc49d8) — works but exposes trampoline bug

`--aot-trace-out PATH` dumps every recorded ROM block PC at replay
end. `--aot-trace-in PATH` reads them as scan seeds.

Generated:
- /tmp/pe_trace.txt: 24052 PE block PCs.
- /tmp/mk_trace.txt: 21565 MK block PCs.

With PE seeded: 31764 thumb blocks compiled in 28s, 74.48% coverage,
**but 32 divs vs scalar**. Same kind of bug that surfaces with
sweep>=25KB on MK. The trampoline-mode placeholder has subtle
correctness issues with real game blocks at scale. Cycle drift
starts at frame 180 (line 3 of hashes) with +68 cycles drift.

The fix is phase 1's per-format inline IR — replace the whole-block
trampoline with per-instruction emit. The bug may resolve naturally
once we're not going through the trampoline.

## Phase 1 plan

**Goal**: inline top-3 most-executed Thumb formats per A7 provisional
ordering (F1 LSL/LSR/ASR + F3 imm8 + F4 ALU). Each format gets a
per-instr emit fn that writes inline LLVM IR for the handler body.
Unsupported formats fall through to a per-instruction trampoline
call (different from the current per-block trampoline).

**Architecture (per-instruction dispatch)**:

```rust
// In compile_thumb_block, for each opcode k:
//   - emit K=2 abort check IR if k odd && k != 0
//   - decode opcode → format
//   - if format in {F3 MOV imm8, ...}: emit inline IR
//   - else: emit aot_thumb_step trampoline call
//   - check return: if 1 (PipelineFlushed), branch to exit_blk
// At exit, return 0.
```

This is the JIT branch's architecture. Port the relevant patterns
from `git show shape-opt/apr22:arm7tdmi/src/dynarec.rs`.

**Phase 1 sub-steps**:
1. arm7tdmi-aot/src/emit/mod.rs scaffold + per-instr architecture.
2. arm7tdmi-aot/src/diff.rs DiffBus + diff_thumb framework (per A6).
3. compile_thumb_block refactored to per-instruction dispatch.
4. Inline F3 MOV imm8 emit fn + diff test.
5. SDL replay verifies divs == 0 (with trace seeds, if possible).
6. Inline F1 LSL/LSR/ASR + F3 ADD/SUB/CMP imm8 + F4 ALU.
7. SDL replay + harness measurement → phase 1 acceptance.

## Reminder loop

CronCreate scheduled at 7,22,37,52 every hour. Each fire re-reads
docs/aot-llvm-program.md, this file, continues from resume marker.

## Recent commits on this branch

- dbc49d8 phase 0d: trace-driven aot entry points
- c4deab4 phase 0 ACCEPT: harness + measurement
- 9df4c0e phase 0 step 4c: Bcc-as-Linear + coverage counters
- 758b0d9 phase 0 step 4b: placeholder block emit
- 9dc0326 phase 0 step 4a: compile_rom plumbing + SDL --aot
- 44411d1 phase 0 step 3: arm7tdmi AOT hook
- d2e6164 phase 0 step 2: scan.rs
- 762493e phase 0 step 1: bus + table
- (audits A0-A8 across 3 batches)
