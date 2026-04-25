# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 0 scaffold (step 4d of ~9).
**Next deliverable:** investigate remaining MK 1-div with sweep=64KB
post-Bcc-Linear fix, then parallel-by-page compile (per A5
mitigation 1) so wider sweep is feasible within budget.

## What's done — 21 commits on aot-apr25

### Phase 0 audits A0-A8
All 9 audits committed across 3 batches. See `docs/findings-*.md`.

### Phase 0 scaffold steps
- **step 1** (`762493e`): bus regions + 2-level PC→fn table.
  16-bit top index after BIOS/ROM collision bug.
- **step 2** (`d2e6164`): scan.rs ROM reachability.
  cart entry decoded from bytes 0-3, PE/MK verified.
- **step 3** (`44411d1`): arm7tdmi `aot_dispatch` hook.
  step_block.try_aot_dispatch() top of chain loop.
- **step 4a** (`9dc0326`): compile_rom plumbing + SDL --aot flag.
  empty table; AOT path engaged but always misses; 0 divs.
- **step 4b** (`758b0d9`): placeholder block emit (trampoline-mode).
  `aot_replay_thumb_block_for<I>` drives THUMB_LUT handlers
  with K=2 abort cadence; LLVM emit one trampoline call per block.
- **step 4c** (`9df4c0e`): Bcc-as-Linear scan fix + coverage counters
  + drop inter-AOT abort.
  scan no longer terminates at Bcc; runtime trampoline handles
  taken/not-taken via handler's CpuAction. fixes block-length
  mismatch with scalar's traced path. MK 4-divs → 1-div at sweep=64KB.

## Verified clean state

- `--features aot` builds cleanly.
- 23+ unit tests pass.
- SDL `--aot` (default sweep=0):
  - pokeemerald: 0 divs, ~540 fps (matches scalar — table empty).
  - mario kart: 0 divs, ~390 fps.

## Pending

### Step 4d: MK 1-div root cause + parallel compile

**MK 1-div with AOT_SWEEP_CAP_KB=64**: down from 4-divs but not 0.
The remaining div is at frame ~96 (~5760-frame index). Cycle drift
starts around frame 45 (~2700-frame index) — small (~9-77 cycles),
oscillates sign. Some specific instruction handling differs
between AOT and scalar in subtle cycle accounting. Bisect:
- sweep=0: 0 divs
- sweep=1KB: 1 div
- sweep=2KB: 1 div
- sweep=4KB through 64KB: 4 divs (pre-fix), 1 div (post-Bcc fix)

To diagnose: instrument the AOT trampoline with cycle-by-cycle
logging vs scalar's same-replay. Find the first instruction where
cycles differ. Likely candidates: variable-cycle instructions
(LDM/STM, mul, LDR with idle cycle).

**Parallel compile (per A5 mitigation 1)**: current sweep=64KB
takes 30s to compile (way over phase-0 5s budget). Can't expand
sweep without parallelizing. 22 cores available, expect ~10×
speedup → ~3s for sweep=64KB, ~10s for sweep=256KB.

Implementation: per-ROM-page LLVM Module + parallel build via
std::thread, link into shared engine on main thread.

### Step 5: phase 0 acceptance test + commit

Once 4d gates green:
- run `bash scripts/aot_measure.sh` (script TBD per V1+V2+V3+V4)
- coverage > 80% with sweep=512KB+
- divs == 0 both ROMs
- |drift| < 1000
- replay determinism check

Then phase 0 ships and we move to phase 1: inline F1/F3/F4 IR.

## Reminder loop

CronCreate scheduled at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, re-read this file, continue from
the resume marker.

## Recent commits on this branch

- 9df4c0e phase 0 step 4c: Bcc-as-Linear + coverage counters
- 758b0d9 phase 0 scaffold step 4b: placeholder block emit
- 9dc0326 phase 0 scaffold step 4a: compile_rom plumbing
- 44411d1 phase 0 scaffold step 3: arm7tdmi AOT hook
- d2e6164 phase 0 scaffold step 2: scan.rs
- 762493e phase 0 scaffold step 1: bus + table
- (audits A0-A8)
