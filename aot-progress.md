# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 0 scaffold (step 4b of ~9).
**Next deliverable:** placeholder block emit — LLVM fn per block
that calls `aot_replay_thumb_block_for<I>` trampoline with baked-in
opcodes_ptr + len + mode. Trampoline runs the block via THUMB_LUT
handler dispatch + K=2 abort check, mirrors scalar replay.
Coverage > 80% after this lands.

## What's done (16 commits on aot-apr25)

### Phase 0 audits A0-A8 (3 batched commits)

All 9 audits committed: pre-flight ABI sanity passing, no handler
reads pipeline (inline-fetch safe), IO regions mapped, indirect
fallback decided, ROM scan algorithm specced, compile budget
measured (277µs/block → parallel-by-page mandatory), diff infra
specced, shape distribution provisional ordering, debug hooks
specced. See `docs/findings-*.md`.

### Phase 0 scaffold step 1: bus + table (1 commit)

`arm7tdmi-aot/src/bus.rs` — region constants + classifiers.
`arm7tdmi-aot/src/table.rs` — two-level PC→fn table. **Bug fix:
8-bit top index collides BIOS w/ ROM; switched to 16-bit (65536-entry
heap-boxed top, 256KB leaves).** `aot_lookup` is `#[inline(always)]`
with `get_unchecked` on hot path.

### Phase 0 scaffold step 2: scan.rs (1 commit)

ROM reachability per A4. Cart entry decoded from ROM bytes 0-3 as
ARM B (verified PE: 0x08000204; MK: 0x080000c0). Classifier
distinguishes Linear / DirectBranch / Conditional / IndirectBranch /
ExceptionEdge / BlReturn for Thumb (F18 B, F16 Bcc, F17 SWI, F5 BX,
F19 BL pair, F14 POP{pc}, F4-/F5 ALU) and ARM (B/BL with cond, BX,
SWI, LDR PC, LDM with R15, ALU Rd=PC). 32-instr cap per I16,
validation per I22. **23 unit tests passing including real-ROM
smoke.**

### Phase 0 scaffold step 3: arm7tdmi AOT hook (1 commit)

New `aot_dispatch` feature on arm7tdmi gates two raw fields on
Arm7tdmiCore: `aot_table: *const u8` and
`aot_lookup_fn: Option<fn(*const u8, u32) -> usize>`. arm7tdmi stays
inkwell-free. `step_block` calls `try_aot_dispatch()` at the top of
each chain iter; on hit dispatches the CompiledFn (single-pointer
ABI per I8), interprets I8 return bits, reloads pipeline at branch
target, continues chain. On miss falls through to existing
block_cache path.

**Cold-start guard per I15**: lookup gated on
`cpu.pipeline[0] != 0`, so first dispatcher tick after reset always
falls to scalar. AOT takes over once pipeline is bootstrapped.

`arm7tdmi-aot::enable_aot_on(cpu, &table)` is the public install
entry point. Casts `&AotTable` to `*const u8`, registers
`aot_lookup_for_hook` (which casts back inside arm7tdmi-aot).

### Phase 0 scaffold step 4a: compile_rom plumbing + SDL --aot (1 commit)

`compile_rom(rom, base, entry_pc, mode) -> AotTable`: phase 0a
returns empty table — scan runs but no IR emitted yet. Step 4b will
add placeholder emit.

New `--features aot` on rustboyadvance-sdl2 (depends on
arm7tdmi-aot + aot_dispatch core feature). New `--aot` flag wired
BEFORE skip_bios per I11. AotTable boxed for static lifetime.

**Smoke test on pokeemerald: 0 divs vs scalar reference.** AOT path
engaged (lookup runs, always misses with empty table), dispatcher
falls through to scalar correctly.

## Pending in phase 0 scaffold

### Step 4b: placeholder block emit (next)

For each `BlockSpec` from scan, emit an LLVM fn that calls
`aot_replay_thumb_block_for<I>(cpu_ctx, opcodes_ptr, len, mode)`.
The trampoline runs the block via THUMB_LUT handler dispatch with
K=2 abort check, mirroring scalar replay. Coverage > 80% after.

Architecture:
- `arm7tdmi-aot/src/replay.rs` — the trampoline:
  `pub unsafe extern "C" fn aot_replay_thumb_block_for<I>(cpu_ctx,
  opcodes_ptr, len, abort_check) -> u32`. Drives THUMB_LUT
  handler dispatch with the same fetch + pipeline + cpsr semantics
  as scalar `replay_cached_block`. Returns I8 ABI bits.
- `arm7tdmi-aot/src/emit.rs` — LLVM IR emit:
  `pub fn emit_placeholder_block(compiler, spec) -> CompiledFn`.
  Creates a per-block fn that calls the trampoline with baked-in
  opcodes_ptr (allocated in a per-block Vec<u32>, kept alive by
  AotTable).
- `compile_rom` — for each scanned block, allocate the opcodes
  vec, emit IR, register the resulting fn in the AotTable.
- Per A5 + I17: parallel-by-page compile is a phase-0
  recommendation but the placeholder emit is small enough we can
  start serial and parallelize if budget exceeds 5s for PE.

### Step 4c: SDL replay verifies acceptance

Run `bash scripts/aot_measure.sh` (or just /tmp/sdl_divs.sh + MK
equivalent with `--aot` arg). Phase 0 acceptance:
- divs == 0 ✓ (already proven with empty table)
- |drift| < 1000
- coverage > 80% (need step 4b for this)
- replay determinism

### Step 5+ (post phase 0): per-format inline IR

Phase 1 inlines the top-3 hottest formats per A7 measurement
(provisional: F1 + F3 + F4). Each inline phase replaces trampoline
calls with direct IR for that format.

## Reminder loop

CronCreate scheduled at minutes 7,22,37,52 every hour. Each fire
re-reads docs/aot-llvm-program.md, re-reads this file, continues
from the resume marker.

## Recent commits on this branch

- 9dc0326 phase 0 scaffold step 4a: compile_rom plumbing + SDL --aot flag
- 44411d1 phase 0 scaffold step 3: arm7tdmi AOT dispatch hook
- d2e6164 phase 0 scaffold step 2: scan.rs (ROM reachability per A4)
- 762493e phase 0 scaffold step 1: bus regions + PC->fn table
- dae1ccb phase 0 audits A5-A8
- 9a8bec6 phase 0 audits A1-A4
- 469ae6a A0: arm7tdmi-aot scaffold + pre-flight ABI sanity passing
- a4a6c0e docs: aot-llvm autoresearch program
- 192fa88 strip llvm jit, cache_interp scalar only
