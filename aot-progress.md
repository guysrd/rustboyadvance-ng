# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 0 scaffold (step 2 of ~8).
**Next deliverable:** scan.rs — ROM reachability per A4 /
docs/findings-rom-scan.md.

## What's done

### A0-A8 phase-0 audits (all complete)

- **A0** (`docs/findings-pre-flight.md`) — pre-flight ABI sanity.
  Toolchain confirmed: LLVM_SYS_181_PREFIX=/usr/lib/llvm-18,
  inkwell 0.9 with llvm18-1-prefer-dynamic, JIT executes a
  constant fn returning 42.
- **A1** (`docs/findings-pipeline-read.md`) — no Thumb or ARM
  handler reads `cpu.pipeline[]`. Inline-fetch optimization
  (skip read_16, keep add_cycles) is safe for ALL formats.
  Pipeline state needs reload on AOT→scalar exit.
- **A2** (`docs/findings-io-regions.md`) — region map. Inlinable RAM
  regions: BIOS read, EWRAM, IWRAM, cart ROM read. Real bus call:
  IOMEM/PALRAM/VRAM/OAM/SRAM. Functions in bus.rs.
- **A3** (`docs/findings-indirect.md`) — simple "miss → scalar
  forever" fallback for phase 0. Trace-pass deferred to phase 9.
- **A4** (`docs/findings-rom-scan.md`) — scan algorithm specced.
  Cart entry from ROM bytes 0-3 as ARM B (verified PE, MK).
  classify_insn: Linear / DirectBranch / Conditional / IndirectBranch
  / ExceptionEdge. I22 validation per target.
- **A5** (`docs/findings-compile-budget.md`) — measured 277 us/block
  at -O3. Serial PE = 27.7s, MK = 8.3s — both over budget.
  DECISION: parallel-by-page mandatory.
- **A6** (`docs/findings-diffinfra.md`) — DiffBus + diff_thumb
  framework specced.
- **A7** (`docs/findings-shape-distribution.md`) — provisional
  ordering: F1 + F3 + F4 lead phase 1. Real measurement runs as
  part of phase-0 scaffold (profile_shapes example).
- **A8** (`docs/findings-debug-hooks.md`) — --dump-ir / --dump-asm
  / --trace-block / --trace-block-instr specced.

### Phase 0 scaffold step 1 (committed: 762493e)

- `arm7tdmi-aot/src/bus.rs` — region constants + classifiers.
- `arm7tdmi-aot/src/table.rs` — two-level PC→fn table with inlined
  lookup. **GOTCHA fixed**: 8-bit top index collides BIOS (0x0)
  with ROM (0x08000000); both have bits 16-23 = 0. Use 16-bit top
  (= pc >> 16). Heap-boxed top (512KB upfront).
- `CompiledFn` ABI (per I8): `extern "C" fn(cpu_ctx, pc_out) -> u32`.
  Single-pointer cpu_ctx, no separate gpr_ptr — avoids noalias bug.
- 7 unit tests passing (bus region classification + table
  insert/lookup).

## Pending in phase 0 scaffold

- **step 2: scan.rs** — implement A4. Decode cart entry B,
  walk reachable basic blocks, classify each insn, queue targets,
  cap blocks at 32 instr (I16). Validate per I22.

- **step 3: arm7tdmi hook for inlined lookup** —
  `arm7tdmi/src/cpu.rs` gets `*const AotTable` field under the new
  `aot_dispatch` feature in arm7tdmi (or a non-feature pub field
  with a sentinel null when unused). `replay_cached_block` calls
  `arm7tdmi_aot::aot_lookup` inline before falling through to
  scalar. Issue: `arm7tdmi` doesn't depend on `arm7tdmi-aot` (avoid
  inkwell in core). Solution: `arm7tdmi` defines the lookup signature
  via a fn-pointer or trait, `arm7tdmi-aot` populates it.

- **step 4: compile_rom (parallel-by-page)** — for each page,
  spawn a thread that creates a Context + Module, emits placeholder
  block fns (one trampoline call to scalar replay), -O3 compiles.
  Main thread `add_module` per finished page, populates AotTable.
  Per I17 compile-then-publish: AotTable returned only after ALL
  pages compiled.

- **step 5: enable_aot_on** — wires the AotTable into the CPU's
  hook field. Called from SDL frontend immediately after
  `GameBoyAdvance::new` (per I11).

- **step 6: diff.rs + dump.rs** — DiffBus per A6, debug hooks
  per A8.

- **step 7: SDL frontend** — `--aot` flag (calls
  `enable_aot_on(rom_bytes, entry_pc)`), `--dump-ir/--dump-asm`/
  `--trace-block/--trace-block-instr` flags. Compile progress on
  stderr.

- **step 8: scripts/aot_measure.sh** — V1 (replay diff) + V2
  (drift) + V3 (diff_failures) + V4 (determinism) gates.

- **step 9: profile_shapes example** — A7 actual measurement.

## Phase 0 acceptance gates

- divs == 0 on PE + MK
- |drift| < 1000 cycles
- coverage > 80%
- replay determinism (V4)
- diff_failures == 0

After phase 0 ships green, phase 1 starts: inline F1 + F3 + F4
emit fns (or whatever the actual A7 measurement shows hottest).

## Reminder loop

CronCreate scheduled at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, read this file, continue from
the resume marker.

## Recent commits on this branch

- 762493e phase 0 scaffold step 1: bus regions + PC->fn table
- (audits A5-A8 commit)
- (audits A1-A4 commit)
- (A0: arm7tdmi-aot scaffold)
- a4a6c0e docs: aot-llvm autoresearch program
- 192fa88 strip llvm jit, cache_interp scalar only
