# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 0 scaffold (audits done, code next).
**Next deliverable:** phase 0 scaffold commit.

## Accepted phases

- A0 — pre-flight ABI sanity. PASSED. arm7tdmi-aot scaffold compiles,
  inkwell+LLVM-18+prefer-dynamic JIT-executes a constant fn returning
  42 on this host. Toolchain confirmed.

- A1 — handler-pipeline-read audit. NO Thumb or ARM handler reads
  `cpu.pipeline[]`. Inline-fetch optimization (skip read_16, keep
  add_cycles) is safe for ALL formats. Pipeline state needs reload
  on AOT→scalar exit.

- A2 — IO-region map. Inlinable RAM regions: BIOS read, EWRAM, IWRAM,
  cart ROM read. Real bus call: IOMEM/PALRAM/VRAM/OAM/SRAM. Region
  check at compile time when address constant; runtime ladder
  otherwise. const IO_REGIONS + is_io_addr/is_rom_region/
  is_inlinable_ram fns specced for arm7tdmi-aot/src/bus.rs.

- A3 — indirect-branch fallback. Strategy: simple "miss → scalar
  forever" for phase 0. Trace-pass deferred to phase 9 if coverage
  measurement shows we need it. Coverage thresholds: >80% green,
  50-80% green-but-watch, <50% blocking.

- A4 — ROM scan strategy. Cart entry decoded from ROM bytes 0-3 as
  ARM B (verified PE: 0xea00007f → 0x08000204; MK: 0xea00002e →
  0x080000c0). BIOS pc=0 ARM. Scan walks BB queue, classifies each
  insn, terminates on indirect/exception, caps at 32 instr per block
  per I16. I22 validation per target.

- A5 — compile budget. Measured 277 us/block at -O3 (1387 ms for
  5000 blocks on Intel Core Ultra 9 185H). Serial PE = 27.7s,
  MK = 8.3s — both over budget. DECISION: parallel-by-page is
  mandatory for phase 0 scaffold. 22 cores, realistic 10× speedup,
  PE projects to 2.8s ✓.

- A6 — differential test infra. DiffBus (bus with cycle counter,
  no scheduler events). diff_thumb(opcode, gpr, cpsr) compares
  scalar handler vs AOT IR on identical inputs. phase-0 ships
  scaffold; phase 1+ adds per-format tests.

- A7 — instruction-format distribution. Provisional ordering:
  F1 + F3 + F4 lead phase 1. Actual measurement runs as part of
  phase-0 scaffold (profile_shapes example binary) and reranks
  if needed.

- A8 — debug hooks. --dump-ir <pc>, --dump-asm <pc>,
  --trace-block <pc>, --trace-block-instr <pc> specced.

## Pending

**Phase 0 scaffold commit (next):**
- arm7tdmi-aot/src/scan.rs (per A4)
- arm7tdmi-aot/src/bus.rs (per A2)
- arm7tdmi-aot/src/table.rs (two-level PC→fn, I18 inlined lookup)
- arm7tdmi-aot/src/diff.rs (per A6)
- arm7tdmi-aot/src/dump.rs (per A8)
- arm7tdmi-aot/src/lib.rs additions: compile_rom, enable_aot_on
- arm7tdmi-aot/examples/profile_shapes.rs (per A7)
- Parallel-by-page compile (per A5)
- Placeholder per-block emit (one trampoline call to scalar)
- arm7tdmi/src/cpu.rs: AotDispatchHook + raw ptr field for I18 lookup
- arm7tdmi/src/cache.rs: replay_cached_block checks AOT first
- platform/rustboyadvance-sdl2: --aot + dump flags + enable_aot
  call ordering per I11
- scripts/aot_measure.sh — V1+V2+V3+V4 harness

**Phase 0 gates:**
- divs == 0 on PE + MK
- |drift| < 1000
- coverage > 80%
- replay determinism (V4)
- diff_failures == 0 (trivially passes since aot side is placeholder)

After phase 0 ships and the gates are green, phase 1 starts:
inline F1 + F3 + F4 (or whatever the actual A7 measurement
shows).

## Reminder loop

CronCreate scheduled at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, read this file, continue from
the resume marker.

## Recent commits on this branch

- a4a6c0e docs: aot-llvm autoresearch program
- 192fa88 strip llvm jit, cache_interp scalar only
- (audits committed in 2 batches: A1-A4, A5-A8)
- (next: phase-0 scaffold)
