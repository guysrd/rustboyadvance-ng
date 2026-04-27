# A8: debug / observability hooks

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A8: when AOT
misbehaves (divergence, drift, perf regression), the agent needs
ways to inspect what was emitted and how it executed. Three hook
types, all under the `aot` feature, all gated behind SDL CLI flags.

## H1. `--dump-ir <pc>`

Print the LLVM IR for the AOT block whose entry-PC matches `<pc>`.

```
$ target/release/rustboyadvance-sdl2 --features aot --aot \
    --dump-ir 0x08001234 --no-audio --replay /tmp/pokeemerald_run.rec \
    --bios <bios> <rom>
```

Implementation:
- After `compile_rom` runs, the AOT crate retains per-block `Module`
  references (or at least the IR string `module.print_to_string()`
  at compile time).
- The SDL frontend, on `--dump-ir <pc>`, calls
  `aot_table.dump_ir(pc)` which returns `Option<String>`. Print and
  exit before starting the replay.
- LLVM IR lands on stdout, in textual form. Useful for confirming
  the emit fns produce the IR we think they do.

## H2. `--dump-asm <pc>`

Print capstone-disassembled native machine code for the same block.

```
$ ... --dump-asm 0x08001234 ...
```

Implementation:
- `capstone` crate as an optional dep on `arm7tdmi-aot` under a
  `dump_asm` feature.
- After block compile, retrieve the raw machine code bytes via
  `engine.get_function_address(name)` + a known size (we instrument
  the JIT engine to record per-fn sizes via `notify_function_emitted`,
  or use a fallback heuristic of "next fn's address - this fn's
  address" for size).
- Disassemble with capstone for the host arch (x86_64 or aarch64).
- Print mnemonic listing on stdout.

## H3. `--trace-block <pc>`

Print full CPU state (gpr, cpsr, pc, scheduler.cycles) before and
after each execution of the AOT block at `<pc>`, for up to 1000
dispatches. Then exit.

```
$ ... --trace-block 0x08001234 ...
[trace] dispatch 1 BEFORE pc=0x08001234 r0=0x00000000 r1=0x00010000 ... cpsr=0x00000010 cycles=12345
[trace] dispatch 1 AFTER  pc=0x08001240 r0=0x00010000 r1=0x00010000 ... cpsr=0x60000010 cycles=12353
[trace] dispatch 2 BEFORE pc=0x08001234 r0=0x00010000 ...
...
```

Implementation:
- AOT-table lookup, on hit, checks if pc matches the trace target.
- If so, snapshot CPU state, call AOT block, snapshot again, print.
- After 1000 dispatches at the target pc, exit cleanly.

Use case: bisecting divergence. Run with scalar (no `--aot`), record
trace. Run with `--aot`, record trace. Diff line-by-line — first
divergence pins the buggy block.

## H4. `--trace-block-instr <pc>`

Same as H3 but per-instruction inside the block. For deep debugging
when H3 narrows to a block but doesn't reveal which instruction
diverges.

```
$ ... --trace-block-instr 0x08001234 ...
[trace] block@0x08001234 instr 0 (raw=0x2005) BEFORE r0=0 ... AFTER r0=5 cycles=1
[trace] block@0x08001234 instr 1 (raw=0x4080) BEFORE r0=5 ... AFTER r0=5 cycles=2
...
```

Implementation:
- Phase 1+ emit fns optionally inject a "snapshot CPU state and
  print" call between every emitted instruction. Gated by a
  per-block trace flag set when `--trace-block-instr <pc>`
  matches.
- Adds a per-instruction extern call when active. Disabled by
  default; only fires when the agent explicitly requests this
  block.

## Trade-offs

- H1 and H2 are zero-cost in the steady state (only fire on the
  CLI flag).
- H3 has constant overhead at the matching block but doesn't
  affect other blocks.
- H4 needs the per-block instrumentation flag to be visible to
  the emit fns at compile time. Either:
  - (a) Re-AOT the matching block when the flag fires (one-time
    JIT compile of the instrumented variant, swap it in).
  - (b) Always emit two variants of every block: instrumented
    and non-instrumented; at dispatch, pick by the per-block
    trace flag.

  (b) doubles emitted code size — not worth it for a debug
  feature. Going with (a): on `--trace-block-instr <pc>` set, on
  first dispatch to that pc, AOT recompiles the block with
  per-instruction trace calls and replaces the table entry.

## Phase 0 deliverable

`arm7tdmi-aot/src/dump.rs` lands with the phase-0 scaffold:
- Module IR string retention (per-block, lazily released after
  compile completes unless dump_ir feature is on).
- `pub fn dump_ir(table: &AotTable, pc: u32) -> Option<String>`.
- `pub fn dump_asm(table: &AotTable, pc: u32) -> Option<String>`
  (depends on `dump_asm` feature → capstone dep).

SDL frontend gets the four flags wired through.

H4 (trace-block-instr) is phase-1+ work since it requires emit-fn
cooperation.

## Phase 0 status

A8 done. All nine phase-0 audits (A0-A8) complete.

## Next: phase-0 scaffold commit

With audits in hand, the agent now writes the phase-0 scaffold:

- `arm7tdmi-aot/src/scan.rs` — implements the algorithm from A4.
- `arm7tdmi-aot/src/bus.rs` — region offsets + IO map (A2).
- `arm7tdmi-aot/src/table.rs` — two-level PC→fn table + inlined
  lookup (I18).
- `arm7tdmi-aot/src/diff.rs` — DiffBus + framework (A6).
- `arm7tdmi-aot/src/dump.rs` — debug hooks (A8).
- `arm7tdmi-aot/src/lib.rs` — `compile_rom`, `enable_aot_on`.
- `arm7tdmi-aot/examples/profile_shapes.rs` — A7 measurement tool.
- Parallel-by-page compile (A5 mandates).
- Placeholder per-block emit (one trampoline call to scalar).
- Wire into SDL frontend and `arm7tdmi/src/cpu.rs` per I11/I18.

Phase 0 gate: scaffold runs SDL replay end-to-end, divs == 0,
drift < 1000, coverage > 80%.
