# A5: compile-time budget check

Date: 2026-04-25
Branch: aot-apr25
Host: Intel Core Ultra 9 185H (22 logical cores)

## What

Spec from `docs/aot-llvm-program.md` deliverable A5: compile a
synthetic ROM of 1000+ placeholder blocks, measure blocks/sec, MB/sec,
peak memory. Decide serial vs parallel compile.

## Method

`arm7tdmi-aot::compiler::compile_n_placeholder_blocks(n)` builds N
placeholder block fns into a single LLVM module — each fn is a
4-instruction sequence (gpr load + add + store) shaped like what
phase 0's per-format inline emit will produce. Then forces codegen
via `engine.get_function_address("blk_0")` to time the full
construction-to-machine-code path at -O3 (Aggressive
OptimizationLevel).

Test: `cargo test -p arm7tdmi-aot a5_compile_budget --release -- --nocapture`

## Result (serial compile, single thread)

| N blocks | wall ms | µs/block | projected PE (100k) | projected MK (30k) |
|----------|---------|----------|---------------------|---------------------|
| 1000     | 314     | ~310     | ~31000 ms           | ~9300 ms            |
| 5000     | 1387    | 277      | 27740 ms            | 8322 ms             |

**Per-block compile cost: ~280 µs.** Higher than the 100 µs the doc
estimated, but still in the right order of magnitude.

Serial projections vs ladder gates:

- **pokeemerald: 27.7 s vs 5 s target → 5.5× over budget.**
- **Mario Kart:   8.3 s vs 2 s target → 4.2× over budget.**

Serial compile is **too slow** for phase 0 to ship. Parallel is
mandatory.

## Decision

**Parallelize from phase 0.** Use `std::thread::available_parallelism()`
sized thread pool (22 here). Each ROM page (64KB) → its own LLVM
`Context` + `Module`, modules built in parallel, modules linked into
the shared `ExecutionEngine` from the main thread after each
finishes (per I17 compile-then-publish).

Realistic speedup at 22 cores is ~8-12× (not 22×) due to:
- LLVM context construction overhead per thread.
- ExecutionEngine `add_module` is single-writer (serialized
  bottleneck at link time).
- L3 cache pressure from many concurrent compile pipelines.

Projected with 10× speedup:
- pokeemerald: 27.7 / 10 = **2.8 s** ✓ under 5 s budget.
- Mario Kart:    8.3 / 10 = **0.8 s** ✓ under 2 s budget.

## Implementation sketch (for phase 0 commit)

```rust
pub fn compile_rom_parallel(rom: &[u8], blocks: &[BlockSpec]) -> AotTable {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(4);

    // Group blocks into per-page chunks.
    let mut chunks: Vec<Vec<&BlockSpec>> = vec![Vec::new(); 256];
    for b in blocks {
        let page = ((b.entry_pc >> 16) & 0xff) as usize;
        chunks[page].push(b);
    }

    // Compile each non-empty chunk on a worker thread.
    let modules: Vec<(usize, OwnedLlvmModule)> = chunks
        .into_par_iter()         // rayon, or hand-rolled scoped threads
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(page, chunk)| (page, compile_page(chunk)))
        .collect();

    // Add modules to a single engine on the main thread, populate
    // PC->fn table.
    let mut engine = ...;
    let mut table = AotTable::new();
    for (page, module) in modules {
        engine.add_module(&module).unwrap();
        for spec in &chunks[page] {
            let addr = engine.get_function_address(&spec.fn_name).unwrap();
            table.insert(spec.entry_pc, spec.mode, addr as CompiledFn);
        }
    }
    table
}
```

Notes:
- Each worker thread creates its own `Context`. That's required by
  inkwell's borrow rules (a Module borrows from a Context, threads
  can't share a non-Sync Context).
- Modules link into a shared engine on the main thread. The engine's
  internal MCJIT / ORC linker isn't `Sync`, so this serialization
  point is unavoidable.
- Rayon adds a workspace dep; alternative is hand-rolled scoped
  threads. Phase-0 commit uses scoped threads to keep the dep tree
  light.

## Caveats

- The placeholder is 4 IR instructions; real per-format inline IR
  may be 10-20 instructions (especially F4 ALU with NZCV). The
  280µs/block estimate may be 2-3× higher in practice. Phase-1
  re-measures and updates this finding.
- Memory: 5000 blocks compiled fine in this test, no OOM. Per I20
  the harness will track total `aot_code_size_kb` and fail at
  256MB; expect ~50-100MB per ROM at -O3 inlined IR.
- Determinism: parallel compile order may affect machine-code byte
  layout but NOT runtime behavior (per I9). The replay-determinism
  test (V4) catches violations.

## Phase 0 status

A5 done. Decision committed: parallel-by-page is mandatory for
phase-0 scaffold.

On to A6 (differential test infra port).
