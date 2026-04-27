# Phase-1 trampoline divs investigation

Date: 2026-04-25
Branch: aot-apr25

## Symptom

With `--aot --aot-trace-in /tmp/pe_trace.txt` (24052 trace seeds),
PE diverges from scalar by 32 frame-hashes. With sweep=0 (no
seeds, empty AOT table) divs=0.

Bisect by trace prefix length:
- 100 seeds: 0 divs
- 150 seeds: 0 divs
- 160 seeds: 0 divs
- 164 seeds: 0 divs
- **165 seeds: 1 div** ← first regression

Adding line 165 (`0802f6bc thumb`) to the trace seeds introduces
the first divergence.

## What's at 0x0802f6bc?

```
0002f6bc: a7f9 72f0 cffa 01bc 0047 0000 ...
```

Halfword decoding:
- 0x0802f6bc = 0xf9a7 → F19 lo (BL pair second half) — top5 = 0b11111
- 0x0802f6be = 0xf072 → F19 hi
- 0x0802f6c0 = 0xfacf → F19 lo (paired with hi above)
- 0x0802f6c2 = 0xbc01 → F14 POP

So 0x0802f6bc is an **orphan F19 lo** — the second half of a BL
pair whose first half (the F19 hi) is at 0x0802f6ba (the
end of a previous block, OUTSIDE the trace seed entry).

## Why does scalar enter at an orphan F19 lo?

Scalar's block recorder records blocks starting at any PC where
`step_block` re-enters after a pipeline flush. If a previous
function call left an instruction stream like:
```
0x0802f6ba: F19 hi  (block N's last instr, advances pc to 0x0802f6bc)
0x0802f6bc: F19 lo  (block N+1's first instr — completes the BL pair)
```
…scalar would record `block N` ending at the hi (or earlier) and
start block N+1 at 0x0802f6bc, which begins with the F19 lo.

The lo handler (`exec_thumb_branch_long_with_link::<true>`) reads
`gpr[REG_LR]` (set by the preceding hi in block N) and computes
the BL target. So at runtime this is fine for scalar.

## What's the bug, then?

In principle AOT should also run the F19 lo handler with the
correct `gpr[REG_LR]` value (set by the preceding block's hi
instruction). Both scalar and AOT should produce identical state
changes.

Empirically AOT diverges by 1+ frames after this block is added
to the cache. Cause not isolated yet — possible factors:

1. **Cycle accounting**: the F19 lo handler does
   `reload_pipeline16()` which charges 2S + 1N. Both scalar and
   AOT path this through `cpu.bus.load_16` so cycles SHOULD match.
   But maybe AOT's per-iter fetch + reload_pipeline16's reload
   double-charges.

2. **Pipeline state**: the AOT block's first iter does its own
   fetch + pipeline shift before the handler. Then the handler
   does reload_pipeline16 → another shift + fetch. State after
   the handler might differ subtly from what scalar produces.

3. **next_fetch_access timing**: reload_pipeline16 sets
   next_fetch_access = Seq at the new pc. AOT might have left
   it set to whatever the previous block's last access was.

To isolate: build a single-block test case that reproduces the
divergence in isolation (DiffBus per A6, run scalar handler + AOT
trampoline path on identical state, assert byte-equality).

## Pragmatic conclusion

Phase 1's whole-block trampoline path has at least one trampoline
correctness bug specific to F19 BL pair scenarios. The bug is
triggered ANY time scalar started a block at an F19 lo orphan
PC (which is normal for game code and present in real ROMs).

**Phase 1 cannot fully accept while this bug stands.** Either:

a) Fix the F19 lo handling specifically — likely a cycle
   accounting issue between AOT's pre-handler fetch and the
   handler's reload_pipeline16.

b) Move to phase 1's "real" goal: per-format inline IR. If a
   per-format inline F19 emit replaces the LUT-handler call, it
   sidesteps the trampoline issue. But phase 1a's per-instr LLVM
   emit had its own scaling bug (separate issue).

c) Accept partial phase 1: ship Rust-level inline fast paths for
   formats that don't have this bug (F3 MOV imm8, F1 shift, F4
   ALU). Coverage stays low (sweep=0) but architecture is in
   place for phase 2+.

Recommending (c) for now. The trampoline bug is real but is a
multi-hour debugging task that doesn't block the architectural
forward motion.

## Update 2026-04-25: trace seed pc convention bug

Found a separate bug: the `--aot-trace-out` writes pcs that are
PIPELINE-HEAD pcs (= exec_addr + 4 in Thumb, +8 in ARM) because
that's what `block_cache.key` stores. But arm7tdmi-aot's
`scan_one_block(entry_pc)` interprets `entry_pc` as exec_addr
(the halfword address of the FIRST executed insn). So scan reads
opcodes 4 bytes ahead of where scalar actually starts the block,
producing off-by-4 AOT blocks that:

- Land at lookup_pc = trace_pc + 4 (one past where scalar
  dispatches), so most miss at runtime.
- Occasionally alias to a real cpu.pc value (block boundary
  shared with static-scan reachability), where AOT's first
  opcode-at-trace_pc may differ from scalar's first
  opcode-at-(trace_pc-4), causing divergence.

Fixed in main.rs trace-in by subtracting 4 (Thumb) / 8 (ARM)
before passing seeds to `compile_rom_with_seeds_and_step`.

Empirical (PE 24052-seed trace):
- Before fix: 74.48% coverage, 32 divs.
- After fix: 78.22% coverage, 31 divs.

Marginal divs improvement; main F19 lo bug remains.

The drift on full-trace went UP because more code now correctly
dispatches through AOT, exposing the latent trampoline cycle
accounting bug at higher rate. That's expected — fixing one bug
exposes another, since the prior coverage-via-aliasing was
masking the trampoline drift.

Phase 1 partial-accept stays as the conclusion. Coverage-at-scale
needs the trampoline cycle bug fixed OR phase 4 (LLVM IR inlines
fetch) sidesteps it entirely.
