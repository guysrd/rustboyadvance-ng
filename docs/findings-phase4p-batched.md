# Phase-4-prime continuation: make 19-fmt actually beat scalar

## Why reopen this

The prior postmortem (`findings-phase4-prime-postmortem.md`) declared
inline-IR a structural net regression vs whole-block. That conclusion
was correct *for the specific framing* "stack more formats" — but
conflated it with the broader claim "this 19-fmt code can't be made
faster." The 19-fmt path has untouched optimisation headroom that
attacks the diagnosed cause directly (i-cache pressure / per-opcode
extern-call boundary cost).

Current measured gap on quiet system (per controlled bench
`controlled-compare-WB-vs-INLINE` row of results.tsv):

    19-fmt PE 431 / MK 364
    scalar PE 503 / MK 368
    gap to close: PE +17%, MK +1%

## Hypothesis

Per-opcode `aot_thumb_fetch_only` extern call is the dominant cost
in the 19-fmt path. With a 32-instruction block, that's 32 indirect
calls → 32 register-save/restore round-trips → optimiser barrier
between each opcode (LLVM can't fold across the extern). Every
inlined format's IR ends with this same call. Removing it across
the board is the single biggest single change available.

Direct evidence from prior bisect:
- F3 step 2 (commit afb1ebd) baked cycles inline as
  `*sched_ts_ptr += K`. PE *appeared faster* (no measurable win, but
  divs broke from WAITCNT-stale baked constants — rejected on V1 not
  perf). The IR pattern was correct; the WAITCNT correctness wasn't.

## Plan ladder

Three sequential phases. Each is one commit, env-gated default-off,
verified by V1+V2+V4 gates before next phase begins.

### 4P-A: WAITCNT generation counter (infrastructure)

The blocker for ALL inline-cycle approaches is WAITCNT-stale risk.
PE's BIOS (or game init) writes WAITCNT, cycle_luts shift, baked
constants go wrong. Per I7 the proper fix is synchronous AOT recompile;
implementing that is a phase 9 problem.

**4P-A scope (this commit, data-only):**
1. SysBus gains `aot_gen_counter: u32` bumped on `on_waitcnt_written`.
2. SysBus exposes `aot_gen_counter_ptr() -> *const u32` and
   `aot_gen_counter() -> u32`.
3. CpuOffsets gains `aot_gen_counter_ptr: u64` and `aot_gen_baked: u32`.
4. SDL frontend captures both and passes to the AOT compiler.

No IR emission. Behaviourally a no-op until 4P-B uses it.

**Deferred to 4P-B (the prologue check itself):**
A naïve "on gen mismatch return 0b10" infinite-loops the dispatcher
because 0b10 yields to the outer run loop, which immediately re-enters
step_block, which AOT-dispatches the same pc, which gen-mismatches
again — and the scheduler never advances because no cycles get
charged. So the prologue check needs a poison path that actually
disables AOT for the affected block (or for the session). Options:

- **Per-block poison flag:** AOT lookup is gated on a `block_poisoned`
  bit; mismatched IR sets the bit + returns 0b10; next dispatcher
  tick falls through to scalar.
- **Session-wide AOT disable:** mismatched IR clears the AOT lookup
  fn pointer (sets to noop). Per I19 second-WAITCNT-write disable
  rule. Loses subsequent AOT for the rest of the session — fine
  given WAITCNT writes are once-per-session in practice.

4P-B picks one and lands it together with the per-opcode inline
cycle replacement. This file gets updated when that happens.

### 4P-B: per-opcode inline cycle add (replaces fetch_only's cycle role)

With gen-check in place, baked cycles are safe. Replace
`aot_thumb_fetch_only` extern call per opcode with:

    *sched_ts_ptr += baked_thumb_seq_cycles_for_page

(or `_nonseq` based on next_fetch_access at compile time — page is
known per opcode since opcodes are at known ROM addresses).

Pipeline state is NOT updated mid-block; instead one block-exit
helper restores pipeline[0/1] for the next block (single extern call
per block, not per opcode).

Mid-block opcodes that read pipeline (none of the 19 inline formats
do — they were chosen specifically because they don't): not affected.

The block-exit pipeline restore: known compile-time, since the AOT
scan knows the exit pc and the next two opcodes (or fetches them
real-time from ROM region — slice index, no bus dispatch).

Gated `AOT_BAKED_CYCLES=1` (requires AOT_GEN_CHECK=1). Verify divs,
bench.

### 4P-C: lazy flag materialization (if 4P-B doesn't cross scalar)

Per program-doc phase 6. State machine in compiler tracks dirty flag
state (N/Z/C/V) per builder block. cpsr write deferred until:
- Opcode reads cpsr (Bcc, ADC/SBC carry-in).
- Block exit.

Estimated +5-10% on flag-heavy code from program-doc.

## Out of scope

- Phase 7 (block chaining): orthogonal lever, deferred.
- Phase 8 (ARM-mode IR): MK win, but requires phase-4-prime to land
  first (mirroring infra).
- Phase 9 (persistent cache): UX, no fps.
- LLVM intrinsics (fshr/uadd.with.overflow): marginal, deferred.

## Acceptance gate per phase

V1 (0 divs both ROMs at sw=64KB), V2 (drift < 1000), V4 (determinism).
Plus per-phase fps gate:
- 4P-A: no fps regression vs current 19-fmt baseline (it's a no-op
  unless gen mismatches at runtime, which shouldn't happen during
  recorded replays).
- 4P-B: > whole-block on at least one ROM (target: PE > 480, MK > 372).
- 4P-C: > scalar on at least one ROM (target: PE > 503, MK > 368).

If 4P-B doesn't hit "> whole-block" then the hypothesis is wrong
and we revert; cron loop falls back to phase 8 ARM.

## Single-firing scope

This firing: 4P-A foundation only. Gen counter + bumping +
prologue check + env gate. No behaviour change; safety net for
the next firings.

Next firings: 4P-B per-format conversion (one or two formats per
firing).
