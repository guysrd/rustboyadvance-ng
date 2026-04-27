# findings-ladder.md — phase-1 permanent-fail re-plan

Per docs/aot-llvm-program.md "Phase-N permanent-fail re-planning":
phase 1's gate is `score > 0 on at least one ROM`, where score is
`(pe_fps_aot - pe_fps_scalar) + (mk_fps_aot - mk_fps_scalar)`.

After many attempts and ~3 weeks of firings, phase 1 has not met
this gate. This doc formalizes the re-plan.

## What phase 1 was supposed to deliver

> Inline top-3 most-executed Thumb formats from A7. Expected: +5-10%.
> Gate: divs+drift+determinism + score>0 on at least one ROM.

## What got built

The phase-1 architecture is in place and correct:

- Two-level PC->fn table (top 65536 x leaf 32768) keyed by
  pipeline-head pc.
- BIOS exception-vector seeded scan + cart-entry-pc + B/Bcc/BL
  static-reachability scan.
- Thumb whole-block trampoline (`aot_replay_thumb_block_for<I>`) and
  per-instr trampoline (`AOT_USE_PER_INSTR=1`).
- ARM whole-block trampoline (`aot_replay_arm_block_for<I>`,
  phase-8 scaffolding).
- 17 Thumb formats inlined as Rust fast paths in `aot_thumb_step`
  (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14,
  F15, F16-bcc, F19-hi) — all bit-exact with scalar handlers.
- Inter-block abort check parity with scalar (commit 82d4170 fixed
  the at-scale divs bug).
- F3 MOV imm8 inline LLVM IR (gated AOT_INLINE_F3=1) with inline
  cycle accumulation via `*sched_ts_ptr += K` (commit afb1ebd).
- Phase-4 infrastructure: `bus.scheduler_timestamp_ptr()` +
  `bus.thumb_fetch_cycles(page)` + `read_16_no_cycles` trait method
  + `CpuOffsets` carrying pc/gpr/cpsr/nfa/pipeline + per-page cycle
  costs.

Correctness: 0 fb-hash divs both ROMs at sweep=0..64KB; 1 transient
div at trace=24052 with -126 cycles drift (under the 1000-cycle
gate). Determinism: identical fb-hash sequences run-to-run.

## Measured fps from each attempt

5x5 noise characterization on commit a00492f (post-hygiene):

| Configuration            | PE fps | PE gap  | MK fps | MK gap  |
|--------------------------|--------|---------|--------|---------|
| scalar (cached_interp)   | 560.2  | (ref)   | 397.7  | (ref)   |
| AOT sweep=0 (1.4% MK cov)| 553.3  | -1.2%   | 388.1  | -2.4%   |
| AOT sweep=64KB whole-blk | 501.2  | -10.5%  | 395.5  | -0.6%   |

PE trace-in 24052 entries (77% coverage): 476.8 fps vs 542.1 scalar
= -12.1% (commit ab4b7ab).

Phase-1 score at sweep=64KB: `-59 + -2 = -61`. Both ROMs negative.

Per-instr (AOT_USE_PER_INSTR=1) is correct at sweep=4-64KB but is
always slower than whole-block (extern boundary per opcode).

## Hypothesis: why phase 1 doesn't deliver

The cached interpreter scalar is a tight inline Rust loop with
PGO'd direct handler calls — the LLVM optimizer (rustc backend)
sees the entire loop end-to-end and inlines aggressively.

The phase-1 trampoline architecture pays a cost the scalar doesn't:

1. **Per-block dispatch**: `try_aot_dispatch` does a mode check, a
   fn-pointer indirect call into `aot_lookup_for_hook`, the table
   walk, and an unsafe transmute + extern "C" call into the JIT'd
   block fn. That's ~3 indirect-call layers vs scalar's direct
   inline `replay_cached_block` call.

2. **Per-iter trampoline cost**: The whole-block JIT'd code calls
   `rba_aot_replay_thumb_<id>` which calls `aot_replay_thumb_block_for`
   which loops calling `cpu.aot_thumb_step`. That `aot_thumb_step`
   call is a non-inlined extern boundary every iteration. LLVM
   can't see through it.

3. **`aot_thumb_step` overhead**: Even though 17 formats are inlined
   as Rust fast paths, each call still does a `bus.load_16` (real
   bus call with page-region match), pipeline shift, pc update,
   and a 17-way bit-mask format detection chain. Scalar's inline
   loop pays the same per-iter work but the optimizer can fold
   redundancies across iters; behind the trampoline boundary it
   can't.

Empirical decomposition (5x5 + per-dispatch math): ~11ns of overhead
per AOT dispatch hit. At 181M hits per PE replay, that's ~2s on a
27s replay = ~7.5% (and we're seeing ~10% in the latest run).

The F3 MOV imm8 inline IR experiment (phase 4 step 1+2) confirmed:
inline IR ops cost roughly what the saved extern boundary did.
A per-format inline IR pass that beats scalar requires inlining
**all** hot formats simultaneously — half-measures don't work
because any opcode that falls through to the step trampoline pays
the full extern boundary, dragging down the average.

## Re-plan options

### Option A: drop phase 1, move to "phase 4 proper" as the only path

Phase 4 in the original ladder was "inline fetch + abort check"
(+20-30%). Reframe it as:

**Phase 4-prime: full inline IR per Thumb format.** Emit LLVM IR
that does all hot Thumb formats inline (no trampoline call). Each
format fragment includes:

- Inline cycle accounting (`*sched_ts_ptr += K`, K baked per page).
- Inline pipeline maintenance (constant stores from baked opcodes).
- Inline gpr/cpsr/pc updates per format semantics.
- IO-region region-dispatch ladder for memory ops (per I3).
- IR-level direct branches for in-block Bcc.

Estimated scope: 12-15 formats x ~50-100 lines of IR Rust each
= ~600-1200 lines. Not parallelizable across cron firings;
needs sustained multi-day work.

Acceptance: PE_fps_aot >= scalar at sweep=64KB. Anything less means
the architecture still loses; pivot to Option B.

### Option B: pivot to direct Rust fn-ptr dispatch, drop LLVM JIT

Replace the LLVM JIT entirely with direct Rust fn pointers in the
PC->fn table. Each "compiled" block is just a Rust trampoline fn
specialized at scan time (e.g., per-format pre-dispatched table).
No IR codegen, no extern boundary into LLVM-emitted code, no
per-block JIT compile time.

Pros:
- Zero compile time at ROM load (currently 70-76s at sw=64KB).
- Direct Rust fn-ptr dispatch can be inlined by rustc/LTO.
- No LLVM extern boundary.

Cons:
- Loses the optimizer's view of "the whole block as IR".
- Phase 4-prime is impossible from this architecture.
- Effectively reduces to "fancier cached interp" — gain over
  cached_interp may be small.

### Option C: ship phase 1 as-is, document the ceiling

Mark phase 1 as "shippable but not faster than scalar." Disable
`--aot` by default in SDL frontend (or ship behind a build feature).
Pause the autoresearch program until either someone commits to
multi-day work or the phase-1 architecture gets a fundamental
re-think.

Acceptance: AOT off by default; AOT-on path still passes V1+V2+V4.
gba-tests pass under `--features cached_interp,aot` (per I26).

## Recommendation

**Option A (phase 4-prime) is the only path that meets the program
doc's success target** (`pe_fps_aot > pe_fps_scalar * 1.05 AND
mk_fps_aot > mk_fps_scalar * 1.05`). Options B and C either give up
on the program goal or commit to a downgrade.

But Option A requires multi-day sustained work. The 15-min cron
cadence has produced ~10+ "small experiment, hit noise floor"
firings. The next operator should either:

1. **Schedule a sustained block** (say, 8 hours of active work
   without context switches) to do the IR-emit grind across
   12-15 formats in one push.

2. **Pause the cron loop** and resume only when ready to commit
   to (1).

Continuing the current cadence is sunk cost; each firing makes
the same observation and reverts.

## What stays in the tree

Regardless of which option lands: keep the phase-1 infrastructure
(scan, table, dispatcher hook, BIOS scan, ARM scaffolding, F3
inline IR, CpuOffsets, fetch_only extern). It's all reusable for
phase 4-prime. Only `--aot` default should change (off vs on).

## Sign-off

Phase 1's correctness deliverables ARE accepted. The fps deliverable
is permanent-fail per program-doc gate. Moving program state from
"phase 1 in progress" to "phase 1 fps-rejected; awaiting phase 4-prime
commitment".
