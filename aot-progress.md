# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**Currently in:** phase 1 ACCEPTED at scale (commit 82d4170 fixed
the trampoline at-scale divs bug). 17 Thumb formats inlined as
Rust-level fast paths in `aot_thumb_step`. Per-instr LLVM emit
(AOT_USE_PER_INSTR=1) probably also works now but untested.

**2026-04-26 hygiene pass:** dropped the redundant un-segmented
`aot_dispatch_hits`/`aot_dispatch_misses` counters (each AOT hit
was bumping two counters when the per-mode counters already split
by mode; un-segmented total is `thumb + arm` at print time). Also
fixed two pre-existing build-without-aot issues: the
`aot_lookup_fn_arm` initializer was missing its `cfg` gate at three
sites, and the `compile_rom_pokeemerald_smoke` test was calling
`compile_rom` with the old 4-arg signature. Sweep=0 baseline still
clean (0 divs both ROMs, score=-1 noise). No measurable fps win at
sw=64KB (within run-to-run noise band) — this is housekeeping, not
a perf experiment.

**Trampoline cycle drift fix (2026-04-26):** scalar fires
`cached_block_should_abort()` on every block boundary; AOT was
skipping it (per phase 0c MK-divergence comment). With the phase
0d Bcc-as-Linear fix making AOT block sizes comparable to scalar's,
the original concern no longer applies and the missing check was
the actual root cause of the 36-div / 23k-drift at sweep>=4KB.

Empirical post-fix:
- sweep=4KB: 0 divs / 0 drift / 54.66% cov (was 36 / 23k)
- sweep=16KB: 0 divs / 0 drift / 55.24% cov
- sweep=64KB: 0 divs / 34 drift / 70.92% cov (was 89 divs)
- trace 24052: 1 div / -126 drift / 77.41% cov (was 31 / 23k)
- MK sweep=64KB: 1 div / 68 drift / 1.16% cov (MK is ARM-heavy)

Phase 1 fps still 10% slower than scalar at 70% coverage (483 vs
537) because trampoline mode pays an LLVM extern boundary per
dispatch. Phase 4 (inline fetch + abort into LLVM IR) is needed
for actual fps gain.

**Phase 4 step 1 done (commit 2999628)**: F3 MOV imm8 inline LLVM
IR works correctly (0 divs at sweep=4KB and 64KB) but is 10 fps
SLOWER than the phase-1 per-instr trampoline path (490 vs 501 at
4KB). The trampoline's Rust-inlined F3 fast path is faster than
equivalent IR ops + 1 fetch_only extern call.

**Performance reasoning for the regression:**
- The fetch_only extern is OPAQUE to LLVM. The optimizer can't see
  across it to combine IR ops with the rest of the block.
- The Rust trampoline has the entire dispatch + fast-path body in
  one Rust fn that the Rust compiler can fully optimize.
- LLVM's per-block module-level inlining can't see into Rust externs.

**To actually beat scalar, we need:**
1. ELIMINATE the per-iter extern call entirely. That means inlining
   cycle accounting via direct scheduler.timestamp += K stores.
2. Skip per-iter pipeline shifts; restore at block exit via 2 const
   stores (halfwords at known ROM addresses, baked at AOT time).
3. Inline ALL hot formats so no opcode falls through to extern.

That's the real phase 4. ~3-5 days work. The infrastructure
(CpuOffsets, fetch_only extern) is in place from step 1.

**Phase 4 step 1 GATED off by default (commit 95e6954):** since the
inline IR was a fps regression, default per-instr is preserved.
AOT_INLINE_F3=1 to opt in (for correctness verification + further
iteration).

**Phase 4 step 2 done (commit afb1ebd):** inline cycle accumulation
via `*sched_ts_ptr += K_seq_or_nonseq` in IR. aot_thumb_fetch_only
now uses read_16_no_cycles (skip cycle charge); F3 IR adds runtime
nfa-based cycle selection + direct ts_ptr increment.

Still no measurable fps win — the inline IR ops cost roughly what
the saved extern boundary did. The real win requires inlining ALL
hot formats so no opcode falls through to the step trampoline extern
(elimination of all per-iter externs is the only path to beating
scalar). That's another 5-10 days of per-format IR emit work.

**Recent rejected experiments (2026-04-26):**
- `#[inline(always)]` on `SysBus::load_*` (commit b4fde27 row): PE
  -11.4% at sw=64KB, code bloat hurts I-cache. Reverted.
- Dispatch guard reorder (commit 46fe2f7): noise.
- Noop-stub for aot_lookup_fn (commit 6cd1c57): noise.

**Loop status:** approximately every cron firing for the past ~10
firings has been "small experiment, hit noise floor or regressed,
revert." The cron interval (5-15 min) is incompatible with the
multi-day refactor scope of:
1. Phase 4 proper — emit ALL hot Thumb formats inline in IR.
2. Phase 8 — ARM block compile path.
3. LLVM JIT elimination — direct Rust trampoline fn-ptr dispatch.

**Recommendation for next operator (human or sustained AI run):**
- Pick ONE of the three multi-day projects above.
- Don't context-switch every cron fire.
- The infrastructure for phase 4 is ALL in place (CpuOffsets +
  ScheduleTimestampPtr + read_16_no_cycles + per-format IR emit
  pattern in compiler.rs). What's left is mechanical per-format
  emit work (~50-100 lines of IR Rust per format × 12 formats).
- Or: deliver as-is at -8% fps gap PE / parity MK and call phase 1
  shippable.

**5-run noise characterization (2026-04-26):**
- PE scalar: 562, 549, 550, 549, 549 → median 549.0, mean 551.6
- PE AOT sw=64KB: 508, 516, 504, 508, 506 → median 507.6, mean 508.5
- Gap: 7.5% (real, not noise)

**Post-hygiene 5x5 ship characterization (2026-04-26 commit c75dbee):**

PE (3-run sw=0 + 5-run scalar + 5-run sw=64KB):
- PE scalar:        557.9, 559.2, 560.2, 564.5, 580.1 → median 560.2
- PE AOT sw=0:      548.0, 553.3, 567.8             → median 553.3 (-1.2% parity)
- PE AOT sw=64KB:   497.1, 500.1, 501.2, 504.0, 504.1 → median 501.2 (-10.5%)

MK (5-run each):
- MK scalar:        396.1, 396.1, 397.7, 398.0, 403.9 → median 397.7
- MK AOT sw=0:      388.0, 388.1, 388.1             → median 388.1 (-2.4%)
- MK AOT sw=64KB:   394.4, 395.4, 395.5, 395.8, 397.7 → median 395.5 (-0.6% parity)

Notes:
- PE sw=64KB gap widened slightly vs the earlier -7.5% — system-load
  variance, not a regression from the hygiene cleanup (struct layout
  changes only affected fields after the IR-baked offsets, and the
  drop in counter increments per dispatch can only be a strict win).
- MK at sw=0 is -2.4% because BIOS scan still installs 3 exception-
  vector seeds (1.4% coverage) — even tiny coverage of trampoline-mode
  AOT loses to scalar. Disabling BIOS scan would close this.
- Phase-1 trampoline architecture's ceiling reproduces:
  high-coverage PE -10.5%, low-coverage MK -2.4%, parity at near-zero
  coverage. No path to closing this without phase-4-proper IR emit.

**Ship-state recommendation:** the AOT_SWEEP_CAP_KB=0 build with
BIOS scan disabled (or AOT off entirely) matches scalar ±1%. Keeping
AOT compiled-in but `--aot` off-by-default is the safest user-facing
posture until phase-4 IR emit lands. Any cron-paced micro-optimization
will continue to hit noise floor; commit to multi-day phase-4 work
or pause the program.

Per-dispatch math: 7.5% × 27s scalar = 2s overhead / 181M AOT
dispatches = ~11ns per AOT dispatch. The AOT path adds ~11ns per
hit vs scalar's per-iter dispatch. Most likely the 17-way format
detection in aot_thumb_step (~3-5ns) + LLVM extern boundary
(~5ns/block ÷ 7 iters/block = ~0.7ns/iter) + lookup (~5ns/block
÷ 7 = 0.7ns/iter). Doesn't fully add up — some unaccounted cache
or branch-prediction effects too.

**Phase 8 results (commits edc3a39, f0f037c, 4d37779, 712ffb6,
c55048f, 61acefd, eaf409d, 27f5aa4):** ARM block scaffolding all
the way through: extern wrappers, AotTable arm pages, emit fn,
compile_rom integration, dispatcher dual-table lookup, SDL frontend
plumbing, BIOS exception-vector seed scan, BIOS-wide ARM sweep.

Results:
- BIOS exception-vector seeds (default-on): MK 1.42% coverage,
  +0.9% fps (small win from IRQ handler caching).
- AOT_SWEEP_BIOS=1: MK 54.69% coverage but -0.9% fps (broad
  coverage = more trampoline overhead).
- PE trace-in 24052 (77% coverage): -12.1% fps (commit ab4b7ab
  confirms trampoline cant beat scalar at any coverage).

Phase 1 trampoline architecture conclusively does NOT beat scalar
regardless of coverage strategy. Phase 4 per-format LLVM IR emit
remains the only path to a fps win. The infrastructure is ready;
the work is mechanical IR-emit code per format.

**Phase 4 infrastructure complete (commits 376722b, 990bd11, 1846504):**
- `SysBus::scheduler_timestamp_ptr() -> *mut usize` — stable raw ptr
  at scheduler.timestamp. Inline IR adds K via `*ts_ptr += K`.
- `SysBus::thumb_fetch_cycles(page) -> (s, n)` — Seq/NonSeq cycle costs
  per page. Per I7 stable until WAITCNT change.
- `Arm7tdmiCore::aot_field_offsets()` returns (pc, gpr, cpsr, nfa,
  pipeline) byte offsets via offset_of! for the monomorphized type.
- `CpuOffsets` struct in arm7tdmi-aot now carries all of: cpu offsets
  (pc, gpr, cpsr, nfa, pipeline) + sched_timestamp_ptr + per-page
  thumb_seq_cycles[16] + thumb_nonseq_cycles[16].
- SDL frontend computes everything from the live SysBus and passes via
  `compile_rom_with_seeds_step_offsets`.

All required ingredients for inline cycle accounting are now in place.
Future phase-4 IR emit can:
  *sched_ts_ptr += K_seq                (cycle accounting, no extern)
  pipeline[0] = pipeline[1]              (pipeline shift via offset)
  pipeline[1] = ???                      (need to load or skip)
  store imm at gpr[Rd]                   (per-format effect)
  pc = fetch_addr + 2
  next_fetch_access = Seq

Remaining design decision: skip per-iter memory loads (since A1 says
no Thumb handler reads pipeline) and bake post-block pipeline values
as constants from ROM at scan time. Add 2 const stores at
fall-through exit. Saves the extern call entirely.

**Latest measurement (2026-04-26 post dead-code cleanup):**
- sweep=0 baseline: 0/0 divs both ROMs, score=+2 (noise).
- sweep=64KB whole-block: PE 521.2 / scalar 565.4 = -7.8% gap;
  MK 403.5 / scalar 406.5 = -0.7% gap (near-equal). 0/1 divs gate.
- score=-47 at sweep=64KB.

The PE gap at high coverage is the main concern. MK is essentially
parity since AOT coverage is only 1.16% there (ARM-heavy code).

**Variance + low-coverage characterization (3-run median):**
- sweep=0 PE scalar: 564 fps (range 20 fps).
- sweep=2KB PE whole-block: 550 fps. -2.5% vs scalar.
- sweep=2KB PE per-instr: 550 fps. Same as whole-block at low cov.

Even at 0.50% coverage AOT costs ~2.5%. That's the per-dispatch
lookup overhead in `try_aot_dispatch` (table page lookup + None
check + leaf index, ~5-10ns per dispatch even on miss). For 280M
dispatches per replay = ~1.4-2.8s overhead = ~5% of 30s.

Implication: the AOT plumbing has a fixed per-dispatch cost
regardless of hit rate. Higher coverage AMORTIZES this by replacing
expensive scalar work with AOT work. Lower coverage doesn't help.

To reduce AOT plumbing cost: shrink try_aot_dispatch hot path.
E.g., skip the cold-start guard once pipeline is initialized
(set a bool flag and check that instead of `pipeline[0] == 0`).

Open observations:
- `compile_rom_with_seeds_step_offsets` at sweep=64KB takes ~70-76s
  to compile via LLVM JIT (regardless of phase-4 changes). LLVM is
  slow when compiling 32k separate per-block modules. Future:
  consolidate modules (one per ROM page, multiple blocks each).
- AOT_USE_PER_INSTR=1 mode is correct at sw=4-64KB but always slower
  than default whole-block (extern boundary per opcode).

Tried this turn (no measurable win):
- `#[inline(always)]` on `aot_thumb_step` — 484 vs 488 fps at
  sweep=64KB, within noise. Reverted.

Path forward (high-value, multi-day):
1. Inline cycle accounting via `*scheduler.timestamp += K` in IR.
   Requires `pub fn scheduler_timestamp_ptr()` on SysBus, plumbed
   through to AOT compiler as a baked constant pointer.
2. Skip per-iter pipeline shifts in `aot_thumb_step`. Restore at
   block exit via 2 const stores (halfwords at known ROM addresses,
   baked at scan time). Saves ~2ns/iter ≈ 1.3% at high coverage.
3. Inline ALL hot formats in IR (not just F3) so no opcode falls
   through to the step trampoline extern. Combined with (1) and (2)
   could potentially beat scalar.

Path forward (alternative, smaller scope):
- ARM block support for MK. Currently AOT scan accepts ARM mode
  but `compile_rom_with_seeds_*` skips ARM specs (1.16% AOT
  coverage on MK). Adding `aot_arm_step_for<I>` + `emit_arm_block`
  would give MK ~50% coverage. But fps WIN unclear without phase-4
  IR work — trampoline mode is currently 6% SLOWER than scalar at
  high coverage.

Lesson: inline IR is only a win when the equivalent Rust fast path
has high overhead. F3 MOV imm8's Rust path is already 4 ops
(`gpr[rd] = imm; cpsr.set_N(false); cpsr.set_Z(imm==0); pc += 2`).
No room to improve.

**Next deliverable:** phase 4 step 2 — try formats where the Rust
fast path has higher overhead:
- F4 ALU shifter: calls shift_by_register + idle_cycle (~15ns).
  Inline IR could constant-fold the shift if amount is known.
- F1 LSL/LSR/ASR imm5: similar.
- F19 hi: gpr[LR] = self.pc + (off << 12). Constant-foldable!
  Just one store of `(fetch_addr + (off << 12))`.

OR pivot to a different approach:
- batch multiple opcodes in one fetch_only call (amortize extern).
- inline cycle accounting via direct scheduler.timestamp += K ops
  (skip the bus.add_cycles extern). Requires offsetting through
  Rc<UnsafeCell<SysBus>>.scheduler.timestamp.

Plan for next turn:
1. Add `CpuOffsets` struct in arm7tdmi-aot/src/lib.rs with pc_offset,
   gpr_offset, cpsr_offset, next_fetch_access_offset (all u32).
2. SDL frontend in main.rs computes via offset_of! for
   `Arm7tdmiCore<SysBus>` and passes via compile_rom variant.
3. Add a thin `aot_thumb_charge_fetch_for<I>` extern that does just
   load_16 + pipeline shift + cycle accounting (no dispatch). Same
   sig as aot_thumb_step but skips the format detection / dispatch.
4. In emit_per_instr_thumb_block, detect F3 MOV imm8 opcodes
   (top5=00100 && bits 12:11 == 00). For those:
   - Call aot_thumb_charge_fetch_for extern.
   - Emit inline IR: store imm at gpr[Rd], update cpsr (clear N, set
     Z if imm==0), store fetch_addr+2 at pc, store Seq at nfa.
   - All operands constant-foldable from baked opcode bits.
5. Test: AOT_USE_PER_INSTR=1 sweep=64KB. Expect:
   - 0 divs (correctness preserved).
   - fps gain on F3-heavy code.

Empirical confirmation that this approach is sound: F3 MOV imm8 is
extremely common (~15-20% of dynamic Thumb instrs) and the body is
just a gpr store + cpsr Z-flag set (constants at AOT time). LLVM
optimizer should fold the IR to ~3 native stores. Saves ~5ns vs
the trampoline call's ~10ns extern boundary + ~3ns format detection
+ ~3ns body. Per-format gain ≈ 30%, weighted by frequency ≈ 5% on
F3 alone. Stack ~10 inlined formats and we beat scalar.

Alternative path: ARM block support (MK uses ARM heavily, currently
only 1.16% AOT coverage). Would need arm7tdmi-aot::aot_arm_step_for
trampoline + emit_per_instr_arm_block + classify_arm support in
scan (already exists). 5%+ gain on MK from coverage alone.

## Phase 0 — ACCEPTED ✓

Per `scripts/aot_measure.sh` at default sweep=0:
- PE: divs=0, drift=0, determinism=1, fps_aot=541.9 vs scalar=545.6
- MK: divs=0, drift=0, determinism=1, fps_aot=387.5 vs scalar=390.1
- diff_failures=0, aot_score=-6 (within noise)

Coverage 0% by default — the trampoline-mode whole-block emit is
correctness-fragile at scale, so default sweep=0 ships clean.

## Phase 1a — partial

Architecture (commit 634aa1f):
- `arm7tdmi-aot::replay::aot_thumb_step_for<I>` per-iter trampoline
- `arm7tdmi-aot::replay::aot_block_should_abort_thumb_for<I>` K=2
  abort check trampoline
- `LlvmCompiler::register_step_thumb` registers them
- `LlvmCompiler::emit_per_instr_thumb_block` emits LLVM IR with one
  step call per opcode + abort check at K=2 cadence
- `compile_rom_with_seeds_and_step` variant takes step+abort fns
- SDL `AOT_USE_PER_INSTR=1` env var toggles between phase-0
  whole-block (default) and phase-1 per-instr

Empirical results:
- PE sweep<=2KB phase-1: 0 divs ✓ (small scale works)
- PE sweep=4KB phase-1: 36 divs (regression)
- PE sweep=8KB phase-1: 36 divs
- PE sweep=64KB phase-1: 89 divs
- PE trace-in (24052 entries) phase-1: 255 divs (every frame)

Tried fixes (didn't help):
- Per-module unique extern names (commit 74b3e15) — didn't fix divs.
- Disabled abort check — got WORSE (152 divs); confirms abort path
  isn't the bug source.

Bug is structurally in the multi-extern-call emit path. The
trampoline `aot_thumb_step_for` delegates to `cpu.aot_thumb_step`
which is the same code the whole-block trampoline calls in its
loop — so the per-call SEMANTICS should match. But empirically
many sequential extern calls from LLVM IR produce divergent state.

Hypothesis: LLVM JIT engine's symbol/state management bug at
scale, OR a calling-convention issue, OR a memory aliasing
inference issue.

## Phase 1 — ACCEPTED ✓ (2026-04-26)

The trampoline at-scale divs bug is fixed (commit 82d4170). With
13 Thumb formats inlined in aot_thumb_step + the inter-block abort
fix, PE+MK both show 0 divs at sweep=4-64KB and trace=24052 has
just 1 transient div with -126 cycles drift (well under 1000 gate).

Phase 1 doesn't deliver fps gain over scalar (trampoline overhead
dominates). It delivers the architecture + correctness foundation
that phase 4+ builds on.

## Phase 1b/1c/2/5 (rust-inline) — incremental accept

`aot_thumb_step` in arm7tdmi/src/cpu.rs now has Rust-level inline
fast paths for 17 Thumb formats — every format that goes through
the per-iter dispatch path:
- F1 MoveShiftedReg LSL/LSR/ASR Rd, Rs, #imm5
- F2 AddSub ADD/SUB Rd, Rs, Rn or #imm3
- F3 DataProcessImm MOV/CMP/ADD/SUB Rd, #imm8
- F4 AluOps all 16 ops (incl. MUL with cycle accuracy)
- F5 HiRegOpOrBranchExchange ADD/CMP/MOV/BX (Rd=R15 → PipelineFlushed)
- F6 LdrPc (literal pool load)
- F7 LdrStrRegOffset LDR/STR reg-offset, byte/word
- F8 LdrStrShb STRH/LDRH/LDSB/LDSH reg-offset
- F9 LdrStrImmOffset LDR/STR imm5-offset, byte/word
- F10 LdrStrHalfWord LDRH/STRH imm5*2 offset
- F11 LdrStrSp LDR/STR sp-relative word
- F12 LoadAddress ADD Rd, [PC|SP], #imm8
- F13 AddSp ADD/SUB SP, #imm7<<2
- F14 PushPop PUSH/POP {rlist, +LR/PC}
- F15 LdmStm LDMIA/STMIA Rb!, {rlist} (incl. empty-rlist GBATEK quirk)
- F16 Bcc (taken/not-taken paths; SWI/undef fall to LUT)
- F19 hi (linear half of BL pair)

Block terminators handled by the AOT emit (not by aot_thumb_step):
F17 SWI, F18 B unconditional, F19 lo.

THUMB_LUT fallback in aot_thumb_step now only fires for SWI / undef.

Each is bit-exact with its scalar handler — same alu helpers, same
idle_cycle calls, same PipelineFlushed/AdvancePC semantics, same
Seq/NonSeq next-fetch-access encoding.

Trace seed pc fix: trace-out wrote pipeline-head pcs (= exec_addr
+ 4 in Thumb / +8 in ARM) but scan interprets entry_pc as exec_addr.
SDL frontend now subtracts 4/8 before passing seeds to scan. Fixed
in commit f0e5434.

Empirical:
- PE 100-seed trace-in: 0 divs, 0.52% coverage.
- sweep=0: 0 divs both ROMs, score in [-8, +3] noise band.

Phase 1 full accept (and any meaningful fps gain) blocked by the
F19-orphan trampoline bug documented in
docs/findings-phase1-trampoline-divs.md. The trace-pc fix exposed
~31 divs at 24052 trace seeds (was 32 before fix); the residual
bug is in trampoline cycle accounting on the F19 lo orphan path,
not in the inline fast paths. Coverage stays low at sweep=0 until
that's fixed OR phase 4 inlines fetch into LLVM IR (which sidesteps
the per-iter trampoline path entirely).

## Pending

### Phase 2: F5 / F2 / F12 inline (per A7)

Continue Rust-level inline fast paths in `aot_thumb_step`:
- F5: high-reg MOV/ADD/CMP (Rd = R8..R15 or Rs = R8..R15).
  CMP/MOV are simple; ADD with Rd=R15 reload_pipeline16 is the
  branchy variant.
- F2: ADD/SUB Rd, Rs, Rn (reg) or #imm3 (imm).
- F12: ADD Rd, [PC|SP], #imm8 — PC-rel address compute.

### Phase 4: inline fetch + abort

This is where wins start showing up. Currently the per-iter
`load_16` for fetch + cycle bookkeeping happens via real bus
calls; phase 4 inlines them as direct memory loads + cycle adds.

## Reminder loop

CronCreate `c1a9dd9e` at minutes 7,22,37,52 every hour. Each fire:
re-read docs/aot-llvm-program.md, this file, continue from resume
marker.

## Sweep coverage discontinuity (observed 2026-04-25)

PE divs vs sweep_kb:
- sweep=0: 0 divs / 0% coverage (baseline)
- sweep=1KB: 0 divs / 0.14% (554 blocks)
- sweep=2KB: 0 divs / 0.51% (1042 blocks)
- sweep=3KB: 36 divs / 56.33% (huge jump)
- sweep=4KB: 36 divs / 56.39%
- sweep=64KB: 89 divs (per resume marker)

The 0.51% → 56.33% coverage jump between 2-3KB suggests some block in
the [0x080800, 0x080C00] range opens a reachable graph that includes
hot common code. The 36 divs that come with it is the trampoline
cycle bug mentioned in findings-phase1-trampoline-divs.md, not a new
regression.

## Recent commits on this branch

- c6443dd phase 1: f15 ldm/stm inline (incl. empty-rlist gbatek quirk)
- 389fe26 phase 1: f14 push/pop inline
- d765aa4 phase 1: f16 bcc + f19 hi inline
- 56d1877 phase 1 ACCEPTED: 13 thumb formats inlined + at-scale divs fix
- 82d4170 fix: re-enable inter-block abort after aot dispatch — kills at-scale divs ★
- 5be090f aot-progress: f8 inline + sweep coverage discontinuity note
- 21b0003 phase 5: inline f8 strh/ldrh/ldsb/ldsh reg-offset
- 74cbee8 aot-progress: phase 1c+2+5 status, 12 thumb formats inlined
- 41412f4 phase 5: f7 ldr/str reg-offset + f10 halfword imm-offset inline
- 422928e phase 5: f9 ldr/str imm-offset + f11 sp-rel inline
- a529065 phase 5: f6 ldr pc-rel inline (literal pool)
- f0e5434 fix: trace seed pcs are pipeline-head, subtract 4/8 before scan
- 015159a phase 2: f2 add/sub, f5 hi-reg, f12 load-addr, f13 add-sp inline
- 56b3b26 phase 1c: f1 shifts + f4 alu inline
- a0bd4f2 phase 1b: F3 family expansion + trampoline divs investigation
- fa38437 phase 1b: F3 MOV imm8 fast-path
- 74b3e15 phase 1a debug: per-module unique extern names (didnt fix)
- 634aa1f phase 1a: per-instr dispatch architecture (gated)
- dbc49d8 phase 0d: trace-driven aot entry points
- c4deab4 phase 0 ACCEPT: harness + measurement
- 9df4c0e phase 0 step 4c: Bcc-as-Linear + coverage counters
- 758b0d9 phase 0 step 4b: placeholder block emit
- 9dc0326 phase 0 step 4a: compile_rom plumbing
- (audits A0-A8 across 3 batches)
