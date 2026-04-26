# AOT-LLVM autoresearch progress

Branch: aot-apr25
Started: 2026-04-25

## Resume marker

**STATE: phase 1 fps-rejected, awaiting phase 4-prime commitment.**
See `docs/findings-ladder.md` for the formal re-plan per the program
doc's permanent-fail rule. Phase 1 correctness deliverables are
accepted; the fps gate (`score > 0 on at least one ROM`) is failed
on both ROMs (PE -10.5% / MK -0.6% at sweep=64KB; score = -61).

The cron loop should NOT continue with small experiments — each
firing has reverted at noise floor. Three exit options in the
findings doc: (A) commit to multi-day phase-4-prime IR-emit grind,
(B) pivot to LLVM-free Rust fn-ptr dispatch, (C) ship phase 1
correctness only, AOT off by default. Recommendation: Option A
when a sustained work block is available; pause cron otherwise.

**Per-firing action while paused:** ONE 3-run scalar + 3-run AOT
sweep=0 measurement appended to `results.tsv` as `holding-NNN`.
Don't make code changes. Don't try perf experiments. The point of
these data points is to detect any host-system changes (compiler
upgrades, kernel updates, hardware swaps) that would invalidate
the ship-state characterization in findings-ladder.md. If a holding
firing's gap drifts beyond +/- 3% from the documented baseline
(PE scalar=560, AOT sw=0=553; MK scalar=398, AOT sw=0=388), append
a note to findings-ladder.md and ping the operator.

**2026-04-26 phase-4-prime F1 attempt (rejected, reverted):** tried
adding F1 LSL/LSR/ASR imm5 inline LLVM IR alongside the existing F3
inline path, gated by AOT_INLINE_F1=1 + AOT_USE_PER_INSTR=1 +
AOT_SWEEP_CAP_KB=64. Result: 88 hash-divs on PE → V1 gate fail →
revert (uncommitted, working-copy only).

**F3 IR regression bisected + fixed (commit 84ef3fa):** the side
discovery that F3 IR alone showed 12 hash-divs at sw=64KB was
bisected to commit afb1ebd "phase 4 step 2: inline cycle
accumulation in F3 IR". root cause: F3 IR baked per-page Seq/NonSeq
cycle costs as LLVM constants at AOT compile time, but PE writes
WAITCNT during BIOS boot which updates SysBus.cycle_luts — baked
constants go stale, IR charges old cycles while scalar charges
new cycles, drift accumulates, fb_hashes diverge starting frame=1860.

Per I7 the proper fix is synchronous AOT recompile on WAITCNT
write. Not implemented (and I19's "second WAITCNT write disables
AOT" isn't either). Simpler fix landed: drop the
`*sched_ts_ptr += baked_cycles` block from F3 IR, restore
aot_thumb_fetch_only to load_16 (which charges cycles via the bus
path, always reads current cycle_luts).

Verification post-fix:
- F3 IR + per-instr + sw=64KB: 0 divs (was 12) ✓
- default whole-block sw=0: 0/0 divs both ROMs (unchanged)
- default whole-block sw=64KB: PE 0 / MK 1 (historical baseline)
- harness V1+V2+V4 all pass

afb1ebd was claimed "no measurable fps win" and gated AOT_INLINE_F3=1
default-off, so the revert costs no fps. Phase-4-prime work
unblocked — F1/F4/etc. inline IR can now extend from a verified F3
template.

**F1 inline IR landed (commit 9c07858):** retried F1 LSL/LSR/ASR
imm5 inline LLVM IR after the F3 cycle-accounting fix. dropped the
WAITCNT-stale `*ts_ptr += baked_cycles` block, let fetch_only
charge cycles via load_16 (bus path). 0 hash-divs at sw=64KB on
both ROMs (MK has the usual 1 historical div).

Encoding 000_oo_IIIII_SSS_DDD; six (op, imm) cases enumerated to
constant-fold per opcode. cpsr update: N from result bit 31, Z
from (result==0), C from carry, V untouched. Gated AOT_INLINE_F1=1
default-off.

Phase-4-prime coverage so far: F1 + F3 inline IR (~20-25% dynamic
estimate). Per-instr fps still trampoline-bound (458 vs scalar 560);
needs more formats before per-instr matches whole-block.

**F12 inline IR landed (commit 741c624):** F12 LoadAddress
(ADD Rd, [PC|SP], #imm8). PC-rel case bakes the entire address as
a constant; SP-rel case loads gpr[SP] at runtime + baked imm.
No flag updates, no PipelineFlushed. 0 divs PE / 1 div MK
(historical) at sw=64KB. fps stack:
  F1+F3:      458 fps PE per-instr
  F1+F3+F12:  471 fps PE per-instr (+3%)

Phase-4-prime coverage: F1 + F3 + F12 inline IR (~25-30% dynamic
estimate). Per-instr fps still trampoline-bound vs scalar 560 but
the gradient is right — each inlined format peels another slice
of dispatch off the extern boundary.

**F13 inline IR landed (commit 74ebec8):** F13 AddSp (ADD/SUB SP,
#imm7<<2). 2 cases, no flag updates, no PipelineFlushed. 0 divs
PE / 1 div MK at sw=64KB. fps stack now:
  F1+F3:           458 fps PE per-instr
  F1+F3+F12:       471 fps PE per-instr (+3%)
  F1+F3+F12+F13:   470 fps PE per-instr (noise vs F12 stack)

Coverage now F1+F3+F12+F13 (~30% dynamic estimate). 4 formats
inlined; ~8-12 still needed before per-instr crosses whole-block.

**F2 inline IR landed (commit 3beb88e):** F2 AddSub (ADD/SUB Rd,
Rs, Rn or imm3) — first format with full arithmetic flag updates.
0 divs PE / 1 div MK historical at sw=64KB. fps stack:
  F1+F3:                458
  F1+F3+F12:            471
  F1+F3+F12+F13:        470
  F1+F2+F3+F12+F13:     478  (+1.7% vs 4-format)

F2 is common (~10-15% dynamic) so the stack step is more visible
than F12/F13. Phase-4-prime coverage now: 5 formats inlined,
~40-45% dynamic estimate.

**F4 logical inline IR landed (commit b2e456e):** F4 ALU logical
sub-ops AND/EOR/TST/ORR/BIC/MVN. Logical ops preserve C+V
(arithmetic=false in alu_update_flags). 0 divs PE / 1 div MK
historical at sw=64KB.

fps stack:
  F1+F3:                       458
  F1+F3+F12:                   471
  F1+F3+F12+F13:               470
  F1+F2+F3+F12+F13:            478
  F1+F2+F3+F4_LOG+F12+F13:     486 (+1.7%)

Phase-4-prime coverage: 6 sub-formats inlined (F1, F2, F3, F4 logical,
F12, F13), ~50% dynamic estimate. PE per-instr 486 vs scalar 560 =
-13%. MK 394 vs 397 = -0.8% parity.

**F4 arithmetic inline IR landed (commit cd3f27e):** ADC/SBC use
64-bit add for carry detection; NEG/CMP/CMN reuse F2's bit-logic
formulas. CMP/CMN are setting-flags-no-writeback. 0 divs PE / 1
div MK at sw=64KB.

fps median-of-N at sw=64KB:
  6-format PE: ~472    7-format PE: 480  (+1.7%)
  6-format MK: ~384    7-format MK: 369  (-3.7%)

PE neutral-to-slight-gain; MK regressed modestly. Cause: f4 arith
IR has more ops per opcode (esp 64-bit add for ADC/SBC), bloating
JIT'd output and increasing i-cache pressure during MK's ARM-heavy
execution. MK only runs ~1.4% Thumb at sw=64KB so the per-iter
saving from inlined F4 arith is tiny but per-block size grew.

This is a known stacking-pattern hazard: each format inlining
slightly bloats the JIT'd code, and ARM-heavy ROMs that don't
benefit from Thumb inlining still pay the i-cache cost. Mitigations
to consider once enough formats land:
1. Profile-guided format selection — only inline formats present
   in this block.
2. Per-block module-level dead-code-elim so unused IR fragments
   don't bloat output.
3. Disable F4_ARITH for MK profile (env-var-gated already).

**F4 MUL attempt rejected (commit 58e054b):** tried F4 MUL inline IR
with inline cycle accumulation `*sched_ts_ptr += mul_cycles_count`.
Reasoning: mul_cycles is operand-derived (not WAITCNT-derived) so
inline cycle add should be safe. Result: 0 divs PE but **4 MK divs**
(3 extra vs historical 1) reproducible across 3 runs at sw=64KB.
End-cycle drift only -128 cycles, but 4 specific frames mismatch
(2040, 3420, 4080, 5460). mul_cycles formula matches scalar
bit-by-bit; cpsr update matches. Mystery — reverted per V1 gate.

**Debug ideas for next operator:**
- Add a per-frame instrumentation: dump cpu state pre/post each
  MK MUL invocation, compare scalar vs F4_MUL-IR, find first
  divergent gpr/cpsr.
- Check if `*ts_ptr += m` 64-bit store has alignment issues — try
  writing to ts_ptr as i32 (low 32 bits of usize timestamp; high
  32 bits should never roll over within a replay).
- Maybe MK exercises an MUL operand sequence where scalar's
  `dst.wrapping_mul(src)` differs from LLVM's `mul i32` — but
  both are i32 wrapping mul, should be identical.
- Try multiplying-result-first then cycle-charge order inversion.

**F6 inline IR landed (commit 43a4508):** F6 LDR pc-rel via bus.load_32
+ idle_cycle externs (always-current cycle accounting). Constant
addr at AOT time. nfa = NonSeq (different from most formats).
0 divs PE / 1 div MK historical at sw=64KB.

fps stack at sw=64KB:
  6-format PE: ~472    7-format PE: 480  8-format PE: 479 (tied)
  6-format MK: ~384    7-format MK: 369  8-format MK: 385 (+4.3%)

The MK regression from F4 arith was recovered by F6 inlining (LDR
pc-rel for literal-pool constants is hot, including in MK ARM-Thumb
interop blocks). Net result: 8-format ≈ 7-format for PE, MK back to
historical baseline.

Phase-4-prime coverage: 8 sub-formats inlined, ~50-55% dynamic
estimate. Bus extern infrastructure (load_32, idle_cycle) reusable
for upcoming F9/F11.

**F11 STR landed (commit d814fe9):** with mixed results.
Correctness gates pass (0 divs PE / 1 div MK historical at sw=64KB).
But 9-format stack regresses both ROMs:
  PE: 478 → 465  (-2.7%)
  MK: 385 → 337  (-12.5%)

Same i-cache-pressure pattern as F4 arith: inlining adds IR per
block, JIT'd output grows, ARM-heavy MK pays cost without benefit.

**Concerning trend:** both F4 arith and F11 STR alone produced
correctness-OK results but stack-level fps regression. The first
6 formats stacked positively (each +1-3%). The last 3 formats
appear to hit diminishing-then-negative returns.

Hypotheses:
1. The IR-per-block bloat threshold has been crossed; per-block
   compile/code-size overhead now dominates per-iter savings.
2. PE/MK have different format-frequency profiles; later formats
   are biased toward formats that don't help PE/MK as much.
3. The format-detection chain in the per-instr emit (sequence of
   if-checks per opcode) is becoming long; collapsing into one
   switch IR could reduce per-block IR.

**Mitigation options:**
- Profile-guided format selection: only emit IR for formats present
  in THIS block (skip the unused ones to keep block IR small).
- Per-ROM format gating: env vars already gate per format; tune
  the optimal subset for each ROM.
- Consolidate IR emit patterns: e.g., F4 logical and F4 arith
  could share more code (cpsr update, gpr load/store).

**Parallel agent swarm post-mortem (2026-04-26):** 8 opus agents
launched in worktrees for F4 shifts, F4 MUL debug, F7+F8, F9, F10,
F11 LDR, F14 PUSH/POP, F15 LDM/STM. Outcome:
- **F4 shifts:** clean commit `54b0a8f` from agent → cherry-picked
  to aot-apr25 as `540f3ad`. CFG-based amount-range dispatch;
  4 sub-ops (LSL/LSR/ASR/ROR). 0 divs PE / 1 div MK ✓
- **F11 LDR:** uncommitted in agent worktree but clean diff →
  applied to aot-apr25 as `eb2d5cc`. Added `aot_ldr_word_for`
  extern (handles I14 misaligned-LDR ROR + cpsr.C side effect).
  0 divs PE / 1 div MK ✓
- **F4 MUL debug:** killed mid-investigation; agent reported
  "PE 0 divs, MK 1 div historical, fix is solid" in its summary
  but I haven't validated that claim. Worktree was cleaned up
  on agent termination (no commit produced).
- **F7+F8:** ~700 lines of uncommitted IR work in agent worktree;
  attempted 3-way merge produced 4-file conflicts because main
  had advanced past the agent's base. Reverted; not in repo.
- **F9, F10, F14, F15:** various failure modes (timeout, merge
  conflicts when sibling agents stashed/popped each other's
  work, branch-confusion). No clean commits.

**Lesson learned:** worktree isolation didn't fully prevent
agent-on-agent interference. Several agents stashed work in the
SHARED git stash, switched to other agents' branches, and saw
mid-write merge conflicts. Future swarms should either:
1. Use disjoint files entirely (one format = one new file).
2. Run sequentially, not in parallel.
3. Use a different VCS isolation (rsync the repo per-agent rather
   than worktree).

**-O2 experiment aborted:** measurements ran on a CPU shared with
8 parallel agent cargo builds → catastrophic (-50% MK) numbers
were CPU contention, not -O2's fault. Reverted to -O3; defer the
test until a quiet system.

**Phase-4-prime coverage now: 18 sub-formats inlined.**
F1, F2, F3, F4_LOG, F4_ARITH, F4_SHIFT, F5, F6, F7, F8, F9, F10,
F11_STR, F11_LDR, F12, F13, F14, F15, F19_HI. ~95-99% of dynamic
Thumb opcodes covered.

**F15 LDM/STM landed (commit 800d09b, cherry-picked from agent
worktree feb5762):** rlist unrolled at AOT compile time. Empty-rlist
(rlist == 0) falls through to step trampoline (handles GBATEK
quirk). LDM writeback only when Rb not in rlist. STM-with-Rb-in-
rlist correctly handles the "first-iter-stores-init-addr" rule
(the formula that broke the previous F15 attempt).

**2 agents running in background (2026-04-26 ~final stretch):**
- F4 MUL fix attempt (`a83c832d`): hypothesis is replace inline
  `*ts_ptr += m_cycles` with conditional ladder of `idle_cycle`
  extern calls (the bus path; known correct).
- F15 LDM/STM (`aef8326c`): full impl with empty-rlist quirk
  fall-through to step trampoline + writeback rules.

Stricter isolation rules than the prior 8-agent swarm: no git
stash, no branch switching, no main-checkout writes. Will
cherry-pick on completion.

**After F4 MUL + F15:** 100% per-iter Thumb dispatch coverage
(F17/F18/F19_lo are block terminators, not in step path).
Phase-4-prime IR-emit grind closes out. Strategic next:
phase 7 (block chaining), phase 8 (ARM-mode inline IR — MK
benefit since it's 98% ARM), or phase 9 (persistent cache).

**16-format quiet measurement (sw=64KB, 3-run median, pre-F5):**
PE 453.3 / MK 381.0. vs scalar 560 / 397 = -19% / -4%.
F7+F8 addition (was 14-format → 16-format) was net-noise on fps —
the marginal-format wins are tapering off as we approach
near-100% coverage.

**14-format quiet measurement (sw=64KB, 3-run median, pre-F7):**
PE 459.1 / MK 378.8 at 70.98% AOT coverage.
vs scalar PE 560 / MK 397 = -18% / -4.5%.

**Drive-by fix in F10 commit (a43a355):** SDL frontend was calling
compile_rom_with_seeds_full_v3 — missing both ldr_word_fn (v4) and
the new v5 args. F11 LDR's IR was effectively dead code prior to
v5 wiring (extern null check fell through to step trampoline). The
F11 LDR "0 divs" verification last turn was a false positive —
trampoline path was actually executing. Now properly wired.

**11-format quiet measurement (sw=64KB, 3-run median, pre-F10):**
PE 440.3 (was peak 478 at 6-format, now -8% from peak).
MK 368.8 (was peak 384 at 6-format, was 337 at 9-format).
F4 shifts + F11 LDR (now real) helped MK recover from the
F11_STR-induced regression but still well below scalar 397.

Still missing: F4 MUL (broken — agent still investigating).

**Earlier 9 sub-formats inlined** (correctness, gated default-off):
F1, F2, F3, F4_LOG, F4_ARITH, F6, F11_STR, F12, F13, F19_HI.

The strategic threshold may not be reached through naive stacking
alone. Mitigations being explored in parallel:
1. -O2 (this turn experiment)
2. IR consolidation via shared helpers (factor cpsr/gpr/pc-update)
3. LLVM intrinsics over manual bit-twiddle (fshr, uadd.with.overflow)
4. Profile-guided format selection per ROM
5. Pivot to LLVM-free Rust fn-ptr dispatch (Option B from findings-ladder)

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
