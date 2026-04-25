# docs/aot-llvm-program.md

Autoresearch program (Karpathy format) for an **AOT-LLVM** dispatcher
that beats the cached-interpreter scalar baseline on `pokeemerald` and
`Mario Kart Super Circuit`. One scalar to maximize, one command that
prints it, one TSV log of every experiment, loop forever.

This is the runbook an agent follows, commit by commit. The
retrospective on why JIT-mode trampoline dispatch failed lives at
`docs/llvm-jit-status.md` on the `shape-opt/apr22` branch. Read it
before this doc.

---

## Operating rhythm

**Set up a 15-minute reminder loop before starting.** Use the agent's
scheduling capability (e.g. `/schedule` or `ScheduleWakeup`) to fire
every 15 minutes with the prompt: *"continue the AOT autoresearch
loop per docs/aot-llvm-program.md, follow the experiment loop
section, do not stop until phase 9 or two consecutive permanent-fails
trigger replan."* The reminder is what keeps the loop running across
context turns; without it, work stalls at the first compaction.

**Commit style: human, never Anthropic.** Every git commit on this
branch is authored by the agent as the human user, NOT as Claude. No
`Co-Authored-By: Claude...` trailer. No "Generated with Claude
Code". Use lowercase, brief, slightly-imperfect English with at least
one spelling mistake or grammar quirk per message — that's the user's
voice. Example acceptable:

    phase 1: inline f3 mov imm8, +7% PE +4% MK

    inlined the F3 MOV imm8 path so the LLVM IR writes gpr[rd]
    directly insted of going through the trampoline. flag update
    is just clearing N/Z bits in cpsr because imm8 cant be negative
    or set carry/overflow. SDL replay 0 divs both ROMs.

Example NOT acceptable (any of these flags = redo):

    feat(dynarec): inline F3 MOV immediate handling

    Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

    🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## Hypothesis

Cached interpreter wins today (~540 PE / ~383 MK fps) because it runs
as a tight inline Rust loop with PGO'd direct handler calls — no extern
boundary per instruction. The LLVM JIT lost (~190 PE / ~92 MK) because
every per-instruction trampoline call paid a function-call boundary
that scalar never paid.

**Claim:** if we compile each ROM-resident Thumb (and ARM) instruction
to **inline LLVM IR** at ROM-load time — no trampoline calls per instr,
direct bus access, inlined scheduler advance, inlined abort check, lazy
flags, block chain — LLVM's optimizer can match or beat scalar. rustc's
backend IS LLVM, and rustc-LTO'd scalar handlers are the existence
proof of the codegen ceiling.

### Tying design choices to specific JIT-era lessons

Every architecture decision below traces to a thing that broke on the
JIT branch. Don't regress to JIT-style thinking.

- **JIT used per-instruction trampolines** because the LLVM module saw
  the bus as opaque. **AOT inlines bus access via baked offsets.**
- **JIT had a separate abort-check trampoline every other iter.**
  **AOT inlines the abort check** as loads + cmp + br, no extern.
- **JIT compiled at runtime per block.** **AOT compiles at ROM load.**
- **JIT's chain attempt regressed** because tail-call wasn't reliably
  optimized and per-instr cost dominated chain savings. **AOT chains
  only after per-instr cost is gone** (phase 7), with stack-overflow
  detection (I10).
- **JIT's K=2 abort cadence was load-bearing** — k=4 broke divs.
  **AOT keeps K=2.**
- **JIT's cpsr round-trip was redundant.** **AOT skips it.**
- **JIT's standalone per-format compile fns each created a new
  module.** **AOT emit fns take a `Builder<'ctx>`** and emit into
  a shared module.

---

## The single scalar

    aot_score = (pe_fps_aot - pe_fps_scalar) + (mk_fps_aot - mk_fps_scalar)

Negative → AOT slower. Zero → matched. Positive → beating. Gate is
**monotone per ROM** — no PE-only wins that regress MK and vice versa.

**Quantified success target:** the program is "done" when
`pe_fps_aot > pe_fps_scalar * 1.05 AND mk_fps_aot > mk_fps_scalar *
1.05` (5% over scalar on both ROMs, well outside noise). Anything
between -2% and +5% is "in the noise gray zone" — keep iterating.
Below -2% on either ROM after phase 5 is a sign the architecture is
wrong; trigger replan.

Hard fails (regardless of fps):

- `pe_divs > 0` or `mk_divs > 0`  — framebuffer-hash diff between AOT
  and scalar over the recorded replay.
- `|pe_drift| > 1000` or `|mk_drift| > 1000` cycles — end-of-replay
  emulated-cycle delta vs scalar.
- Any per-instruction differential test failure (gpr/cpsr/pc/cycles
  mismatch on a constructed input).
- Replay non-determinism: same ROM + same .rec across two runs of
  the same build must produce identical fb hashes.

Secondary metrics, printed but not gated:

- `aot_compile_ms_<rom>` — wall time for the at-load AOT pass.
- `aot_code_size_kb_<rom>` — total emitted machine code; warn at
  64MB, fail at 256MB (per I20).
- `aot_coverage_pct_<rom>` — % of dispatcher ticks (block-entry
  events) during the replay whose entry-PC lookup hit a non-null
  AOT table slot.

### Run-to-run variance

Harness runs SDL replay **3 times per ROM** and takes median fps.
Single-shot fps has 3-5% jitter on this host; median-of-3 stabilizes
to ~1-2%. Setup step 7 characterizes noise; raise N to 5 if 2σ > 5%.

---

## Constraints / Invariants

These are hard-set across the whole program. Don't relax without
writing `findings-invariant-<name>.md` first.

**I1. cpu.gpr is the source of truth.** AOT does *not* allocate guest
registers to LLVM SSA values across instruction boundaries. Each
instruction loads from `cpu.gpr[Rn]`, computes, stores back. LLVM's
mem2reg may promote to SSA *within a block* where safe; let it.
(The Cranelift dynarec's `GprCache` experiment caused silent
divergences — same class of bug here would cost weeks.)

**I2. K=2 abort cadence.** Mid-block yield check fires before every
odd instruction (k=1, 3, 5, …) and never on k=0. K=4 broke divs in
the JIT branch. Coded as `const ABORT_CADENCE: usize = 2;`.

**I3. IO region needs real bus calls.** Memory operations targeting
`0x04000000-0x040003FE` (IO regs), `0x05000000-0x05000400` (palette),
`0x06000000-0x06017FFF` (VRAM), `0x07000000-0x070003FF` (OAM),
`0x0E000000-0x0E00FFFF` (cartridge backup) are side-effecting and
emit a real `bus.read_*` / `bus.write_*` call. Pure-RAM regions
(BIOS / EWRAM / IWRAM / ROM) inline. Region check is at compile
time when the address is constant; runtime branch-on-region when
the address is reg-offset.

**I4. IWRAM blocks are scalar-only.** AOT compiles ROM (BIOS +
cartridge) blocks. IWRAM is recorded-and-cached the existing way and
runs through scalar replay. Rationale: IWRAM SMC is rare but legal,
and the existing `block_cache_dirty` + `flush()` mechanics already
handle invalidation.

**I5. ROM is read-only.** Once an AOT block compiles, it's valid for
the process lifetime. No ROM-write invalidation path needed.

**I6. Pipeline state at AOT entry.** When the dispatcher transitions
scalar → AOT block:
- `cpu.pc == block.entry_pc & !1` (pipeline-head pc, Thumb bit
  stripped).
- `cpu.pipeline[0]` contains the first instruction's opcode.
- `cpu.pipeline[1]` contains the second instruction's opcode (or any
  value if the block has only one instruction).
- `cpu.next_fetch_access` is the access type for the *next* fetch.

**I7. Cycle costs recompute on WAITCNT change (synchronous).**
Per-region cycle costs are baked as constants at AOT compile time.
The bus maintains a u32 generation counter that increments on any
WAITCNT write. The first WAITCNT write at runtime triggers a
**synchronous** AOT-cache flush + recompile pass (blocks the CPU
thread for the duration). This blocks gameplay for ~5s once per
ROM; in practice WAITCNT is set during BIOS/cartridge init and
never changed thereafter, so this fires zero or one times per
session. If it fires repeatedly (some weird game does it), the
recompile pass falls back to "permanently scalar for this ROM
session" after the second WAITCNT change to avoid recompile thrash.

**I8. Compiled fn ABI — single-pointer cpu_ctx.**
`extern "C" fn(*mut u8 cpu_ctx, *mut u32 pc_out) -> u32`. The
emit fns derive ALL state through `cpu_ctx`: gpr at
`cpu_ctx + gpr_offset`, cpsr at `cpu_ctx + cpsr_offset`, pipeline
at `cpu_ctx + pipeline_offset`, etc. **Do NOT pass `gpr_ptr` as a
separate argument**, because LLVM's argument-promotion / noalias
inference may treat it as not-aliasing `cpu_ctx`, leading to
miscompiles when both writes hit the same memory. Single-pointer
ABI is the simplest fix.

`pc_out` is the only other arg, written only when return = `0b01`.

Return:
- `0`     → fall-through to next block.
- `0b01`  → branch fired, target in `*pc_out` with Thumb bit in bit 0.
- `0b10`  → mid-block abort, dispatcher yields.

**I9. Determinism.** AOT compile order, FxHashMap iteration order,
parallel-compile thread scheduling — none of these may affect
*runtime behavior*. They MAY affect emitted machine code bytes.
The test gate uses runtime fb-hashes, not binary diff. The harness
runs the SAME .rec twice and verifies identical fb-hash sequences
(determinism check, hard fail).

**I10. Block chain stack safety.** Block chain (phase 7) uses LLVM
tail call. `set_tail_call(true)` is best-effort. Phase 7 includes:
1. Post-compile asm-dump check confirming the tail-call slot
   actually emitted `jmp` (or arch equivalent), not `call`. If
   not, chain disables itself and dispatcher falls through.
2. Chain-depth counter passed via cpu_ctx, decremented per chain
   step; on underflow (depth > 256), chain returns to dispatcher.

**I11. AOT enable ordering.** `enable_aot()` MUST be called before
the first instruction is fetched. SDL frontend (and fps_bench) call
it immediately after `GameBoyAdvance::new`, BEFORE `skip_bios` /
`frame()`.

**I12. Save-state ownership.** AOT table is owned by the *bus* (or
held in an `Rc` shared with the bus) so save_state / restore_state
on the CPU don't drop it. After save-state restore, AOT lookup
gracefully misses on non-block-aligned PCs and falls to scalar
until the next block boundary.

**I13. gpr aliasing inside cpu_ctx.** Per I8: there is no separate
`gpr_ptr` argument; gpr is accessed through `cpu_ctx + gpr_offset`.
This avoids aliasing surprises with LLVM's noalias inference. Emit
fns:
- Load `gpr[Rn]`: `gep i8 cpu_ctx, gpr_offset + Rn*4` → `load i32`.
- Store `gpr[Rd]`: same gep → `store i32`.
- LLVM's mem2reg may promote within a basic block (good).

**I14. Misaligned word load ROR semantics.** ARM7TDMI `LDR Rd, [Rn]`
with `Rn & 3 != 0` doesn't fault — it loads from the aligned
address and ROR's the result by `(Rn & 3) * 8` bits. CPSR.C is
unaffected for `LDR` proper (the `LDRH` half-word case has its
own quirk). Emit IR:

    aligned_addr = addr & !3
    raw          = load i32, aligned_addr
    shift        = (addr & 3) * 8
    result       = (raw >> shift) | (raw << (32 - shift))   ; ROR
    ; for shift==0 the `raw << 32` is UB in IR — guard:
    if shift == 0: result = raw

The CPSR.C bug fixes on shape-opt/apr22 (F7/F9/F11) are exactly
this same logic; port the test cases.

**I15. Cold-start pipeline init.** After power-on / reset, pipeline
is zeroed. Scalar's `reload_pipeline32` fetches the first two
instructions before the first execute. AOT's first block ASSUMES
pipeline[0] is valid (per I6). The AOT lookup is gated on
`cpu.pipeline[0] != 0` (or a "pipeline_initialized: bool" flag on
the CPU struct, set after first `reload_pipeline*`). The very
first dispatcher tick after reset always falls to scalar, scalar
runs the BIOS reset vector, pipeline becomes valid, AOT takes over
on the next dispatcher tick.

**I16. Per-block instruction cap.** Blocks > 32 instructions get
**split** at AOT compile time into chained sub-blocks of ≤ 32
instructions each. Split point is just before instruction 32; the
first sub-block falls through to the second via the chain
mechanism. Bounds emitted code size per block to ~5KB (32 × ~150
bytes after -O3) and prevents pathological i-cache thrash.

**I17. Compile-then-publish.** AOT's per-page parallel compile is
behind a "table not yet ready" flag. The dispatcher uses scalar
until `compile_rom()` returns. `compile_rom` only returns when
ALL pages have finished compiling AND been added to the
ExecutionEngine. This avoids the race where a fast page calls
into a still-compiling slow page.

**I18. Dispatcher lookup is inlined.** `replay_cached_block`'s
AOT-table check is *inlined* into the cached_interp scalar
dispatcher, NOT an extern call. Implementation: the AOT crate
exposes a raw `*const AotTable` pointer that arm7tdmi stores in
the CPU struct (under the `aot` feature). The lookup is a
`#[inline(always)]` fn that does the 2-level array index dance
inline. That keeps the hot path one indirect call deep, not two.

**I19. WAITCNT recompile strategy.** First WAITCNT write fires a
synchronous flush + recompile (per I7). Second WAITCNT write
within the same session disables AOT for the session (sets a
"poisoned" flag, dispatcher uses scalar permanently from then on).
This bounds recompile pathology.

**I20. Code size budget.** Total `aot_code_size_kb` per ROM:
- < 64 MB → green.
- 64-256 MB → warn (logged, agent investigates which blocks are
  bloated; usually means I16 split threshold needs lowering).
- > 256 MB → hard fail (the experiment is malformed; revert).

**I21. AOT cache disposal.** When the user reloads ROM (or the
process terminates), the AOT ExecutionEngine drops, releasing JIT
pages to the OS. Phase-0 deliverable A0 verifies repeated
`compile_rom` followed by drop doesn't leak (RSS stable across N
cycles).

**I22. Scan target validation.** Every direct-branch target the
scanner queues is validated:
- Address inside ROM region (BIOS or cartridge ROM as relevant).
- Mode-correct alignment (Thumb pc & 1 == 0; ARM pc & 3 == 0).
- (Heuristic) decodes to a non-trivially-broken instruction (e.g.,
  doesn't decode to all-zero or all-one halfwords, which usually
  means data-as-code).

Scanner skips invalid targets; logs them in
`findings-rom-scan.md` for review.

**I23. Inkwell module memory accounting.** Each call to
`engine.add_module(...)` retains the module + its JIT'd code in
the engine. `compile_rom` adds N modules (one per ROM page). On
re-load (phase-9 persistent cache, or test harness compiling
twice), the engine drops + recreates rather than `remove_module`
per old module (simpler, less error-prone).

**I24. Replay determinism.** Two runs of the SAME .rec file on the
SAME build must produce IDENTICAL fb-hash sequences. Verify in
phase 0 by running the harness twice and comparing.

**I25. Pre-flight ABI sanity.** Phase-0 deliverable A0: compile
a constant-returning function via inkwell, JIT-execute it, assert
it returns 42. The JIT branch had this exact test
(`compile_constant_42`); port it. If A0 fails, no point on phase
1 — environment isn't set up.

**I26. gba-tests under both feature combos.** gba-tests passes
under `--features cached_interp` AND `--features cached_interp,aot`.
With AOT enabled but no real cartridge ROM, AOT-miss falls to
scalar and gba-tests behave identically. If gba-tests breaks under
either, the dispatcher's scalar fallback is broken.

**I27. Workspace clippy / rustfmt.** New crate `arm7tdmi-aot`
passes `cargo clippy --workspace --features aot -- -D warnings`
and `cargo fmt --check`. CI doesn't run AOT tests but DOES run
clippy/fmt — keep the new crate green.

**I28. Inkwell version pin.** `arm7tdmi-aot` pins
`inkwell = "0.9"` with `features = ["llvm18-1-prefer-dynamic"]`.
Same as the JIT branch. Don't bump versions during the
experiment loop — that's a separate research thread.

---

## Setup

1. Tag the run, e.g. `aot-apr25`. Branch `aot-llvm/<tag>` must not
   already exist.
2. `git checkout -b aot-llvm/<tag>` from the head of `aot-llvm`.
3. **Set up the 15-minute reminder loop.** Use the agent's
   scheduling capability now, BEFORE writing any code:

        # pseudocode
        ScheduleWakeup(delaySeconds=900,
                       prompt="continue AOT autoresearch loop per
                               docs/aot-llvm-program.md, follow the
                               experiment loop section, do not stop
                               until phase 9 or two replan triggers",
                       reason="AOT loop heartbeat")

   The loop fires every 15 minutes. Each fire: re-read
   `aot-progress.md`, identify the current phase, continue from
   wherever the previous turn left off.

4. Read every in-scope file end-to-end before touching anything:
   - `arm7tdmi/src/cache.rs`
   - `arm7tdmi/src/cpu.rs::replay_cached_block`
   - `arm7tdmi/src/thumb/exec.rs`, `arm7tdmi/src/arm/exec.rs`
   - `arm7tdmi/src/memory.rs`, `core/src/sysbus.rs`
   - `arm7tdmi/src/psr.rs`
   - `arm7tdmi/src/exception.rs`
   - `core/src/iodev.rs` and IO sub-modules (used by I3)
   - `core/src/sched.rs` (`add_cycles` + `has_events_due`)

5. Environment checks. **The libpolly-18 workaround is mandatory:**

        # LLVM 18 toolchain
        export LLVM_SYS_181_PREFIX=/usr/lib/llvm-18
        ls "$LLVM_SYS_181_PREFIX"/bin/llvm-config-18

        # IMPORTANT: arm7tdmi-aot's inkwell dep needs:
        #   features = ["llvm18-1-prefer-dynamic"]
        # because Ubuntu 24.04's llvm-18-dev ships WITHOUT libPolly.a.
        # Static linking via the default llvm18-1 feature fails:
        #   "could not find native static library Polly"
        # The prefer-dynamic feature links libLLVM-18.so.1 which
        # doesn't separately depend on Polly.

        # SDL display
        echo "DISPLAY=$DISPLAY"

        # ROMs and recordings
        ls /home/user/pokeemerlad/pokeemerald/pokeemerald.gba
        ls "/tmp/mks/Mario Kart - Super Circuit (USA).gba"
        ls /tmp/pokeemerald_run.rec /tmp/mks.rec
        ls core/benches/roms/normatt_gba_bios.bin

6. Regenerate scalar reference hashes:

        cargo build --release -p rustboyadvance-sdl2 --features cached_interp
        target/release/rustboyadvance-sdl2 --no-audio --replay /tmp/pokeemerald_run.rec \
            --frame-hash-every 60 --bios core/benches/roms/normatt_gba_bios.bin \
            /home/user/pokeemerlad/pokeemerald/pokeemerald.gba 2>/dev/null \
            | grep "fb_hash" > /tmp/aot_pe_scalar_ref.txt
        # ... same for MK ...

7. **Characterize noise.** Run baseline (AOT off → scalar) five
   times back-to-back. Compute σ on per-ROM fps. Harness uses
   median-of-3; if 2σ > 5% of scalar fps, raise to median-of-5.

8. Initialize `results.tsv` (untracked):

        printf 'commit\taot_score\tpe_fps_aot\tpe_fps_scalar\tmk_fps_aot\tmk_fps_scalar\tpe_drift\tmk_drift\tcompile_ms\tcoverage_pct\tdivs_pe\tdivs_mk\tdiff_failures\tstatus\tdescription\n' > results.tsv

9. Run the baseline and record (status `baseline`).

10. **Profile the instruction-format distribution** (drives phase 1-3
    ordering — see Phase-0 deliverable A7):

        bash scripts/aot_profile_distribution.sh > docs/findings-shape-distribution.md

    Phase-1 emit-fn ordering = top-N most-executed formats by ROM,
    NOT a fixed list.

---

## Architecture sketch

    ┌─────────────────────────────────────────────────────────────────┐
    │ AOT pass at ROM load (new):                                     │
    │   scan ROM (BIOS + cartridge) → reachable Thumb/ARM blocks      │
    │   for each block (cap 32 instrs per I16, split if longer):      │
    │     emit_inline_block_ir(...) into per-page LLVM module         │
    │   parallel compile modules with -O3                             │
    │   publish two-level PC→fn table ATOMICALLY (per I17)            │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ replay_cached_block (existing, modified) checks AOT FIRST:      │
    │   inlined lookup (per I18) → on hit: call compiled fn           │
    │   miss → record-and-trace → scalar replay                       │
    └─────────────────────────────────────────────────────────────────┘

### What "AOT miss" covers permanently

- **SWI / undefined / IRQ entry** — exception handler sets pc to
  BIOS vector dynamically; scalar handles entry, AOT picks back up
  on the return path.
- **IWRAM-resident code (I4).**
- **Mode-flip points** the AOT scan didn't trace.
- **Indirect branches** the static reachability scan couldn't
  resolve (see fallback below).
- **WAITCNT-stale blocks** during recompile (I7) or
  permanently-scalar after second WAITCNT change (I19).
- **Cold-start first dispatcher tick** until pipeline is initialized
  (I15).
- **Save-state restored at non-block-aligned pc** until next
  pipeline flush (I12).

### Indirect branch fallback

Static scan misses ~30-40% of code on real games (jump tables, BX
register, return addresses on the stack). For PCs the scan misses,
the dispatcher falls through to the existing record-and-cache path:
the block records under cached_interp, runs scalar, stays scalar.

If miss rate is too high, phase-9 adds a trace pass: log every
runtime-observed BX / LDR-PC target, feed as a "known-PC list" into
AOT's static scan as a second pass.

### IO region handling (I3)

AOT emit for Thumb LDR/STR (any width) checks the address's region
at compile time (constant base+offset case) or runtime (reg-offset
case):

- **Constant address resolves to** BIOS / EWRAM / IWRAM / ROM →
  emit inline pointer arithmetic via baked region-base offset. No
  extern call. Scheduler advance inlined.
- **Constant address resolves to** IO / palette / VRAM / OAM / SRAM
  → emit `bus.read_*` / `bus.write_*` call.
- **Reg-offset address** → emit a region-dispatch ladder:
  `if addr in [iwram_base, iwram_end]` → inline; `else` → real call.

### Misaligned word load ROR (I14)

Every `LDR` emit applies the alignment ROR per I14. The same logic
ports from the shape-opt/apr22 CPSR.C fix commits. Don't skip this
even for the "constant address known to be aligned" case — the
emit is the same logic; LLVM constant-folds when shift is known
to be 0.

### Variable-latency ARM ops

ARM MUL / MLA / UMULL / UMLAL / SMULL / SMLAL: cycle counts depend
on operand value. Phase 8 emits a real `bus.add_cycles(mul_cycles
(op2), ...)` call where `mul_cycles` is a small extern fn doing the
CLZ detection. One extern boundary per mul instruction; multiplies
are < 0.5% of dynamic instr in target ROMs so this is fine.

Phase 8a optionally inlines the CLZ ladder if A7 shows mul > 1%
dynamic.

### Thumb F19 (BL pair)

Two consecutive Thumb halfwords act as one logical branch-with-link.
AOT scan detects the F19 hi half and emits ONE IR fragment for the
pair: compute target, write LR, set pc, return `0b01`. NEVER emit
two independent F19-half emits. Block terminator is the F19 lo
opcode.

### In-block branches (CFG within block)

A block can contain a non-tail Bcc or B that doesn't terminate the
block (Body::CondBranch / Body::UncondBranch in dynarec lineage):

- **Bcc not taken** → continue.
- **Bcc taken** → block ends, return `0b01` with target.

Emit:

    eval cond against current cpsr → %cond_v
    br %cond_v, label %taken_exit, label %not_taken_continue

    taken_exit:
      compute target (constant since pc-rel known at compile time)
      store target to *pc_out
      return 0b01

    not_taken_continue:
      ; next instruction's emit picks up here

In-body unconditional B: simpler — block ends unconditionally,
target is constant.

### Block splitting for I16

Blocks > 32 instructions split at AOT compile time. Sub-block N
ends with an unconditional fall-through to sub-block N+1's pc. The
chain mechanism (phase 7) tail-calls between them. Pre-phase-7,
sub-blocks just return 0 and the dispatcher rolls forward — slower
but correct.

### PC→fn table data structure (two-level, lookup inlined per I18)

Top-level: 256-entry array indexed by `(pc >> 16) & 0xff`. Each entry
is `Option<Box<[Option<CompiledFn>; 32768]>>`. Pages allocated lazily.

Leaf: 32768-entry dense array indexed by `(pc >> 1) & 0x7fff`.

Lookup (inlined into `replay_cached_block`):

    fn aot_lookup_inline(table: *const AotTable, pc: u32) -> Option<CompiledFn> {
        let top = (pc >> 16) as usize & 0xff;
        let leaf = unsafe { (*table).pages[top].as_ref()? };
        let idx = (pc >> 1) as usize & 0x7fff;
        leaf[idx]
    }

`#[inline(always)]`, no extern call.

### ROMs with decompression / runtime-loaded code

Some games decompress code from ROM into IWRAM at boot. AOT compiles
the COMPRESSED ROM bytes, which are never executed as instructions.
Wasted compile, harmless. The actually-running code lives in IWRAM
and is scalar-only per I4. Coverage will be lower on these games;
that's OK.

### Compiler crate boundary

AOT code lives in a NEW crate `arm7tdmi-aot` (sibling of `arm7tdmi`
in the workspace). Cargo.toml depends on `arm7tdmi`, `inkwell` (with
`llvm18-1-prefer-dynamic`), `rustc-hash`. `build.rs` enforces LLVM
18 presence with a clean error.

`arm7tdmi-aot` exposes:
- `pub struct AotTable`
- `pub fn compile_rom(rom: &[u8], entry_pc: u32) -> AotTable`
- `pub fn enable_aot_on<I>(cpu: &mut Arm7tdmiCore<I>, table: AotTable)`
- `pub fn aot_lookup(table: &AotTable, pc: u32) -> Option<CompiledFn>`
  (`#[inline(always)]`, used by arm7tdmi via raw ptr per I18)

`arm7tdmi/src/cpu.rs` exposes a `*const AotTable` field (under
`aot` feature) populated by `enable_aot_on`. Lookup is inlined at
the dispatcher call site.

### Graceful degradation on missing LLVM

`arm7tdmi-aot/build.rs`:

    fn main() {
        let prefix = std::env::var("LLVM_SYS_181_PREFIX")
            .expect("arm7tdmi-aot requires LLVM 18; set LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 or build without --features aot");
        if !std::path::Path::new(&prefix).join("bin/llvm-config-18").exists() {
            panic!("LLVM 18 not found at {} — install llvm-18 or fix LLVM_SYS_181_PREFIX", prefix);
        }
    }

---

## Phase 0 deliverables

Before phase 1's first inline-IR commit, the agent **must** write
and commit these as audit artifacts:

**A0. ABI sanity pre-flight** (`docs/findings-pre-flight.md`).
Compile a constant-returning function via inkwell, JIT-execute it,
assert it returns 42. Port `compile_constant_42` from
shape-opt/apr22's `dynarec_llvm.rs`. If A0 fails, environment
isn't set up — fix that before anything else.

**A1. Handler-pipeline-read audit** (`docs/findings-pipeline-read.md`).
Grep every Thumb and ARM handler for `cpu.pipeline[`. Decide per
handler: inline fetch can elide `read_16` (no read), or must keep
`load_16` (handler reads pipeline[1]).

**A2. IO-region map** (`docs/findings-io-regions.md`). Every IO
register a real game touches via LDR/STR. Confirm against
`resources/gbatek`. Output: `const IO_REGIONS: &[(u32, u32)]` in
`arm7tdmi-aot/src/bus.rs` + a `fn is_io_addr(addr: u32) -> bool`.

**A3. Indirect-branch fallback decision**
(`docs/findings-indirect.md`). Run scalar+log-bx-targets on
pokeemerald + MK. Histogram. Decide miss-strategy.

**A4. ROM-scan strategy spec** (`docs/findings-rom-scan.md`).
Cartridge entry: ROM byte 0 = ARM B; decode 24-bit signed offset;
target = `0x08000000 + 8 + (offset << 2)`. The 0xC0-byte cartridge
header lives between entry B and game code — DO NOT scan as code.
BIOS entry: pc = 0 in ARM mode. Scan algorithm: queue
(entry_pc, mode), walk basic blocks, validate targets per I22.

**A5. Compile-time budget check**
(`docs/findings-compile-budget.md`). Compile synthetic ROM of 1000
blocks. Measure blocks/sec, MB/sec, peak memory. Decide serial vs
parallel.

**A6. Differential test infra port**
(`docs/findings-diffinfra.md`). Port from shape-opt/apr22
`dynarec_pattern_differential.rs`. Diff bus = new `DiffBus` (NOT
SimpleMemory) with backing memory + real `Scheduler`. Each AOT
format emit gets a paired test asserting bit-equality across gpr
/ cpsr / pc / cycles.

**A7. Instruction-format distribution profile**
(`docs/findings-shape-distribution.md`). Run scalar+log-format on
target ROMs. Print % dynamic instruction count by Thumb format and
ARM class. Phase-1/2/3 emit-fn ordering = actual hot top-N from
this profile.

**A8. Debug / observability hooks**
(`docs/findings-debug-hooks.md`). Spec:
- `--dump-ir <pc>` → `module.print_to_string()` for that block.
- `--dump-asm <pc>` → capstone-disassembled native code.
- `--trace-block <pc>` → gpr/cpsr/cycles before+after each
  execution of that block, for 1000 dispatches.
- `--trace-block-instr <pc>` → same but per-instruction inside the
  block (deep debugging).

These nine audits land BEFORE the phase 0 scaffold commit. Not
optional. Every phase 1+ decision references one.

---

## Phased experiment ladder

| # | Experiment | Expected (PE / MK) | Gate |
|---|------------|--------------------|------|
| 0 | Scaffold: ROM scan, two-level PC→fn table (I18 inlined lookup), dispatcher hook, block split (I16). Every block emits placeholder trampoline call. | =scalar | divs==0; drift<1000; coverage>80%; determinism (I24) |
| 1 | Inline top-3 most-executed Thumb formats from A7. | +5-10% | gates + score>0 on at least one ROM |
| 2 | Inline next 3 hot formats from A7. | +10-15% | both ROMs score≥phase1 |
| 3 | Inline remaining no-memory ALU/load-addr formats (F12, F13). | +15-20% | both ROMs score≥phase2 |
| 4 | Inline fetch (per A1) + inline abort check (per A2/I3). | +20-30% | both ROMs score≥phase3 |
| 5 | Inline memory ops F6-F11 (RAM inlines per I3+I14, IO real bus call). | +20-30% | both ROMs score≥phase4 |
| 6 | Lazy flag materialization. | +10-15% | both ROMs score≥phase5 |
| 7 | Block chain via Rc<AtomicUsize> + LLVM tail call + I10 stack safety. | +10-15% | both ROMs score≥phase6 |
| 8 | ARM compile path (data-proc / mem / branch + cond field per-instr wrapper + variable-latency mul via real cycle-cost call). | +5-10% on MK | both ROMs score≥phase7 |
| 8a | (Optional) Inline CLZ ladder for mul cycles — only if A7 shows mul > 1%. | +1-2% | both ROMs score≥phase8 |
| 9 | Persistent compiled-blob cache (ROM SHA-256 keyed) + indirect-branch trace seeding (per A3). | first-launch faster | divs==0 |

**Early-stop rule:** if phase 5 wins (`aot_score > +5%` on both
ROMs per the quantified target), agent may skip 6-8 and go to
phase 9. Document early-stop in `findings-early-stop.md`.

**Phase-N permanent-fail re-planning:** if phase N's gate fails
twice with different approaches, STOP and write
`findings-ladder.md`:
- What was phase N supposed to deliver?
- Measured fps/correctness from each attempt.
- Hypothesis for why it doesn't deliver as expected.
- Re-plan: split phase N, drop it (move to N+1 with reduced
  expected gain), or reorder.

Re-planning is a commit (the new ladder, status `replan`) but
not a code commit. The experiment loop resumes.

---

## Compile-time budget

AOT compile happens at ROM load. Targets:

- pokeemerald (16MB, ~100k blocks): < 5s.
- Mario Kart (4MB, ~30k blocks): < 2s.
- BIOS (16KB, ~few hundred blocks): < 100ms.

100k × 100µs serial = 10s — over budget. Mitigations in order:

1. **Per-page parallel compile.** Each ROM 64KB page → its own
   LLVM module. Modules build on `std::thread` pool sized to
   `available_parallelism()`. Each thread owns its `Context`
   (inkwell borrow rules). Modules link into the shared engine
   via `add_module` from the main thread. **Compile-then-publish
   per I17** — table available only after ALL pages done.

2. **Cap module size.** Smaller modules compile faster.

3. **Phase-9 persistent cache.** Hash ROM, save
   `target/release/.aot-cache/<sha>.{ll,o}` after first compile,
   skip on subsequent runs.

`aot_compile_ms_<rom>` is logged but not gated. If it exceeds
target after phase 5, mitigation 1 ships ahead of phase 6.

### First-launch UX

Default: progress bar. SDL window opens immediately; AOT pass
runs on background thread; main thread reports `compiled X / Y`
to stderr. Headless replay just prints start/end timestamps.

---

## scripts/aot_measure.sh sketch

    #!/usr/bin/env bash
    set -e
    cd /home/user/pokeemerlad/rustboyadvance-ng

    LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 \
        cargo build --release --workspace --features aot \
        2>&1 | tail -3

    BIN=target/release/rustboyadvance-sdl2

    measure_rom() {
        local rom_label="$1" rom_path="$2" rec="$3" ref_hashes="$4"
        # AOT × 3 → median fps
        # scalar × 3 → median fps
        # divs = diff(aot[1].hashes, ref.hashes)
        # drift = end_cycle_aot - end_cycle_scalar
        # determinism = diff(aot[1].hashes, aot[2].hashes) must be 0
        echo "$aot_fps_med $scalar_fps_med $divs $drift ..."
    }

    PE_RESULTS=$(measure_rom pe ...)
    MK_RESULTS=$(measure_rom mk ...)

    DIFF_FAILS=$(LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 \
        cargo test --workspace --features aot aot_diff_ 2>&1 \
        | grep -c '^test .* FAILED')

    AOT_SCORE=$((... compute ...))

    cat <<OUT
    ---
    aot_score:           $AOT_SCORE
    pe_fps_aot:          $PE_FPS_AOT
    pe_fps_scalar:       $PE_FPS_SCALAR
    pe_drift:            $PE_DRIFT
    pe_divs:             $PE_DIVS
    pe_determinism_ok:   $PE_DETERMINISM
    mk_fps_aot:          $MK_FPS_AOT
    mk_fps_scalar:       $MK_FPS_SCALAR
    mk_drift:            $MK_DRIFT
    mk_divs:             $MK_DIVS
    mk_determinism_ok:   $MK_DETERMINISM
    aot_compile_ms_pe:   $COMPILE_MS_PE
    aot_compile_ms_mk:   $COMPILE_MS_MK
    aot_coverage_pct_pe: $COVERAGE_PE
    aot_coverage_pct_mk: $COVERAGE_MK
    aot_code_size_kb_pe: $CODE_KB_PE
    aot_code_size_kb_mk: $CODE_KB_MK
    diff_failures:       $DIFF_FAILS
    ---
    OUT

The agent fills in extraction details during phase 0 against the
actual SDL output format (which the agent also designs in phase 0).

---

## Output format

`scripts/aot_measure.sh` prints exactly the canonical block above.
Order is fixed and machine-parseable.

`*_divs` — fb-hash-line diff between AOT and scalar; non-zero → revert.
`*_drift` — `aot_end_cycle - scalar_end_cycle`; `|drift|>1000` → revert.
`*_determinism_ok` — `1` if two AOT runs match exactly, `0` otherwise;
                     `0` → revert.
`diff_failures` — per-instruction differential test failures; > 0 → revert.

---

## Logging results

Append one row to `results.tsv` per experiment:

    printf '%s\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%d\t%d\t%d\t%.1f\t%d\t%d\t%d\t%s\t%s\n' \
        "$(git rev-parse --short HEAD)" \
        "$AOT_SCORE" \
        "$PE_FPS_AOT" "$PE_FPS_SCALAR" \
        "$MK_FPS_AOT" "$MK_FPS_SCALAR" \
        "$PE_DRIFT" "$MK_DRIFT" \
        "$COMPILE_MS" "$COVERAGE_PCT" \
        "$PE_DIVS" "$MK_DIVS" \
        "$DIFF_FAILURES" \
        "$STATUS" \
        "$DESCRIPTION" \
        >> results.tsv

`STATUS` ∈ {`baseline`, `accept`, `reject_divs`, `reject_drift`,
`reject_diff`, `reject_determinism`, `reject_perf`, `reject_compile`,
`reject_size`, `replan`}.

`DESCRIPTION` is one short imperative line, **human-style with
imperfect grammar/spelling per the commit-style rule**.

---

## The experiment loop

For each phase:

1. **Read.** Open every file you'll touch end-to-end. NO grep-only
   navigation.
2. **Plan.** Write `findings-phaseN.md` *before* code: IR shape per
   format, what verifies it, what could break, what you'll measure.
3. **Test infra first.** Phase 1+: write the per-format differential
   test BEFORE writing the emit fn. Make it pass.
4. **Implement.** ONE commit per phase, human-style commit message
   (no Anthropic attribution, intentional minor mistakes per the
   Operating-rhythm rule).
5. **Test.** `bash scripts/aot_measure.sh > run.log`. Parse. Append
   to `results.tsv`. Any gate fail: `git reset --hard HEAD~1`,
   write `findings-phaseN-rejected.md`, re-plan.
6. **Validate.** `cargo test --workspace --features cached_interp`
   for scalar correctness; `cargo test --workspace --features aot
   aot_diff_` for AOT differential. Per I26 verify both pass.
   Per I27 run `cargo clippy --workspace --features aot -- -D
   warnings` and `cargo fmt --check`.
7. **Update.** Append phase result to `aot-progress.md`.
8. **Reschedule.** If the 15-min reminder hasn't fired in this
   phase, the next firing handles continuation. Do not stop the
   loop.

Repeat until phase 9, until early-stop fires, OR until two
consecutive permanent-fails trigger replan.

---

## Critical files

### New (added by the agent during the program)

- **`arm7tdmi-aot/`** — new workspace member crate. Cargo.toml
  depends on `arm7tdmi`, `inkwell = "0.9"` with
  `features = ["llvm18-1-prefer-dynamic"]` (per I28),
  `rustc-hash`. `build.rs` per "Graceful degradation" section.
- `arm7tdmi-aot/src/lib.rs` — exports.
- `arm7tdmi-aot/src/scan.rs` — static reachability per A4 + I22.
- `arm7tdmi-aot/src/emit/mod.rs` — per-format emitters.
- `arm7tdmi-aot/src/emit/thumb_*.rs` — one file per Thumb format.
- `arm7tdmi-aot/src/emit/arm.rs` — phase-8 ARM emit + cond wrapper
  + variable-latency mul.
- `arm7tdmi-aot/src/bus.rs` — offsets via `std::mem::offset_of!` +
  inlined RAM-region bus ops + region-dispatch ladder for
  reg-offset addresses.
- `arm7tdmi-aot/src/abort.rs` — inlined K=2 abort check.
- `arm7tdmi-aot/src/table.rs` — two-level PC→fn table + inlined
  lookup (I18).
- `arm7tdmi-aot/src/chain.rs` — phase-7 chain + I10 stack safety.
- `arm7tdmi-aot/src/diff.rs` — `DiffBus` + per-format differential
  test harness.
- `arm7tdmi-aot/src/dump.rs` — A8 debug hooks.
- `scripts/aot_measure.sh` — the harness.
- `scripts/aot_profile_distribution.sh` — A7 profile pass.
- `docs/findings-phase*.md` — one per phase.
- `docs/findings-pre-flight.md`, `findings-pipeline-read.md`,
  `findings-io-regions.md`, `findings-indirect.md`,
  `findings-rom-scan.md`, `findings-compile-budget.md`,
  `findings-diffinfra.md`, `findings-shape-distribution.md`,
  `findings-debug-hooks.md` — the nine Phase-0 audit deliverables.
- `aot-progress.md` (untracked) — one-line summary per accepted
  phase. Used by the 15-min reminder loop to know where to resume.

### Modified

- `Cargo.toml` (workspace) — add `arm7tdmi-aot` member.
- `arm7tdmi/src/cpu.rs` — adds `*const AotTable` field under `aot`
  feature, populated by `enable_aot_on`. Inlined lookup at the
  `replay_cached_block` call site (I18).
- `arm7tdmi/src/cache.rs::replay_cached_block` — checks AOT lookup
  first.
- `core/src/sysbus.rs` — exposes `&mut Scheduler` getter for AOT
  bus offset baking. WAITCNT-write side bumps `aot_gen_counter`
  (I7) and triggers synchronous recompile (I19).
- `platform/rustboyadvance-sdl2/src/options.rs` — `--aot`,
  `--dump-ir`, `--dump-asm`, `--trace-block`,
  `--trace-block-instr` flags under `aot` feature.
- `platform/rustboyadvance-sdl2/src/main.rs` — calls
  `arm7tdmi_aot::compile_rom + enable_aot_on` immediately after
  `GameBoyAdvance::new`, BEFORE `skip_bios` / `frame()` (I11).
  Surfaces compile progress on stderr.
- `platform/rustboyadvance-sdl2/Cargo.toml` — adds `aot` feature
  gating `arm7tdmi-aot` dep.
- `fps_bench/src/main.rs` — same `enable_aot` ordering.

### Untouched

- `arm7tdmi/src/thumb/`, `arm7tdmi/src/arm/` — scalar handlers stay
  source-of-truth and AOT-miss fallback.
- `core/src/gpu/`, `core/src/sound/` — orthogonal SIMD work
  (separate program).

---

## Validation

Three independent gates (V1-V3) all must pass; plus determinism
(V4).

### V1. SDL replay framebuffer-hash diff

Per ROM:

    target/release/rustboyadvance-sdl2 --features aot \
        --aot --no-audio --replay /tmp/<rom>.rec \
        --frame-hash-every 60 --bios <bios> <rom> \
        2>/dev/null | grep "fb_hash" > /tmp/aot_<rom>.txt

    diff <(awk '{print $4}' /tmp/aot_<rom>.txt) \
         <(awk '{print $4}' /tmp/<rom>_scalar_ref.txt) | grep -c '^>'

Zero divs is acceptance criterion. Divs > 0 → revert.

### V2. End-cycle drift gate

`|drift| > 1000` cycles → revert.

### V3. Per-instruction differential tests

Per format, paired test on `Arm7tdmiCore<DiffBus>`:

```rust
#[test]
fn aot_diff_thumb_f3_mov_imm8() {
    let mut scalar_cpu = make_diff_cpu();
    let mut aot_cpu = make_diff_cpu();
    setup_state(&mut scalar_cpu, /* gpr / cpsr / memory */);
    setup_state(&mut aot_cpu, /* same */);

    let opcode = 0x2005; // MOV r0, #5
    scalar_handler_for_thumb(&mut scalar_cpu, opcode);
    aot_emit_and_run(&mut aot_cpu, opcode);

    assert_eq!(scalar_cpu.gpr, aot_cpu.gpr);
    assert_eq!(scalar_cpu.cpsr.get(), aot_cpu.cpsr.get());
    assert_eq!(scalar_cpu.pc, aot_cpu.pc);
    assert_eq!(scalar_cpu.bus.scheduler.cycles,
               aot_cpu.bus.scheduler.cycles);
}
```

Run with:

    LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 \
        cargo test --workspace --features aot aot_diff_

Failure → revert. Coverage: every AOT-supported format × every
input-shape variant (zero-result, sign-bit, carry-out, overflow,
register-clash, misaligned-load, etc.).

### V4. Replay determinism (I24)

Same .rec on same build, run twice, fb-hash sequences must be
identical. Implemented as a step in the harness:

    A_HASH=$(md5sum /tmp/aot_pe_run1.txt | awk '{print $1}')
    B_HASH=$(md5sum /tmp/aot_pe_run2.txt | awk '{print $1}')
    [ "$A_HASH" = "$B_HASH" ] || echo "DETERMINISM_FAIL"

Mismatch → revert.

### gba-tests scope

Per I26, gba-tests must pass under BOTH:
- `cargo test --workspace --features cached_interp`
- `cargo test --workspace --features cached_interp,aot`

In the second case AOT-miss falls to scalar (no real ROM in test
harness). If gba-tests breaks under either, dispatcher's scalar
fallback is broken.

### CI scope

Experiment loop runs **locally only.** GitHub Actions runners
don't have LLVM 18 readily. CI gates only `cargo test --features
cached_interp` (no AOT) — verifies scalar didn't regress. CI also
runs clippy / rustfmt per I27.

---

## Notes for the next agent

- **Set the 15-min reminder loop FIRST.** Without it, work stalls
  at the next compaction.
- **Commit messages: human, imperfect, no Anthropic attribution.**
  Re-read the Operating-rhythm section if unsure.
- Don't merge phases. Each phase is one commit.
- Don't pre-optimize. -O3 once, ship the IR, measure.
- Don't relax I1-I28 without writing the corresponding
  `findings-invariant-<name>.md` first.
- Don't re-introduce per-instruction trampolines under any name.
- If a phase fails twice with different approaches, STOP. Re-plan.
- Phase-0 audits A0-A8 are not optional, not skippable.
- After each accepted phase, dump IR + asm for one example block
  of each newly-supported format using A8 hooks. Save under
  `docs/asm-samples/phaseN-<format>.{ll,s}`. Future regressions
  diff against these.
- Keep `aot-progress.md` up to date; the 15-min reminder loop reads
  it to know where to resume.

---

## Provenance

- Predecessor branch: `shape-opt/apr22` (LLVM JIT trampoline-mode,
  retired). See `docs/llvm-jit-status.md` there for the why.
- Predecessor program: `docs/dynarec-arm64-program.md` (Cranelift
  shape-optimization autoresearch, also Karpathy format).
- Pivot rationale: trampoline-per-instruction dispatch can't beat
  scalar's inline loop. AOT lets the LLVM optimizer see the *whole*
  block as inline IR — the only architecture where the optimizer
  can actually win.
