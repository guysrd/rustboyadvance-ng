//! Block cache for the cached interpreter.
//!
//! The baseline interpreter in `cpu.rs` fetches every instruction, hashes it
//! into the 4096-entry ARM LUT or the 1024-entry Thumb LUT, calls the
//! resolved handler function, and returns to the `single_step()` wrapper —
//! which then re-checks bus-master state and IRQ pending. A lot of that is
//! redundant across a run of non-branching instructions.
//!
//! This module stores, per (entry-PC, CPU state) pair, the sequence of raw
//! instruction words and their pre-resolved handler fn pointers that were
//! executed until the pipeline flushed. Re-entry at the same PC executes the
//! recorded sequence with a tight loop that skips the LUT hash and the
//! per-instruction single_step wrapper overhead.
//!
//! Accuracy is preserved because:
//!   * Memory-access cycle costs are paid by the handlers' `load_*`/`store_*`
//!     calls — those still run exactly as before, advancing the scheduler.
//!   * Scheduler events do not fire mid-instruction today (they fire between
//!     `single_step()` calls in `gba.rs::run`), and this module preserves
//!     that by returning to the outer loop after each executed block — which
//!     is at least as often as a pipeline flush.
//!   * Self-modifying code is handled by flushing the cache on any write to
//!     writable memory regions (see `SysBus::write_*`).

use std::rc::Rc;
#[cfg(feature = "dynarec")]
use std::sync::atomic::AtomicUsize;

use rustc_hash::FxHashMap;

use crate::cpu::{Arm7tdmiCore, CpuAction};
use crate::memory::MemoryInterface;

#[cfg(feature = "dynarec")]
use crate::dynarec::DynarecCompiler;

/// Per-block link-time chain slot. Compiled blocks whose tail is a known
/// fall-through (`Tail::Body`) bake in the address of this slot and, on
/// exit, atomically load it. If non-null, the value is a
/// `CompiledThumbFn` pointer for the immediately-following block — the
/// compiled code direct-calls it instead of returning to the dispatcher.
/// Stored as `AtomicUsize` so codegen can treat it as a plain 64-bit
/// load without any type casting concerns. Null until the successor
/// block is itself compiled and the cache links them.
///
/// Heap-allocated behind `Rc` so the waiters table can hold a clone
/// without outliving the owning Block.
#[cfg(feature = "dynarec")]
pub type ChainSlot = AtomicUsize;

#[cfg(feature = "shape_profile")]
use crate::dynarec::shape_profile::{self, ShapeId};

/// Fn pointer shape produced by the unified Thumb mem+branch compile path
/// in the `dynarec` module. Same four args and return value semantics that
/// try_compile_thumb_mem_block_with_branch hands out.
#[cfg(feature = "dynarec")]
pub type CompiledThumbFn =
    extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32;

/// One recorded instruction inside a block.
///
/// Handler fn signatures differ between ARM and Thumb (u32 vs u16 insn word),
/// so we encode the mode in an enum. A uniform `u32`-argument handler would
/// require trampolines; this enum costs one jump-table branch per instruction
/// but keeps the handlers untouched.
#[derive(Clone, Copy)]
pub enum DecodedInstr<I: MemoryInterface> {
    Arm {
        raw: u32,
        handler: fn(&mut Arm7tdmiCore<I>, u32) -> CpuAction,
    },
    Thumb {
        raw: u16,
        handler: fn(&mut Arm7tdmiCore<I>, u16) -> CpuAction,
    },
}

/// Cache key = entry PC with the Thumb bit folded into bit 0.
/// ARM entry PCs are 4-byte aligned so bit 0 is always 0 in ARM mode;
/// Thumb entry PCs are 2-byte aligned so bit 0 is free for us.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey(u32);

impl BlockKey {
    #[inline]
    pub fn new(pc: u32, thumb: bool) -> Self {
        BlockKey(pc | (thumb as u32))
    }
}

/// A recorded straight-line run of instructions, terminated by whatever
/// PipelineFlushed the last time we executed it.
pub struct Block<I: MemoryInterface> {
    pub instrs: Vec<DecodedInstr<I>>,
    /// PC at which this block begins, with the Thumb bit in bit 0. Needed
    /// by the dynarec compiler to fold PC relative branch targets at
    /// codegen time. Populated even when the `dynarec` feature is off so
    /// future diagnostics can use it.
    pub entry_pc: u32,
    /// Optional compiled block. Set by `finish_record` when the dynarec
    /// compiler successfully lowers all the recorded opcodes. None for
    /// blocks that contain any shape the dynarec doesn't yet support.
    #[cfg(feature = "dynarec")]
    pub compiled: Option<CompiledThumbFn>,
    /// Link-time block-chaining slot. Allocated before compile so the
    /// codegen can bake in its address and emit a load-and-tail-call at
    /// the block epilogue. `None` when the block didn't compile, or
    /// when the compiled tail isn't chain-eligible (e.g. the compile
    /// path opted out). Remains `AtomicUsize(0)` until the successor
    /// block is linked in by `finish_record` or by the waiters drain.
    #[cfg(feature = "dynarec")]
    pub chain_slot: Option<Rc<ChainSlot>>,
    /// BlockKey-format key of the block that follows this one on the
    /// fall-through path: `(entry_pc & !1) + 2*len | thumb_bit`. Only
    /// populated when the compiled block's tail is a known fall-through
    /// (`Tail::Body` today). Used by `finish_record` to resolve the
    /// chain target and to register this block with the waiters map.
    #[cfg(feature = "dynarec")]
    pub fallthrough_key: Option<BlockKey>,
    /// Shape classification (only under `shape_profile` feature). Set
    /// alongside `compiled` in `finish_record` so the dispatcher can
    /// `shape_profile::tick` the right counter on every replay.
    /// None when the block didn't compile (falls back to interpreter)
    /// or when the dynarec feature is off.
    #[cfg(feature = "shape_profile")]
    pub shape: Option<ShapeId>,
}

impl<I: MemoryInterface> Block<I> {
    fn new(entry_pc: u32) -> Self {
        Block {
            instrs: Vec::with_capacity(8),
            entry_pc,
            #[cfg(feature = "dynarec")]
            compiled: None,
            #[cfg(feature = "dynarec")]
            chain_slot: None,
            #[cfg(feature = "dynarec")]
            fallthrough_key: None,
            #[cfg(feature = "shape_profile")]
            shape: None,
        }
    }
}

/// Per-CPU block cache.
///
/// Blocks are held behind `Rc` so the executor can clone a handle at block
/// entry and then safely call handlers that may mutate memory (and therefore
/// invalidate the cache) without dangling references.
///
/// The cache is split into two maps keyed by entry-PC region:
///   * `rom_blocks` — blocks whose first instruction lives in BIOS or ROM.
///     These addresses are read-only on a real GBA (and by this emulator's
///     SysBus, which treats writes to BIOS/ROM as no-ops), so no write can
///     invalidate them. Keeping them warm across RAM writes is the main win
///     on games like pokeemerald that hammer IWRAM for state updates.
///   * `ram_blocks` — blocks starting in EWRAM or IWRAM. Flushed whenever
///     any RAM write happens, since self-modifying code (rare but legal)
///     could have overwritten a cached instruction.
///
/// Empty in the default build — the struct and all its methods compile to
/// no-ops unless the `cached_interp` feature is on.
pub struct BlockCache<I: MemoryInterface> {
    rom_blocks: FxHashMap<BlockKey, Rc<Block<I>>>,
    ram_blocks: FxHashMap<BlockKey, Rc<Block<I>>>,
    /// Block currently being recorded. `None` when not in a recording pass.
    recording: Option<(BlockKey, Block<I>)>,
    /// Dynarec compiler used to attempt lowering each newly recorded Thumb
    /// block to native code in `finish_record`. None until
    /// `enable_dynarec` is called, which the CPU constructor does when the
    /// dynarec feature is on.
    #[cfg(feature = "dynarec")]
    compiler: Option<DynarecCompiler>,
    /// Link-time block-chaining waiters. Keyed by the target block's
    /// `BlockKey`; each entry is the list of chain slots belonging to
    /// already-compiled blocks that fall through to this target and
    /// are waiting for the target to be compiled. When `finish_record`
    /// lands a new compiled block at some key K, it drains
    /// `waiters[K]`, storing its own compiled fn pointer into every
    /// slot — that retroactively links every predecessor that was
    /// parked here. ROM-only (we never compile RAM blocks), so
    /// waiters survive `flush()` but are cleared by `flush_all()`.
    #[cfg(feature = "dynarec")]
    waiters: FxHashMap<BlockKey, Vec<Rc<ChainSlot>>>,
}

/// Try to JIT compile the recorded block's Thumb instructions. Bails out
/// (returns None) for any block that isn't all Thumb, or whose shapes the
/// dynarec doesn't yet support.
/// Minimum block length to attempt compilation. Short blocks (just a
/// handful of instructions) don't amortize the fixed dispatch overhead
/// of a compiled block entry: fetch_n trampoline call, cpsr
/// load/store, extern C call through a fn pointer. Empirical: at 1
/// instruction per block the dynarec is ~25% slower than the cached
/// interpreter on pokeemerald; at >=4 instructions it pulls even or
/// ahead. Gate compilation until we cross that line.
/// Set to 1 per user directive (2026-04-24): reach high compilation
/// rate first, optimize per-dispatch speed later. This compiles
/// every block the back-end shape-decoder supports, regardless of
/// length. Expect initial fps regression (compiled dispatch is
/// ~3× slower than interp today) but a much larger surface for
/// follow-up codegen optimizations to claim.
#[cfg(feature = "dynarec")]
const DYNAREC_MIN_BLOCK_LEN: usize = 1;

/// Maximum block length to attempt compilation. Longer blocks suffer
/// from abort-latency: scalar `replay_cached_block` checks for IRQ /
/// DMA / RAM-dirty every other instruction and can bail out; a compiled
/// block runs to completion. On Mario Kart this caused measurable
/// framebuffer divergence from scalar that scaled with block length.
///
/// History of this constant:
///   - 4: the conservative setting that kept fb-hash parity with
///     scalar but made 0% of real game-code blocks compilable on
///     either pokeemerald or MK.
///   - 16834 (effectively uncapped, as `record_instr`'s 64-cap and
///     pipeline flushes limit actual length; sweep at 8/16/32/64
///     shows compile rate saturates around 16 on MK): 267 MK blocks
///     compile, 7 pokeemerald blocks compile. Costs 2 MK frame-hash
///     regressions (124 → 126 diverging frames vs scalar) — a drift
///     caused by scheduler-events that scalar aborts on mid-block
///     but compiled runs through. Pokeemerald stays at 0 divergences
///     at any cap. Tunable via `DYNAREC_DEBUG=max=N` for bisection.
#[cfg(feature = "dynarec")]
const DYNAREC_MAX_BLOCK_LEN: usize = 16834;

/// Debug knob for bisecting dynarec correctness bugs. Reads the env
/// var once at first use and caches the parse. Supported values:
///
///   DYNAREC_DEBUG=off           disable ALL dynarec compilation
///                               (effectively cached_interp-only)
///   DYNAREC_DEBUG=max=N         compile blocks only if length <= N
///   DYNAREC_DEBUG=no-mem        skip compilation of blocks containing
///                               any memory op (F7/F8/F9/F10/F11/F14)
///   DYNAREC_DEBUG=no-branch     skip blocks whose last instr is Bcc/B/BL/BX
///   DYNAREC_DEBUG=no-dp-long    skip blocks with 6+ pure-DP instructions
///   DYNAREC_DEBUG=no-chain      compile blocks as before but don't emit
///                               the link-time chain check in epilogues.
///                               Keeps chain-slot allocation + linking
///                               alive so the difference is a pure codegen
///                               bisect (chaining-on vs chaining-off).
///
/// Multiple tokens can be combined with commas (evaluated as AND):
///   DYNAREC_DEBUG=no-mem,no-branch
///
/// Unset (default): compile everything the unified compiler supports.
/// Used to narrow down which compile path miscompiles pokeemerald.
#[cfg(feature = "dynarec")]
#[derive(Default, Debug)]
struct DynarecDebug {
    off: bool,
    no_mem: bool,
    no_branch: bool,
    no_cond_branch: bool, // skip only format-16 Bcc
    no_uncond_branch: bool, // skip only format-18 B
    no_bx: bool, // skip only format-5 BX
    no_pop_pc: bool, // skip only POP{PC}
    no_dp_long: bool,
    no_chain: bool,
    max_len: Option<usize>,
}

#[cfg(feature = "dynarec")]
fn dynarec_debug() -> &'static DynarecDebug {
    use std::sync::OnceLock;
    static CELL: OnceLock<DynarecDebug> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut d = DynarecDebug::default();
        if let Ok(s) = std::env::var("DYNAREC_DEBUG") {
            for tok in s.split(',').map(str::trim) {
                match tok {
                    "off" => d.off = true,
                    "no-mem" => d.no_mem = true,
                    "no-branch" => d.no_branch = true,
                    "no-cond-branch" => d.no_cond_branch = true,
                    "no-uncond-branch" => d.no_uncond_branch = true,
                    "no-bx" => d.no_bx = true,
                    "no-pop-pc" => d.no_pop_pc = true,
                    "no-dp-long" => d.no_dp_long = true,
                    "no-chain" => d.no_chain = true,
                    t if t.starts_with("max=") => {
                        if let Ok(n) = t[4..].parse() {
                            d.max_len = Some(n);
                        }
                    }
                    "" => {}
                    other => eprintln!("DYNAREC_DEBUG: unknown token {:?}", other),
                }
            }
            eprintln!("DYNAREC_DEBUG active: {:?}", d);
        }
        d
    })
}

/// Compile result including the block's chain slot (for link-time
/// chaining). The chain slot is always allocated when compilation
/// succeeds UNLESS `DYNAREC_DEBUG=no-chain` is set, in which case
/// codegen skips the chain check and the slot is `None`.
#[cfg(feature = "dynarec")]
struct CompileResult {
    func: CompiledThumbFn,
    chain_slot: Option<Rc<ChainSlot>>,
}

#[cfg(feature = "dynarec")]
fn try_compile_thumb<I: MemoryInterface>(
    compiler: &mut DynarecCompiler,
    block: &Block<I>,
) -> Option<CompileResult> {
    if !compiler.has_bus() {
        return None;
    }
    let dbg = dynarec_debug();
    if dbg.off {
        return None;
    }
    if block.instrs.len() < DYNAREC_MIN_BLOCK_LEN {
        return None;
    }
    if block.instrs.len() > DYNAREC_MAX_BLOCK_LEN {
        return None;
    }
    if let Some(max) = dbg.max_len
        && block.instrs.len() > max
    {
        return None;
    }
    let mut raws: Vec<u16> = Vec::with_capacity(block.instrs.len());
    for instr in &block.instrs {
        match instr {
            DecodedInstr::Thumb { raw, .. } => raws.push(*raw),
            DecodedInstr::Arm { .. } => return None,
        }
    }
    if raws.is_empty() {
        return None;
    }
    // Reject ONLY dynamic-target branch terminators (BX, POP{PC},
    // BL pair). Their targets come from runtime values so can't be
    // statically chained. Bcc (format 16) and B (format 18) have
    // compile-time-known targets and now have chain emission on
    // both their taken and fall-through paths — compiling them
    // unlocks chain links that were impossible under the old
    // blanket branch-filter.
    if let Some(&last) = raws.last() {
        let top4 = last >> 12;
        // BX: 0100_0111_xxxx_xxxx (format 5 BX subset).
        if (last & 0xFF00) == 0x4700 {
            return None;
        }
        // POP{PC}: 1011_1101_xxxx_xxxx (format 14 with R=1).
        if (last & 0xFF00) == 0xBD00 {
            return None;
        }
        // BL pair (format 19): top4 == 0b1111. First half sets LR,
        // second half branches. Either half is unsupported as a tail
        // terminator in the current compiler.
        if top4 == 0b1111 {
            return None;
        }
    }
    // Debug knobs: skip blocks whose shape matches a suspect classifier.
    if dbg.no_mem && raws.iter().any(|&op| is_thumb_mem_opcode(op)) {
        return None;
    }
    if dbg.no_branch
        && raws
            .last()
            .map(|&op| is_thumb_branch_opcode(op))
            .unwrap_or(false)
    {
        return None;
    }
    if let Some(&last) = raws.last() {
        let top4 = last >> 12;
        // Bcc: 0b1101 xxxx (format 16). Also masks out SWI 0b1101_1111 at runtime,
        // but classify_tail handles that.
        if dbg.no_cond_branch && top4 == 0b1101 {
            return None;
        }
        // Unconditional B (format 18): 0b11100 xxx (mask 0xF800 == 0xE000)
        if dbg.no_uncond_branch && (last & 0xF800) == 0xE000 {
            return None;
        }
        // BX (format 5): 0b0100_0111_xxxx
        if dbg.no_bx && (last & 0xFF00) == 0x4700 {
            return None;
        }
        // POP{PC}: 0b1011_1101_xxxx_xxxx (format 14 with R=1 and PC set)
        if dbg.no_pop_pc && (last & 0xFF00) == 0xBD00 {
            return None;
        }
    }
    if dbg.no_dp_long
        && raws.len() >= 6
        && !raws.iter().any(|&op| is_thumb_mem_opcode(op))
        && !raws
            .last()
            .map(|&op| is_thumb_branch_opcode(op))
            .unwrap_or(false)
    {
        return None;
    }
    // block.entry_pc is self.pc at step_block entry with the Thumb bit
    // OR'd into bit 0. Masking off that bit gives the pipeline-head pc,
    // which equals block_start_addr + 4 per Thumb pipeline convention.
    // The dynarec compile API takes entry_pc = block_start_addr (what
    // the unit tests use), so subtract 4 to match.
    let block_start_addr = (block.entry_pc & !1).wrapping_sub(4);
    // Allocate the chain slot up front so its stable address can be
    // baked into the generated code. `no-chain` debug knob disables
    // chain-slot allocation entirely; codegen then emits the original
    // dispatcher-return epilogue.
    let chain_slot: Option<Rc<ChainSlot>> = if dbg.no_chain {
        None
    } else {
        Some(Rc::new(AtomicUsize::new(0)))
    };
    let func = compiler.try_compile_thumb_mem_block_with_branch(
        &raws,
        block_start_addr,
        chain_slot.as_deref(),
    )?;
    Some(CompileResult { func, chain_slot })
}

/// Cheap detectors that mirror the block-level classifier rules in
/// `dynarec::shape_profile`. Kept here so the DYNAREC_DEBUG gate can
/// filter without pulling in shape_profile (which is an independent
/// feature and not always compiled in).
#[cfg(feature = "dynarec")]
fn is_thumb_mem_opcode(op: u16) -> bool {
    let top4 = op >> 12;
    match top4 {
        0b0101 | 0b0110 | 0b0111 | 0b1000 | 0b1001 => true,
        0b1011 => (op & 0x0600) == 0x0400,
        _ => false,
    }
}

#[cfg(feature = "dynarec")]
fn is_thumb_branch_opcode(op: u16) -> bool {
    let top4 = op >> 12;
    match top4 {
        0b1101 => true,
        0b1110 => (op & 0xF800) == 0xE000,
        0b1111 => true,
        0b0100 => (op & 0xFF00) == 0x4700,
        _ => false,
    }
}

/// Classify a guest PC by memory region for the block-cache split.
/// See GBA memory map in resources/gbatek: BIOS is 0x0000_0000-0x0000_3FFF,
/// ROM (WS0/1/2) at 0x0800_0000-0x0DFF_FFFF.
#[inline]
fn is_rom_address(pc: u32) -> bool {
    let top = pc & 0xff00_0000;
    // BIOS range (top byte 0x00) or any cartridge wait-state region.
    top == 0x0000_0000 || (0x0800_0000..=0x0d00_0000).contains(&top)
}

impl<I: MemoryInterface> Default for BlockCache<I> {
    fn default() -> Self {
        BlockCache {
            rom_blocks: FxHashMap::with_capacity_and_hasher(2048, Default::default()),
            ram_blocks: FxHashMap::with_capacity_and_hasher(256, Default::default()),
            recording: None,
            #[cfg(feature = "dynarec")]
            compiler: None,
            #[cfg(feature = "dynarec")]
            waiters: FxHashMap::default(),
        }
    }
}

impl<I: MemoryInterface> BlockCache<I> {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn get(&self, key: BlockKey) -> Option<Rc<Block<I>>> {
        let pc = key.0 & !1;
        if is_rom_address(pc) {
            self.rom_blocks.get(&key).cloned()
        } else {
            self.ram_blocks.get(&key).cloned()
        }
    }

    /// Called on any RAM write to invalidate RAM-region blocks. ROM-region
    /// blocks are left alone — no write can reach that memory anyway.
    #[inline]
    pub fn flush(&mut self) {
        self.ram_blocks.clear();
        // If we happen to be mid-recording a RAM-region block, drop it too.
        // ROM recordings stay valid.
        if let Some((key, _)) = &self.recording
            && !is_rom_address(key.0 & !1)
        {
            self.recording = None;
        }
    }

    /// Flush everything (ROM and RAM). Used only by diagnostics/tests; normal
    /// invalidation goes through `flush()` which preserves ROM entries.
    #[allow(dead_code)]
    #[inline]
    pub fn flush_all(&mut self) {
        self.rom_blocks.clear();
        self.ram_blocks.clear();
        self.recording = None;
        // Chain waiters reference chain slots inside the now-cleared
        // ROM blocks. The Rc keeps the slots alive, but there's no
        // point holding onto them — nothing will drain them now the
        // blocks they reference are gone.
        #[cfg(feature = "dynarec")]
        self.waiters.clear();
    }

    /// Install a Cranelift-backed dynarec compiler. After this, any
    /// newly recorded Thumb block that matches a supported shape will be
    /// JIT compiled by `finish_record` and its fn pointer stashed on the
    /// Block for `step_block` to dispatch to. Call once per CPU after
    /// construction (the CPU knows how to build a trampoline struct
    /// with its concrete MemoryInterface impl).
    #[cfg(feature = "dynarec")]
    pub fn enable_dynarec(&mut self, compiler: DynarecCompiler) {
        self.compiler = Some(compiler);
    }

    /// True if a dynarec compiler has been installed.
    #[cfg(feature = "dynarec")]
    pub fn has_dynarec(&self) -> bool {
        self.compiler.is_some()
    }

    /// Begin trace-recording a new block starting at `key`.
    #[inline]
    pub fn begin_record(&mut self, key: BlockKey) {
        self.recording = Some((key, Block::new(key.0)));
    }

    /// Append one executed instruction to the in-progress recording, if any.
    #[inline]
    pub fn record_instr(&mut self, instr: DecodedInstr<I>) {
        if let Some((_, block)) = &mut self.recording {
            // Cap block length to avoid runaway recordings on pathological code
            // paths (long loops that never pipeline-flush would just trace forever).
            // 64 is arbitrary but comfortably larger than typical basic blocks.
            if block.instrs.len() < 64 {
                block.instrs.push(instr);
            }
        }
    }

    /// Finish the current recording and insert it into the appropriate cache
    /// half based on the block's entry region. Called on pipeline flush or
    /// when the block length cap is reached.
    ///
    /// When the dynarec feature is on and a compiler has been installed,
    /// this also tries to JIT compile the recorded block into a native fn.
    /// On compile success the fn pointer is stashed on the Block and
    /// step_block can dispatch to it; on compile failure (any unsupported
    /// shape) the Block is left with compiled=None and replays via the
    /// interpreter handler loop.
    #[inline]
    pub fn finish_record(&mut self) {
        if let Some((_key, block)) = self.recording.as_ref()
            && block.instrs.is_empty()
        {
            self.recording = None;
            return;
        }

        #[cfg(feature = "dynarec")]
        let Some((key, mut block)) = self.recording.take() else {
            return;
        };
        #[cfg(not(feature = "dynarec"))]
        let Some((key, block)) = self.recording.take() else {
            return;
        };

        #[cfg(feature = "dynarec")]
        {
            // Only compile ROM blocks. RAM blocks get flushed on every RAM
            // write, so compilation would burn a Cranelift codegen pass
            // for a single use. ROM blocks stay warm for the whole run.
            let pc = key.0 & !1;
            if is_rom_address(pc)
                && let Some(compiler) = self.compiler.as_mut()
                && let Some(result) = try_compile_thumb(compiler, &block)
            {
                block.compiled = Some(result.func);
                block.chain_slot = result.chain_slot;
                // Fall-through key: (entry_pc + 2*len) with the Thumb
                // bit preserved — the block-cache lookup key for the
                // next block on the sequential path.
                let len_bytes = (block.instrs.len() as u32).wrapping_mul(2);
                let fallthrough_pc = (block.entry_pc & !1).wrapping_add(len_bytes);
                block.fallthrough_key = Some(BlockKey::new(fallthrough_pc, true));
                // Tag the block with its shape category so the dispatcher
                // can `shape_profile::tick` the right counter on every
                // invocation. Only meaningful when the compile succeeded
                // — interpreter-path blocks don't have a bench-comparable
                // shape.
                #[cfg(feature = "shape_profile")]
                {
                    let raws: Vec<u16> = block
                        .instrs
                        .iter()
                        .filter_map(|i| match i {
                            DecodedInstr::Thumb { raw, .. } => Some(*raw),
                            DecodedInstr::Arm { .. } => None,
                        })
                        .collect();
                    block.shape = Some(shape_profile::classify_thumb(&raws));
                }
            }
        }

        #[cfg(feature = "dynarec")]
        let block_compiled_fn = block.compiled;
        #[cfg(feature = "dynarec")]
        let block_fallthrough = block.fallthrough_key;
        #[cfg(feature = "dynarec")]
        let block_chain_slot = block.chain_slot.clone();

        let pc = key.0 & !1;
        if is_rom_address(pc) {
            self.rom_blocks.insert(key, Rc::new(block));
        } else {
            self.ram_blocks.insert(key, Rc::new(block));
        }

        // Link-time chaining bookkeeping. Only meaningful under
        // dynarec and only when we actually compiled the block we
        // just inserted.
        #[cfg(feature = "dynarec")]
        if is_rom_address(pc)
            && let (Some(compiled), Some(chain_slot)) =
                (block_compiled_fn, block_chain_slot.as_ref())
        {
            // (a) Try to link THIS block's chain slot to its
            // fallthrough target if the target is already
            // compiled. Otherwise park the slot on the waiters
            // list for that target.
            if let Some(ft_key) = block_fallthrough {
                if let Some(target_block) = self.rom_blocks.get(&ft_key) {
                    if let Some(target_fn) = target_block.compiled {
                        chain_slot.store(
                            target_fn as usize,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    // Target exists but didn't compile: slot stays null
                    // forever. No point parking in waiters — the target
                    // won't ever transition to compiled (rom blocks
                    // are recorded once).
                } else {
                    // Target block not recorded yet. Park our slot so
                    // that when the target eventually compiles, it can
                    // retroactively link us.
                    self.waiters
                        .entry(ft_key)
                        .or_default()
                        .push(Rc::clone(chain_slot));
                }
            }

            // (b) Drain any waiters that were parked against
            // THIS block's key — retroactively link their chain
            // slots to our compiled fn. The key is `key` (the
            // BlockKey we inserted under), which carries the
            // Thumb bit in bit 0.
            if let Some(mut parked) = self.waiters.remove(&key) {
                let compiled_addr = compiled as usize;
                for slot in parked.drain(..) {
                    slot.store(compiled_addr, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// Drop the in-progress recording without saving. Used when we bail out of
    /// a partial recording (e.g. we were mid-block when the caller decided to
    /// invalidate).
    #[allow(dead_code)]
    #[inline]
    pub fn abort_record(&mut self) {
        self.recording = None;
    }

    #[inline]
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.rom_blocks.len() + self.ram_blocks.len()
    }

    /// Diagnostic: compile-rate stats for the ROM-block half of the
    /// cache. Returns `(total_rom_blocks, with_compiled_fn,
    /// with_chain_slot_linked)`.  A block counts as "chain-linked"
    /// when its chain slot has been populated with a non-null
    /// target — i.e. a real fall-through successor was compiled.
    /// Used by benchmarks / diagnostic runs to estimate how much of
    /// the hot code is on the dynarec fast path.
    #[cfg(feature = "dynarec")]
    pub fn compile_stats(&self) -> (usize, usize, usize) {
        let mut compiled = 0;
        let mut linked = 0;
        for block in self.rom_blocks.values() {
            if block.compiled.is_some() {
                compiled += 1;
            }
            if let Some(slot) = &block.chain_slot
                && slot.load(std::sync::atomic::Ordering::Relaxed) != 0
            {
                linked += 1;
            }
        }
        (self.rom_blocks.len(), compiled, linked)
    }
}

#[cfg(all(test, feature = "dynarec"))]
mod tests {
    use super::*;
    use crate::dynarec::{DynarecCompiler, trampolines};
    use crate::SimpleMemory;

    fn thumb(raw: u16, handler: fn(&mut Arm7tdmiCore<SimpleMemory>, u16) -> CpuAction)
        -> DecodedInstr<SimpleMemory>
    {
        DecodedInstr::Thumb { raw, handler }
    }

    fn stub_thumb_handler(_cpu: &mut Arm7tdmiCore<SimpleMemory>, _insn: u16) -> CpuAction {
        CpuAction::AdvancePC(crate::memory::MemoryAccess::Seq)
    }

    #[test]
    fn block_carries_compiled_fn_when_dynarec_wired() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );
        assert!(cache.has_dynarec());

        // Record a block of DYNAREC_MIN_BLOCK_LEN Thumb instructions
        // at a ROM address so it gets a slot in rom_blocks AND exceeds
        // the minimum-length gate for compilation.
        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        for _ in 0..super::DYNAREC_MIN_BLOCK_LEN {
            cache.record_instr(thumb(0x2005, stub_thumb_handler));
        }
        cache.finish_record();

        let block = cache.get(key).expect("block in cache");
        assert!(block.compiled.is_some(),
                "dynarec should have compiled this supported shape");
    }

    #[test]
    fn empty_block_not_compiled() {
        // Below DYNAREC_MIN_BLOCK_LEN guard when MIN > len. With
        // MIN=1 (post 2026-04-24 coverage push) any non-empty block
        // can compile, so the test that used to assert "1-instr
        // block doesn't compile" is no longer meaningful. Empty
        // blocks still can't compile (they'd have no body to emit).
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );
        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        // No record_instr calls — empty block.
        cache.finish_record();
        // Empty blocks aren't inserted at all by finish_record.
        assert!(cache.get(key).is_none(), "empty block should not be stored");
    }

    #[test]
    fn block_compiled_is_none_when_dynarec_not_wired() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        assert!(!cache.has_dynarec());

        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        cache.record_instr(thumb(0x2005, stub_thumb_handler));
        cache.finish_record();

        let block = cache.get(key).expect("block in cache");
        assert!(block.compiled.is_none());
    }

    #[test]
    fn block_compile_bails_on_mixed_arm_thumb() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        fn stub_arm_handler(_cpu: &mut Arm7tdmiCore<SimpleMemory>, _insn: u32) -> CpuAction {
            CpuAction::AdvancePC(crate::memory::MemoryAccess::Seq)
        }

        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        cache.record_instr(thumb(0x2005, stub_thumb_handler));
        cache.record_instr(DecodedInstr::Arm {
            raw: 0xE3A0_0001,
            handler: stub_arm_handler,
        });
        cache.finish_record();

        let block = cache.get(key).expect("block in cache");
        assert!(block.compiled.is_none(), "mixed ARM/Thumb rejects");
    }

    #[test]
    fn block_compile_bails_on_unsupported_thumb_shape() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        // Thumb format 6 PC-relative LDR: 0b0100_1ddd_iiiiiiii. 0x4900
        // is LDR R1, [PC, #0]. Not yet supported by the dynarec.
        cache.record_instr(thumb(0x4900, stub_thumb_handler));
        cache.finish_record();

        let block = cache.get(key).expect("block in cache");
        assert!(block.compiled.is_none(), "unsupported shape -> no compile");
    }

    /// Record a compiled Thumb block of 4 stub MOVs at the given
    /// `entry_pc`. 4 is an arbitrary len that's compilable at the
    /// currently-shipped MIN/MAX bounds and keeps the fall-through
    /// arithmetic explicit (target = entry_pc + 8).
    const STUB_BLOCK_LEN: usize = 4;
    fn record_stub_block_at(cache: &mut BlockCache<SimpleMemory>, entry_pc: u32) {
        let key = BlockKey::new(entry_pc, true);
        cache.begin_record(key);
        for _ in 0..STUB_BLOCK_LEN {
            cache.record_instr(thumb(0x2005, stub_thumb_handler));
        }
        cache.finish_record();
    }

    /// Eager-link case: block A is recorded + compiled first, its
    /// chain slot ends up parked in the waiters map against the key
    /// of its fall-through target. When block B is later compiled
    /// and inserted at that key, the waiter drain populates A's
    /// chain slot with B's compiled-fn pointer.
    #[test]
    fn chaining_waiter_drain_links_predecessor_to_successor() {
        use std::sync::atomic::Ordering;
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        // Block A at 0x0800_0000 × 4 Thumb instrs (8 bytes) →
        // fallthrough_key points at 0x0800_0008.
        record_stub_block_at(&mut cache, 0x0800_0000);
        // Block B lives at A's fallthrough. Compiling B should drain
        // A from the waiters map.
        record_stub_block_at(&mut cache, 0x0800_0008);

        let block_a = cache.get(BlockKey::new(0x0800_0000, true)).expect("A");
        let block_b = cache.get(BlockKey::new(0x0800_0008, true)).expect("B");

        let a_chain = block_a.chain_slot.as_ref().expect("A has slot");
        let b_fn = block_b.compiled.expect("B compiled") as usize;
        assert_eq!(
            a_chain.load(Ordering::Relaxed),
            b_fn,
            "waiter drain should link A's chain slot to B's compiled fn",
        );
    }

    /// Immediate-link case: block B is recorded first. When A is
    /// recorded afterwards, its fall-through lookup finds B already
    /// compiled and links the chain slot synchronously without going
    /// through the waiters map.
    #[test]
    fn chaining_immediate_link_when_target_compiled_first() {
        use std::sync::atomic::Ordering;
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        record_stub_block_at(&mut cache, 0x0800_0008);
        record_stub_block_at(&mut cache, 0x0800_0000);

        let block_a = cache.get(BlockKey::new(0x0800_0000, true)).expect("A");
        let block_b = cache.get(BlockKey::new(0x0800_0008, true)).expect("B");

        let a_chain = block_a.chain_slot.as_ref().expect("A has slot");
        let b_fn = block_b.compiled.expect("B compiled") as usize;
        assert_eq!(
            a_chain.load(Ordering::Relaxed),
            b_fn,
            "eager link should set A's chain slot to B's compiled fn",
        );
    }

    /// RAM-region blocks never compile (they'd be flushed on every
    /// RAM write, so burning a Cranelift codegen pass for a single
    /// use is pointless). A block recorded at a RAM-region PC
    /// therefore ends up with compiled=None, chain_slot=None,
    /// fallthrough_key=None — no compiled-fn pointer to chain to,
    /// and no way for a predecessor to chain into it.
    #[test]
    fn chaining_skips_ram_region_blocks() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );
        // 0x0300_0000 is IWRAM — RAM region, not ROM.
        let key = BlockKey::new(0x0300_0000, true);
        cache.begin_record(key);
        for _ in 0..DYNAREC_MIN_BLOCK_LEN {
            cache.record_instr(thumb(0x2005, stub_thumb_handler));
        }
        cache.finish_record();
        let block = cache.get(key).expect("RAM block in cache");
        assert!(block.compiled.is_none(), "RAM blocks never compile");
        assert!(block.chain_slot.is_none(), "RAM blocks never get a chain slot");
        assert!(block.fallthrough_key.is_none(), "RAM blocks never get a fallthrough key");
    }

    /// `flush_all` is a diagnostic path that blows away the whole
    /// cache. The waiters map holds Rc<ChainSlot> clones against
    /// ROM blocks that got cleared; those waiter entries are now
    /// orphaned and must also be cleared so a future block at the
    /// same target PC doesn't try to link to the now-dropped
    /// predecessor.
    #[test]
    fn chaining_flush_all_clears_waiters() {
        use std::sync::atomic::Ordering;
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        // Record a block whose fall-through target doesn't exist
        // yet — this parks the block's chain slot in the waiters
        // map.
        record_stub_block_at(&mut cache, 0x0800_0000);
        let block_a = cache.get(BlockKey::new(0x0800_0000, true)).expect("A");
        let a_chain_pre = block_a.chain_slot.as_ref().expect("A has slot");
        assert_eq!(
            a_chain_pre.load(Ordering::Relaxed),
            0,
            "target not recorded yet, chain stays null",
        );
        drop(block_a);

        // Flush everything. Waiters entries must be cleared too.
        cache.flush_all();

        // Now record the target block at the fall-through key.
        // Since the predecessor is gone AND its waiter entry was
        // cleared, nothing retroactively links to the new block.
        record_stub_block_at(&mut cache, 0x0800_0008);
        let block_b = cache.get(BlockKey::new(0x0800_0008, true)).expect("B");
        assert!(block_b.compiled.is_some(), "B still compiles normally");
        // A was flushed; confirm it's gone.
        assert!(
            cache.get(BlockKey::new(0x0800_0000, true)).is_none(),
            "flush_all should have removed A",
        );
    }
}
