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

use rustc_hash::FxHashMap;

use crate::cpu::{Arm7tdmiCore, CpuAction};
use crate::memory::MemoryInterface;

#[cfg(feature = "dynarec")]
use crate::dynarec::DynarecCompiler;

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
#[cfg(feature = "dynarec")]
const DYNAREC_MIN_BLOCK_LEN: usize = 4;

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
    no_dp_long: bool,
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
                    "no-dp-long" => d.no_dp_long = true,
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

#[cfg(feature = "dynarec")]
fn try_compile_thumb<I: MemoryInterface>(
    compiler: &mut DynarecCompiler,
    block: &Block<I>,
) -> Option<CompiledThumbFn> {
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
    if let Some(max) = dbg.max_len {
        if block.instrs.len() > max {
            return None;
        }
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
    compiler.try_compile_thumb_mem_block_with_branch(&raws, block_start_addr)
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
            {
                block.compiled = try_compile_thumb(compiler, &block);
                // Tag the block with its shape category so the dispatcher
                // can `shape_profile::tick` the right counter on every
                // invocation. Only meaningful when the compile succeeded
                // — interpreter-path blocks don't have a bench-comparable
                // shape.
                #[cfg(feature = "shape_profile")]
                if block.compiled.is_some() {
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

        let pc = key.0 & !1;
        if is_rom_address(pc) {
            self.rom_blocks.insert(key, Rc::new(block));
        } else {
            self.ram_blocks.insert(key, Rc::new(block));
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
    fn block_not_compiled_when_below_min_length() {
        let mut cache: BlockCache<SimpleMemory> = BlockCache::new();
        cache.enable_dynarec(
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>()),
        );

        // Single instruction block, below DYNAREC_MIN_BLOCK_LEN.
        let key = BlockKey::new(0x0800_0000, true);
        cache.begin_record(key);
        cache.record_instr(thumb(0x2005, stub_thumb_handler));
        cache.finish_record();

        let block = cache.get(key).expect("block in cache");
        assert!(block.compiled.is_none(),
                "short block should not be compiled (amortization gate)");
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
}
