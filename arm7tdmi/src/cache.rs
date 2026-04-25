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

/// Fn pointer shape for a JIT-compiled Thumb block. The LLVM backend
/// emits functions matching this signature (gpr_ptr, cpsr_ptr,
/// pc_out_ptr, cpu_ctx). Return value:
///   bit 0 = branch taken (pc_out populated, caller reloads pipeline)
///   bit 1 = mid-block abort (handler already wrote pc/pipeline state)
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
    /// Optional compiled block. Set by `finish_record` when the LLVM
    /// dynarec compiler successfully lowers all the recorded opcodes.
    /// None for ARM blocks (cached_interp scalar fallback) and any
    /// block the LLVM backend rejects.
    #[cfg(feature = "dynarec")]
    pub compiled: Option<CompiledThumbFn>,
}

impl<I: MemoryInterface> Block<I> {
    fn new(entry_pc: u32) -> Self {
        Block {
            instrs: Vec::with_capacity(8),
            entry_pc,
            #[cfg(feature = "dynarec")]
            compiled: None,
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
    /// LLVM-via-inkwell compiler. `finish_record` calls
    /// `compile_thumb_block` on every all-Thumb ROM block; ARM blocks
    /// and any block the backend rejects fall through to the
    /// cached_interp scalar replay path. Set by `enable_dynarec` on
    /// the owning CPU.
    #[cfg(feature = "dynarec")]
    llvm_compiler: Option<crate::dynarec::LlvmCompiler>,
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
            llvm_compiler: None,
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

    /// Install the LLVM-backed dynarec compiler. After this, every
    /// newly recorded all-Thumb ROM block is handed to
    /// `compile_thumb_block` in `finish_record` and the resulting fn
    /// pointer is stashed on the Block for `step_block` to dispatch
    /// to. Call once per CPU after construction.
    #[cfg(feature = "dynarec")]
    pub fn enable_dynarec(&mut self, compiler: crate::dynarec::LlvmCompiler) {
        self.llvm_compiler = Some(compiler);
    }

    /// True if the dynarec compiler has been installed.
    #[cfg(feature = "dynarec")]
    pub fn has_dynarec(&self) -> bool {
        self.llvm_compiler.is_some()
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
            // Recording length cap — prevents runaway recordings on
            // pathological code (a tight loop with no pipeline flush
            // would otherwise trace forever). Raised from 64 to 16864
            // per user directive so `DYNAREC_MAX_BLOCK_LEN = 16834` is
            // actually reachable: previously the 64-cap here was the
            // *real* limit regardless of what MAX was set to.
            if block.instrs.len() < 16864 {
                block.instrs.push(instr);
            }
        }
    }

    /// Finish the current recording and insert it into the appropriate cache
    /// half based on the block's entry region. Called on pipeline flush or
    /// when the block length cap is reached.
    ///
    /// When the dynarec feature is on and the LLVM compiler has been
    /// installed, every all-Thumb ROM block is handed to the backend.
    /// On success the resulting fn pointer is stashed on the Block and
    /// step_block can dispatch to it; on rejection the Block is left
    /// with compiled=None and replays via the interpreter handler loop.
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
            // Only compile ROM blocks. RAM blocks get flushed on every
            // RAM write, so compiling them would burn a codegen pass
            // for a single use. ROM blocks stay warm for the whole run.
            let pc = key.0 & !1;
            if is_rom_address(pc)
                && let Some(c) = self.llvm_compiler.as_mut()
            {
                // Only all-Thumb blocks reach the backend. ARM blocks
                // fall back to cached_interp scalar replay.
                let raws_opt: Option<Vec<u16>> = block
                    .instrs
                    .iter()
                    .map(|i| match i {
                        DecodedInstr::Thumb { raw, .. } => Some(*raw),
                        DecodedInstr::Arm { .. } => None,
                    })
                    .collect();
                if let Some(raws) = raws_opt
                    && let Some(f) = c.compile_thumb_block(
                        &raws,
                        (block.entry_pc & !1).wrapping_sub(4),
                    )
                {
                    let f_typed: CompiledThumbFn = unsafe {
                        std::mem::transmute::<
                            crate::dynarec::CompiledFn,
                            CompiledThumbFn,
                        >(f)
                    };
                    block.compiled = Some(f_typed);
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

    /// Diagnostic: compile-rate stats for the ROM-block half of the
    /// cache. Returns `(total_rom_blocks, with_compiled_fn)`. Used by
    /// benchmarks and diagnostic runs to estimate how much of the
    /// hot code is on the dynarec fast path.
    #[cfg(feature = "dynarec")]
    pub fn compile_stats(&self) -> (usize, usize) {
        let mut compiled = 0;
        for block in self.rom_blocks.values() {
            if block.compiled.is_some() {
                compiled += 1;
            }
        }
        (self.rom_blocks.len(), compiled)
    }
}

