use std::fmt;

use ansi_term::Style;
use bit::BitIndex;
use log::debug;
use num::FromPrimitive;
use serde::{Deserialize, Serialize};

#[cfg(feature = "debugger")]
use ansi_term::Colour;

use rustboyadvance_utils::{Shared, WeakPointer};

pub use super::exception::Exception;
use super::reg_string;

use super::{Addr, CpuMode, CpuState, arm::ArmCond, psr::RegPSR};

use super::memory::{MemoryAccess, MemoryInterface};
use MemoryAccess::*;

use cfg_if::cfg_if;

#[cfg(feature = "debugger")]
use super::thumb::ThumbFormat;

#[cfg(feature = "debugger")]
use super::arm::ArmFormat;

#[cfg_attr(not(feature = "debugger"), repr(transparent))]
pub struct ThumbInstructionInfo<I: MemoryInterface> {
    pub handler_fn: fn(&mut Arm7tdmiCore<I>, insn: u16) -> CpuAction,
    #[cfg(feature = "debugger")]
    pub fmt: ThumbFormat,
}

#[cfg_attr(not(feature = "debugger"), repr(transparent))]
pub struct ArmInstructionInfo<I: MemoryInterface> {
    pub handler_fn: fn(&mut Arm7tdmiCore<I>, insn: u32) -> CpuAction,
    #[cfg(feature = "debugger")]
    pub fmt: ArmFormat,
}

cfg_if! {
    if #[cfg(feature = "debugger")] {
        use super::DecodedInstruction;
        use super::arm::ArmInstruction;
        use super::thumb::ThumbInstruction;

    } else {

    }
}

pub enum CpuAction {
    AdvancePC(MemoryAccess),
    PipelineFlushed,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct BankedRegisters {
    // r13 and r14 are banked for all modes. System&User mode share them
    pub gpr_banked_r13: [u32; 6],
    pub gpr_banked_r14: [u32; 6],
    // r8-r12 are banked for fiq mode
    pub gpr_banked_old_r8_12: [u32; 5],
    pub gpr_banked_fiq_r8_12: [u32; 5],
    pub spsr_bank: [RegPSR; 6],
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SavedCpuState {
    pub pc: u32,
    pub gpr: [u32; 15],
    next_fetch_access: MemoryAccess,
    pipeline: [u32; 2],

    pub cpsr: RegPSR,
    pub(super) spsr: RegPSR,

    pub(super) banks: BankedRegisters,
}

#[cfg(feature = "debugger")]
#[derive(Default, Clone, Debug)]
pub struct DebuggerState {
    pub last_executed: Option<DecodedInstruction>,
    /// store the gpr before executing an instruction to show diff in the Display impl
    pub gpr_previous: [u32; 15],
    pub breakpoints: Vec<u32>,
    pub verbose: bool,
    pub trace_opcodes: bool,
    pub trace_exceptions: bool,
}

pub struct Arm7tdmiCore<I: MemoryInterface> {
    pub pc: u32,
    pub bus: Shared<I>,

    pub(crate) next_fetch_access: MemoryAccess,
    pub(crate) pipeline: [u32; 2],
    pub gpr: [u32; 15],

    pub cpsr: RegPSR,
    pub spsr: RegPSR,

    pub banks: BankedRegisters,

    /// Hardware breakpoints for use by gdb
    breakpoints: Vec<Addr>,

    /// Deprecated in-house debugger state
    #[cfg(feature = "debugger")]
    pub dbg: DebuggerState,

    /// Block cache for the cached interpreter. Only populated when the
    /// `cached_interp` feature is on; zero-sized otherwise.
    #[cfg(feature = "cached_interp")]
    pub block_cache: super::cache::BlockCache<I>,

    /// AOT dispatch hook (per I18 in docs/aot-llvm-program.md).
    /// Opaque pointer to an AotTable defined in the arm7tdmi-aot
    /// crate; we keep it as raw `*const u8` here so this crate
    /// stays inkwell-free. `enable_aot_hook` (called by the AOT
    /// crate) populates both fields.
    ///
    /// Lookup signature: `fn(table, pc) -> usize`. Returns the
    /// CompiledFn address as a usize (cast back at dispatch site),
    /// or 0 if no AOT block exists at that pc.
    ///
    /// CompiledFn signature (per I8): `extern "C" fn(*mut u8 cpu_ctx,
    /// *mut u32 pc_out) -> u32` returning 0 / 0b01 (branch) / 0b10
    /// (abort).
    #[cfg(feature = "aot_dispatch")]
    pub aot_table: *const u8,
    #[cfg(feature = "aot_dispatch")]
    pub aot_lookup_fn: Option<fn(*const u8, u32) -> usize>,
}

// BlockCache holds handler function pointers keyed by entry-PC; cloning a CPU
// with a populated cache would share those fn pointers (cheap), but the cache
// is a transient runtime optimization and we'd rather not clone it by accident.
// Implement Clone manually so we can reset the cache on clone.
impl<I: MemoryInterface> Clone for Arm7tdmiCore<I> {
    fn clone(&self) -> Self {
        Arm7tdmiCore {
            pc: self.pc,
            bus: self.bus.clone(),
            next_fetch_access: self.next_fetch_access,
            pipeline: self.pipeline,
            gpr: self.gpr,
            cpsr: self.cpsr,
            spsr: self.spsr,
            banks: self.banks.clone(),
            breakpoints: self.breakpoints.clone(),
            #[cfg(feature = "debugger")]
            dbg: self.dbg.clone(),
            #[cfg(feature = "cached_interp")]
            block_cache: super::cache::BlockCache::new(),
            // AOT hook intentionally NOT cloned: cloned CPUs are
            // typically used for tests / save-states which don't
            // share the AOT table state. Re-installing the hook is
            // the caller's responsibility.
            #[cfg(feature = "aot_dispatch")]
            aot_table: std::ptr::null(),
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn: None,
        }
    }
}

impl<I: MemoryInterface> Arm7tdmiCore<I> {
    pub fn new(bus: Shared<I>) -> Arm7tdmiCore<I> {
        let cpsr = RegPSR::new(0x0000_00D3);
        Arm7tdmiCore {
            bus,
            pc: 0,
            gpr: [0; 15],
            pipeline: [0; 2],
            next_fetch_access: MemoryAccess::NonSeq,
            cpsr,
            spsr: Default::default(),
            banks: BankedRegisters::default(),

            breakpoints: Vec::new(),

            #[cfg(feature = "debugger")]
            dbg: DebuggerState::default(),

            #[cfg(feature = "cached_interp")]
            block_cache: super::cache::BlockCache::new(),
            #[cfg(feature = "aot_dispatch")]
            aot_table: std::ptr::null(),
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn: None,
        }
    }

    pub fn weak_ptr(&mut self) -> WeakPointer<Arm7tdmiCore<I>> {
        WeakPointer::new(self as *mut Arm7tdmiCore<I>)
    }

    /// Install the AOT dispatch hook (per I18). The arm7tdmi-aot
    /// crate calls this with its `*const AotTable` (cast to
    /// `*const u8`) and an `aot_lookup` fn that returns the
    /// CompiledFn address as a `usize` (0 if no AOT block exists
    /// at that pc).
    ///
    /// Caller MUST call this before the first `step_block` per I11
    /// — otherwise the cold-start BIOS boot misses the AOT cache
    /// and runs scalar.
    #[cfg(feature = "aot_dispatch")]
    pub fn install_aot_hook(
        &mut self,
        table: *const u8,
        lookup_fn: fn(*const u8, u32) -> usize,
    ) {
        self.aot_table = table;
        self.aot_lookup_fn = Some(lookup_fn);
    }

    /// Try to dispatch an AOT-compiled block at the current pc. Returns
    /// `Some(can_chain)` on hit, `None` on miss (caller falls through
    /// to block_cache + scalar replay).
    ///
    /// Per I8 ABI: CompiledFn returns
    ///   0     → fall-through (caller chains).
    ///   0b01  → branch fired; dispatcher reloads pipeline at *pc_out.
    ///   0b10  → mid-block abort; caller yields to outer run loop.
    ///
    /// Per I15 cold-start: AOT lookup is gated on
    /// `cpu.pipeline[0] != 0` (a zero pipeline means the CPU just
    /// reset and scalar must bootstrap before AOT takes over). See
    /// findings-pipeline-read.md for the rationale.
    #[cfg(feature = "aot_dispatch")]
    #[inline]
    fn try_aot_dispatch(&mut self) -> Option<bool> {
        let lookup = self.aot_lookup_fn?;
        // Cold-start guard (I15): skip AOT until scalar has fetched
        // at least one instruction. This avoids dispatching into an
        // AOT block whose first iter would re-fetch pipeline[0] at
        // a stale pc.
        if self.pipeline[0] == 0 {
            return None;
        }
        let fn_addr = lookup(self.aot_table, self.pc);
        if fn_addr == 0 {
            return None;
        }
        let f: unsafe extern "C" fn(*mut u8, *mut u32) -> u32 =
            unsafe { std::mem::transmute(fn_addr) };
        let cpu_ctx = self as *mut Arm7tdmiCore<I> as *mut u8;
        let mut pc_out: u32 = 0;
        let ret = unsafe { f(cpu_ctx, &mut pc_out) };
        if ret & 0b10 != 0 {
            return Some(false); // mid-block abort, yield
        }
        if ret & 0b01 != 0 {
            // Branch fired. Apply mode + reload pipeline at target.
            let thumb_bit = pc_out & 1 != 0;
            self.cpsr.set_state(if thumb_bit { CpuState::THUMB } else { CpuState::ARM });
            self.pc = pc_out & if thumb_bit { !1 } else { !3 };
            if thumb_bit { self.reload_pipeline16() } else { self.reload_pipeline32() }
        }
        Some(true) // can chain
    }

    pub fn from_saved_state(bus: Shared<I>, state: SavedCpuState) -> Arm7tdmiCore<I> {
        Arm7tdmiCore {
            bus,

            pc: state.pc,
            cpsr: state.cpsr,
            gpr: state.gpr,
            banks: state.banks,
            spsr: state.spsr,

            pipeline: state.pipeline,
            next_fetch_access: state.next_fetch_access,

            breakpoints: Vec::new(), // TODO include breakpoints in saved state

            // savestate does not keep debugger related information, so just reinitialize to default
            #[cfg(feature = "debugger")]
            dbg: DebuggerState::default(),

            #[cfg(feature = "cached_interp")]
            block_cache: super::cache::BlockCache::new(),
            // AOT hook is process-level state, not save-state state.
            // Caller re-installs after restore.
            #[cfg(feature = "aot_dispatch")]
            aot_table: std::ptr::null(),
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn: None,
        }
    }

    pub fn save_state(&self) -> SavedCpuState {
        SavedCpuState {
            cpsr: self.cpsr,
            pc: self.pc,
            gpr: self.gpr,
            spsr: self.spsr,
            banks: self.banks.clone(),
            pipeline: self.pipeline,
            next_fetch_access: self.next_fetch_access,
        }
    }

    pub fn restore_state(&mut self, state: SavedCpuState) {
        self.pc = state.pc;
        self.cpsr = state.cpsr;
        self.gpr = state.gpr;
        self.spsr = state.spsr;
        self.banks = state.banks;
        self.pipeline = state.pipeline;
        self.next_fetch_access = state.next_fetch_access;
    }

    pub fn set_memory_interface(&mut self, i: Shared<I>) {
        self.bus = i;
    }

    pub fn add_breakpoint(&mut self, addr: Addr) {
        debug!("adding breakpoint {:08x}", addr);
        self.breakpoints.push(addr);
    }

    pub fn del_breakpoint(&mut self, addr: Addr) {
        if let Some(pos) = self.breakpoints.iter().position(|x| *x == addr) {
            debug!("deleting breakpoint {:08x}", addr);
            self.breakpoints.remove(pos);
        }
    }

    pub fn check_breakpoint(&self) -> Option<u32> {
        let next_pc = self.get_next_pc();
        for bp in &self.breakpoints {
            if (*bp & !1) == next_pc {
                return Some(*bp);
            }
        }
        None
    }

    #[cfg(feature = "debugger")]
    pub fn set_verbose(&mut self, v: bool) {
        self.dbg.verbose = v;
    }

    pub fn get_reg(&self, r: usize) -> u32 {
        match r {
            0..=14 => self.gpr[r],
            15 => self.pc,
            _ => panic!("invalid register {}", r),
        }
    }

    #[inline]
    /// Gets PC of the currently executed instruction in arm mode
    pub fn pc_arm(&self) -> u32 {
        self.pc.wrapping_sub(8)
    }

    #[inline]
    /// Gets PC of the currently executed instruction in thumb mode
    pub fn pc_thumb(&self) -> u32 {
        self.pc.wrapping_sub(4)
    }

    pub fn get_reg_user(&mut self, r: usize) -> u32 {
        match r {
            0..=7 => self.gpr[r],
            8..=12 => {
                if self.cpsr.mode() == CpuMode::Fiq {
                    self.gpr[r]
                } else {
                    self.banks.gpr_banked_old_r8_12[r - 8]
                }
            }
            13 => self.banks.gpr_banked_r13[0],
            14 => self.banks.gpr_banked_r14[0],
            _ => panic!("invalid register"),
        }
    }

    pub fn set_reg(&mut self, r: usize, val: u32) {
        match r {
            0..=14 => self.gpr[r] = val,
            15 => {
                self.pc = {
                    match self.cpsr.state() {
                        CpuState::THUMB => val & !1,
                        CpuState::ARM => val & !3,
                    }
                }
            }
            _ => panic!("invalid register"),
        }
    }

    pub fn set_reg_user(&mut self, r: usize, val: u32) {
        match r {
            0..=7 => self.gpr[r] = val,
            8..=12 => {
                if self.cpsr.mode() == CpuMode::Fiq {
                    self.gpr[r] = val;
                } else {
                    self.banks.gpr_banked_old_r8_12[r - 8] = val;
                }
            }
            13 => {
                self.banks.gpr_banked_r13[0] = val;
            }
            14 => {
                self.banks.gpr_banked_r14[0] = val;
            }
            _ => panic!("invalid register"),
        }
    }

    pub fn copy_registers(&self) -> [u32; 15] {
        self.gpr
    }

    pub(super) fn change_mode(&mut self, old_mode: CpuMode, new_mode: CpuMode) {
        let new_index = new_mode.bank_index();
        let old_index = old_mode.bank_index();

        if new_index == old_index {
            return;
        }

        let banks = &mut self.banks;

        banks.spsr_bank[old_index] = self.spsr;
        banks.gpr_banked_r13[old_index] = self.gpr[13];
        banks.gpr_banked_r14[old_index] = self.gpr[14];

        self.spsr = banks.spsr_bank[new_index];
        self.gpr[13] = banks.gpr_banked_r13[new_index];
        self.gpr[14] = banks.gpr_banked_r14[new_index];

        if new_mode == CpuMode::Fiq {
            for r in 0..5 {
                banks.gpr_banked_old_r8_12[r] = self.gpr[r + 8];
                self.gpr[r + 8] = banks.gpr_banked_fiq_r8_12[r];
            }
        } else if old_mode == CpuMode::Fiq {
            for r in 0..5 {
                banks.gpr_banked_fiq_r8_12[r] = self.gpr[r + 8];
                self.gpr[r + 8] = banks.gpr_banked_old_r8_12[r];
            }
        }
        self.cpsr.set_mode(new_mode);
    }

    /// Resets the cpu
    pub fn reset(&mut self) {
        self.exception(Exception::Reset, 0);
    }

    pub fn word_size(&self) -> usize {
        match self.cpsr.state() {
            CpuState::ARM => 4,
            CpuState::THUMB => 2,
        }
    }

    pub(super) fn get_required_multipiler_array_cycles(&self, rs: u32) -> usize {
        if rs & 0xff == rs {
            1
        } else if rs & 0xffff == rs {
            2
        } else if rs & 0xffffff == rs {
            3
        } else {
            4
        }
    }

    #[inline(always)]
    pub(super) fn check_arm_cond(&self, cond: ArmCond) -> bool {
        use ArmCond::*;
        match cond {
            Invalid => {
                // TODO - we would normally want to panic here
                false
            }
            EQ => self.cpsr.Z(),
            NE => !self.cpsr.Z(),
            HS => self.cpsr.C(),
            LO => !self.cpsr.C(),
            MI => self.cpsr.N(),
            PL => !self.cpsr.N(),
            VS => self.cpsr.V(),
            VC => !self.cpsr.V(),
            HI => self.cpsr.C() && !self.cpsr.Z(),
            LS => !self.cpsr.C() || self.cpsr.Z(),
            GE => self.cpsr.N() == self.cpsr.V(),
            LT => self.cpsr.N() != self.cpsr.V(),
            GT => !self.cpsr.Z() && (self.cpsr.N() == self.cpsr.V()),
            LE => self.cpsr.Z() || (self.cpsr.N() != self.cpsr.V()),
            AL => true,
        }
    }

    #[cfg(feature = "debugger")]
    fn debugger_record_step(&mut self, d: DecodedInstruction) {
        self.dbg.gpr_previous = self.copy_registers();
        self.dbg.last_executed = Some(d);
    }

    fn step_arm_exec(&mut self, insn: u32) -> CpuAction {
        let hash = (((insn >> 16) & 0xff0) | ((insn >> 4) & 0xf)) as usize;
        let arm_info = &Self::ARM_LUT[hash];
        #[cfg(feature = "debugger")]
        self.debugger_record_step(DecodedInstruction::Arm(ArmInstruction::new(
            insn,
            self.pc.wrapping_sub(8),
            arm_info.fmt,
        )));
        (arm_info.handler_fn)(self, insn)
    }

    fn step_thumb_exec(&mut self, insn: u16) -> CpuAction {
        let thumb_info = &Self::THUMB_LUT[(insn >> 6) as usize];
        #[cfg(feature = "debugger")]
        self.debugger_record_step(DecodedInstruction::Thumb(ThumbInstruction::new(
            insn,
            self.pc.wrapping_sub(4),
            thumb_info.fmt,
        )));
        (thumb_info.handler_fn)(self, insn)
    }

    /// 2S + 1N
    #[inline(always)]
    pub fn reload_pipeline16(&mut self) {
        self.pipeline[0] = self.load_16(self.pc, NonSeq) as u32;
        self.advance_thumb();
        self.pipeline[1] = self.load_16(self.pc, Seq) as u32;
        self.advance_thumb();
        self.next_fetch_access = Seq;
    }

    /// 2S + 1N
    #[inline(always)]
    pub fn reload_pipeline32(&mut self) {
        self.pipeline[0] = self.load_32(self.pc, NonSeq);
        self.advance_arm();
        self.pipeline[1] = self.load_32(self.pc, Seq);
        self.advance_arm();
        self.next_fetch_access = Seq;
    }

    #[inline]
    pub(super) fn advance_thumb(&mut self) {
        self.pc = self.pc.wrapping_add(2)
    }

    #[inline]
    pub(super) fn advance_arm(&mut self) {
        self.pc = self.pc.wrapping_add(4)
    }

    #[inline]
    pub fn get_decoded_opcode(&self) -> u32 {
        self.pipeline[0]
    }

    #[inline]
    pub fn get_prefetched_opcode(&self) -> u32 {
        self.pipeline[1]
    }

    /// Cached-interpreter entry point. Equivalent to `step()` but executes a
    /// whole cached block per call instead of a single instruction.
    ///
    /// On a cache miss at the current PC, run one pipeline step at a time
    /// (identical to `step()` except it also records the resolved handler and
    /// the raw instruction word into the block cache). The recording stops at
    /// the first pipeline flush or after 64 recorded instructions.
    ///
    /// On a cache hit, replay the recorded handlers directly, skipping the
    /// hash+LUT lookup per instruction and the `single_step()` wrapper checks
    /// that the outer loop does today.
    ///
    /// Accuracy: per-instruction memory accesses still go through the bus and
    /// advance the scheduler exactly as before. Scheduler events still fire
    /// between `step_block` calls in the outer loop; worst-case overshoot per
    /// block is bounded by the 64-instruction record cap.
    /// Max number of blocks to chain through in a single `step_block`
    /// call before bailing back to the outer run loop. In the common
    /// case scheduler events or IRQs fire long before this and break
    /// the chain, but having a hard cap prevents a pathological tight
    /// loop with no abort-triggering events from monopolizing the
    /// emulator thread.
    #[cfg(feature = "cached_interp")]
    const CHAIN_MAX_DEPTH: u32 = 16;

    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn step_block(&mut self) {
        // Iterate through contiguous fall-through / pipeline-flush
        // chains so the gba::run → single_step → cpu_step → step_block
        // overhead (~20 ns at the measurement site) is amortized over
        // multiple blocks instead of every block. Each iteration
        // dispatches one block and then checks the same abort
        // conditions the inner scalar replay loop checks (RAM dirty,
        // IRQ/DMA/halt/scheduler) before continuing to the next
        // block. The loop terminates on abort, on a cache miss (fall
        // through to recording), on Thumb/ARM state flip, or on the
        // chain-depth cap.
        for _ in 0..Self::CHAIN_MAX_DEPTH {
            // If any RAM write happened since the last block started,
            // blow the whole cache. Coarse but cheap and correct.
            if self.bus.take_block_cache_dirty() {
                self.block_cache.flush();
                return;
            }

            // AOT fast path (per I18). When a hook is installed and
            // the current pc matches a compiled block, dispatch it
            // directly without going through block_cache. On miss
            // (None), fall through to the normal cache-or-record
            // path.
            #[cfg(feature = "aot_dispatch")]
            if let Some(can_chain) = self.try_aot_dispatch() {
                if !can_chain {
                    return;
                }
                if self.bus.cached_block_should_abort() {
                    return;
                }
                continue;
            }

            let thumb = matches!(self.cpsr.state(), CpuState::THUMB);
            let key = super::cache::BlockKey::new(self.pc, thumb);

            let can_chain = if let Some(block) = self.block_cache.get(key) {
                self.replay_cached_block(&block, thumb)
            } else {
                self.block_cache.begin_record(key);
                self.record_new_block(thumb);
                self.block_cache.finish_record();
                // New recording — return to outer loop once so the
                // scheduler can advance and we don't chain through a
                // just-recorded block on the same tick.
                return;
            };
            if !can_chain {
                return;
            }

            // Inter-block abort check: scalar replay does this every
            // other instruction inside the block; between chained
            // blocks we do it once per block boundary.
            if self.bus.cached_block_should_abort() {
                return;
            }
        }
    }

    /// Replay a cached block. Returns `true` when the caller can
    /// chain into the next block at the updated PC (normal fall-
    /// through or pipeline flush with pipeline already reloaded at
    /// the new target), `false` when it must return to the outer
    /// run loop (abort condition, mode flip, or self-modifying
    /// cache-dirty mid-block).
    #[cfg(feature = "cached_interp")]
    fn replay_cached_block(
        &mut self,
        block: &super::cache::Block<I>,
        entry_thumb: bool,
    ) -> bool {
        use super::cache::DecodedInstr;

        // Always run the first instruction before checking abort conditions —
        // that guarantees forward progress even when (for example) an IRQ is
        // pending but CPU has IRQs disabled, so cpu_interrupt() is a no-op and
        // would otherwise loop us forever between the outer run() while and
        // step_block's early return.
        let mut instr_idx: u32 = 0;

        for instr in &block.instrs {
            // Abort checks run at entry of iterations 1, 3, 5, ... — every
            // other instruction rather than every one. Worst-case latency for
            // servicing an IRQ that became pending inside a block is now ~1
            // extra instruction (~a few cycles), which the GBA's interrupt
            // model tolerates. Scheduler events get the same slack, bounded
            // by block length (capped at 64). gba-tests/nes still passes.
            if instr_idx != 0 && instr_idx & 1 == 1 {
                // If the CPU state flipped ARM<->Thumb since block entry, the
                // rest of this block no longer matches; bail and let the next
                // step_block call re-resolve or re-record.
                if matches!(self.cpsr.state(), CpuState::THUMB) != entry_thumb {
                    return false;
                }

                // If an earlier instr in this block wrote to RAM, remaining
                // cached handlers were resolved from potentially stale memory.
                if self.bus.take_block_cache_dirty() {
                    self.block_cache.flush();
                    return false;
                }

                // Yield to the outer loop at about per-two-instruction
                // granularity (pending IRQ, newly active DMA, halt, scheduler
                // overshoot).
                if self.bus.cached_block_should_abort() {
                    return false;
                }
            }
            instr_idx = instr_idx.wrapping_add(1);

            match *instr {
                DecodedInstr::Arm { raw: expected, handler } => {
                    let pc = self.pc & !3;
                    let fetched_now = self.load_32(pc, self.next_fetch_access);
                    let insn = self.pipeline[0];
                    self.pipeline[0] = self.pipeline[1];
                    self.pipeline[1] = fetched_now;
                    debug_assert_eq!(
                        insn, expected,
                        "cached-block ARM instr mismatch at pc={:#x}: \
                         cached={:#010x} actual={:#010x}",
                        pc.wrapping_sub(8), expected, insn,
                    );
                    let cond = ArmCond::from_u8(insn.bit_range(28..32) as u8)
                        .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
                    if cond != ArmCond::AL && !self.check_arm_cond(cond) {
                        self.advance_arm();
                        self.next_fetch_access = MemoryAccess::NonSeq;
                        continue;
                    }
                    match handler(self, insn) {
                        CpuAction::AdvancePC(access) => {
                            self.next_fetch_access = access;
                            self.advance_arm();
                        }
                        CpuAction::PipelineFlushed => return true,
                    }
                }
                DecodedInstr::Thumb { raw: expected, handler } => {
                    let pc = self.pc & !1;
                    let fetched_now = self.load_16(pc, self.next_fetch_access);
                    let insn = self.pipeline[0];
                    self.pipeline[0] = self.pipeline[1];
                    self.pipeline[1] = fetched_now as u32;
                    debug_assert_eq!(
                        insn as u16, expected,
                        "cached-block Thumb instr mismatch at pc={:#x}: \
                         cached={:#06x} actual={:#06x}",
                        pc.wrapping_sub(4), expected, insn as u16,
                    );
                    match handler(self, insn as u16) {
                        CpuAction::AdvancePC(access) => {
                            self.advance_thumb();
                            self.next_fetch_access = access;
                        }
                        CpuAction::PipelineFlushed => return true,
                    }
                }
            }
        }
        // Block finished normally (all recorded instructions
        // executed without a pipeline flush). Fall-through to the
        // next block at the advanced PC; caller can chain into it.
        true
    }

    #[cfg(feature = "cached_interp")]
    fn record_new_block(&mut self, entry_thumb: bool) {
        use super::cache::DecodedInstr;

        // Same check-every-other-iter policy as replay_cached_block.
        // The 64-cap here must match the cap in BlockCache::record_instr; going
        // past it would just silently drop recordings. Both need to change
        // together if the limit is tuned.
        for instr_idx in 0..64u32 {
            if instr_idx != 0 && instr_idx & 1 == 1 {
                if matches!(self.cpsr.state(), CpuState::THUMB) != entry_thumb {
                    return;
                }

                if self.bus.take_block_cache_dirty() {
                    self.block_cache.abort_record();
                    self.block_cache.flush();
                    return;
                }

                if self.bus.cached_block_should_abort() {
                    return;
                }
            }

            if entry_thumb {
                let pc = self.pc & !1;
                let fetched_now = self.load_16(pc, self.next_fetch_access);
                let insn = self.pipeline[0];
                self.pipeline[0] = self.pipeline[1];
                self.pipeline[1] = fetched_now as u32;
                let handler = Self::THUMB_LUT[(insn >> 6) as usize].handler_fn;
                self.block_cache.record_instr(DecodedInstr::Thumb {
                    raw: insn as u16,
                    handler,
                });
                match handler(self, insn as u16) {
                    CpuAction::AdvancePC(access) => {
                        self.advance_thumb();
                        self.next_fetch_access = access;
                    }
                    CpuAction::PipelineFlushed => return,
                }
            } else {
                let pc = self.pc & !3;
                let fetched_now = self.load_32(pc, self.next_fetch_access);
                let insn = self.pipeline[0];
                self.pipeline[0] = self.pipeline[1];
                self.pipeline[1] = fetched_now;
                let hash = (((insn >> 16) & 0xff0) | ((insn >> 4) & 0xf)) as usize;
                let handler = Self::ARM_LUT[hash].handler_fn;
                self.block_cache.record_instr(DecodedInstr::Arm {
                    raw: insn,
                    handler,
                });
                let cond = ArmCond::from_u8(insn.bit_range(28..32) as u8)
                    .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
                if cond != ArmCond::AL && !self.check_arm_cond(cond) {
                    self.advance_arm();
                    self.next_fetch_access = MemoryAccess::NonSeq;
                    continue;
                }
                match handler(self, insn) {
                    CpuAction::AdvancePC(access) => {
                        self.next_fetch_access = access;
                        self.advance_arm();
                    }
                    CpuAction::PipelineFlushed => return,
                }
            }
        }
    }

    /// Perform a pipeline step
    /// If an instruction was executed in this step, return it.
    #[inline]
    pub fn step(&mut self) {
        match self.cpsr.state() {
            CpuState::ARM => {
                let pc = self.pc & !3;

                let fetched_now = self.load_32(pc, self.next_fetch_access);
                let insn = self.pipeline[0];
                self.pipeline[0] = self.pipeline[1];
                self.pipeline[1] = fetched_now;
                let cond = ArmCond::from_u8(insn.bit_range(28..32) as u8)
                    .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
                if cond != ArmCond::AL && !self.check_arm_cond(cond) {
                    self.advance_arm();
                    self.next_fetch_access = MemoryAccess::NonSeq;
                    return;
                }
                match self.step_arm_exec(insn) {
                    CpuAction::AdvancePC(access) => {
                        self.next_fetch_access = access;
                        self.advance_arm();
                    }
                    CpuAction::PipelineFlushed => {}
                }
            }
            CpuState::THUMB => {
                let pc = self.pc & !1;

                let fetched_now = self.load_16(pc, self.next_fetch_access);
                let insn = self.pipeline[0];
                self.pipeline[0] = self.pipeline[1];
                self.pipeline[1] = fetched_now as u32;
                match self.step_thumb_exec(insn as u16) {
                    CpuAction::AdvancePC(access) => {
                        self.advance_thumb();
                        self.next_fetch_access = access;
                    }
                    CpuAction::PipelineFlushed => {}
                }
            }
        }
    }

    /// Get's the address of the next instruction that is going to be executed
    pub fn get_next_pc(&self) -> Addr {
        let insn_size = self.word_size() as u32;
        self.pc - 2 * insn_size
    }

    pub fn get_cpu_state(&self) -> CpuState {
        self.cpsr.state()
    }
}

impl<I: MemoryInterface> fmt::Debug for Arm7tdmiCore<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ARM7TDMI Core Status:")?;
        writeln!(f, "\tCPSR: {}", self.cpsr)?;
        writeln!(f, "\tGeneral Purpose Registers:")?;
        let reg_normal_style = Style::new().bold();
        let gpr = self.copy_registers();
        for (i, gp) in gpr.iter().enumerate() {
            let mut reg_name = reg_string(i).to_string();
            reg_name.make_ascii_uppercase();
            let entry = format!("\t{:-3} = 0x{:08x}", reg_name, gp);
            write!(
                f,
                "{}{}",
                reg_normal_style.paint(entry),
                if (i + 1) % 4 == 0 { "\n" } else { "" }
            )?;
        }
        let pc = format!("\tPC  = 0x{:08x}", self.get_next_pc());
        writeln!(f, "{}", reg_normal_style.paint(pc))
    }
}

#[cfg(feature = "debugger")]
impl<I: MemoryInterface> fmt::Display for Arm7tdmiCore<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ARM7TDMI Core Status:")?;
        writeln!(f, "\tCPSR: {}", self.cpsr)?;
        writeln!(f, "\tGeneral Purpose Registers:")?;
        let reg_normal_style = Style::new().bold();
        let reg_dirty_style = Colour::Black.bold().on(Colour::Yellow);
        let gpr = self.copy_registers();
        for (i, gp) in gpr.iter().enumerate() {
            let mut reg_name = reg_string(i).to_string();
            reg_name.make_ascii_uppercase();

            let style = if gpr[i] != self.dbg.gpr_previous[i] {
                &reg_dirty_style
            } else {
                &reg_normal_style
            };

            let entry = format!("\t{:-3} = 0x{:08x}", reg_name, gp);

            write!(
                f,
                "{}{}",
                style.paint(entry),
                if (i + 1) % 4 == 0 { "\n" } else { "" }
            )?;
        }
        let pc = format!("\tPC  = 0x{:08x}", self.get_next_pc());
        writeln!(f, "{}", reg_normal_style.paint(pc))
    }
}

include!(concat!(env!("OUT_DIR"), "/arm_lut.rs"));
include!(concat!(env!("OUT_DIR"), "/thumb_lut.rs"));
