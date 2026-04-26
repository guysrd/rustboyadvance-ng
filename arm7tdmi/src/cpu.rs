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
use super::registers_consts::{REG_LR, REG_PC, REG_SP};
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

/// No-op AOT lookup stub. Returns 0 always. The default for
/// `aot_lookup_fn` so the dispatcher can call it unconditionally
/// without an Option discriminator branch. Replaced by the real
/// lookup fn when AOT is enabled via `install_aot_hook`.
#[cfg(feature = "aot_dispatch")]
pub fn aot_lookup_noop(_table: *const u8, _pc: u32) -> usize {
    0
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
    /// AOT lookup hot-path fn ptr. Initialized to a no-op stub that
    /// always returns 0 so `try_aot_dispatch` can call it
    /// unconditionally without an `Option` discriminator branch.
    /// Replaced by the real lookup fn (set by `enable_aot_hook`)
    /// when AOT is enabled. Saves ~0.5ns per dispatch vs Option check.
    #[cfg(feature = "aot_dispatch")]
    pub aot_lookup_fn: fn(*const u8, u32) -> usize,
    /// Phase-8 ARM-mode lookup. Dispatcher uses this when cpsr=ARM.
    /// Defaults to the no-op stub (returns 0) so try_aot_dispatch can
    /// call it unconditionally.
    #[cfg(feature = "aot_dispatch")]
    pub aot_lookup_fn_arm: fn(*const u8, u32) -> usize,
    /// Per-mode coverage counters bumped from `step_block`. The
    /// un-segmented total is just `thumb + arm` — recovered at print
    /// time, no extra runtime add per dispatch.
    #[cfg(feature = "aot_dispatch")]
    pub aot_dispatch_hits_thumb: u64,
    #[cfg(feature = "aot_dispatch")]
    pub aot_dispatch_hits_arm: u64,
    #[cfg(feature = "aot_dispatch")]
    pub aot_dispatch_misses_thumb: u64,
    #[cfg(feature = "aot_dispatch")]
    pub aot_dispatch_misses_arm: u64,
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
            aot_lookup_fn: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn_arm: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_arm: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_arm: 0,
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
            aot_lookup_fn: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn_arm: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_arm: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_arm: 0,
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
        self.aot_lookup_fn = lookup_fn;
    }

    /// Phase-8: install ARM-mode AOT lookup hook. Called alongside
    /// `install_aot_hook` (which sets the Thumb hook + table ptr).
    /// The same `aot_table` is used by both lookups.
    #[cfg(feature = "aot_dispatch")]
    pub fn install_aot_hook_arm(
        &mut self,
        lookup_fn: fn(*const u8, u32) -> usize,
    ) {
        self.aot_lookup_fn_arm = lookup_fn;
    }

    /// AOT-side helper: per-iter Thumb step (fetch + pipeline shift +
    /// THUMB_LUT handler dispatch + AdvancePC bookkeeping). Mirrors
    /// what scalar `replay_cached_block` does for one iteration.
    /// Phase-4 cpu state offsets for the AOT inline-IR emit.
    /// next_fetch_access + pipeline are `pub(crate)` so external
    /// crates can't use `std::mem::offset_of!` directly — exposing
    /// the offsets via this fn keeps the field visibility narrow
    /// while letting the AOT compiler bake them as constants.
    /// Returns (pc, gpr, cpsr, next_fetch_access, pipeline).
    pub fn aot_field_offsets() -> (usize, usize, usize, usize, usize) {
        (
            std::mem::offset_of!(Arm7tdmiCore<I>, pc),
            std::mem::offset_of!(Arm7tdmiCore<I>, gpr),
            std::mem::offset_of!(Arm7tdmiCore<I>, cpsr),
            std::mem::offset_of!(Arm7tdmiCore<I>, next_fetch_access),
            std::mem::offset_of!(Arm7tdmiCore<I>, pipeline),
        )
    }

    /// arm7tdmi-aot's phase-0 trampoline calls this once per opcode.
    ///
    /// Returns:
    ///   0 → AdvancePC (continue to next instruction).
    ///   1 → PipelineFlushed (handler updated cpu.pc + cpu.pipeline
    ///       at branch target; caller exits the block).
    ///
    /// Per A1 audit: no Thumb handler reads `pipeline[]`, so the
    /// `read_16` fetch is for cycle accounting only. Phase 0 keeps
    /// the actual load to match scalar exactly; phase 4 (per the
    /// ladder) inlines `add_cycles` and elides the load.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_thumb_step(&mut self, fetch_addr: u32, insn: u32) -> u32 {
        let access = self.next_fetch_access;
        let val = self.load_16(fetch_addr, access);
        self.pipeline[0] = self.pipeline[1];
        self.pipeline[1] = val as u32;
        // Pipeline-head pc convention: scalar's self.pc at handler-call
        // time IS `fetch_addr` (= exec_addr + 4). After AdvancePC
        // scalar advances pc by 2 → exec_addr + 6 = fetch_addr + 2.
        self.pc = fetch_addr;

        // Phase-1 Rust-level inline fast paths: skip THUMB_LUT + indirect
        // handler call for opcodes the AOT path explicitly knows how to
        // execute. Saves ~3-5ns per inlined instr (one indirect call).
        // Each fast path is bit-exact with the corresponding scalar
        // handler in arm7tdmi/src/thumb/exec.rs (verified by SDL replay
        // diff). Add formats here as their bit-exact equivalent is
        // ported.
        let top3 = (insn >> 13) & 0x7;
        if top3 == 0b000 {
            // F1 MoveShiftedReg (LSL/LSR/ASR Rd, Rs, #imm5) — bits 15:13 = 000
            // AND bits 12:11 != 0b11 (the 0b11 case is F2 AddSub).
            // Encoding: 000_oo_IIIII_SSS_DDD where oo ∈ {LSL=0, LSR=1, ASR=2}.
            // Mirrors thumb/exec.rs::exec_thumb_move_shifted_reg.
            let bs_op_bits = ((insn >> 11) & 0x3) as u8;
            if bs_op_bits != 0b11 {
                let imm = ((insn >> 6) & 0x1f) as u32;
                let rs = ((insn >> 3) & 0x7) as usize;
                let rd = (insn & 0x7) as usize;
                let mut carry = self.cpsr.C();
                let bsop = match bs_op_bits {
                    0 => crate::BarrelShiftOpCode::LSL,
                    1 => crate::BarrelShiftOpCode::LSR,
                    2 => crate::BarrelShiftOpCode::ASR,
                    _ => unsafe { std::hint::unreachable_unchecked() },
                };
                let op2 = self.barrel_shift_op(bsop, self.gpr[rs], imm, &mut carry, true);
                self.gpr[rd] = op2;
                self.alu_update_flags(op2, false, carry, self.cpsr.V());
                self.next_fetch_access = MemoryAccess::Seq;
                self.pc = fetch_addr.wrapping_add(2);
                return 0; // AdvancePC
            }
            // F2 AddSub — bits 15:11 = 00011 (the 0b11 case above).
            // Encoding: 00011_I_S_NNN_SSS_DDD where:
            //   I=bit 10 (1 → imm3), S=bit 9 (1 → SUB), NNN=bits 8:6 (Rn or imm3).
            // Mirrors thumb/exec.rs::exec_thumb_add_sub.
            let sub = (insn >> 9) & 0x1 != 0;
            let imm_flag = (insn >> 10) & 0x1 != 0;
            let rn_or_imm = ((insn >> 6) & 0x7) as u32;
            let rs = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let op1 = self.gpr[rs];
            let op2 = if imm_flag { rn_or_imm } else { self.gpr[rn_or_imm as usize] };
            let mut carry = self.cpsr.C();
            let mut overflow = self.cpsr.V();
            let result = if sub {
                self.alu_sub_flags(op1, op2, &mut carry, &mut overflow)
            } else {
                self.alu_add_flags(op1, op2, &mut carry, &mut overflow)
            };
            self.alu_update_flags(result, true, carry, overflow);
            self.gpr[rd] = result;
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        if top3 == 0b001 {
            // F3 MOV/CMP/ADD/SUB Rd, #imm8 — bits 15:13 = 0b001.
            // Encoding: 001_oo_RRR_IIIIIIII (oo selects op).
            //   00 MOV: Rd = imm8        (bit-exact with exec_thumb_data_process_imm<0,RD>)
            //   01 CMP: temp = Rd - imm8 (no writeback) (op=1)
            //   10 ADD: Rd = Rd + imm8                  (op=2)
            //   11 SUB: Rd = Rd - imm8                  (op=3)
            // Mirrors thumb/exec.rs::exec_thumb_data_process_imm — same
            // helpers (alu_add_flags / alu_sub_flags / alu_update_flags).
            let op = ((insn >> 11) & 0x3) as u8;
            let rd = ((insn >> 8) & 0x7) as usize;
            let imm = (insn & 0xff) as u32;
            let op1 = self.gpr[rd];
            let mut carry = self.cpsr.C();
            let mut overflow = self.cpsr.V();
            let result = match op {
                0 => imm,                                                       // MOV
                1 | 3 => self.alu_sub_flags(op1, imm, &mut carry, &mut overflow), // CMP / SUB
                2 => self.alu_add_flags(op1, imm, &mut carry, &mut overflow),   // ADD
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            let arithmetic = op == 2 || op == 3; // ADD / SUB
            self.alu_update_flags(result, arithmetic, carry, overflow);
            if op != 1 {
                self.gpr[rd] = result; // skip writeback for CMP
            }
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F4 ALU ops — raw & 0xfc00 == 0x4000 (top6 = 010000).
        // Encoding: 010000_OOOO_SSS_DDD; OOOO indexes ThumbAluOps.
        // Mirrors thumb/exec.rs::exec_thumb_alu_ops.
        if (insn & 0xfc00) == 0x4000 {
            let op = ((insn >> 6) & 0xf) as u8;
            let rs = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let dst = self.gpr[rd];
            let src = self.gpr[rs];
            let mut carry = self.cpsr.C();
            let mut overflow = self.cpsr.V();
            // Helper: shift-by-register pattern with one idle cycle (used by
            // LSL/LSR/ASR/ROR variants of F4).
            let result = match op {
                0b0000 | 0b1000 => dst & src,                                       // AND / TST
                0b0001 => dst ^ src,                                                // EOR
                0b0010 => {                                                          // LSL
                    let r = self.shift_by_register(crate::BarrelShiftOpCode::LSL, rd, rs, &mut carry);
                    self.idle_cycle();
                    r
                }
                0b0011 => {                                                          // LSR
                    let r = self.shift_by_register(crate::BarrelShiftOpCode::LSR, rd, rs, &mut carry);
                    self.idle_cycle();
                    r
                }
                0b0100 => {                                                          // ASR
                    let r = self.shift_by_register(crate::BarrelShiftOpCode::ASR, rd, rs, &mut carry);
                    self.idle_cycle();
                    r
                }
                0b0111 => {                                                          // ROR
                    let r = self.shift_by_register(crate::BarrelShiftOpCode::ROR, rd, rs, &mut carry);
                    self.idle_cycle();
                    r
                }
                0b0101 => self.alu_adc_flags(dst, src, &mut carry, &mut overflow),  // ADC
                0b0110 => self.alu_sbc_flags(dst, src, &mut carry, &mut overflow),  // SBC
                0b1001 => self.alu_sub_flags(0, src, &mut carry, &mut overflow),    // NEG
                0b1010 => self.alu_sub_flags(dst, src, &mut carry, &mut overflow),  // CMP
                0b1011 => self.alu_add_flags(dst, src, &mut carry, &mut overflow),  // CMN
                0b1100 => dst | src,                                                 // ORR
                0b1101 => {                                                          // MUL
                    let m = self.get_required_multipiler_array_cycles(src);
                    for _ in 0..m {
                        self.idle_cycle();
                    }
                    carry = false;
                    overflow = false;
                    dst.wrapping_mul(src)
                }
                0b1110 => dst & (!src),                                              // BIC
                0b1111 => !src,                                                       // MVN
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            // is_arithmetic: ADC / SBC / NEG / CMP / CMN.
            let arithmetic = matches!(op, 0b0101 | 0b0110 | 0b1001 | 0b1010 | 0b1011);
            self.alu_update_flags(result, arithmetic, carry, overflow);
            // is_setting_flags (no writeback): TST / CMP / CMN.
            let setting_flags = matches!(op, 0b1000 | 0b1010 | 0b1011);
            if !setting_flags {
                self.gpr[rd] = result;
            }
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F5 HiRegOpOrBranchExchange — raw & 0xfc00 == 0x4400.
        // Encoding: 010001_OO_H1_H2_SSS_DDD where OO ∈ {ADD=0, CMP=1, MOV=2, BX=3}.
        // Mirrors thumb/exec.rs::exec_thumb_hi_reg_op_or_bx. PipelineFlushed
        // cases (BX always; ADD/MOV with Rd=R15) return 1 — same as scalar.
        if (insn & 0xfc00) == 0x4400 {
            let op = ((insn >> 8) & 0x3) as u8;
            let h1 = ((insn >> 7) & 0x1) as usize;
            let h2 = ((insn >> 6) & 0x1) as usize;
            let rs_low = ((insn >> 3) & 0x7) as usize;
            let rd_low = (insn & 0x7) as usize;
            let dst_reg = if h1 == 1 { rd_low + 8 } else { rd_low };
            let src_reg = if h2 == 1 { rs_low + 8 } else { rs_low };
            // Per pc_thumb / get_reg semantics: at this point self.pc =
            // fetch_addr (pipeline-head pc), so get_reg(15) returns the
            // pipeline-head pc which is what scalar's handler also sees.
            if op == 3 {
                // BX — always pipeline flush.
                self.branch_exchange(self.get_reg(src_reg));
                return 1; // PipelineFlushed
            }
            let op1 = self.get_reg(dst_reg);
            let op2 = self.get_reg(src_reg);
            match op {
                0 => {
                    // ADD
                    self.set_reg(dst_reg, op1.wrapping_add(op2));
                    if dst_reg == REG_PC {
                        self.reload_pipeline16();
                        return 1; // PipelineFlushed
                    }
                }
                1 => {
                    // CMP
                    let mut carry = self.cpsr.C();
                    let mut overflow = self.cpsr.V();
                    let result = self.alu_sub_flags(op1, op2, &mut carry, &mut overflow);
                    self.alu_update_flags(result, true, carry, overflow);
                }
                2 => {
                    // MOV
                    self.set_reg(dst_reg, op2);
                    if dst_reg == REG_PC {
                        self.reload_pipeline16();
                        return 1; // PipelineFlushed
                    }
                }
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F6 LDR PC-relative (load from literal pool) — raw & 0xf800 == 0x4800.
        // Encoding: 01001_DDD_IIIIIIII (Rd, word8 = imm << 2).
        // Mirrors thumb/exec.rs::exec_thumb_ldr_pc which uses (pc & !3) + ofs
        // where pc is pipeline-head (= exec_addr + 4).
        if (insn & 0xf800) == 0x4800 {
            let rd = ((insn >> 8) & 0x7) as usize;
            let imm = ((insn & 0xff) << 2) as u32;
            let addr = (self.pc & !3).wrapping_add(imm);
            self.gpr[rd] = self.ldr_word(addr, MemoryAccess::NonSeq);
            self.idle_cycle();
            // F6 returns AdvancePC(NonSeq), not Seq.
            self.next_fetch_access = MemoryAccess::NonSeq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F7 LDR/STR with reg offset — raw & 0xf200 == 0x5000.
        // Encoding: 0101_LB_0_OOO_BBB_DDD; L=bit 11 (load), B=bit 10 (byte), O=Ro.
        // addr = gpr[Rb] + gpr[Ro]. Mirrors thumb/exec.rs::exec_thumb_ldr_str_reg_offset.
        if (insn & 0xf200) == 0x5000 {
            let load = (insn >> 11) & 0x1 != 0;
            let byte = (insn >> 10) & 0x1 != 0;
            let ro = ((insn >> 6) & 0x7) as usize;
            let rb = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let addr = self.gpr[rb].wrapping_add(self.gpr[ro]);
            if load {
                let data = if byte {
                    self.load_8(addr, MemoryAccess::NonSeq) as u32
                } else {
                    self.ldr_word(addr, MemoryAccess::NonSeq)
                };
                self.gpr[rd] = data;
                self.idle_cycle();
                self.next_fetch_access = MemoryAccess::Seq;
            } else {
                let value = self.gpr[rd];
                if byte {
                    self.store_8(addr, value as u8, MemoryAccess::NonSeq);
                } else {
                    self.store_aligned_32(addr, value, MemoryAccess::NonSeq);
                }
                self.next_fetch_access = MemoryAccess::NonSeq;
            }
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F8 LDR/STR sign-extended/halfword reg-offset — raw & 0xf200 == 0x5200.
        // Encoding: 0101_HS_1_OOO_BBB_DDD; H=bit 11 (halfword), S=bit 10 (sign-ext).
        //   (S,H) = (0,0) STRH, (0,1) LDRH, (1,0) LDSB, (1,1) LDSH
        // Mirrors thumb/exec.rs::exec_thumb_ldr_str_shb. Always returns
        // AdvancePC(NonSeq).
        if (insn & 0xf200) == 0x5200 {
            let halfword = (insn >> 11) & 0x1 != 0;
            let sign_extend = (insn >> 10) & 0x1 != 0;
            let ro = ((insn >> 6) & 0x7) as usize;
            let rb = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let addr = self.gpr[rb].wrapping_add(self.gpr[ro]);
            match (sign_extend, halfword) {
                (false, false) => {
                    // STRH
                    self.store_aligned_16(addr, self.gpr[rd] as u16, MemoryAccess::NonSeq);
                }
                (false, true) => {
                    // LDRH
                    self.gpr[rd] = self.ldr_half(addr, MemoryAccess::NonSeq);
                    self.idle_cycle();
                }
                (true, false) => {
                    // LDSB — load_8 then sign-extend i8 → i32 → u32
                    let val = self.load_8(addr, MemoryAccess::NonSeq) as i8 as i32 as u32;
                    self.gpr[rd] = val;
                    self.idle_cycle();
                }
                (true, true) => {
                    // LDSH
                    self.gpr[rd] = self.ldr_sign_half(addr, MemoryAccess::NonSeq);
                    self.idle_cycle();
                }
            }
            self.next_fetch_access = MemoryAccess::NonSeq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F10 LDRH/STRH with imm5*2 offset — raw & 0xf000 == 0x8000.
        // Encoding: 1000_L_IIIII_BBB_DDD; L=bit 11 (load).
        // offset = imm5 << 1. Mirrors thumb/exec.rs::exec_thumb_ldr_str_halfword.
        if (insn & 0xf000) == 0x8000 {
            let load = (insn >> 11) & 0x1 != 0;
            let imm5 = ((insn >> 6) & 0x1f) as i32;
            let rb = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let base = self.gpr[rb] as i32;
            let addr = base.wrapping_add(imm5 << 1) as u32;
            if load {
                let data = self.ldr_half(addr, MemoryAccess::NonSeq);
                self.idle_cycle();
                self.gpr[rd] = data;
                self.next_fetch_access = MemoryAccess::Seq;
            } else {
                self.store_aligned_16(addr, self.gpr[rd] as u16, MemoryAccess::NonSeq);
                self.next_fetch_access = MemoryAccess::NonSeq;
            }
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F9 LDR/STR with imm5 offset — raw & 0xe000 == 0x6000.
        // Encoding: 011_BL_IIIII_BBB_DDD; B=bit 12 (1 → byte), L=bit 11 (1 → load).
        // Mirrors thumb/exec.rs::exec_thumb_ldr_str_imm_offset → do_exec_thumb_ldr_str.
        if (insn & 0xe000) == 0x6000 {
            let byte = (insn >> 12) & 0x1 != 0;
            let load = (insn >> 11) & 0x1 != 0;
            let imm5 = ((insn >> 6) & 0x1f) as u32;
            let rb = ((insn >> 3) & 0x7) as usize;
            let rd = (insn & 0x7) as usize;
            let offset = if byte { imm5 } else { imm5 << 2 };
            let addr = self.gpr[rb].wrapping_add(offset);
            if load {
                let data = if byte {
                    self.load_8(addr, MemoryAccess::NonSeq) as u32
                } else {
                    self.ldr_word(addr, MemoryAccess::NonSeq)
                };
                self.gpr[rd] = data;
                self.idle_cycle();
                self.next_fetch_access = MemoryAccess::Seq;
            } else {
                let value = self.gpr[rd];
                if byte {
                    self.store_8(addr, value as u8, MemoryAccess::NonSeq);
                } else {
                    self.store_aligned_32(addr, value, MemoryAccess::NonSeq);
                }
                self.next_fetch_access = MemoryAccess::NonSeq;
            }
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F11 LDR/STR SP-relative (word) — raw & 0xf000 == 0x9000.
        // Encoding: 1001_L_DDD_IIIIIIII; L=bit 11 (1 → load).
        // word8 = imm << 2. Mirrors thumb/exec.rs::exec_thumb_ldr_str_sp.
        if (insn & 0xf000) == 0x9000 {
            let load = (insn >> 11) & 0x1 != 0;
            let rd = ((insn >> 8) & 0x7) as usize;
            let word8 = ((insn & 0xff) << 2) as u32;
            let addr = self.gpr[REG_SP].wrapping_add(word8);
            if load {
                let data = self.ldr_word(addr, MemoryAccess::NonSeq);
                self.idle_cycle();
                self.gpr[rd] = data;
                self.next_fetch_access = MemoryAccess::Seq;
            } else {
                self.store_aligned_32(addr, self.gpr[rd], MemoryAccess::NonSeq);
                self.next_fetch_access = MemoryAccess::NonSeq;
            }
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F12 LoadAddress (ADD Rd, [PC|SP], #imm8) — raw & 0xf000 == 0xa000.
        // Encoding: 1010_S_DDD_IIIIIIII; S=bit 11 (1 → SP, 0 → PC).
        // Mirrors thumb/exec.rs::exec_thumb_load_address.
        if (insn & 0xf000) == 0xa000 {
            let sp = (insn >> 11) & 0x1 != 0;
            let rd = ((insn >> 8) & 0x7) as usize;
            let imm = ((insn & 0xff) << 2) as u32; // word8: imm << 2
            // self.pc here is fetch_addr = exec_addr + 4. pc_thumb() = pc - 4.
            // Per scalar: (pc_thumb() & !2) + 4 + imm = ((fetch_addr - 4) & !2) + 4 + imm.
            let val = if sp {
                self.gpr[REG_SP].wrapping_add(imm)
            } else {
                ((self.pc.wrapping_sub(4)) & !0b10).wrapping_add(4).wrapping_add(imm)
            };
            self.gpr[rd] = val;
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F13 AddSp (ADD/SUB SP, #imm7<<2) — raw & 0xff00 == 0xb000.
        // Encoding: 10110000_S_IIIIIII; S=bit 7 (1 → SUB).
        // Mirrors thumb/exec.rs::exec_thumb_add_sp.
        if (insn & 0xff00) == 0xb000 {
            let sub = (insn >> 7) & 0x1 != 0;
            let offset = ((insn & 0x7f) << 2) as i32;
            let sp = self.gpr[REG_SP] as i32;
            self.gpr[REG_SP] = if sub {
                sp.wrapping_sub(offset) as u32
            } else {
                sp.wrapping_add(offset) as u32
            };
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC
        }

        // F15 LDM/STM — raw & 0xf000 == 0xc000.
        // Encoding: 1100_L_BBB_RRRRRRRR; L=bit 11 (load), B=Rb (bits 10:8).
        // Mirrors thumb/exec.rs::exec_thumb_ldm_stm. Always returns
        // AdvancePC(NonSeq) unless empty-rlist LDM (which flushes pipeline).
        if (insn & 0xf000) == 0xc000 {
            let load = (insn >> 11) & 0x1 != 0;
            let rb = ((insn >> 8) & 0x7) as usize;
            let rlist = (insn & 0xff) as u8;
            let align_preserve = self.gpr[rb] & 3;
            let mut addr = self.gpr[rb] & !3;
            if rlist != 0 {
                if load {
                    let mut access = MemoryAccess::NonSeq;
                    for r in 0..8 {
                        if (rlist >> r) & 1 != 0 {
                            let val = self.load_32(addr, access);
                            access = MemoryAccess::Seq;
                            addr = addr.wrapping_add(4);
                            self.gpr[r] = val;
                        }
                    }
                    self.idle_cycle();
                    if (rlist >> rb) & 1 == 0 {
                        self.gpr[rb] = addr.wrapping_add(align_preserve);
                    }
                } else {
                    let mut first = true;
                    let mut access = MemoryAccess::NonSeq;
                    let count = (rlist.count_ones() as u32).wrapping_sub(1);
                    for r in 0..8 {
                        if (rlist >> r) & 1 != 0 {
                            let v = if r != rb {
                                self.gpr[r]
                            } else if first {
                                addr
                            } else {
                                addr.wrapping_add(count.wrapping_mul(4))
                            };
                            self.store_32(addr, v, access);
                            access = MemoryAccess::Seq;
                            addr = addr.wrapping_add(4);
                            first = false;
                        }
                        // Mirrors scalar's quirky "set rb every iter" pattern.
                        self.gpr[rb] = addr.wrapping_add(align_preserve);
                    }
                }
            } else {
                // Empty rlist edge case (GBATEK ARMv4 quirk):
                // LDM empty: loads PC from addr; STM empty: stores PC+2 at addr.
                // Both: rb += 0x40.
                if load {
                    let val = self.load_32(addr, MemoryAccess::NonSeq);
                    self.pc = val & !1;
                    self.reload_pipeline16();
                    addr = addr.wrapping_add(0x40);
                    self.gpr[rb] = addr.wrapping_add(align_preserve);
                    return 1; // PipelineFlushed
                } else {
                    self.store_32(addr, self.pc.wrapping_add(2), MemoryAccess::NonSeq);
                    addr = addr.wrapping_add(0x40);
                    self.gpr[rb] = addr.wrapping_add(align_preserve);
                }
            }
            self.next_fetch_access = MemoryAccess::NonSeq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC(NonSeq)
        }

        // F14 PUSH/POP — raw & 0xf600 == 0xb400.
        // Encoding: 1011_L_10_R_RRRRRRRR; L=bit 11 (1=POP), R=bit 8 (LR/PC).
        // Mirrors thumb/exec.rs::exec_thumb_push_pop.
        if (insn & 0xf600) == 0xb400 {
            let pop = (insn >> 11) & 0x1 != 0;
            let flag_r = (insn >> 8) & 0x1 != 0;
            let rlist = (insn & 0xff) as u8;
            if pop {
                let mut access = MemoryAccess::NonSeq;
                for r in 0..8 {
                    if (rlist >> r) & 1 != 0 {
                        let stack_addr = self.gpr[REG_SP] & !3;
                        self.gpr[r] = self.load_32(stack_addr, access);
                        access = MemoryAccess::Seq;
                        self.gpr[REG_SP] = self.gpr[REG_SP].wrapping_add(4);
                    }
                }
                if flag_r {
                    // pop! 1-arg in scalar uses Seq.
                    let stack_addr = self.gpr[REG_SP] & !3;
                    let val = self.load_32(stack_addr, MemoryAccess::Seq);
                    self.set_reg(REG_PC, val);
                    self.gpr[REG_SP] = self.gpr[REG_SP].wrapping_add(4);
                    self.pc &= !1;
                    self.reload_pipeline16();
                    self.idle_cycle();
                    return 1; // PipelineFlushed
                }
                self.idle_cycle();
                self.next_fetch_access = MemoryAccess::NonSeq;
                self.pc = fetch_addr.wrapping_add(2);
                return 0; // AdvancePC(NonSeq)
            } else {
                // PUSH
                let mut access = MemoryAccess::NonSeq;
                if flag_r {
                    self.gpr[REG_SP] = self.gpr[REG_SP].wrapping_sub(4);
                    let stack_addr = self.gpr[REG_SP] & !3;
                    self.store_32(stack_addr, self.gpr[REG_LR], access);
                    access = MemoryAccess::Seq;
                }
                for r in (0..8).rev() {
                    if (rlist >> r) & 1 != 0 {
                        self.gpr[REG_SP] = self.gpr[REG_SP].wrapping_sub(4);
                        let stack_addr = self.gpr[REG_SP] & !3;
                        self.store_32(stack_addr, self.gpr[r], access);
                        access = MemoryAccess::Seq;
                    }
                }
                self.next_fetch_access = MemoryAccess::NonSeq;
                self.pc = fetch_addr.wrapping_add(2);
                return 0; // AdvancePC(NonSeq)
            }
        }

        // F16 Bcc — raw & 0xf000 == 0xd000.
        // Encoding: 1101_CCCC_IIIIIIII where CCCC=cond, IIIIIIII signed imm8.
        // SWI (cond=0xF) and undefined (cond=0xE) handled via LUT below.
        // Mirrors thumb/exec.rs::exec_thumb_branch_with_cond.
        if (insn & 0xf000) == 0xd000 {
            let cond = ((insn >> 8) & 0xf) as u8;
            if cond < 0xe {
                let cond_enum = match num::FromPrimitive::from_u8(cond) {
                    Some(c) => c,
                    None => unsafe { std::hint::unreachable_unchecked() },
                };
                if !self.check_arm_cond(cond_enum) {
                    // Not taken — AdvancePC(Seq).
                    self.next_fetch_access = MemoryAccess::Seq;
                    self.pc = fetch_addr.wrapping_add(2);
                    return 0;
                }
                // Taken — same offset math as bcond_offset(): sign-extend
                // the 8-bit imm to 32 bits then shift left by 1 so it's a
                // halfword offset.
                let offset = ((((insn & 0xff) as u32) << 24) as i32) >> 23;
                self.pc = (self.pc as i32).wrapping_add(offset) as u32;
                self.reload_pipeline16();
                return 1; // PipelineFlushed
            }
            // SWI / undefined — fall through to LUT.
        }

        // F19 hi (top5=11110) — Linear part of BL pair. Sets gpr[LR].
        // Mirrors thumb/exec.rs::exec_thumb_branch_long_with_link<false>.
        // F19 lo (top5=11111) is the terminator and goes through LUT
        // for now (handler does reload_pipeline16 + flag setup).
        if (insn >> 11) == 0b11110 {
            // off = (insn.offset11() << 21) >> 9 (sign-extend 11-bit to 32-bit then << 12)
            let off = (((insn & 0x7ff) as u32) << 21) as i32 >> 9;
            self.gpr[REG_LR] = (self.pc as i32).wrapping_add(off) as u32;
            self.next_fetch_access = MemoryAccess::Seq;
            self.pc = fetch_addr.wrapping_add(2);
            return 0; // AdvancePC(Seq)
        }

        // Fallback: LUT + handler dispatch (unsupported format).
        let thumb_info = &Self::THUMB_LUT[((insn >> 6) as usize) & 0x3FF];
        match (thumb_info.handler_fn)(self, insn as u16) {
            CpuAction::AdvancePC(next_access) => {
                self.next_fetch_access = next_access;
                self.pc = fetch_addr.wrapping_add(2);
                0
            }
            CpuAction::PipelineFlushed => 1,
        }
    }

    /// Phase-4 helper: do per-iter fetch + cycle accounting + pipeline
    /// shift. The block emit pairs this with inline LLVM IR for the
    /// instruction's effect.  No dispatch, no pc update.
    ///
    /// Cycles are charged via the bus path so per-page costs always
    /// reflect the current WAITCNT state (per I7).  Earlier phase-4
    /// step 2 (commit afb1ebd) tried inline cycle accumulation in IR
    /// with baked Seq/NonSeq constants, but PE writes WAITCNT during
    /// BIOS boot — the baked constants silently went stale and the
    /// inline-IR path picked up ~12 hash-divs at sw=64KB.  Reverting
    /// to bus-charged cycles here closes that hole.  Re-introducing
    /// inline cycle IR requires implementing I7 (synchronous recompile
    /// on WAITCNT write) first.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_thumb_fetch_only(&mut self, fetch_addr: u32) {
        let access = self.next_fetch_access;
        let val = self.load_16(fetch_addr, access);
        self.pipeline[0] = self.pipeline[1];
        self.pipeline[1] = val as u32;
    }

    /// AOT-side helper: word-sized LDR with I14 misaligned-LDR ROR
    /// semantics. Used by F11 LDR sp-rel inline IR (and future F9 LDR
    /// word) which has a runtime-computed address — unlike F6 where
    /// the address is constant-aligned at AOT compile time.
    ///
    /// Forwards to the private `ldr_word` in `memory.rs`. Side effect:
    /// when `addr & 3 != 0`, sets `cpsr.C` from the rotated result's
    /// top bit (per I14). The IR caller must NOT separately update
    /// cpsr.C around this call.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_ldr_word(&mut self, addr: u32, access: MemoryAccess) -> u32 {
        self.ldr_word(addr, access)
    }

    /// Forwards to the private `ldr_half` in `memory.rs`. Misaligned-addr
    /// (`addr & 1 != 0`) ROR side effect on cpsr.C — same shape as
    /// `aot_ldr_word`'s I14 behavior but for half-word load.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_ldr_half(&mut self, addr: u32, access: MemoryAccess) -> u32 {
        self.ldr_half(addr, access)
    }

    /// Forwards to the private `ldr_sign_half` in `memory.rs`. F8 LDSH
    /// uses this; misaligned-addr (`addr & 1 != 0`) does sign-extended
    /// byte load instead of halfword.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_ldr_sign_half(&mut self, addr: u32, access: MemoryAccess) -> u32 {
        self.ldr_sign_half(addr, access)
    }

    /// AOT-side helper: mid-block abort check (K=2 cadence per I2).
    /// Mirrors the scalar `replay_cached_block` per-iter abort guard.
    /// Returns true if the AOT block should yield to the dispatcher.
    ///
    /// The block was recorded as Thumb only (per phase-0 scope); the
    /// mode-flip check fires if cpu.cpsr.state() flipped to ARM.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_block_should_abort_thumb(&mut self) -> bool {
        if !matches!(self.cpsr.state(), CpuState::THUMB) {
            return true;
        }
        if self.bus.take_block_cache_dirty() {
            self.block_cache.flush();
            return true;
        }
        self.bus.cached_block_should_abort()
    }

    /// AOT-side helper: per-iter ARM step (fetch + pipeline shift +
    /// cond check + ARM_LUT handler dispatch + AdvancePC bookkeeping).
    /// Phase-8 scaffolding — mirrors `aot_thumb_step` but for ARM mode.
    ///
    /// Returns:
    ///   0 → AdvancePC (continue to next instruction).
    ///   1 → PipelineFlushed (handler updated cpu.pc + cpu.pipeline
    ///       at branch target; caller exits the block).
    ///
    /// Pipeline-head pc convention: scalar's self.pc at handler-call
    /// time IS `fetch_addr` (= exec_addr + 8 in ARM mode). After
    /// AdvancePC scalar advances pc by 4 → exec_addr + 12 = fetch_addr + 4.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_arm_step(&mut self, fetch_addr: u32, insn: u32) -> u32 {
        let access = self.next_fetch_access;
        let val = self.load_32(fetch_addr, access);
        self.pipeline[0] = self.pipeline[1];
        self.pipeline[1] = val;
        self.pc = fetch_addr;

        // ARM cond field (bits 28..32). AL = 0xE = always; skip cond
        // check on AL for the common case.
        let cond_bits = ((insn >> 28) & 0xf) as u8;
        if cond_bits != 0xE {
            // Cond is not AL; check it.
            let cond = match num::FromPrimitive::from_u8(cond_bits) {
                Some(c) => c,
                None => unsafe { std::hint::unreachable_unchecked() },
            };
            if !self.check_arm_cond(cond) {
                // Cond false — skip handler entirely. Scalar mirror:
                // `advance_arm(); next_fetch_access = NonSeq;`.
                self.next_fetch_access = MemoryAccess::NonSeq;
                self.pc = fetch_addr.wrapping_add(4);
                return 0; // AdvancePC
            }
        }

        // Dispatch via ARM_LUT (same hash as scalar's step_arm_exec).
        let hash = (((insn >> 16) & 0xff0) | ((insn >> 4) & 0xf)) as usize;
        let arm_info = &Self::ARM_LUT[hash];
        match (arm_info.handler_fn)(self, insn) {
            CpuAction::AdvancePC(next_access) => {
                self.next_fetch_access = next_access;
                self.pc = fetch_addr.wrapping_add(4);
                0
            }
            CpuAction::PipelineFlushed => 1,
        }
    }

    /// AOT-side helper: ARM mode mid-block abort check. Mirrors
    /// `aot_block_should_abort_thumb` but the mode-flip check fires
    /// if cpu state flipped to Thumb.
    #[cfg(feature = "cached_interp")]
    #[inline]
    pub fn aot_block_should_abort_arm(&mut self) -> bool {
        if matches!(self.cpsr.state(), CpuState::THUMB) {
            return true;
        }
        if self.bus.take_block_cache_dirty() {
            self.block_cache.flush();
            return true;
        }
        self.bus.cached_block_should_abort()
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
        // Phase-8: select Thumb or ARM lookup based on mode. Both
        // default to the no-op stub (returns 0) so try_aot_dispatch
        // can call unconditionally without an Option discriminator.
        let is_thumb = matches!(self.cpsr.state(), CpuState::THUMB);
        let lookup = if is_thumb {
            self.aot_lookup_fn
        } else {
            self.aot_lookup_fn_arm
        };
        let fn_addr = lookup(self.aot_table, self.pc);
        if fn_addr == 0 {
            return None;
        }
        // Per-mode hit counter (phase-8 debugging).
        if is_thumb {
            self.aot_dispatch_hits_thumb = self.aot_dispatch_hits_thumb.wrapping_add(1);
        } else {
            self.aot_dispatch_hits_arm = self.aot_dispatch_hits_arm.wrapping_add(1);
        }
        // Cold-start guard (I15): skip AOT until scalar has fetched
        // at least one instruction. AT cold start cpu.pc=0 (BIOS reset
        // vector) and BIOS isn't in the AOT table, so the lookup above
        // returns 0 and we early-out before reaching here. This guard
        // is only needed for save-state restores into AOT-keyed PCs
        // (per I12) — but pipeline[0] is also restored from save-state
        // so it's never 0 in practice. Keep guard for safety; it's
        // out of the cold-start mainline path.
        if self.pipeline[0] == 0 {
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
            aot_lookup_fn: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_lookup_fn_arm: aot_lookup_noop,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_hits_arm: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_thumb: 0,
            #[cfg(feature = "aot_dispatch")]
            aot_dispatch_misses_arm: 0,
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
                // try_aot_dispatch already bumps the per-mode hit
                // counters; the un-segmented total is just the sum.
                if !can_chain {
                    return;
                }
                // Inter-block abort check (matches scalar's
                // replay_cached_block exit). Without this, AOT
                // accumulates ~0.0001 cycles drift per dispatch
                // because scalar yields to the outer loop on every
                // block boundary while AOT chains blindly to the next
                // block — IRQs/DMA events fire ~1 instr late on AOT
                // vs scalar. Bisected to seed 0x080008ca (a hot 7-instr
                // loop); 18950 cycles drift over the PE replay with
                // just that seed enabled.
                //
                // Phase 0c worried this would fire too often vs scalar
                // because AOT's scan splits at Bcc-as-Branch making
                // smaller blocks. Phase 0d's Bcc-as-Linear fix made
                // AOT's blocks comparable to scalar's recorded blocks,
                // so the cadence concern no longer applies.
                if self.bus.cached_block_should_abort() {
                    return;
                }
                continue;
            }
            #[cfg(feature = "aot_dispatch")]
            if self.aot_lookup_fn as *const () != aot_lookup_noop as *const ()
                || self.aot_lookup_fn_arm as *const () != aot_lookup_noop as *const ()
            {
                // Hook installed (thumb or arm), lookup missed —
                // count as scalar dispatch.  Only per-mode counters now;
                // the un-segmented total is `thumb + arm`.
                if matches!(self.cpsr.state(), CpuState::THUMB) {
                    self.aot_dispatch_misses_thumb =
                        self.aot_dispatch_misses_thumb.wrapping_add(1);
                } else {
                    self.aot_dispatch_misses_arm =
                        self.aot_dispatch_misses_arm.wrapping_add(1);
                }
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
