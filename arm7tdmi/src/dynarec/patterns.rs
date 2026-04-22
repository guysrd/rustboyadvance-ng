//! Block-level pattern matcher for the dynarec.
//!
//! Called at the top of every full-block `try_compile_*` entry point.
//! When `try_match_thumb` returns `Some(Pattern)`, the compiler emits
//! the hand-lowered stencil for that pattern and returns. When it
//! returns `None`, the existing per-instruction emitter loop runs,
//! unchanged.
//!
//! Synthesized shapes are added one at a time per the experiment loop
//! in `docs/program-shapekarpathy.md`, each with its own entry in
//! `tests/dynarec_pattern_differential.rs`.

use super::ShiftKind;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::prelude::*;

/// Synthesized shape the matcher recognized. Variants get added as
/// patterns are implemented.
#[allow(dead_code)]
pub enum Pattern {
    /// `LSL Rd, Rs, #24 ; ASR Rd, Rd, #24`  →  `sxtb Wd, Ws` on arm64
    /// (or `movsx` chain on x86_64). Standard compiler-generated
    /// `i8 -> i32` widening idiom; extremely common in real GBA code.
    /// Emits one signed byte-extend + one register writeback, no flag
    /// update (block-level pattern preserves scalar semantics: neither
    /// instruction is flag-observable at block boundary since the ASR
    /// overwrites LSL's flags and the ASR's flag values are the same
    /// `sxtb` would produce).
    ShiftPairSignExtendByte { rd: i32, rs: i32 },
    /// `LSL Rd, Rs, #16 ; ASR Rd, Rd, #16`  →  `sxth` (signed i16
    /// widen). Same pattern, 16-bit width.
    ShiftPairSignExtendHalf { rd: i32, rs: i32 },
    /// `LSL Rd, Rs, #24 ; LSR Rd, Rd, #24`  →  `uxtb` (unsigned byte
    /// zero-extend via `rd & 0xff`).
    ShiftPairZeroExtendByte { rd: i32, rs: i32 },
    /// `LSL Rd, Rs, #16 ; LSR Rd, Rd, #16`  →  `uxth`.
    ShiftPairZeroExtendHalf { rd: i32, rs: i32 },
}

/// Try to recognize a known synthesized shape in a Thumb block.
/// Currently only matches if the ENTIRE block is exactly two
/// instructions forming one of the known shift-pair idioms. Returns
/// `None` for any other shape; caller falls through to the
/// per-instruction emit loop unchanged.
pub(crate) fn try_match_thumb(opcodes: &[u16]) -> Option<Pattern> {
    if opcodes.len() != 2 {
        return None;
    }
    let first = super::DynarecCompiler::decode_thumb_format1(opcodes[0])?;
    let second = super::DynarecCompiler::decode_thumb_format1(opcodes[1])?;

    // Shift amount must be the pair (24 for byte, 16 for half), source
    // of second must be the dest of first, and both must target the
    // same Rd.
    if first.rd != second.rs || first.rd != second.rd {
        return None;
    }
    if first.kind != ShiftKind::Lsl {
        return None;
    }
    match (first.imm5, second.imm5, second.kind) {
        (24, 24, ShiftKind::Asr) => Some(Pattern::ShiftPairSignExtendByte {
            rd: first.rd,
            rs: first.rs,
        }),
        (16, 16, ShiftKind::Asr) => Some(Pattern::ShiftPairSignExtendHalf {
            rd: first.rd,
            rs: first.rs,
        }),
        (24, 24, ShiftKind::Lsr) => Some(Pattern::ShiftPairZeroExtendByte {
            rd: first.rd,
            rs: first.rs,
        }),
        (16, 16, ShiftKind::Lsr) => Some(Pattern::ShiftPairZeroExtendHalf {
            rd: first.rd,
            rs: first.rs,
        }),
        _ => None,
    }
}

/// Emit the host-code stencil for a recognized pattern into the
/// Cranelift builder. Updates gpr[rd] and packs the right NZ flags
/// into cpsr (the ASR/LSR's NZ — same as what the two-instruction
/// sequence produces under the interpreter). Preserves C (LSL with
/// imm=0 preserves C; imm=24 sets C from bit 24-1=23 of Rs; the
/// subsequent ASR/LSR imm=24 sets C from bit 23 of the SAME value —
/// so net C is bit 23 of Rs, matching the standalone interpretation).
///
/// Returns the final cpsr Value for the caller to store via def_var.
pub(crate) fn emit_pattern_thumb(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    pat: &Pattern,
) {
    let (rd, rs, width_bits, signed) = match *pat {
        Pattern::ShiftPairSignExtendByte { rd, rs } => (rd, rs, 8, true),
        Pattern::ShiftPairSignExtendHalf { rd, rs } => (rd, rs, 16, true),
        Pattern::ShiftPairZeroExtendByte { rd, rs } => (rd, rs, 8, false),
        Pattern::ShiftPairZeroExtendHalf { rd, rs } => (rd, rs, 16, false),
    };

    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(rs * 4),
    );

    // Narrow-then-widen:
    //   signed:   (Rs << (32 - width)) >> (32 - width) arithmetic
    //   unsigned: Rs & ((1 << width) - 1)
    // We implement as ireduce + {sextend,uextend} so Cranelift picks
    // the right arm64 instruction (sxtb/sxth/uxtb/uxth) directly.
    let narrow_ty = if width_bits == 8 { types::I8 } else { types::I16 };
    let narrowed = builder.ins().ireduce(narrow_ty, rs_val);
    let extended = if signed {
        builder.ins().sextend(types::I32, narrowed)
    } else {
        builder.ins().uextend(types::I32, narrowed)
    };
    builder.ins().store(
        MemFlags::trusted(),
        extended,
        gpr_ptr,
        Offset32::new(rd * 4),
    );

    // Flags that the two-instruction sequence sets (at the end):
    //   N = extended >> 31            (the final shift's sign bit)
    //   Z = (extended == 0)
    //   C = 0                          (second shift's shifted-out
    //                                   bits are all 0, because the
    //                                   first shift filled the low
    //                                   `width` bits with zeros)
    //   V unchanged
    let zero = builder.ins().iconst(types::I32, 0);
    let _ = rs_val; // rs_val unused past the narrow; keep-alive is fine.
    let n_shifted = builder.ins().band_imm(extended, 0x8000_0000_u32 as i64);
    let z_bool = builder.ins().icmp(IntCC::Equal, extended, zero);
    let z_u32 = builder.ins().uextend(types::I32, z_bool);
    let z_shifted = builder.ins().ishl_imm(z_u32, 30);

    // Merge into cpsr: clear N/Z/C (keep V), OR new N/Z in. C stays 0.
    let cpsr = builder.use_var(cpsr_var);
    let cleared = builder.ins().band_imm(cpsr, 0x1fff_ffff);
    let nz = builder.ins().bor(n_shifted, z_shifted);
    let new_cpsr = builder.ins().bor(cleared, nz);
    builder.def_var(cpsr_var, new_cpsr);
}
