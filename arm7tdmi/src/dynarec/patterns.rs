//! Block-level pattern matcher for the dynarec.
//!
//! Called at the top of every full-block `try_compile_*` entry point.
//! When `try_match` returns `Some(Pattern)`, the compiler emits the
//! hand-lowered stencil for that pattern and returns. When it returns
//! `None`, the existing per-instruction emitter loop runs, unchanged.
//!
//! Stub today: `try_match` always returns `None`, so the compiled
//! binary is byte-identical to the pre-plan build. Synthesized shapes
//! (mul-by-constant, div-by-constant, CLZ polyfill, popcount SWAR,
//! branchless abs/min/max, shift-pair sign extend, byte swap, GBA BIOS
//! CpuSet fold, dead-DP-chain DCE) are added one at a time per the
//! experiment loop in docs/program-shapekarpathy.md, each with its own
//! entry in `tests/dynarec_pattern_differential.rs`.
//!
//! Naming: "Thumb pattern" for Thumb-block matchers, "ARM pattern" for
//! ARM-block matchers. Patterns that span both stay on the Thumb side
//! for now since that's where the agent loop starts.

/// Synthesized shape the matcher recognized. Variants will be added as
/// patterns are implemented; today the enum is empty and `try_match`
/// always returns `None`.
pub enum Pattern {
    // e.g. MulByConst { rd: u8, rs: u8, imm: u32 },
    //      DivByConstMagic { rd: u8, rs: u8, magic: u32, shift: u8 },
    //      ClzPolyfill { rd: u8, rs: u8 },
    //      PopcountSwar { rd: u8, rs: u8 },
    //      BranchlessAbs { rd: u8, rs: u8 },
    //      BranchlessMinMax { rd: u8, ra: u8, rb: u8, is_max: bool },
    //      ShiftPairSignExtend { rd: u8, rs: u8, width: u8 /* 8 or 16 */ },
    //      ByteSwap { rd: u8, rs: u8 },
    //      BiosCpuSet { mode: CpuSetMode },
    //      DeadDpChain { skipped: u8 },
}

/// Try to recognize a known synthesized shape in a Thumb block. Returns
/// `None` by default; patterns are added one at a time by the research
/// loop.
#[allow(dead_code)]
pub(crate) fn try_match_thumb(_opcodes: &[u16]) -> Option<Pattern> {
    None
}

/// Try to recognize a known synthesized shape in an ARM block.
#[allow(dead_code)]
pub(crate) fn try_match_arm(_opcodes: &[u32]) -> Option<Pattern> {
    None
}
