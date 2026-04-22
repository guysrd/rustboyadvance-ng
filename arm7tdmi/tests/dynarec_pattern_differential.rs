//! Differential-execution tests for synthesized shapes recognized by
//! `arm7tdmi::dynarec::patterns`. Each newly-added pattern lands here
//! with crafted opcode sequences + boundary cases that assert gpr +
//! NZCV match the scalar interpreter bit-for-bit.
//!
//! Empty today. The first synthesized shape added to `patterns.rs` by
//! the research loop (see docs/program-shapekarpathy.md) brings its
//! corresponding test here.
#![cfg(feature = "dynarec")]

// Imports kept ready so the first synthesized-shape test doesn't need
// to rediscover the helper module.
#[allow(unused_imports)]
use arm7tdmi::dynarec::test_utils::{differential, differential_with_flags};

/// Sanity: the helper-import path resolves and the crate-feature gate
/// is correct. Removed / replaced once real pattern tests land.
#[test]
fn helper_imports_resolve() {
    // A trivial differential that the dynarec already handles — MOV
    // R0,#7 via ARM imm DP.
    differential(&[0xE3A0_0007u32], [0u32; 15]);
}
