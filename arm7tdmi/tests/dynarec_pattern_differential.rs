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
/// is correct. Kept as a baseline test alongside pattern tests.
#[test]
fn helper_imports_resolve() {
    differential(&[0xE3A0_0007u32], [0u32; 15]);
}

// --- Shift-pair sign/zero extend patterns ---
//
// The block-level matcher in `arm7tdmi::dynarec::patterns` recognizes
// these two-instruction idioms and emits a direct narrow+extend
// stencil. These tests drive the dynarec against the scalar
// interpreter for each case to prove the pattern is bit-exact.
//
// Thumb format 1 encoding: 000 oo imm5 Rs Rd
//   oo=00 LSL, oo=01 LSR, oo=10 ASR
// For Rd=R1, Rs=R0, imm5=24:
//   LSL #24 R1, R0  = 0x0601
//   LSR #24 R1, R1  = 0x0E09  (Rs=R1, Rd=R1)
//   ASR #24 R1, R1  = 0x1609
// For imm5=16:
//   LSL #16 R1, R0  = 0x0401
//   LSR #16 R1, R1  = 0x0C09
//   ASR #16 R1, R1  = 0x1409
//
// Thumb-only differential helper: the `differential` function in
// test_utils handles ARM; for Thumb we need a separate path. Since the
// bench already exercises the full dynarec path end-to-end, and the
// pattern stencil's flag packing mirrors the two-instruction sequence's
// final state, we fabricate a minimal pair via
// `DynarecCompiler::try_compile_thumb_block` and compare gpr+flags to
// what the scalar Thumb handler produces.

use arm7tdmi::dynarec::DynarecCompiler;
use arm7tdmi::{SimpleMemory, cpu::Arm7tdmiCore};
use rustboyadvance_utils::Shared;

fn run_scalar_thumb(opcodes: &[u16], initial_gpr: [u32; 15]) -> ([u32; 15], u32) {
    // Build a tiny SimpleMemory program and drive each Thumb opcode
    // through its handler fn. We step the handlers directly (no
    // pipeline dance) like `differential` does for ARM.
    let mut mem = SimpleMemory::new(1024);
    let mut program = Vec::with_capacity(opcodes.len() * 2);
    for &op in opcodes {
        program.extend_from_slice(&op.to_le_bytes());
    }
    mem.load_program(&program);
    let mem_shared = Shared::new(mem);
    let mut cpu = Arm7tdmiCore::new(mem_shared);
    cpu.gpr = initial_gpr;
    // Drop default CPSR bits that might perturb flag comparisons. The
    // scalar Thumb handlers read Cpu state directly, no pipeline
    // staging needed.
    for &op in opcodes {
        let hash = (op >> 6) as usize & 0x3ff;
        let info = &Arm7tdmiCore::<SimpleMemory>::THUMB_LUT[hash];
        (info.handler_fn)(&mut cpu, op);
    }
    (cpu.gpr, cpu.cpsr.get() & 0xF000_0000)
}

fn run_dynarec_thumb(opcodes: &[u16], initial_gpr: [u32; 15]) -> ([u32; 15], u32) {
    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_thumb_block(opcodes)
        .expect("dynarec should compile this Thumb block");
    let mut gpr = initial_gpr;
    let mut cpsr = 0u32;
    func(gpr.as_mut_ptr(), &mut cpsr);
    (gpr, cpsr & 0xF000_0000)
}

fn differential_thumb(opcodes: &[u16], initial_gpr: [u32; 15]) {
    let (sgpr, scpsr) = run_scalar_thumb(opcodes, initial_gpr);
    let (dgpr, dcpsr) = run_dynarec_thumb(opcodes, initial_gpr);
    assert_eq!(
        sgpr, dgpr,
        "gpr diverged for opcodes {:x?}: scalar={:?} dyn={:?}",
        opcodes, sgpr, dgpr
    );
    assert_eq!(
        scpsr, dcpsr,
        "CPSR NZCV diverged for opcodes {:x?}: scalar={:#010x} dyn={:#010x}",
        opcodes, scpsr, dcpsr
    );
}

#[test]
fn pattern_sxtb_positive_values() {
    // LSL R1, R0, #24 ; ASR R1, R1, #24
    let block: [u16; 2] = [0x0601, 0x1609];
    for &r0 in &[0u32, 1, 0x7f, 0xff, 0x80, 0x100, 0x1234_5678, 0xffff_ff7f] {
        let mut gpr = [0u32; 15];
        gpr[0] = r0;
        differential_thumb(&block, gpr);
    }
}

#[test]
fn pattern_sxth_various() {
    // LSL R1, R0, #16 ; ASR R1, R1, #16
    let block: [u16; 2] = [0x0401, 0x1409];
    for &r0 in &[0u32, 1, 0x7fff, 0xffff, 0x8000, 0x1_0000, 0x1234_5678, 0xdead_beef] {
        let mut gpr = [0u32; 15];
        gpr[0] = r0;
        differential_thumb(&block, gpr);
    }
}

#[test]
fn pattern_uxtb_various() {
    // LSL R1, R0, #24 ; LSR R1, R1, #24
    let block: [u16; 2] = [0x0601, 0x0e09];
    for &r0 in &[0u32, 1, 0xff, 0x100, 0x8000_00ab, 0xffff_ff01] {
        let mut gpr = [0u32; 15];
        gpr[0] = r0;
        differential_thumb(&block, gpr);
    }
}

#[test]
fn pattern_uxth_various() {
    // LSL R1, R0, #16 ; LSR R1, R1, #16
    let block: [u16; 2] = [0x0401, 0x0c09];
    for &r0 in &[0u32, 1, 0xffff, 0x1_0000, 0x8000_abcd, 0xdead_beef] {
        let mut gpr = [0u32; 15];
        gpr[0] = r0;
        differential_thumb(&block, gpr);
    }
}
