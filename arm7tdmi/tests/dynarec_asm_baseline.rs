//! Per-shape arm64 disasm baseline — prints what Cranelift emits for each
//! shape the dynarec supports. Gated behind `dynarec_asm_dump`.
//!
//! Run with:
//!     cargo test -p arm7tdmi --features dynarec_asm_dump \
//!                --test dynarec_asm_baseline -- --nocapture
//!
//! The agent reads the printed disasm when picking an experiment and
//! updates the "current arm64" field in results.tsv comments for the
//! shape they're about to tackle. These tests are not asserting anything
//! about the emitted bytes — only that the compile path runs without
//! panicking. Content changes are expected when any emit_* or pattern
//! lands.
#![cfg(feature = "dynarec_asm_dump")]

use arm7tdmi::dynarec::dump;

fn print_shape(name: &str, disasm: Result<String, String>) {
    println!("=== SHAPE={name} ===");
    match disasm {
        Ok(s) => print!("{s}"),
        Err(e) => println!("<disasm error: {e}>"),
    }
    println!("=== END ===");
}

#[test]
fn shape_thumb_mov_imm() {
    // MOV R1, #42 — Thumb format 3, op=MOV.
    print_shape("thumb_mov_imm", dump::dump_thumb_block(&[0x212a]));
}

#[test]
fn shape_thumb_add_imm() {
    // ADD R1, R1, #1 — Thumb format 3, op=ADD.
    print_shape("thumb_add_imm", dump::dump_thumb_block(&[0x3101]));
}

#[test]
fn shape_thumb_cmp_imm() {
    // CMP R1, #5 — Thumb format 3, op=CMP.
    print_shape("thumb_cmp_imm", dump::dump_thumb_block(&[0x2905]));
}

#[test]
fn shape_thumb_mov_chain() {
    // MOV R0,#1 ; MOV R1,#2 ; ADD R0,R0,R1 — exercises DP + register-form ADD.
    print_shape(
        "thumb_mov_chain",
        dump::dump_thumb_block(&[0x2001, 0x2102, 0x1840]),
    );
}

#[test]
fn shape_arm_mov_imm() {
    // MOV R1, #42
    print_shape(
        "arm_mov_imm",
        dump::dump_arm_imm_block(&[0xE3A0_102Au32]),
    );
}

#[test]
fn shape_arm_cmp_imm() {
    // CMP R0, #5  — sets NZCV, no gpr write.
    print_shape(
        "arm_cmp_imm",
        dump::dump_arm_imm_block(&[0xE350_0005u32]),
    );
}
