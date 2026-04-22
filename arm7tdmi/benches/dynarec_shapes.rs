//! Per-shape Criterion micro-bench for the dynarec.
//!
//! Each bench compiles one representative block once, then tight-loops
//! invocations of the resulting fn pointer with `black_box`. ns/iter
//! per shape is what scripts/dynarec_measure.sh multiplies against the
//! per-shape call count from a shape_profile profile run.
//!
//! Gated behind the `bench` feature so the default build doesn't pull
//! Criterion. Run with:
//!
//!     cargo bench -p arm7tdmi --bench dynarec_shapes \
//!                 --features dynarec,bench -- --quiet
//!
//! Add a new bench here when a new synthesized shape lands in
//! `arm7tdmi::dynarec::patterns`.
#![cfg(feature = "bench")]

use arm7tdmi::dynarec::DynarecCompiler;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_thumb_mov_imm(c: &mut Criterion) {
    // MOV R1, #42 — single-instruction Thumb format 3 block.
    let mut compiler = DynarecCompiler::new();
    let func = compiler.try_compile_thumb_block(&[0x212a]).unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("thumb_mov_imm", |b| {
        b.iter(|| {
            func(
                black_box(gpr.as_mut_ptr()),
                black_box(&mut cpsr as *mut u32),
            );
            black_box(&mut gpr);
        })
    });
}

fn bench_thumb_add_imm(c: &mut Criterion) {
    // ADD R1, R1, #1
    let mut compiler = DynarecCompiler::new();
    let func = compiler.try_compile_thumb_block(&[0x3101]).unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("thumb_add_imm", |b| {
        b.iter(|| {
            func(black_box(gpr.as_mut_ptr()), black_box(&mut cpsr as *mut u32));
            black_box(&mut gpr);
        })
    });
}

fn bench_thumb_cmp_imm(c: &mut Criterion) {
    // CMP R1, #5 — sets NZCV.
    let mut compiler = DynarecCompiler::new();
    let func = compiler.try_compile_thumb_block(&[0x2905]).unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("thumb_cmp_imm", |b| {
        b.iter(|| {
            func(black_box(gpr.as_mut_ptr()), black_box(&mut cpsr as *mut u32));
            black_box(&mut cpsr);
        })
    });
}

fn bench_thumb_dp_chain(c: &mut Criterion) {
    // MOV R0,#1 ; MOV R1,#2 ; ADD R0,R0,R1
    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_thumb_block(&[0x2001, 0x2102, 0x1840])
        .unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("thumb_dp_chain", |b| {
        b.iter(|| {
            func(black_box(gpr.as_mut_ptr()), black_box(&mut cpsr as *mut u32));
            black_box(&mut gpr);
        })
    });
}

fn bench_arm_mov_imm(c: &mut Criterion) {
    // MOV R1, #42
    let mut compiler = DynarecCompiler::new();
    let func = compiler.try_compile_imm_block(&[0xE3A0_102Au32]).unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("arm_mov_imm", |b| {
        b.iter(|| {
            func(black_box(gpr.as_mut_ptr()), black_box(&mut cpsr as *mut u32));
            black_box(&mut gpr);
        })
    });
}

fn bench_arm_cmp_imm(c: &mut Criterion) {
    // CMP R0, #5 — sets NZCV.
    let mut compiler = DynarecCompiler::new();
    let func = compiler.try_compile_imm_block(&[0xE350_0005u32]).unwrap();
    let mut gpr = [0u32; 15];
    let mut cpsr = 0u32;
    c.bench_function("arm_cmp_imm", |b| {
        b.iter(|| {
            func(black_box(gpr.as_mut_ptr()), black_box(&mut cpsr as *mut u32));
            black_box(&mut cpsr);
        })
    });
}

criterion_group!(
    benches,
    bench_thumb_mov_imm,
    bench_thumb_add_imm,
    bench_thumb_cmp_imm,
    bench_thumb_dp_chain,
    bench_arm_mov_imm,
    bench_arm_cmp_imm,
);
criterion_main!(benches);
