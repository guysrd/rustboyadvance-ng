//! Differential-execution helpers shared between the in-crate
//! `#[cfg(test)] mod tests` and the `tests/dynarec_pattern_differential.rs`
//! integration test binary. Only compiled under `dynarec`.
//!
//! Runs the same ARM opcode sequence through the scalar interpreter and
//! through the dynarec, then asserts gpr (and optionally NZCV) match.
//! Kept as plain `pub` functions rather than behind a feature because
//! the arm7tdmi crate is workspace-only and not shipped on crates.io.

use crate::dynarec::DynarecCompiler;

/// Differential-execution: interpreter vs dynarec, gpr-only check.
pub fn differential(opcodes: &[u32], initial_gpr: [u32; 15]) {
    use crate::cpu::{Arm7tdmiCore, CpuAction};
    use rustboyadvance_utils::Shared;

    // Interpreter run
    let mut program = Vec::with_capacity(opcodes.len() * 4);
    for op in opcodes {
        program.extend_from_slice(&op.to_le_bytes());
    }
    let mut mem = crate::SimpleMemory::new(1024);
    mem.load_program(&program);
    let mem_shared = Shared::new(mem);
    let mut cpu = Arm7tdmiCore::new(mem_shared);
    cpu.gpr = initial_gpr;
    for &op in opcodes {
        let _: CpuAction = {
            let hash = (((op >> 16) & 0xff0) | ((op >> 4) & 0xf)) as usize;
            let arm_info = &Arm7tdmiCore::<crate::SimpleMemory>::ARM_LUT[hash];
            (arm_info.handler_fn)(&mut cpu, op)
        };
    }
    let interp_gpr = cpu.gpr;

    // Dynarec run
    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_imm_block(opcodes)
        .expect("dynarec should support these opcodes");
    let mut dyn_gpr = initial_gpr;
    let mut dyn_cpsr = 0u32;
    func(dyn_gpr.as_mut_ptr(), &mut dyn_cpsr);

    assert_eq!(
        dyn_gpr, interp_gpr,
        "dynarec and interpreter diverged on block {:x?}",
        opcodes
    );
}

/// Differential-execution: interpreter vs dynarec, gpr + NZCV check.
pub fn differential_with_flags(
    opcodes: &[u32],
    initial_gpr: [u32; 15],
    initial_cpsr: u32,
) {
    use crate::cpu::Arm7tdmiCore;
    use rustboyadvance_utils::Shared;

    let mut program = Vec::with_capacity(opcodes.len() * 4);
    for op in opcodes {
        program.extend_from_slice(&op.to_le_bytes());
    }
    let mut mem = crate::SimpleMemory::new(1024);
    mem.load_program(&program);
    let mem_shared = Shared::new(mem);
    let mut cpu = Arm7tdmiCore::new(mem_shared);
    cpu.gpr = initial_gpr;
    cpu.cpsr = crate::psr::RegPSR::new(initial_cpsr);
    for &op in opcodes {
        let hash = (((op >> 16) & 0xff0) | ((op >> 4) & 0xf)) as usize;
        let arm_info = &Arm7tdmiCore::<crate::SimpleMemory>::ARM_LUT[hash];
        let _ = (arm_info.handler_fn)(&mut cpu, op);
    }
    let interp_gpr = cpu.gpr;
    let interp_cpsr = cpu.cpsr.get() & 0xF000_0000;

    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_imm_block(opcodes)
        .expect("dynarec should support these opcodes");
    let mut dyn_gpr = initial_gpr;
    let mut dyn_cpsr = initial_cpsr;
    func(dyn_gpr.as_mut_ptr(), &mut dyn_cpsr);
    let dyn_cpsr_flags = dyn_cpsr & 0xF000_0000;

    assert_eq!(
        dyn_gpr, interp_gpr,
        "gpr diverged on {:x?}\ninterp={:?}\ndynrec={:?}",
        opcodes, interp_gpr, dyn_gpr
    );
    assert_eq!(
        dyn_cpsr_flags, interp_cpsr,
        "NZCV diverged on {:x?}: interp={:#010x} dynarec={:#010x}",
        opcodes, interp_cpsr, dyn_cpsr_flags
    );
}
