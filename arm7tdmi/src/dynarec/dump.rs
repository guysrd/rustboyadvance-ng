//! Cranelift-disasm dump helper for the shape-optimization research loop.
//!
//! Compiles a block via the normal `try_compile_*` path, then disassembles
//! the resulting host machine code using capstone so the agent can read
//! what Cranelift actually emits per shape.
//!
//! Auto-detects host arch (aarch64 on Android / Apple Silicon, x86_64
//! otherwise) so the same helper works on both the dev desktop and the
//! PGO target. Output format is plain text: one instruction per line.
//!
//! Feature-gated behind `dynarec_asm_dump`. Not in the default build.
//!
//! LIMITATIONS (best-effort):
//!   - Cranelift's JIT writes the compiled function into a shared code
//!     page alongside other functions. We don't currently have a clean
//!     hook to read the compiled-function size back out of the module,
//!     so we disassemble a fixed window and rely on capstone stopping
//!     cleanly at the first `ret` or invalid instruction.
//!   - capstone errors are surfaced to the caller as `Err(String)`.

use super::DynarecCompiler;
use capstone::prelude::*;

/// Size of the disasm window after the function pointer. Pokeemerald
/// mem-blocks are ~500 bytes post-codegen; 2 KiB is a comfortable
/// upper bound.
const DISASM_WINDOW_BYTES: usize = 2048;

fn build_capstone() -> Result<Capstone, String> {
    #[cfg(target_arch = "aarch64")]
    {
        Capstone::new()
            .arm64()
            .mode(arch::arm64::ArchMode::Arm)
            .detail(false)
            .build()
            .map_err(|e| format!("capstone arm64 init: {e}"))
    }
    #[cfg(target_arch = "x86_64")]
    {
        Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .detail(false)
            .build()
            .map_err(|e| format!("capstone x86_64 init: {e}"))
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Err("unsupported host arch for capstone dump".to_string())
    }
}

fn disasm_at(ptr: *const u8) -> Result<String, String> {
    let cs = build_capstone()?;
    // SAFETY: we're reading executable-page bytes owned by the JITModule;
    // they're valid for at least the function size, which is guaranteed
    // to be >= 1 and almost always < 2 KiB for the shapes we compile.
    // Reading past end-of-function lands on adjacent JIT code (also
    // valid memory) or on zero-init fill; capstone will either decode
    // it as adjacent instructions (harmless) or stop at an invalid
    // encoding.
    let window = unsafe { std::slice::from_raw_parts(ptr, DISASM_WINDOW_BYTES) };
    let insns = cs
        .disasm_all(window, ptr as u64)
        .map_err(|e| format!("capstone disasm: {e}"))?;
    let mut out = String::new();
    for insn in insns.iter() {
        out.push_str(&format!(
            "  {:#018x}  {:8}  {}\n",
            insn.address(),
            insn.mnemonic().unwrap_or(""),
            insn.op_str().unwrap_or(""),
        ));
        // Heuristic cutoff: first `ret` (x86) or `ret` (arm64 mnemonic).
        if insn.mnemonic().map(|m| m == "ret" || m == "retq").unwrap_or(false) {
            break;
        }
    }
    Ok(out)
}

/// Dump the arm64/x86_64 disassembly Cranelift produces for the given
/// Thumb imm block, using `try_compile_thumb_block`.
pub fn dump_thumb_block(opcodes: &[u16]) -> Result<String, String> {
    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_thumb_block(opcodes)
        .ok_or("try_compile_thumb_block returned None")?;
    disasm_at(func as *const u8)
}

/// Dump the arm64/x86_64 disassembly for an ARM immediate-DP block.
pub fn dump_arm_imm_block(opcodes: &[u32]) -> Result<String, String> {
    let mut compiler = DynarecCompiler::new();
    let func = compiler
        .try_compile_imm_block(opcodes)
        .ok_or("try_compile_imm_block returned None")?;
    disasm_at(func as *const u8)
}

/// Dump the disassembly for the UNIFIED Thumb compile path (the one
/// the block cache actually uses in production). Uses stub bus
/// trampolines, so the disassembly shows real calls into the Rust
/// trampoline table — useful for seeing what thumb_fetch_n / the
/// load/store helpers look like in the generated code.
pub fn dump_thumb_unified_block(opcodes: &[u16], entry_pc: u32) -> Result<String, String> {
    use crate::cpu::Arm7tdmiCore;
    use crate::memory::{MemoryAccess, MemoryInterface};
    // Stub bus where every trampoline is a no-op. We only care about
    // the emitted code shape, not its runtime behavior.
    unsafe extern "C" fn stub1(_ctx: *mut u8, _a: u32) -> u32 { 0 }
    unsafe extern "C" fn stub2(_ctx: *mut u8, _a: u32, _b: u32) {}
    unsafe extern "C" fn stub_ctx(_ctx: *mut u8) {}
    unsafe extern "C" fn stub_ctx_r(_ctx: *mut u8) -> u32 { 0 }
    unsafe extern "C" fn stub_ctx_pc(_ctx: *mut u8, _pc: u32) {}
    unsafe extern "C" fn stub_fetch_n(_ctx: *mut u8, _pc: u32, _count: u32) {}
    let trampolines = super::BusTrampolines {
        load_32: stub1, store_32: stub2, load_8: stub1, store_8: stub2,
        load_with_idle_32: stub1, load_with_idle_8: stub1,
        set_next_fetch_nonseq: stub_ctx,
        pay_thumb_fetch_extra_nonseq: stub_ctx_pc,
        idle_cycle: stub_ctx,
        thumb_fetch_n: stub_fetch_n,
        chain_abort_check: stub_ctx_r,
    };
    let _ = std::marker::PhantomData::<(Arm7tdmiCore<crate::SimpleMemory>, MemoryAccess)>;
    let _ = std::any::type_name::<dyn MemoryInterface>;
    let mut compiler = DynarecCompiler::new_with_bus(trampolines);
    let func = compiler
        .try_compile_thumb_mem_block_with_branch(opcodes, entry_pc, None)
        .ok_or("try_compile_thumb_mem_block_with_branch returned None")?;
    disasm_at(func as *const u8)
}
