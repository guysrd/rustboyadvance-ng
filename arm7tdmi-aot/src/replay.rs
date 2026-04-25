//! Trampoline for the phase-0 placeholder block emit.
//!
//! Each AOT-compiled block is, for phase 0, just a thin LLVM
//! wrapper around `aot_replay_thumb_block_for<I>`. The wrapper
//! bakes in the block's opcodes_ptr + len + entry_pc as constants
//! and calls this trampoline once per dispatch; the trampoline
//! does the per-instruction fetch + handler dispatch + K=2 abort
//! check (matches scalar `replay_cached_block`).
//!
//! Phase 1+ replaces these trampoline calls with per-format inline
//! IR (one emit fn per Thumb format). This module stays as the
//! fallback for formats not yet inlined and for phase 0 acceptance.

use arm7tdmi::Arm7tdmiCore;
use arm7tdmi::memory::MemoryInterface;
// Suppress unused warning when the pub-helpers feature gate is off.
#[allow(unused_imports)]
use arm7tdmi::CpuAction;

/// Phase-0 trampoline ABI. Caller (SDL frontend with `I = SysBus`)
/// passes `arm7tdmi_aot::replay::aot_replay_thumb_block_for::<SysBus>`
/// as the `replay_thumb_fn` arg of `compile_rom`.
pub type AotReplayFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    opcodes_ptr: *const u32,
    opcodes_len: u32,
    entry_pc: u32,
) -> u32;

/// K=2 abort cadence (per I2). Don't change without re-validating
/// SDL divs — k=4 broke divs on both ROMs in the JIT branch.
const ABORT_CADENCE_MASK: u32 = 1;

/// Per-I monomorphized trampoline that runs an entire Thumb block
/// through scalar handler dispatch with K=2 abort cadence.
///
/// Returns I8 ABI bits:
///   0     → fall-through to next block (caller chains).
///   0b01  → branch fired (handler updated cpu.pc + pipeline; caller
///           reads cpu.pc). NOTE: phase-0 placeholder doesn't write
///           pc_out — the caller's wrapper writes 0 there. The
///           dispatcher `try_aot_dispatch` reloads pipeline based on
///           cpu.pc which the handler already set. (For phase 1+
///           inline IR this changes — branches will write pc_out.)
///   0b10  → mid-block abort, dispatcher yields.
///
/// # Safety
/// `cpu_ctx` must point to a valid `Arm7tdmiCore<I>`. `opcodes_ptr`
/// must point to a contiguous array of `opcodes_len` u32s (Thumb
/// halfwords zero-extended).
#[inline]
pub unsafe extern "C" fn aot_replay_thumb_block_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    opcodes_ptr: *const u32,
    opcodes_len: u32,
    entry_pc: u32,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };

    for k in 0..opcodes_len {
        // Mid-block abort check (K=2 per I2). Skip k=0 — block
        // always makes forward progress.
        if k != 0 && (k & ABORT_CADENCE_MASK) == 1 && cpu.aot_block_should_abort_thumb() {
            return 0b10;
        }

        let insn = unsafe { *opcodes_ptr.add(k as usize) };
        let exec_addr = entry_pc.wrapping_add(2u32.wrapping_mul(k));
        let fetch_addr = exec_addr.wrapping_add(4);

        // Per-iter step via the pub helper on Arm7tdmiCore (returns
        // 1 = PipelineFlushed, 0 = AdvancePC). Keeps the field
        // accesses inside the arm7tdmi crate.
        if cpu.aot_thumb_step(fetch_addr, insn) == 1 {
            // Handler updated cpu.pc + cpu.pipeline at the branch
            // target. Per phase-0 ABI, return 0 (NOT 0b01) — the
            // dispatcher just reads cpu.pc on re-entry.
            // For phase 1+ this changes to 0b01 with target in pc_out.
            return 0;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    /// We can't easily test the trampoline without a full bus impl
    /// (it dispatches THUMB_LUT handlers which call into bus). The
    /// real exercise is via the SDL replay test in step 4c.
    #[test]
    fn signature_compiles() {
        // Ensures the type compiles for some I impl; the crate's
        // own tests don't have a bus impl available so this is just
        // a sanity check.
        use super::AotReplayFn;
        let _: Option<AotReplayFn> = None;
    }
}
