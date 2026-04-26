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

/// Phase-0 whole-block trampoline ABI. Used by the placeholder
/// `compile_thumb_block`. Phase-1+ uses the per-instruction
/// trampolines below (`AotStepFn`, `AotAbortFn`) for per-format
/// inline IR.
pub type AotReplayFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    opcodes_ptr: *const u32,
    opcodes_len: u32,
    entry_pc: u32,
) -> u32;

/// Phase-1 per-iter Thumb step trampoline. Mirrors the JIT branch's
/// `thumb_step_with_fetch_for<I>`. Drives one instruction's
/// dispatch:
///   - fetch at fetch_addr (cycle accounting + pipeline shift +
///     cpu.pc = fetch_addr).
///   - dispatch THUMB_LUT handler.
///   - on AdvancePC: cpu.pc = fetch_addr + 2, cpu.next_fetch_access
///     = handler-supplied access. Returns 0.
///   - on PipelineFlushed: handler already updated cpu.pc + cpu.pipeline.
///     Returns 1.
pub type AotStepFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    fetch_addr: u32,
    insn: u32,
) -> u32;

/// Per-I monomorphized step trampoline. SDL frontend passes
/// `aot_thumb_step_for::<SysBus>`.
pub unsafe extern "C" fn aot_thumb_step_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    fetch_addr: u32,
    insn: u32,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    cpu.aot_thumb_step(fetch_addr, insn)
}

/// Phase-4 fetch-only trampoline: does just fetch + cycle accounting
/// + pipeline shift, NO dispatch. The block emit pairs this with
/// inline LLVM IR for the actual instruction effect.
pub type AotFetchOnlyFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    fetch_addr: u32,
);

pub unsafe extern "C" fn aot_thumb_fetch_only_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    fetch_addr: u32,
) {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    cpu.aot_thumb_fetch_only(fetch_addr);
}

/// Phase-4 helper: bus-side aligned word load with cycle accounting.
/// Used by F6 LDR pc-rel inline IR (and future F9/F11 paths).
/// `addr` MUST be 4-byte aligned (caller guarantees per F6 semantics
/// — see arm7tdmi/src/cpu.rs aot_thumb_step F6 path: addr =
/// (fetch_addr & !3) + (imm8 << 2), always word-aligned).
/// `access_byte`: 0 = NonSeq, 1 = Seq (matches MemoryAccess repr).
pub type AotLoad32Fn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32;

pub unsafe extern "C" fn aot_load_32_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32 {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.bus.load_32(addr, access)
}

/// Phase-4 helper: charge one idle cycle (`bus.idle_cycle()`).
/// Used by F6 (and future shift-by-reg / MUL paths) to avoid the
/// inline `*ts_ptr += 1` pattern that the F4 MUL attempt suspected
/// of LLVM-codegen issues (3 reproducible MK divs at sw=64KB).
/// Bus path is always correct.
pub type AotIdleCycleFn = unsafe extern "C" fn(cpu_ctx: *mut u8);

pub unsafe extern "C" fn aot_idle_cycle_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
) {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    cpu.bus.idle_cycle();
}

/// Phase-4 helper: bus-side aligned word store with cycle accounting.
/// Used by F11 STR sp-rel inline IR (and future F9 STR paths).
/// Stores at `addr & ~3` (matches scalar's `store_aligned_32`).
/// `access_byte`: 0 = NonSeq, 1 = Seq.
pub type AotStore32Fn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u32,
    access_byte: u8,
);

pub unsafe extern "C" fn aot_store_32_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u32,
    access_byte: u8,
) {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.bus.store_32(addr & !0x3, val, access);
}

/// Phase-4 helper: word LDR with I14 misaligned-LDR ROR semantics
/// (handles `addr & 3 != 0` rotate + cpsr.C side effect). Used by
/// F11 LDR sp-rel inline IR — F11 has a runtime address (gpr[SP] +
/// imm) so can hit the misaligned path, unlike F6 which is
/// constant-aligned. `access_byte`: 0 = NonSeq, 1 = Seq.
pub type AotLdrWordFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32;

pub unsafe extern "C" fn aot_ldr_word_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32 {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.aot_ldr_word(addr, access)
}

/// Phase-4 helper: halfword LDRH with misaligned-addr ROR (addr & 1 != 0
/// rotates by 8 bits and sets cpsr.C from result bit 31 — same shape
/// as I14 word-LDR ROR but at halfword granularity). Used by F10 LDRH
/// inline IR. Returns u32 (zero-extended halfword, possibly rotated).
pub type AotLdrHalfFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32;

pub unsafe extern "C" fn aot_ldr_half_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32 {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.aot_ldr_half(addr, access)
}

/// Phase-4 helper: bus-side aligned 16-bit store with cycle accounting.
/// Used by F10 STRH inline IR.  Stores at `addr & ~1` (matches scalar's
/// `store_aligned_16`). `access_byte`: 0 = NonSeq, 1 = Seq.
pub type AotStore16Fn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u16,
    access_byte: u8,
);

pub unsafe extern "C" fn aot_store_16_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u16,
    access_byte: u8,
) {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.bus.store_16(addr & !0x1, val, access);
}

/// Phase-4 helpers: bus-side byte load/store with cycle accounting.
/// Used by F9 LDRB/STRB inline IR. No alignment concerns (byte ops).
pub type AotLoad8Fn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u8;

pub unsafe extern "C" fn aot_load_8_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u8 {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.bus.load_8(addr, access)
}

pub type AotStore8Fn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u8,
    access_byte: u8,
);

pub unsafe extern "C" fn aot_store_8_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    val: u8,
    access_byte: u8,
) {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.bus.store_8(addr, val, access);
}

/// Phase-4 helper: signed halfword load (LDSH). Wraps cpu.aot_ldr_sign_half
/// (= cpu.ldr_sign_half) which handles the misaligned case: if addr & 1,
/// do a sign-extended byte load instead. Returns u32 (sign-extended).
pub type AotLdrSignHalfFn = unsafe extern "C" fn(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32;

pub unsafe extern "C" fn aot_ldr_sign_half_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    addr: u32,
    access_byte: u8,
) -> u32 {
    use arm7tdmi::memory::MemoryAccess;
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    let access = if access_byte == 1 { MemoryAccess::Seq } else { MemoryAccess::NonSeq };
    cpu.aot_ldr_sign_half(addr, access)
}

/// Phase-8 ARM step trampoline. Mirrors `aot_thumb_step_for` but
/// dispatches via ARM_LUT instead of THUMB_LUT and uses 32-bit fetch.
pub unsafe extern "C" fn aot_arm_step_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    fetch_addr: u32,
    insn: u32,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    cpu.aot_arm_step(fetch_addr, insn)
}

pub unsafe extern "C" fn aot_block_should_abort_arm_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    if cpu.aot_block_should_abort_arm() { 1 } else { 0 }
}

/// Phase-1 mid-block abort check (K=2 cadence, called from
/// compile_thumb_block before iters with k odd && k != 0).
/// Returns 1 if the AOT block should yield to the dispatcher
/// (mode-flip, block-cache-dirty, or scheduler abort).
pub type AotAbortFn = unsafe extern "C" fn(cpu_ctx: *mut u8) -> u32;

pub unsafe extern "C" fn aot_block_should_abort_thumb_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };
    if cpu.aot_block_should_abort_thumb() { 1 } else { 0 }
}

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

/// Phase-8 whole-block ARM trampoline. Mirrors `aot_replay_thumb_block_for`
/// but uses 4-byte-stride per-iter (ARM is 32-bit) and ARM-specific
/// mode-flip / step / abort helpers.
#[inline]
pub unsafe extern "C" fn aot_replay_arm_block_for<I: MemoryInterface>(
    cpu_ctx: *mut u8,
    opcodes_ptr: *const u32,
    opcodes_len: u32,
    entry_pc: u32,
) -> u32 {
    let cpu = unsafe { &mut *(cpu_ctx as *mut Arm7tdmiCore<I>) };

    for k in 0..opcodes_len {
        if k != 0 && (k & ABORT_CADENCE_MASK) == 1 && cpu.aot_block_should_abort_arm() {
            return 0b10;
        }

        let insn = unsafe { *opcodes_ptr.add(k as usize) };
        let exec_addr = entry_pc.wrapping_add(4u32.wrapping_mul(k));
        let fetch_addr = exec_addr.wrapping_add(8);

        if cpu.aot_arm_step(fetch_addr, insn) == 1 {
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
