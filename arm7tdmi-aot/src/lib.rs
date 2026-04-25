//! AOT-LLVM dispatcher for the ARM7TDMI core.
//!
//! At ROM load time, scan reachable Thumb/ARM blocks, emit inline
//! LLVM IR for each, batch-compile the resulting modules, and
//! populate a fast PC->fn lookup table. The dispatcher in
//! `arm7tdmi::cache::replay_cached_block` queries the table before
//! falling through to the cached_interp scalar replay path.
//!
//! See `docs/aot-llvm-program.md` for the autoresearch program
//! (Karpathy format) that drives the development of this crate.
//! Invariants I1-I28 in that doc are NOT optional.

pub mod bus;
pub mod compiler;
pub mod scan;
pub mod table;

pub use compiler::LlvmCompiler;
pub use scan::{BlockEnd, BlockSpec, Mode, scan_rom};
pub use table::{AotTable, CompiledFn, aot_lookup};

use arm7tdmi::Arm7tdmiCore;
use arm7tdmi::memory::MemoryInterface;

/// `arm7tdmi`-side hook signature. The dispatcher calls
/// `aot_lookup_fn(table_ptr, pc)` and dispatches the returned
/// CompiledFn (cast from usize) when non-zero.
fn aot_lookup_for_hook(table_ptr: *const u8, pc: u32) -> usize {
    // SAFETY: arm7tdmi installed this fn alongside a
    // `*const AotTable` pointer (both via `enable_aot_on`); we cast
    // back to the same type. Lifetime: the AotTable is owned by the
    // bus side (per I12) for the process lifetime once enabled.
    if table_ptr.is_null() {
        return 0;
    }
    let table = unsafe { &*(table_ptr as *const AotTable) };
    match aot_lookup(table, pc) {
        Some(f) => f as usize,
        None => 0,
    }
}

/// Install an `AotTable` on the given CPU. Must be called BEFORE the
/// first `step_block` (per I11). The CPU keeps a raw pointer to the
/// table — caller is responsible for keeping the table alive (don't
/// drop the AotTable while the CPU is still using it).
///
/// Phase-0 scaffold note: this hands the table off as a raw pointer
/// because arm7tdmi doesn't link inkwell. The table is a plain
/// PC->fn map — no LLVM types in its public surface.
pub fn enable_aot_on<I: MemoryInterface>(cpu: &mut Arm7tdmiCore<I>, table: &AotTable) {
    let table_ptr = table as *const AotTable as *const u8;
    cpu.install_aot_hook(table_ptr, aot_lookup_for_hook);
}
