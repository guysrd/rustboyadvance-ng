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

/// Phase-0 entry point. Scan ROM + populate an AotTable with placeholder
/// blocks (none yet — phase-0 step-4b will add trampoline-mode emit).
///
/// `rom`        — full ROM bytes (cartridge or BIOS).
/// `rom_base`   — GBA address-space base for `rom` (e.g. `0x08000000`
///                for cart, `0x00000000` for BIOS).
/// `entry_pc`   — initial entry point. For cart, decode via
///                `scan::cart_entry_pc(rom)` first. For BIOS, pass 0.
/// `entry_mode` — the entry's CPU mode (typically `Mode::Arm`).
///
/// Returns an `AotTable` populated with no blocks for now — coverage
/// will be 0% until step-4b adds the emit. The returned table is
/// passed to `enable_aot_on(cpu, &table)` and the caller keeps it
/// alive for the CPU's lifetime.
pub fn compile_rom(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
) -> AotTable {
    let _blocks = scan_rom(rom, rom_base, vec![(entry_pc, entry_mode)]);
    // Phase-0 step-4a: scan runs but no IR is emitted yet. The
    // table starts empty; the dispatcher misses every lookup and
    // falls through to scalar replay. Phase-0 step-4b adds the
    // placeholder emit (one trampoline call per block).
    AotTable::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: compile_rom on the real pokeemerald ROM doesn't
    /// panic and returns a usable (empty) table. Skipped if ROM
    /// file isn't present.
    #[test]
    fn compile_rom_pokeemerald_smoke() {
        let path = "/home/user/pokeemerlad/pokeemerald/pokeemerald.gba";
        let rom = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let entry = scan::cart_entry_pc(&rom).expect("decode cart B");
        let table = compile_rom(&rom, 0x0800_0000, entry, Mode::Arm);
        // Phase-0 step-4a: no blocks compiled yet.
        assert_eq!(table.block_count(), 0);
    }
}
