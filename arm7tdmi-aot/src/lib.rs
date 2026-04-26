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
pub mod replay;
pub mod scan;
pub mod table;

pub use compiler::{CpuOffsets, LlvmCompiler};
pub use scan::{BlockEnd, BlockSpec, Mode, scan_rom};
pub use table::{AotTable, CompiledFn, aot_lookup, aot_lookup_arm};

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

/// Phase-8 ARM-mode lookup hook. Dispatcher selects between this and
/// `aot_lookup_for_hook` based on cpu.cpsr.state() at dispatch time.
fn aot_lookup_arm_for_hook(table_ptr: *const u8, pc: u32) -> usize {
    if table_ptr.is_null() {
        return 0;
    }
    let table = unsafe { &*(table_ptr as *const AotTable) };
    match aot_lookup_arm(table, pc) {
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
///
/// Phase-8: also installs the ARM lookup hook. Dispatcher uses
/// thumb hook when cpsr=THUMB and arm hook when cpsr=ARM.
pub fn enable_aot_on<I: MemoryInterface>(cpu: &mut Arm7tdmiCore<I>, table: &AotTable) {
    let table_ptr = table as *const AotTable as *const u8;
    cpu.install_aot_hook(table_ptr, aot_lookup_for_hook);
    cpu.install_aot_hook_arm(aot_lookup_arm_for_hook);
}

/// Phase-0 entry point. Scan ROM, emit a placeholder block for each
/// reachable Thumb block, return a populated AotTable. ARM blocks are
/// SKIPPED in phase 0 (they fall to scalar via the AOT-miss path).
///
/// `rom`        — full ROM bytes (cartridge or BIOS).
/// `rom_base`   — GBA address-space base for `rom` (e.g. `0x08000000`
///                for cart, `0x00000000` for BIOS).
/// `entry_pc`   — initial entry point. For cart, decode via
///                `scan::cart_entry_pc(rom)` first. For BIOS, pass 0.
/// `entry_mode` — the entry's CPU mode (typically `Mode::Arm`).
/// `replay_thumb_fn` — caller-supplied per-I monomorphized
///                trampoline (e.g.
///                `arm7tdmi_aot::replay::aot_replay_thumb_block_for::<SysBus>`).
pub fn compile_rom(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
    replay_thumb_fn: replay::AotReplayFn,
) -> AotTable {
    compile_rom_with_seeds(rom, rom_base, entry_pc, entry_mode, &[], replay_thumb_fn)
}

/// Variant that takes additional seed entry points (typically from a
/// trace-out file). Each seed adds a (pc, mode) pair to the scan
/// queue alongside the static entry_pc + sweep candidates.
pub fn compile_rom_with_seeds(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
    seeds: &[(u32, Mode)],
    replay_thumb_fn: replay::AotReplayFn,
) -> AotTable {
    compile_rom_with_seeds_and_step(
        rom,
        rom_base,
        entry_pc,
        entry_mode,
        seeds,
        replay_thumb_fn,
        None,
        None,
    )
}

/// Phase-1 variant: takes per-instruction step + abort trampolines so
/// `compile_thumb_block` can dispatch per-opcode (per-format inline
/// IR or step trampoline call) instead of one whole-block trampoline.
/// When step+abort are None, falls back to the phase-0 whole-block
/// emit. When both are Some, uses the new per-instruction emit.
pub fn compile_rom_with_seeds_and_step(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
    seeds: &[(u32, Mode)],
    replay_thumb_fn: replay::AotReplayFn,
    step_thumb_fn: Option<replay::AotStepFn>,
    abort_thumb_fn: Option<replay::AotAbortFn>,
) -> AotTable {
    compile_rom_with_seeds_step_offsets(
        rom, rom_base, entry_pc, entry_mode, seeds,
        replay_thumb_fn, step_thumb_fn, abort_thumb_fn, None, None,
    )
}

/// Phase-4 variant of compile_rom_with_seeds_and_step that ALSO
/// takes optional cpu state offsets and a fetch-only trampoline.
/// When both `cpu_offsets` and `fetch_only_thumb_fn` are Some,
/// emit_per_instr_thumb_block can emit inline LLVM IR for selected
/// formats (F3 MOV imm8 first; more to come). When either is None,
/// falls back to per-iter trampoline calls (phase 1 behavior).
pub fn compile_rom_with_seeds_step_offsets(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
    seeds: &[(u32, Mode)],
    replay_thumb_fn: replay::AotReplayFn,
    step_thumb_fn: Option<replay::AotStepFn>,
    abort_thumb_fn: Option<replay::AotAbortFn>,
    cpu_offsets: Option<CpuOffsets>,
    fetch_only_thumb_fn: Option<replay::AotFetchOnlyFn>,
) -> AotTable {
    compile_rom_with_seeds_full(
        rom, rom_base, entry_pc, entry_mode, seeds,
        replay_thumb_fn, step_thumb_fn, abort_thumb_fn,
        cpu_offsets, fetch_only_thumb_fn, None,
    )
}

/// Phase-8 variant that also accepts an ARM whole-block trampoline.
/// When `replay_arm_fn` is Some, ARM specs are emitted via
/// `emit_placeholder_arm_block` and inserted into the AotTable's ARM
/// page lookup. When None, ARM specs are skipped (phase-1 behavior).
pub fn compile_rom_with_seeds_full(
    rom: &[u8],
    rom_base: u32,
    entry_pc: u32,
    entry_mode: Mode,
    seeds: &[(u32, Mode)],
    replay_thumb_fn: replay::AotReplayFn,
    step_thumb_fn: Option<replay::AotStepFn>,
    abort_thumb_fn: Option<replay::AotAbortFn>,
    cpu_offsets: Option<CpuOffsets>,
    fetch_only_thumb_fn: Option<replay::AotFetchOnlyFn>,
    replay_arm_fn: Option<replay::AotReplayFn>,
) -> AotTable {
    // Phase-0 scan strategy: static reachability from the supplied
    // entry can't get past the first indirect branch. To get >0%
    // coverage on the SDL replay we ALSO sweep aligned halfwords as
    // candidate Thumb entries. The scanner's I22 validation filters
    // out non-code halfwords; over-approximation is harmless
    // (never dispatched at runtime).
    //
    // Compile time bound (per A5): ~280 µs/block emit. To fit phase 0
    // budget we cap sweep at 64KB per ROM. That gives 32k candidates,
    // ~5-10k unique blocks after dedup, ~3 seconds compile. Coverage
    // will be partial; phase 0 follow-up does parallel-by-page +
    // larger cap, OR phase 9 trace-pass (per A3) replaces sweep with
    // runtime-observed PCs.
    let mut entries: Vec<(u32, Mode)> = vec![(entry_pc, entry_mode)];
    // Trace-driven seeds (typically from --aot-trace-in). These are
    // runtime-observed block entry pcs — they're real entries, no
    // sweep-style false positives. THE primary path for getting
    // coverage > 80%.
    entries.extend_from_slice(seeds);
    eprintln!("AOT: {} seed entry points (from --aot-trace-in)", seeds.len());
    // Sweep cap: 0 = disabled (table only contains the static-reachable
    // entry; coverage near 0% but divs == 0 trivially). Default 0 for
    // phase 0 step 4b — empirical: sweep > 0 causes MK to diverge by
    // 4 frames vs scalar (PE remains 0 divs even with sweep=64KB).
    // The over-AOT'd blocks aren't being dispatched at non-block-entry
    // pcs (cpu.pc is always pipeline-head), so the divergence cause is
    // subtler than initially suspected — needs phase-0c investigation.
    // Override at runtime: AOT_SWEEP_CAP_KB=64 to re-enable for testing.
    let sweep_cap_kb: usize = std::env::var("AOT_SWEEP_CAP_KB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let sweep_cap_bytes: usize = sweep_cap_kb * 1024;
    let sweep_end = rom.len().min(sweep_cap_bytes);
    let arm_sweep = std::env::var("AOT_SWEEP_ARM").map(|v| v == "1").unwrap_or(false);
    if rom_base == 0x0800_0000 {
        // Skip the 0xC0-byte cartridge header.
        let mut off = 0xC0;
        while off + 1 < sweep_end {
            entries.push((rom_base.wrapping_add(off as u32), Mode::Thumb));
            off += 2;
        }
        if arm_sweep {
            let mut arm_off = 0xC0;
            // ARM is 4-byte aligned. Round up to next multiple of 4.
            arm_off = (arm_off + 3) & !3;
            while arm_off + 3 < sweep_end {
                entries.push((rom_base.wrapping_add(arm_off as u32), Mode::Arm));
                arm_off += 4;
            }
        }
    } else {
        // BIOS sweep (16KB total).
        let mut off = 0;
        while off + 1 < sweep_end {
            entries.push((rom_base.wrapping_add(off as u32), Mode::Thumb));
            off += 2;
        }
    }
    eprintln!(
        "AOT: queueing {} sweep entry candidates (arm sweep: {})",
        entries.len(), arm_sweep,
    );

    let blocks = scan_rom(rom, rom_base, entries);
    let mut compiler = match LlvmCompiler::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("AOT: failed to create LlvmCompiler: {} (table will be empty)", e);
            return AotTable::new();
        }
    };
    compiler.register_replay_thumb(replay_thumb_fn);
    let use_per_instr = match (step_thumb_fn, abort_thumb_fn) {
        (Some(s), Some(a)) => {
            compiler.register_step_thumb(s, a);
            true
        }
        _ => false,
    };
    if let Some(off) = cpu_offsets {
        compiler.register_cpu_offsets(off);
    }
    if let Some(fo) = fetch_only_thumb_fn {
        compiler.register_fetch_only_thumb(fo);
    }
    if let Some(arm) = replay_arm_fn {
        compiler.register_replay_arm(arm);
    }

    let mut table = AotTable::new();
    let mut emitted = 0usize;
    let mut skipped_arm = 0usize;
    let mut arm_emitted = 0usize;
    let mut emit_failed = 0usize;
    let arm_enabled = replay_arm_fn.is_some();
    for spec in &blocks {
        if spec.mode != Mode::Thumb {
            // Phase-8: emit ARM blocks too (when replay_arm_fn is set).
            if arm_enabled {
                let opcodes_ptr = table.intern_arm_opcodes(&spec.opcodes);
                let opcodes_len = spec.opcodes.len() as u32;
                match compiler.emit_placeholder_arm_block(opcodes_ptr, opcodes_len, spec.entry_pc) {
                    Some(f) => {
                        // ARM lookup_pc = exec_addr + 8 (pipeline-head pc).
                        let lookup_pc = spec.entry_pc.wrapping_add(8);
                        table.insert_arm(lookup_pc, f);
                        arm_emitted += 1;
                    }
                    None => {
                        emit_failed += 1;
                    }
                }
            } else {
                skipped_arm += 1;
            }
            continue;
        }
        let opcodes_u16: Vec<u16> = spec.opcodes.iter().map(|&o| o as u16).collect();
        let emit_result = if use_per_instr {
            compiler.emit_per_instr_thumb_block(&opcodes_u16, spec.entry_pc)
        } else {
            let opcodes_ptr = table.intern_opcodes(&spec.opcodes);
            let opcodes_len = spec.opcodes.len() as u32;
            compiler.emit_placeholder_thumb_block(opcodes_ptr, opcodes_len, spec.entry_pc)
        };
        match emit_result {
            Some(f) => {
                let lookup_pc = spec.entry_pc.wrapping_add(4);
                table.insert(lookup_pc, f);
                emitted += 1;
            }
            None => {
                emit_failed += 1;
            }
        }
    }
    eprintln!(
        "AOT: scanned {} blocks; emitted {} thumb, {} arm, skipped {} arm-no-trampoline, {} failed",
        blocks.len(), emitted, arm_emitted, skipped_arm, emit_failed
    );
    // Keep the compiler (and its ExecutionEngine) alive by
    // converting it to a Box<dyn> — dropped only when the AotTable
    // drops. Per I23 we drop+recreate engine across compile_rom
    // calls; for now, leak it via Box::leak so the JIT'd code
    // pages stay valid for the AotTable's life.
    // TODO(phase 9): proper engine ownership tied to AotTable lifetime.
    Box::leak(Box::new(compiler));
    table
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
