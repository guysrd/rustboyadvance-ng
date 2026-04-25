//! Static reachability scan for ROM (per A4 — see
//! `docs/findings-rom-scan.md`).
//!
//! Walks basic blocks starting from given entry points, decoding
//! enough of each instruction to classify control flow:
//! Linear / DirectBranch / Conditional / IndirectBranch /
//! ExceptionEdge. Terminates each scan path on indirect branch or
//! exception. Direct branch targets queue for further scanning.
//! Caps each block at 32 instructions per I16 (the rest split into
//! a chained sub-block via fall-through).
//!
//! Phase-0 scaffold: produces `Vec<BlockSpec>` for the AOT compile
//! pass to walk. The actual IR emission for each block is the next
//! step.

use crate::bus;

/// CPU mode (Thumb or ARM). Local copy to avoid pulling the full
/// arm7tdmi::CpuState dep into scan logic; converted at call
/// boundaries.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Mode {
    Arm,
    Thumb,
}

impl Mode {
    #[inline]
    pub fn insn_bytes(self) -> u32 {
        match self {
            Mode::Arm => 4,
            Mode::Thumb => 2,
        }
    }
}

/// A reachable block found by the scan. Phase-0 emit takes one of
/// these and produces an LLVM fn placed in the AotTable at
/// `entry_pc`.
#[derive(Clone, Debug)]
pub struct BlockSpec {
    pub entry_pc: u32,
    pub mode: Mode,
    /// Raw u32 opcodes (Thumb upper bits zero). One entry per
    /// instruction.
    pub opcodes: Vec<u32>,
    pub end_kind: BlockEnd,
}

/// How a block ends. Drives queue-target generation in scan_rom.
#[derive(Clone, Debug)]
pub enum BlockEnd {
    /// Direct unconditional branch with statically-known target +
    /// mode. The target is queued for further scanning.
    DirectBranch { target: u32, mode: Mode },
    /// Conditional branch (Bcc or ARM cond field): both target and
    /// fall-through are live. Both queued.
    Conditional { target: u32 },
    /// Branch-with-link (BL): target queued; the return address is
    /// dynamic (BX LR later) and is handled by the indirect-branch
    /// fallback (A3 — scalar replay for that path).
    BlReturn { target: u32 },
    /// BX register / LDR PC / POP {pc,...} / LDM with R15 in rlist —
    /// target only known at runtime. Terminates this scan path.
    IndirectBranch,
    /// SWI / undefined / unimplemented — terminates this scan path.
    ExceptionEdge,
    /// Hit the I16 per-block instruction cap. Fall-through is live.
    BlockLengthCap,
}

/// Per-block instruction cap (I16).
pub const MAX_BLOCK_INSTRS: usize = 32;

/// Decode the cartridge entry-point B instruction at ROM byte 0
/// per A4. Returns the target PC (in 0x08000000+ space) where game
/// code actually starts.
pub fn cart_entry_pc(rom: &[u8]) -> Option<u32> {
    if rom.len() < 4 {
        return None;
    }
    // ROM byte 0..3 (LE) = ARM B instruction at pc 0x08000000.
    let raw = u32::from_le_bytes([rom[0], rom[1], rom[2], rom[3]]);
    // Validate: cond=AL (0xE), opcode bits 27:25 = 101, L bit clear.
    let cond = (raw >> 28) & 0xF;
    let kind = (raw >> 25) & 0x7;
    let l = (raw >> 24) & 0x1;
    if cond != 0xE || kind != 0b101 || l != 0 {
        return None;
    }
    // 24-bit signed offset, sign-extended, shifted left by 2,
    // added to (pc + 8) where pc here is 0x08000000.
    let off24 = raw & 0x00FF_FFFF;
    let signed = if off24 & 0x0080_0000 != 0 {
        (off24 | 0xFF00_0000) as i32
    } else {
        off24 as i32
    };
    let target = 0x0800_0008u32.wrapping_add((signed as u32).wrapping_mul(4));
    Some(target)
}

/// Validate a candidate target per I22. Returns true if scan
/// should follow it.
fn validate_target(pc: u32, mode: Mode, rom_base: u32, rom_len: usize) -> bool {
    // Mode-correct alignment.
    match mode {
        Mode::Thumb => {
            if pc & 1 != 0 {
                return false;
            }
        }
        Mode::Arm => {
            if pc & 3 != 0 {
                return false;
            }
        }
    }
    // Inside ROM (BIOS or cartridge).
    in_rom_range(pc, rom_base, rom_len)
}

fn in_rom_range(pc: u32, rom_base: u32, rom_len: usize) -> bool {
    pc >= rom_base && (pc as u64) < (rom_base as u64 + rom_len as u64)
}

/// Read a Thumb halfword from ROM at the given pc (assuming pc is
/// in [rom_base, rom_base+rom_len)).
fn read_thumb_at(rom: &[u8], rom_base: u32, pc: u32) -> Option<u16> {
    let off = (pc - rom_base) as usize;
    if off + 1 >= rom.len() {
        return None;
    }
    Some(u16::from_le_bytes([rom[off], rom[off + 1]]))
}

/// Read an ARM word from ROM at the given pc.
fn read_arm_at(rom: &[u8], rom_base: u32, pc: u32) -> Option<u32> {
    let off = (pc - rom_base) as usize;
    if off + 3 >= rom.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        rom[off],
        rom[off + 1],
        rom[off + 2],
        rom[off + 3],
    ]))
}

/// Lightweight per-instruction classifier. Returns one of the four
/// CFG-relevant outcomes; doesn't attempt full decode.
#[derive(Clone, Debug)]
enum Class {
    /// Falls through to next instruction.
    Linear,
    /// Block terminator with statically-known target.
    Branch(BlockEnd),
}

fn classify_thumb(op: u16, pc: u32) -> Class {
    let top4 = op >> 12;
    let top5 = (op >> 11) as u32;
    let top8 = (op >> 8) as u32;

    match top4 {
        // F16 Bcc / SWI.
        0b1101 => {
            let cond = top8 & 0xF;
            if cond == 0xF {
                // SWI (1101_1111_imm8).
                Class::Branch(BlockEnd::ExceptionEdge)
            } else if cond == 0xE {
                // 0b1101_1110 is undefined.
                Class::Branch(BlockEnd::ExceptionEdge)
            } else {
                // Bcc imm8: target = pc + 4 + sign_extend(imm8) << 1.
                let imm8 = (op & 0xFF) as i8 as i32;
                let target = pc.wrapping_add(4).wrapping_add((imm8 as u32) << 1);
                Class::Branch(BlockEnd::Conditional { target })
            }
        }
        // F18 B (1110_0xxxxxxxxxxx).
        0b1110 => {
            // 11-bit signed offset, shifted left by 1.
            let off11 = (op & 0x07FF) as u32;
            let signed = if off11 & 0x0400 != 0 {
                (off11 | 0xFFFF_F800) as i32
            } else {
                off11 as i32
            };
            let target = pc.wrapping_add(4).wrapping_add((signed as u32) << 1);
            Class::Branch(BlockEnd::DirectBranch {
                target,
                mode: Mode::Thumb,
            })
        }
        // F19 BL pair handled at the block level (caller looks at hi+lo).
        0b1111 => {
            // F19 hi (top5 = 0b11110): linear, the next halfword is the lo
            // and the pair completes there.
            // F19 lo (top5 = 0b11111): always preceded by F19 hi in well-formed
            // code; classify standalone as exception (orphan lo). For phase 0
            // we emit Linear on hi and the caller's block walker special-cases
            // the pair.
            if top5 == 0b11110 {
                Class::Linear
            } else {
                // F19 lo without preceding hi — orphan, treat as exception.
                Class::Branch(BlockEnd::ExceptionEdge)
            }
        }
        // Format 5 high-register ops including BX.
        0b0100 => {
            // BX: 0b0100_0111_xxxx (mask 0xFF87 == 0x4700; bit 7 distinguishes).
            if (op & 0xFF87) == 0x4700 {
                Class::Branch(BlockEnd::IndirectBranch)
            } else {
                // F4/F5 ALU/high-reg — Linear (high-reg writes to PC are
                // possible but rare; treat as Linear and let scan miss them
                // for now — they fall to scalar via the indirect-fallback).
                Class::Linear
            }
        }
        // Format 14 PUSH/POP with PC variants.
        0b1011 => {
            // POP {...,pc}: 0b1011_1101_xxxx_xxxx (mask 0xFF00 == 0xBD00).
            if (op & 0xFF00) == 0xBD00 {
                Class::Branch(BlockEnd::IndirectBranch)
            } else {
                Class::Linear
            }
        }
        _ => Class::Linear,
    }
}

fn classify_arm(op: u32, pc: u32) -> Class {
    // Cond field. cond=NV (0xF) is mostly reserved; treat as exception.
    let cond = (op >> 28) & 0xF;
    if cond == 0xF {
        return Class::Branch(BlockEnd::ExceptionEdge);
    }

    // SWI: (op >> 24) & 0xF == 0xF.
    if (op >> 24) & 0xF == 0xF {
        return Class::Branch(BlockEnd::ExceptionEdge);
    }

    // BX: 0001_0010_1111_1111_1111_0001_Rm (after cond).
    if (op & 0x0FFF_FFF0) == 0x012F_FF10 {
        return Class::Branch(BlockEnd::IndirectBranch);
    }

    // B / BL: bits 27:25 = 101.
    if (op >> 25) & 0x7 == 0b101 {
        let l = (op >> 24) & 0x1;
        let off24 = op & 0x00FF_FFFF;
        let signed = if off24 & 0x0080_0000 != 0 {
            (off24 | 0xFF00_0000) as i32
        } else {
            off24 as i32
        };
        let target = pc.wrapping_add(8).wrapping_add((signed as u32).wrapping_mul(4));
        if l == 0 {
            // Conditional B if cond != AL, else direct.
            if cond == 0xE {
                return Class::Branch(BlockEnd::DirectBranch {
                    target,
                    mode: Mode::Arm,
                });
            } else {
                return Class::Branch(BlockEnd::Conditional { target });
            }
        } else {
            // BL: target queued, block continues at fall-through (lr = pc+4
            // restores after subroutine). Treat as block terminator that
            // queues the target.
            return Class::Branch(BlockEnd::BlReturn { target });
        }
    }

    // LDM with R15 in rlist (bit 15 set): IndirectBranch.
    // 100xx1xx_xxxxxxxx_1xxxxxxx_xxxxxxxx (LDM = bit 20 = 1; R15 in rlist
    // = bit 15 = 1).
    if (op >> 25) & 0x7 == 0b100 && (op >> 20) & 0x1 == 1 && (op >> 15) & 0x1 == 1 {
        return Class::Branch(BlockEnd::IndirectBranch);
    }

    // LDR Rd=PC: bit 27:25 = 010 or 011 (LDR/STR), bit 20 = 1 (load),
    // bits 15:12 = 1111 (Rd=PC).
    let kind = (op >> 25) & 0x7;
    if (kind == 0b010 || kind == 0b011) && (op >> 20) & 0x1 == 1 && (op >> 12) & 0xF == 0xF {
        return Class::Branch(BlockEnd::IndirectBranch);
    }

    // ALU with Rd = PC (bits 15:12 = 1111).
    if (op >> 26) & 0x3 == 0b00 && (op >> 12) & 0xF == 0xF {
        // Could be MSR (cond_00010xx0_x1111_xxxxxxx_xxxxxxxx) which writes
        // PSR, not PC; cheap distinguish: bit 23 = 1 and bit 21:20 = 10
        // marks MSR. Skip this refinement for now — false positives just
        // reject the block which is conservative.
        return Class::Branch(BlockEnd::IndirectBranch);
    }

    Class::Linear
}

/// Scan one block starting at `pc`. Returns the BlockSpec or None
/// if pc is outside ROM.
fn scan_one_block(rom: &[u8], rom_base: u32, pc: u32, mode: Mode) -> Option<BlockSpec> {
    if !in_rom_range(pc, rom_base, rom.len()) {
        return None;
    }
    let mut opcodes: Vec<u32> = Vec::with_capacity(8);
    let mut cur = pc;
    let bytes = mode.insn_bytes();

    // F19 BL pair tracking: when we see F19 hi we MUST also consume
    // the next halfword as F19 lo. The pair is one logical branch.
    let mut prev_was_f19_hi = false;

    for _ in 0..MAX_BLOCK_INSTRS {
        if !in_rom_range(cur, rom_base, rom.len()) {
            // Off the end — treat as IndirectBranch terminator.
            return Some(BlockSpec {
                entry_pc: pc,
                mode,
                opcodes,
                end_kind: BlockEnd::IndirectBranch,
            });
        }

        let raw: u32 = match mode {
            Mode::Thumb => read_thumb_at(rom, rom_base, cur)? as u32,
            Mode::Arm => read_arm_at(rom, rom_base, cur)?,
        };
        opcodes.push(raw);

        // F19 BL pair: if previous was hi, this halfword IS the lo and
        // completes the pair. The pair acts as a BlReturn.
        if prev_was_f19_hi {
            // Compute BL target from hi (opcodes[len-2]) + lo (raw).
            let hi = opcodes[opcodes.len() - 2] as u16;
            let lo = raw as u16;
            let pc_at_hi = cur.wrapping_sub(2);  // hi was 2 bytes before lo
            let target = thumb_bl_target(pc_at_hi, hi, lo);
            return Some(BlockSpec {
                entry_pc: pc,
                mode,
                opcodes,
                end_kind: BlockEnd::BlReturn { target },
            });
        }

        let class = match mode {
            Mode::Thumb => classify_thumb(raw as u16, cur),
            Mode::Arm => classify_arm(raw, cur),
        };

        match class {
            Class::Linear => {
                // Check if this was F19 hi — if so, set the flag and
                // continue (the next iter will consume lo).
                if mode == Mode::Thumb && (raw as u16 >> 11) == 0b11110 {
                    prev_was_f19_hi = true;
                }
                cur = cur.wrapping_add(bytes);
            }
            Class::Branch(end) => {
                return Some(BlockSpec {
                    entry_pc: pc,
                    mode,
                    opcodes,
                    end_kind: end,
                });
            }
        }
    }

    // Hit length cap.
    Some(BlockSpec {
        entry_pc: pc,
        mode,
        opcodes,
        end_kind: BlockEnd::BlockLengthCap,
    })
}

fn thumb_bl_target(pc_at_hi: u32, hi: u16, lo: u16) -> u32 {
    // F19 hi: 11110_HHHHHHHHHHH (high 11 bits, signed when sign-extended).
    // F19 lo: 11111_LLLLLLLLLLL (low 11 bits, unsigned).
    // target = pc_at_hi + 4 + (sign_ext(hi[10:0]) << 12) + (lo[10:0] << 1)
    let hi_off = (hi & 0x07FF) as u32;
    let signed_hi = if hi_off & 0x0400 != 0 {
        (hi_off | 0xFFFF_F800) as i32
    } else {
        hi_off as i32
    };
    let lo_off = (lo & 0x07FF) as u32;
    pc_at_hi
        .wrapping_add(4)
        .wrapping_add((signed_hi as u32) << 12)
        .wrapping_add(lo_off << 1)
}

/// The phase-0 scan entry. Walks reachable blocks from given entry
/// points, returns Vec<BlockSpec> for the compile pass.
///
/// `rom` is the full ROM byte slice (BIOS, cartridge — caller picks).
/// `rom_base` is the GBA-space base address (e.g. 0x08000000 for cart).
pub fn scan_rom(rom: &[u8], rom_base: u32, entry_points: Vec<(u32, Mode)>) -> Vec<BlockSpec> {
    use std::collections::{HashSet, VecDeque};
    let mut visited: HashSet<(u32, Mode)> = HashSet::new();
    let mut queue: VecDeque<(u32, Mode)> = entry_points.into_iter().collect();
    let mut blocks: Vec<BlockSpec> = Vec::new();

    while let Some((pc, mode)) = queue.pop_front() {
        if !validate_target(pc, mode, rom_base, rom.len()) {
            continue;
        }
        if visited.contains(&(pc, mode)) {
            continue;
        }
        visited.insert((pc, mode));

        // Skip pcs that map to IO regions (per I3, they wouldn't be code).
        // Note: rom_base is BIOS or cartridge; both are non-IO. This guards
        // against bogus targets.
        if bus::is_io_addr(pc) {
            continue;
        }

        let block = match scan_one_block(rom, rom_base, pc, mode) {
            Some(b) => b,
            None => continue,
        };

        // Queue successors.
        match &block.end_kind {
            BlockEnd::DirectBranch { target, mode: tm } => {
                queue.push_back((*target, *tm));
            }
            BlockEnd::Conditional { target } => {
                queue.push_back((*target, mode));
                let ft = pc.wrapping_add(block.opcodes.len() as u32 * mode.insn_bytes());
                queue.push_back((ft, mode));
            }
            BlockEnd::BlReturn { target } => {
                queue.push_back((*target, mode));
                // Return path is dynamic — A3 fallback handles via scalar.
            }
            BlockEnd::BlockLengthCap => {
                let ft = pc.wrapping_add(block.opcodes.len() as u32 * mode.insn_bytes());
                queue.push_back((ft, mode));
            }
            BlockEnd::IndirectBranch | BlockEnd::ExceptionEdge => {
                // No queue.
            }
        }
        blocks.push(block);
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cart_entry_pokeemerald_pattern() {
        // ROM bytes 7f 00 00 ea = ARM B with offset 0x7f, target = 0x08000204.
        let mut rom = vec![0u8; 1024];
        rom[0..4].copy_from_slice(&[0x7f, 0x00, 0x00, 0xea]);
        let target = cart_entry_pc(&rom).expect("decode B");
        assert_eq!(target, 0x0800_0204);
    }

    #[test]
    fn cart_entry_mks_pattern() {
        // ROM bytes 2e 00 00 ea = ARM B with offset 0x2e, target = 0x080000c0.
        let mut rom = vec![0; 1024];
        rom[0..4].copy_from_slice(&[0x2e, 0x00, 0x00, 0xea]);
        let target = cart_entry_pc(&rom).expect("decode B");
        assert_eq!(target, 0x0800_00c0);
    }

    #[test]
    fn cart_entry_rejects_non_b() {
        // 0x00000000 = invalid (cond=0, kind=0).
        let rom = vec![0; 1024];
        assert!(cart_entry_pc(&rom).is_none());
    }

    #[test]
    fn classify_thumb_b_unconditional() {
        // F18 B with offset 4: pc + 4 + (4 << 1) = pc + 12.
        let op = 0b11100_00000000100u16; // top5=11100 + offset 4
        let pc = 0x0800_0000;
        match classify_thumb(op, pc) {
            Class::Branch(BlockEnd::DirectBranch { target, mode }) => {
                assert_eq!(target, 0x0800_000c);
                assert_eq!(mode, Mode::Thumb);
            }
            other => panic!("expected DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn classify_thumb_bcc() {
        // F16 Bcc with cond=0 (EQ), offset 4: pc + 4 + (4 << 1) = pc + 12.
        let op = 0b1101_0000_0000_0100u16;
        let pc = 0x0800_0000;
        match classify_thumb(op, pc) {
            Class::Branch(BlockEnd::Conditional { target }) => {
                assert_eq!(target, 0x0800_000c);
            }
            other => panic!("expected Conditional, got {:?}", other),
        }
    }

    #[test]
    fn classify_thumb_bx() {
        // F5 BX r0: 0x4700.
        let op = 0x4700u16;
        match classify_thumb(op, 0x0800_0000) {
            Class::Branch(BlockEnd::IndirectBranch) => {}
            other => panic!("expected IndirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn classify_thumb_swi() {
        // SWI 0: 0xDF00.
        match classify_thumb(0xDF00u16, 0x0800_0000) {
            Class::Branch(BlockEnd::ExceptionEdge) => {}
            other => panic!("expected ExceptionEdge, got {:?}", other),
        }
    }

    #[test]
    fn classify_thumb_pop_pc() {
        // POP {pc}: 0xBD00 (rlist = 0).
        match classify_thumb(0xBD00u16, 0x0800_0000) {
            Class::Branch(BlockEnd::IndirectBranch) => {}
            other => panic!("expected IndirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn classify_thumb_linear_alu() {
        // F3 MOV r0, #5: 0x2005.
        match classify_thumb(0x2005u16, 0x0800_0000) {
            Class::Linear => {}
            other => panic!("expected Linear, got {:?}", other),
        }
    }

    #[test]
    fn classify_arm_b_unconditional() {
        // EA00007F = AL B with offset 0x7f, target = pc + 8 + 0x1fc.
        let pc = 0x0800_0000;
        match classify_arm(0xEA00_007F, pc) {
            Class::Branch(BlockEnd::DirectBranch { target, mode }) => {
                assert_eq!(target, 0x0800_0204);
                assert_eq!(mode, Mode::Arm);
            }
            other => panic!("expected DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn classify_arm_bx() {
        // BX r0: cond=AL, opcode 012FFF10.
        match classify_arm(0xE12F_FF10, 0x0800_0000) {
            Class::Branch(BlockEnd::IndirectBranch) => {}
            other => panic!("expected IndirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn classify_arm_swi() {
        // SWI 0: cond=AL, opcode 0xEFnnnnnn.
        match classify_arm(0xEF00_0000, 0x0800_0000) {
            Class::Branch(BlockEnd::ExceptionEdge) => {}
            other => panic!("expected ExceptionEdge, got {:?}", other),
        }
    }

    #[test]
    fn scan_tiny_thumb_program() {
        // Thumb program at rom_base 0x08000000:
        //   0: MOV r0, #5     (F3 imm)        bytes: 05 20
        //   2: MOV r1, #6                     bytes: 06 21
        //   4: SWI 0          (terminates)    bytes: 00 df
        // Single block, ends in ExceptionEdge — scan doesn't queue
        // anything past the SWI.
        let mut rom = vec![0u8; 1024];
        rom[..6].copy_from_slice(&[0x05, 0x20, 0x06, 0x21, 0x00, 0xdf]);
        let blocks = scan_rom(&rom, 0x0800_0000, vec![(0x0800_0000, Mode::Thumb)]);
        assert_eq!(blocks.len(), 1, "expected 1 block, got {}: {:?}", blocks.len(), blocks);
        let b = &blocks[0];
        assert_eq!(b.entry_pc, 0x0800_0000);
        assert_eq!(b.mode, Mode::Thumb);
        assert_eq!(b.opcodes, vec![0x2005, 0x2106, 0xdf00]);
        assert!(matches!(b.end_kind, BlockEnd::ExceptionEdge));
    }

    /// Smoke test against the real BIOS — should produce a moderate
    /// number of reachable blocks (BIOS is 16KB, has SWI dispatch
    /// table + several handlers).
    #[test]
    fn scan_bios_smoke() {
        let path = "core/benches/roms/normatt_gba_bios.bin";
        let rom = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return,
        };
        eprintln!("BIOS size: {} bytes", rom.len());
        // BIOS reset vector at pc=0 ARM. Plus the standard exception
        // vectors at 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C.
        let entries: Vec<(u32, Mode)> = (0..8).map(|i| (i * 4, Mode::Arm)).collect();
        let blocks = scan_rom(&rom, 0, entries);
        eprintln!("scanned {} blocks from BIOS exception vectors", blocks.len());
        assert!(blocks.len() >= 5, "BIOS scan should find >=5 blocks, got {}", blocks.len());
    }

    /// Smoke test against a real ROM: scan from cartridge entry,
    /// confirm we find a non-trivial number of blocks. Skipped if
    /// the ROM file isn't present (so CI without the ROM doesn't
    /// fail).
    #[test]
    fn scan_pokeemerald_smoke() {
        let path = "/home/user/pokeemerlad/pokeemerald/pokeemerald.gba";
        let rom = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let entry = cart_entry_pc(&rom).expect("cart entry decode");
        eprintln!("pokeemerald cart entry pc: {:#010x}", entry);
        // Scan from cartridge entry in ARM mode.
        // The entry usually does ARM setup then BX r0 / BX lr to switch
        // to Thumb. Our scan terminates at the indirect branch; it'll
        // produce 1-3 blocks in the ARM bootstrap.
        let blocks = scan_rom(&rom, 0x0800_0000, vec![(entry, Mode::Arm)]);
        eprintln!("scanned {} blocks from ARM entry", blocks.len());
        assert!(blocks.len() >= 1, "expected at least 1 block from cart entry");
        // Each block should have at least 1 opcode.
        for b in &blocks {
            assert!(!b.opcodes.is_empty());
        }
    }

    #[test]
    fn scan_with_direct_branch_queues_target() {
        // Thumb program: B forward by 4 halfwords, then a SWI at target.
        // Forward B:
        //   target = pc + 4 + (off11 << 1) = 0x08000000 + 4 + (off11 << 1)
        //   want target = 0x08000010 (16 bytes ahead = 8 halfwords)
        //   off11 = (0x10 - 4) >> 1 = 6
        //   F18 B with off=6: top5=11100, body=0x006 → 0xe006.
        let mut rom = vec![0u8; 0x40];
        // pc=0: B +6 (target = 0x08000010)
        rom[0..2].copy_from_slice(&[0x06, 0xe0]);
        // pc=0x10: SWI 0 (terminator)
        rom[0x10..0x12].copy_from_slice(&[0x00, 0xdf]);
        // The bytes between are 0; classifier will see them as Linear
        // (top4 of 0x0000 = 0b0000 = F1 — Linear). They WILL be scanned
        // if reached; but the B at pc=0 jumps over them, so they're
        // not queued.
        let blocks = scan_rom(&rom, 0x0800_0000, vec![(0x0800_0000, Mode::Thumb)]);
        assert_eq!(blocks.len(), 2, "expected B-block + target-block, got {}", blocks.len());
        // Block 0: the B itself.
        assert_eq!(blocks[0].entry_pc, 0x0800_0000);
        assert!(matches!(
            blocks[0].end_kind,
            BlockEnd::DirectBranch { target: 0x0800_0010, mode: Mode::Thumb }
        ));
        // Block 1: the SWI.
        assert_eq!(blocks[1].entry_pc, 0x0800_0010);
        assert!(matches!(blocks[1].end_kind, BlockEnd::ExceptionEdge));
    }
}
