# A4: ROM-scan strategy spec

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A4: the static
reachability algorithm that turns a ROM byte-blob into a set of
basic blocks AOT can compile.

## Verified entry-point conventions

Cartridge ROM bytes 0-3 are an ARM B instruction. Decoding:

```
ARM B encoding: cond[31:28] | 101[27:25] | L[24] | offset_24[23:0]
target = pc + 8 + (sign_extend(offset_24) << 2)
where pc = 0x08000000 (cartridge base) at the entry.
```

Confirmed on actual ROMs:

| ROM | bytes 0-3 (LE) | parsed insn | target |
|-----|----------------|-------------|--------|
| pokeemerald | `7f 00 00 ea` = 0xea00007f | B AL, off=0x7f | 0x08000204 |
| Mario Kart  | `2e 00 00 ea` = 0xea00002e | B AL, off=0x2e | 0x080000c0 |

Both target a PC AFTER the 0xC0-byte cartridge header. The header
itself (0x080000C0 = 0x08000000 + 0xC0 in game-relative addressing,
sometimes with first 4 bytes being the entry B itself) **must NOT
be scanned as code** — it's nintendo-logo + game-id + checksum
bytes.

## BIOS entry: `pc = 0x00000000`, ARM mode (reset vector).

## Scan algorithm

```
struct BlockSpec {
    entry_pc: u32,
    mode: CpuState,           // ARM or THUMB
    length_words: u32,        // # halfwords (Thumb) or words (ARM)
    opcodes: Vec<u32>,        // raw instruction values
    end_kind: BlockEnd,
}

enum BlockEnd {
    DirectBranch(u32, CpuState),  // target known, mode known
    BlReturn(u32),                // BL: target known, return is dynamic (lr will hold pc+4)
    Conditional(u32),             // Bcc: target known, fall-through is also live
    IndirectBranch,               // BX register / LDR PC / POP {pc} — terminates scan
    ExceptionEdge,                // SWI / undefined / unimplemented
    BlockLengthCap,               // hit per-block 32-instr cap (I16) — fall-through to next block
}

fn scan_rom(rom_bytes: &[u8], rom_base: u32, entry_points: Vec<(u32, CpuState)>) -> Vec<BlockSpec> {
    let mut visited = HashSet::new();   // (pc, mode) pairs already scanned
    let mut queue = entry_points.into_iter().collect::<VecDeque<_>>();
    let mut out = Vec::new();

    while let Some((pc, mode)) = queue.pop_front() {
        // I22 validation:
        if visited.contains(&(pc, mode)) { continue; }
        if !is_in_rom(pc, rom_base, rom_bytes.len()) { continue; }
        if mode == THUMB && (pc & 1) != 0 { continue; }
        if mode == ARM && (pc & 3) != 0 { continue; }

        visited.insert((pc, mode));
        let block = scan_one_block(rom_bytes, rom_base, pc, mode);
        match &block.end_kind {
            BlockEnd::DirectBranch(target, target_mode) => {
                queue.push_back((*target, *target_mode));
            }
            BlockEnd::Conditional(target) => {
                queue.push_back((*target, mode));
                // fall-through also live
                let ft = pc + block.length_words * mode.insn_bytes();
                queue.push_back((ft, mode));
            }
            BlockEnd::BlReturn(target) => {
                queue.push_back((*target, mode));
                // return address: lr = pc + 4 (Thumb) / pc + 4 (ARM after BL).
                // We don't queue it; the called subroutine ending in BX LR
                // is an indirect branch and terminates that scan path.
                // Return paths are picked up by the indirect-branch
                // fallback (A3) — scalar handles them.
            }
            BlockEnd::BlockLengthCap => {
                let ft = pc + block.length_words * mode.insn_bytes();
                queue.push_back((ft, mode));
            }
            BlockEnd::IndirectBranch | BlockEnd::ExceptionEdge => {
                // terminate this scan path
            }
        }
        out.push(block);
    }
    out
}

fn scan_one_block(rom: &[u8], rom_base: u32, pc: u32, mode: CpuState) -> BlockSpec {
    let mut opcodes = Vec::new();
    let mut cur_pc = pc;
    let insn_bytes = match mode { CpuState::ARM => 4, CpuState::THUMB => 2 };

    for _ in 0..32 {  // I16 per-block cap
        if !is_in_rom(cur_pc, rom_base, rom.len()) { break; }
        let raw = read_insn(rom, rom_base, cur_pc, mode);
        opcodes.push(raw);
        let class = classify_insn(raw, mode, cur_pc);
        match class {
            InsnClass::Linear => {
                cur_pc += insn_bytes;
            }
            InsnClass::DirectBranch(target, target_mode) => {
                return BlockSpec { entry_pc: pc, mode, length_words: opcodes.len() as u32,
                                   opcodes, end_kind: BlockEnd::DirectBranch(target, target_mode) };
            }
            // ... etc per BlockEnd variants ...
        }
    }
    // Hit cap → split per I16
    BlockSpec { ..., end_kind: BlockEnd::BlockLengthCap }
}
```

## Instruction classification

`classify_insn` is a lightweight decoder that returns one of:

```
enum InsnClass {
    Linear,                                  // ALU, mem, MUL, etc — fall through
    DirectBranch(u32, CpuState),             // B/BL — target known, mode known
    Conditional(u32),                        // Bcc — target known, fall-through also possible
    IndirectBranch,                          // BX Rm, LDR PC, POP {pc, ...}, etc
    ExceptionEdge,                           // SWI, undefined
}
```

For Thumb:
- F18 B (top4=0b1110): Linear if the surrounding block isn't ending,
  or DirectBranch if it's the last instr of a basic block. Convention
  per the dynarec branch's classifier: a B IS the end (terminator)
  unless it's an in-body B (Body::UncondBranch), which the AOT
  detects later. For initial scan: B = DirectBranch terminator.
- F16 Bcc (top4=0b1101, bits 11:8 != 0xF): Conditional.
  bits 11:8 == 0xF is SWI (1101_1111_imm8).
- F17 SWI (1101_1111_imm8): ExceptionEdge.
- F19 BL pair: hi (0xF000-0xF7FF) + lo (0xF800-0xFFFF). Hi is
  Linear (we scan into the lo). The pair as a unit is BlReturn.
  Special-case in classify_insn — track "saw F19 hi" state.
- F5 BX Rm (0x4700-0x47FF, mask 0xFF87 = 0x4700): IndirectBranch.
- F14 POP {pc, ...} (0xBD00-0xBDFF mask 0xFF00 = 0xBD00): IndirectBranch.
- Everything else: Linear.

For ARM:
- B/BL (top 3 bits 101): DirectBranch / Linear-with-link.
- BX (cond_0001_0010_1111_1111_1111_0001_Rm): IndirectBranch.
- LDM with R15 in rlist: IndirectBranch.
- LDR Rd=PC: IndirectBranch.
- MOV/ADD/SUB Rd=PC variants: IndirectBranch.
- SWI: ExceptionEdge.
- Multi-cond cases: trickier. ARM cond field per-instruction means
  even an unconditional ALU may not always execute. For scan
  purposes, treat conditional non-branch instructions as Linear
  (they don't change CFG).

## Validation per I22

`is_in_rom(pc, base, len)` — pc must be in `[base, base+len)`.

`mode-correct alignment` — Thumb pc & 1 == 0; ARM pc & 3 == 0.

`heuristic decode check` — if the decoded raw value is
0x00000000 or 0xFFFFFFFF, that's overwhelmingly data, not code.
Skip that target. (Same heuristic the dynarec branch's
recording-loop used implicitly via the LUT producing
no-op/unreachable handlers.)

## Output

The scan produces `Vec<BlockSpec>`. Phase 0 scaffold:
- For each BlockSpec, emit a placeholder block fn (single
  trampoline call to scalar replay).
- Insert into the two-level PC→fn table (per I18).
- Compile-then-publish per I17.

## Phase 0 status

A4 done. On to A5 (compile-time budget check).
