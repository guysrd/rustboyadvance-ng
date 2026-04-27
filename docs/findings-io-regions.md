# A2: IO-region map

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A2: enumerate which
GBA memory regions can be inlined as direct buffer access (no side
effects) vs which must emit a real `bus.read_*` / `bus.write_*` call.

This drives invariant I3 (IO needs real bus calls) for every memory
op AOT emits.

## GBA memory map

From `core/src/sysbus.rs::consts` (which matches gbatek):

| Region | Upper 8 bits | Range | Type | AOT inlinable? |
|--------|--------------|-------|------|----------------|
| BIOS | 0x00 | 0x00000000-0x00003FFF | ROM, read-only | **YES (read only)** |
| (open bus) | 0x01 | — | — | NO (real call) |
| EWRAM | 0x02 | 0x02000000-0x0203FFFF | 256KB RAM | **YES** |
| IWRAM | 0x03 | 0x03000000-0x03007FFF | 32KB RAM | **YES** |
| IOMEM | 0x04 | 0x04000000-0x040003FE | IO registers | NO (side effects) |
| PALRAM | 0x05 | 0x05000000-0x050003FF | palette | NO (GPU side effects) |
| VRAM | 0x06 | 0x06000000-0x06017FFF | video RAM | NO (GPU side effects) |
| OAM | 0x07 | 0x07000000-0x070003FF | sprite regs | NO (GPU side effects) |
| GAMEPAK WS0 LO | 0x08 | 0x08000000-0x08FFFFFF | cart ROM | **YES (read only)** |
| GAMEPAK WS0 HI | 0x09 | 0x09000000-0x09FFFFFF | cart ROM mirror | **YES (read only)** |
| GAMEPAK WS1 LO | 0x0A | 0x0A000000-0x0AFFFFFF | cart ROM mirror | **YES (read only)** |
| GAMEPAK WS1 HI | 0x0B | 0x0B000000-0x0BFFFFFF | cart ROM mirror | **YES (read only)** |
| GAMEPAK WS2 LO | 0x0C | 0x0C000000-0x0CFFFFFF | cart ROM mirror | **YES (read only)** |
| GAMEPAK WS2 HI | 0x0D | 0x0D000000-0x0DFFFFFF | cart ROM mirror | **YES (read only)** |
| SRAM LO | 0x0E | 0x0E000000-0x0E00FFFF | cart backup | NO (real call — flash/EEPROM quirks) |
| SRAM HI | 0x0F | 0x0F000000-0x0F00FFFF | SRAM mirror | NO (real call) |

## Decision matrix

For each LDR/STR (any width) at AOT compile time:

```rust
match (addr_upper_8, op) {
    (0x00, Read)                        => inline_read_bios,
    (0x00, Write)                       => no_op,            // BIOS is ROM
    (0x02, _)                           => inline_ewram,
    (0x03, _)                           => inline_iwram,
    (0x04 | 0x05 | 0x06 | 0x07, _)      => real_bus_call,    // IO/GPU
    (0x08..=0x0D, Read)                 => inline_cart_rom,
    (0x08..=0x0D, Write)                => no_op,            // cart is ROM
    (0x0E | 0x0F, _)                    => real_bus_call,    // SRAM/backup
    _                                   => real_bus_call,    // open bus
}
```

For reg-offset addresses (region not known at compile time): emit a
runtime `match (addr >> 24) & 0xF` ladder. The fast paths inline; IO
falls to real call.

## Implementation in arm7tdmi-aot/src/bus.rs

```rust
pub fn is_io_addr(addr: u32) -> bool {
    let region = (addr >> 24) & 0xff;
    matches!(region, 0x04 | 0x05 | 0x06 | 0x07 | 0x0E | 0x0F)
}

pub fn is_rom_region(addr: u32) -> bool {
    let region = (addr >> 24) & 0xff;
    matches!(region, 0x00 | 0x08..=0x0D)
}

pub fn is_inlinable_ram(addr: u32) -> bool {
    let region = (addr >> 24) & 0xff;
    matches!(region, 0x02 | 0x03)
}

pub const IO_REGIONS: &[(u32, u32)] = &[
    (0x04000000, 0x04000400),  // IOMEM
    (0x05000000, 0x05000400),  // PALRAM
    (0x06000000, 0x06018000),  // VRAM
    (0x07000000, 0x07000400),  // OAM
    (0x0E000000, 0x0E010000),  // SRAM_LO
    (0x0F000000, 0x0F010000),  // SRAM_HI (mirror)
];
```

## Why VRAM/PALRAM/OAM aren't direct-RAM inlinable

Even though VRAM/PALRAM/OAM ARE backing-buffer reads/writes, they
trigger GPU-side state transitions that the SysBus implementation
encapsulates (e.g., palette write may invalidate render caches,
VRAM write may need bounds-mask alignment for mode-specific regions,
OAM write affects sprite rendering for the next scanline). Going
direct would require reproducing all that GPU state-machine logic
in IR — too brittle. Real bus call is correct AND maintains the
invariant I3 says.

## Cycle accounting note

ALL memory regions need cycle accounting (per the wait-state LUT in
sysbus). The "inlinable RAM" decision affects whether the actual
buffer access is inlined; cycle accounting is ALWAYS inlined as a
separate `add_cycles` call regardless. Per A1, the fetch elision
optimization keeps add_cycles + skips read_16 — same pattern here:
keep cycle accounting always, inline buffer access when safe.

## Phase 0 status

A2 done. On to A3 (indirect-branch fallback decision).
