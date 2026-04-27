//! GBA memory region constants and helpers (per A2 — see
//! `docs/findings-io-regions.md`).
//!
//! The AOT emit fns use these to decide whether a memory operation
//! can be inlined as direct buffer access (RAM regions) or must emit
//! a real `bus.read_*` / `bus.write_*` extern call (IO regions).

/// Memory page (top byte of address) constants. Match
/// `core::sysbus::consts::PAGE_*`.
pub mod page {
    pub const BIOS: u8 = 0x00;
    pub const EWRAM: u8 = 0x02;
    pub const IWRAM: u8 = 0x03;
    pub const IOMEM: u8 = 0x04;
    pub const PALRAM: u8 = 0x05;
    pub const VRAM: u8 = 0x06;
    pub const OAM: u8 = 0x07;
    pub const GAMEPAK_WS0_LO: u8 = 0x08;
    pub const GAMEPAK_WS0_HI: u8 = 0x09;
    pub const GAMEPAK_WS1_LO: u8 = 0x0A;
    pub const GAMEPAK_WS1_HI: u8 = 0x0B;
    pub const GAMEPAK_WS2_LO: u8 = 0x0C;
    pub const GAMEPAK_WS2_HI: u8 = 0x0D;
    pub const SRAM_LO: u8 = 0x0E;
    pub const SRAM_HI: u8 = 0x0F;
}

/// True if `addr` is in an IO / side-effecting region (per I3, must
/// emit a real bus call rather than direct buffer access).
#[inline]
pub fn is_io_addr(addr: u32) -> bool {
    let region = ((addr >> 24) & 0xff) as u8;
    matches!(
        region,
        page::IOMEM | page::PALRAM | page::VRAM | page::OAM | page::SRAM_LO | page::SRAM_HI
    )
}

/// True if `addr` is in a read-only ROM region (BIOS or cartridge).
/// Reads inline as direct buffer access; writes are no-ops.
#[inline]
pub fn is_rom_region(addr: u32) -> bool {
    let region = ((addr >> 24) & 0xff) as u8;
    matches!(
        region,
        page::BIOS
            | page::GAMEPAK_WS0_LO
            | page::GAMEPAK_WS0_HI
            | page::GAMEPAK_WS1_LO
            | page::GAMEPAK_WS1_HI
            | page::GAMEPAK_WS2_LO
            | page::GAMEPAK_WS2_HI
    )
}

/// True if `addr` is in a writable RAM region that can inline as
/// direct buffer access (no GPU/IO side effects).
#[inline]
pub fn is_inlinable_ram(addr: u32) -> bool {
    let region = ((addr >> 24) & 0xff) as u8;
    matches!(region, page::EWRAM | page::IWRAM)
}

/// All IO regions as (start, end_exclusive) ranges. For
/// reg-offset address bounds checks at AOT compile time.
pub const IO_REGIONS: &[(u32, u32)] = &[
    (0x0400_0000, 0x0400_0400), // IOMEM
    (0x0500_0000, 0x0500_0400), // PALRAM
    (0x0600_0000, 0x0601_8000), // VRAM
    (0x0700_0000, 0x0700_0400), // OAM
    (0x0E00_0000, 0x0E01_0000), // SRAM_LO
    (0x0F00_0000, 0x0F01_0000), // SRAM_HI mirror
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_classification() {
        // BIOS region: ROM read, no IO.
        assert!(is_rom_region(0x0000_1234));
        assert!(!is_io_addr(0x0000_1234));
        assert!(!is_inlinable_ram(0x0000_1234));

        // EWRAM: inlinable RAM.
        assert!(is_inlinable_ram(0x0203_F000));
        assert!(!is_io_addr(0x0203_F000));
        assert!(!is_rom_region(0x0203_F000));

        // IWRAM: inlinable RAM.
        assert!(is_inlinable_ram(0x0300_4000));

        // IOMEM: IO.
        assert!(is_io_addr(0x0400_0000));
        assert!(!is_rom_region(0x0400_0000));

        // PALRAM/VRAM/OAM: IO.
        assert!(is_io_addr(0x0500_0200));
        assert!(is_io_addr(0x0600_4000));
        assert!(is_io_addr(0x0700_0100));

        // Cart ROM mirrors: ROM.
        assert!(is_rom_region(0x0800_1000));
        assert!(is_rom_region(0x0A00_0000));
        assert!(is_rom_region(0x0D00_0000));

        // SRAM: IO (because of flash/EEPROM quirks).
        assert!(is_io_addr(0x0E00_0000));
        assert!(is_io_addr(0x0F00_0500));
    }
}
