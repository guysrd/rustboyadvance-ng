use std::path::PathBuf;

use clap::Parser;
use rustboyadvance_core::{
    cartridge::{BackupType, GamepakBuilder},
    prelude::Cartridge,
};

#[derive(Parser, Debug)]
#[command(name = "rustboyadvance-sdl2")]
pub struct Options {
    /// Rom file to emulate, may be a raw dump from a cartridge or a compiled ELF file
    #[arg(name = "ROM")]
    pub rom: PathBuf,

    /// Bios file to use
    #[arg(long, default_value = "gba_bios.bin")]
    pub bios: PathBuf,

    /// Skip running the bios boot animation and jump straight to the ROM
    #[arg(long)]
    pub skip_bios: bool,

    /// Do not output sound
    #[arg(long)]
    pub _silent: bool,

    /// Initalize gdbserver and wait for a connection from gdb
    #[arg(short = 'd', long)]
    pub gdbserver: bool,

    #[arg(long = "port", default_value = "1337")]
    pub gdbserver_port: u16,

    /// Force emulation of RTC, use for games that have RTC but the emulator fails to detect
    #[arg(long)]
    pub rtc: bool,

    /// Override save type, useful for troublemaking games that fool the auto detection
    #[arg(long, default_value = "autodetect", value_enum)]
    pub save_type: Option<BackupType>,

    /// Record keypad input changes to this path for later replay. Every
    /// change to the keypad bitmask is written with the current emulated
    /// cycle count, producing a deterministic input trace the benchmark
    /// harness can replay against any CPU build.
    #[arg(long = "record-input", value_name = "PATH")]
    pub record_input: Option<PathBuf>,

    /// Feed a previously-recorded RBAREC01 input trace into the emulator
    /// in place of live keyboard/controller input. The recording drives
    /// the GBA keypad bitmask at the exact emulated cycles it was captured
    /// at, so the run is deterministic across CPU builds. When the last
    /// recorded event is consumed the emulator exits.
    #[arg(long = "replay", value_name = "PATH", conflicts_with = "record_input")]
    pub replay: Option<PathBuf>,

    /// Disable all audio output. Skips opening the SDL audio device and
    /// installs a NullAudio sink instead. Intended for CI / measurement
    /// runs where the audio pipeline adds noise and no one's listening.
    #[arg(long = "no-audio")]
    pub no_audio: bool,

    /// Enable the Cranelift dynarec for CPU dispatch. Off by default
    /// because (a) on pokeemerald PGO'd cached-interp currently beats
    /// the compiled path end-to-end and (b) dynarec has known latent
    /// correctness issues in some games that gba-tests don't catch.
    /// Turn on for profiling / measurement runs; expect possible
    /// visual glitches in some ROMs until the dispatch path is
    /// hardened. Built into any binary compiled with `--features
    /// dynarec` (or `--features shape_profile`).
    #[arg(long = "jit")]
    pub jit: bool,

    /// Print an FNV-1a hash of the framebuffer every N frames while
    /// replaying (format: `fb_hash: frame=N cycle=C hash=HHHHHHHH`).
    /// Used to diff scalar vs jit output on real SDL without going
    /// through fps_bench. Only meaningful in --replay mode.
    #[arg(long = "frame-hash-every", value_name = "N")]
    pub frame_hash_every: Option<u64>,

    #[cfg(feature = "debugger")]
    #[arg(long, default_value = None)]
    pub script_file: Option<String>,
}

type DynError = Box<dyn std::error::Error>;

impl Options {
    pub fn cartridge_from_opts(&self) -> Result<Cartridge, DynError> {
        let mut builder = GamepakBuilder::new()
            .save_type(self.save_type.unwrap_or(BackupType::AutoDetect))
            .file(&self.rom);
        if self.rtc {
            builder = builder.with_rtc();
        }
        Ok(builder.build()?)
    }

    pub fn savestate_path(&self) -> PathBuf {
        self.rom.with_extension("savestate")
    }

    pub fn rom_name(&self) -> &str {
        self.rom.file_name().unwrap().to_str().unwrap()
    }
}
