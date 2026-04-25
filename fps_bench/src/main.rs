use std::path::PathBuf;

use clap::Parser;

use rustboyadvance_core::prelude::*;
use rustboyadvance_utils::FpsCounter;

mod replay;

#[derive(Parser, Debug)]
#[command(name = "fps_bench")]
struct Options {
    /// BIOS file to use.
    bios: PathBuf,
    /// ROM file to run.
    rom: PathBuf,

    /// Path to a recording produced by `rustboyadvance-sdl2 --record-input`.
    /// If set, fps_bench runs exactly as long as the recording lasts while
    /// feeding the recorded keypad changes into the emulator at the cycle
    /// counts they were captured at, and reports aggregate FPS stats at end.
    ///
    /// If unset, fps_bench runs indefinitely and prints per-second FPS (the
    /// original "how fast is idle gameplay" benchmark).
    #[arg(long = "replay", value_name = "PATH")]
    replay: Option<PathBuf>,

    /// No-op on the `aot-llvm` branch — LLVM JIT was retired here.
    /// Reserved for the AOT-LLVM dispatcher when that lands.
    #[arg(long = "jit")]
    jit: bool,

    /// Run the full BIOS boot animation instead of skipping it. Matches
    /// what the SDL frontend does by default — useful when comparing
    /// fps_bench against SDL under dynarec because skip_bios leaves
    /// some games in a state they don't reach during normal play.
    #[arg(long = "full-bios")]
    full_bios: bool,

    /// Print a 64-bit hash of the framebuffer every N frames to stdout,
    /// prefixed "fb_hash: frame=N cycle=C hash=HHHHHHHH". Used to diff
    /// interpreter vs dynarec output: run twice (with and without --jit)
    /// and the first diverging line narrows the mis-compiled cycle range.
    #[arg(long = "frame-hash-every", value_name = "N")]
    frame_hash_every: Option<u64>,
}

fn main() {
    let opts = Options::parse();

    let bios = read_bin_file(&opts.bios).expect("failed to read bios file");
    let rom = read_bin_file(&opts.rom).expect("failed to read rom file");

    let gamepak = GamepakBuilder::new()
        .take_buffer(rom.into_boxed_slice())
        .with_sram()
        .without_backup_to_file()
        .build()
        .unwrap();

    let mut gba = GameBoyAdvance::new(bios.into_boxed_slice(), gamepak, NullAudio::new());
    if !opts.full_bios {
        gba.skip_bios();
    }

    if opts.jit {
        eprintln!(
            "--jit is a no-op on this branch (cached_interp scalar only). \
             AOT-LLVM dispatcher landing in a future commit."
        );
    }

    match opts.replay {
        Some(path) => run_replay(&mut gba, &path, opts.frame_hash_every),
        None => run_idle(&mut gba),
    }
}

/// Small non-crypto hash suitable for spotting framebuffer divergence.
/// FNV-1a over u32 pixels; 64-bit output. Avoids pulling in sha2.
fn hash_framebuffer(fb: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &px in fb {
        for b in px.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Original benchmark loop: run forever, print FPS once per second.
fn run_idle(gba: &mut GameBoyAdvance) {
    let mut fps_counter = FpsCounter::default();
    loop {
        gba.frame();
        if let Some(fps) = fps_counter.tick() {
            println!("FPS: {}", fps);
        }
    }
}

/// Replay loop: drive the emulator with recorded keypad edges until the
/// recording runs out of events AND the emulator's cycle counter passes the
/// last recorded edge, then report aggregate timing.
fn run_replay(
    gba: &mut GameBoyAdvance,
    path: &std::path::Path,
    frame_hash_every: Option<u64>,
) {
    let mut replayer = replay::Replayer::load(path).expect("failed to load recording");
    let last_cycle = replayer.last_cycle();
    eprintln!(
        "Replaying {} events, terminating at cycle {}",
        replayer.len(),
        last_cycle
    );

    let mut fps_counter = FpsCounter::default();
    let wall_start = std::time::Instant::now();
    let mut frames: u64 = 0;

    loop {
        // Apply any keypad edges whose recorded cycle has already elapsed.
        // This runs BEFORE the frame so an edge whose cycle lands inside the
        // upcoming frame's cycle range is applied one frame late at worst —
        // the same granularity the recorder captures at (once per frame).
        replayer.apply_due(gba.cycles() as u64, gba.get_key_state_mut());

        gba.frame();
        frames += 1;

        if let Some(n) = frame_hash_every {
            if n > 0 && frames % n == 0 {
                let h = hash_framebuffer(gba.get_frame_buffer());
                println!(
                    "fb_hash: frame={} cycle={} hash={:016x}",
                    frames,
                    gba.cycles(),
                    h
                );
            }
        }

        if let Some(fps) = fps_counter.tick() {
            println!("FPS: {}", fps);
        }

        // Terminate once the recording is fully consumed and the emulator
        // has run past the trailing event's cycle. This ensures both builds
        // execute the same total amount of emulated work before reporting.
        if replayer.exhausted() && gba.cycles() as u64 >= last_cycle {
            break;
        }
    }

    let elapsed = wall_start.elapsed().as_secs_f64();
    let avg_fps = frames as f64 / elapsed;
    println!("---");
    println!(
        "replay done: {} frames in {:.2}s wall, {:.1} avg fps ({} emulated cycles)",
        frames,
        elapsed,
        avg_fps,
        gba.cycles()
    );
    #[cfg(feature = "dynarec")]
    {
        let (total, compiled, linked) = gba.cpu.block_cache.compile_stats();
        let pct = if total > 0 {
            100.0 * compiled as f64 / total as f64
        } else {
            0.0
        };
        let link_pct = if compiled > 0 {
            100.0 * linked as f64 / compiled as f64
        } else {
            0.0
        };
        println!(
            "block cache: {} rom blocks, {} compiled ({:.1}%), {} chain-linked ({:.1}% of compiled)",
            total, compiled, pct, linked, link_pct,
        );
    }
}
