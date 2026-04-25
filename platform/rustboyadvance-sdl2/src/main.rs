use sdl2::controller::Button;
use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::{self};

use log::info;

use clap::Parser;

use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::time;

use flexi_logger::*;

mod audio;
mod input;
mod options;
mod recorder;
mod video;

use rustboyadvance_core::prelude::*;
use rustboyadvance_core::prelude::NullAudio;

use rustboyadvance_utils::FpsCounter;

use rustboyadvance_core::cartridge::loader::{LoadRom, load_from_file};

mod replay;

const LOG_DIR: &str = ".logs";

fn ask_download_bios() {
    const OPEN_SOURCE_BIOS_URL: &str =
        "https://github.com/Nebuleon/ReGBA/raw/master/bios/gba_bios.bin";
    println!(
        "Missing BIOS file. If you don't have the original GBA BIOS, you can download an open-source bios from {}",
        OPEN_SOURCE_BIOS_URL
    );
    std::process::exit(0);
}

fn load_bios(bios_path: &Path) -> Box<[u8]> {
    match read_bin_file(bios_path) {
        Ok(bios) => bios.into_boxed_slice(),
        _ => {
            ask_download_bios();
            unreachable!()
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(LOG_DIR)
        .unwrap_or_else(|_| panic!("could not create log directory ({})", LOG_DIR));
    flexi_logger::Logger::try_with_env_or_str("info")
        .unwrap()
        .log_to_file(FileSpec::default().directory(LOG_DIR))
        .duplicate_to_stderr(Duplicate::Debug)
        .format_for_files(default_format)
        .format_for_stderr(colored_default_format)
        .start()
        .unwrap();

    let opts = options::Options::parse();

    info!("Initializing SDL2 context");
    let sdl_context = sdl2::init().expect("failed to initialize sdl2");

    let controller_subsystem = sdl_context.game_controller()?;
    let controller_mappings =
        include_str!("../../../external/SDL_GameControllerDB/gamecontrollerdb.txt");
    controller_subsystem.load_mappings_from_read(&mut Cursor::new(controller_mappings))?;

    let available_controllers = (0..controller_subsystem.num_joysticks()?)
        .filter(|&id| controller_subsystem.is_game_controller(id))
        .collect::<Vec<u32>>();

    let mut active_controller = match available_controllers.first() {
        Some(&id) => {
            let controller = controller_subsystem.open(id)?;
            info!("Found game controller: {}", controller.name());
            Some(controller)
        }
        _ => {
            info!("No game controllers were found");
            None
        }
    };

    let mut renderer = video::init(&sdl_context)?;
    // --no-audio routes past the SDL audio device entirely. Useful for
    // measurement runs where the audio pipeline adds noise and nothing
    // is listening. Window + video still come up normally.
    let (audio_interface, mut _sdl_audio_device): (DynAudioInterface, _) = if opts.no_audio {
        info!("--no-audio: installing NullAudio sink, skipping SDL audio device");
        (NullAudio::new(), None)
    } else {
        audio::create_audio_player(&sdl_context)
    };
    let rom_name = opts.rom_name();

    let bios_bin = load_bios(&opts.bios);

    let mut gba = Box::new(GameBoyAdvance::new(
        bios_bin.clone(),
        opts.cartridge_from_opts()?,
        audio_interface,
    ));

    // let gba_raw_ptr = Box::into_raw(gba) as usize;
    // static mut gba_raw: usize = 0;
    // unsafe { gba_raw = gba_raw_ptr };
    // let mut gba = unsafe {Box::from_raw(gba_raw_ptr as *mut GameBoyAdvance) };

    // std::panic::set_hook(Box::new(|panic_info| {
    //     let gba = unsafe {Box::from_raw(gba_raw as *mut GameBoyAdvance) };
    //     println!("System crashed Oh No!!! {:?}", gba.cpu);
    //     let normal_panic = std::panic::take_hook();
    //     normal_panic(panic_info);
    // }));

    if opts.skip_bios {
        println!("Skipping bios animation..");
        gba.skip_bios();
    }

    // Dynarec dispatcher control. Off by default; --jit turns the
    // LLVM-backed dispatcher on. ARM blocks and any LLVM-rejected
    // blocks fall back to cached_interp scalar replay.
    if opts.jit {
        #[cfg(feature = "dynarec")]
        {
            info!("Enabling LLVM dynarec dispatcher");
            gba.cpu.enable_dynarec();
        }
        #[cfg(not(feature = "dynarec"))]
        log::warn!(
            "--jit requested but this binary was built without dynarec; ignoring"
        );
    }

    if opts.gdbserver {
        gba.start_gdbserver(opts.gdbserver_port);
    }

    // Input recorder: created lazily if --record-input was passed. We record
    // the initial keypad state at cycle 0 so the replayer starts from an
    // identical snapshot regardless of the host's SDL event timing.
    let mut recorder = match &opts.record_input {
        Some(path) => {
            info!("Recording keypad input to {:?}", path);
            Some(recorder::Recorder::create(path, *gba.get_key_state())?)
        }
        None => None,
    };

    // Input replayer: created lazily if --replay was passed. Mutually
    // exclusive with --record-input (enforced by clap). When present, the
    // replayer drives the keypad from the recorded trace; the SDL event
    // loop still runs (so the user can Escape out, resize the window,
    // etc) but per-frame key-state edits from keyboard/controller get
    // overwritten by `apply_due` right before `gba.frame()`. Emulator
    // exits once the last recorded edge is consumed AND the emulator's
    // cycle counter has passed its cycle stamp.
    let mut replayer = match &opts.replay {
        Some(path) => {
            let r = replay::Replayer::load(path)
                .map_err(|e| format!("failed to load replay {:?}: {}", path, e))?;
            info!(
                "Replaying {:?}: {} events, last cycle {}",
                path,
                r.len(),
                r.last_cycle()
            );
            Some(r)
        }
        None => None,
    };
    let replay_start = time::Instant::now();
    let mut replay_frames: u64 = 0;
    let mut last_present = time::Instant::now();
    const REPLAY_PRESENT_INTERVAL: time::Duration = time::Duration::from_millis(16); // ~60Hz

    let mut vsync = true;
    let mut fps_counter = FpsCounter::default();
    const FRAME_TIME: time::Duration = time::Duration::new(0, 1_000_000_000u32 / 60);
    let mut event_pump = sdl_context.event_pump()?;
    'running: loop {
        let start_time = time::Instant::now();
        for event in event_pump.poll_iter() {
            match event {
                Event::KeyDown {
                    scancode: Some(scancode),
                    ..
                } => match scancode {
                    Scancode::Space => vsync = false,
                    k => input::on_keyboard_key_down(gba.get_key_state_mut(), k),
                },
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => match scancode {
                    Scancode::Escape => break 'running,
                    #[cfg(feature = "debugger")]
                    Scancode::F1 => {
                        let mut debugger = Debugger::new();
                        info!("starting debugger...");
                        debugger
                            .repl(&mut gba, opts.script_file.as_deref())
                            .unwrap();
                        info!("ending debugger...")
                    }
                    Scancode::F2 => gba.start_gdbserver(opts.gdbserver_port),
                    Scancode::F5 => {
                        info!("Saving state ...");
                        let save = gba.save_state()?;
                        write_bin_file(&opts.savestate_path(), &save)?;
                        info!(
                            "Saved to {:?} ({})",
                            opts.savestate_path(),
                            bytesize::ByteSize::b(save.len() as u64)
                        );
                    }
                    Scancode::F9 => {
                        if opts.savestate_path().is_file() {
                            let save = read_bin_file(&opts.savestate_path())?;
                            info!("Restoring state from {:?}...", opts.savestate_path());
                            let (audio_interface, _sdl_audio_device_new) =
                                audio::create_audio_player(&sdl_context);
                            _sdl_audio_device = _sdl_audio_device_new;
                            let rom = match load_from_file(&opts.rom)? {
                                LoadRom::Raw(data) => data.into_boxed_slice(),
                                LoadRom::Elf { data, .. } => data.into_boxed_slice(),
                            };
                            *gba = GameBoyAdvance::from_saved_state(
                                &save,
                                bios_bin.clone(),
                                rom,
                                audio_interface,
                            )?;
                            info!("Restored!");
                        } else {
                            info!("Savestate not created, please create one by pressing F5");
                        }
                    }
                    Scancode::Space => vsync = true,
                    k => input::on_keyboard_key_up(gba.get_key_state_mut(), k),
                },
                Event::ControllerButtonDown { button, .. } => match button {
                    Button::RightStick => vsync = !vsync,
                    b => input::on_controller_button_down(gba.get_key_state_mut(), b),
                },
                Event::ControllerButtonUp { button, .. } => {
                    input::on_controller_button_up(gba.get_key_state_mut(), button);
                }
                Event::ControllerAxisMotion { axis, value, .. } => {
                    input::on_axis_motion(gba.get_key_state_mut(), axis, value);
                }
                Event::ControllerDeviceRemoved { which, .. } => {
                    let removed = if let Some(active_controller) = &active_controller {
                        active_controller.instance_id() == which
                    } else {
                        false
                    };
                    if removed {
                        let name = active_controller
                            .map(|controller| controller.name())
                            .unwrap();
                        info!("Removing game controller: {}", name);
                        active_controller = None;
                    }
                }
                Event::ControllerDeviceAdded { which, .. } => {
                    if active_controller.is_none() {
                        let controller = controller_subsystem.open(which)?;
                        info!("Adding game controller: {}", controller.name());
                        active_controller = Some(controller);
                    }
                }
                Event::Quit { .. } => break 'running,
                Event::DropFile { .. } => {
                    todo!("impl DropFile again")
                }
                _ => {}
            }
        }

        // Snapshot the keypad after all SDL events have been applied; if
        // anything changed, record the edge. Doing this once per frame
        // (rather than per-event) coalesces bursts of events into a single
        // observation, which is what the replayer sees anyway.
        if let Some(rec) = &mut recorder {
            let _ = rec.observe(gba.cycles(), *gba.get_key_state());
        }

        // Replay drives the keypad. Apply any edges whose recorded cycle
        // has already passed; apply_due writes directly into the keypad
        // bitmask, overwriting whatever the user typed into SDL. Stop
        // the emulator once the last recorded edge is in the past so
        // the replay run has a well-defined end (identical across CPU
        // builds).
        if let Some(r) = &mut replayer {
            let now = gba.cycles() as u64;
            r.apply_due(now, gba.get_key_state_mut());
            if r.exhausted() && now >= r.last_cycle() {
                let elapsed = replay_start.elapsed().as_secs_f64();
                let fps = (replay_frames as f64) / elapsed.max(1e-9);
                // Stable machine-parseable summary for the measure harness.
                println!(
                    "replay done: {} frames in {:.2}s wall, {:.1} avg fps ({} emulated cycles)",
                    replay_frames, elapsed, fps, now
                );
                #[cfg(feature = "dynarec")]
                {
                    let (total, compiled) = gba.cpu.block_cache.compile_stats();
                    let pct = if total > 0 {
                        100.0 * compiled as f64 / total as f64
                    } else {
                        0.0
                    };
                    println!(
                        "block cache: {} rom blocks, {} compiled ({:.1}%)",
                        total, compiled, pct,
                    );
                    let cc = gba.cpu.dispatch_compiled_count;
                    let ic = gba.cpu.dispatch_interp_count;
                    let total_d = cc + ic;
                    let cc_pct = if total_d > 0 {
                        100.0 * cc as f64 / total_d as f64
                    } else {
                        0.0
                    };
                    println!(
                        "dispatch: {} total, {} compiled ({:.2}%), {} interp",
                        total_d, cc, cc_pct, ic,
                    );
                }
                break 'running;
            }
        }

        if gba.is_debugger_attached() {
            gba.debugger_run()
        } else {
            gba.frame();
            if replayer.is_some() {
                replay_frames += 1;
                if let Some(n) = opts.frame_hash_every {
                    if n > 0 && replay_frames % n == 0 {
                        let fb = gba.get_frame_buffer();
                        // FNV-1a over the u32 pixels, matching
                        // fps_bench's hash scheme so output is
                        // cross-comparable.
                        let mut h: u64 = 0xcbf29ce484222325;
                        for &px in fb {
                            for b in px.to_le_bytes() {
                                h ^= b as u64;
                                h = h.wrapping_mul(0x100000001b3);
                            }
                        }
                        println!(
                            "fb_hash: frame={} cycle={} hash={:016x}",
                            replay_frames,
                            gba.cycles(),
                            h,
                        );
                    }
                }
            }
        }
        // In replay mode we rate-limit the SDL present to ~60Hz wall-clock.
        // The emulator itself runs flat-out (no vsync) at 1000+ FPS, so
        // presenting every emulator frame causes the SDL texture upload
        // to race the GPU writing scanlines, producing visible tearing.
        // By sampling the backbuffer ~60x per wall second we show a clean
        // view of the game to the user while keeping the underlying
        // throughput measurement untouched. The emulator framebuffer is
        // bit-identical to interp anyway (verified by fps_bench
        // --frame-hash-every diffs over 3 min of pokeemerald replay).
        let should_present = if replayer.is_some() {
            let now = time::Instant::now();
            if now.duration_since(last_present) >= REPLAY_PRESENT_INTERVAL {
                last_present = now;
                true
            } else {
                false
            }
        } else {
            true
        };
        if should_present {
            renderer.render(gba.get_frame_buffer());
        }

        if let Some(fps) = fps_counter.tick() {
            let title = format!("{} ({} fps)", rom_name, fps);
            renderer.set_window_title(&title);
        }

        // Replay runs flat out regardless of vsync — the whole point is
        // to measure wall time against a fixed amount of emulated work.
        if vsync && replayer.is_none() {
            let time_passed = start_time.elapsed();
            let delay = FRAME_TIME.checked_sub(time_passed);
            match delay {
                None => {}
                Some(delay) => {
                    spin_sleep::sleep(delay);
                }
            };
        }
    }

    Ok(())
}
