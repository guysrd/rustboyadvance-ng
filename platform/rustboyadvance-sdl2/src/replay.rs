//! Reader side of the SDL frontend input recorder. File format matches
//! what the recorder in `recorder.rs` emits and what `fps_bench`'s replay
//! module consumes: 8-byte `RBAREC01` magic then a stream of 10-byte
//! records (u64 LE cycle + u16 LE keystate).
//!
//! Kept as a copy of `fps_bench/src/replay.rs` rather than a shared crate
//! because the file is trivial and the two platforms depend on disjoint
//! runtime environments (SDL vs a bare core). If it grows any logic
//! beyond "load + apply_due + exhausted", promote it to utils.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RBAREC01";

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub cycle: u64,
    pub state: u16,
}

pub struct Replayer {
    events: Vec<Event>,
    cursor: usize,
}

impl Replayer {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("input recording magic mismatch: got {:?}", magic),
            ));
        }

        let mut events = Vec::new();
        let mut buf = [0u8; 10];
        loop {
            match file.read_exact(&mut buf) {
                Ok(()) => {
                    let cycle = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                    let state = u16::from_le_bytes(buf[8..10].try_into().unwrap());
                    events.push(Event { cycle, state });
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        Ok(Replayer { events, cursor: 0 })
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn last_cycle(&self) -> u64 {
        self.events.last().map(|e| e.cycle).unwrap_or(0)
    }

    pub fn apply_due(&mut self, now: u64, key_state: &mut u16) {
        while self.cursor < self.events.len() && self.events[self.cursor].cycle <= now {
            *key_state = self.events[self.cursor].state;
            self.cursor += 1;
        }
    }

    pub fn exhausted(&self) -> bool {
        self.cursor >= self.events.len()
    }
}
