//! Per-shape execution-count profiler for the dynarec.
//!
//! The shape-opt measure harness (`scripts/dynarec_measure.sh`) computes
//!
//!     weighted_cycles = sum over shapes of (ns_per_shape * W_<shape>)
//!
//! where `ns_per_shape` is measured by Criterion and `W_<shape>` is a
//! hand-picked constant reflecting how often the shape fires per second
//! in real gameplay. The hand-picked numbers were educated guesses.
//! This module turns them into measurements.
//!
//! ## Mechanism
//!
//! At block compile time (in `BlockCache::finish_record`), we classify
//! the block by inspecting its raw opcodes and tag the `Block` with a
//! `ShapeId`. At dispatch time (in `Arm7tdmiCore::replay_cached_block`),
//! when we invoke the compiled function we also `tick(shape)` to bump
//! that shape's AtomicU64 counter. At SDL-replay exit the frontend
//! calls `dump()` which prints a parseable block:
//!
//!     shape_profile:thumb_mov_imm    8_245_123
//!     shape_profile:thumb_add_imm    4_891_200
//!     ...
//!
//! The measure script greps this block, divides each count by the
//! replay wall time to get calls/sec, and either prints the suggested
//! `W_*` values or auto-updates them (see retraining section in
//! docs/dynarec-arm64-program.md).
//!
//! ## Why this file is cfg-gated
//!
//! A `tick` call on every compiled-block invocation has a measurable
//! cost even when it's just an atomic add (it defeats Cranelift's
//! ability to keep the block-entry path branchless). Gating everything
//! behind `#[cfg(feature = "shape_profile")]` keeps the default build
//! byte-identical — you opt into profiling by rebuilding with
//! `--features shape_profile`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Shape categories. One variant per micro-bench in
/// `arm7tdmi/benches/dynarec_shapes.rs` (so `W_*` retraining is a
/// straight one-to-one replacement) plus `Other` for everything
/// that doesn't match a bench's trigger.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum ShapeId {
    ThumbMovImm = 0,
    ThumbAddImm = 1,
    ThumbCmpImm = 2,
    ThumbDpChain = 3,
    ArmMovImm = 4,
    ArmCmpImm = 5,
    ShiftPairSxtb = 6,
    Other = 7,
}

pub const N_SHAPES: usize = 8;

/// Human-readable names in the same order as the enum discriminants.
/// Used by `dump()` to label each counter in the output block.
const SHAPE_NAMES: [&str; N_SHAPES] = [
    "thumb_mov_imm",
    "thumb_add_imm",
    "thumb_cmp_imm",
    "thumb_dp_chain",
    "arm_mov_imm",
    "arm_cmp_imm",
    "shift_pair_sxtb",
    "other",
];

/// Per-shape execution counters, bumped at dispatch time.
static SHAPE_COUNTS: [AtomicU64; N_SHAPES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Bump the execution counter for `shape`. Called from
/// `replay_cached_block` right before invoking the compiled function.
///
/// Uses `Relaxed` ordering because we don't need synchronization with
/// any other thread's state — the counters are a running total read at
/// exit, and lost increments from races cost at most a handful of
/// samples out of millions.
#[inline]
pub fn tick(shape: ShapeId) {
    SHAPE_COUNTS[shape as usize].fetch_add(1, Ordering::Relaxed);
}

/// Reset all counters to zero. Useful for isolating the profile of a
/// specific replay pass from the BIOS / startup frames that preceded
/// it — call this right before the replay loop begins so the dump
/// only reflects the measured window.
pub fn reset() {
    for c in &SHAPE_COUNTS {
        c.store(0, Ordering::Relaxed);
    }
}

/// Snapshot the counters into a fixed-size array, in enum-order.
/// Caller can post-process this (e.g. divide by wall time for
/// calls/sec) without holding a reference into the atomics.
pub fn snapshot() -> [u64; N_SHAPES] {
    let mut out = [0u64; N_SHAPES];
    for (i, c) in SHAPE_COUNTS.iter().enumerate() {
        out[i] = c.load(Ordering::Relaxed);
    }
    out
}

/// Format the current counter snapshot as the exact block the measure
/// script greps. One line per shape, prefixed with `shape_profile:`
/// and the shape name, followed by the count. Underscores in the
/// count are for human readability and are stripped on the script
/// side with a simple `tr -d '_'`.
pub fn dump() -> String {
    let snap = snapshot();
    let mut out = String::new();
    out.push_str("--- shape_profile ---\n");
    for i in 0..N_SHAPES {
        out.push_str(&format!(
            "shape_profile:{:<16} {}\n",
            SHAPE_NAMES[i], snap[i]
        ));
    }
    out
}

/// Classify a Thumb block (sequence of 16-bit opcodes) into one of the
/// `ShapeId` categories. Called from `BlockCache::finish_record` when
/// compilation succeeds, and the result stored on the `Block` so the
/// dispatcher can tick the right counter on every replay.
///
/// The classification intentionally mirrors the `bench_*` functions
/// in `arm7tdmi/benches/dynarec_shapes.rs` — any shape not recognizable
/// as a bench shape falls into `Other`, which is where the agent's
/// attention goes when dumping (if `other` dominates the profile, the
/// bench set doesn't cover a hot path).
pub fn classify_thumb(raws: &[u16]) -> ShapeId {
    // Single-instruction Thumb format 3 blocks. Bit pattern:
    //   format 3 = 001_<op>_<rd>_<imm8>
    //   op: 00=MOV, 01=CMP, 10=ADD, 11=SUB
    if raws.len() == 1 {
        let op = raws[0];
        if op & 0xE000 == 0x2000 {
            let subop = (op >> 11) & 0b11;
            match subop {
                0b00 => return ShapeId::ThumbMovImm,
                0b01 => return ShapeId::ThumbCmpImm,
                0b10 => return ShapeId::ThumbAddImm,
                _ => {}
            }
        }
    }
    // Shift-pair sign extend: LSL #24 ; ASR #24 (and the halfword /
    // unsigned variants), compiled by patterns.rs as a single
    // sxtb/sxth/uxtb/uxth stencil.
    if raws.len() == 2
        && is_thumb_shift_pair_sign_or_zero_extend(raws[0], raws[1])
    {
        return ShapeId::ShiftPairSxtb;
    }
    // Any other multi-instruction Thumb block is a "dp_chain" — that's
    // the bench that most closely approximates these in ns/iter.
    if raws.len() >= 2 {
        return ShapeId::ThumbDpChain;
    }
    ShapeId::Other
}

/// Classify an ARM block (sequence of 32-bit opcodes). Thumb and ARM
/// blocks are kept separate so the classification can't confuse a
/// Thumb format 3 MOV with an ARM data-processing MOV.
pub fn classify_arm(opcodes: &[u32]) -> ShapeId {
    if opcodes.len() == 1 {
        let op = opcodes[0];
        // ARM data-processing, immediate operand2. Bits 27..26 = 00,
        // bit 25 = 1 (immediate form). Opcode is bits 24..21.
        let is_dp_imm = (op >> 26) & 0b11 == 0b00 && (op >> 25) & 1 == 1;
        if is_dp_imm {
            let dp_op = (op >> 21) & 0b1111;
            match dp_op {
                0b1101 => return ShapeId::ArmMovImm, // MOV
                0b1010 => return ShapeId::ArmCmpImm, // CMP
                _ => {}
            }
        }
    }
    ShapeId::Other
}

/// Detect the two-instruction LSL+ASR / LSL+LSR idiom that
/// patterns.rs lowers to `sxtb / sxth / uxtb / uxth`. Matches the
/// exact trigger set of
/// `patterns::try_match_shift_pair_sign_or_zero_extend` — if that
/// matcher evolves, this heuristic needs to match.
///
/// LSL #imm5  = 000_00_<imm5>_<rs>_<rd>, opcode group 0b000 with the
/// shift-type field = 0b00.
/// ASR #imm5  = 000_10_<imm5>_<rs>_<rd>, shift-type 0b10.
/// LSR #imm5  = 000_01_<imm5>_<rs>_<rd>, shift-type 0b01.
fn is_thumb_shift_pair_sign_or_zero_extend(first: u16, second: u16) -> bool {
    // Both must be Thumb format 1 (top 3 bits == 000) AND not format 2
    // (which has top 5 bits == 00011).
    let is_format1 = |op: u16| (op & 0xE000) == 0x0000 && (op & 0x1800) != 0x1800;
    if !is_format1(first) || !is_format1(second) {
        return false;
    }
    let first_shift_type = (first >> 11) & 0b11;
    let second_shift_type = (second >> 11) & 0b11;
    let first_imm5 = (first >> 6) & 0b11111;
    let second_imm5 = (second >> 6) & 0b11111;
    // Matches:
    //   LSL #24 + ASR #24   → sxtb
    //   LSL #16 + ASR #16   → sxth
    //   LSL #24 + LSR #24   → uxtb (redundant with `and #0xff` but common)
    //   LSL #16 + LSR #16   → uxth
    if first_shift_type != 0b00 {
        return false;
    }
    match (first_imm5, second_shift_type, second_imm5) {
        (24, 0b10, 24) => true, // sxtb
        (16, 0b10, 16) => true, // sxth
        (24, 0b01, 24) => true, // uxtb
        (16, 0b01, 16) => true, // uxth
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_thumb_format3_mov_is_mov_imm() {
        // MOV R1, #42
        assert_eq!(classify_thumb(&[0x212a]), ShapeId::ThumbMovImm);
    }

    #[test]
    fn single_thumb_format3_add_is_add_imm() {
        // ADD R1, #1
        assert_eq!(classify_thumb(&[0x3101]), ShapeId::ThumbAddImm);
    }

    #[test]
    fn single_thumb_format3_cmp_is_cmp_imm() {
        // CMP R1, #5
        assert_eq!(classify_thumb(&[0x2905]), ShapeId::ThumbCmpImm);
    }

    #[test]
    fn three_thumb_instrs_is_dp_chain() {
        // MOV R0,#1 ; MOV R1,#2 ; ADD R0,R0,R1
        assert_eq!(
            classify_thumb(&[0x2001, 0x2102, 0x1840]),
            ShapeId::ThumbDpChain
        );
    }

    #[test]
    fn shift_pair_sxtb_recognized() {
        // LSL R1, R0, #24 ; ASR R1, R1, #24
        assert_eq!(classify_thumb(&[0x0601, 0x1609]), ShapeId::ShiftPairSxtb);
    }

    #[test]
    fn single_arm_mov_imm_recognized() {
        // MOV R1, #42 (encoded E3A0102A)
        assert_eq!(classify_arm(&[0xE3A0_102A]), ShapeId::ArmMovImm);
    }

    #[test]
    fn single_arm_cmp_imm_recognized() {
        // CMP R0, #5 (E3500005)
        assert_eq!(classify_arm(&[0xE350_0005]), ShapeId::ArmCmpImm);
    }

    #[test]
    fn tick_increments_counter() {
        reset();
        tick(ShapeId::ThumbMovImm);
        tick(ShapeId::ThumbMovImm);
        tick(ShapeId::ArmCmpImm);
        let snap = snapshot();
        assert_eq!(snap[ShapeId::ThumbMovImm as usize], 2);
        assert_eq!(snap[ShapeId::ArmCmpImm as usize], 1);
        assert_eq!(snap[ShapeId::ThumbAddImm as usize], 0);
    }

    #[test]
    fn dump_includes_all_shape_names() {
        reset();
        tick(ShapeId::ThumbMovImm);
        let s = dump();
        for name in &SHAPE_NAMES {
            assert!(
                s.contains(&format!("shape_profile:{:<16}", name)),
                "dump missing shape {}: {}",
                name,
                s
            );
        }
    }
}
