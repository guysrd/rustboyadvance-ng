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

/// Shape categories. Three groups:
///
/// 1. **Single-instruction bench shapes** — one variant per `bench_*`
///    function in `arm7tdmi/benches/dynarec_shapes.rs` that measures
///    a single-opcode compiled block. Retaining these makes `W_*`
///    retraining a one-to-one replacement.
///
/// 2. **Block-structural buckets** — the program mandates block-level
///    optimization targets (see the experiment idea menu, part C in
///    `docs/dynarec-arm64-program.md`). A single `thumb_dp_chain`
///    bucket for every multi-instr block was too coarse — a 6-instr
///    block with an embedded LDR has a very different cost curve from
///    the 3-instr MOV/MOV/ADD triplet the bench measures. These
///    buckets split by (length × mem-op presence × terminator) so
///    the agent can see which cost-class moves after each experiment.
///
/// 3. **Pattern-matched shapes** — specific idioms lowered by
///    `patterns.rs`. Today that's only `ShiftPairSxtb`; as new
///    patterns land (MUL-by-const, UDIV-by-const, abs, CLZ,
///    popcount, MIN/MAX, byteswap, CpuSet-fold, DCE), **each
///    gets its own variant here** so the counter can tell
///    "pattern X fired N times" apart from the plain structural
///    fallback it replaced. Add the variant alongside the
///    `patterns.rs` classifier entry and the `bench_<pattern>`
///    in `dynarec_shapes.rs`; see the "Each new synthesized shape
///    lands with..." checklist in `program-shapekarpathy.md`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum ShapeId {
    // --- Group 1: single-instruction bench shapes ---
    ThumbMovImm = 0,
    ThumbAddImm = 1,
    ThumbCmpImm = 2,
    ArmMovImm = 3,
    ArmCmpImm = 4,
    // --- Group 2: block-structural buckets ---
    /// 2 Thumb instructions, all data-processing (no mem, no branch).
    ThumbDpPairPure = 5,
    /// 3–5 Thumb instructions, all data-processing.
    ThumbDpShortPure = 6,
    /// 6+ Thumb instructions, all data-processing.
    ThumbDpLongPure = 7,
    /// Any Thumb block containing at least one memory op (format
    /// 7/8/9/10/11/14 LDR/STR/PUSH/POP), regardless of length.
    ThumbBlockWithMem = 8,
    /// Same as the above groups but with a branch-terminator detected
    /// in the last opcode (Bcc / B / BX / BL pair). Separate bucket
    /// because emit_conditional_* paths cost different from fallthrough.
    ThumbBlockWithBranch = 9,
    // --- Group 3: pattern-matched shapes (add new variants as
    //            patterns.rs grows) ---
    ShiftPairSxtb = 10,
    // --- Catch-all ---
    Other = 11,
}

pub const N_SHAPES: usize = 12;

/// Human-readable names in the same order as the enum discriminants.
/// Used by `dump()` to label each counter in the output block. Keep
/// this in lockstep with the enum above; there's a unit test that
/// verifies the count matches.
const SHAPE_NAMES: [&str; N_SHAPES] = [
    "thumb_mov_imm",
    "thumb_add_imm",
    "thumb_cmp_imm",
    "arm_mov_imm",
    "arm_cmp_imm",
    "thumb_dp_pair_pure",
    "thumb_dp_short_pure",
    "thumb_dp_long_pure",
    "thumb_block_with_mem",
    "thumb_block_with_branch",
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
/// Classification order (first match wins, most specific → most
/// general):
///
/// 1. Pattern-matched shapes (ShiftPairSxtb and future additions) —
///    these are the highest-signal buckets, so check first.
/// 2. Single-instruction bench shapes (Thumb format 3 MOV/ADD/CMP).
/// 3. Multi-instruction structural buckets, split by:
///    a. Has any memory op → ThumbBlockWithMem
///    b. Has a branch terminator → ThumbBlockWithBranch
///    c. Otherwise pure DP → length bucket (Pair / Short / Long)
/// 4. Everything else → Other (a non-zero `other` count on the
///    profile is a signal that a common block shape isn't being
///    recognized; the classifier needs an update).
pub fn classify_thumb(raws: &[u16]) -> ShapeId {
    // Pattern: shift-pair sign/zero extend. 2-opcode trigger.
    if raws.len() == 2
        && is_thumb_shift_pair_sign_or_zero_extend(raws[0], raws[1])
    {
        return ShapeId::ShiftPairSxtb;
    }
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
    // Multi-instruction structural buckets.
    if raws.len() >= 2 {
        if raws.iter().any(|&op| is_thumb_mem_opcode(op)) {
            return ShapeId::ThumbBlockWithMem;
        }
        if raws
            .last()
            .map(|&op| is_thumb_branch_opcode(op))
            .unwrap_or(false)
        {
            return ShapeId::ThumbBlockWithBranch;
        }
        return match raws.len() {
            2 => ShapeId::ThumbDpPairPure,
            3..=5 => ShapeId::ThumbDpShortPure,
            _ => ShapeId::ThumbDpLongPure,
        };
    }
    ShapeId::Other
}

/// True if `op` is any Thumb memory-transfer opcode we'd want to
/// flag on a block-level classifier. Covers formats 7, 8, 9, 10, 11,
/// 14 — the full LDR/STR/LDRB/STRB/LDRH/STRH/PUSH/POP family. Doesn't
/// need to be tight — false positives just promote a block from a
/// Pure bucket to WithMem, which is the conservative direction for a
/// block-cost classifier.
fn is_thumb_mem_opcode(op: u16) -> bool {
    let top4 = op >> 12;
    // Format 7 (01010 / 01011... LDR/STR reg offset, LDRB/STRB reg
    // offset, LDRH/STRH reg offset): top 4 bits 0101.
    // Format 8 (0101...): also 0101, distinguished by bit 9.
    // Format 9 (011x...): top 4 bits 0110 or 0111 → LDR/STR imm offset
    // word or byte.
    // Format 10 (1000...): LDRH/STRH imm offset.
    // Format 11 (1001...): SP-relative LDR/STR.
    // Format 14 (1011x10x...): PUSH/POP; specifically 1011_x10x_rrrrrrrr.
    match top4 {
        0b0101 | 0b0110 | 0b0111 | 0b1000 | 0b1001 => true,
        0b1011 => (op & 0x0600) == 0x0400, // PUSH/POP (format 14)
        _ => false,
    }
}

/// True if `op` is a Thumb branch terminator we'd want to flag. Covers
/// format 16 (Bcc), format 17 (SWI — not a branch but acts like one),
/// format 18 (B unconditional), format 19 (BL long). Plus BX / BLX
/// from format 5 (with Hi-reg bit set).
fn is_thumb_branch_opcode(op: u16) -> bool {
    let top4 = op >> 12;
    match top4 {
        // Format 16 (1101): Bcc OR SWI (1101_1111 = SWI, else Bcc).
        0b1101 => true,
        // Format 18 (11100): unconditional B.
        0b1110 => (op & 0xF800) == 0xE000,
        // Format 19 (1111): BL high/low half — both halves are
        // branch-related.
        0b1111 => true,
        // Format 5 BX (010001_11_0_rs_000): bits 15..10 = 010001, bits
        // 9..8 = 11. Mask: 0xFF00 == 0x4700.
        0b0100 => (op & 0xFF00) == 0x4700,
        _ => false,
    }
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
    fn two_thumb_dp_instrs_classified_as_dp_pair_pure() {
        // MOV R0,#1 ; ADD R0,R0,R1  (2 instructions, all DP)
        // 0x2001 = MOV R0,#1 (format 3)
        // 0x1840 = ADD R0,R0,R1 (format 2)
        assert_eq!(
            classify_thumb(&[0x2001, 0x1840]),
            ShapeId::ThumbDpPairPure
        );
    }

    #[test]
    fn three_thumb_instrs_is_dp_short_pure() {
        // MOV R0,#1 ; MOV R1,#2 ; ADD R0,R0,R1  (all DP, length 3)
        assert_eq!(
            classify_thumb(&[0x2001, 0x2102, 0x1840]),
            ShapeId::ThumbDpShortPure
        );
    }

    #[test]
    fn six_thumb_instrs_is_dp_long_pure() {
        // Six format-3 MOVs (length 6, all DP)
        assert_eq!(
            classify_thumb(&[0x2001, 0x2102, 0x2203, 0x2304, 0x2405, 0x2506]),
            ShapeId::ThumbDpLongPure
        );
    }

    #[test]
    fn block_with_ldr_is_block_with_mem() {
        // 0x6801 = LDR R1, [R0, #0]  (format 9, word load)
        // 0x2002 = MOV R0,#2 (format 3)
        assert_eq!(
            classify_thumb(&[0x2002, 0x6801]),
            ShapeId::ThumbBlockWithMem
        );
    }

    #[test]
    fn block_with_pop_is_block_with_mem() {
        // 0xBD00 = POP {PC} (format 14)
        // 0x2001 = MOV R0,#1 (format 3)
        assert_eq!(
            classify_thumb(&[0x2001, 0xBD00]),
            ShapeId::ThumbBlockWithMem
        );
    }

    #[test]
    fn block_with_bcc_terminator_is_block_with_branch() {
        // 0x2001 = MOV R0,#1 (format 3)
        // 0xD001 = BEQ +1 (format 16 conditional branch)
        assert_eq!(
            classify_thumb(&[0x2001, 0xD001]),
            ShapeId::ThumbBlockWithBranch
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
