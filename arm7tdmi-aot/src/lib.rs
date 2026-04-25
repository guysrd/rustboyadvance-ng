//! AOT-LLVM dispatcher for the ARM7TDMI core.
//!
//! At ROM load time, scan reachable Thumb/ARM blocks, emit inline
//! LLVM IR for each, batch-compile the resulting modules, and
//! populate a fast PC->fn lookup table. The dispatcher in
//! `arm7tdmi::cache::replay_cached_block` queries the table before
//! falling through to the cached_interp scalar replay path.
//!
//! See `docs/aot-llvm-program.md` for the autoresearch program
//! (Karpathy format) that drives the development of this crate.
//! Invariants I1-I28 in that doc are NOT optional.

pub mod compiler;

pub use compiler::LlvmCompiler;
