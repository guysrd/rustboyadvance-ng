//! ARM → native-code dynarec, backed by Cranelift.
//!
//! Early-stage scaffolding. The goal is to translate hot ARM data-processing
//! instructions into native machine code at block-record time and call the
//! compiled block from `Arm7tdmiCore::step_block` instead of replaying the
//! cached interpreter's decoded instruction stream.
//!
//! Current scope (MVP):
//!   - Cranelift JIT module wrapper (one per CPU instance).
//!   - Compile-and-invoke of a trivial function that manipulates the CPU's
//!     general-purpose register array. Proves the end-to-end path works
//!     without pulling in the full ARM decoder yet.
//!   - Unit test that runs the compiled function against a real gpr array
//!     and asserts the expected post-state.
//!
//! The next increments will:
//!   1. Define a `CpuCtx` layout struct matching the in-memory layout of
//!      `Arm7tdmiCore`'s gpr / pc / cpsr fields, and pass a pointer of that
//!      type through Cranelift.
//!   2. Per-instruction codegen for MOV/ADD/SUB/MVN (immediate and register
//!      forms) without memory access or flag-setting.
//!   3. Flag materialization on request (S-bit-set instructions).
//!   4. Memory access via a callback into the bus (trampoline).
//!   5. Condition-code check wrapper per instruction.
//!   6. Branch handling that either falls through to the next block lookup
//!      or re-enters the dispatcher.
//!
//! Until all of that is in place, `compile_block` returns `None` and the
//! cached interpreter keeps running. This module compiles (and its tests
//! pass) only when the `dynarec` cargo feature is enabled.

use cranelift::codegen::ir::immediates::Offset32;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

/// Differential-execution helpers, shared between `#[cfg(test)] mod tests`
/// and `tests/dynarec_pattern_differential.rs`. Always compiled under
/// `dynarec` so integration test binaries can link to them.
pub mod test_utils;

/// Per-shape execution-count profiler, gated behind the `shape_profile`
/// feature. Drives the `W_*` retraining step in scripts/dynarec_measure.sh
/// — the counters here tell the agent how many times per second each
/// bench shape's compiled block actually runs in real gameplay, replacing
/// the hand-picked constants with measured ones. Pure no-op on the
/// default build.
#[cfg(feature = "shape_profile")]
pub mod shape_profile;

/// Block-level pattern matcher: given a Thumb/ARM opcode block, try to
/// recognize a hand-lowered shape (mul-by-constant, div-by-constant, CLZ,
/// popcount, etc.) and emit a tighter host-code stencil. Returning `None`
/// means "no pattern matched, fall through to the per-instruction emitters
/// as usual." Stub for the shape-optimization research loop (see
/// docs/program-shapekarpathy.md).
pub(crate) mod patterns;

/// Cranelift-disasm dump helper: same code paths as production, but
/// bypasses define_function so we can pull `CompiledCode::code_buffer()`
/// and hand it to capstone for arm64 disassembly. Used by the research
/// loop's baseline tests; not in the default build.
#[cfg(feature = "dynarec_asm_dump")]
pub mod dump;

/// Rust side trampoline signatures the dynarec calls into for memory ops.
/// The opaque *mut u8 is a "cpu ctx" pointer; whoever constructs the
/// DynarecCompiler supplies trampolines that interpret that pointer the
/// right way for their CPU type. A typical trampoline would cast the
/// pointer to `*mut Arm7tdmiCore<MyBus>` and forward to `cpu.load_32`.
pub type BusLoad32Fn = unsafe extern "C" fn(*mut u8, u32) -> u32;
pub type BusStore32Fn = unsafe extern "C" fn(*mut u8, u32, u32);
pub type BusLoad8Fn = unsafe extern "C" fn(*mut u8, u32) -> u32;
pub type BusStore8Fn = unsafe extern "C" fn(*mut u8, u32, u32);
/// Like load_32/load_8 but additionally pays the +1I "internal" cycle
/// that scalar LDR / LDRB / LDRH all charge after the data fetch.
/// Used by codegen for *data* loads (LDR family) so cycle accounting
/// stays parity-correct with the interpreter; instruction fetches
/// (`thumb_fetch_n`) deliberately don't pay +1I.
pub type BusLoadIdle32Fn = unsafe extern "C" fn(*mut u8, u32) -> u32;
pub type BusLoadIdle8Fn = unsafe extern "C" fn(*mut u8, u32) -> u32;
/// Forces `cpu.next_fetch_access = NonSeq`. Called from compiled blocks
/// right before return when the LAST emitted instruction is a STORE
/// (or any other shape that would have returned `CpuAction::AdvancePC(NonSeq)`
/// in the interpreter), so the very next post-block fetch in the cached
/// interpreter loop pays NonSeq cycles like it would have under scalar
/// dispatch.
pub type BusSetNextFetchNonSeqFn = unsafe extern "C" fn(*mut u8);
/// Pay the *extra* cycles a NonSeq Thumb fetch costs over a Seq one
/// for the address `fetch_pc` — i.e. `n_cycles16[page] - s_cycles16[page]`.
/// Called from compiled blocks once per intermediate STORE in the body,
/// passing the PC of the next fetch (the one that would have been
/// NonSeq in scalar but was pre-paid as Seq by `thumb_fetch_n`).
pub type BusPayThumbFetchExtraNonSeqFn = unsafe extern "C" fn(*mut u8, u32);
/// Standalone +1I idle cycle — scalar POP/POP{PC} add this at the end
/// of the whole multi-load (see exec_thumb_push_pop comment "Idle 1 cycle").
pub type BusIdleCycleFn = unsafe extern "C" fn(*mut u8);
/// Pay scheduler cycles for N Thumb fetches and update CPU pipeline/pc.
/// Called once at each compiled Thumb block's entry so cycle accounting
/// stays parity-correct with the interpreter.
pub type BusThumbFetchNFn = unsafe extern "C" fn(*mut u8, u32, u32);
/// Link-time block-chaining guard. Called from a compiled block's
/// Tail::Body epilogue immediately before it tail-calls its linked
/// successor block. Returns non-zero when the outer dispatcher must
/// get control back instead of staying on the compiled hot path —
/// mirrors the abort conditions that scalar `replay_cached_block`
/// checks every other instruction:
///   - RAM dirty (self-modifying code flush).
///   - `cached_block_should_abort` (IRQ / DMA / halt / scheduler event).
/// Returns 0 to continue into the chain target.
pub type BusChainAbortCheckFn = unsafe extern "C" fn(*mut u8) -> u32;
/// Mid-block abort state-writeback. Called from compiled blocks when
/// `chain_abort_check` fires mid-body to restore CPU state to what a
/// scalar replay would have left at the SAME abort point. Sets
/// `cpu.pc`, `cpu.pipeline[0/1]`, and `cpu.next_fetch_access` so the
/// next `replay_cached_block` dispatch reads the right instruction.
/// pipeline values are baked in at codegen (they are known raw
/// opcodes from the recorded block).
pub type BusAbortMidBlockFn =
    unsafe extern "C" fn(*mut u8, u32 /* pc */, u32 /* pipe0 */, u32 /* pipe1 */);

/// Generic bus trampolines. Instantiate with the concrete MemoryInterface
/// type of your CPU. The `cpu_ctx` opaque pointer passed to the dynarec
/// compiled block must be a `*mut Arm7tdmiCore<I>` cast to `*mut u8`.
/// Data accesses use MemoryAccess::NonSeq, matching the ARM7TDMI
/// interpreter's LDR/STR path.
pub mod trampolines {
    use crate::cpu::Arm7tdmiCore;
    use crate::memory::{MemoryAccess, MemoryInterface};

    pub unsafe extern "C" fn load_32<I: MemoryInterface>(ctx: *mut u8, addr: u32) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.load_32(addr, MemoryAccess::NonSeq)
    }
    pub unsafe extern "C" fn store_32<I: MemoryInterface>(ctx: *mut u8, addr: u32, value: u32) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.store_32(addr, value, MemoryAccess::NonSeq);
    }
    /// Sequential-access variants of load_32 / store_32 for the
    /// 2nd+ slot of a multi-load/store (PUSH/POP, LDM/STM). Scalar
    /// pays 1 NonSeq cycle for the first access and 1 Seq for each
    /// subsequent — mirroring that here avoids cycle-accounting
    /// drift that accumulates over many multi-access ops.
    pub unsafe extern "C" fn load_32_seq<I: MemoryInterface>(ctx: *mut u8, addr: u32) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.load_32(addr, MemoryAccess::Seq)
    }
    pub unsafe extern "C" fn store_32_seq<I: MemoryInterface>(ctx: *mut u8, addr: u32, value: u32) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.store_32(addr, value, MemoryAccess::Seq);
    }
    /// 16-bit halfword STRH — mirrors scalar `store_aligned_16`
    /// (masks low bit of address before the store).
    pub unsafe extern "C" fn store_16<I: MemoryInterface>(ctx: *mut u8, addr: u32, value: u32) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.store_16(addr & !0x1, value as u16, MemoryAccess::NonSeq);
    }
    /// 16-bit halfword LDRH with +1I. Mirrors scalar `ldr_half`:
    /// on misaligned addr, rotates the halfword right by 8 so the
    /// byte-at-address ends in the high lane (ARM7TDMI quirk).
    pub unsafe extern "C" fn load_with_idle_16<I: MemoryInterface>(
        ctx: *mut u8,
        addr: u32,
    ) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        let v = if addr & 0x1 != 0 {
            let loaded = cpu.load_16(addr & !0x1, MemoryAccess::NonSeq) as u32;
            loaded.rotate_right(8)
        } else {
            cpu.load_16(addr, MemoryAccess::NonSeq) as u32
        };
        cpu.idle_cycle();
        v
    }
    pub unsafe extern "C" fn load_8<I: MemoryInterface>(ctx: *mut u8, addr: u32) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.load_8(addr, MemoryAccess::NonSeq) as u32
    }
    pub unsafe extern "C" fn store_8<I: MemoryInterface>(ctx: *mut u8, addr: u32, value: u32) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.store_8(addr, value as u8, MemoryAccess::NonSeq);
    }

    /// LDR data-load trampoline: load_32 + the "+1I" idle cycle every
    /// scalar LDR / LDRH / LDRB charges after the data access. Applies
    /// the unaligned-rotate scalar `ldr_word` does: if addr & 3 != 0,
    /// the returned word is rotated right by `(addr & 3) * 8` bits so
    /// the byte at `addr` ends up in the low lane, matching what the
    /// ARM7TDMI actually returns for misaligned LDR.
    pub unsafe extern "C" fn load_with_idle_32<I: MemoryInterface>(
        ctx: *mut u8,
        addr: u32,
    ) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        let v = cpu.load_32(addr, MemoryAccess::NonSeq);
        let rotated = if addr & 3 != 0 {
            let rotation = (addr & 3) << 3;
            v.rotate_right(rotation)
        } else {
            v
        };
        cpu.idle_cycle();
        rotated
    }
    pub unsafe extern "C" fn load_with_idle_8<I: MemoryInterface>(
        ctx: *mut u8,
        addr: u32,
    ) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        let v = cpu.load_8(addr, MemoryAccess::NonSeq) as u32;
        cpu.idle_cycle();
        v
    }

    /// Force the CPU's `next_fetch_access` to NonSeq. Called from
    /// compiled blocks just before return when the last emitted
    /// instruction is a STORE — scalar dispatch would have set
    /// next_fetch_access via `CpuAction::AdvancePC(NonSeq)`, this
    /// trampoline mirrors that for the cached-interp loop's fetch
    /// after the compiled block returns.
    pub unsafe extern "C" fn set_next_fetch_nonseq<I: MemoryInterface>(ctx: *mut u8) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.next_fetch_access = MemoryAccess::NonSeq;
    }

    /// Standalone +1I idle cycle. Used by codegen for POP and POP{PC}
    /// which in scalar add a single idle cycle at the end of the
    /// multi-load (not per-register, unlike LDR which pairs with
    /// load_with_idle_32).
    pub unsafe extern "C" fn idle_cycle<I: MemoryInterface>(ctx: *mut u8) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.idle_cycle();
    }

    /// Compensation trampoline for the in-block STORE-followed-by-X
    /// case. `thumb_fetch_n` pre-paid every fetch after the first as
    /// Seq, but scalar STR sets the next fetch's access to NonSeq via
    /// `CpuAction::AdvancePC(NonSeq)`. For each intermediate STORE in
    /// a compiled block we charge the missing `n - s` cycles for the
    /// fetch at `fetch_pc` here. Forwards to the bus's
    /// `pay_thumb_fetch_extra_nonseq` (default is no-op for test buses,
    /// SysBus implements via the cycle LUT).
    pub unsafe extern "C" fn pay_thumb_fetch_extra_nonseq<I: MemoryInterface>(
        ctx: *mut u8,
        fetch_pc: u32,
    ) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.bus.pay_thumb_fetch_extra_nonseq(fetch_pc);
    }

    /// Pay the scheduler cycles for N Thumb instruction fetches at the
    /// start of a compiled block, and update the CPU's pipeline[0/1] +
    /// next_fetch_access + pc so post-block state matches what the
    /// interpreter would have produced after running N iterations of
    /// replay_cached_block's Thumb loop.
    ///
    /// first_fetch_pc is the address of the FIRST fetch the interpreter
    /// would have done, which equals (block_start_addr + 4) in Thumb
    /// per the pipeline-head convention. The fetch at iteration k lands
    /// at first_fetch_pc + 2*(k-1).
    ///
    /// Pipeline semantics after this call match iter-N-exit in the
    /// interpreter:
    ///   - pipeline[0] = fetched value at iter N-1 (for count >= 2) or
    ///     the old pipeline[1] (for count == 1)
    ///   - pipeline[1] = fetched value at iter N
    ///   - next_fetch_access = Seq
    ///   - pc = first_fetch_pc + 2*count
    ///
    /// The first fetch uses whatever access mode the CPU had on entry
    /// (typically Seq mid-run, NonSeq right after a pipeline flush);
    /// subsequent fetches are all Seq, matching the interpreter.
    /// Global counters for thumb_fetch_n — gated on the
    /// DYNAREC_TIME_FETCH_N env var being set at dump time. Using
    /// relaxed atomic counters (one u64 of ns accumulated + one u64
    /// of call count) so we can report avg-ns-per-call at replay end
    /// and decide if thumb_fetch_n is the bottleneck.
    pub static FETCH_N_TOTAL_NS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    pub static FETCH_N_CALLS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    pub unsafe extern "C" fn thumb_fetch_n<I: MemoryInterface>(
        ctx: *mut u8,
        first_fetch_pc: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let time_it = std::env::var_os("DYNAREC_TIME_FETCH_N").is_some();
        let t0 = if time_it { Some(std::time::Instant::now()) } else { None };
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        let mut access = cpu.next_fetch_access;
        let mut prev_fetched: u32 = cpu.pipeline[1];
        let mut last_fetched: u32 = 0;
        for i in 0..count {
            let pc = first_fetch_pc.wrapping_add(2 * i);
            let val = cpu.load_16(pc, access) as u32;
            access = MemoryAccess::Seq;
            if i > 0 {
                prev_fetched = last_fetched;
            }
            last_fetched = val;
        }
        cpu.pipeline[0] = prev_fetched;
        cpu.pipeline[1] = last_fetched;
        cpu.next_fetch_access = MemoryAccess::Seq;
        cpu.pc = first_fetch_pc.wrapping_add(2 * count);
        if let Some(t0) = t0 {
            let ns = t0.elapsed().as_nanos() as u64;
            FETCH_N_TOTAL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            FETCH_N_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Chain-abort check called from the compiled-block epilogue right
    /// before tail-calling the chained successor. Returns non-zero when
    /// the compiled hot path must bail to the outer dispatcher —
    /// mirrors the checks scalar `replay_cached_block` does every
    /// other instruction. Only queried when a chain slot is linked;
    /// unlinked blocks skip straight to normal return.
    ///
    /// If `take_block_cache_dirty` is true this ALSO flushes the RAM
    /// side of the cache, matching scalar's behavior at the same
    /// check point (cpu.rs:632-635).
    #[cfg(feature = "cached_interp")]
    pub unsafe extern "C" fn chain_abort_check<I: MemoryInterface>(ctx: *mut u8) -> u32 {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        if cpu.bus.take_block_cache_dirty() {
            cpu.block_cache.flush();
            return 1;
        }
        if cpu.bus.cached_block_should_abort() {
            return 1;
        }
        0
    }
    /// Stand-in for builds without `cached_interp` — the whole
    /// chaining path is a no-op there because block-cache state
    /// doesn't exist. Always returns 1 ("abort") so that even if the
    /// chain slot happens to be non-null the compiled code bails to
    /// the dispatcher instead of calling through a stale pointer.
    #[cfg(not(feature = "cached_interp"))]
    pub unsafe extern "C" fn chain_abort_check<I: MemoryInterface>(_ctx: *mut u8) -> u32 {
        1
    }

    /// Mid-block abort writeback. Called by compiled code when
    /// `chain_abort_check` fires between body items — restores
    /// the CPU state a scalar replay would have left at the same
    /// point so the next `replay_cached_block` dispatch resumes
    /// from the correct pipeline[0]. pipeline values + pc are
    /// constants baked in at codegen time.
    pub unsafe extern "C" fn abort_mid_block<I: MemoryInterface>(
        ctx: *mut u8,
        pc: u32,
        pipe0: u32,
        pipe1: u32,
    ) {
        let cpu = unsafe { &mut *(ctx as *mut Arm7tdmiCore<I>) };
        cpu.pc = pc;
        cpu.pipeline[0] = pipe0;
        cpu.pipeline[1] = pipe1;
        cpu.next_fetch_access = MemoryAccess::Seq;
    }

    /// Fill a `BusTrampolines` with pointers to the generic trampolines
    /// monomorphized for `I`. Call once at compiler construction time and
    /// hand the returned struct to `DynarecCompiler::new_with_bus`.
    pub fn for_cpu<I: MemoryInterface>() -> super::BusTrampolines {
        super::BusTrampolines {
            load_32: load_32::<I>,
            store_32: store_32::<I>,
            load_8: load_8::<I>,
            store_8: store_8::<I>,
            load_32_seq: load_32_seq::<I>,
            store_32_seq: store_32_seq::<I>,
            store_16: store_16::<I>,
            load_with_idle_16: load_with_idle_16::<I>,
            load_with_idle_32: load_with_idle_32::<I>,
            load_with_idle_8: load_with_idle_8::<I>,
            set_next_fetch_nonseq: set_next_fetch_nonseq::<I>,
            pay_thumb_fetch_extra_nonseq: pay_thumb_fetch_extra_nonseq::<I>,
            idle_cycle: idle_cycle::<I>,
            thumb_fetch_n: thumb_fetch_n::<I>,
            chain_abort_check: chain_abort_check::<I>,
            abort_mid_block: abort_mid_block::<I>,
        }
    }
}

/// Optional set of bus trampolines. When None the dynarec will refuse to
/// compile anything that needs memory access (returns None from compile
/// paths). When set, compiled blocks can call into these at runtime.
#[derive(Clone, Copy)]
pub struct BusTrampolines {
    pub load_32: BusLoad32Fn,
    pub store_32: BusStore32Fn,
    pub load_8: BusLoad8Fn,
    pub store_8: BusStore8Fn,
    /// Sequential-access load/store for the 2nd+ access in a
    /// multi-access op (PUSH/POP/LDM/STM). Same signature as the
    /// NonSeq variants; internal trampoline switches access mode.
    pub load_32_seq: BusLoad32Fn,
    pub store_32_seq: BusStore32Fn,
    /// 16-bit halfword access for Thumb format 10 LDRH / STRH imm.
    pub store_16: BusStore32Fn,
    pub load_with_idle_16: BusLoadIdle32Fn,
    /// Data-load with implicit +1I idle cycle (LDR/LDRH/LDRB family).
    pub load_with_idle_32: BusLoadIdle32Fn,
    pub load_with_idle_8: BusLoadIdle8Fn,
    /// Force `cpu.next_fetch_access = NonSeq` for post-block-store fixup.
    pub set_next_fetch_nonseq: BusSetNextFetchNonSeqFn,
    /// Pay the `n - s` cycle delta for a single intermediate STORE's
    /// next fetch.
    pub pay_thumb_fetch_extra_nonseq: BusPayThumbFetchExtraNonSeqFn,
    /// Standalone +1I idle cycle for POP/POP{PC} end-of-load idle.
    pub idle_cycle: BusIdleCycleFn,
    pub thumb_fetch_n: BusThumbFetchNFn,
    /// Link-time block-chaining abort guard. Consulted from compiled
    /// Tail::Body epilogues before a chained tail-call.
    pub chain_abort_check: BusChainAbortCheckFn,
    /// Mid-block abort state writeback (pc + pipeline[0/1] + next_fetch_access).
    pub abort_mid_block: BusAbortMidBlockFn,
}

/// Handle to a Cranelift JIT module. One per CPU instance; freed on CPU drop.
///
/// Wrapping the Cranelift state here keeps the module lifetime tied to the
/// CPU so generated code pages are reclaimed when the emulator tears down.
pub struct DynarecCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    next_id: u64,
    /// Resolved imports for the bus trampolines. Only populated when the
    /// compiler was built with `new_with_bus`. The FuncId values are stable
    /// for the lifetime of this compiler and can be re-referenced in every
    /// compiled function via `module.declare_func_in_func`.
    bus_imports: Option<BusImports>,
}

/// Imported function handles registered with the JITModule for the bus
/// trampolines. `declare_func_in_func` re uses these across blocks.
#[derive(Clone, Copy)]
struct BusImports {
    load_32: FuncId,
    store_32: FuncId,
    /// Declared for completeness — all LDRB-family codegen goes
    /// through `load_with_idle_8` so the +1I cycle is paid. Kept
    /// around so future scalar-load use sites don't have to redo
    /// the JIT import plumbing.
    #[allow(dead_code)]
    load_8: FuncId,
    store_8: FuncId,
    load_32_seq: FuncId,
    store_32_seq: FuncId,
    store_16: FuncId,
    load_with_idle_16: FuncId,
    load_with_idle_32: FuncId,
    load_with_idle_8: FuncId,
    set_next_fetch_nonseq: FuncId,
    pay_thumb_fetch_extra_nonseq: FuncId,
    idle_cycle: FuncId,
    thumb_fetch_n: FuncId,
    chain_abort_check: FuncId,
    abort_mid_block: FuncId,
}

impl DynarecCompiler {
    pub fn new() -> Self {
        Self::build(None)
    }

    /// Build a dynarec compiler wired to a set of bus trampolines. Needed
    /// before any LDR/STR compilation can succeed.
    pub fn new_with_bus(bus: BusTrampolines) -> Self {
        Self::build(Some(bus))
    }

    fn build(bus: Option<BusTrampolines>) -> Self {
        let isa_builder = cranelift_native::builder()
            .expect("host architecture not supported by Cranelift");
        let mut flag_builder = settings::builder();
        // We don't need stack unwinding for these JIT blocks — they're
        // never in the middle of a panic unwind, and any exception path
        // returns via the normal fn pointer. Disabling unwind_info
        // shrinks the emitted prologue/epilogue (no CFI directives).
        flag_builder
            .set("unwind_info", "false")
            .expect("set unwind_info=false");
        let flags = settings::Flags::new(flag_builder);
        let isa = isa_builder
            .finish(flags)
            .expect("failed to build Cranelift ISA for host");

        let mut jit_builder =
            JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // If the caller passed trampolines, register them as JIT symbols so
        // declare_function(Linkage::Import, ...) can resolve to the real
        // rust side fn pointers at link time.
        if let Some(b) = bus {
            jit_builder.symbol("rba_bus_load_32",            b.load_32              as *const u8);
            jit_builder.symbol("rba_bus_store_32",           b.store_32             as *const u8);
            jit_builder.symbol("rba_bus_load_8",             b.load_8               as *const u8);
            jit_builder.symbol("rba_bus_store_8",            b.store_8              as *const u8);
            jit_builder.symbol("rba_bus_load_32_seq",        b.load_32_seq          as *const u8);
            jit_builder.symbol("rba_bus_store_32_seq",       b.store_32_seq         as *const u8);
            jit_builder.symbol("rba_bus_store_16",           b.store_16             as *const u8);
            jit_builder.symbol("rba_bus_load_with_idle_16",  b.load_with_idle_16    as *const u8);
            jit_builder.symbol("rba_bus_load_with_idle_32",  b.load_with_idle_32    as *const u8);
            jit_builder.symbol("rba_bus_load_with_idle_8",   b.load_with_idle_8     as *const u8);
            jit_builder.symbol("rba_set_next_fetch_nonseq",  b.set_next_fetch_nonseq as *const u8);
            jit_builder.symbol("rba_pay_thumb_fetch_extra_nonseq",
                                                              b.pay_thumb_fetch_extra_nonseq as *const u8);
            jit_builder.symbol("rba_idle_cycle",             b.idle_cycle           as *const u8);
            jit_builder.symbol("rba_thumb_fetch_n",          b.thumb_fetch_n        as *const u8);
            jit_builder.symbol("rba_chain_abort_check",      b.chain_abort_check    as *const u8);
            jit_builder.symbol("rba_abort_mid_block",        b.abort_mid_block      as *const u8);
        }

        let mut module = JITModule::new(jit_builder);
        let ctx = module.make_context();

        let bus_imports = bus.map(|_| {
            let ptr_ty = module.isa().pointer_type();
            // load_32: extern "C" fn(*mut u8, u32) -> u32
            let mut sig_load_32 = module.make_signature();
            sig_load_32.params.push(AbiParam::new(ptr_ty));
            sig_load_32.params.push(AbiParam::new(types::I32));
            sig_load_32.returns.push(AbiParam::new(types::I32));
            let load_32 = module
                .declare_function("rba_bus_load_32", Linkage::Import, &sig_load_32)
                .expect("declare load_32 failed");

            // store_32: extern "C" fn(*mut u8, u32, u32)
            let mut sig_store_32 = module.make_signature();
            sig_store_32.params.push(AbiParam::new(ptr_ty));
            sig_store_32.params.push(AbiParam::new(types::I32));
            sig_store_32.params.push(AbiParam::new(types::I32));
            let store_32 = module
                .declare_function("rba_bus_store_32", Linkage::Import, &sig_store_32)
                .expect("declare store_32 failed");

            // load_8: extern "C" fn(*mut u8, u32) -> u32 (zero extended)
            let mut sig_load_8 = module.make_signature();
            sig_load_8.params.push(AbiParam::new(ptr_ty));
            sig_load_8.params.push(AbiParam::new(types::I32));
            sig_load_8.returns.push(AbiParam::new(types::I32));
            let load_8 = module
                .declare_function("rba_bus_load_8", Linkage::Import, &sig_load_8)
                .expect("declare load_8 failed");

            // store_8: extern "C" fn(*mut u8, u32, u32)
            let mut sig_store_8 = module.make_signature();
            sig_store_8.params.push(AbiParam::new(ptr_ty));
            sig_store_8.params.push(AbiParam::new(types::I32));
            sig_store_8.params.push(AbiParam::new(types::I32));
            let store_8 = module
                .declare_function("rba_bus_store_8", Linkage::Import, &sig_store_8)
                .expect("declare store_8 failed");

            // load_with_idle_32: same signature as load_32; charges +1I post-load.
            let mut sig_load_idle_32 = module.make_signature();
            sig_load_idle_32.params.push(AbiParam::new(ptr_ty));
            sig_load_idle_32.params.push(AbiParam::new(types::I32));
            sig_load_idle_32.returns.push(AbiParam::new(types::I32));
            let load_with_idle_32 = module
                .declare_function("rba_bus_load_with_idle_32", Linkage::Import, &sig_load_idle_32)
                .expect("declare load_with_idle_32 failed");

            // load_32_seq / store_32_seq: same signatures as
            // load_32 / store_32, just different access type.
            let load_32_seq = module
                .declare_function("rba_bus_load_32_seq", Linkage::Import, &sig_load_32)
                .expect("declare load_32_seq failed");
            let store_32_seq = module
                .declare_function("rba_bus_store_32_seq", Linkage::Import, &sig_store_32)
                .expect("declare store_32_seq failed");

            // 16-bit halfword access (Thumb format 10 LDRH/STRH imm)
            let store_16 = module
                .declare_function("rba_bus_store_16", Linkage::Import, &sig_store_32)
                .expect("declare store_16 failed");
            let load_with_idle_16 = module
                .declare_function("rba_bus_load_with_idle_16", Linkage::Import, &sig_load_idle_32)
                .expect("declare load_with_idle_16 failed");

            // load_with_idle_8: same signature as load_8; charges +1I post-load.
            let mut sig_load_idle_8 = module.make_signature();
            sig_load_idle_8.params.push(AbiParam::new(ptr_ty));
            sig_load_idle_8.params.push(AbiParam::new(types::I32));
            sig_load_idle_8.returns.push(AbiParam::new(types::I32));
            let load_with_idle_8 = module
                .declare_function("rba_bus_load_with_idle_8", Linkage::Import, &sig_load_idle_8)
                .expect("declare load_with_idle_8 failed");

            // set_next_fetch_nonseq: extern "C" fn(*mut u8)
            let mut sig_nonseq = module.make_signature();
            sig_nonseq.params.push(AbiParam::new(ptr_ty));
            let set_next_fetch_nonseq = module
                .declare_function("rba_set_next_fetch_nonseq", Linkage::Import, &sig_nonseq)
                .expect("declare set_next_fetch_nonseq failed");

            // pay_thumb_fetch_extra_nonseq: extern "C" fn(*mut u8, u32)
            let mut sig_pay_extra = module.make_signature();
            sig_pay_extra.params.push(AbiParam::new(ptr_ty));
            sig_pay_extra.params.push(AbiParam::new(types::I32));
            let pay_thumb_fetch_extra_nonseq = module
                .declare_function(
                    "rba_pay_thumb_fetch_extra_nonseq",
                    Linkage::Import,
                    &sig_pay_extra,
                )
                .expect("declare pay_thumb_fetch_extra_nonseq failed");

            // idle_cycle: extern "C" fn(*mut u8)
            let mut sig_idle = module.make_signature();
            sig_idle.params.push(AbiParam::new(ptr_ty));
            let idle_cycle = module
                .declare_function("rba_idle_cycle", Linkage::Import, &sig_idle)
                .expect("declare idle_cycle failed");

            // thumb_fetch_n: extern "C" fn(*mut u8, u32, u32)
            let mut sig_fetch_n = module.make_signature();
            sig_fetch_n.params.push(AbiParam::new(ptr_ty));
            sig_fetch_n.params.push(AbiParam::new(types::I32));
            sig_fetch_n.params.push(AbiParam::new(types::I32));
            let thumb_fetch_n = module
                .declare_function("rba_thumb_fetch_n", Linkage::Import, &sig_fetch_n)
                .expect("declare thumb_fetch_n failed");

            // chain_abort_check: extern "C" fn(*mut u8) -> u32
            let mut sig_chain_abort = module.make_signature();
            sig_chain_abort.params.push(AbiParam::new(ptr_ty));
            sig_chain_abort.returns.push(AbiParam::new(types::I32));
            let chain_abort_check = module
                .declare_function("rba_chain_abort_check", Linkage::Import, &sig_chain_abort)
                .expect("declare chain_abort_check failed");

            // abort_mid_block: extern "C" fn(*mut u8, u32, u32, u32)
            let mut sig_abort_mid = module.make_signature();
            sig_abort_mid.params.push(AbiParam::new(ptr_ty));
            sig_abort_mid.params.push(AbiParam::new(types::I32));
            sig_abort_mid.params.push(AbiParam::new(types::I32));
            sig_abort_mid.params.push(AbiParam::new(types::I32));
            let abort_mid_block = module
                .declare_function("rba_abort_mid_block", Linkage::Import, &sig_abort_mid)
                .expect("declare abort_mid_block failed");

            BusImports {
                load_32,
                store_32,
                load_8,
                store_8,
                load_32_seq,
                store_32_seq,
                store_16,
                load_with_idle_16,
                load_with_idle_32,
                load_with_idle_8,
                set_next_fetch_nonseq,
                pay_thumb_fetch_extra_nonseq,
                idle_cycle,
                thumb_fetch_n,
                chain_abort_check,
                abort_mid_block,
            }
        });

        DynarecCompiler {
            module,
            builder_context: FunctionBuilderContext::new(),
            ctx,
            next_id: 0,
            bus_imports,
        }
    }

    /// True if this compiler was built with bus trampolines wired up, and
    /// therefore can compile memory ops.
    pub fn has_bus(&self) -> bool {
        self.bus_imports.is_some()
    }

    /// Compile a sequence of supported ARM data-processing instructions
    /// into native code. Supported shapes:
    ///
    /// Writeback (S=0 or S=1):
    ///   - MOV Rd, op2    op=1101    writes Rd = op2
    ///   - MVN Rd, op2    op=1111    writes Rd = !op2
    ///   - ADD Rd, Rn,op2 op=0100    writes Rd = Rn+op2
    ///   - SUB Rd, Rn,op2 op=0010    writes Rd = Rn-op2
    ///
    /// Compare-only (S=1 mandatory, no writeback):
    ///   - CMP Rn, op2    op=1010    flags from Rn-op2
    ///   - CMN Rn, op2    op=1011    flags from Rn+op2
    ///   - TST Rn, op2    op=1000    flags from Rn&op2
    ///   - TEQ Rn, op2    op=1001    flags from Rn^op2
    ///
    /// operand2 is either an immediate (I=1) or a register with no shift.
    /// All 14 ARM condition codes (EQ/NE/HS/LO/MI/PL/VS/VC/HI/LS/GE/LT/GT/LE/AL)
    /// are supported via runtime CPSR.NZCV check.
    ///
    /// For S=1 / compare-only instructions, the dynarec computes and writes
    /// back N, Z, C, V to the caller-owned CPSR word at `cpsr_ptr`.
    ///
    /// Returns `None` if any opcode is not one of the supported shapes.
    pub fn try_compile_imm_block(
        &mut self,
        opcodes: &[u32],
    ) -> Option<extern "C" fn(*mut u32, *mut u32)> {
        for &insn in opcodes {
            if Self::decode_supported_dp(insn).is_none() {
                return None;
            }
        }

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));   // gpr_ptr
        sig.params.push(AbiParam::new(ptr_type));   // cpsr_ptr

        self.next_id += 1;
        let name = format!("dynarec_imm_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        // Does any instruction in the block either read or write CPSR?
        // If not, we skip the CPSR load+store at function entry/exit
        // entirely — saves ~2 x86 instructions per compiled call on
        // single-shape blocks like "MOV R1, #42" (S=0, cond=AL).
        let touches_cpsr = opcodes.iter().any(|&insn| {
            let dec = Self::decode_supported_dp(insn).expect("pre-validated");
            dec.s || dec.cond != ArmCond::Al
        });

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];

            // Load CPSR only if we'll actually use it. When no instr
            // reads flags (cond=AL) or writes flags (S=0), the var
            // stays unused and Cranelift eliminates both the load and
            // the matching store at function exit.
            let cpsr_var = builder.declare_var(types::I32);
            if touches_cpsr {
                let cpsr_initial = builder.ins().load(
                    types::I32,
                    MemFlags::trusted(),
                    cpsr_ptr,
                    0,
                );
                builder.def_var(cpsr_var, cpsr_initial);
            } else {
                let zero = builder.ins().iconst(types::I32, 0);
                builder.def_var(cpsr_var, zero);
            }

            // Dead-flag-write pre-pass, mirror of the one in
            // try_compile_thumb_block (exp 10). If instruction k's
            // flag-write is wholly covered by k+1's flag-write AND
            // instruction k+1 has cond=AL AND S=1, then k's flag update
            // is observably dead.
            //
            // ARM DP flag-write masks (NZCV bitmask: N=8, Z=4, C=2, V=1):
            //   ADD/SUB/CMP/CMN:        0b1111 (full NZCV)
            //   MOV/MVN/TST/TEQ:        0b1100 (NZ only — preserves CV)
            // When S=0, the instr writes no flags → doesn't kill anything
            // downstream and can't itself be dead-eliminated (its write
            // was already zero).
            fn arm_dp_flag_write_mask(dec: &DecodedDp) -> u8 {
                if !dec.s {
                    return 0;
                }
                match dec.op {
                    DpOp::Add | DpOp::Sub | DpOp::Cmp | DpOp::Cmn => 0b1111,
                    DpOp::Mov | DpOp::Mvn | DpOp::Tst | DpOp::Teq => 0b1100,
                }
            }
            let decoded: Vec<DecodedDp> = opcodes
                .iter()
                .map(|&i| Self::decode_supported_dp(i).expect("pre-validated"))
                .collect();
            let n = decoded.len();
            let mut skip_flags: Vec<bool> = vec![false; n];
            for k in 0..n.saturating_sub(1) {
                let cur = arm_dp_flag_write_mask(&decoded[k]);
                let next_reads_flags = decoded[k + 1].cond != ArmCond::Al;
                let next_writes_full = arm_dp_flag_write_mask(&decoded[k + 1]);
                if cur != 0
                    && !next_reads_flags
                    && (cur & !next_writes_full) == 0
                {
                    skip_flags[k] = true;
                }
            }

            for (k, dec) in decoded.iter().enumerate() {
                if skip_flags[k] {
                    emit_conditional_instr_no_flags(
                        &mut builder, gpr_ptr, cpsr_var, *dec,
                    );
                } else {
                    emit_conditional_instr(&mut builder, gpr_ptr, cpsr_var, *dec);
                }
            }

            if touches_cpsr {
                let cpsr_final = builder.use_var(cpsr_var);
                builder.ins().store(
                    MemFlags::trusted(),
                    cpsr_final,
                    cpsr_ptr,
                    0,
                );
            }

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        // SAFETY: declared signature matches the transmute target.
        Some(unsafe {
            std::mem::transmute::<*const u8, extern "C" fn(*mut u32, *mut u32)>(code)
        })
    }

    /// Variant of `try_compile_imm_block` that accepts an optional trailing
    /// ARM B / BL as a block terminator. The returned function writes the
    /// new PC value (already adjusted to the ARM branch's PC+8+offset<<2
    /// semantics) into `*pc_out` and returns 1 iff the branch was taken,
    /// otherwise returns 0 and leaves `*pc_out` untouched.
    ///
    /// `entry_pc` is the address of the first instruction in `opcodes` as
    /// the ARM pipeline sees it (i.e. the pc the interpreter would be at
    /// while decoding the first instr, which equals the instruction's own
    /// address plus 8 per ARM pipeline convention). We fold that into the
    /// generated code so the runtime doesn't need to know where the block
    /// lived.
    ///
    /// Returns None if any instruction other than the optional trailing B/BL
    /// isn't a supported DP shape, or if a B/BL shows up anywhere other
    /// than the last slot.
    pub fn try_compile_block_with_branch(
        &mut self,
        opcodes: &[u32],
        entry_pc: u32,
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u32) -> u32> {
        if opcodes.is_empty() {
            return None;
        }

        // Branch may only appear as the last instruction.
        let (body_opcodes, tail) = opcodes.split_at(opcodes.len() - 1);
        let tail_insn = tail[0];

        // First: every body instruction must be a DP shape we can emit.
        for &insn in body_opcodes {
            if Self::decode_supported_dp(insn).is_none() {
                return None;
            }
            if Self::decode_branch(insn).is_some() {
                // B/BL found mid block; reject until we know how to handle
                // mid block early return vs scheduler interleaving cleanly.
                return None;
            }
        }

        // The tail can be either a DP shape (no branch) OR a B/BL.
        let tail_is_dp = Self::decode_supported_dp(tail_insn).is_some();
        let tail_is_branch = Self::decode_branch(tail_insn);
        if !tail_is_dp && tail_is_branch.is_none() {
            return None;
        }

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // gpr_ptr
        sig.params.push(AbiParam::new(ptr_type)); // cpsr_ptr
        sig.params.push(AbiParam::new(ptr_type)); // pc_out
        sig.returns.push(AbiParam::new(types::I32));

        self.next_id += 1;
        let name = format!("dynarec_branch_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];
            let pc_out = builder.block_params(entry)[2];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            // Pre-compute the entry pc of each instruction as if the ARM
            // pipeline were live. The first body instr is at entry_pc-8 from
            // the branch's perspective (since ARM PC during execute is
            // current_instr+8). For the tail branch we use entry_pc + 8 +
            // (body_len * 4), matching how the interpreter would have
            // advanced pc by the time B itself executes.
            let body_len = body_opcodes.len() as u32;

            for (i, &insn) in body_opcodes.iter().enumerate() {
                let dec = Self::decode_supported_dp(insn).expect("pre-validated");
                // Reject MOV/etc to R15 in compiled path: that would flush
                // the pipeline and we don't model it yet.
                if dec.rd == 15 || (matches!(dec.operand2, Operand2::Reg(15))) {
                    return None;
                }
                let _ = i;
                emit_conditional_instr(&mut builder, gpr_ptr, cpsr_var, dec);
            }

            // Fall through vs branch-taken return path.
            let took_branch_var = builder.declare_var(types::I32);
            let zero = builder.ins().iconst(types::I32, 0);
            builder.def_var(took_branch_var, zero);

            if let Some(br) = tail_is_branch {
                // pc of the B instruction itself, in linear layout from block
                // entry.
                let branch_insn_pc = entry_pc.wrapping_add(body_len.wrapping_mul(4));
                // ARM B semantics: new_pc = (pc_of_B + 8) + sign_extend(imm24) * 4.
                let target = branch_insn_pc
                    .wrapping_add(8)
                    .wrapping_add((br.offset24_signed as i32 * 4) as u32);

                let cond_pass = emit_cond_check(&mut builder, cpsr_var, br.cond);
                let taken = builder.create_block();
                let not_taken = builder.create_block();
                builder.ins().brif(cond_pass, taken, &[], not_taken, &[]);

                builder.switch_to_block(taken);
                builder.seal_block(taken);
                // If BL, write LR = pc_of_B + 4.
                if br.link {
                    let lr_val = builder
                        .ins()
                        .iconst(types::I32, (branch_insn_pc.wrapping_add(4)) as i64);
                    builder.ins().store(
                        MemFlags::trusted(),
                        lr_val,
                        gpr_ptr,
                        Offset32::new(14 * 4),
                    );
                }
                let target_val = builder.ins().iconst(types::I32, target as i64);
                builder.ins().store(MemFlags::trusted(), target_val, pc_out, 0);
                let one = builder.ins().iconst(types::I32, 1);
                builder.def_var(took_branch_var, one);
                builder.ins().jump(not_taken, &[]);

                builder.switch_to_block(not_taken);
                builder.seal_block(not_taken);
            } else {
                // Trailing DP instruction, same as body.
                let dec = Self::decode_supported_dp(tail_insn).expect("pre-validated");
                if dec.rd == 15 || matches!(dec.operand2, Operand2::Reg(15)) {
                    return None;
                }
                emit_conditional_instr(&mut builder, gpr_ptr, cpsr_var, dec);
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder
                .ins()
                .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);

            let ret_val = builder.use_var(took_branch_var);
            builder.ins().return_(&[ret_val]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u32, *mut u32) -> u32,
            >(code)
        })
    }

    /// Classify an ARM opcode as B or BL. Returns None for everything else.
    /// Encoding (ARMv4):
    ///   cond[31:28] | 101L[27:24] | imm24[23:0]
    /// L=0 -> B, L=1 -> BL. imm24 is a signed 24-bit value; the effective
    /// offset is sign_extend(imm24) << 2, added to PC+8.
    fn decode_branch(insn: u32) -> Option<DecodedBranch> {
        let cond_bits = (insn >> 28) & 0xf;
        if cond_bits == 0xF {
            return None;
        }
        // Top 7 bits of the non-cond field must be 0b1010xxx (B) or 0b1011xxx (BL).
        if (insn >> 25) & 0b111 != 0b101 {
            return None;
        }
        let link = ((insn >> 24) & 1) != 0;
        // Sign-extend the 24 bit immediate into i32.
        let raw = insn & 0x00ff_ffff;
        let signed = ((raw as i32) << 8) >> 8; // arithmetic shift to sign-extend
        Some(DecodedBranch {
            cond: ArmCond::from_bits(cond_bits as u8),
            link,
            offset24_signed: signed,
        })
    }

    /// Compile a block that mixes supported data processing shapes with ARM
    /// LDR / STR immediate (word size, pre indexed, no writeback). Calls
    /// into the bus trampolines the compiler was built with.
    ///
    /// Signature:
    ///   extern "C" fn(
    ///     gpr_ptr: *mut u32,
    ///     cpsr_ptr: *mut u32,
    ///     cpu_ctx: *mut u8,   // opaque context passed through to trampolines
    ///   )
    ///
    /// Returns None if the compiler was built without bus trampolines, or if
    /// any opcode isn't a supported DP or mem shape.
    pub fn try_compile_mem_block(
        &mut self,
        opcodes: &[u32],
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u8)> {
        let imports = self.bus_imports?;

        // Classify each instr as DP or Mem. Reject early if anything is
        // unsupported so we don't half emit.
        enum BlockItem {
            Dp(DecodedDp),
            Mem(DecodedMem),
        }
        let mut items = Vec::with_capacity(opcodes.len());
        for &insn in opcodes {
            if let Some(m) = Self::decode_mem_immediate(insn) {
                // PC relative loads aren't emitted yet; reject.
                if m.rn == 15 || m.rd == 15 {
                    return None;
                }
                items.push(BlockItem::Mem(m));
            } else if let Some(d) = Self::decode_supported_dp(insn) {
                if d.rd == 15 || matches!(d.operand2, Operand2::Reg(15)) {
                    return None;
                }
                items.push(BlockItem::Dp(d));
            } else {
                return None;
            }
        }

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // gpr_ptr
        sig.params.push(AbiParam::new(ptr_type)); // cpsr_ptr
        sig.params.push(AbiParam::new(ptr_type)); // cpu_ctx

        self.next_id += 1;
        let name = format!("dynarec_mem_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];
            let cpu_ctx = builder.block_params(entry)[2];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            // Reference imports once in this function.
            let load_idle_32_ref =
                self.module.declare_func_in_func(imports.load_with_idle_32, builder.func);
            let store_32_ref =
                self.module.declare_func_in_func(imports.store_32, builder.func);
            let load_idle_8_ref =
                self.module.declare_func_in_func(imports.load_with_idle_8, builder.func);
            let store_8_ref =
                self.module.declare_func_in_func(imports.store_8, builder.func);

            for item in &items {
                match item {
                    BlockItem::Dp(dec) => {
                        emit_conditional_instr(&mut builder, gpr_ptr, cpsr_var, *dec);
                    }
                    BlockItem::Mem(mem) => {
                        emit_conditional_mem(
                            &mut builder,
                            gpr_ptr,
                            cpsr_var,
                            cpu_ctx,
                            load_idle_32_ref,
                            store_32_ref,
                            load_idle_8_ref,
                            store_8_ref,
                            *mem,
                        );
                    }
                }
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder
                .ins()
                .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<*const u8, extern "C" fn(*mut u32, *mut u32, *mut u8)>(
                code,
            )
        })
    }

    /// Classify an ARM LDR / STR immediate opcode. Encoding:
    ///   cond[31:28] | 01[27:26] | 0[25=I] P U B W L [Rn] [Rd] imm12
    /// Required: pre indexed (P=1), no writeback (W=0), immediate form (I=0).
    /// B=0 is word, B=1 is unsigned byte. U picks add vs subtract.
    fn decode_mem_immediate(insn: u32) -> Option<DecodedMem> {
        let cond_bits = (insn >> 28) & 0xf;
        if cond_bits == 0xF {
            return None;
        }
        // Bits [27:26] must be 01 for load/store single.
        if (insn >> 26) & 0b11 != 0b01 {
            return None;
        }
        // I must be 0 (immediate form, not register). Bit 25.
        if (insn >> 25) & 1 != 0 {
            return None;
        }
        let p = (insn >> 24) & 1 != 0;
        let u = (insn >> 23) & 1 != 0;
        let b = (insn >> 22) & 1 != 0;
        let w = (insn >> 21) & 1 != 0;
        let l = (insn >> 20) & 1 != 0;

        // Pre indexed, no writeback. U and B both accepted now.
        if !p || w {
            return None;
        }

        let rn = ((insn >> 16) & 0xf) as i32;
        let rd = ((insn >> 12) & 0xf) as i32;
        let imm12 = insn & 0xfff;

        Some(DecodedMem {
            cond: ArmCond::from_bits(cond_bits as u8),
            load: l,
            byte: b,
            add: u,
            rd,
            rn,
            offset: imm12,
        })
    }

    /// Compile a block of Thumb 16 bit opcodes. Supports:
    ///   - format 1: LSL/LSR/ASR Rd, Rs, #imm5 (with shifter carry)
    ///   - format 2: ADD/SUB Rd, Rs, Rn/imm3
    ///   - format 3: MOV/CMP/ADD/SUB Rd, #imm8
    ///   - format 4 logical subset: AND/EOR/ORR/BIC/MVN/TST/CMP/CMN Rd, Rs
    ///   - format 5 non-branch: ADD/CMP/MOV Hi registers (no flag updates
    ///     on ADD/MOV, CMP still sets flags). BX deferred to the branch
    ///     block path.
    ///
    ///   001_oo_ddd_iiiiiiii
    ///     oo = 00 MOV, 01 CMP, 10 ADD, 11 SUB
    ///     Rd = ddd
    ///     imm8 = low byte
    ///
    /// All four mnemonics update CPSR.NZCV unconditionally (Thumb has no
    /// per instruction cond field outside IT blocks, which ARMv4 doesn't
    /// have anyway). CMP does not writeback. MOV just sets N and Z; ADD
    /// and SUB update full NZCV. The compiled fn signature matches the ARM
    /// DP one so the caller logic is the same:
    ///
    ///   extern "C" fn(*mut u32 gpr, *mut u32 cpsr)
    ///
    /// Returns None for any unsupported encoding.
    pub fn try_compile_thumb_block(
        &mut self,
        opcodes: &[u16],
    ) -> Option<extern "C" fn(*mut u32, *mut u32)> {
        // Each opcode must classify as one of the supported formats.
        enum ThumbItem {
            F1(DecodedThumb1),
            F2(DecodedThumb2),
            F3(DecodedThumb3),
            F4(DecodedThumb4),
            F5(DecodedThumb5),
        }
        let mut items: Vec<ThumbItem> = Vec::with_capacity(opcodes.len());
        for &op in opcodes {
            // Format 1 has to be tried before format 2 because 00011xx... is
            // format 2 but 000xx... otherwise is format 1. decode_thumb_format1
            // explicitly rejects the format 2 bit pattern.
            if let Some(d) = Self::decode_thumb_format1(op) {
                items.push(ThumbItem::F1(d));
            } else if let Some(d) = Self::decode_thumb_format2(op) {
                items.push(ThumbItem::F2(d));
            } else if let Some(d) = Self::decode_thumb_format3(op) {
                items.push(ThumbItem::F3(d));
            } else if let Some(d) = Self::decode_thumb_format4_logical(op) {
                items.push(ThumbItem::F4(d));
            } else if let Some(d) = Self::decode_thumb_format5_non_branch(op) {
                items.push(ThumbItem::F5(d));
            } else {
                return None;
            }
        }

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));

        self.next_id += 1;
        let name = format!("dynarec_thumb_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        // Block-level pattern pre-pass: if the whole block matches a
        // known synthesized shape (shift-pair sign/zero extend, etc),
        // emit its hand-lowered stencil and return. Stencils are in
        // arm7tdmi::dynarec::patterns.
        let matched_pattern = patterns::try_match_thumb(opcodes);

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            if let Some(pattern) = &matched_pattern {
                patterns::emit_pattern_thumb(
                    &mut builder,
                    gpr_ptr,
                    cpsr_var,
                    pattern,
                );
                let cpsr_final = builder.use_var(cpsr_var);
                builder
                    .ins()
                    .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
                builder.ins().return_(&[]);
                builder.finalize();
            } else {

            // Dead flag-write analysis: if instruction k's flag write is
            // wholly covered by instruction k+1's flag write AND k+1
            // doesn't read any flags (cond=AL), then instruction k's
            // flag update is dead and can be skipped. This matters for
            // chains like [MOV #1, MOV #2, ADD] where the first two
            // MOVs' NZ updates are overwritten by the ADD.
            //
            // Flag write masks per-item:
            //   F3 MOV:        NZ         (preserves CV, so writes_covered_by ⊆ NZ)
            //   F3 CMP/ADD/SUB: NZCV
            //   F4 logical:    NZ         (preserves CV)
            //   F4 CMP/CMN:    NZCV
            //   F1 shift:      NZC        (shifter-carry writes C; V preserved)
            //   F2 ADD/SUB:    NZCV
            //   F5 MOV/ADD:    (none)
            //   F5 CMP:        NZCV
            //
            // Using NZCV bitmask: N=8, Z=4, C=2, V=1.
            fn flag_write_mask(item: &ThumbItem) -> u8 {
                match item {
                    ThumbItem::F3(d) => match d.op {
                        Thumb3Op::Mov => 0b1100,       // N, Z
                        _ => 0b1111,                    // NZCV
                    },
                    ThumbItem::F4(d) => match d.op {
                        Thumb4Op::Tst | Thumb4Op::Cmp | Thumb4Op::Cmn => 0b1111,
                        _ => 0b1100, // logical: N, Z
                    },
                    ThumbItem::F1(_) => 0b1110,         // N, Z, C
                    ThumbItem::F2(_) => 0b1111,
                    ThumbItem::F5(d) => match d.op {
                        Thumb5Op::Cmp => 0b1111,
                        _ => 0,
                    },
                }
            }
            let n = items.len();
            let mut skip: Vec<bool> = vec![false; n];
            // Thumb has implicit AL cond on formats 1-5, so no flag reads
            // between items. Dead-write condition = next_write ⊇ current_write.
            for k in 0..n.saturating_sub(1) {
                let cur = flag_write_mask(&items[k]);
                let next = flag_write_mask(&items[k + 1]);
                if cur != 0 && (cur & !next) == 0 {
                    skip[k] = true;
                }
            }

            for (k, item) in items.iter().enumerate() {
                let skip_fu = skip[k];
                match item {
                    ThumbItem::F1(d) => emit_thumb_format1(&mut builder, gpr_ptr, cpsr_var, *d),
                    ThumbItem::F2(d) => {
                        if skip_fu {
                            emit_thumb_format2_no_flags(&mut builder, gpr_ptr, *d);
                        } else {
                            emit_thumb_format2(&mut builder, gpr_ptr, cpsr_var, *d);
                        }
                    }
                    ThumbItem::F3(d) => {
                        if skip_fu {
                            emit_thumb_format3_no_flags(&mut builder, gpr_ptr, *d);
                        } else {
                            emit_thumb_format3(&mut builder, gpr_ptr, cpsr_var, *d);
                        }
                    }
                    ThumbItem::F4(d) => emit_thumb_format4_logical(&mut builder, gpr_ptr, cpsr_var, *d),
                    ThumbItem::F5(d) => emit_thumb_format5_non_branch(&mut builder, gpr_ptr, cpsr_var, *d),
                }
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder
                .ins()
                .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
            builder.ins().return_(&[]);
            builder.finalize();
            } // end else (pattern match fell through to per-instruction path)
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<*const u8, extern "C" fn(*mut u32, *mut u32)>(code)
        })
    }

    /// Thumb variant that mixes DP shapes with memory ops. Calls into the
    /// bus trampolines registered at compiler construction time, like the
    /// ARM `try_compile_mem_block`. Returns None if the compiler was built
    /// without `new_with_bus`, or if any opcode is not a supported shape.
    pub fn try_compile_thumb_mem_block(
        &mut self,
        opcodes: &[u16],
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u8)> {
        let imports = self.bus_imports?;

        enum MemItem {
            F1(DecodedThumb1),
            F2(DecodedThumb2),
            F3(DecodedThumb3),
            F4(DecodedThumb4),
            F5(DecodedThumb5),
            F9(DecodedThumb9),
            F11(DecodedThumb11),
            F14(DecodedThumb14),
        }
        let mut items = Vec::with_capacity(opcodes.len());
        for &op in opcodes {
            if let Some(d) = Self::decode_thumb_format14_non_pc(op) {
                items.push(MemItem::F14(d));
            } else if let Some(d) = Self::decode_thumb_format11(op) {
                items.push(MemItem::F11(d));
            } else if let Some(d) = Self::decode_thumb_format9(op) {
                items.push(MemItem::F9(d));
            } else if let Some(d) = Self::decode_thumb_format1(op) {
                items.push(MemItem::F1(d));
            } else if let Some(d) = Self::decode_thumb_format2(op) {
                items.push(MemItem::F2(d));
            } else if let Some(d) = Self::decode_thumb_format3(op) {
                items.push(MemItem::F3(d));
            } else if let Some(d) = Self::decode_thumb_format4_logical(op) {
                items.push(MemItem::F4(d));
            } else if let Some(d) = Self::decode_thumb_format5_non_branch(op) {
                items.push(MemItem::F5(d));
            } else {
                return None;
            }
        }

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));

        self.next_id += 1;
        let name = format!("dynarec_thumb_mem_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];
            let cpu_ctx = builder.block_params(entry)[2];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            // load_32: plain no-idle variant used by PUSH/POP codegen
            // (format 14) where scalar charges +1I once at the end of
            // the whole multi-load, not per-register.
            let load_32_ref =
                self.module.declare_func_in_func(imports.load_32, builder.func);
            // load_with_idle_*: used by single-LDR codegen (formats 9, 11,
            // and the ARM mem path) where scalar charges +1I per LDR.
            let load_idle_32_ref =
                self.module.declare_func_in_func(imports.load_with_idle_32, builder.func);
            let load_idle_8_ref =
                self.module.declare_func_in_func(imports.load_with_idle_8, builder.func);
            let store_32_ref =
                self.module.declare_func_in_func(imports.store_32, builder.func);
            let store_8_ref =
                self.module.declare_func_in_func(imports.store_8, builder.func);
            let load_32_seq_ref =
                self.module.declare_func_in_func(imports.load_32_seq, builder.func);
            let store_32_seq_ref =
                self.module.declare_func_in_func(imports.store_32_seq, builder.func);
            let idle_cycle_ref =
                self.module.declare_func_in_func(imports.idle_cycle, builder.func);

            for item in &items {
                match item {
                    MemItem::F1(d) => emit_thumb_format1(&mut builder, gpr_ptr, cpsr_var, *d),
                    MemItem::F2(d) => emit_thumb_format2(&mut builder, gpr_ptr, cpsr_var, *d),
                    MemItem::F3(d) => emit_thumb_format3(&mut builder, gpr_ptr, cpsr_var, *d),
                    MemItem::F4(d) => emit_thumb_format4_logical(&mut builder, gpr_ptr, cpsr_var, *d),
                    MemItem::F5(d) => emit_thumb_format5_non_branch(&mut builder, gpr_ptr, cpsr_var, *d),
                    MemItem::F9(d) => emit_thumb_format9(
                        &mut builder, gpr_ptr, cpu_ctx,
                        load_idle_32_ref, store_32_ref, load_idle_8_ref, store_8_ref,
                        *d,
                    ),
                    MemItem::F11(d) => emit_thumb_format11(
                        &mut builder, gpr_ptr, cpu_ctx,
                        load_idle_32_ref, store_32_ref,
                        *d,
                    ),
                    MemItem::F14(d) => emit_thumb_format14(
                        &mut builder, gpr_ptr, cpu_ctx,
                        load_32_ref, store_32_ref,
                        load_32_seq_ref, store_32_seq_ref,
                        idle_cycle_ref, *d,
                    ),
                }
            }
            // Track if the last emitted instruction is a STORE (any of
            // the F9/F11/F14 store shapes) or a halfword load with
            // NonSeq follow up. Scalar STR returns
            // `CpuAction::AdvancePC(NonSeq)` which sets next_fetch_access
            // to NonSeq for the post-block fetch in the cached-interp
            // loop; mirror that here.
            let last_is_nonseq_advance = matches!(
                items.last(),
                Some(MemItem::F9(d)) if !d.load
            ) || matches!(
                items.last(),
                Some(MemItem::F11(d)) if !d.load
            ) || matches!(
                items.last(),
                Some(MemItem::F14(d)) if d.push
            );
            if last_is_nonseq_advance {
                let nonseq_ref = self
                    .module
                    .declare_func_in_func(imports.set_next_fetch_nonseq, builder.func);
                builder.ins().call(nonseq_ref, &[cpu_ctx]);
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder
                .ins()
                .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u32, *mut u8),
            >(code)
        })
    }

    /// Classify a Thumb format 14 PUSH / POP (non PC variant):
    ///   1011_L_10_R_rrrrrrrr
    ///     L = 0 PUSH, 1 POP
    ///     R = include LR (PUSH) or PC (POP).
    ///     rrrrrrrr = R0..R7 inclusion bitmap.
    /// POP with R=1 (POP{PC}) is a block terminator and gets handled in the
    /// branch block path, so it's rejected here.
    /// Empty register list (including R) is UNPREDICTABLE per spec,
    /// rejected.
    fn decode_thumb_format14_non_pc(op: u16) -> Option<DecodedThumb14> {
        if (op >> 12) & 0xF != 0b1011 {
            return None;
        }
        let load = (op >> 11) & 1 != 0;
        // Bits [10:9] must be 10.
        if (op >> 9) & 0b11 != 0b10 {
            return None;
        }
        let extra = (op >> 8) & 1 != 0;
        let reg_list = (op & 0xff) as u8;

        // Defer POP{PC} to try_compile_thumb_block_with_branch.
        if load && extra {
            return None;
        }
        // Empty effective register list is UNPREDICTABLE.
        if reg_list == 0 && !extra {
            return None;
        }

        Some(DecodedThumb14 {
            push: !load,
            extra_reg: extra,
            reg_list,
        })
    }

    /// Classify a Thumb format 11 SP relative LDR/STR:
    ///   1001_L_ddd_iiiiiiii
    ///     L = 0 STR, 1 LDR. Always word sized. imm8 is scaled by 4.
    /// addr = gpr[13] (SP) + imm8 * 4.
    fn decode_thumb_format11(op: u16) -> Option<DecodedThumb11> {
        if (op >> 12) & 0xF != 0b1001 {
            return None;
        }
        let load = (op >> 11) & 1 != 0;
        let rd = ((op >> 8) & 0b111) as i32;
        let imm8 = (op & 0xff) as u32;
        let offset = imm8 * 4;
        Some(DecodedThumb11 { load, rd, offset })
    }

    /// Classify a Thumb format 9 LDR/STR immediate offset:
    ///   011_B_L_iiiii_sss_ddd
    /// B = 0 word (offset = imm5 * 4), B = 1 byte (offset = imm5).
    /// L = 0 STR, 1 LDR.
    fn decode_thumb_format9(op: u16) -> Option<DecodedThumb9> {
        if (op >> 13) & 0b111 != 0b011 {
            return None;
        }
        let byte = (op >> 12) & 1 != 0;
        let load = (op >> 11) & 1 != 0;
        let imm5 = ((op >> 6) & 0b11111) as u32;
        let rs = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        let offset = if byte { imm5 } else { imm5 * 4 };
        Some(DecodedThumb9 { load, byte, offset, rs, rd })
    }

    /// Thumb format 6: PC-relative LDR.  LDR Rd, [PC, #imm8*4]
    /// Encoding: 0100_1[Rd:3][imm8:8].  Loads a word from
    /// `((pc+4) & !3) + imm8*4` into Rd.  Since the pc of any given
    /// instruction in a cached block is known at codegen time, the
    /// effective address is a constant — emit can fold it and skip
    /// the runtime alignment/add.
    fn decode_thumb_format6(op: u16) -> Option<DecodedThumb6> {
        if (op >> 11) & 0b11111 != 0b01001 {
            return None;
        }
        let rd = ((op >> 8) & 0b111) as i32;
        let imm8 = (op & 0xFF) as u32;
        Some(DecodedThumb6 { rd, imm8 })
    }

    /// Thumb format 13: ADD/SUB SP, #imm7 << 2. Register-only ALU
    /// op, no memory, no flag write. Encoding: `1011_0000_S_imm7`.
    fn decode_thumb_format13(op: u16) -> Option<DecodedThumb13> {
        if (op & 0xFF00) != 0xB000 {
            return None;
        }
        let sub = (op >> 7) & 1 != 0;
        let imm7 = (op & 0x7F) as u32;
        Some(DecodedThumb13 {
            sub,
            offset: imm7 * 4,
        })
    }

    /// Thumb format 8 LDSB: sign-extended byte load Rd = (i8)[Rb+Ro].
    /// Encoding: `0101_10_1_Ro_Rb_Rd` (H=0 sign=1).
    /// Execution: 1N load + 1I, AdvancePC(NonSeq).
    /// Only the LDSB sub-op of F8 is supported here — STRH/LDRH share
    /// the F10 halfword trampolines (which currently drift), and LDSH
    /// requires an ldr_sign_half trampoline we haven't written. LDSB
    /// is self-contained.
    fn decode_thumb_format8_ldsb(op: u16) -> Option<DecodedThumb8Ldsb> {
        if (op >> 12) & 0xF != 0b0101 {
            return None;
        }
        if (op >> 9) & 1 == 0 {
            return None; // F7 (bit 9 = 0)
        }
        // H=bit 11, S=bit 10. LDSB is H=0, S=1 → bits 11-10 = 01.
        if (op >> 10) & 0b11 != 0b01 {
            return None;
        }
        let ro = ((op >> 6) & 0b111) as i32;
        let rb = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        Some(DecodedThumb8Ldsb { ro, rb, rd })
    }

    /// Thumb format 10: LDRH / STRH imm5 offset.
    /// Encoding: `1000_L_imm5_Rs_Rd`. Address = Rs + imm5*2.
    #[allow(dead_code)]
    fn decode_thumb_format10(op: u16) -> Option<DecodedThumb10> {
        if (op >> 12) & 0xF != 0b1000 {
            return None;
        }
        let load = (op >> 11) & 1 != 0;
        let imm5 = ((op >> 6) & 0b11111) as u32;
        let rs = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        Some(DecodedThumb10 { load, rs, rd, offset: imm5 * 2 })
    }

    /// Thumb format 15: LDM / STM IA base-write-back register list.
    /// Encoding: `1100_L_Rb_rlist`. L=1 LDMIA, L=0 STMIA. Address
    /// starts at Rb & !3. Each listed R0..R7 is load/stored in
    /// ascending order, base Rb incremented by 4 per register.
    /// Empty rlist is a special case (loads/stores PC, Rb += 0x40);
    /// the classifier rejects those so the common case stays simple.
    /// STM with Rb in the list + NOT first writes `addr + 4*(count-1)`
    /// to memory for THAT reg (idiosyncratic ARMv4 behavior) — also
    /// rejected from compilation to keep codegen tractable.
    fn decode_thumb_format15(op: u16) -> Option<DecodedThumb15> {
        if (op >> 12) & 0xF != 0b1100 {
            return None;
        }
        let load = (op >> 11) & 1 != 0;
        let rb = ((op >> 8) & 0b111) as i32;
        let rlist = (op & 0xFF) as u8;
        if rlist == 0 {
            return None; // empty-rlist edge case
        }
        // STM with Rb in rlist: 3 ARM sub-cases (first=store addr,
        // later=store addr+4*(count-1)). Reject to keep it simple.
        if !load && (rlist & (1 << rb)) != 0 {
            return None;
        }
        Some(DecodedThumb15 { load, rb, rlist })
    }

    /// Thumb format 7: LDR/STR register offset (word/byte).
    /// Encoding: `0101_L_B_0_Ro_Rb_Rd` — bit 9 must be 0 to
    /// distinguish from F8 (bit 9 = 1, sign-extended).
    /// Address = Rb + Ro. L=1 load, B=1 byte.
    fn decode_thumb_format7(op: u16) -> Option<DecodedThumb7> {
        // F7 and F8 share top-5 0b0101, differ on bit 9.
        if (op >> 12) & 0xF != 0b0101 {
            return None;
        }
        if (op >> 9) & 1 != 0 {
            return None; // F8
        }
        let load = (op >> 11) & 1 != 0;
        let byte = (op >> 10) & 1 != 0;
        let ro = ((op >> 6) & 0b111) as i32;
        let rb = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        Some(DecodedThumb7 { load, byte, ro, rb, rd })
    }

    /// Thumb format 12: load address PC-rel or SP-rel into Rd.
    /// Encoding: `1010_L_Rd_imm8`. L=0 PC, L=1 SP.
    ///   PC case: Rd = ((instr_pc & !2) + 4) + imm8*4 — folded at
    ///   codegen since instr_pc is known.
    ///   SP case: Rd = SP + imm8*4.
    /// No memory access, no flag update, AdvancePC(Seq).
    fn decode_thumb_format12(op: u16) -> Option<DecodedThumb12> {
        if (op >> 12) & 0xF != 0b1010 {
            return None;
        }
        let sp = (op >> 11) & 1 != 0;
        let rd = ((op >> 8) & 0b111) as i32;
        let imm8 = (op & 0xFF) as u32;
        Some(DecodedThumb12 {
            sp,
            rd,
            offset: imm8 * 4,
        })
    }

    /// Thumb variant of `try_compile_block_with_branch`: compiles a block
    /// whose body is supported Thumb shapes plus an optional trailing BX Rs.
    /// BX is a block terminator that writes the target pc (preserving bit 0
    /// as the ARM/Thumb mode signal, per ARM BX convention) into *pc_out
    /// and returns 1. If there's no BX tail the block compiles as usual
    /// and returns 0.
    pub fn try_compile_thumb_block_with_branch(
        &mut self,
        opcodes: &[u16],
        entry_pc: u32,
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u32) -> u32> {
        if opcodes.is_empty() {
            return None;
        }

        let (body_opcodes, tail) = opcodes.split_at(opcodes.len() - 1);
        let tail_insn = tail[0];

        // Classify body: every non-tail opcode must be one of the supported
        // straight-line Thumb shapes. Branching shapes (BX, fmt 16, fmt 18)
        // only allowed in the tail slot.
        enum BodyItem {
            F1(DecodedThumb1),
            F2(DecodedThumb2),
            F3(DecodedThumb3),
            F4(DecodedThumb4),
            F5(DecodedThumb5),
        }
        let mut body: Vec<BodyItem> = Vec::with_capacity(body_opcodes.len());
        for &op in body_opcodes {
            if Self::decode_thumb_bx(op).is_some()
                || Self::decode_thumb_format16(op).is_some()
                || Self::decode_thumb_format18(op).is_some()
            {
                return None;
            }
            if let Some(d) = Self::decode_thumb_format1(op) {
                body.push(BodyItem::F1(d));
            } else if let Some(d) = Self::decode_thumb_format2(op) {
                body.push(BodyItem::F2(d));
            } else if let Some(d) = Self::decode_thumb_format3(op) {
                body.push(BodyItem::F3(d));
            } else if let Some(d) = Self::decode_thumb_format4_logical(op) {
                body.push(BodyItem::F4(d));
            } else if let Some(d) = Self::decode_thumb_format5_non_branch(op) {
                body.push(BodyItem::F5(d));
            } else {
                return None;
            }
        }

        // Tail classification: prefer branch shapes over body shapes since
        // the body also accepts fmt 5 ADD/CMP/MOV which share the 010001
        // prefix. BX / fmt 16 / fmt 18 each have their own distinguishing
        // prefix so there's no overlap to worry about.
        let tail_bx = Self::decode_thumb_bx(tail_insn);
        let tail_pc_branch = Self::decode_thumb_format18(tail_insn)
            .or_else(|| Self::decode_thumb_format16(tail_insn));
        let tail_body = if tail_bx.is_none() && tail_pc_branch.is_none() {
            if let Some(d) = Self::decode_thumb_format1(tail_insn) {
                Some(BodyItem::F1(d))
            } else if let Some(d) = Self::decode_thumb_format2(tail_insn) {
                Some(BodyItem::F2(d))
            } else if let Some(d) = Self::decode_thumb_format3(tail_insn) {
                Some(BodyItem::F3(d))
            } else if let Some(d) = Self::decode_thumb_format4_logical(tail_insn) {
                Some(BodyItem::F4(d))
            } else if let Some(d) = Self::decode_thumb_format5_non_branch(tail_insn) {
                Some(BodyItem::F5(d))
            } else {
                return None;
            }
        } else {
            None
        };

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.returns.push(AbiParam::new(types::I32));

        self.next_id += 1;
        let name = format!("dynarec_thumb_branch_block_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];
            let pc_out = builder.block_params(entry)[2];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            for item in &body {
                match item {
                    BodyItem::F1(d) => emit_thumb_format1(&mut builder, gpr_ptr, cpsr_var, *d),
                    BodyItem::F2(d) => emit_thumb_format2(&mut builder, gpr_ptr, cpsr_var, *d),
                    BodyItem::F3(d) => emit_thumb_format3(&mut builder, gpr_ptr, cpsr_var, *d),
                    BodyItem::F4(d) => emit_thumb_format4_logical(&mut builder, gpr_ptr, cpsr_var, *d),
                    BodyItem::F5(d) => emit_thumb_format5_non_branch(&mut builder, gpr_ptr, cpsr_var, *d),
                }
            }

            let took_branch_var = builder.declare_var(types::I32);
            let zero = builder.ins().iconst(types::I32, 0);
            builder.def_var(took_branch_var, zero);

            if let Some(bx) = tail_bx {
                // target = gpr[bx.rs]. Preserve bit 0 (mode signal).
                let target = builder.ins().load(
                    types::I32,
                    MemFlags::trusted(),
                    gpr_ptr,
                    Offset32::new(bx.rs * 4),
                );
                builder.ins().store(MemFlags::trusted(), target, pc_out, 0);
                let one = builder.ins().iconst(types::I32, 1);
                builder.def_var(took_branch_var, one);
            } else if let Some(br) = tail_pc_branch {
                // Static target at codegen time.
                // pc_of_branch_in_block = entry_pc + 2 * (body_len)
                // current_pc_at_execute = pc_of_branch + 4 (Thumb pipeline)
                // final_target           = current_pc + (offset << 1)
                let body_len = body.len() as u32;
                let branch_pc = entry_pc.wrapping_add(body_len.wrapping_mul(2));
                let target = branch_pc
                    .wrapping_add(4)
                    .wrapping_add((br.offset_signed << 1) as u32)
                    | 1; // stay in Thumb mode
                let cond_pass = emit_cond_check(&mut builder, cpsr_var, br.cond);
                let taken = builder.create_block();
                let not_taken = builder.create_block();
                builder.ins().brif(cond_pass, taken, &[], not_taken, &[]);

                builder.switch_to_block(taken);
                builder.seal_block(taken);
                let t_val = builder.ins().iconst(types::I32, target as i64);
                builder.ins().store(MemFlags::trusted(), t_val, pc_out, 0);
                let one = builder.ins().iconst(types::I32, 1);
                builder.def_var(took_branch_var, one);
                builder.ins().jump(not_taken, &[]);

                builder.switch_to_block(not_taken);
                builder.seal_block(not_taken);
            } else if let Some(item) = tail_body {
                match item {
                    BodyItem::F1(d) => emit_thumb_format1(&mut builder, gpr_ptr, cpsr_var, d),
                    BodyItem::F2(d) => emit_thumb_format2(&mut builder, gpr_ptr, cpsr_var, d),
                    BodyItem::F3(d) => emit_thumb_format3(&mut builder, gpr_ptr, cpsr_var, d),
                    BodyItem::F4(d) => emit_thumb_format4_logical(&mut builder, gpr_ptr, cpsr_var, d),
                    BodyItem::F5(d) => emit_thumb_format5_non_branch(&mut builder, gpr_ptr, cpsr_var, d),
                }
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder
                .ins()
                .store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
            let ret_val = builder.use_var(took_branch_var);
            builder.ins().return_(&[ret_val]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u32, *mut u32) -> u32,
            >(code)
        })
    }

    /// Unified Thumb block compiler. Accepts any mix of supported Thumb
    /// body shapes plus an optional trailing terminator (BX, B, Bcc, or
    /// POP{regs,pc}). Always takes all four params even if a given block
    /// doesn't use all of them, so `step_block` integration has a single
    /// dispatch shape.
    ///
    /// Signature:
    ///   extern "C" fn(
    ///     gpr: *mut u32,
    ///     cpsr: *mut u32,
    ///     pc_out: *mut u32,
    ///     cpu_ctx: *mut u8,
    ///   ) -> u32     // 0 = fall through, 1 = branch taken, pc_out written
    ///
    /// Body shapes (non-terminating):
    ///   format 1/2/3/4-logical/5-nb/9/11/14-non-PC
    /// Tail shapes (terminating, only in last slot):
    ///   BX, format 16 Bcc, format 18 B, POP{regs, pc}
    ///
    /// Requires bus trampolines (new_with_bus). Returns None for any
    /// unsupported encoding or if the compiler has no bus.
    ///
    /// `chain_slot`, when `Some`, is a stable-address slot whose
    /// 64-bit payload is baked in to the compiled code as the chain
    /// target (a `CompiledThumbFn` pointer). The caller is expected
    /// to (1) allocate the slot before calling this function, (2)
    /// keep it alive for the lifetime of the returned compiled fn,
    /// and (3) write the successor block's compiled fn pointer into
    /// the slot when the block cache links them. When the slot is
    /// non-null at runtime AND the chain-abort check returns 0, the
    /// compiled epilogue tail-calls the slot value instead of
    /// returning to the outer dispatcher — the link-time block
    /// chaining hot path. `None` disables chain emission (pure
    /// dispatcher-return codegen, equivalent to the pre-chaining
    /// behavior).
    ///
    /// Chain emission is only meaningful for `Tail::Body` tails
    /// (fall-through blocks). `Tail::Bx/PcBranch/PopPc` end a block
    /// by definition — they already set `pc_out` + `taken=1` and
    /// return, so chain emission is a no-op for them even when a
    /// slot is supplied.
    pub fn try_compile_thumb_mem_block_with_branch(
        &mut self,
        opcodes: &[u16],
        entry_pc: u32,
        chain_slot: Option<&crate::cache::ChainSlot>,
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32> {
        if opcodes.is_empty() {
            return None;
        }
        let imports = self.bus_imports?;
        let chain_slot_addr: Option<i64> =
            chain_slot.map(|s| s as *const crate::cache::ChainSlot as i64);

        enum Body {
            F1(DecodedThumb1),
            F2(DecodedThumb2),
            F3(DecodedThumb3),
            F4(DecodedThumb4),
            F5(DecodedThumb5),
            F6(DecodedThumb6),
            F7(DecodedThumb7),
            F8Ldsb(DecodedThumb8Ldsb),
            F9(DecodedThumb9),
            #[allow(dead_code)]
            F10(DecodedThumb10),
            F11(DecodedThumb11),
            F12(DecodedThumb12),
            F13(DecodedThumb13),
            F14(DecodedThumb14),
            F15(DecodedThumb15),
        }
        enum Tail {
            Bx(DecodedThumbBx),
            PcBranch(DecodedThumbPcBranch),
            PopPc(DecodedThumb14),
            Body(Body),
        }

        /// True for body items whose scalar handler returns
        /// `CpuAction::AdvancePC(NonSeq)`. The dynarec needs to charge
        /// `(n - s)` extra cycles after each such item so the next
        /// fetch matches scalar's NonSeq access. For format 14 this is
        /// BOTH push AND pop (scalar exec_thumb_push_pop initializes
        /// `result = CpuAction::AdvancePC(NonSeq)` before the POP/PUSH
        /// branches). For format 9/11 only the STORE path returns
        /// NonSeq; the LDR path returns Seq.
        fn body_item_is_nonseq_advance(item: &Body) -> bool {
            match item {
                Body::F7(d) => !d.load,
                // F8 always NonSeq advance (both loads + stores per
                // scalar exec_thumb_ldr_str_shb's return value).
                Body::F8Ldsb(_) => true,
                Body::F9(d) => !d.load,
                // F10 STRH returns NonSeq; LDRH returns Seq.
                Body::F10(d) => !d.load,
                Body::F11(d) => !d.load,
                // F15 LDM: AdvancePC(NonSeq). STM: AdvancePC(NonSeq). Always NonSeq.
                Body::F15(_) => true,
                // F14 PUSH and POP BOTH return AdvancePC(NonSeq) in
                // scalar (exec_thumb_push_pop initializes `result =
                // CpuAction::AdvancePC(NonSeq)` before the direction
                // branches, and only POP{PC} overrides with
                // PipelineFlushed). The previous `d.push` gate missed
                // the POP case, undercounting the post-block fetch
                // cycles on blocks ending with POP — one of several
                // drifts surfaced by the a7fb782 BX+POP lift.
                Body::F14(_) => true,
                _ => false,
            }
        }

        fn classify_body(op: u16) -> Option<Body> {
            // F13 must come BEFORE F14 because both have top4==0b1011
            // but F13 uses middle=0b0000 while F14 uses middle=0b010x
            // / 0b110x. Try F14 first to narrow then F13.
            if let Some(d) = DynarecCompiler::decode_thumb_format15(op) {
                Some(Body::F15(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format14_non_pc(op) {
                Some(Body::F14(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format13(op) {
                Some(Body::F13(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format12(op) {
                Some(Body::F12(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format11(op) {
                Some(Body::F11(d))
            // F10 codegen + trampoline are plumbed but classifier
            // intentionally SKIPS format10 until the 10270-cycle drift
            // on real-SDL pokeemerald is root-caused. Pre-existing
            // code paths already exercise `store_16` /
            // `load_with_idle_16` for the test stubs, so the trampoline
            // wiring is kept live.
            } else if let Some(d) = DynarecCompiler::decode_thumb_format9(op) {
                Some(Body::F9(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format7(op) {
                Some(Body::F7(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format8_ldsb(op) {
                Some(Body::F8Ldsb(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format6(op) {
                Some(Body::F6(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format1(op) {
                Some(Body::F1(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format2(op) {
                Some(Body::F2(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format3(op) {
                Some(Body::F3(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format4_logical(op) {
                Some(Body::F4(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format5_non_branch(op) {
                Some(Body::F5(d))
            } else {
                None
            }
        }
        fn classify_tail(op: u16) -> Option<Tail> {
            if let Some(d) = DynarecCompiler::decode_thumb_bx(op) {
                Some(Tail::Bx(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format18(op) {
                Some(Tail::PcBranch(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_format16(op) {
                Some(Tail::PcBranch(d))
            } else if let Some(d) = DynarecCompiler::decode_thumb_pop_pc(op) {
                Some(Tail::PopPc(d))
            } else {
                classify_body(op).map(Tail::Body)
            }
        }

        let (body_opcodes, tail_slot) = opcodes.split_at(opcodes.len() - 1);
        let mut body: Vec<Body> = Vec::with_capacity(body_opcodes.len());
        for &op in body_opcodes {
            // Terminators not allowed in body.
            if Self::decode_thumb_bx(op).is_some()
                || Self::decode_thumb_format16(op).is_some()
                || Self::decode_thumb_format18(op).is_some()
                || Self::decode_thumb_pop_pc(op).is_some()
            {
                return None;
            }
            body.push(classify_body(op)?);
        }
        let tail = classify_tail(tail_slot[0])?;

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // gpr
        sig.params.push(AbiParam::new(ptr_type)); // cpsr
        sig.params.push(AbiParam::new(ptr_type)); // pc_out
        sig.params.push(AbiParam::new(ptr_type)); // cpu_ctx
        sig.returns.push(AbiParam::new(types::I32));

        self.next_id += 1;
        let name = format!("dynarec_thumb_unified_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        // Keep a copy for `import_signature` so `call_indirect` can
        // target a slot-loaded function pointer with the same ABI as
        // this compiled block. Needed only when chain emission is
        // enabled (`chain_slot_addr` is `Some`).
        let self_sig_template = sig.clone();
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpsr_ptr = builder.block_params(entry)[1];
            let pc_out = builder.block_params(entry)[2];
            let cpu_ctx = builder.block_params(entry)[3];

            let cpsr_var = builder.declare_var(types::I32);
            let cpsr_initial =
                builder.ins().load(types::I32, MemFlags::trusted(), cpsr_ptr, 0);
            builder.def_var(cpsr_var, cpsr_initial);

            // load_32: plain no-idle variant for PUSH/POP (format 14).
            let load_32_ref =
                self.module.declare_func_in_func(imports.load_32, builder.func);
            // load_with_idle_*: per-LDR +1I variants for formats 9/11.
            let load_idle_32_ref =
                self.module.declare_func_in_func(imports.load_with_idle_32, builder.func);
            let load_idle_8_ref =
                self.module.declare_func_in_func(imports.load_with_idle_8, builder.func);
            let store_32_ref =
                self.module.declare_func_in_func(imports.store_32, builder.func);
            let store_8_ref =
                self.module.declare_func_in_func(imports.store_8, builder.func);
            // Sequential-access variants for 2nd+ slot of PUSH/POP.
            // Scalar pays 1N + (N-1)S cycles over the whole multi-
            // access; using all-NonSeq compiled-side caused per-POP
            // drift that accumulated into hash divergence on real
            // SDL (measured 2026-04-24).
            let load_32_seq_ref =
                self.module.declare_func_in_func(imports.load_32_seq, builder.func);
            let store_32_seq_ref =
                self.module.declare_func_in_func(imports.store_32_seq, builder.func);
            // 16-bit halfword (F10) trampolines.
            let store_16_ref =
                self.module.declare_func_in_func(imports.store_16, builder.func);
            let load_idle_16_ref =
                self.module.declare_func_in_func(imports.load_with_idle_16, builder.func);
            let fetch_n_ref =
                self.module.declare_func_in_func(imports.thumb_fetch_n, builder.func);
            // pay_extra_nonseq: charges (n - s) cycles for the next fetch
            // after each in-body STORE (mirrors scalar STR's
            // `CpuAction::AdvancePC(NonSeq)`). Address-invariant within a
            // block since all fetches share the same region/page, so we
            // can pass `entry_pc` for every call.
            let pay_extra_nonseq_ref = self
                .module
                .declare_func_in_func(imports.pay_thumb_fetch_extra_nonseq, builder.func);
            // +1I idle helper for POP / POP{PC} end-of-load pad.
            let idle_cycle_ref = self
                .module
                .declare_func_in_func(imports.idle_cycle, builder.func);
            let entry_pc_val = builder.ins().iconst(types::I32, entry_pc as i64);

            // Pay fetch cycles for the whole block up front and let the
            // trampoline stage pipeline[0/1] + pc + next_fetch_access so
            // the register-only block body can run without any further
            // bus access on the instruction side. first_fetch_pc is
            // block_start_addr + 4 per the Thumb pipeline-head convention;
            // entry_pc in this compile API is block_start_addr.
            let total_count = (body.len() as u32) + 1; // body + tail
            let first_fetch_pc_val = builder
                .ins()
                .iconst(types::I32, (entry_pc.wrapping_add(4)) as i64);
            let count_val = builder.ins().iconst(types::I32, total_count as i64);
            builder.ins().call(fetch_n_ref, &[cpu_ctx, first_fetch_pc_val, count_val]);

            // NZCV bitmask: N=8, Z=4, C=2, V=1. Mirrors the table
            // in the `try_compile_thumb_block` dead-flag pass.
            fn flag_write_mask_body(item: &Body) -> u8 {
                match item {
                    Body::F1(_) => 0b1110, // NZC (shifter carry; V preserved)
                    Body::F2(_) => 0b1111, // ADD/SUB reg: NZCV
                    Body::F3(d) => match d.op {
                        Thumb3Op::Mov => 0b1100, // NZ; preserves CV
                        _ => 0b1111,
                    },
                    Body::F4(d) => match d.op {
                        Thumb4Op::Tst | Thumb4Op::Cmp | Thumb4Op::Cmn => 0b1111,
                        _ => 0b1100, // logical ops: NZ
                    },
                    Body::F5(d) => match d.op {
                        Thumb5Op::Cmp => 0b1111,
                        _ => 0, // MOV/ADD high-reg don't write flags
                    },
                    // Memory ops don't write flags.
                    Body::F6(_) | Body::F7(_) | Body::F8Ldsb(_) | Body::F9(_) | Body::F10(_) | Body::F11(_) | Body::F12(_) | Body::F13(_) | Body::F14(_) | Body::F15(_) => 0,
                }
            }
            // Dead-flag-write pass. Thumb data-proc instrs always
            // update flags (there's no optional S-bit); the cost of
            // emitting the full NZCV computation is ~20 host
            // instructions per data-proc on x86_64 (per the ASM
            // dump). If body[k]'s flag writes are all overwritten by
            // body[k+1] before anything reads them, the write is
            // dead and we can emit the _no_flags variant (which
            // just skips the flag math). AL-cond Thumb has no flag
            // reads between items, so the analysis is a simple
            // next-writes-superset check per position. Only F2 and
            // F3 have _no_flags emitters today; other shapes fall
            // back to full emission even when dead.
            let body_len = body.len();
            let mut skip_flags: Vec<bool> = vec![false; body_len];
            if body_len > 0 {
                // Tail is always Tail::Body today (branch-terminator
                // filter in the cache rejects everything else). If
                // that filter is lifted, the tail's flag read/write
                // analysis needs extending.
                let tail_mask = match &tail {
                    Tail::Body(b) => flag_write_mask_body(b),
                    Tail::Bx(_) | Tail::PopPc(_) => 0, // don't write NZCV
                    Tail::PcBranch(_) => 0, // B/Bcc don't write flags
                };
                let tail_reads_flags = match &tail {
                    Tail::PcBranch(br) => br.cond != ArmCond::Al,
                    _ => false,
                };
                for k in 0..body_len {
                    let cur = flag_write_mask_body(&body[k]);
                    if cur == 0 {
                        continue; // mem op; nothing to skip
                    }
                    let next = if k + 1 < body_len {
                        flag_write_mask_body(&body[k + 1])
                    } else if tail_reads_flags {
                        0 // live — Bcc will read them
                    } else {
                        tail_mask
                    };
                    if (cur & !next) == 0 {
                        skip_flags[k] = true;
                    }
                }
            }

            let emit_body = |builder: &mut FunctionBuilder, item: &Body, skip_flag_write: bool, instr_pc: u32| {
                match item {
                    Body::F1(d) => emit_thumb_format1(builder, gpr_ptr, cpsr_var, *d),
                    Body::F2(d) => {
                        if skip_flag_write {
                            emit_thumb_format2_no_flags(builder, gpr_ptr, *d);
                        } else {
                            emit_thumb_format2(builder, gpr_ptr, cpsr_var, *d);
                        }
                    }
                    Body::F3(d) => {
                        if skip_flag_write {
                            emit_thumb_format3_no_flags(builder, gpr_ptr, *d);
                        } else {
                            emit_thumb_format3(builder, gpr_ptr, cpsr_var, *d);
                        }
                    }
                    Body::F4(d) => emit_thumb_format4_logical(builder, gpr_ptr, cpsr_var, *d),
                    Body::F5(d) => emit_thumb_format5_non_branch(builder, gpr_ptr, cpsr_var, *d),
                    Body::F6(d) => emit_thumb_format6(
                        builder, gpr_ptr, cpu_ctx, load_idle_32_ref, *d, instr_pc,
                    ),
                    Body::F7(d) => emit_thumb_format7(
                        builder, gpr_ptr, cpu_ctx,
                        load_idle_32_ref, store_32_ref, load_idle_8_ref, store_8_ref, *d,
                    ),
                    Body::F8Ldsb(d) => emit_thumb_format8_ldsb(
                        builder, gpr_ptr, cpu_ctx, load_idle_8_ref, *d,
                    ),
                    Body::F9(d) => emit_thumb_format9(
                        builder, gpr_ptr, cpu_ctx,
                        load_idle_32_ref, store_32_ref, load_idle_8_ref, store_8_ref, *d,
                    ),
                    Body::F10(d) => emit_thumb_format10(
                        builder, gpr_ptr, cpu_ctx,
                        load_idle_16_ref, store_16_ref, *d,
                    ),
                    Body::F11(d) => emit_thumb_format11(
                        builder, gpr_ptr, cpu_ctx,
                        load_idle_32_ref, store_32_ref, *d,
                    ),
                    Body::F12(d) => emit_thumb_format12(builder, gpr_ptr, *d, instr_pc),
                    Body::F13(d) => emit_thumb_format13(builder, gpr_ptr, *d),
                    Body::F14(d) => emit_thumb_format14(
                        builder, gpr_ptr, cpu_ctx,
                        load_32_ref, store_32_ref,
                        load_32_seq_ref, store_32_seq_ref,
                        idle_cycle_ref, *d,
                    ),
                    Body::F15(d) => emit_thumb_format15(
                        builder, gpr_ptr, cpu_ctx,
                        load_32_ref, store_32_ref,
                        load_32_seq_ref, store_32_seq_ref,
                        idle_cycle_ref, *d,
                    ),
                }
            };

            // Mid-block abort-check setup. Declared up-front so every
            // abort point in the body loop can share the same FuncRefs.
            let abort_chain_ref = self
                .module
                .declare_func_in_func(imports.chain_abort_check, builder.func);
            let abort_mid_ref = self
                .module
                .declare_func_in_func(imports.abort_mid_block, builder.func);

            for (k, item) in body.iter().enumerate() {
                let instr_pc = entry_pc.wrapping_add((2 * k) as u32);
                emit_body(&mut builder, item, skip_flags[k], instr_pc);
                // Mid-block abort check — match scalar's
                // `instr_idx != 0 && instr_idx & 1 == 1` cadence:
                // scalar checks AT THE START of iters 1, 3, 5, ...
                // which is AFTER iters 0, 2, 4, ... finish. Mirror
                // by checking after body[k] at even k. Skip if
                // pipeline[1] at the abort point would be past the
                // block (k+2 >= opcodes.len()) — we'd need a bus
                // fetch to reconstruct it. Minor drift residue at
                // the last abort-eligible position; catches the
                // bulk of the inter-event drift.
                if k % 2 == 0 && k + 2 < opcodes.len() {
                    let abort_call = builder.ins().call(abort_chain_ref, &[cpu_ctx]);
                    let abort = builder.inst_results(abort_call)[0];
                    let abort_nz = builder.ins().icmp_imm(IntCC::NotEqual, abort, 0);
                    let do_abort_blk = builder.create_block();
                    let cont_blk = builder.create_block();
                    builder.ins().brif(abort_nz, do_abort_blk, &[], cont_blk, &[]);

                    builder.switch_to_block(do_abort_blk);
                    builder.seal_block(do_abort_blk);
                    // Abort point: pc points at body[k+1]'s pipeline-head.
                    let abort_pc = entry_pc.wrapping_add((2 * (k + 1)) as u32).wrapping_add(4);
                    let pipe0 = opcodes[k + 1] as u32;
                    let pipe1 = opcodes[k + 2] as u32;
                    let pc_val = builder.ins().iconst(types::I32, abort_pc as i64);
                    let p0 = builder.ins().iconst(types::I32, pipe0 as i64);
                    let p1 = builder.ins().iconst(types::I32, pipe1 as i64);
                    builder
                        .ins()
                        .call(abort_mid_ref, &[cpu_ctx, pc_val, p0, p1]);
                    // Flush cpsr_var, set abort-signal return (bit 1 = 2).
                    let cpsr_cur = builder.use_var(cpsr_var);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), cpsr_cur, cpsr_ptr, 0);
                    let abort_ret = builder.ins().iconst(types::I32, 2);
                    builder.ins().return_(&[abort_ret]);

                    builder.switch_to_block(cont_blk);
                    builder.seal_block(cont_blk);
                }
                // Compensate for the under-counted next fetch when this
                // body item is a STORE: scalar would have charged NonSeq
                // for the fetch that follows, but `thumb_fetch_n` paid
                // Seq up front. One trampoline call per intermediate
                // store covers both:
                //   - in-body STORE → next body item's fetch
                //   - last body STORE → tail/branch fetch
                if body_item_is_nonseq_advance(item) {
                    builder
                        .ins()
                        .call(pay_extra_nonseq_ref, &[cpu_ctx, entry_pc_val]);
                }
            }

            let took_var = builder.declare_var(types::I32);
            let zero = builder.ins().iconst(types::I32, 0);
            builder.def_var(took_var, zero);

            match &tail {
                Tail::Body(b) => {
                    // Tail body is the LAST flag-writer in the block;
                    // the caller (dispatcher, next block's cpsr load)
                    // sees whatever flags it writes, so we never skip
                    // its flag update.
                    let tail_instr_pc = entry_pc.wrapping_add((2 * body.len()) as u32);
                    emit_body(&mut builder, b, false, tail_instr_pc);
                    // Mirror scalar's `CpuAction::AdvancePC(NonSeq)` for the
                    // post-block fetch: when the tail body item is a STORE
                    // (STR/STRB/PUSH), scalar STR returns NonSeq, so the
                    // NEXT block's first fetch should be charged NonSeq.
                    // thumb_fetch_n set `next_fetch_access = Seq` at block
                    // entry for this block's fetches, but the POST-block
                    // fetch is paid by the cached-interp loop out of
                    // `cpu.next_fetch_access`, so we need to flip it to
                    // NonSeq here if the last instruction was a store.
                    let tail_is_nonseq = body_item_is_nonseq_advance(b);
                    if tail_is_nonseq {
                        let nonseq_ref = self
                            .module
                            .declare_func_in_func(imports.set_next_fetch_nonseq, builder.func);
                        builder.ins().call(nonseq_ref, &[cpu_ctx]);
                    }
                    // Link-time block-chaining check. When a `chain_slot`
                    // was supplied at compile time, emit a load of the
                    // slot's 64-bit payload. If non-null AND the
                    // chain-abort check returns 0, tail-call the slot
                    // value (a `CompiledThumbFn`) with the same 4-arg
                    // signature instead of returning to the outer
                    // dispatcher. Null-slot or abort-nonzero falls
                    // through to the normal `return took_var` epilogue.
                    //
                    // Correctness: the abort check mirrors what scalar
                    // `replay_cached_block` does every other instruction
                    // (RAM dirty / IRQ / DMA / halt / scheduler event).
                    // Chain emission runs only for fall-through `Tail::Body`
                    // blocks, which cannot flip the ARM/Thumb state in
                    // the body, so we don't need an extra state-flip
                    // guard here.
                    if let Some(slot_addr) = chain_slot_addr {
                        let chain_abort_ref = self
                            .module
                            .declare_func_in_func(imports.chain_abort_check, builder.func);
                        let self_sig_ref =
                            builder.import_signature(self_sig_template.clone());

                        let slot_addr_val = builder
                            .ins()
                            .iconst(types::I64, slot_addr);
                        let chain_fn_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            slot_addr_val,
                            0,
                        );

                        let chain_try_blk = builder.create_block();
                        let chain_call_blk = builder.create_block();
                        let chain_merge_blk = builder.create_block();

                        let is_nonnull = builder
                            .ins()
                            .icmp_imm(IntCC::NotEqual, chain_fn_ptr, 0);
                        builder.ins().brif(
                            is_nonnull,
                            chain_try_blk,
                            &[],
                            chain_merge_blk,
                            &[],
                        );

                        builder.switch_to_block(chain_try_blk);
                        builder.seal_block(chain_try_blk);
                        let abort_call =
                            builder.ins().call(chain_abort_ref, &[cpu_ctx]);
                        let abort = builder.inst_results(abort_call)[0];
                        let abort_nz = builder
                            .ins()
                            .icmp_imm(IntCC::NotEqual, abort, 0);
                        builder.ins().brif(
                            abort_nz,
                            chain_merge_blk,
                            &[],
                            chain_call_blk,
                            &[],
                        );

                        builder.switch_to_block(chain_call_blk);
                        builder.seal_block(chain_call_blk);
                        // Flush our cpsr_var to *cpsr_ptr before
                        // handing off — the tail-called block reads
                        // cpsr fresh from *cpsr_ptr at its own entry.
                        let cpsr_cur = builder.use_var(cpsr_var);
                        builder.ins().store(
                            MemFlags::trusted(),
                            cpsr_cur,
                            cpsr_ptr,
                            0,
                        );
                        let indirect_call = builder.ins().call_indirect(
                            self_sig_ref,
                            chain_fn_ptr,
                            &[gpr_ptr, cpsr_ptr, pc_out, cpu_ctx],
                        );
                        let chained_ret = builder.inst_results(indirect_call)[0];
                        builder.ins().return_(&[chained_ret]);

                        builder.switch_to_block(chain_merge_blk);
                        builder.seal_block(chain_merge_blk);
                    }
                }
                Tail::Bx(bx) => {
                    let target = builder.ins().load(
                        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(bx.rs * 4),
                    );
                    builder.ins().store(MemFlags::trusted(), target, pc_out, 0);
                    let one = builder.ins().iconst(types::I32, 1);
                    builder.def_var(took_var, one);
                }
                Tail::PcBranch(br) => {
                    let body_len = body.len() as u32;
                    let branch_pc = entry_pc.wrapping_add(body_len.wrapping_mul(2));
                    let target = branch_pc
                        .wrapping_add(4)
                        .wrapping_add((br.offset_signed << 1) as u32)
                        | 1;
                    let cond_pass = emit_cond_check(&mut builder, cpsr_var, br.cond);
                    let taken_blk = builder.create_block();
                    let fallthrough_blk = builder.create_block();
                    let merge_blk = builder.create_block();
                    builder.ins().brif(cond_pass, taken_blk, &[], fallthrough_blk, &[]);

                    builder.switch_to_block(taken_blk);
                    builder.seal_block(taken_blk);
                    let t_val = builder.ins().iconst(types::I32, target as i64);
                    builder.ins().store(MemFlags::trusted(), t_val, pc_out, 0);
                    let one = builder.ins().iconst(types::I32, 1);
                    builder.def_var(took_var, one);
                    builder.ins().jump(merge_blk, &[]);

                    // Fall-through (cond failed): emit chain check on
                    // the same `chain_slot` (fall-through target key)
                    // that Tail::Body uses. Equivalent semantics —
                    // scalar would advance past the Bcc and execute
                    // the next sequential block; if that block is
                    // compiled we tail-call it instead of returning
                    // to the dispatcher.
                    builder.switch_to_block(fallthrough_blk);
                    builder.seal_block(fallthrough_blk);
                    if let Some(slot_addr) = chain_slot_addr {
                        let chain_abort_ref = self
                            .module
                            .declare_func_in_func(imports.chain_abort_check, builder.func);
                        let self_sig_ref =
                            builder.import_signature(self_sig_template.clone());
                        let slot_addr_val =
                            builder.ins().iconst(types::I64, slot_addr);
                        let chain_fn_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            slot_addr_val,
                            0,
                        );
                        let chain_try_blk = builder.create_block();
                        let chain_call_blk = builder.create_block();
                        let is_nonnull = builder
                            .ins()
                            .icmp_imm(IntCC::NotEqual, chain_fn_ptr, 0);
                        builder.ins().brif(
                            is_nonnull,
                            chain_try_blk,
                            &[],
                            merge_blk,
                            &[],
                        );
                        builder.switch_to_block(chain_try_blk);
                        builder.seal_block(chain_try_blk);
                        let abort_call =
                            builder.ins().call(chain_abort_ref, &[cpu_ctx]);
                        let abort = builder.inst_results(abort_call)[0];
                        let abort_nz = builder
                            .ins()
                            .icmp_imm(IntCC::NotEqual, abort, 0);
                        builder.ins().brif(
                            abort_nz,
                            merge_blk,
                            &[],
                            chain_call_blk,
                            &[],
                        );
                        builder.switch_to_block(chain_call_blk);
                        builder.seal_block(chain_call_blk);
                        let cpsr_cur = builder.use_var(cpsr_var);
                        builder.ins().store(
                            MemFlags::trusted(),
                            cpsr_cur,
                            cpsr_ptr,
                            0,
                        );
                        let indirect_call = builder.ins().call_indirect(
                            self_sig_ref,
                            chain_fn_ptr,
                            &[gpr_ptr, cpsr_ptr, pc_out, cpu_ctx],
                        );
                        let chained_ret = builder.inst_results(indirect_call)[0];
                        builder.ins().return_(&[chained_ret]);
                    } else {
                        builder.ins().jump(merge_blk, &[]);
                    }

                    builder.switch_to_block(merge_blk);
                    builder.seal_block(merge_blk);
                }
                Tail::PopPc(dec) => {
                    let sp = builder.ins().load(
                        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(13 * 4),
                    );
                    // Scalar POP{PC}: 1 NonSeq access for the FIRST
                    // load (lowest register or PC if rlist empty),
                    // Seq for every access after. Mirror that split
                    // so cycle accounting is drift-free.
                    let mut byte_offset: i64 = 0;
                    let mut access_count = 0u32;
                    for i in 0..8 {
                        if dec.reg_list & (1 << i) != 0 {
                            let addr = builder.ins().iadd_imm(sp, byte_offset);
                            let ref_fn = if access_count == 0 { load_32_ref } else { load_32_seq_ref };
                            let call = builder.ins().call(ref_fn, &[cpu_ctx, addr]);
                            let v = builder.inst_results(call)[0];
                            builder.ins().store(
                                MemFlags::trusted(), v, gpr_ptr, Offset32::new(i * 4),
                            );
                            byte_offset += 4;
                            access_count += 1;
                        }
                    }
                    let pc_addr = builder.ins().iadd_imm(sp, byte_offset);
                    let pc_ref_fn = if access_count == 0 { load_32_ref } else { load_32_seq_ref };
                    let pc_call = builder.ins().call(pc_ref_fn, &[cpu_ctx, pc_addr]);
                    let pc_val = builder.inst_results(pc_call)[0];
                    // Scalar Thumb POP{PC} stays in Thumb regardless of
                    // the popped value's bit 0 (exec_thumb_push_pop only
                    // does `self.pc &= !1`; never switches to ARM via
                    // branch_exchange). Force bit 0 = 1 so the caller's
                    // thumb_bit check correctly lands in Thumb mode.
                    let pc_val_thumb = builder.ins().bor_imm(pc_val, 1);
                    builder.ins().store(MemFlags::trusted(), pc_val_thumb, pc_out, 0);
                    byte_offset += 4;
                    let new_sp = builder.ins().iadd_imm(sp, byte_offset);
                    builder.ins().store(
                        MemFlags::trusted(), new_sp, gpr_ptr, Offset32::new(13 * 4),
                    );
                    // Scalar POP{PC} adds +1I at the end of the load
                    // sequence (see exec_thumb_push_pop "// Idle 1 cycle").
                    // Mirror for parity.
                    builder.ins().call(idle_cycle_ref, &[cpu_ctx]);
                    let one = builder.ins().iconst(types::I32, 1);
                    builder.def_var(took_var, one);
                }
            }

            let cpsr_final = builder.use_var(cpsr_var);
            builder.ins().store(MemFlags::trusted(), cpsr_final, cpsr_ptr, 0);
            let ret = builder.use_var(took_var);
            builder.ins().return_(&[ret]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32,
            >(code)
        })
    }

    /// Compile a standalone POP {regs, pc} as a block terminator. Pops
    /// every listed register in R0..R7 order from SP, SP+4, ... then
    /// pops PC from the next word and writes it (bit 0 preserved for the
    /// ARM/Thumb mode signal) into *pc_out. Updates SP by 4 * total
    /// count. Always returns 1 (branch always taken, POP{PC} is
    /// unconditional).
    ///
    /// Signature:
    ///   extern "C" fn(
    ///     gpr: *mut u32,
    ///     cpsr: *mut u32,
    ///     pc_out: *mut u32,
    ///     cpu_ctx: *mut u8,
    ///   ) -> u32
    ///
    /// Returns None if the compiler was built without bus trampolines or
    /// the opcode isn't a POP with R=1.
    pub fn try_compile_thumb_pop_pc(
        &mut self,
        opcode: u16,
    ) -> Option<extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32> {
        let dec = Self::decode_thumb_pop_pc(opcode)?;
        let imports = self.bus_imports?;

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // gpr
        sig.params.push(AbiParam::new(ptr_type)); // cpsr (unused but keep shape)
        sig.params.push(AbiParam::new(ptr_type)); // pc_out
        sig.params.push(AbiParam::new(ptr_type)); // cpu_ctx
        sig.returns.push(AbiParam::new(types::I32));

        self.next_id += 1;
        let name = format!("dynarec_thumb_pop_pc_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let pc_out = builder.block_params(entry)[2];
            let cpu_ctx = builder.block_params(entry)[3];

            let load_32_ref =
                self.module.declare_func_in_func(imports.load_32, builder.func);

            let sp = builder.ins().load(
                types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(13 * 4),
            );

            // Pop each R0..R7 in the list, low-to-high.
            let mut byte_offset: i64 = 0;
            for i in 0..8 {
                if dec.reg_list & (1 << i) != 0 {
                    let addr = builder.ins().iadd_imm(sp, byte_offset);
                    let call = builder.ins().call(load_32_ref, &[cpu_ctx, addr]);
                    let v = builder.inst_results(call)[0];
                    builder.ins().store(
                        MemFlags::trusted(), v, gpr_ptr, Offset32::new(i * 4),
                    );
                    byte_offset += 4;
                }
            }
            // Pop PC from the next slot.
            let pc_addr = builder.ins().iadd_imm(sp, byte_offset);
            let pc_call = builder.ins().call(load_32_ref, &[cpu_ctx, pc_addr]);
            let pc_val = builder.inst_results(pc_call)[0];
            builder.ins().store(MemFlags::trusted(), pc_val, pc_out, 0);
            byte_offset += 4;

            // Update SP.
            let new_sp = builder.ins().iadd_imm(sp, byte_offset);
            builder.ins().store(
                MemFlags::trusted(), new_sp, gpr_ptr, Offset32::new(13 * 4),
            );

            let one = builder.ins().iconst(types::I32, 1);
            builder.ins().return_(&[one]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32,
            >(code)
        })
    }

    /// Classify a POP {regs, pc} (format 14 with L=1, R=1). This is a
    /// block terminator: the popped PC value becomes the new program
    /// counter, so control flow leaves the compiled block.
    fn decode_thumb_pop_pc(op: u16) -> Option<DecodedThumb14> {
        if (op >> 12) & 0xF != 0b1011 {
            return None;
        }
        // L must be 1 (POP), R must be 1 (include PC), bits 10:9 = 10.
        if (op >> 9) & 0b111 != 0b110 {
            return None;
        }
        if (op >> 8) & 1 == 0 {
            return None;
        }
        let reg_list = (op & 0xff) as u8;
        Some(DecodedThumb14 {
            push: false,
            extra_reg: true,
            reg_list,
        })
    }

    /// Classify a Thumb format 18 unconditional branch:
    ///   11100_iiiiiiiiiii   (11 bit signed offset)
    /// Target = current pc (= instr + 4) + sign_extend(imm11) << 1.
    fn decode_thumb_format18(op: u16) -> Option<DecodedThumbPcBranch> {
        if (op >> 11) & 0b11111 != 0b11100 {
            return None;
        }
        let raw = (op & 0x07FF) as i32;
        let signed = (raw << 21) >> 21; // sign extend 11 bits
        Some(DecodedThumbPcBranch {
            cond: ArmCond::Al,
            offset_signed: signed,
        })
    }

    /// Classify a Thumb format 16 conditional branch:
    ///   1101_cccc_iiiiiiii
    /// cond 0xE (AL) is reserved here (that'd be format 18), cond 0xF is
    /// SWI (format 17). Both rejected. Target = pc + sign_extend(imm8) << 1.
    fn decode_thumb_format16(op: u16) -> Option<DecodedThumbPcBranch> {
        if (op >> 12) & 0xF != 0b1101 {
            return None;
        }
        let cond_bits = ((op >> 8) & 0xF) as u8;
        if cond_bits == 0xE || cond_bits == 0xF {
            return None;
        }
        let raw = (op & 0xFF) as i32;
        let signed = (raw << 24) >> 24; // sign extend 8 bits
        Some(DecodedThumbPcBranch {
            cond: ArmCond::from_bits(cond_bits),
            offset_signed: signed,
        })
    }

    /// Classify a Thumb BX (format 5 with oo=11). Encoding:
    ///   010001_11_0_H2_sss_000
    /// H1 and the low 3 bits are SBZ (should be zero); if set, we reject
    /// rather than silently compiling UNPREDICTABLE behavior.
    fn decode_thumb_bx(op: u16) -> Option<DecodedThumbBx> {
        if (op >> 10) & 0b111111 != 0b010001 {
            return None;
        }
        let oo = (op >> 8) & 0b11;
        if oo != 0b11 {
            return None;
        }
        let h1 = (op >> 7) & 1;
        if h1 != 0 {
            return None; // SBZ
        }
        if op & 0b111 != 0 {
            return None; // SBZ low bits
        }
        let h2 = (op >> 6) & 1;
        let rs_raw = (op >> 3) & 0b111;
        let rs = (rs_raw | (h2 << 3)) as i32;
        // PC as source would mean "BX PC" which flushes into a known
        // constant pc + 4 (Thumb) / pc + 8 (ARM). Deferred for now.
        if rs == 15 {
            return None;
        }
        Some(DecodedThumbBx { rs })
    }

    /// Classify a Thumb 16 bit opcode as format 5 (Hi register op),
    /// non-branch mnemonics only. Encoding:
    ///   010001_oo_H1_H2_sss_ddd
    ///     oo = 00 ADD, 01 CMP, 10 MOV, 11 BX
    ///     H1 selects Rd in upper bank (R8-R15), H2 same for Rs.
    ///     Full reg index = (H1<<3) | ddd  (and similarly H2|sss).
    ///
    /// Rejected here (return None):
    ///   - oo=11 (BX). Deferred to try_compile_block_with_branch.
    ///   - Any reg index = 15 (PC). Handling PC as source would need pc
    ///     folding; as dest it would flush the pipeline. Either way not
    ///     in the straight line compile path.
    ///   - oo=00/01/10 with both H1=0 and H2=0. That encoding is
    ///     UNPREDICTABLE per the ARM spec; bail to the interpreter.
    fn decode_thumb_format5_non_branch(op: u16) -> Option<DecodedThumb5> {
        if (op >> 10) & 0b111111 != 0b010001 {
            return None;
        }
        let oo = (op >> 8) & 0b11;
        if oo == 0b11 {
            return None; // BX handled elsewhere
        }
        let h1 = (op >> 7) & 1;
        let h2 = (op >> 6) & 1;
        if h1 == 0 && h2 == 0 {
            return None; // UNPREDICTABLE per spec
        }
        let rd_raw = op & 0b111;
        let rs_raw = (op >> 3) & 0b111;
        let rd = (rd_raw | (h1 << 3)) as i32;
        let rs = (rs_raw | (h2 << 3)) as i32;
        if rd == 15 || rs == 15 {
            return None;
        }
        let mnemonic = match oo {
            0b00 => Thumb5Op::Add,
            0b01 => Thumb5Op::Cmp,
            0b10 => Thumb5Op::Mov,
            _ => unreachable!(),
        };
        Some(DecodedThumb5 { op: mnemonic, rd, rs })
    }

    /// Classify a Thumb 16 bit opcode as format 1 (LSL/LSR/ASR Rd, Rs,
    /// #imm5). Encoding:
    ///   000_oo_iiiii_sss_ddd
    ///     oo = 00 LSL, 01 LSR, 10 ASR  (11 is format 2 add/sub, rejected)
    ///     imm5 = iiiii
    ///     Rs = sss, Rd = ddd
    pub(crate) fn decode_thumb_format1(op: u16) -> Option<DecodedThumb1> {
        if (op >> 13) & 0b111 != 0b000 {
            return None;
        }
        let oo = (op >> 11) & 0b11;
        if oo == 0b11 {
            // That's format 2.
            return None;
        }
        let imm5 = ((op >> 6) & 0b11111) as u32;
        let rs = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        let kind = match oo {
            0b00 => ShiftKind::Lsl,
            0b01 => ShiftKind::Lsr,
            0b10 => ShiftKind::Asr,
            _ => unreachable!(),
        };
        Some(DecodedThumb1 { kind, imm5, rs, rd })
    }

    /// Classify a Thumb 16 bit opcode as one of the format 4 logical
    /// subset: AND/EOR/ORR/BIC/MVN/TST/CMP/CMN Rd, Rs. Encoding:
    ///   010000_oooo_sss_ddd
    /// Only the logical / compare mnemonics are handled here. Shift ops
    /// (LSL/LSR/ASR/ROR) need barrel shifter carry handling, ADC/SBC need
    /// carry in, NEG needs signed negation semantics, MUL is its own
    /// timing. Those are all rejected for now.
    fn decode_thumb_format4_logical(op: u16) -> Option<DecodedThumb4> {
        if (op >> 10) & 0b111111 != 0b010000 {
            return None;
        }
        let op_bits = (op >> 6) & 0xf;
        let rs = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        let mnemonic = match op_bits {
            0b0000 => Thumb4Op::And,
            0b0001 => Thumb4Op::Eor,
            0b1000 => Thumb4Op::Tst,
            0b1010 => Thumb4Op::Cmp,
            0b1011 => Thumb4Op::Cmn,
            0b1100 => Thumb4Op::Orr,
            0b1110 => Thumb4Op::Bic,
            0b1111 => Thumb4Op::Mvn,
            // Unsupported: 0010 LSL, 0011 LSR, 0100 ASR, 0101 ADC,
            // 0110 SBC, 0111 ROR, 1001 NEG, 1101 MUL.
            _ => return None,
        };
        Some(DecodedThumb4 { op: mnemonic, rd, rs })
    }

    /// Classify a Thumb 16 bit opcode as format 2 (ADD/SUB Rd, Rs, Rn
    /// OR ADD/SUB Rd, Rs, #imm3). Encoding:
    ///   00011_I_Op_nnn_sss_ddd
    ///     I = 0 register (Rn = nnn), 1 immediate (imm3 = nnn)
    ///     Op = 0 ADD, 1 SUB
    ///     Rs = sss, Rd = ddd
    fn decode_thumb_format2(op: u16) -> Option<DecodedThumb2> {
        if (op >> 11) & 0b11111 != 0b00011 {
            return None;
        }
        let imm_form = (op >> 10) & 1 != 0;
        let sub = (op >> 9) & 1 != 0;
        let rn_or_imm = ((op >> 6) & 0b111) as u32;
        let rs = ((op >> 3) & 0b111) as i32;
        let rd = (op & 0b111) as i32;
        let operand = if imm_form {
            Thumb2Operand::Imm3(rn_or_imm)
        } else {
            Thumb2Operand::Reg(rn_or_imm as i32)
        };
        Some(DecodedThumb2 {
            sub,
            rd,
            rs,
            operand,
        })
    }

    /// Classify a Thumb 16 bit opcode as format 3 (MOV/CMP/ADD/SUB Rd,
    /// #imm8). Top 3 bits of the opcode must be 0b001.
    fn decode_thumb_format3(op: u16) -> Option<DecodedThumb3> {
        if (op >> 13) & 0b111 != 0b001 {
            return None;
        }
        let oo = (op >> 11) & 0b11;
        let rd = ((op >> 8) & 0b111) as i32;
        let imm8 = (op & 0xff) as u32;
        let mnemonic = match oo {
            0b00 => Thumb3Op::Mov,
            0b01 => Thumb3Op::Cmp,
            0b10 => Thumb3Op::Add,
            0b11 => Thumb3Op::Sub,
            _ => unreachable!(),
        };
        Some(DecodedThumb3 { op: mnemonic, rd, imm8 })
    }

    /// End to end smoke test for the bus trampoline plumbing. Builds a tiny
    /// function with signature
    ///     extern "C" fn(gpr_ptr: *mut u32, cpu_ctx: *mut u8, addr: u32)
    /// that calls the registered rba_bus_load_32 with (cpu_ctx, addr) and
    /// stores the returned value into gpr[0]. Lets a unit test verify the
    /// Rust -> Cranelift -> Rust callback round trip works without needing
    /// the full LDR decoder landed yet.
    ///
    /// Panics if this compiler was not built with `new_with_bus`.
    pub fn compile_bus_load_32_stub(
        &mut self,
    ) -> extern "C" fn(*mut u32, *mut u8, u32) {
        let imports = self
            .bus_imports
            .expect("compile_bus_load_32_stub requires new_with_bus()");

        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));   // gpr_ptr
        sig.params.push(AbiParam::new(ptr_type));   // cpu_ctx
        sig.params.push(AbiParam::new(types::I32)); // addr

        self.next_id += 1;
        let name = format!("dynarec_bus_stub_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            let cpu_ctx = builder.block_params(entry)[1];
            let addr = builder.block_params(entry)[2];

            // Reference the import inside this function so we can call it.
            let callee = self
                .module
                .declare_func_in_func(imports.load_32, builder.func);
            let call = builder.ins().call(callee, &[cpu_ctx, addr]);
            let loaded = builder.inst_results(call)[0];

            // gpr[0] = loaded
            builder
                .ins()
                .store(MemFlags::trusted(), loaded, gpr_ptr, Offset32::new(0));

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*mut u32, *mut u8, u32),
            >(code)
        }
    }

    /// Classify an ARM opcode as one of the supported data-processing shapes
    /// (immediate or register with no shift). Returns the decoded fields
    /// needed for codegen, or None for any unsupported encoding.
    fn decode_supported_dp(insn: u32) -> Option<DecodedDp> {
        let cond_bits = (insn >> 28) & 0xf;
        // NV (0xF) is reserved / invalid in ARMv4; skip.
        if cond_bits == 0xF {
            return None;
        }
        let cond = ArmCond::from_bits(cond_bits as u8);
        let class = (insn >> 20) & 0xff;
        // Data-processing bits [27:26] = 0b00.
        // Immediate form has bit 5 (of class) = 1 (= insn bit 25 = I).
        // Register form has I = 0.
        if (class & 0b1100_0000) != 0b0000_0000 {
            return None;
        }
        let i_bit = (class >> 5) & 1;
        let s_bit = (class & 1) != 0;
        let op_raw = (class >> 1) & 0xf;
        let op = match op_raw {
            0b1101 => DpOp::Mov,
            0b1111 => DpOp::Mvn,
            0b0100 => DpOp::Add,
            0b0010 => DpOp::Sub,
            0b1010 if s_bit => DpOp::Cmp,
            0b1011 if s_bit => DpOp::Cmn,
            0b1000 if s_bit => DpOp::Tst,
            0b1001 if s_bit => DpOp::Teq,
            _ => return None,
        };
        let rn = ((insn >> 16) & 0xf) as i32;
        let rd = ((insn >> 12) & 0xf) as i32;
        if !(0..15).contains(&rd) || !(0..15).contains(&rn) {
            return None;
        }

        let operand2 = if i_bit == 1 {
            // Immediate form: 8-bit value rotated right by 2 × rot4.
            let imm8 = insn & 0xff;
            let rot = ((insn >> 8) & 0xf) * 2;
            Operand2::Imm(imm8.rotate_right(rot))
        } else {
            // Register form: require no shift (shift_imm=0, shift_type=00)
            // and bit 4 = 0 (distinguishes from shift-by-register, which
            // has different timing and pipeline semantics).
            let shift_field = (insn >> 4) & 0xff;
            if shift_field != 0 {
                return None;
            }
            let rm = (insn & 0xf) as i32;
            if !(0..15).contains(&rm) {
                return None;
            }
            Operand2::Reg(rm)
        };

        Some(DecodedDp { cond, op, rd, rn, operand2, s: s_bit })
    }

    /// Compile a stub function that reads `gpr[1]` and writes it into
    /// `gpr[0]`. Used as an end-to-end smoke test for the JIT pipeline —
    /// proves codegen, define, and finalize all round-trip before we invest
    /// in the real ARM decoder→IR lowering.
    ///
    /// Returns an `extern "C"` fn pointer; the closure-wrapper is kept alive
    /// by the JITModule owned by this compiler, so the returned pointer is
    /// valid until `DynarecCompiler` drops.
    pub fn compile_mov_r0_r1_stub(&mut self) -> extern "C" fn(*mut u32) {
        let ptr_type = self.module.isa().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        self.next_id += 1;
        let name = format!("dynarec_stub_{}", self.next_id);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare_function failed");
        self.ctx.func.signature = sig;

        {
            let mut builder =
                FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let gpr_ptr = builder.block_params(entry)[0];
            // Load gpr[1] (4-byte offset into the u32 array).
            let r1 =
                builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), gpr_ptr, 4);
            // Store into gpr[0] (offset 0).
            builder
                .ins()
                .store(MemFlags::trusted(), r1, gpr_ptr, 0);
            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function failed");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions failed");

        let code = self.module.get_finalized_function(func_id);
        // SAFETY: the pointer Cranelift returns is a valid executable
        // function matching the signature we declared. We've pinned its
        // lifetime to `self` by keeping the JITModule alive here.
        unsafe { std::mem::transmute::<*const u8, extern "C" fn(*mut u32)>(code) }
    }
}

impl Default for DynarecCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Subset of ARM data-processing opcodes the dynarec can currently emit.
/// Cmp/Cmn/Tst/Teq are compare-only (no Rd writeback, S=1 mandatory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DpOp {
    Mov,
    Mvn,
    Add,
    Sub,
    Cmp,
    Cmn,
    Tst,
    Teq,
}

impl DpOp {
    fn is_compare_only(self) -> bool {
        matches!(self, DpOp::Cmp | DpOp::Cmn | DpOp::Tst | DpOp::Teq)
    }
}

/// The second operand of a data-processing instruction.
#[derive(Clone, Copy, Debug)]
enum Operand2 {
    Imm(u32),
    Reg(i32),
}

/// A decoded ARM B or BL instruction. `offset24_signed` is the raw 24 bit
/// immediate sign extended to i32 (not yet shifted by 2).
#[derive(Clone, Copy, Debug)]
struct DecodedBranch {
    cond: ArmCond,
    link: bool,
    offset24_signed: i32,
}

/// Thumb format 3 mnemonic: MOV / CMP / ADD / SUB Rd, #imm8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Thumb3Op { Mov, Cmp, Add, Sub }

#[derive(Clone, Copy, Debug)]
struct DecodedThumb3 {
    op: Thumb3Op,
    rd: i32,
    imm8: u32,
}

/// Thumb format 2: ADD/SUB Rd, Rs, (Rn | #imm3). operand picks between
/// a register source or a 3 bit immediate.
#[derive(Clone, Copy, Debug)]
enum Thumb2Operand {
    Reg(i32),
    Imm3(u32),
}

#[derive(Clone, Copy, Debug)]
struct DecodedThumb2 {
    sub: bool, // false = ADD, true = SUB
    rd: i32,
    rs: i32,
    operand: Thumb2Operand,
}

/// Thumb format 4 logical subset mnemonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Thumb4Op { And, Eor, Orr, Bic, Mvn, Tst, Cmp, Cmn }

#[derive(Clone, Copy, Debug)]
struct DecodedThumb4 {
    op: Thumb4Op,
    rd: i32,
    rs: i32,
}

/// Thumb format 5 (Hi register) mnemonic, non-branch. ADD/MOV don't
/// update flags in this form; CMP does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Thumb5Op { Add, Cmp, Mov }

#[derive(Clone, Copy, Debug)]
struct DecodedThumb5 {
    op: Thumb5Op,
    rd: i32,
    rs: i32,
}

/// Thumb format 6 PC-relative LDR.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb6 {
    rd: i32,
    imm8: u32,
}

/// Thumb format 15 LDM/STM (base write-back).
#[derive(Clone, Copy, Debug)]
struct DecodedThumb15 {
    load: bool,
    rb: i32,
    rlist: u8,
}

/// Thumb format 7 register-offset LDR/STR (word/byte).
#[derive(Clone, Copy, Debug)]
struct DecodedThumb7 {
    load: bool,
    byte: bool,
    ro: i32,
    rb: i32,
    rd: i32,
}

/// Thumb format 8 LDSB — sign-extended byte load with reg offset.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb8Ldsb {
    ro: i32,
    rb: i32,
    rd: i32,
}

/// Thumb format 10 halfword LDRH/STRH imm5.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb10 {
    load: bool,
    rs: i32,
    rd: i32,
    offset: u32,
}

/// Thumb format 13 ADD/SUB SP, #imm.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb13 {
    sub: bool,
    offset: u32,
}

/// Thumb format 12 load address (PC/SP-rel into Rd).
#[derive(Clone, Copy, Debug)]
struct DecodedThumb12 {
    sp: bool,
    rd: i32,
    offset: u32,
}

/// Thumb format 9 LDR/STR immediate offset (word or unsigned byte).
#[derive(Clone, Copy, Debug)]
struct DecodedThumb9 {
    load: bool,   // true = LDR/LDRB, false = STR/STRB
    byte: bool,   // true = byte size, false = word
    offset: u32,  // already scaled (word: imm5*4, byte: imm5)
    rs: i32,      // base register
    rd: i32,      // dest / src register
}

/// Thumb format 11 SP relative LDR/STR word.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb11 {
    load: bool,
    rd: i32,
    offset: u32, // already scaled by 4
}

/// Thumb format 14 PUSH/POP register list (non PC variant).
/// `extra_reg` is LR for PUSH or PC for POP. POP with extra_reg=true is
/// rejected by the classifier because it's a block terminator handled
/// elsewhere.
#[derive(Clone, Copy, Debug)]
struct DecodedThumb14 {
    push: bool,      // true = PUSH (L=0), false = POP (L=1)
    extra_reg: bool, // R bit
    reg_list: u8,    // R0..R7 bitmap
}

impl DecodedThumb14 {
    /// Total number of registers transferred by this instruction.
    fn count(&self) -> u32 {
        self.reg_list.count_ones() + self.extra_reg as u32
    }
}

/// Thumb BX Rs (format 5 with oo=11). Reads gpr[rs] at runtime and jumps
/// there, preserving bit 0 as the ARM/Thumb mode signal.
#[derive(Clone, Copy, Debug)]
struct DecodedThumbBx {
    rs: i32,
}

/// Thumb PC relative branch (format 16 conditional or format 18
/// unconditional). Target pc is computed at codegen time from
/// entry_pc + in-block offset + 4 (pipeline) + (offset << 1).
#[derive(Clone, Copy, Debug)]
struct DecodedThumbPcBranch {
    cond: ArmCond,           // Al for format 18
    offset_signed: i32,      // already sign extended, NOT yet shifted by 1
}

/// ARM shifter operation kind. Only the three Thumb format 1 variants
/// for now. Format 4 reg shifts (LSL/LSR/ASR/ROR with register amount)
/// would extend this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShiftKind { Lsl, Lsr, Asr }

#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodedThumb1 {
    pub(crate) kind: ShiftKind,
    pub(crate) imm5: u32,
    pub(crate) rs: i32,
    pub(crate) rd: i32,
}

/// A decoded ARM LDR / STR immediate instruction. Pre indexed, no writeback,
/// any offset sign, either word or byte size. The dynarec emits this by
/// calling into the bus trampolines at runtime.
#[derive(Clone, Copy, Debug)]
struct DecodedMem {
    cond: ArmCond,
    /// true = LDR, false = STR
    load: bool,
    /// true = byte, false = word.
    byte: bool,
    /// true = add offset to base, false = subtract.
    add: bool,
    rd: i32,
    rn: i32,
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
struct DecodedDp {
    cond: ArmCond,
    op: DpOp,
    rd: i32,
    rn: i32,
    operand2: Operand2,
    /// Update CPSR.NZCV after executing this instruction. Compare-only ops
    /// (CMP/CMN/TST/TEQ) always have s=true.
    s: bool,
}

/// ARM condition-code mnemonics. Evaluated against the NZCV bits of CPSR
/// before each instruction body runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmCond {
    Eq, Ne, Hs, Lo, Mi, Pl, Vs, Vc,
    Hi, Ls, Ge, Lt, Gt, Le, Al,
}

impl ArmCond {
    fn from_bits(bits: u8) -> Self {
        use ArmCond::*;
        match bits & 0xf {
            0x0 => Eq, 0x1 => Ne, 0x2 => Hs, 0x3 => Lo,
            0x4 => Mi, 0x5 => Pl, 0x6 => Vs, 0x7 => Vc,
            0x8 => Hi, 0x9 => Ls, 0xA => Ge, 0xB => Lt,
            0xC => Gt, 0xD => Le, 0xE => Al,
            _ => Al, // 0xF handled upstream
        }
    }
}

/// Emit Cranelift IR that evaluates an ARM condition code against the CPSR
/// variable and returns a `bool` (i8 in IR) that's true when the condition
/// passes.
fn emit_cond_check(
    builder: &mut FunctionBuilder,
    cpsr_var: Variable,
    cond: ArmCond,
) -> Value {
    let cpsr = builder.use_var(cpsr_var);
    // Flag bit positions in CPSR: N=31, Z=30, C=29, V=28.
    let one = builder.ins().iconst(types::I32, 1);
    let n = {
        let shifted = builder.ins().ushr_imm(cpsr, 31);
        builder.ins().band(shifted, one)
    };
    let z = {
        let shifted = builder.ins().ushr_imm(cpsr, 30);
        builder.ins().band(shifted, one)
    };
    let c = {
        let shifted = builder.ins().ushr_imm(cpsr, 29);
        builder.ins().band(shifted, one)
    };
    let v = {
        let shifted = builder.ins().ushr_imm(cpsr, 28);
        builder.ins().band(shifted, one)
    };
    let zero = builder.ins().iconst(types::I32, 0);

    let true_val = builder.ins().iconst(types::I8, 1);

    match cond {
        ArmCond::Al => true_val,
        ArmCond::Eq => builder.ins().icmp(IntCC::NotEqual, z, zero),
        ArmCond::Ne => builder.ins().icmp(IntCC::Equal, z, zero),
        ArmCond::Hs => builder.ins().icmp(IntCC::NotEqual, c, zero),
        ArmCond::Lo => builder.ins().icmp(IntCC::Equal, c, zero),
        ArmCond::Mi => builder.ins().icmp(IntCC::NotEqual, n, zero),
        ArmCond::Pl => builder.ins().icmp(IntCC::Equal, n, zero),
        ArmCond::Vs => builder.ins().icmp(IntCC::NotEqual, v, zero),
        ArmCond::Vc => builder.ins().icmp(IntCC::Equal, v, zero),
        ArmCond::Hi => {
            // C=1 && Z=0
            let c_set = builder.ins().icmp(IntCC::NotEqual, c, zero);
            let z_clear = builder.ins().icmp(IntCC::Equal, z, zero);
            builder.ins().band(c_set, z_clear)
        }
        ArmCond::Ls => {
            // C=0 || Z=1
            let c_clear = builder.ins().icmp(IntCC::Equal, c, zero);
            let z_set = builder.ins().icmp(IntCC::NotEqual, z, zero);
            builder.ins().bor(c_clear, z_set)
        }
        ArmCond::Ge => builder.ins().icmp(IntCC::Equal, n, v),
        ArmCond::Lt => builder.ins().icmp(IntCC::NotEqual, n, v),
        ArmCond::Gt => {
            // Z=0 && N==V
            let z_clear = builder.ins().icmp(IntCC::Equal, z, zero);
            let nv_eq = builder.ins().icmp(IntCC::Equal, n, v);
            builder.ins().band(z_clear, nv_eq)
        }
        ArmCond::Le => {
            // Z=1 || N!=V
            let z_set = builder.ins().icmp(IntCC::NotEqual, z, zero);
            let nv_ne = builder.ins().icmp(IntCC::NotEqual, n, v);
            builder.ins().bor(z_set, nv_ne)
        }
    }
}

/// Emit a conditionally-executed data-processing instruction. For AL cond
/// this is the same as emit_data_processing_imm; for any other cond we
/// wrap the body in a brif/skip pattern.
fn emit_conditional_instr(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedDp,
) {
    if dec.cond == ArmCond::Al {
        emit_data_processing_imm(builder, gpr_ptr, cpsr_var, dec);
        return;
    }

    let cond_result = emit_cond_check(builder, cpsr_var, dec.cond);
    let body = builder.create_block();
    let merge = builder.create_block();
    builder.ins().brif(cond_result, body, &[], merge, &[]);
    builder.switch_to_block(body);
    builder.seal_block(body);
    emit_data_processing_imm(builder, gpr_ptr, cpsr_var, dec);
    builder.ins().jump(merge, &[]);
    builder.switch_to_block(merge);
    builder.seal_block(merge);
}

/// Same as `emit_conditional_instr` but skips the flag-update portion
/// of the contained DP emit. Used by the dead-flag-write pass in
/// `try_compile_imm_block` when the next instruction in the block
/// unconditionally overwrites this one's NZCV footprint.
///
/// For a conditional instruction (cond != AL) whose body is flag-dead,
/// the dead-flag analysis still emits the cond-check + body, the body
/// just doesn't pack flags. That's strictly correct since the next
/// unconditional flag-write overwrites.
fn emit_conditional_instr_no_flags(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    mut dec: DecodedDp,
) {
    // Trick: pass S=false into the emit path so emit_data_processing_imm
    // skips the emit_flag_update call entirely. Result + writeback are
    // unchanged.
    dec.s = false;
    emit_conditional_instr(builder, gpr_ptr, cpsr_var, dec);
}

/// Emit a Thumb format 14 PUSH or POP register list.
///
/// Address ordering (ARMv4 PUSH/POP = STMDB/LDMIA on SP):
///   PUSH:  new_sp = SP - 4*count; write registers low-to-high to
///          addresses new_sp, new_sp+4, ... (lowest reg at lowest addr).
///          End SP = new_sp.
///   POP:   read registers low-to-high from SP, SP+4, ... up to
///          SP + 4*(count-1). End SP = SP + 4*count.
///
/// The register list for PUSH is R0..R7 ordered, then LR. For POP the
/// order is R0..R7, then PC. We only handle POP without PC here
/// (classifier rejects).
fn emit_thumb_format14(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_32_seq_ref: cranelift::codegen::ir::FuncRef,
    store_32_seq_ref: cranelift::codegen::ir::FuncRef,
    idle_cycle_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb14,
) {
    let count = dec.count();
    let bytes = (count as i64) * 4;
    let sp = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(13 * 4),
    );

    let start_addr = if dec.push {
        builder.ins().iadd_imm(sp, -bytes)
    } else {
        sp
    };

    // Walk the register list low-to-high.  First access is NonSeq,
    // all following accesses are Seq — mirrors scalar PUSH/POP
    // cycle accounting (first load_32 NonSeq, rest Seq).
    let mut byte_offset = 0i64;
    let mut access_count = 0u32;
    for i in 0..8 {
        if dec.reg_list & (1 << i) != 0 {
            let addr = builder.ins().iadd_imm(start_addr, byte_offset);
            if dec.push {
                let v = builder.ins().load(
                    types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(i * 4),
                );
                let ref_fn = if access_count == 0 { store_32_ref } else { store_32_seq_ref };
                builder.ins().call(ref_fn, &[cpu_ctx, addr, v]);
            } else {
                let ref_fn = if access_count == 0 { load_32_ref } else { load_32_seq_ref };
                let call = builder.ins().call(ref_fn, &[cpu_ctx, addr]);
                let v = builder.inst_results(call)[0];
                builder.ins().store(
                    MemFlags::trusted(), v, gpr_ptr, Offset32::new(i * 4),
                );
            }
            byte_offset += 4;
            access_count += 1;
        }
    }
    // LR bit for PUSH (extra_reg = LR = R14). POP with PC bit was rejected
    // by the classifier.
    if dec.extra_reg && dec.push {
        let addr = builder.ins().iadd_imm(start_addr, byte_offset);
        let lr = builder.ins().load(
            types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(14 * 4),
        );
        let ref_fn = if access_count == 0 { store_32_ref } else { store_32_seq_ref };
        builder.ins().call(ref_fn, &[cpu_ctx, addr, lr]);
        // byte_offset += 4; (unused after this, kept for readability)
    }

    // Update SP.
    let new_sp = if dec.push {
        start_addr // which is sp - bytes
    } else {
        builder.ins().iadd_imm(sp, bytes)
    };
    builder.ins().store(
        MemFlags::trusted(), new_sp, gpr_ptr, Offset32::new(13 * 4),
    );

    // Scalar POP (format 14, non-PC) adds a single idle cycle at the
    // end of the whole multi-load (see thumb/exec.rs:exec_thumb_push_pop
    // "// Idle 1 cycle"). Mirror that here so cycle accounting matches.
    // PUSH has no trailing idle cycle.
    if !dec.push {
        builder.ins().call(idle_cycle_ref, &[cpu_ctx]);
    }
}

/// Emit a Thumb format 11 SP relative LDR/STR (word). Base register is
/// hardcoded to R13 (SP) in the encoding.
///
/// LDR uses the *_with_idle trampoline to charge the +1I scalar adds
/// after every LDR. STR uses the plain store_32; scalar STR doesn't
/// add an internal cycle (its NonSeq next-fetch effect is handled by
/// the per-block last-instruction fixup).
fn emit_thumb_format11(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb11,
) {
    let sp = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(13 * 4),
    );
    let offset = builder.ins().iconst(types::I32, dec.offset as i64);
    let addr = builder.ins().iadd(sp, offset);
    if dec.load {
        let call = builder.ins().call(load_idle_32_ref, &[cpu_ctx, addr]);
        let v = builder.inst_results(call)[0];
        builder.ins().store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
    } else {
        let rd_val = builder.ins().load(
            types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
        );
        builder.ins().call(store_32_ref, &[cpu_ctx, addr, rd_val]);
    }
}

/// Emit a Thumb format 9 LDR/STR immediate offset. addr = gpr[rs] + offset.
/// Word form uses load_with_idle_32/store_32 trampolines; byte form uses
/// load_with_idle_8/store_8 with zero extension (LDRB) or low-byte
/// truncation (STRB). The "_with_idle" load trampolines charge +1I per
/// LDR/LDRB to match the scalar `idle_cycle()` after each data fetch.
/// Thumb format 6 (PC-relative LDR) — LDR Rd, [PC, #imm8*4]. Since
/// the pc of each instruction in a compiled block is known at
/// codegen time, the effective address is a constant folded here.
/// `instr_pc` is the pc of the instruction being emitted — this is
/// the pc Thumb semantics call "PC", which is aligned then offset.
fn emit_thumb_format6(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb6,
    instr_pc: u32,
) {
    // Thumb format 6 spec: addr = ((PC + 4) & ~3) + imm8 * 4, where
    // PC is the address of the LDR instruction. (PC + 4 because the
    // Thumb pipeline-head convention leaves PC pointing two instrs
    // ahead during decode.)
    let addr = instr_pc
        .wrapping_add(4)
        .wrapping_add(0xFFFF_FFFC_u32 & 0)  // no-op, documents the align
        & !3_u32;
    let addr = addr.wrapping_add(dec.imm8.wrapping_mul(4));
    let addr_val = builder.ins().iconst(types::I32, addr as i64);
    let call = builder.ins().call(load_idle_32_ref, &[cpu_ctx, addr_val]);
    let v = builder.inst_results(call)[0];
    builder
        .ins()
        .store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
}

/// Thumb format 8 LDSB — sign-extended byte load. Addr = Rb + Ro,
/// load 1 byte via load_with_idle_8, sign-extend to i32.
fn emit_thumb_format8_ldsb(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_8_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb8Ldsb,
) {
    let rb_val = builder.ins().load(
        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rb * 4),
    );
    let ro_val = builder.ins().load(
        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.ro * 4),
    );
    let addr = builder.ins().iadd(rb_val, ro_val);
    let call = builder.ins().call(load_idle_8_ref, &[cpu_ctx, addr]);
    let byte = builder.inst_results(call)[0];
    // Sign-extend low byte to i32: shift left 24, arith-shift-right 24.
    let shl = builder.ins().ishl_imm(byte, 24);
    let signed = builder.ins().sshr_imm(shl, 24);
    builder
        .ins()
        .store(MemFlags::trusted(), signed, gpr_ptr, Offset32::new(dec.rd * 4));
}

/// Thumb format 10 halfword LDRH/STRH imm. Address = Rs + offset.
fn emit_thumb_format10(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_16_ref: cranelift::codegen::ir::FuncRef,
    store_16_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb10,
) {
    let rs_val = builder.ins().load(
        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rs * 4),
    );
    let addr = builder.ins().iadd_imm(rs_val, dec.offset as i64);
    if dec.load {
        let call = builder.ins().call(load_idle_16_ref, &[cpu_ctx, addr]);
        let v = builder.inst_results(call)[0];
        builder
            .ins()
            .store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
    } else {
        let rd_val = builder.ins().load(
            types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
        );
        builder.ins().call(store_16_ref, &[cpu_ctx, addr, rd_val]);
    }
}

/// Thumb format 15 LDM/STM register list. First access NonSeq, rest
/// Seq. LDM: +1I at end, NO base write-back if Rb in rlist. STM: no
/// idle cycle. Base write-back happens AFTER the multi-access, with
/// `align_preserve = Rb & 3` reapplied. Empty rlist + STM-with-Rb-in-
/// rlist are filtered out by `decode_thumb_format15`.
fn emit_thumb_format15(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_32_seq_ref: cranelift::codegen::ir::FuncRef,
    store_32_seq_ref: cranelift::codegen::ir::FuncRef,
    idle_cycle_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb15,
) {
    let rb_val = builder.ins().load(
        types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rb * 4),
    );
    let align_preserve = builder.ins().band_imm(rb_val, 3);
    // addr starts at Rb & !3.
    let addr0 = builder.ins().band_imm(rb_val, !3i64);
    let mut byte_offset: i64 = 0;
    let mut access_count = 0u32;
    for i in 0..8 {
        if dec.rlist & (1 << i) != 0 {
            let addr = builder.ins().iadd_imm(addr0, byte_offset);
            if dec.load {
                let ref_fn = if access_count == 0 { load_32_ref } else { load_32_seq_ref };
                let call = builder.ins().call(ref_fn, &[cpu_ctx, addr]);
                let v = builder.inst_results(call)[0];
                builder.ins().store(
                    MemFlags::trusted(), v, gpr_ptr, Offset32::new(i * 4),
                );
            } else {
                let v = builder.ins().load(
                    types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(i * 4),
                );
                let ref_fn = if access_count == 0 { store_32_ref } else { store_32_seq_ref };
                builder.ins().call(ref_fn, &[cpu_ctx, addr, v]);
            }
            byte_offset += 4;
            access_count += 1;
        }
    }
    // LDM: +1I at end.
    if dec.load {
        builder.ins().call(idle_cycle_ref, &[cpu_ctx]);
    }
    // Base write-back: addr + 4*count + align_preserve.
    //   LDM case: only write back if Rb NOT in rlist. Scalar skips
    //   the update when the register it loaded was Rb itself.
    //   STM case: classifier rejected Rb-in-rlist, so always update.
    let skip_writeback = dec.load && (dec.rlist & (1 << dec.rb)) != 0;
    if !skip_writeback {
        let new_base = builder.ins().iadd_imm(addr0, byte_offset);
        let new_base_preserved = builder.ins().iadd(new_base, align_preserve);
        builder.ins().store(
            MemFlags::trusted(),
            new_base_preserved,
            gpr_ptr,
            Offset32::new(dec.rb * 4),
        );
    }
}

/// Thumb format 7 register-offset LDR/STR.  Address = Rb + Ro.
/// Word load uses `load_with_idle_32` (scalar: ldr_word + +1I).
/// Byte load uses `load_with_idle_8`. STR uses plain store_32 /
/// store_8.  Mirrors scalar `do_exec_thumb_ldr_str`.
fn emit_thumb_format7(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_idle_8_ref: cranelift::codegen::ir::FuncRef,
    store_8_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb7,
) {
    let rb_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rb * 4),
    );
    let ro_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.ro * 4),
    );
    let addr = builder.ins().iadd(rb_val, ro_val);
    match (dec.load, dec.byte) {
        (true, false) => {
            let call = builder.ins().call(load_idle_32_ref, &[cpu_ctx, addr]);
            let v = builder.inst_results(call)[0];
            builder
                .ins()
                .store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
        }
        (true, true) => {
            let call = builder.ins().call(load_idle_8_ref, &[cpu_ctx, addr]);
            let v = builder.inst_results(call)[0];
            builder
                .ins()
                .store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
        }
        (false, false) => {
            let rd_val = builder.ins().load(
                types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
            );
            builder.ins().call(store_32_ref, &[cpu_ctx, addr, rd_val]);
        }
        (false, true) => {
            let rd_val = builder.ins().load(
                types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
            );
            builder.ins().call(store_8_ref, &[cpu_ctx, addr, rd_val]);
        }
    }
}

/// Thumb format 12 load address. SP case: Rd = SP + offset.
/// PC case: Rd is a constant `((instr_pc & !2) + 4 + offset)`,
/// folded here since instr_pc is known at codegen.
fn emit_thumb_format12(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    dec: DecodedThumb12,
    instr_pc: u32,
) {
    let val = if dec.sp {
        let sp = builder.ins().load(
            types::I32,
            MemFlags::trusted(),
            gpr_ptr,
            Offset32::new(13 * 4),
        );
        builder.ins().iadd_imm(sp, dec.offset as i64)
    } else {
        // scalar: (pc_thumb() & !0b10) + 4 + offset. pc_thumb() = instr_pc.
        let folded = (instr_pc & !0b10_u32)
            .wrapping_add(4)
            .wrapping_add(dec.offset);
        builder.ins().iconst(types::I32, folded as i64)
    };
    builder
        .ins()
        .store(MemFlags::trusted(), val, gpr_ptr, Offset32::new(dec.rd * 4));
}

/// Thumb format 13 ADD SP, #imm / SUB SP, #imm. Pure SP update.
fn emit_thumb_format13(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    dec: DecodedThumb13,
) {
    let sp = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(13 * 4),
    );
    let delta = dec.offset as i64;
    let new_sp = if dec.sub {
        builder.ins().iadd_imm(sp, -delta)
    } else {
        builder.ins().iadd_imm(sp, delta)
    };
    builder
        .ins()
        .store(MemFlags::trusted(), new_sp, gpr_ptr, Offset32::new(13 * 4));
}

fn emit_thumb_format9(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_idle_8_ref: cranelift::codegen::ir::FuncRef,
    store_8_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedThumb9,
) {
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );
    let offset = builder.ins().iconst(types::I32, dec.offset as i64);
    let addr = builder.ins().iadd(rs_val, offset);

    match (dec.load, dec.byte) {
        (true, false) => {
            let call = builder.ins().call(load_idle_32_ref, &[cpu_ctx, addr]);
            let v = builder.inst_results(call)[0];
            builder.ins().store(MemFlags::trusted(), v, gpr_ptr, Offset32::new(dec.rd * 4));
        }
        (true, true) => {
            let call = builder.ins().call(load_idle_8_ref, &[cpu_ctx, addr]);
            let v = builder.inst_results(call)[0];
            let zero_ext = builder.ins().band_imm(v, 0xff);
            builder.ins().store(MemFlags::trusted(), zero_ext, gpr_ptr, Offset32::new(dec.rd * 4));
        }
        (false, false) => {
            let rd_val = builder.ins().load(
                types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
            );
            builder.ins().call(store_32_ref, &[cpu_ctx, addr, rd_val]);
        }
        (false, true) => {
            let rd_val = builder.ins().load(
                types::I32, MemFlags::trusted(), gpr_ptr, Offset32::new(dec.rd * 4),
            );
            let byte_val = builder.ins().band_imm(rd_val, 0xff);
            builder.ins().call(store_8_ref, &[cpu_ctx, addr, byte_val]);
        }
    }
}

/// Emit a Thumb format 5 non-branch op (ADD/CMP/MOV with Hi registers).
/// ADD and MOV do not update flags in this form. CMP updates N/Z/C/V
/// just like CMP in format 4 / ARM DP S bit.
fn emit_thumb_format5_non_branch(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedThumb5,
) {
    let rd_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );

    match dec.op {
        Thumb5Op::Mov => {
            // Rd = Rs, no flag update.
            builder.ins().store(
                MemFlags::trusted(),
                rs_val,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        Thumb5Op::Add => {
            // Rd = Rd + Rs, no flag update.
            let result = builder.ins().iadd(rd_val, rs_val);
            builder.ins().store(
                MemFlags::trusted(),
                result,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        Thumb5Op::Cmp => {
            // flags from Rd - Rs, no writeback. Reuse the ARM DP S bit
            // path via emit_flag_update with DpOp::Cmp.
            let result = builder.ins().isub(rd_val, rs_val);
            let new_cpsr = emit_flag_update(builder, cpsr_var, DpOp::Cmp, rd_val, rs_val, result);
            builder.def_var(cpsr_var, new_cpsr);
        }
    }
}

/// Emit a Thumb format 1 shift by immediate (LSL/LSR/ASR Rd, Rs, #imm5).
/// Writes N, Z, and C (shifter carry) to CPSR. Preserves V.
///
/// ARM7TDMI barrel shifter special cases:
///   LSL #0:  result = Rs, C preserved
///   LSR #0:  decoded as LSR #32 -> result = 0,  C = bit 31 of Rs
///   ASR #0:  decoded as ASR #32 -> result = Rs arith shift by 31 (all sign
///            bits), C = bit 31 of Rs
///   LSL #n (1..31): result = Rs << n,     C = bit (32-n) of Rs
///   LSR #n (1..31): result = Rs >> n,     C = bit (n-1) of Rs
///   ASR #n (1..31): result = (i32)Rs >> n, C = bit (n-1) of Rs
fn emit_thumb_format1(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedThumb1,
) {
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );

    // Precompute result and shifter-out C bit at codegen time by folding
    // the constant imm5 into specific instruction sequences. This avoids
    // runtime branching on the shift amount.
    let one = builder.ins().iconst(types::I32, 1);
    let preserve_c = {
        // Current C bit from cpsr_var, shifted down to bit 0.
        let cpsr = builder.use_var(cpsr_var);
        let shifted = builder.ins().ushr_imm(cpsr, 29);
        builder.ins().band(shifted, one)
    };

    let (result, new_c) = match (dec.kind, dec.imm5) {
        (ShiftKind::Lsl, 0) => {
            // LSL #0: no shift, C preserved.
            (rs_val, preserve_c)
        }
        (ShiftKind::Lsl, n) => {
            // result = Rs << n ; C = bit (32 - n) of Rs
            let r = builder.ins().ishl_imm(rs_val, n as i64);
            let c_shift = 32 - n as i64;
            let c_raw = builder.ins().ushr_imm(rs_val, c_shift);
            let c = builder.ins().band(c_raw, one);
            (r, c)
        }
        (ShiftKind::Lsr, 0) => {
            // LSR #0 means LSR #32: result = 0, C = bit 31 of Rs.
            let r = builder.ins().iconst(types::I32, 0);
            let c_raw = builder.ins().ushr_imm(rs_val, 31);
            let c = builder.ins().band(c_raw, one);
            (r, c)
        }
        (ShiftKind::Lsr, n) => {
            let r = builder.ins().ushr_imm(rs_val, n as i64);
            let c_raw = builder.ins().ushr_imm(rs_val, (n - 1) as i64);
            let c = builder.ins().band(c_raw, one);
            (r, c)
        }
        (ShiftKind::Asr, 0) => {
            // ASR #0 means ASR #32: result = all sign bits, C = bit 31.
            let r = builder.ins().sshr_imm(rs_val, 31);
            let c_raw = builder.ins().ushr_imm(rs_val, 31);
            let c = builder.ins().band(c_raw, one);
            (r, c)
        }
        (ShiftKind::Asr, n) => {
            let r = builder.ins().sshr_imm(rs_val, n as i64);
            let c_raw = builder.ins().ushr_imm(rs_val, (n - 1) as i64);
            let c = builder.ins().band(c_raw, one);
            (r, c)
        }
    };

    builder.ins().store(
        MemFlags::trusted(),
        result,
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );

    // N, Z from result. C is new_c. V preserved.
    let zero = builder.ins().iconst(types::I32, 0);
    let n = builder.ins().ushr_imm(result, 31);
    let z_bool = builder.ins().icmp(IntCC::Equal, result, zero);
    let z = builder.ins().uextend(types::I32, z_bool);
    let v = {
        let cpsr = builder.use_var(cpsr_var);
        let shifted = builder.ins().ushr_imm(cpsr, 28);
        builder.ins().band(shifted, one)
    };

    let cpsr = builder.use_var(cpsr_var);
    let mask = builder.ins().iconst(types::I32, 0x0fff_ffff);
    let cleared = builder.ins().band(cpsr, mask);
    let n_shifted = builder.ins().ishl_imm(n, 31);
    let z_shifted = builder.ins().ishl_imm(z, 30);
    let c_shifted = builder.ins().ishl_imm(new_c, 29);
    let v_shifted = builder.ins().ishl_imm(v, 28);
    let nz = builder.ins().bor(n_shifted, z_shifted);
    let cv = builder.ins().bor(c_shifted, v_shifted);
    let flags = builder.ins().bor(nz, cv);
    let new_cpsr = builder.ins().bor(cleared, flags);
    builder.def_var(cpsr_var, new_cpsr);
}

/// Emit a Thumb format 4 logical / compare op. All mnemonics in this
/// subset update NZ flags. AND/EOR/ORR/BIC/MVN writeback to Rd; TST/CMP/CMN
/// do not. C and V behave like the equivalent ARM DP S bit path:
/// preserved for logical, computed for CMP/CMN.
fn emit_thumb_format4_logical(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedThumb4,
) {
    let rd_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );

    let (result, dp_equivalent, writeback) = match dec.op {
        Thumb4Op::And => (builder.ins().band(rd_val, rs_val), DpOp::Tst, true),
        Thumb4Op::Eor => (builder.ins().bxor(rd_val, rs_val), DpOp::Teq, true),
        Thumb4Op::Orr => {
            // ARM ORR flag semantics == TST (logical, N/Z from result,
            // preserve C/V). The Tst branch of emit_flag_update reads the
            // existing cpsr C/V and leaves them. Same goes for EOR vs TEQ.
            let r = builder.ins().bor(rd_val, rs_val);
            (r, DpOp::Tst, true)
        }
        Thumb4Op::Bic => {
            // Rd = Rd & ~Rs
            let not_rs = builder.ins().bnot(rs_val);
            let r = builder.ins().band(rd_val, not_rs);
            (r, DpOp::Tst, true)
        }
        Thumb4Op::Mvn => {
            // Rd = ~Rs  (Rd value ignored as source).
            let r = builder.ins().bnot(rs_val);
            (r, DpOp::Mov, true)
        }
        Thumb4Op::Tst => (builder.ins().band(rd_val, rs_val), DpOp::Tst, false),
        Thumb4Op::Cmp => (builder.ins().isub(rd_val, rs_val), DpOp::Cmp, false),
        Thumb4Op::Cmn => (builder.ins().iadd(rd_val, rs_val), DpOp::Cmn, false),
    };

    if writeback {
        builder.ins().store(
            MemFlags::trusted(),
            result,
            gpr_ptr,
            Offset32::new(dec.rd * 4),
        );
    }

    let new_cpsr = emit_flag_update(builder, cpsr_var, dp_equivalent, rd_val, rs_val, result);
    builder.def_var(cpsr_var, new_cpsr);
}

/// Emit a Thumb format 2 add/sub. Always updates full NZCV (Thumb
/// ADD/SUB behave like ARM DP with S=1 always in ARMv4 outside IT blocks).
fn emit_thumb_format2(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedThumb2,
) {
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );
    // For Imm3 shape (imm3 ∈ [0,7], always non-negative), rhs sign
    // bit is known 0, so the same simplified V formula Thumb3 uses
    // applies:
    //   ADD V = ~rs & result & 0x8000_0000
    //   SUB V = rs & ~result & 0x8000_0000
    // Saves 2 CLIF ops per Thumb2 imm3 instruction vs general.
    let is_small_positive_imm = matches!(dec.operand, Thumb2Operand::Imm3(_));
    let rhs = match dec.operand {
        Thumb2Operand::Imm3(v) => builder.ins().iconst(types::I32, v as i64),
        Thumb2Operand::Reg(rn) => builder.ins().load(
            types::I32,
            MemFlags::trusted(),
            gpr_ptr,
            Offset32::new(rn * 4),
        ),
    };
    let (result, dp_equivalent) = if dec.sub {
        (builder.ins().isub(rs_val, rhs), DpOp::Sub)
    } else {
        (builder.ins().iadd(rs_val, rhs), DpOp::Add)
    };
    builder.ins().store(
        MemFlags::trusted(),
        result,
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );

    if is_small_positive_imm {
        // Inlined flag update with simplified V (mirrors emit_thumb_format3
        // ADD/SUB path).
        let zero = builder.ins().iconst(types::I32, 0);
        let n_shifted = builder.ins().band_imm(result, 0x8000_0000_u32 as i64);
        let z_bool = builder.ins().icmp(IntCC::Equal, result, zero);
        let z_u32 = builder.ins().uextend(types::I32, z_bool);
        let z_shifted = builder.ins().ishl_imm(z_u32, 30);
        let (c_bool, v_shifted) = if dec.sub {
            let c = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, rs_val, rhs);
            let not_result = builder.ins().bnot(result);
            let v_bits = builder.ins().band(rs_val, not_result);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            (c, builder.ins().ushr_imm(v_top, 3))
        } else {
            let c = builder.ins().icmp(IntCC::UnsignedLessThan, result, rs_val);
            let not_rs = builder.ins().bnot(rs_val);
            let v_bits = builder.ins().band(not_rs, result);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            (c, builder.ins().ushr_imm(v_top, 3))
        };
        let c_u32 = builder.ins().uextend(types::I32, c_bool);
        let c_shifted = builder.ins().ishl_imm(c_u32, 29);
        let cpsr = builder.use_var(cpsr_var);
        let cleared = builder.ins().band_imm(cpsr, 0x0fff_ffff);
        let nz = builder.ins().bor(n_shifted, z_shifted);
        let cv = builder.ins().bor(c_shifted, v_shifted);
        let flags = builder.ins().bor(nz, cv);
        let new_cpsr = builder.ins().bor(cleared, flags);
        builder.def_var(cpsr_var, new_cpsr);
    } else {
        // Reg operand → rhs sign could be anything; use general path.
        let new_cpsr =
            emit_flag_update(builder, cpsr_var, dp_equivalent, rs_val, rhs, result);
        builder.def_var(cpsr_var, new_cpsr);
    }
}

/// Emit a Thumb format 2 instruction with the flag update SKIPPED.
/// Counterpart to `emit_thumb_format2` used by the dead-flag-write pass.
/// Drops the entire cpsr pack sequence; still performs the ADD/SUB
/// and writes the result back to rd.
fn emit_thumb_format2_no_flags(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    dec: DecodedThumb2,
) {
    let rs_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rs * 4),
    );
    let rhs = match dec.operand {
        Thumb2Operand::Imm3(v) => builder.ins().iconst(types::I32, v as i64),
        Thumb2Operand::Reg(rn) => builder.ins().load(
            types::I32,
            MemFlags::trusted(),
            gpr_ptr,
            Offset32::new(rn * 4),
        ),
    };
    let result = if dec.sub {
        builder.ins().isub(rs_val, rhs)
    } else {
        builder.ins().iadd(rs_val, rhs)
    };
    builder.ins().store(
        MemFlags::trusted(),
        result,
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );
}

/// Emit a Thumb format 3 instruction with the flag update SKIPPED.
/// Used by the dead-flag-write pass in `try_compile_thumb_block` when
/// the next in-block instruction overwrites this one's flag bits.
/// Drops the entire cpsr pack/merge sequence; still writes back the
/// result for ADD/SUB/MOV.
fn emit_thumb_format3_no_flags(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    dec: DecodedThumb3,
) {
    let imm8 = builder.ins().iconst(types::I32, dec.imm8 as i64);
    match dec.op {
        Thumb3Op::Mov => {
            builder.ins().store(
                MemFlags::trusted(),
                imm8,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        Thumb3Op::Cmp => {
            // Compare-only with no flag write is a true no-op; do
            // nothing. This path is only reached if the dead-flag pass
            // marked it skippable, which implies the next instr
            // overwrites NZCV — effectively CMP vanishes.
        }
        Thumb3Op::Add => {
            let rd_val = builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
            let result = builder.ins().iadd(rd_val, imm8);
            builder.ins().store(
                MemFlags::trusted(),
                result,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        Thumb3Op::Sub => {
            let rd_val = builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
            let result = builder.ins().isub(rd_val, imm8);
            builder.ins().store(
                MemFlags::trusted(),
                result,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
    }
}

/// Emit a Thumb format 3 immediate8 instruction. Always updates NZCV.
/// MOV: Rd = imm8, sets N=0 (imm8 < 0x80 always clears top bit),
///      Z=(imm8==0), preserves C and V.
/// CMP: flags from Rd - imm8. No writeback.
/// ADD: Rd = Rd + imm8. Full NZCV.
/// SUB: Rd = Rd - imm8. Full NZCV.
fn emit_thumb_format3(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedThumb3,
) {
    let imm8 = builder.ins().iconst(types::I32, dec.imm8 as i64);

    // MOV imm8 fast path: imm8 ∈ [0, 255] so N is always 0, C/V are
    // preserved, Z is a compile-time constant (imm8==0). Skip the
    // general emit_flag_update + skip the rd_val load (MOV doesn't
    // read rd).
    if let Thumb3Op::Mov = dec.op {
        builder.ins().store(
            MemFlags::trusted(),
            imm8,
            gpr_ptr,
            Offset32::new(dec.rd * 4),
        );
        let cpsr = builder.use_var(cpsr_var);
        let cleared = builder.ins().band_imm(cpsr, 0x3fff_ffff);
        let new_cpsr = if dec.imm8 == 0 {
            builder.ins().bor_imm(cleared, 0x4000_0000_u32 as i64)
        } else {
            cleared
        };
        builder.def_var(cpsr_var, new_cpsr);
        return;
    }

    // For Thumb3 ADD/SUB/CMP, imm8 ∈ [0, 255] (always positive → sign
    // bit always 0). Specializing the V-flag formula:
    //   ADD V = ~rd & result & 0x8000_0000  (4 CLIF ops vs 6 general)
    //   SUB V = rd & ~result & 0x8000_0000  (4 CLIF ops vs 6 general)
    // Saves 2 ops per instruction on this hot shape.
    let rd_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rd * 4),
    );
    let zero = builder.ins().iconst(types::I32, 0);

    let (result, c_bool, v_shifted, writeback) = match dec.op {
        Thumb3Op::Mov => unreachable!(),
        Thumb3Op::Cmp | Thumb3Op::Sub => {
            let result = builder.ins().isub(rd_val, imm8);
            let c = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, rd_val, imm8);
            let not_result = builder.ins().bnot(result);
            let v_bits = builder.ins().band(rd_val, not_result);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            let v = builder.ins().ushr_imm(v_top, 3);
            (result, c, v, matches!(dec.op, Thumb3Op::Sub))
        }
        Thumb3Op::Add => {
            let result = builder.ins().iadd(rd_val, imm8);
            let c = builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, result, rd_val);
            let not_rd = builder.ins().bnot(rd_val);
            let v_bits = builder.ins().band(not_rd, result);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            let v = builder.ins().ushr_imm(v_top, 3);
            (result, c, v, true)
        }
    };

    if writeback {
        builder.ins().store(
            MemFlags::trusted(),
            result,
            gpr_ptr,
            Offset32::new(dec.rd * 4),
        );
    }

    // N/Z pack via the same icmp+uextend+ishl pattern emit_flag_update
    // uses on ADD/SUB (exp 4 confirmed select() regresses on x86_64).
    let n_shifted = builder.ins().band_imm(result, 0x8000_0000_u32 as i64);
    let z_bool = builder.ins().icmp(IntCC::Equal, result, zero);
    let z_u32 = builder.ins().uextend(types::I32, z_bool);
    let z_shifted = builder.ins().ishl_imm(z_u32, 30);
    let c_u32 = builder.ins().uextend(types::I32, c_bool);
    let c_shifted = builder.ins().ishl_imm(c_u32, 29);

    let cpsr = builder.use_var(cpsr_var);
    let cleared = builder.ins().band_imm(cpsr, 0x0fff_ffff);
    let nz = builder.ins().bor(n_shifted, z_shifted);
    let cv = builder.ins().bor(c_shifted, v_shifted);
    let flags = builder.ins().bor(nz, cv);
    let new_cpsr = builder.ins().bor(cleared, flags);
    builder.def_var(cpsr_var, new_cpsr);
}

/// Emit a conditionally executed LDR / STR immediate. Loads rn, adjusts by
/// the signed offset, then calls the right size trampoline (load_32 /
/// store_32 for word, load_8 / store_8 for byte). For LDR the returned value
/// is stored into gpr[rd]; for STR the value from gpr[rd] is passed out.
fn emit_conditional_mem(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_idle_8_ref: cranelift::codegen::ir::FuncRef,
    store_8_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedMem,
) {
    if dec.cond == ArmCond::Al {
        emit_mem_body(
            builder, gpr_ptr, cpu_ctx,
            load_idle_32_ref, store_32_ref, load_idle_8_ref, store_8_ref,
            dec,
        );
        return;
    }

    let cond_result = emit_cond_check(builder, cpsr_var, dec.cond);
    let body = builder.create_block();
    let merge = builder.create_block();
    builder.ins().brif(cond_result, body, &[], merge, &[]);
    builder.switch_to_block(body);
    builder.seal_block(body);
    emit_mem_body(
        builder, gpr_ptr, cpu_ctx,
        load_idle_32_ref, store_32_ref, load_idle_8_ref, store_8_ref,
        dec,
    );
    builder.ins().jump(merge, &[]);
    builder.switch_to_block(merge);
    builder.seal_block(merge);
}

/// Emit the address-compute + load/store sequence for an ARM single-data
/// transfer (LDR/STR/LDRB/STRB). LDR uses the *_with_idle trampoline so
/// the +1I idle cycle scalar adds is paid; STR uses the plain store_*.
fn emit_mem_body(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpu_ctx: Value,
    load_idle_32_ref: cranelift::codegen::ir::FuncRef,
    store_32_ref: cranelift::codegen::ir::FuncRef,
    load_idle_8_ref: cranelift::codegen::ir::FuncRef,
    store_8_ref: cranelift::codegen::ir::FuncRef,
    dec: DecodedMem,
) {
    // addr = gpr[rn] +/- imm12  (U bit from encoding picks add vs sub).
    let rn_val = builder.ins().load(
        types::I32,
        MemFlags::trusted(),
        gpr_ptr,
        Offset32::new(dec.rn * 4),
    );
    let offset = builder.ins().iconst(types::I32, dec.offset as i64);
    let addr = if dec.add {
        builder.ins().iadd(rn_val, offset)
    } else {
        builder.ins().isub(rn_val, offset)
    };

    match (dec.load, dec.byte) {
        (true, false) => {
            let call = builder.ins().call(load_idle_32_ref, &[cpu_ctx, addr]);
            let loaded = builder.inst_results(call)[0];
            builder.ins().store(
                MemFlags::trusted(),
                loaded,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        (true, true) => {
            // LDRB returns u32 zero extended via the u32 sig on the
            // trampoline. Mask to 8 bits on the trampoline side; here we
            // just store what comes back.
            let call = builder.ins().call(load_idle_8_ref, &[cpu_ctx, addr]);
            let loaded = builder.inst_results(call)[0];
            let byte_only = builder.ins().band_imm(loaded, 0xff);
            builder.ins().store(
                MemFlags::trusted(),
                byte_only,
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
        }
        (false, false) => {
            let rd_val = builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
            builder.ins().call(store_32_ref, &[cpu_ctx, addr, rd_val]);
        }
        (false, true) => {
            let rd_val = builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                gpr_ptr,
                Offset32::new(dec.rd * 4),
            );
            // STRB writes only the low byte. Trampoline is responsible for
            // truncation; we mask here too so the contract is explicit.
            let byte_val = builder.ins().band_imm(rd_val, 0xff);
            builder.ins().call(store_8_ref, &[cpu_ctx, addr, byte_val]);
        }
    }
}

/// Emit Cranelift IR for a single decoded data-processing instruction.
/// - Loads operand2 and (if applicable) rn from the gpr array.
/// - Computes the result.
/// - Writes the result back to gpr[rd] unless the op is compare-only.
/// - If S=1, updates the caller's cpsr variable with new NZCV flags.
fn emit_data_processing_imm(
    builder: &mut FunctionBuilder,
    gpr_ptr: Value,
    cpsr_var: Variable,
    dec: DecodedDp,
) {
    let op2 = match dec.operand2 {
        Operand2::Imm(v) => builder.ins().iconst(types::I32, v as i64),
        Operand2::Reg(rm) => builder.ins().load(
            types::I32,
            MemFlags::trusted(),
            gpr_ptr,
            Offset32::new(rm * 4),
        ),
    };

    // rn is only meaningful for ops other than MOV/MVN; loading it
    // unconditionally is safe (register file access is cheap, LLVM will
    // remove dead loads) but not worth the cycle if we can skip.
    let need_rn = !matches!(dec.op, DpOp::Mov | DpOp::Mvn);
    let rn = if need_rn {
        builder.ins().load(
            types::I32,
            MemFlags::trusted(),
            gpr_ptr,
            Offset32::new(dec.rn * 4),
        )
    } else {
        // Placeholder — unused.
        builder.ins().iconst(types::I32, 0)
    };

    let result = match dec.op {
        DpOp::Mov => op2,
        DpOp::Mvn => builder.ins().bnot(op2),
        DpOp::Add | DpOp::Cmn => builder.ins().iadd(rn, op2),
        DpOp::Sub | DpOp::Cmp => builder.ins().isub(rn, op2),
        DpOp::Tst => builder.ins().band(rn, op2),
        DpOp::Teq => builder.ins().bxor(rn, op2),
    };

    // Writeback (unless compare-only).
    if !dec.op.is_compare_only() {
        builder.ins().store(
            MemFlags::trusted(),
            result,
            gpr_ptr,
            Offset32::new(dec.rd * 4),
        );
    }

    // Flag update.
    if dec.s {
        let new_cpsr = emit_flag_update(builder, cpsr_var, dec.op, rn, op2, result);
        builder.def_var(cpsr_var, new_cpsr);
    }
}

/// Produce a new CPSR value with N/Z/C/V updated per the ARM rules for the
/// given opcode.
fn emit_flag_update(
    builder: &mut FunctionBuilder,
    cpsr_var: Variable,
    op: DpOp,
    rn: Value,
    op2: Value,
    result: Value,
) -> Value {
    let zero = builder.ins().iconst(types::I32, 0);

    // N at bit 31 directly: `result & 0x8000_0000`. Saves the
    // ushr_imm(31) + ishl_imm(31) round-trip that previously went
    // through bit 0.
    let n_shifted = builder
        .ins()
        .band_imm(result, 0x8000_0000_u32 as i64);

    // Z at bit 30: (result == 0) → i8 {0, 1} → uextend → shift up.
    let z_bool = builder.ins().icmp(IntCC::Equal, result, zero);
    let z_u32 = builder.ins().uextend(types::I32, z_bool);
    let z_shifted = builder.ins().ishl_imm(z_u32, 30);

    // C/V depend on op:
    //   ADD/CMN: C = unsigned carry-out; V = signed overflow.
    //   SUB/CMP: C = NOT borrow; V = signed overflow.
    //   Logical (TST/TEQ/MOV/MVN with S=1): C unchanged, V unchanged.
    //
    // For logical ops we skip the C/V compute AND the cpsr C/V
    // round-trip entirely: the merge mask at the bottom preserves
    // bits 29/28 in place when `preserve_cv` is true.
    let (c_shifted, v_shifted, preserve_cv) = match op {
        DpOp::Add | DpOp::Cmn => {
            let c_bool = builder.ins().icmp(IntCC::UnsignedLessThan, result, rn);
            let c_u32 = builder.ins().uextend(types::I32, c_bool);
            let c_shifted = builder.ins().ishl_imm(c_u32, 29);
            // V: (~(rn ^ op2) & (rn ^ result)) & 0x8000_0000, shifted to bit 28.
            let xor_ab = builder.ins().bxor(rn, op2);
            let nxor_ab = builder.ins().bnot(xor_ab);
            let xor_ar = builder.ins().bxor(rn, result);
            let v_bits = builder.ins().band(nxor_ab, xor_ar);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            let v_shifted = builder.ins().ushr_imm(v_top, 3);
            (c_shifted, v_shifted, false)
        }
        DpOp::Sub | DpOp::Cmp => {
            let c_bool = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, rn, op2);
            let c_u32 = builder.ins().uextend(types::I32, c_bool);
            let c_shifted = builder.ins().ishl_imm(c_u32, 29);
            let xor_ab = builder.ins().bxor(rn, op2);
            let xor_ar = builder.ins().bxor(rn, result);
            let v_bits = builder.ins().band(xor_ab, xor_ar);
            let v_top = builder.ins().band_imm(v_bits, 0x8000_0000_u32 as i64);
            let v_shifted = builder.ins().ushr_imm(v_top, 3);
            (c_shifted, v_shifted, false)
        }
        DpOp::Mov | DpOp::Mvn | DpOp::Tst | DpOp::Teq => {
            // Logical ops: leave C/V in place in cpsr. zero contributes
            // nothing to the OR below.
            (zero, zero, true)
        }
    };

    // Merge new NZCV into CPSR. For ADD/SUB/CMP/CMN we clear all four
    // top bits (0x0fff_ffff) and OR the new NZCV in. For MOV/MVN/TST/TEQ
    // we keep bits 29/28 (C/V) in place, so we only clear bits 31/30
    // (0x3fff_ffff) and OR in the new N/Z only.
    let cpsr = builder.use_var(cpsr_var);
    // band_imm folds the mask into a single arm64 `and` with an encoded
    // immediate, vs the two-op iconst+band sequence that previously
    // forced the mask into a register first. Semantics identical.
    let cleared = if preserve_cv {
        builder.ins().band_imm(cpsr, 0x3fff_ffffi64)
    } else {
        builder.ins().band_imm(cpsr, 0x0fff_ffffi64)
    };
    let nz = builder.ins().bor(n_shifted, z_shifted);
    if preserve_cv {
        builder.ins().bor(cleared, nz)
    } else {
        let cv = builder.ins().bor(c_shifted, v_shifted);
        let flags = builder.ins().bor(nz, cv);
        builder.ins().bor(cleared, flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Differential helpers moved to pub test_utils so integration tests
    // (tests/dynarec_pattern_differential.rs) can call them without
    // duplicating the interpreter-setup boilerplate.
    use crate::dynarec::test_utils::{differential, differential_with_flags};

    #[test]
    fn differential_cmp_various() {
        // CMP R0, #5 with R0 = {3, 5, 10, 0, 0xFFFFFFFF, 0x80000000}
        let block = [0xE350_0005u32];
        for &r0 in &[3u32, 5, 10, 0, 0xFFFFFFFFu32, 0x80000000u32] {
            let mut gpr = [0u32; 15];
            gpr[0] = r0;
            differential_with_flags(&block, gpr, 0);
        }
    }

    #[test]
    fn differential_adds_overflow_cases() {
        // ADDS R0, R0, #1 at interesting boundary values.
        let block = [0xE290_0001u32];
        for &r0 in &[
            0u32,
            1,
            0x7FFF_FFFF, // signed overflow
            0xFFFF_FFFF, // unsigned overflow/wrap
            0x8000_0000,
            0xFFFF_FFFE,
        ] {
            let mut gpr = [0u32; 15];
            gpr[0] = r0;
            differential_with_flags(&block, gpr, 0);
        }
    }

    #[test]
    fn differential_subs_borrow_cases() {
        // SUBS R0, R0, #1
        let block = [0xE250_0001u32];
        for &r0 in &[
            0u32,
            1,
            0x8000_0000, // signed overflow via subtract
            0x7FFF_FFFF,
            0xFFFF_FFFF,
        ] {
            let mut gpr = [0u32; 15];
            gpr[0] = r0;
            differential_with_flags(&block, gpr, 0);
        }
    }

    #[test]
    fn differential_tst_teq() {
        // TST R0, #0xFF; TEQ R0, #0xFF
        for opcode in &[0xE310_00FFu32, 0xE330_00FFu32] {
            for &r0 in &[0u32, 1, 0xFF, 0x100, 0xFFFF_FFFF, 0x8000_0000] {
                let mut gpr = [0u32; 15];
                gpr[0] = r0;
                differential_with_flags(&[*opcode], gpr, 0);
            }
        }
    }

    #[test]
    fn differential_mov_imm_sequence() {
        // Three MOVs with different Rd, different immediates.
        let block = [0xE3A0_0001u32, 0xE3A0_1002u32, 0xE3A0_20FFu32];
        differential(&block, [0; 15]);
    }

    #[test]
    fn differential_arithmetic_chain() {
        // MOV R0, #10; ADD R1, R0, #5; SUB R2, R1, #3; MVN R3, #0
        let block = [
            0xE3A0_000Au32,
            0xE280_1005u32,
            0xE241_2003u32,
            0xE3E0_3000u32,
        ];
        differential(&block, [0; 15]);
    }

    #[test]
    fn differential_register_form_chain() {
        // MOV R0, #10; MOV R1, #20; ADD R2, R0, R1; SUB R3, R1, R0
        let block = [
            0xE3A0_000Au32,
            0xE3A0_1014u32,
            0xE080_2001u32,
            0xE041_3000u32,
        ];
        differential(&block, [0; 15]);
    }

    #[test]
    fn differential_nonzero_initial_state() {
        // ADD R2, R0, R1 with gpr[0]=100, gpr[1]=50 → gpr[2]=150
        let block = [0xE080_2001u32];
        let mut initial = [0u32; 15];
        initial[0] = 100;
        initial[1] = 50;
        differential(&block, initial);
    }


    #[test]
    fn compile_and_run_mov_stub() {
        let mut compiler = DynarecCompiler::new();
        let func = compiler.compile_mov_r0_r1_stub();

        let mut gpr = [0u32; 15];
        gpr[1] = 0xDEAD_BEEF;
        func(gpr.as_mut_ptr());
        assert_eq!(gpr[0], 0xDEAD_BEEF);
        // Verify we didn't scribble anywhere else.
        assert_eq!(gpr[2], 0);
        assert_eq!(gpr[14], 0);
    }

    #[test]
    fn compile_real_mov_imm_sequence() {
        let mut compiler = DynarecCompiler::new();
        // Encode three MOV immediates:
        //   E3A0_0001  MOV R0, #1
        //   E3A0_1002  MOV R1, #2
        //   E3A0_20FF  MOV R2, #255
        let block = [0xE3A0_0001u32, 0xE3A0_1002u32, 0xE3A0_20FFu32];
        let func = compiler.try_compile_imm_block(&block).expect("should compile");

        let mut gpr = [0u32; 15];
        // Pre-poison to prove the writes actually happen.
        for v in gpr.iter_mut() {
            *v = 0xFFFF_FFFF;
        }
        { let mut cpsr = 0u32; func(gpr.as_mut_ptr(), &mut cpsr); }
        assert_eq!(gpr[0], 1);
        assert_eq!(gpr[1], 2);
        assert_eq!(gpr[2], 255);
        // Unchanged registers keep their poison.
        assert_eq!(gpr[3], 0xFFFF_FFFF);
    }

    #[test]
    fn compile_register_form_add() {
        let mut compiler = DynarecCompiler::new();
        // Block:
        //   MOV R0, #10         E3A0_000A
        //   MOV R1, #20         E3A0_1014
        //   ADD R2, R0, R1      E080_2001   (register-form, no shift)
        //   SUB R3, R1, R0      E041_3000   (register-form, no shift)
        let block = [
            0xE3A0_000Au32,
            0xE3A0_1014u32,
            0xE080_2001u32,
            0xE041_3000u32,
        ];
        let func = compiler
            .try_compile_imm_block(&block)
            .expect("should compile");

        let mut gpr = [0u32; 15];
        { let mut cpsr = 0u32; func(gpr.as_mut_ptr(), &mut cpsr); }
        assert_eq!(gpr[0], 10);
        assert_eq!(gpr[1], 20);
        assert_eq!(gpr[2], 30);
        assert_eq!(gpr[3], 10);
    }

    #[test]
    fn reject_register_form_with_shift() {
        let mut compiler = DynarecCompiler::new();
        // ADD R0, R1, R2, LSL #1 — shift nonzero, should not compile yet.
        let block = [0xE081_0082u32];
        assert!(compiler.try_compile_imm_block(&block).is_none());
    }

    #[test]
    fn compile_add_sub_mvn_mixed() {
        let mut compiler = DynarecCompiler::new();
        // Block:
        //   MOV R0, #10         E3A0_000A
        //   ADD R1, R0, #5      E280_1005
        //   SUB R2, R1, #3      E241_2003
        //   MVN R3, #0          E3E0_3000   (result = 0xFFFF_FFFF)
        let block = [
            0xE3A0_000Au32,
            0xE280_1005u32,
            0xE241_2003u32,
            0xE3E0_3000u32,
        ];
        let func = compiler.try_compile_imm_block(&block).expect("should compile");

        let mut gpr = [0u32; 15];
        { let mut cpsr = 0u32; func(gpr.as_mut_ptr(), &mut cpsr); }
        assert_eq!(gpr[0], 10);
        assert_eq!(gpr[1], 15);
        assert_eq!(gpr[2], 12);
        assert_eq!(gpr[3], 0xFFFF_FFFF);
    }

    #[test]
    fn reject_unsupported_opcode_returns_none() {
        let mut compiler = DynarecCompiler::new();
        // An LDR opcode (not data-processing-immediate). Should be rejected.
        let block = [0xE590_0000u32];
        assert!(compiler.try_compile_imm_block(&block).is_none());

        // NV (0xF) condition is reserved in ARMv4 and should be rejected.
        let block = [0xF3A0_0001u32];
        assert!(compiler.try_compile_imm_block(&block).is_none());
    }

    #[test]
    fn cmp_sets_zero_flag() {
        let mut compiler = DynarecCompiler::new();
        // CMP R0, #5    E350_0005
        let block = [0xE350_0005u32];
        let func = compiler.try_compile_imm_block(&block).expect("CMP supported");

        // R0 == 5 → Z=1, N=0, C=1 (no borrow), V=0
        let mut gpr = [0u32; 15];
        gpr[0] = 5;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!((cpsr >> 30) & 1, 1, "Z should be set for equal");
        assert_eq!((cpsr >> 31) & 1, 0, "N should be clear");
        assert_eq!((cpsr >> 29) & 1, 1, "C should be set (no borrow)");
        assert_eq!((cpsr >> 28) & 1, 0, "V should be clear");
        assert_eq!(gpr[0], 5, "CMP must not write Rd");
    }

    #[test]
    fn cmp_sets_negative_flag() {
        let mut compiler = DynarecCompiler::new();
        // CMP R0, #5   R0=3 → result = -2, N=1, Z=0, C=0 (borrow)
        let block = [0xE350_0005u32];
        let func = compiler.try_compile_imm_block(&block).expect("CMP supported");

        let mut gpr = [0u32; 15];
        gpr[0] = 3;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!((cpsr >> 31) & 1, 1, "N should be set");
        assert_eq!((cpsr >> 30) & 1, 0, "Z should be clear");
        assert_eq!((cpsr >> 29) & 1, 0, "C should be clear (borrow)");
    }

    #[test]
    fn tst_sets_zero_on_no_overlap() {
        let mut compiler = DynarecCompiler::new();
        // TST R0, #0x0F   E310_000F
        let block = [0xE310_000Fu32];
        let func = compiler.try_compile_imm_block(&block).expect("TST supported");

        // R0 = 0xF0 → R0 AND 0x0F = 0 → Z=1
        let mut gpr = [0u32; 15];
        gpr[0] = 0xF0;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!((cpsr >> 30) & 1, 1);
        assert_eq!(gpr[0], 0xF0, "TST must not write Rd");

        // R0 = 0xF1 → AND = 1 → Z=0, N=0
        let mut gpr = [0u32; 15];
        gpr[0] = 0xF1;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!((cpsr >> 30) & 1, 0);
        assert_eq!((cpsr >> 31) & 1, 0);
    }

    #[test]
    fn adds_sets_all_flags_correctly() {
        let mut compiler = DynarecCompiler::new();
        // ADDS R0, R0, #1   E290_0001
        let block = [0xE290_0001u32];
        let func = compiler.try_compile_imm_block(&block).expect("ADDS supported");

        // 0xFFFF_FFFF + 1 = 0 → Z=1, C=1 (carry), N=0, V=0
        let mut gpr = [0u32; 15];
        gpr[0] = 0xFFFF_FFFF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0);
        assert_eq!((cpsr >> 30) & 1, 1, "Z");
        assert_eq!((cpsr >> 29) & 1, 1, "C");
        assert_eq!((cpsr >> 31) & 1, 0, "N");
        assert_eq!((cpsr >> 28) & 1, 0, "V");

        // 0x7FFF_FFFF + 1 = 0x8000_0000 → N=1, V=1 (signed overflow)
        let mut gpr = [0u32; 15];
        gpr[0] = 0x7FFF_FFFF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0x8000_0000);
        assert_eq!((cpsr >> 31) & 1, 1, "N");
        assert_eq!((cpsr >> 28) & 1, 1, "V");
        assert_eq!((cpsr >> 29) & 1, 0, "C");
    }

    #[test]
    fn moveq_with_z_flag_set_writes_register() {
        let mut compiler = DynarecCompiler::new();
        // MOVEQ R0, #42    →  03A0_002A
        let block = [0x03A0_002Au32];
        let func = compiler
            .try_compile_imm_block(&block)
            .expect("MOVEQ is supported");

        // CPSR with Z=1 → condition passes → write happens
        let mut gpr = [0u32; 15];
        let mut cpsr: u32 = 1 << 30;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 42);

        // CPSR with Z=0 → condition fails → gpr unchanged
        let mut gpr = [7u32; 15];
        let mut cpsr: u32 = 0;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 7);
    }

    #[test]
    fn movne_complements_moveq() {
        let mut compiler = DynarecCompiler::new();
        // MOVNE R0, #99     →  13A0_0063
        let block = [0x13A0_0063u32];
        let func = compiler
            .try_compile_imm_block(&block)
            .expect("MOVNE is supported");

        // CPSR with Z=0 (not equal) → write happens
        let mut gpr = [0u32; 15];
        let mut cpsr: u32 = 0;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 99);

        // CPSR with Z=1 → condition fails → no write
        let mut gpr = [3u32; 15];
        let mut cpsr: u32 = 1 << 30;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 3);
    }

    #[test]
    fn compile_mov_imm_with_rotation() {
        let mut compiler = DynarecCompiler::new();
        // MOV R3, #0xFF000000 encodes as E3A0_34FF:
        //   imm8 = 0xFF, rot = 4 → rotate right 8 bits → 0xFF000000
        let block = [0xE3A0_34FFu32];
        let func = compiler.try_compile_imm_block(&block).expect("should compile");

        let mut gpr = [0u32; 15];
        { let mut cpsr = 0u32; func(gpr.as_mut_ptr(), &mut cpsr); }
        assert_eq!(gpr[3], 0xFF00_0000);
    }

    #[test]
    fn two_independent_compilations_coexist() {
        let mut compiler = DynarecCompiler::new();
        let f1 = compiler.compile_mov_r0_r1_stub();
        let f2 = compiler.compile_mov_r0_r1_stub();

        let mut gpr_a = [0u32; 15];
        gpr_a[1] = 11;
        f1(gpr_a.as_mut_ptr());

        let mut gpr_b = [0u32; 15];
        gpr_b[1] = 22;
        f2(gpr_b.as_mut_ptr());

        assert_eq!(gpr_a[0], 11);
        assert_eq!(gpr_b[0], 22);
    }

    #[test]
    fn decode_unconditional_b_forward() {
        // B #+8 (skip the next instruction). ARM encoding:
        //   cond=AL, opcode 1010, imm24 = 0 means target = pc+8+0 = pc+8.
        // E.g. "EA FF FF FE" is B -8 (infinite loop); for a +8 jump use
        // imm24 = 0.
        //   EA 00 00 00  = B pc+8 (= next+4, i.e. skip one instr)
        let insn = 0xEA00_0000u32;
        let dec = DynarecCompiler::decode_branch(insn)
            .expect("should be a branch");
        assert_eq!(dec.cond, ArmCond::Al);
        assert_eq!(dec.link, false);
        assert_eq!(dec.offset24_signed, 0);

        // BL #-8  cond=AL, opcode 1011, imm24 = 0x_FF_FF_FE (sign-ext'd to
        // -2), target = pc+8 + (-2)*4 = pc.
        let insn = 0xEBFF_FFFEu32;
        let dec = DynarecCompiler::decode_branch(insn)
            .expect("should be a branch");
        assert_eq!(dec.link, true);
        assert_eq!(dec.offset24_signed, -2);
    }

    #[test]
    fn decode_branch_rejects_non_branch() {
        // MOV R0, #5 isn't a branch.
        assert!(DynarecCompiler::decode_branch(0xE3A0_0005u32).is_none());
        // NV (0xF) cond reserved / never-taken, reject.
        assert!(DynarecCompiler::decode_branch(0xFA00_0000u32).is_none());
    }

    #[test]
    fn compile_b_forward_taken() {
        // Block: MOV R0, #1; B +8. Both always taken (AL cond).
        let mut compiler = DynarecCompiler::new();
        let mov = 0xE3A0_0001u32;        // MOV R0, #1
        let b_plus8 = 0xEA00_0000u32;    // B +8 (target pc+8 from branch site)
        let entry_pc: u32 = 0x0800_1000;
        let func = compiler
            .try_compile_block_with_branch(&[mov, b_plus8], entry_pc)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out: u32 = 0xDEAD_BEEF;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);

        assert_eq!(gpr[0], 1, "MOV ran");
        assert_eq!(taken, 1, "branch should report taken");
        // B at entry_pc+4. Target = (entry_pc+4) + 8 + 0 = entry_pc + 12.
        assert_eq!(pc_out, entry_pc.wrapping_add(12));
    }

    #[test]
    fn compile_bl_sets_lr() {
        // BL +8 -> LR = pc_of_BL + 4, target = pc_of_BL + 8.
        let mut compiler = DynarecCompiler::new();
        let bl = 0xEB00_0000u32;
        let entry_pc: u32 = 0x0800_2000;
        let func = compiler
            .try_compile_block_with_branch(&[bl], entry_pc)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);

        assert_eq!(taken, 1);
        assert_eq!(gpr[14], entry_pc.wrapping_add(4), "LR = PC_of_BL + 4");
        assert_eq!(pc_out, entry_pc.wrapping_add(8));
    }

    #[test]
    fn compile_conditional_branch_not_taken() {
        // BEQ +8 with Z=0 (not equal) -> branch not taken, block falls
        // through.
        let mut compiler = DynarecCompiler::new();
        let beq = 0x0A00_0000u32; // cond=EQ, B opcode, imm24=0
        let func = compiler
            .try_compile_block_with_branch(&[beq], 0x0800_3000)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32; // Z=0
        let mut pc_out: u32 = 0xBADD_F00D;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);

        assert_eq!(taken, 0, "BEQ should not fire when Z=0");
        // pc_out must be left alone.
        assert_eq!(pc_out, 0xBADD_F00D);
    }

    #[test]
    fn compile_conditional_branch_taken_when_z_set() {
        let mut compiler = DynarecCompiler::new();
        let beq = 0x0A00_0000u32;
        let entry_pc: u32 = 0x0800_4000;
        let func = compiler
            .try_compile_block_with_branch(&[beq], entry_pc)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr: u32 = 1 << 30; // Z=1
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);

        assert_eq!(taken, 1);
        assert_eq!(pc_out, entry_pc.wrapping_add(8));
    }

    #[test]
    fn branch_body_rejects_unsupported_midblock() {
        // Body contains an unsupported opcode (LDR): should be None.
        let ldr_placeholder = 0xE5101000u32; // LDR R1, [R0]
        let b = 0xEA00_0000u32;
        let mut compiler = DynarecCompiler::new();
        assert!(
            compiler
                .try_compile_block_with_branch(&[ldr_placeholder, b], 0)
                .is_none()
        );
    }

    // --- Bus trampoline plumbing ---
    //
    // These tests exercise the Rust -> Cranelift -> Rust round trip used by
    // the upcoming LDR/STR codegen without decoding any ARM memory ops yet.
    // They depend on a test-only trampoline pair plus a tiny 256 byte fake
    // memory buffer.

    use std::cell::UnsafeCell;

    /// Fake memory used by the bus trampoline tests. `UnsafeCell` so the
    /// `extern "C"` callbacks can mutate it through a raw pointer without
    /// tripping the borrow checker. Safe in tests because each test runs
    /// single threaded and constructs its own buffer.
    struct TestBus {
        bytes: UnsafeCell<[u8; 256]>,
    }

    unsafe extern "C" fn test_load_32(ctx: *mut u8, addr: u32) -> u32 {
        let bus = &*(ctx as *const TestBus);
        let bytes = &*bus.bytes.get();
        let a = addr as usize & 0xFC;
        u32::from_le_bytes([bytes[a], bytes[a + 1], bytes[a + 2], bytes[a + 3]])
    }
    unsafe extern "C" fn test_store_32(ctx: *mut u8, addr: u32, val: u32) {
        let bus = &*(ctx as *const TestBus);
        let bytes = &mut *bus.bytes.get();
        let a = addr as usize & 0xFC;
        let v = val.to_le_bytes();
        bytes[a] = v[0]; bytes[a + 1] = v[1]; bytes[a + 2] = v[2]; bytes[a + 3] = v[3];
    }
    unsafe extern "C" fn test_load_8(ctx: *mut u8, addr: u32) -> u32 {
        let bus = &*(ctx as *const TestBus);
        let bytes = &*bus.bytes.get();
        bytes[addr as usize & 0xFF] as u32
    }
    unsafe extern "C" fn test_store_8(ctx: *mut u8, addr: u32, val: u32) {
        let bus = &*(ctx as *const TestBus);
        let bytes = &mut *bus.bytes.get();
        bytes[addr as usize & 0xFF] = val as u8;
    }
    /// Noop stand-in for the fetch-cycle trampoline. Real integration
    /// supplies `trampolines::thumb_fetch_n::<I>` which requires ctx to
    /// be a real Arm7tdmiCore; these unit tests use a TestBus so the
    /// stub intentionally does nothing.
    unsafe extern "C" fn test_thumb_fetch_n(_ctx: *mut u8, _pc: u32, _count: u32) {}
    /// Noop stand-in for the post-block NonSeq fixup. The real integration
    /// updates `cpu.next_fetch_access`; these unit tests don't have a real
    /// Arm7tdmiCore so the stub is empty.
    unsafe extern "C" fn test_set_next_fetch_nonseq(_ctx: *mut u8) {}
    /// Stand-ins for the +1I LDR variants. These tests check codegen
    /// shape, not cycle accounting, so they just forward to the plain
    /// no-idle variants.
    unsafe extern "C" fn test_load_with_idle_32(ctx: *mut u8, addr: u32) -> u32 {
        unsafe { test_load_32(ctx, addr) }
    }
    unsafe extern "C" fn test_load_with_idle_8(ctx: *mut u8, addr: u32) -> u32 {
        unsafe { test_load_8(ctx, addr) }
    }
    /// No-op stand-in. SimpleMemory / TestBus have uniform fetch cost
    /// (no LUT), so the SysBus override doesn't apply here.
    unsafe extern "C" fn test_pay_thumb_fetch_extra_nonseq(_ctx: *mut u8, _pc: u32) {}
    /// No-op idle trampoline stub; these tests don't observe scheduler state.
    unsafe extern "C" fn test_idle_cycle(_ctx: *mut u8) {}
    /// Chain-abort stub. Tests pass `chain_slot: None` so no chain
    /// tail-call gets emitted (the null-check short-circuits); but
    /// the trampoline is ALSO called by the mid-block-abort codegen
    /// at every even body position. Returning 0 ("no abort") lets
    /// compiled test blocks run to completion. The old "return 1"
    /// was fine before the mid-block-abort codegen existed and now
    /// short-circuits everything.
    unsafe extern "C" fn test_chain_abort_check(_ctx: *mut u8) -> u32 { 0 }
    unsafe extern "C" fn test_abort_mid_block(
        _ctx: *mut u8, _pc: u32, _p0: u32, _p1: u32,
    ) {}

    fn test_trampolines() -> BusTrampolines {
        BusTrampolines {
            load_32: test_load_32,
            store_32: test_store_32,
            load_8: test_load_8,
            store_8: test_store_8,
            load_with_idle_32: test_load_with_idle_32,
            load_with_idle_8: test_load_with_idle_8,
            set_next_fetch_nonseq: test_set_next_fetch_nonseq,
            pay_thumb_fetch_extra_nonseq: test_pay_thumb_fetch_extra_nonseq,
            idle_cycle: test_idle_cycle,
            thumb_fetch_n: test_thumb_fetch_n,
            chain_abort_check: test_chain_abort_check,
            abort_mid_block: test_abort_mid_block,
            load_32_seq: test_load_32,
            store_32_seq: test_store_32,
            store_16: test_store_32,
            load_with_idle_16: test_load_with_idle_32,
        }
    }

    #[test]
    fn bus_stub_round_trip_load_32() {
        let bus = TestBus { bytes: UnsafeCell::new([0u8; 256]) };
        // Seed a known value at offset 0x10.
        unsafe {
            let bytes = &mut *bus.bytes.get();
            bytes[0x10] = 0x11;
            bytes[0x11] = 0x22;
            bytes[0x12] = 0x33;
            bytes[0x13] = 0x44;
        }

        let mut compiler = DynarecCompiler::new_with_bus(test_trampolines());
        assert!(compiler.has_bus());
        let stub = compiler.compile_bus_load_32_stub();

        let mut gpr = [0u32; 15];
        stub(gpr.as_mut_ptr(), &bus as *const TestBus as *mut u8, 0x10);
        assert_eq!(gpr[0], 0x44332211, "round trip through bus trampoline");
    }

    #[test]
    fn plain_compiler_has_no_bus() {
        let c = DynarecCompiler::new();
        assert!(!c.has_bus());
    }

    // --- ARM LDR / STR immediate codegen ---

    fn new_bus_and_compiler() -> (TestBus, DynarecCompiler) {
        (
            TestBus { bytes: UnsafeCell::new([0u8; 256]) },
            DynarecCompiler::new_with_bus(test_trampolines()),
        )
    }

    #[test]
    fn decode_ldr_str_immediate_shapes() {
        // LDR R1, [R0, #4]  ->  E5_90_10_04
        let ldr = 0xE590_1004u32;
        let d = DynarecCompiler::decode_mem_immediate(ldr).expect("LDR");
        assert_eq!(d.cond, ArmCond::Al);
        assert_eq!(d.load, true);
        assert_eq!(d.byte, false);
        assert_eq!(d.add, true);
        assert_eq!(d.rn, 0);
        assert_eq!(d.rd, 1);
        assert_eq!(d.offset, 4);

        // STR R2, [R3, #0x20]  ->  E5_83_20_20
        let str_ = 0xE583_2020u32;
        let d = DynarecCompiler::decode_mem_immediate(str_).expect("STR");
        assert_eq!(d.load, false);
        assert_eq!(d.byte, false);
        assert_eq!(d.rn, 3);
        assert_eq!(d.rd, 2);

        // LDRB R1, [R0, #4]  B=1  ->  E5_D0_10_04
        let ldrb = 0xE5D0_1004u32;
        let d = DynarecCompiler::decode_mem_immediate(ldrb).expect("LDRB");
        assert_eq!(d.load, true);
        assert_eq!(d.byte, true);
        assert_eq!(d.offset, 4);

        // STRB R1, [R0, #4]  B=1,L=0  ->  E5_C0_10_04
        let strb = 0xE5C0_1004u32;
        let d = DynarecCompiler::decode_mem_immediate(strb).expect("STRB");
        assert_eq!(d.load, false);
        assert_eq!(d.byte, true);

        // LDR with negative offset (U=0) accepted now.
        //   E5_10_10_04
        let ldr_neg = 0xE510_1004u32;
        let d = DynarecCompiler::decode_mem_immediate(ldr_neg).expect("LDR -ve");
        assert_eq!(d.add, false);
        assert_eq!(d.offset, 4);

        // Writeback (W=1) still rejected.
        let ldr_wb = 0xE5B0_1004u32;
        assert!(DynarecCompiler::decode_mem_immediate(ldr_wb).is_none());

        // Post indexed (P=0) still rejected.
        let ldr_post = 0xE490_1004u32;
        assert!(DynarecCompiler::decode_mem_immediate(ldr_post).is_none());
    }

    #[test]
    fn compile_ldrb_reads_byte_zero_extended() {
        let (bus, mut compiler) = new_bus_and_compiler();
        unsafe {
            let b = &mut *bus.bytes.get();
            b[0x40] = 0xAB;
        }
        // LDRB R1, [R0, #0x40]
        let ldrb = 0xE5D0_1040u32;
        let func = compiler
            .try_compile_mem_block(&[ldrb])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[1] = 0xDEADBEEF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0x0000_00ABu32, "LDRB zero extends");
    }

    #[test]
    fn compile_strb_truncates_to_byte() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // STRB R2, [R0, #0x50]
        let strb = 0xE5C0_2050u32;
        let func = compiler
            .try_compile_mem_block(&[strb])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        gpr[2] = 0x1234_5678;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(b[0x50], 0x78, "low byte only");
            assert_eq!(b[0x51], 0, "neighbour untouched");
        }
    }

    // --- Thumb format 3 (MOV/CMP/ADD/SUB Rd, #imm8) ---

    #[test]
    fn decode_thumb_format3_shapes() {
        // MOV R0, #0x42  -> 0010 0 000 0100 0010 = 0x2042
        let d = DynarecCompiler::decode_thumb_format3(0x2042).expect("MOV");
        assert_eq!(d.op, Thumb3Op::Mov);
        assert_eq!(d.rd, 0);
        assert_eq!(d.imm8, 0x42);

        // CMP R3, #0x10 -> 0010 1 011 0001 0000 = 0x2B10
        let d = DynarecCompiler::decode_thumb_format3(0x2B10).expect("CMP");
        assert_eq!(d.op, Thumb3Op::Cmp);
        assert_eq!(d.rd, 3);
        assert_eq!(d.imm8, 0x10);

        // ADD R5, #1 -> 0011 0 101 0000 0001 = 0x3501
        let d = DynarecCompiler::decode_thumb_format3(0x3501).expect("ADD");
        assert_eq!(d.op, Thumb3Op::Add);
        assert_eq!(d.rd, 5);

        // SUB R1, #5 -> 0011 1 001 0000 0101 = 0x3905
        let d = DynarecCompiler::decode_thumb_format3(0x3905).expect("SUB");
        assert_eq!(d.op, Thumb3Op::Sub);

        // Not format 3 (top 3 bits != 001)
        assert!(DynarecCompiler::decode_thumb_format3(0x4000).is_none());
        assert!(DynarecCompiler::decode_thumb_format3(0xE000).is_none());
    }

    #[test]
    fn compile_thumb_mov_imm8() {
        let mut compiler = DynarecCompiler::new();
        let mov_r1_42 = 0x2142u16; // MOV R1, #0x42
        let func = compiler
            .try_compile_thumb_block(&[mov_r1_42])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[1], 0x42);
        // Z flag clear (result nonzero), N clear.
        assert_eq!(cpsr & (1 << 30), 0);
        assert_eq!(cpsr & (1 << 31), 0);
    }

    #[test]
    fn compile_thumb_mov_imm8_zero_sets_z() {
        let mut compiler = DynarecCompiler::new();
        let mov_r2_0 = 0x2200u16;
        let func = compiler.try_compile_thumb_block(&[mov_r2_0]).unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[2], 0);
        assert_ne!(cpsr & (1 << 30), 0, "Z set for 0");
    }

    #[test]
    fn compile_thumb_cmp_sets_flags_no_writeback() {
        let mut compiler = DynarecCompiler::new();
        // CMP R0, #5
        let cmp = 0x2805u16;
        let func = compiler.try_compile_thumb_block(&[cmp]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[0] = 5;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 5, "CMP must not writeback");
        assert_ne!(cpsr & (1 << 30), 0, "Z should be set (5 - 5 == 0)");
    }

    #[test]
    fn compile_thumb_add_sub_sequence() {
        let mut compiler = DynarecCompiler::new();
        // MOV R0, #10; ADD R0, #5; SUB R0, #3
        let mov = 0x200Au16;
        let add = 0x3005u16;
        let sub = 0x3803u16;
        let func = compiler.try_compile_thumb_block(&[mov, add, sub]).unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 12);
        // Result nonzero, nonnegative -> Z=0, N=0.
        assert_eq!(cpsr & (1 << 30), 0);
        assert_eq!(cpsr & (1 << 31), 0);
    }

    #[test]
    fn compile_thumb_reject_unsupported() {
        let mut compiler = DynarecCompiler::new();
        // 0x4400 is Thumb format 5 (ADD Hi reg) - not in any currently
        // supported shape.
        assert!(compiler.try_compile_thumb_block(&[0x4400]).is_none());
    }

    // --- Thumb format 2 (ADD/SUB Rd, Rs, Rn|imm3) ---

    #[test]
    fn decode_thumb_format2_shapes() {
        // ADD R0, R1, R2  -> 00011_0_0_010_001_000 = 0b0001_1000_1000_1000 = 0x1888
        let d = DynarecCompiler::decode_thumb_format2(0x1888).expect("ADD reg");
        assert_eq!(d.sub, false);
        assert_eq!(d.rd, 0);
        assert_eq!(d.rs, 1);
        matches!(d.operand, Thumb2Operand::Reg(2));

        // SUB R3, R4, R5 -> 00011_0_1_101_100_011 = 0b0001_1011_0110_0011 = 0x1B63
        let d = DynarecCompiler::decode_thumb_format2(0x1B63).expect("SUB reg");
        assert_eq!(d.sub, true);
        assert_eq!(d.rd, 3);
        assert_eq!(d.rs, 4);
        matches!(d.operand, Thumb2Operand::Reg(5));

        // ADD R0, R1, #7 -> 00011_1_0_111_001_000 = 0b0001_1101_1100_1000 = 0x1DC8
        let d = DynarecCompiler::decode_thumb_format2(0x1DC8).expect("ADD imm3");
        assert_eq!(d.sub, false);
        matches!(d.operand, Thumb2Operand::Imm3(7));

        // SUB R2, R2, #1 -> 00011_1_1_001_010_010 = 0b0001_1111_0101_0010 = 0x1F52
        let d = DynarecCompiler::decode_thumb_format2(0x1F52).expect("SUB imm3");
        assert_eq!(d.sub, true);
        matches!(d.operand, Thumb2Operand::Imm3(1));

        // Not format 2 (top 5 bits != 00011)
        assert!(DynarecCompiler::decode_thumb_format2(0x2000).is_none()); // format 3
        assert!(DynarecCompiler::decode_thumb_format2(0x4000).is_none()); // format 4
    }

    #[test]
    fn compile_thumb_add_reg() {
        let mut compiler = DynarecCompiler::new();
        // ADD R0, R1, R2
        let add_reg = 0x1888u16;
        let func = compiler.try_compile_thumb_block(&[add_reg]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 10;
        gpr[2] = 20;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 30);
        assert_eq!(cpsr & (1 << 30), 0);  // Z clear
    }

    #[test]
    fn compile_thumb_sub_imm3_sets_z_on_zero() {
        let mut compiler = DynarecCompiler::new();
        // SUB R0, R0, #3 where R0 = 3
        let sub = 0x1EC0u16; // 00011_11_011_000_000 = 0b0001_1110_1100_0000 = 0x1EC0
        let func = compiler.try_compile_thumb_block(&[sub]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[0] = 3;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0);
        assert_ne!(cpsr & (1 << 30), 0, "Z should be set");
    }

    // --- Generic trampolines for real CPU ---

    #[test]
    fn thumb_fetch_n_trampoline_matches_interpreter_fetch_sequence() {
        use crate::SimpleMemory;
        use crate::cpu::Arm7tdmiCore;
        use crate::memory::{MemoryAccess, MemoryInterface};
        use rustboyadvance_utils::Shared;

        // Lay out four Thumb instructions in memory at addresses 0, 2, 4, 6.
        let mut mem = SimpleMemory::new(256);
        let program = vec![
            0x11u8, 0x22,   // word at 0
            0x33, 0x44,     // word at 2
            0x55, 0x66,     // word at 4
            0x77, 0x88,     // word at 6
        ];
        mem.load_program(&program);
        let m = Shared::new(mem);
        let mut cpu = Arm7tdmiCore::new(m.clone());

        // Simulate block entry state: pipeline[0] already holds the first
        // instruction (0x2211), pipeline[1] holds the second (0x4433),
        // pc points at the third instruction (0x0004 = A_addr + 4 for
        // A_addr = 0).
        cpu.pipeline[0] = 0x2211;
        cpu.pipeline[1] = 0x4433;
        cpu.pc = 4;
        cpu.next_fetch_access = MemoryAccess::Seq;

        // Run a compiled block of count = 2 instructions.
        // first_fetch_pc = A_addr + 4 = 4 (same as current pc).
        unsafe {
            super::trampolines::thumb_fetch_n::<SimpleMemory>(
                &mut cpu as *mut _ as *mut u8,
                4,
                2,
            );
        }

        // After two iterations of the interpreter loop:
        //   pipeline[0] = fetched at iter 1 = mem[4] = 0x6655
        //   pipeline[1] = fetched at iter 2 = mem[6] = 0x8877
        //   pc = first_fetch_pc + 2*count = 4 + 4 = 8
        assert_eq!(cpu.pipeline[0], 0x6655);
        assert_eq!(cpu.pipeline[1], 0x8877);
        assert_eq!(cpu.pc, 8);
        assert!(matches!(cpu.next_fetch_access, MemoryAccess::Seq));
    }

    #[test]
    fn thumb_fetch_n_with_count_one_preserves_old_pipeline_1() {
        use crate::SimpleMemory;
        use crate::cpu::Arm7tdmiCore;
        use crate::memory::{MemoryAccess, MemoryInterface};
        use rustboyadvance_utils::Shared;

        let mut mem = SimpleMemory::new(256);
        mem.load_program(&vec![0x11, 0x22, 0x33, 0x44]);
        let m = Shared::new(mem);
        let mut cpu = Arm7tdmiCore::new(m.clone());

        cpu.pipeline[0] = 0xAAAA;
        cpu.pipeline[1] = 0xBBBB;
        cpu.pc = 0;
        cpu.next_fetch_access = MemoryAccess::Seq;

        unsafe {
            super::trampolines::thumb_fetch_n::<SimpleMemory>(
                &mut cpu as *mut _ as *mut u8,
                0,
                1,
            );
        }

        // For count=1: pipeline[0] takes the OLD pipeline[1] value,
        // pipeline[1] takes the one newly fetched word (from pc=0).
        assert_eq!(cpu.pipeline[0], 0xBBBB, "old pipeline[1] shifted down");
        assert_eq!(cpu.pipeline[1], 0x2211, "freshly fetched low word at addr 0");
    }

    #[test]
    fn trampolines_for_simple_memory_round_trip() {
        use crate::SimpleMemory;
        use crate::cpu::Arm7tdmiCore;
        use crate::memory::{MemoryAccess, MemoryInterface};
        use rustboyadvance_utils::Shared;

        // Seed a known u32 at offset 0x20 of SimpleMemory via the CPU's
        // bus store API, then read it back through a dynarec compiled
        // LDR that goes through the trampoline.
        let mem = SimpleMemory::new(1024);
        let m = Shared::new(mem);
        let mut cpu = Arm7tdmiCore::new(m.clone());
        MemoryInterface::store_8(&mut cpu, 0x20, 0x11, MemoryAccess::NonSeq);
        MemoryInterface::store_8(&mut cpu, 0x21, 0x22, MemoryAccess::NonSeq);
        MemoryInterface::store_8(&mut cpu, 0x22, 0x33, MemoryAccess::NonSeq);
        MemoryInterface::store_8(&mut cpu, 0x23, 0x44, MemoryAccess::NonSeq);

        let mut compiler =
            DynarecCompiler::new_with_bus(trampolines::for_cpu::<SimpleMemory>());
        // LDR R1, [R0, #0x20]  -> E5_90_10_20
        let func = compiler
            .try_compile_mem_block(&[0xE590_1020u32])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        let mut cpsr = 0u32;
        let cpu_ctx = &mut cpu as *mut Arm7tdmiCore<SimpleMemory> as *mut u8;
        func(gpr.as_mut_ptr(), &mut cpsr, cpu_ctx);
        assert_eq!(gpr[1], 0x4433_2211);
    }

    /// MemoryInterface stub that counts every cycle-bearing event the
    /// dynarec trampolines could trigger: `load_*`, `store_*`, and
    /// `idle_cycle`. Lets us assert the +1I behavior of the *_with_idle
    /// trampolines without having to plug in the full SysBus + Scheduler.
    struct CycleCountingMem {
        loads: u32,
        stores: u32,
        idles: u32,
    }
    impl CycleCountingMem {
        fn new() -> Self {
            Self { loads: 0, stores: 0, idles: 0 }
        }
    }
    impl crate::memory::MemoryInterface for CycleCountingMem {
        fn load_8(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u8 { self.loads += 1; 0 }
        fn load_16(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u16 { self.loads += 1; 0 }
        fn load_32(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u32 { self.loads += 1; 0 }
        fn store_8(&mut self, _: u32, _: u8, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn store_16(&mut self, _: u32, _: u16, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn store_32(&mut self, _: u32, _: u32, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn idle_cycle(&mut self) { self.idles += 1; }
    }

    /// MemoryInterface stub that ALSO counts pay_thumb_fetch_extra_nonseq
    /// calls so we can verify the codegen emits the right number of
    /// post-store compensation calls. Returns a fixed delta of 2 per
    /// call so we can also see in the cycle stream whether the call
    /// landed.
    struct CycleCountingMemWithExtra {
        loads: u32,
        stores: u32,
        idles: u32,
        extras: u32,
    }
    impl CycleCountingMemWithExtra {
        fn new() -> Self {
            Self { loads: 0, stores: 0, idles: 0, extras: 0 }
        }
    }
    impl crate::memory::MemoryInterface for CycleCountingMemWithExtra {
        fn load_8(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u8 { self.loads += 1; 0 }
        fn load_16(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u16 { self.loads += 1; 0 }
        fn load_32(&mut self, _: u32, _: crate::memory::MemoryAccess) -> u32 { self.loads += 1; 0 }
        fn store_8(&mut self, _: u32, _: u8, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn store_16(&mut self, _: u32, _: u16, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn store_32(&mut self, _: u32, _: u32, _: crate::memory::MemoryAccess) { self.stores += 1; }
        fn idle_cycle(&mut self) { self.idles += 1; }
        fn pay_thumb_fetch_extra_nonseq(&mut self, _addr: u32) { self.extras += 1; }
    }

    /// Codegen check for the in-block STR NonSeq compensation: a Thumb
    /// block of [STR, MOV, MOV, BX] should call
    /// `pay_thumb_fetch_extra_nonseq` exactly once (one intermediate
    /// store followed by non-store body items + a branch terminator).
    /// A block of [MOV, MOV, BX] (no stores) should call it zero times.
    #[test]
    fn pay_thumb_fetch_extra_nonseq_emitted_once_per_intermediate_store() {
        use crate::cpu::Arm7tdmiCore;
        use rustboyadvance_utils::Shared;

        // STR R0, [R1] = 0x6008 (Thumb format 9, store word, offset 0)
        // MOV R2, #0   = 0x2200
        // BX  LR       = 0x4770
        let opcodes = [0x6008, 0x2200, 0x2200, 0x4770];

        let m = Shared::new(CycleCountingMemWithExtra::new());
        let mut cpu = Arm7tdmiCore::new(m);
        let mut compiler = DynarecCompiler::new_with_bus(
            super::trampolines::for_cpu::<CycleCountingMemWithExtra>(),
        );
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&opcodes, 0x0800_0000, None)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[14] = 0x0800_1235; // BX LR target
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let _ = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &mut cpu as *mut _ as *mut u8,
        );
        assert_eq!(
            cpu.bus.extras, 1,
            "expected exactly 1 pay_thumb_fetch_extra_nonseq call for the \
             single STR in body[0]; got {}",
            cpu.bus.extras
        );
        assert_eq!(cpu.bus.stores, 1);
    }

    #[test]
    fn pay_thumb_fetch_extra_nonseq_not_emitted_when_no_stores() {
        use crate::cpu::Arm7tdmiCore;
        use rustboyadvance_utils::Shared;

        let opcodes = [0x2001, 0x2002, 0x4770]; // MOV, MOV, BX LR

        let m = Shared::new(CycleCountingMemWithExtra::new());
        let mut cpu = Arm7tdmiCore::new(m);
        let mut compiler = DynarecCompiler::new_with_bus(
            super::trampolines::for_cpu::<CycleCountingMemWithExtra>(),
        );
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&opcodes, 0x0800_0000, None)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[14] = 0;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let _ = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &mut cpu as *mut _ as *mut u8,
        );
        assert_eq!(cpu.bus.extras, 0, "no stores -> no extra-nonseq calls");
    }

    /// Catches the cycle accounting bug the PR #200 reviewer flagged:
    /// scalar Thumb LDR charges +1I after the data fetch, but the
    /// dynarec was calling the plain `load_32` trampoline which did
    /// not. With the fix, `load_with_idle_32` calls `cpu.idle_cycle()`
    /// after the read so the cycle counts match scalar.
    #[test]
    fn load_with_idle_32_invokes_idle_cycle() {
        use crate::cpu::Arm7tdmiCore;
        use rustboyadvance_utils::Shared;

        // Plain load_32: 1 load, 0 idles.
        let m = Shared::new(CycleCountingMem::new());
        let mut cpu = Arm7tdmiCore::new(m.clone());
        unsafe {
            super::trampolines::load_32::<CycleCountingMem>(
                &mut cpu as *mut _ as *mut u8,
                0x20,
            );
        }
        assert_eq!(cpu.bus.loads, 1);
        assert_eq!(cpu.bus.idles, 0, "plain load_32 must not idle");

        // load_with_idle_32: 1 load, 1 idle.
        let m2 = Shared::new(CycleCountingMem::new());
        let mut cpu2 = Arm7tdmiCore::new(m2.clone());
        unsafe {
            super::trampolines::load_with_idle_32::<CycleCountingMem>(
                &mut cpu2 as *mut _ as *mut u8,
                0x20,
            );
        }
        assert_eq!(cpu2.bus.loads, 1);
        assert_eq!(cpu2.bus.idles, 1, "load_with_idle_32 must charge +1I");
    }

    /// Same shape for load_with_idle_8 (LDRB).
    #[test]
    fn load_with_idle_8_invokes_idle_cycle() {
        use crate::cpu::Arm7tdmiCore;
        use rustboyadvance_utils::Shared;

        let m = Shared::new(CycleCountingMem::new());
        let mut cpu = Arm7tdmiCore::new(m.clone());
        unsafe {
            super::trampolines::load_with_idle_8::<CycleCountingMem>(
                &mut cpu as *mut _ as *mut u8,
                0x10,
            );
        }
        assert_eq!(cpu.bus.loads, 1);
        assert_eq!(cpu.bus.idles, 1);
    }

    /// Catches the second leak: a compiled block whose last instruction
    /// is a STORE must leave `cpu.next_fetch_access = NonSeq` so the
    /// post-block fetch in the cached interp loop pays NonSeq cycles.
    /// The trampoline that does this is `set_next_fetch_nonseq`.
    #[test]
    fn set_next_fetch_nonseq_trampoline_flips_access_mode() {
        use crate::SimpleMemory;
        use crate::cpu::Arm7tdmiCore;
        use crate::memory::MemoryAccess;
        use rustboyadvance_utils::Shared;

        let m = Shared::new(SimpleMemory::new(64));
        let mut cpu = Arm7tdmiCore::new(m);
        cpu.next_fetch_access = MemoryAccess::Seq;
        unsafe {
            super::trampolines::set_next_fetch_nonseq::<SimpleMemory>(
                &mut cpu as *mut _ as *mut u8,
            );
        }
        assert!(matches!(cpu.next_fetch_access, MemoryAccess::NonSeq));
    }

    // --- Unified Thumb mem+branch block compiler ---

    #[test]
    fn unified_thumb_fall_through_body_only() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // MOV R0, #5  (no branch, no memory)
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&[0x2005], 0x0800_0000, None)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0xBADC0FFEu32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 0);
        assert_eq!(gpr[0], 5);
        assert_eq!(pc_out, 0xBADC0FFEu32, "no branch, pc_out untouched");
    }

    #[test]
    fn unified_thumb_bx_terminator() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // BX LR  -> 0x4770
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&[0x4770], 0x0800_0000, None)
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[14] = 0x0800_1235;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_1235);
    }

    #[test]
    fn unified_thumb_pop_pc_terminator() {
        let (bus, mut compiler) = new_bus_and_compiler();
        unsafe {
            let b = &mut *bus.bytes.get();
            let pc = 0x0800_5555u32.to_le_bytes();
            b[0x40..0x44].copy_from_slice(&pc);
        }
        // POP {PC}
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&[0xBD00], 0, None)
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x40;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_5555);
        assert_eq!(gpr[13], 0x44);
    }

    #[test]
    fn unified_thumb_mem_body_plus_pop_pc_epilogue() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // Prepare epilogue: stack has R4, PC at SP..SP+8
        unsafe {
            let b = &mut *bus.bytes.get();
            let r4 = 0xCAFEu32.to_le_bytes();
            let pc = 0x0800_1011u32.to_le_bytes();
            b[0x50..0x54].copy_from_slice(&r4);
            b[0x54..0x58].copy_from_slice(&pc);
        }
        // MOV R0, #1 ; STR R0, [SP, #0x10] ; POP {R4, PC}
        //   MOV R0, #1 = 0x2001
        //   STR R0, [SP, #0x10] -> imm8=4 -> 1001_0_000_00000100 = 0x9004
        //   POP {R4, PC}        -> reg_list=0x10 -> 1011_1_10_1_00010000 = 0xBD10
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(
                &[0x2001, 0x9004, 0xBD10],
                0x0800_3000,
                None,
            )
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[13] = 0x50;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_1011);
        assert_eq!(gpr[4], 0xCAFE);
        assert_eq!(gpr[13], 0x58, "SP += 8 (1 reg + PC)");
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(u32::from_le_bytes([b[0x60], b[0x61], b[0x62], b[0x63]]), 1,
                       "STR wrote R0=1 to SP+0x10=0x60");
        }
    }

    #[test]
    fn unified_thumb_bcc_not_taken_keeps_going() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // BEQ +4 with Z=0 -> not taken, pc_out untouched, return 0.
        let func = compiler
            .try_compile_thumb_mem_block_with_branch(&[0xD002], 0x0800_4000, None)
            .unwrap();
        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0xC0FF_EE00u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 0);
        assert_eq!(pc_out, 0xC0FF_EE00);
    }

    #[test]
    fn unified_thumb_rejects_mid_block_terminator() {
        let (_bus, mut compiler) = new_bus_and_compiler();
        // BX followed by MOV -> BX not in last slot -> reject.
        assert!(compiler
            .try_compile_thumb_mem_block_with_branch(&[0x4770, 0x2001], 0, None)
            .is_none());
        // POP{PC} in middle, same.
        assert!(compiler
            .try_compile_thumb_mem_block_with_branch(&[0xBD00, 0x2001], 0, None)
            .is_none());
    }

    #[test]
    fn unified_thumb_requires_bus() {
        let mut compiler = DynarecCompiler::new();
        assert!(compiler
            .try_compile_thumb_mem_block_with_branch(&[0x2005], 0, None)
            .is_none());
    }

    // --- Thumb POP {regs, pc} terminator ---

    #[test]
    fn decode_thumb_pop_pc_shapes() {
        // POP {PC} only -> 1011_1_10_1_00000000 = 0xBD00
        let d = DynarecCompiler::decode_thumb_pop_pc(0xBD00).expect("POP {PC}");
        assert_eq!(d.push, false);
        assert_eq!(d.extra_reg, true);
        assert_eq!(d.reg_list, 0);

        // POP {R4-R7, PC} -> 1011_1_10_1_11110000 = 0xBDF0
        let d = DynarecCompiler::decode_thumb_pop_pc(0xBDF0).expect("POP {r,pc}");
        assert_eq!(d.reg_list, 0xF0);
        assert_eq!(d.count(), 5);

        // PUSH forms rejected.
        assert!(DynarecCompiler::decode_thumb_pop_pc(0xB500).is_none());
        // POP without R (not a terminator) rejected.
        assert!(DynarecCompiler::decode_thumb_pop_pc(0xBC01).is_none());
    }

    #[test]
    fn compile_thumb_pop_pc_only() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // Stage a PC value on the stack.
        unsafe {
            let b = &mut *bus.bytes.get();
            // write 0x0800_1235 (Thumb-marked) at SP
            let v = 0x0800_1235u32.to_le_bytes();
            b[0x40] = v[0]; b[0x41] = v[1]; b[0x42] = v[2]; b[0x43] = v[3];
        }
        // POP {PC}
        let func = compiler.try_compile_thumb_pop_pc(0xBD00).unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x40;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_1235, "Thumb bit preserved");
        assert_eq!(gpr[13], 0x44, "SP advanced by 4");
    }

    #[test]
    fn compile_thumb_pop_regs_and_pc() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // Stage 3 words: R4, R5, PC  at SP..SP+8
        unsafe {
            let b = &mut *bus.bytes.get();
            let r4 = 0xAAu32.to_le_bytes();
            let r5 = 0xBBu32.to_le_bytes();
            let pc = 0x0800_CAFEu32.to_le_bytes();
            b[0x30..0x34].copy_from_slice(&r4);
            b[0x34..0x38].copy_from_slice(&r5);
            b[0x38..0x3C].copy_from_slice(&pc);
        }
        // POP {R4, R5, PC}  reg_list = 0b0011_0000 = 0x30
        //   1011_1_10_1_00110000 = 0xBD30
        let func = compiler.try_compile_thumb_pop_pc(0xBD30).unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x30;
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(
            gpr.as_mut_ptr(), &mut cpsr, &mut pc_out,
            &bus as *const TestBus as *mut u8,
        );
        assert_eq!(taken, 1);
        assert_eq!(gpr[4], 0xAA);
        assert_eq!(gpr[5], 0xBB);
        assert_eq!(pc_out, 0x0800_CAFE);
        assert_eq!(gpr[13], 0x3C, "SP += 12");
    }

    #[test]
    fn compile_thumb_pop_pc_requires_bus() {
        let mut compiler = DynarecCompiler::new();
        assert!(compiler.try_compile_thumb_pop_pc(0xBD00).is_none());
    }

    // --- Thumb format 14 PUSH/POP (non PC variant) ---

    #[test]
    fn decode_thumb_format14_shapes() {
        // PUSH {R0} -> 1011_0_10_0_00000001 = 0b1011_0100_0000_0001 = 0xB401
        let d = DynarecCompiler::decode_thumb_format14_non_pc(0xB401).expect("PUSH R0");
        assert_eq!(d.push, true);
        assert_eq!(d.extra_reg, false);
        assert_eq!(d.reg_list, 0x01);

        // PUSH {R0-R3, LR} -> 1011_0_10_1_00001111 = 0xB50F
        let d = DynarecCompiler::decode_thumb_format14_non_pc(0xB50F).expect("PUSH r,lr");
        assert_eq!(d.push, true);
        assert_eq!(d.extra_reg, true);
        assert_eq!(d.reg_list, 0x0F);
        assert_eq!(d.count(), 5);

        // POP {R4-R7} -> 1011_1_10_0_11110000 = 0xBCF0
        let d = DynarecCompiler::decode_thumb_format14_non_pc(0xBCF0).expect("POP");
        assert_eq!(d.push, false);
        assert_eq!(d.reg_list, 0xF0);

        // POP {R4-R7, PC} must be rejected here (deferred to branch path).
        //   1011_1_10_1_11110000 = 0xBDF0
        assert!(DynarecCompiler::decode_thumb_format14_non_pc(0xBDF0).is_none());

        // Empty list rejection: PUSH {} with R=0.
        //   1011_0_10_0_00000000 = 0xB400
        assert!(DynarecCompiler::decode_thumb_format14_non_pc(0xB400).is_none());

        // Not format 14.
        assert!(DynarecCompiler::decode_thumb_format14_non_pc(0x6000).is_none());
    }

    #[test]
    fn compile_thumb_push_single_reg() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // PUSH {R0}
        let push = 0xB401u16;
        let func = compiler.try_compile_thumb_mem_block(&[push]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x40; // SP
        gpr[0]  = 0x1234_5678;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[13], 0x40 - 4, "SP decremented by 4");
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(u32::from_le_bytes([b[0x3C], b[0x3D], b[0x3E], b[0x3F]]),
                       0x1234_5678);
        }
    }

    #[test]
    fn compile_thumb_push_multiple_with_lr() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // PUSH {R0, R2, LR}
        //   1011_0_10_1_00000101 = 0xB505
        let push = 0xB505u16;
        let func = compiler.try_compile_thumb_mem_block(&[push]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x80;
        gpr[0]  = 0xAA;
        gpr[2]  = 0xBB;
        gpr[14] = 0xCC;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[13], 0x80 - 12, "SP -= 12 for 3 regs");
        unsafe {
            let b = &*bus.bytes.get();
            // R0 at lowest, then R2, then LR
            assert_eq!(b[0x74], 0xAA, "R0 at SP-12");
            assert_eq!(b[0x78], 0xBB, "R2 at SP-8");
            assert_eq!(b[0x7C], 0xCC, "LR at SP-4");
        }
    }

    #[test]
    fn compile_thumb_push_then_pop_roundtrip() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // PUSH {R0, R1}; POP {R2, R3}
        //   1011_0_10_0_00000011 = 0xB403
        //   1011_1_10_0_00001100 = 0xBC0C
        let push = 0xB403u16;
        let pop  = 0xBC0Cu16;
        let func = compiler.try_compile_thumb_mem_block(&[push, pop]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[13] = 0x60;
        gpr[0]  = 0xDEAD;
        gpr[1]  = 0xBEEF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[2], 0xDEAD, "R2 = pushed R0");
        assert_eq!(gpr[3], 0xBEEF, "R3 = pushed R1");
        assert_eq!(gpr[13], 0x60, "SP restored");
    }

    // --- Thumb format 11 SP-relative LDR/STR ---

    #[test]
    fn decode_thumb_format11_shapes() {
        // STR R0, [SP, #4]  imm8=1 scales to 4
        //   1001_0_000_00000001 = 0b1001_0000_0000_0001 = 0x9001
        let d = DynarecCompiler::decode_thumb_format11(0x9001).expect("STR");
        assert_eq!(d.load, false);
        assert_eq!(d.rd, 0);
        assert_eq!(d.offset, 4);

        // LDR R3, [SP, #0x100]  imm8=0x40 scales to 0x100
        //   1001_1_011_01000000 = 0b1001_1011_0100_0000 = 0x9B40
        let d = DynarecCompiler::decode_thumb_format11(0x9B40).expect("LDR");
        assert_eq!(d.load, true);
        assert_eq!(d.rd, 3);
        assert_eq!(d.offset, 0x100);

        // Not format 11.
        assert!(DynarecCompiler::decode_thumb_format11(0x6000).is_none());
    }

    #[test]
    fn compile_thumb_sp_relative_store_then_load() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // Simulated stack base at offset 0x40 in the 256 byte bus buffer.
        // STR R1, [SP, #0]; LDR R2, [SP, #0]
        let str_ = 0x9100u16; // 1001_0_001_00000000
        let ldr  = 0x9A00u16; // 1001_1_010_00000000
        let func = compiler
            .try_compile_thumb_mem_block(&[str_, ldr])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[13] = 0x40;
        gpr[1] = 0xCAFE_BABE;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[2], 0xCAFE_BABE);
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(u32::from_le_bytes([b[0x40], b[0x41], b[0x42], b[0x43]]),
                       0xCAFE_BABE);
        }
    }

    #[test]
    fn compile_thumb_sp_relative_with_offset() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // STR R0, [SP, #8]; LDR R1, [SP, #8]
        let str_ = 0x9002u16; // imm8=2 -> offset 8
        let ldr  = 0x9902u16;
        let func = compiler
            .try_compile_thumb_mem_block(&[str_, ldr])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[13] = 0x30;
        gpr[0]  = 0x1234_5678;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0x1234_5678);
    }

    // --- Thumb format 9 LDR/STR immediate offset ---

    #[test]
    fn decode_thumb_format9_shapes() {
        // STR R0, [R1, #4]  word, offset imm5 = 1 -> scaled to 4
        //   011_0_0_00001_001_000 = 0b0110_0000_0100_1000 = 0x6048
        let d = DynarecCompiler::decode_thumb_format9(0x6048).expect("STR word");
        assert_eq!(d.load, false);
        assert_eq!(d.byte, false);
        assert_eq!(d.offset, 4);
        assert_eq!(d.rs, 1);
        assert_eq!(d.rd, 0);

        // LDR R2, [R3, #0x1C]  imm5 = 7 -> scaled to 28
        //   011_0_1_00111_011_010 = 0b0110_1001_1101_1010 = 0x69DA
        let d = DynarecCompiler::decode_thumb_format9(0x69DA).expect("LDR word");
        assert_eq!(d.load, true);
        assert_eq!(d.offset, 28);

        // STRB R0, [R1, #3] byte offset 3
        //   011_1_0_00011_001_000 = 0b0111_0000_1100_1000 = 0x70C8
        let d = DynarecCompiler::decode_thumb_format9(0x70C8).expect("STRB");
        assert_eq!(d.byte, true);
        assert_eq!(d.offset, 3);

        // LDRB R2, [R3, #1]  011_1_1_00001_011_010 = 0b0111_1000_0101_1010 = 0x785A
        let d = DynarecCompiler::decode_thumb_format9(0x785A).expect("LDRB");
        assert_eq!(d.load, true);
        assert_eq!(d.byte, true);

        // Not format 9 (top 3 bits != 011)
        assert!(DynarecCompiler::decode_thumb_format9(0x2000).is_none());
    }

    #[test]
    fn compile_thumb_ldr_word_immediate() {
        let (bus, mut compiler) = new_bus_and_compiler();
        unsafe {
            let b = &mut *bus.bytes.get();
            b[0x14] = 0x11; b[0x15] = 0x22; b[0x16] = 0x33; b[0x17] = 0x44;
        }
        // LDR R1, [R0, #0x14] -> imm5 = 5
        //   011_0_1_00101_000_001 = 0b0110_1001_0100_0001 = 0x6941
        let ldr = 0x6941u16;
        let func = compiler
            .try_compile_thumb_mem_block(&[ldr])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0x4433_2211);
    }

    #[test]
    fn compile_thumb_strb_truncates() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // STRB R2, [R0, #5]  byte offset 5
        //   011_1_0_00101_000_010 = 0b0111_0001_0100_0010 = 0x7142
        let strb = 0x7142u16;
        let func = compiler.try_compile_thumb_mem_block(&[strb]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        gpr[2] = 0xDEAD_BEEF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(b[5], 0xEF);
            assert_eq!(b[6], 0, "neighbour untouched");
        }
    }

    #[test]
    fn compile_thumb_mem_mixes_with_dp() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // MOV R2, #0x42 ; STR R2, [R0, #8] ; LDR R3, [R0, #8] ; ADD R4, R3, #1
        let mov = 0x2242u16;
        // STR word offset=8 means imm5=2  011_0_0_00010_000_010 = 0x6082
        let str_ = 0x6082u16;
        // LDR word offset=8 imm5=2        011_0_1_00010_000_011 = 0x6883
        let ldr = 0x6883u16;
        // ADD R4, R3, #1 format 2 imm3    0001_1_1_0_001_011_100 = 0x1C5C
        let add = 0x1C5Cu16;
        let func = compiler
            .try_compile_thumb_mem_block(&[mov, str_, ldr, add])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[2], 0x42);
        assert_eq!(gpr[3], 0x42);
        assert_eq!(gpr[4], 0x43);
    }

    #[test]
    fn thumb_mem_block_requires_bus() {
        let mut compiler = DynarecCompiler::new();
        let ldr = 0x6941u16;
        assert!(compiler.try_compile_thumb_mem_block(&[ldr]).is_none());
    }

    // --- Thumb BX as block terminator ---

    #[test]
    fn decode_thumb_bx_shapes() {
        // BX R1  -> 010001_11_0_0_001_000 = 0b0100_0111_0000_1000 = 0x4708
        let d = DynarecCompiler::decode_thumb_bx(0x4708).expect("BX R1");
        assert_eq!(d.rs, 1);

        // BX R14 (LR)  -> 010001_11_0_1_110_000 = 0b0100_0111_0111_0000 = 0x4770
        let d = DynarecCompiler::decode_thumb_bx(0x4770).expect("BX LR");
        assert_eq!(d.rs, 14);

        // H1 set is SBZ violation -> reject.
        //   010001_11_1_0_001_000 = 0b0100_0111_1000_1000 = 0x4788
        assert!(DynarecCompiler::decode_thumb_bx(0x4788).is_none());

        // Low 3 bits nonzero is SBZ violation -> reject.
        assert!(DynarecCompiler::decode_thumb_bx(0x4709).is_none());

        // Not format 5 BX.
        assert!(DynarecCompiler::decode_thumb_bx(0x2000).is_none());
        assert!(DynarecCompiler::decode_thumb_bx(0x4488).is_none()); // ADD Hi
    }

    #[test]
    fn compile_thumb_bx_writes_target_and_returns_1() {
        let mut compiler = DynarecCompiler::new();
        // BX R1 (target in R1).
        let func = compiler
            .try_compile_thumb_block_with_branch(&[0x4708], 0x0800_0000)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[1] = 0x0800_1234; // target, bit 0 = 0 -> ARM mode
        let mut cpsr = 0u32;
        let mut pc_out = 0xDEAD_BEEFu32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_1234);
    }

    #[test]
    fn compile_thumb_bx_preserves_thumb_bit() {
        let mut compiler = DynarecCompiler::new();
        // BX R2
        //   010001_11_0_0_010_000 = 0x4710
        let func = compiler
            .try_compile_thumb_block_with_branch(&[0x4710], 0)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[2] = 0x0800_1235; // bit 0 = 1 -> Thumb mode
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 1);
        assert_eq!(pc_out & 1, 1, "Thumb bit preserved");
        assert_eq!(pc_out & !1, 0x0800_1234);
    }

    #[test]
    fn compile_thumb_block_with_body_and_bx_tail() {
        let mut compiler = DynarecCompiler::new();
        // MOV R0, #5 ; ADD R1, R0, #3 ; BX LR
        let mov = 0x2005u16;        // fmt 3
        let add_imm3 = 0x1CC1u16;   // ADD R1, R0, #3 -> 00011_10_011_000_001 = 0b0001_1100_1100_0001
        let bx_lr = 0x4770u16;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[mov, add_imm3, bx_lr], 0x0800_2000)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[14] = 0x0800_3001; // LR: bit 0 set -> Thumb
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(gpr[0], 5);
        assert_eq!(gpr[1], 8);
        assert_eq!(taken, 1);
        assert_eq!(pc_out, 0x0800_3001);
    }

    #[test]
    fn compile_thumb_block_no_bx_returns_0() {
        let mut compiler = DynarecCompiler::new();
        // Body only, no BX. Should return 0 and leave pc_out alone.
        let mov = 0x2042u16;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[mov], 0)
            .unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0xC0FF_EE00u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 0);
        assert_eq!(pc_out, 0xC0FF_EE00);
        assert_eq!(gpr[0], 0x42);
    }

    #[test]
    fn decode_thumb_format18_unconditional() {
        // B #+8 (forward). imm11 offset from instr+4. imm11 = 0 means target
        // = instr+4+0 = instr+4 (skip no instrs).
        //   11100_00000000000 = 0xE000
        let d = DynarecCompiler::decode_thumb_format18(0xE000).expect("B +0");
        assert_eq!(d.cond, ArmCond::Al);
        assert_eq!(d.offset_signed, 0);

        // Forward 4 instructions: imm11 = 4 -> 0xE004
        let d = DynarecCompiler::decode_thumb_format18(0xE004).unwrap();
        assert_eq!(d.offset_signed, 4);

        // Backward: imm11 = -1 -> 0xE7FF
        let d = DynarecCompiler::decode_thumb_format18(0xE7FF).unwrap();
        assert_eq!(d.offset_signed, -1);

        // Not format 18.
        assert!(DynarecCompiler::decode_thumb_format18(0xE800).is_none());
        assert!(DynarecCompiler::decode_thumb_format18(0x2000).is_none());
    }

    #[test]
    fn decode_thumb_format16_conditional() {
        // BEQ #+4 (cond=0x0, imm8=2 -> target = pc+4+4)
        //   1101_0000_00000010 = 0xD002
        let d = DynarecCompiler::decode_thumb_format16(0xD002).expect("BEQ");
        assert_eq!(d.cond, ArmCond::Eq);
        assert_eq!(d.offset_signed, 2);

        // BNE -4: cond=0x1, imm8 = -2 -> 0xD1FE
        let d = DynarecCompiler::decode_thumb_format16(0xD1FE).unwrap();
        assert_eq!(d.cond, ArmCond::Ne);
        assert_eq!(d.offset_signed, -2);

        // cond AL (0xE) is reserved for format 18, reject here.
        assert!(DynarecCompiler::decode_thumb_format16(0xDE00).is_none());
        // cond 0xF is SWI (format 17), reject.
        assert!(DynarecCompiler::decode_thumb_format16(0xDF00).is_none());
        // Not format 16 at all.
        assert!(DynarecCompiler::decode_thumb_format16(0xC000).is_none());
    }

    #[test]
    fn compile_thumb_format18_taken() {
        let mut compiler = DynarecCompiler::new();
        // B +0: target = branch_pc + 4 + 0, with Thumb bit.
        let b = 0xE000u16;
        let entry_pc = 0x0800_1000u32;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[b], entry_pc)
            .unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 1);
        // branch_pc = entry_pc (body_len = 0). target = entry_pc + 4 | 1.
        assert_eq!(pc_out, (entry_pc + 4) | 1);
    }

    #[test]
    fn compile_thumb_format16_not_taken() {
        let mut compiler = DynarecCompiler::new();
        // BEQ #+4 with Z=0.
        let beq = 0xD002u16;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[beq], 0x0800_2000)
            .unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32; // Z=0
        let mut pc_out = 0xBADD_F00Du32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 0);
        assert_eq!(pc_out, 0xBADD_F00Du32);
    }

    #[test]
    fn compile_thumb_format16_taken_when_z_set() {
        let mut compiler = DynarecCompiler::new();
        let beq = 0xD002u16;
        let entry_pc = 0x0800_3000u32;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[beq], entry_pc)
            .unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr: u32 = 1 << 30;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(taken, 1);
        assert_eq!(pc_out, (entry_pc + 4 + 4) | 1);
    }

    #[test]
    fn compile_thumb_body_then_b_tail() {
        let mut compiler = DynarecCompiler::new();
        // MOV R0, #1 ; B -0
        //   MOV R0, #1 -> 0x2001
        //   B -0 means backward to self (imm11 = -2 -> 0xE7FE)
        let mov = 0x2001u16;
        let b   = 0xE7FEu16;
        let entry_pc = 0x0800_4000u32;
        let func = compiler
            .try_compile_thumb_block_with_branch(&[mov, b], entry_pc)
            .unwrap();

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out = 0u32;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);
        assert_eq!(gpr[0], 1);
        assert_eq!(taken, 1);
        // branch_pc = entry_pc + 2 (one body instr). target = branch_pc + 4 + (-2 << 1) = branch_pc.
        // That's entry_pc + 2. With Thumb bit = (entry_pc + 2) | 1.
        assert_eq!(pc_out, (entry_pc + 2) | 1);
    }

    #[test]
    fn compile_thumb_mid_block_pc_branch_rejected() {
        let mut compiler = DynarecCompiler::new();
        let b  = 0xE000u16; // format 18
        let mov = 0x2042u16;
        assert!(compiler
            .try_compile_thumb_block_with_branch(&[b, mov], 0)
            .is_none());
    }

    #[test]
    fn compile_thumb_mid_block_bx_rejected() {
        let mut compiler = DynarecCompiler::new();
        // BX can only be in the last slot.
        let bx = 0x4708u16;
        let mov = 0x2005u16;
        assert!(compiler
            .try_compile_thumb_block_with_branch(&[bx, mov], 0)
            .is_none());
    }

    // --- Thumb format 5 (ADD/CMP/MOV Hi) ---

    #[test]
    fn decode_thumb_format5_shapes() {
        // ADD R8, R0, R1  (H1=1, H2=0, Rd=0, Rs=1)
        //   010001_00_10_001_000 = 0b0100_0100_1000_1000 = 0x4488
        let d = DynarecCompiler::decode_thumb_format5_non_branch(0x4488).expect("ADD Hi");
        assert_eq!(d.op, Thumb5Op::Add);
        assert_eq!(d.rd, 8);
        assert_eq!(d.rs, 1);

        // MOV R9, R0  (H1=1, H2=0, op=10)
        //   010001_10_10_000_001 = 0b0100_0110_1000_0001 = 0x4681
        let d = DynarecCompiler::decode_thumb_format5_non_branch(0x4681).expect("MOV Hi");
        assert_eq!(d.op, Thumb5Op::Mov);
        assert_eq!(d.rd, 9);
        assert_eq!(d.rs, 0);

        // CMP R10, R11  (H1=1, H2=1, op=01)
        //   010001_01_11_011_010 = 0b0100_0101_1101_1010 = 0x45DA
        let d = DynarecCompiler::decode_thumb_format5_non_branch(0x45DA).expect("CMP Hi");
        assert_eq!(d.op, Thumb5Op::Cmp);
        assert_eq!(d.rd, 10);
        assert_eq!(d.rs, 11);

        // BX encoding (oo=11) must reject here.
        //   010001_11_00_001_000 = 0b0100_0111_0000_1000 = 0x4708  BX R1
        assert!(DynarecCompiler::decode_thumb_format5_non_branch(0x4708).is_none());

        // H1=0, H2=0, op=ADD is UNPREDICTABLE -> reject.
        //   010001_00_00_001_000 = 0b0100_0100_0000_1000 = 0x4408
        assert!(DynarecCompiler::decode_thumb_format5_non_branch(0x4408).is_none());

        // PC (R15) source or dest -> reject.
        //   ADD R15, R0  010001_00_10_000_111 = 0x4487  (Rd=7, H1=1 -> 15)
        assert!(DynarecCompiler::decode_thumb_format5_non_branch(0x4487).is_none());
    }

    #[test]
    fn compile_thumb_format5_add_mov_no_flag_update() {
        let mut compiler = DynarecCompiler::new();
        // ADD R8, R1  (H1=1)  -> R8 += R1
        //   010001_00_10_001_000 = 0x4488
        // MOV R9, R2  (H1=1)
        //   010001_10_10_010_001 = 0b0100_0110_1001_0001 = 0x4691
        let add = 0x4488u16;
        let mov = 0x4691u16;
        let func = compiler
            .try_compile_thumb_block(&[add, mov])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[1] = 5;
        gpr[2] = 42;
        gpr[8] = 10;
        let mut cpsr: u32 = (1 << 30) | (1 << 31); // Z=1, N=1 pre-set
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[8], 15);
        assert_eq!(gpr[9], 42);
        // ADD/MOV in format 5 don't set flags -> N and Z stay where they were.
        assert_ne!(cpsr & (1 << 30), 0, "Z preserved");
        assert_ne!(cpsr & (1 << 31), 0, "N preserved");
    }

    #[test]
    fn compile_thumb_format5_cmp_sets_flags_no_writeback() {
        let mut compiler = DynarecCompiler::new();
        // CMP R10, R11  -> 0x45DA
        let cmp = 0x45DAu16;
        let func = compiler.try_compile_thumb_block(&[cmp]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[10] = 7;
        gpr[11] = 7;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[10], 7);
        assert_eq!(gpr[11], 7);
        assert_ne!(cpsr & (1 << 30), 0, "Z set on equal");
    }

    // --- Thumb format 1 (LSL/LSR/ASR Rd, Rs, #imm5) ---

    #[test]
    fn decode_thumb_format1_shapes() {
        // LSL R0, R1, #3  -> 000_00_00011_001_000 = 0b0000_0000_1100_1000 = 0x00C8
        let d = DynarecCompiler::decode_thumb_format1(0x00C8).expect("LSL");
        assert_eq!(d.kind, ShiftKind::Lsl);
        assert_eq!(d.imm5, 3);
        assert_eq!(d.rs, 1);
        assert_eq!(d.rd, 0);

        // LSR R2, R3, #5  -> 000_01_00101_011_010 = 0b0000_1001_0101_1010 = 0x095A
        let d = DynarecCompiler::decode_thumb_format1(0x095A).expect("LSR");
        assert_eq!(d.kind, ShiftKind::Lsr);
        assert_eq!(d.imm5, 5);

        // ASR R5, R6, #8  -> 000_10_01000_110_101 = 0b0001_0010_0011_0101 = 0x1235
        let d = DynarecCompiler::decode_thumb_format1(0x1235).expect("ASR");
        assert_eq!(d.kind, ShiftKind::Asr);
        assert_eq!(d.imm5, 8);

        // oo = 11 (format 2) must be rejected by format 1 decoder.
        assert!(DynarecCompiler::decode_thumb_format1(0x1800).is_none());

        // Not Thumb format 1 (top 3 bits != 000).
        assert!(DynarecCompiler::decode_thumb_format1(0x2000).is_none()); // fmt 3
        assert!(DynarecCompiler::decode_thumb_format1(0x4000).is_none()); // fmt 4
    }

    #[test]
    fn compile_lsl_imm_regular() {
        let mut compiler = DynarecCompiler::new();
        // LSL R0, R1, #3
        let func = compiler
            .try_compile_thumb_block(&[0x00C8])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[1] = 0x0000_0005;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 5 << 3);
        // C = bit (32-3) = bit 29 of 5 = 0.
        assert_eq!(cpsr & (1 << 29), 0);
    }

    #[test]
    fn compile_lsl_zero_preserves_c() {
        let mut compiler = DynarecCompiler::new();
        // LSL R0, R1, #0  -> no shift, C preserved.
        let func = compiler
            .try_compile_thumb_block(&[0x0008])
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 0xDEAD_BEEF;
        let mut cpsr: u32 = 1 << 29; // Set C = 1 on input.
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0xDEAD_BEEF);
        assert_ne!(cpsr & (1 << 29), 0, "C preserved on LSL #0");
    }

    #[test]
    fn compile_lsl_imm_shifts_out_carry() {
        let mut compiler = DynarecCompiler::new();
        // LSL R0, R1, #1 -> R0 = R1 << 1, C = bit 31 of R1.
        let func = compiler
            .try_compile_thumb_block(&[0x0048])
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 0x8000_0001;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0x0000_0002);
        assert_ne!(cpsr & (1 << 29), 0, "C from shifted out top bit");
    }

    #[test]
    fn compile_lsr_zero_is_lsr_32() {
        let mut compiler = DynarecCompiler::new();
        // LSR R0, R1, #0 -> LSR #32: result = 0, C = bit 31 of R1.
        let func = compiler
            .try_compile_thumb_block(&[0x0808])
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 0x8000_0000;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0);
        assert_ne!(cpsr & (1 << 29), 0, "C = bit 31 of Rs");
        assert_ne!(cpsr & (1 << 30), 0, "Z set when result == 0");
    }

    #[test]
    fn compile_asr_preserves_sign() {
        let mut compiler = DynarecCompiler::new();
        // ASR R0, R1, #4 -> arithmetic right shift.
        let func = compiler
            .try_compile_thumb_block(&[0x1108])
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 0x8000_0000u32;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0xF800_0000u32, "ASR sign extends");
        assert_ne!(cpsr & (1 << 31), 0, "N set (result top bit)");
    }

    #[test]
    fn compile_asr_zero_is_asr_32() {
        let mut compiler = DynarecCompiler::new();
        // ASR R0, R1, #0 -> ASR #32: result = all sign bits of Rs.
        let func = compiler
            .try_compile_thumb_block(&[0x1008])
            .unwrap();

        let mut gpr = [0u32; 15];
        gpr[1] = 0x8000_0000u32;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0xFFFF_FFFFu32, "all ones when sign bit set");

        gpr[1] = 0x0000_0001u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0, "all zeros when sign bit clear");
    }

    // --- Thumb format 4 logical subset ---

    #[test]
    fn decode_thumb_format4_logical_shapes() {
        // AND R0, R1 -> 010000_0000_001_000 = 0b0100_0000_0000_1000 = 0x4008
        let d = DynarecCompiler::decode_thumb_format4_logical(0x4008).expect("AND");
        assert_eq!(d.op, Thumb4Op::And);
        assert_eq!(d.rd, 0);
        assert_eq!(d.rs, 1);

        // ORR R2, R3 -> 010000_1100_011_010 = 0x431A
        let d = DynarecCompiler::decode_thumb_format4_logical(0x431A).expect("ORR");
        assert_eq!(d.op, Thumb4Op::Orr);
        assert_eq!(d.rs, 3);
        assert_eq!(d.rd, 2);

        // TST R4, R5 -> 010000_1000_101_100 = 0x422C
        let d = DynarecCompiler::decode_thumb_format4_logical(0x422C).expect("TST");
        assert_eq!(d.op, Thumb4Op::Tst);

        // CMP R6, R7 -> 010000_1010_111_110 = 0x42BE
        let d = DynarecCompiler::decode_thumb_format4_logical(0x42BE).expect("CMP");
        assert_eq!(d.op, Thumb4Op::Cmp);

        // MVN R1, R2 -> 010000_1111_010_001 = 0x43D1
        let d = DynarecCompiler::decode_thumb_format4_logical(0x43D1).expect("MVN");
        assert_eq!(d.op, Thumb4Op::Mvn);

        // LSL R0, R1 -> 010000_0010_001_000 = 0x4088  (unsupported)
        assert!(DynarecCompiler::decode_thumb_format4_logical(0x4088).is_none());
        // MUL R0, R1 -> 010000_1101_001_000 = 0x4348  (unsupported)
        assert!(DynarecCompiler::decode_thumb_format4_logical(0x4348).is_none());

        // Not format 4 (top 6 bits != 010000)
        assert!(DynarecCompiler::decode_thumb_format4_logical(0x2000).is_none());
    }

    #[test]
    fn compile_thumb_and_orr_eor_bic() {
        let mut compiler = DynarecCompiler::new();
        // AND R0, R1 ; ORR R2, R3 ; EOR R4, R5 ; BIC R6, R7
        let and = 0x4008u16; // 010000_0000_001_000
        let orr = 0x431Au16; // 010000_1100_011_010
        let eor = 0x406Cu16; // 010000_0001_101_100
        let bic = 0x43BEu16; // 010000_1110_111_110
        let func = compiler
            .try_compile_thumb_block(&[and, orr, eor, bic])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0xF0F0_F0F0; gpr[1] = 0x0FF0_0FF0;
        gpr[2] = 0x0000_0001; gpr[3] = 0x0000_0010;
        gpr[4] = 0xAAAA_AAAA; gpr[5] = 0x5555_5555;
        gpr[6] = 0xFFFF_FFFF; gpr[7] = 0x0F0F_0F0F;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 0xF0F0_F0F0 & 0x0FF0_0FF0, "AND R0, R1");
        assert_eq!(gpr[2], 0x0000_0001 | 0x0000_0010, "ORR R2, R3");
        assert_eq!(gpr[4], 0xAAAA_AAAA ^ 0x5555_5555, "EOR R4, R5");
        assert_eq!(gpr[6], 0xFFFF_FFFF & !0x0F0F_0F0F, "BIC R6, R7");
    }

    #[test]
    fn compile_thumb_mvn_complements() {
        let mut compiler = DynarecCompiler::new();
        // MVN R1, R2
        let mvn = 0x43D1u16;
        let func = compiler.try_compile_thumb_block(&[mvn]).unwrap();

        let mut gpr = [0u32; 15];
        gpr[2] = 0x1234_5678;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[1], !0x1234_5678u32);
        // Top bit set -> N should be set.
        assert_ne!(cpsr & (1 << 31), 0, "N set on negative result");
    }

    #[test]
    fn compile_thumb_tst_cmp_cmn_no_writeback() {
        let mut compiler = DynarecCompiler::new();
        // TST R4, R5 ; CMP R4, R5 ; CMN R4, R5
        let tst = 0x422Cu16;
        let cmp = 0x42ACu16; // 010000_1010_101_100 = 0x42AC
        let cmn = 0x42ECu16; // 010000_1011_101_100 = 0x42EC
        let func = compiler
            .try_compile_thumb_block(&[tst, cmp, cmn])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[4] = 0xDEAD_BEEF;
        gpr[5] = 0x1111_1111;
        let before = gpr;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr, before, "TST/CMP/CMN never writeback");
    }

    #[test]
    fn compile_thumb_mix_format2_and_3() {
        let mut compiler = DynarecCompiler::new();
        // MOV R1, #5; MOV R2, #3; ADD R0, R1, R2; SUB R0, R0, #1
        let mov_r1_5  = 0x2105u16;                    // fmt 3
        let mov_r2_3  = 0x2203u16;                    // fmt 3
        let add_r0_r1_r2 = 0x1888u16;                 // fmt 2 reg
        let sub_r0_r0_1  = 0x1E40u16;                 // fmt 2 imm3: 00011_11_001_000_000 = 0x1E40
        let func = compiler
            .try_compile_thumb_block(&[mov_r1_5, mov_r2_3, add_r0_r1_r2, sub_r0_r0_1])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr);
        assert_eq!(gpr[0], 7);
        assert_eq!(gpr[1], 5);
        assert_eq!(gpr[2], 3);
    }

    #[test]
    fn compile_ldr_negative_offset() {
        let (bus, mut compiler) = new_bus_and_compiler();
        unsafe {
            let b = &mut *bus.bytes.get();
            b[0x60] = 0x01; b[0x61] = 0x02; b[0x62] = 0x03; b[0x63] = 0x04;
        }
        // LDR R1, [R0, #-4]  R0 starts at 0x64, addr = 0x60
        //   E5_10_10_04
        let ldr_neg = 0xE510_1004u32;
        let func = compiler
            .try_compile_mem_block(&[ldr_neg])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0x64;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0x0403_0201u32);
    }

    #[test]
    fn compile_ldr_single() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // Seed memory at offset 0x20.
        unsafe {
            let b = &mut *bus.bytes.get();
            b[0x20] = 0xAA; b[0x21] = 0xBB; b[0x22] = 0xCC; b[0x23] = 0xDD;
        }

        // LDR R1, [R0, #0x20]
        let ldr = 0xE590_1020u32;
        let func = compiler
            .try_compile_mem_block(&[ldr])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0; // base = 0
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0xDDCC_BBAA);
    }

    #[test]
    fn compile_str_then_ldr_roundtrip() {
        let (bus, mut compiler) = new_bus_and_compiler();

        // MOV R2, #0x55 (... actually use the imm that encodes simply)
        // We'll set R2 and R3 via gpr initial state and just emit the mem ops.
        // STR R2, [R0, #0x10];  LDR R3, [R0, #0x10]
        let str_ = 0xE580_2010u32;
        let ldr = 0xE590_3010u32;
        let func = compiler
            .try_compile_mem_block(&[str_, ldr])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        gpr[2] = 0x1234_5678;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[3], 0x1234_5678, "STR then LDR should round trip");
        // And the byte layout in the buffer should be little endian.
        unsafe {
            let b = &*bus.bytes.get();
            assert_eq!(b[0x10], 0x78);
            assert_eq!(b[0x11], 0x56);
            assert_eq!(b[0x12], 0x34);
            assert_eq!(b[0x13], 0x12);
        }
    }

    #[test]
    fn compile_mixed_dp_and_mem() {
        let (bus, mut compiler) = new_bus_and_compiler();
        // MOV R2, #0x42;  STR R2, [R0, #4];  LDR R3, [R0, #4];  ADD R4, R3, #1
        let mov  = 0xE3A0_2042u32;
        let str_ = 0xE580_2004u32;
        let ldr  = 0xE590_3004u32;
        let add  = 0xE283_4001u32;
        let func = compiler
            .try_compile_mem_block(&[mov, str_, ldr, add])
            .expect("compiles");

        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[2], 0x42);
        assert_eq!(gpr[3], 0x42);
        assert_eq!(gpr[4], 0x43);
    }

    #[test]
    fn compile_mem_requires_bus() {
        let mut compiler = DynarecCompiler::new();  // no bus
        let ldr = 0xE590_1020u32;
        assert!(compiler.try_compile_mem_block(&[ldr]).is_none());
    }

    #[test]
    fn compile_ldr_conditional_not_taken() {
        let (bus, mut compiler) = new_bus_and_compiler();
        unsafe {
            let b = &mut *bus.bytes.get();
            b[0] = 0x11; b[1] = 0x22; b[2] = 0x33; b[3] = 0x44;
        }
        // LDREQ R1, [R0, #0]
        //   cond=EQ (0x0), rest same as LDR pattern above.
        //   0_590_1000 = 0x0590_1000
        let ldreq = 0x0590_1000u32;
        let func = compiler
            .try_compile_mem_block(&[ldreq])
            .expect("compiles");

        // Z=0 -> not taken -> gpr[1] stays 0
        let mut gpr = [0u32; 15];
        gpr[1] = 0xDEAD_BEEF;
        let mut cpsr = 0u32;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0xDEAD_BEEF, "EQ not taken, no load");

        // Z=1 -> taken
        let mut gpr = [0u32; 15];
        gpr[0] = 0;
        let mut cpsr: u32 = 1 << 30;
        func(gpr.as_mut_ptr(), &mut cpsr, &bus as *const TestBus as *mut u8);
        assert_eq!(gpr[1], 0x4433_2211);
    }

    #[test]
    fn branch_block_with_dp_tail_also_works() {
        // When the block has no terminal B/BL, the compiled fn should just
        // run the DP instructions and return 0 (no branch taken).
        let mut compiler = DynarecCompiler::new();
        let mov0 = 0xE3A0_0005u32;
        let mov1 = 0xE3A0_100Au32;
        let func = compiler
            .try_compile_block_with_branch(&[mov0, mov1], 0x800_5000)
            .expect("compiles");

        let mut gpr = [0u32; 15];
        let mut cpsr = 0u32;
        let mut pc_out: u32 = 0xC0FF_EE00;
        let taken = func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out);

        assert_eq!(taken, 0);
        assert_eq!(pc_out, 0xC0FF_EE00, "no branch, pc_out untouched");
        assert_eq!(gpr[0], 5);
        assert_eq!(gpr[1], 10);
    }
}
