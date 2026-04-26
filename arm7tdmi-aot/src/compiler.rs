//! LLVM JIT compiler scaffold. Owns the leaked `Context` and the
//! `ExecutionEngine` for the lifetime of the AOT pass. Modules built
//! per-block (or per-page in parallel mode) are added here.

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
#[cfg(test)]
use inkwell::execution_engine::JitFunction;

use crate::replay::{
    AotAbortFn, AotFetchOnlyFn, AotIdleCycleFn, AotLdrHalfFn, AotLdrSignHalfFn, AotLdrWordFn,
    AotLoad32Fn, AotLoad8Fn, AotReplayFn, AotStepFn, AotStore16Fn, AotStore32Fn, AotStore8Fn,
};
// Phase-8 ARM trampolines (used by emit_placeholder_arm_block; the per-instr
// arm emit + step-arm + abort-arm registration come in a future commit).
use crate::table::CompiledFn;

/// Per-I cpu state field offsets + bus pointers baked into the
/// compiled LLVM IR (per I8/I13 — single-pointer cpu_ctx ABI).
/// The phase-4 inline-IR emit fns use these to GEP into the cpu_ctx
/// without going through the trampoline, and to do direct cycle
/// accumulation via the scheduler timestamp pointer.
///
/// SDL frontend computes via offset_of! / `bus.scheduler_timestamp_ptr` /
/// `bus.thumb_fetch_cycles(page)` and passes via
/// `LlvmCompiler::register_cpu_offsets`.
///
/// All cpu offsets are byte offsets within the `Arm7tdmiCore<I>`
/// struct. `gpr` points at gpr[0]; gpr[N] is at gpr + N*4.
#[derive(Clone, Copy, Debug)]
pub struct CpuOffsets {
    pub pc: u32,
    pub gpr: u32,
    pub cpsr: u32,
    pub next_fetch_access: u32,
    /// Pipeline at offset `pipeline`; pipeline[0] at +0, pipeline[1] at +4.
    pub pipeline: u32,
    /// Raw ptr at scheduler.timestamp (usize). Stable for the bus's
    /// lifetime. Inline IR adds K cycles via `*sched_ts_ptr += K`.
    pub scheduler_timestamp_ptr: u64,
    /// Per-page Thumb16 fetch cycle costs. Index by page =
    /// (addr >> 24) & 0xf. Used to emit inline cycle-add constants
    /// in IR. Per I7 these are stable until WAITCNT changes (which
    /// triggers AOT recompile).
    pub thumb_seq_cycles: [u32; 16],
    pub thumb_nonseq_cycles: [u32; 16],
    /// Phase-4P-A: raw `*const u32` to the bus's WAITCNT gen counter.
    /// Inline-cycle paths emit a prologue check that the live counter
    /// matches `aot_gen_baked`; on mismatch, return 0b10 (yield to
    /// scalar) so stale baked cycles don't cause divergence.
    /// 0 = no gen check (legacy / disabled).
    pub aot_gen_counter_ptr: u64,
    /// Phase-4P-A: counter value captured at AOT compile time.
    pub aot_gen_baked: u32,
}

/// Compiler handle. The `Context` is leaked to `'static` so the
/// `ExecutionEngine` and modules can borrow it for the process
/// lifetime — same idiom the JIT branch used.
pub struct LlvmCompiler {
    pub(crate) context: &'static Context,
    pub(crate) engine: ExecutionEngine<'static>,
    pub(crate) next_id: u64,
    /// Per-I trampoline registered at construction. Phase-0
    /// placeholder blocks call this; phase 1+ inline IR replaces
    /// the calls one format at a time.
    pub(crate) replay_thumb_fn: Option<AotReplayFn>,
    /// Phase-1 per-instruction trampoline for unsupported formats
    /// (anything not yet inlined). Set via register_step_thumb.
    pub(crate) step_thumb_fn: Option<AotStepFn>,
    /// Phase-1 mid-block abort check for K=2 cadence.
    pub(crate) abort_thumb_fn: Option<AotAbortFn>,
    /// Phase-8: ARM whole-block replay trampoline. Set via
    /// register_replay_arm. None until enabled.
    pub(crate) replay_arm_fn: Option<AotReplayFn>,
    /// Phase-4 cpu state offsets for inline IR emit. None until
    /// register_cpu_offsets is called; emit fns fall through to
    /// the trampoline call when offsets are missing.
    pub(crate) cpu_offsets: Option<CpuOffsets>,
    /// Phase-4 fetch-only trampoline. Pairs with inline IR for the
    /// instruction's effect. Set via register_fetch_only.
    pub(crate) fetch_only_thumb_fn: Option<AotFetchOnlyFn>,
    /// Phase-4 bus.load_32 trampoline (per I3). F6/F9/F11 inline IR
    /// uses this to do the actual word load with cycle accounting.
    pub(crate) load_32_fn: Option<AotLoad32Fn>,
    /// Phase-4 idle-cycle trampoline. F6 uses this for its 1S+1N+1I
    /// cycle accounting. Future F4 shifts/MUL paths will reuse.
    pub(crate) idle_cycle_fn: Option<AotIdleCycleFn>,
    /// Phase-4 bus.store_32 trampoline (per I3). F11 STR sp-rel
    /// (and future F9 STR / F14 PUSH) inline IR uses this to do
    /// the actual word store with cycle accounting via the bus.
    pub(crate) store_32_fn: Option<AotStore32Fn>,
    /// Phase-4 ldr_word trampoline. F11 LDR sp-rel inline IR uses
    /// this; addr is runtime-computed (gpr[SP] + imm) so can hit
    /// the I14 misaligned-LDR ROR path which the extern handles
    /// internally (incl. cpsr.C side effect).
    pub(crate) ldr_word_fn: Option<AotLdrWordFn>,
    /// Phase-4 ldr_half + store_16 trampolines for F10 LDRH/STRH
    /// inline IR. ldr_half handles misaligned-LDRH ROR + cpsr.C side
    /// effect; store_16 wraps store_aligned_16 (chops `addr & ~1`).
    pub(crate) ldr_half_fn: Option<AotLdrHalfFn>,
    pub(crate) store_16_fn: Option<AotStore16Fn>,
    /// Phase-4 byte load/store trampolines for F9 LDRB/STRB.
    pub(crate) load_8_fn: Option<AotLoad8Fn>,
    pub(crate) store_8_fn: Option<AotStore8Fn>,
    /// Phase-4 signed halfword load trampoline for F8 LDSH.
    pub(crate) ldr_sign_half_fn: Option<AotLdrSignHalfFn>,
    /// Phase-4P-B block-exit pipeline-restore trampoline. Set when
    /// AOT_BAKED_CYCLES path is wired by the SDL frontend; the per-
    /// instr emit fn calls this once at block exit when the inline
    /// path skipped per-opcode pipeline shifts.
    pub(crate) pipeline_restore_thumb_fn: Option<crate::replay::AotPipelineRestoreFn>,
}

impl LlvmCompiler {
    /// Build a fresh compiler. Initializes the native target and a
    /// JIT execution engine at -O3 (Aggressive).
    pub fn new() -> Result<Self, String> {
        // SAFETY: leaking a Context is the standard inkwell idiom for
        // multi-module JIT scenarios where modules need a stable parent.
        let context: &'static Context = Box::leak(Box::new(Context::create()));
        let module = context.create_module("aot_root");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| format!("create_jit_execution_engine: {}", e))?;
        Ok(Self {
            context,
            engine,
            next_id: 0,
            replay_thumb_fn: None,
            step_thumb_fn: None,
            abort_thumb_fn: None,
            replay_arm_fn: None,
            cpu_offsets: None,
            fetch_only_thumb_fn: None,
            load_32_fn: None,
            idle_cycle_fn: None,
            store_32_fn: None,
            ldr_word_fn: None,
            ldr_half_fn: None,
            store_16_fn: None,
            load_8_fn: None,
            store_8_fn: None,
            ldr_sign_half_fn: None,
            pipeline_restore_thumb_fn: None,
        })
    }

    /// Phase-4 hook: register cpu state offsets so emit fns can bake
    /// them into IR for direct gpr/cpsr/pc access. Caller (SDL
    /// frontend) computes via `std::mem::offset_of!` for the
    /// monomorphized `Arm7tdmiCore<I>` and passes here.
    pub fn register_cpu_offsets(&mut self, offsets: CpuOffsets) {
        self.cpu_offsets = Some(offsets);
    }

    /// Phase-4 hook: register the fetch-only trampoline. Required
    /// alongside `register_cpu_offsets` for inline IR emit to fire.
    pub fn register_fetch_only_thumb(&mut self, f: AotFetchOnlyFn) {
        self.fetch_only_thumb_fn = Some(f);
    }

    /// Phase-4 hook: register the bus-side load_32 trampoline. F6 LDR
    /// pc-rel inline IR calls this to do the actual word load with
    /// per-I cycle accounting via the bus path (always reads current
    /// cycle_luts; survives WAITCNT writes).
    pub fn register_load_32(&mut self, f: AotLoad32Fn) {
        self.load_32_fn = Some(f);
    }

    /// Phase-4 hook: register the bus-side idle-cycle trampoline.
    pub fn register_idle_cycle(&mut self, f: AotIdleCycleFn) {
        self.idle_cycle_fn = Some(f);
    }

    /// Phase-4 hook: register the bus-side store_32 trampoline.
    pub fn register_store_32(&mut self, f: AotStore32Fn) {
        self.store_32_fn = Some(f);
    }

    /// Phase-4 hook: register the ldr_word trampoline (handles I14
    /// misaligned-LDR ROR + cpsr.C side effect inside the extern).
    /// Used by F11 LDR sp-rel inline IR.
    pub fn register_ldr_word(&mut self, f: AotLdrWordFn) {
        self.ldr_word_fn = Some(f);
    }

    /// Phase-4 hooks: F10 LDRH/STRH externs.
    pub fn register_ldr_half(&mut self, f: AotLdrHalfFn) {
        self.ldr_half_fn = Some(f);
    }

    pub fn register_store_16(&mut self, f: AotStore16Fn) {
        self.store_16_fn = Some(f);
    }

    /// Phase-4 hooks: F9 LDRB/STRB byte externs.
    pub fn register_load_8(&mut self, f: AotLoad8Fn) {
        self.load_8_fn = Some(f);
    }

    pub fn register_store_8(&mut self, f: AotStore8Fn) {
        self.store_8_fn = Some(f);
    }

    pub fn register_ldr_sign_half(&mut self, f: AotLdrSignHalfFn) {
        self.ldr_sign_half_fn = Some(f);
    }

    /// Phase-4P-B: register the block-exit pipeline-restore trampoline.
    /// Required for AOT_BAKED_CYCLES path; without it the IR falls
    /// back to per-opcode fetch_only.
    pub fn register_pipeline_restore_thumb(
        &mut self,
        f: crate::replay::AotPipelineRestoreFn,
    ) {
        self.pipeline_restore_thumb_fn = Some(f);
    }

    /// Register the per-I monomorphized phase-0 whole-block trampoline.
    pub fn register_replay_thumb(&mut self, f: AotReplayFn) {
        self.replay_thumb_fn = Some(f);
    }

    /// Phase-8: register ARM whole-block trampoline (per-I monomorphized).
    pub fn register_replay_arm(&mut self, f: AotReplayFn) {
        self.replay_arm_fn = Some(f);
    }

    /// Register the phase-1 per-instruction trampoline + abort check.
    pub fn register_step_thumb(&mut self, step: AotStepFn, abort: AotAbortFn) {
        self.step_thumb_fn = Some(step);
        self.abort_thumb_fn = Some(abort);
    }

    /// Phase-1 per-instruction Thumb block emit. Each opcode becomes
    /// either inline IR (for supported formats — currently F3 MOV
    /// imm8) or a call to the per-iter step trampoline (for everything
    /// else). K=2 abort check between iters per I2.
    ///
    /// Returns the JIT'd CompiledFn matching the CompiledFn ABI
    /// (`extern "C" fn(cpu_ctx, pc_out) -> u32`).
    ///
    /// `opcodes` is the raw u16 opcode list for the block.
    /// `entry_pc` is the block's first-instruction exec_addr (NOT the
    /// pipeline-head; the ABI for fetch_addr offset = entry_pc + 4).
    pub fn emit_per_instr_thumb_block(
        &mut self,
        opcodes: &[u16],
        entry_pc: u32,
    ) -> Option<CompiledFn> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let step_fn = self.step_thumb_fn?;
        let abort_fn = self.abort_thumb_fn?;

        // Use per-module unique symbol names for the externs.
        // Without this, every block module declares "rba_aot_step"
        // and the JIT engine's global symbol table gets confused
        // across many modules — that was the suspect cause of the
        // sweep>=4KB divergence.
        self.next_id += 1;
        let id = self.next_id;
        let module = self.context.create_module(&format!("aot_blk_pi_{}", id));
        let i8_t = self.context.i8_type();
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        // Imports: step + abort trampolines, unique-named per module.
        let step_sig = i32_t.fn_type(
            &[ptr_t.into(), i32_t.into(), i32_t.into()],
            false,
        );
        let step_name = format!("rba_aot_step_{}", id);
        let step_ref = module.add_function(&step_name, step_sig, None);
        self.engine.add_global_mapping(&step_ref, step_fn as usize);

        let abort_sig = i32_t.fn_type(&[ptr_t.into()], false);
        let abort_name = format!("rba_aot_abort_{}", id);
        let abort_ref = module.add_function(&abort_name, abort_sig, None);
        self.engine.add_global_mapping(&abort_ref, abort_fn as usize);

        // Phase-4 inline IR uses the fetch-only trampoline + cpu offsets.
        // Both must be registered to enable inline IR; otherwise we fall
        // back to the per-iter step trampoline call (phase-1 behavior).
        // Per-format env-var gates: AOT_INLINE_F1=1, AOT_INLINE_F3=1, ...
        let f1_inline_env = std::env::var("AOT_INLINE_F1").map(|v| v == "1").unwrap_or(false);
        let f3_inline_env = std::env::var("AOT_INLINE_F3").map(|v| v == "1").unwrap_or(false);
        let f12_inline_env = std::env::var("AOT_INLINE_F12").map(|v| v == "1").unwrap_or(false);
        let f13_inline_env = std::env::var("AOT_INLINE_F13").map(|v| v == "1").unwrap_or(false);
        let f2_inline_env = std::env::var("AOT_INLINE_F2").map(|v| v == "1").unwrap_or(false);
        // F4 sub-op groups: logical (AND/EOR/TST/ORR/BIC/MVN),
        // arithmetic (ADC/SBC/NEG/CMP/CMN). shifts and MUL still in
        // trampoline path (separate cycle semantics).
        let f4_log_inline_env = std::env::var("AOT_INLINE_F4_LOG").map(|v| v == "1").unwrap_or(false);
        let f4_arith_inline_env = std::env::var("AOT_INLINE_F4_ARITH").map(|v| v == "1").unwrap_or(false);
        // F4 shift sub-ops (LSL=2, LSR=3, ASR=4, ROR=7) — runtime amount
        // (gpr[Rs] & 0xff), per-amount-range CFG to avoid LLVM poison
        // from shift-by-32+. cycle accounting uses idle_cycle extern
        // (NOT inline ts_ptr += 1 — see F4 MUL rejection commit 58e054b).
        let f4_shift_inline_env = std::env::var("AOT_INLINE_F4_SHIFT").map(|v| v == "1").unwrap_or(false);
        // F4 MUL (op=0b1101) — operand-derived cycle count m∈{1,2,3,4}
        // from get_required_multipiler_array_cycles. cycle accounting
        // uses up to 4 conditional idle_cycle extern calls (NOT inline
        // ts_ptr += m — that pattern was rejected in commit 58e054b
        // with 3 reproducible MK divs, root cause was scheduler events
        // firing at wrong cycle relative to the bulk add).
        let f4_mul_inline_env = std::env::var("AOT_INLINE_F4_MUL").map(|v| v == "1").unwrap_or(false);
        let f6_inline_env = std::env::var("AOT_INLINE_F6").map(|v| v == "1").unwrap_or(false);
        // F11 split: STR-only and LDR-only gated separately. Both share
        // top4 = 0x9 but masks are disjoint (STR=0x9000, LDR=0x9800).
        // LDR routes through the ldr_word extern which handles I14
        // misaligned-LDR ROR + cpsr.C side effect inside the extern.
        let f11_str_inline_env = std::env::var("AOT_INLINE_F11_STR").map(|v| v == "1").unwrap_or(false);
        let f11_ldr_inline_env = std::env::var("AOT_INLINE_F11_LDR").map(|v| v == "1").unwrap_or(false);
        let f10_inline_env = std::env::var("AOT_INLINE_F10").map(|v| v == "1").unwrap_or(false);
        let f9_inline_env = std::env::var("AOT_INLINE_F9").map(|v| v == "1").unwrap_or(false);
        let f7_inline_env = std::env::var("AOT_INLINE_F7").map(|v| v == "1").unwrap_or(false);
        let f8_inline_env = std::env::var("AOT_INLINE_F8").map(|v| v == "1").unwrap_or(false);
        let f5_inline_env = std::env::var("AOT_INLINE_F5").map(|v| v == "1").unwrap_or(false);
        let f14_inline_env = std::env::var("AOT_INLINE_F14").map(|v| v == "1").unwrap_or(false);
        let f15_inline_env = std::env::var("AOT_INLINE_F15").map(|v| v == "1").unwrap_or(false);
        let f19hi_inline_env = std::env::var("AOT_INLINE_F19_HI").map(|v| v == "1").unwrap_or(false);
        let any_format_inline = f1_inline_env
            || f2_inline_env
            || f3_inline_env
            || f4_log_inline_env
            || f4_arith_inline_env
            || f4_shift_inline_env
            || f4_mul_inline_env
            || f6_inline_env
            || f11_str_inline_env
            || f11_ldr_inline_env
            || f10_inline_env
            || f9_inline_env
            || f7_inline_env
            || f8_inline_env
            || f5_inline_env
            || f14_inline_env
            || f15_inline_env
            || f12_inline_env
            || f13_inline_env
            || f19hi_inline_env;
        let inline_enabled = self.cpu_offsets.is_some()
            && self.fetch_only_thumb_fn.is_some()
            && any_format_inline;
        // Phase-4P-B: detect whether every opcode in this block matches
        // a registered inline format. If true AND AOT_BAKED_CYCLES=1 AND
        // pipeline_restore is registered, swap fo_ref to point at an
        // in-module IR helper that does ONLY the cycle charge (no
        // load_16, no pipeline shift). Block-exit pipeline_restore
        // call replaces the cumulative pipeline shifts.
        let baked_cycles_env = std::env::var("AOT_BAKED_CYCLES")
            .map(|v| v == "1")
            .unwrap_or(false);
        let block_fully_inline = if baked_cycles_env {
            opcodes.iter().all(|&op| {
                thumb_opcode_in_inline_path(
                    op as u32,
                    f1_inline_env, f2_inline_env, f3_inline_env,
                    f4_log_inline_env, f4_arith_inline_env, f4_shift_inline_env, f4_mul_inline_env,
                    f5_inline_env, f6_inline_env, f7_inline_env, f8_inline_env,
                    f9_inline_env, f10_inline_env,
                    f11_str_inline_env, f11_ldr_inline_env,
                    f12_inline_env, f13_inline_env, f14_inline_env, f15_inline_env,
                    f19hi_inline_env,
                )
            })
        } else {
            false
        };
        let baked_cycles_active = baked_cycles_env
            && block_fully_inline
            && self.pipeline_restore_thumb_fn.is_some()
            && self.cpu_offsets.is_some();

        let (fetch_only_ref, offsets) = if inline_enabled {
            let off = self.cpu_offsets.unwrap();
            let fo_sig = self.context.void_type()
                .fn_type(&[ptr_t.into(), i32_t.into()], false);

            let chosen_fo_ref = if baked_cycles_active {
                // Build the baked-fetch helper IR fn in this module.
                // Body: read nfa byte from cpu_ctx, look up cycles per
                // page from baked seq/nonseq const arrays, add to
                // *sched_ts_ptr. No load_16, no pipeline shift.
                let bf_name = format!("rba_aot_baked_fetch_{}", id);
                let bf_ref = module.add_function(&bf_name, fo_sig, None);
                let bf_entry = self.context.append_basic_block(bf_ref, "entry");
                let bf_b = self.context.create_builder();
                bf_b.position_at_end(bf_entry);
                let bf_cpu = bf_ref.get_nth_param(0).unwrap().into_pointer_value();
                let bf_fa = bf_ref.get_nth_param(1).unwrap().into_int_value();
                // page = (fa >> 24) & 0xf
                let page_shift = bf_b
                    .build_right_shift(bf_fa, i32_t.const_int(24, false), false, "ps")
                    .ok()?;
                let page = bf_b
                    .build_and(page_shift, i32_t.const_int(0xf, false), "page")
                    .ok()?;
                // Read nfa byte (i8).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    bf_b.build_in_bounds_gep(i8_t, bf_cpu, &[nfa_off_v], "nfa_ptr").ok()?
                };
                let nfa_val = bf_b
                    .build_load(i8_t, nfa_ptr, "nfa_val")
                    .ok()?
                    .into_int_value();
                // Build per-page const arrays (i32 x 16) for seq + nonseq cycles.
                let arr_t = i32_t.array_type(16);
                let seq_vals: Vec<inkwell::values::IntValue> = (0..16)
                    .map(|i| i32_t.const_int(off.thumb_seq_cycles[i] as u64, false))
                    .collect();
                let nonseq_vals: Vec<inkwell::values::IntValue> = (0..16)
                    .map(|i| i32_t.const_int(off.thumb_nonseq_cycles[i] as u64, false))
                    .collect();
                let seq_const = i32_t.const_array(&seq_vals);
                let nonseq_const = i32_t.const_array(&nonseq_vals);
                let seq_global = module.add_global(
                    arr_t,
                    None,
                    &format!("rba_seq_cycles_{}", id),
                );
                seq_global.set_initializer(&seq_const);
                seq_global.set_constant(true);
                let nonseq_global = module.add_global(
                    arr_t,
                    None,
                    &format!("rba_nonseq_cycles_{}", id),
                );
                nonseq_global.set_initializer(&nonseq_const);
                nonseq_global.set_constant(true);
                // GEP into both arrays at index `page`, load both.
                let zero_i32 = i32_t.const_int(0, false);
                let seq_elem_ptr = unsafe {
                    bf_b.build_in_bounds_gep(
                        arr_t,
                        seq_global.as_pointer_value(),
                        &[zero_i32, page],
                        "seq_elem",
                    ).ok()?
                };
                let nonseq_elem_ptr = unsafe {
                    bf_b.build_in_bounds_gep(
                        arr_t,
                        nonseq_global.as_pointer_value(),
                        &[zero_i32, page],
                        "nonseq_elem",
                    ).ok()?
                };
                let k_seq = bf_b
                    .build_load(i32_t, seq_elem_ptr, "k_seq")
                    .ok()?
                    .into_int_value();
                let k_nonseq = bf_b
                    .build_load(i32_t, nonseq_elem_ptr, "k_nonseq")
                    .ok()?
                    .into_int_value();
                // is_seq = (nfa == 1)
                let is_seq = bf_b
                    .build_int_compare(
                        IntPredicate::EQ,
                        nfa_val,
                        i8_t.const_int(1, false),
                        "is_seq",
                    )
                    .ok()?;
                let k_i32 = bf_b
                    .build_select(is_seq, k_seq, k_nonseq, "k")
                    .ok()?
                    .into_int_value();
                // *sched_ts_ptr += k (as i64)
                let i64_t_local = self.context.i64_type();
                let k_i64 = bf_b
                    .build_int_z_extend(k_i32, i64_t_local, "k_i64")
                    .ok()?;
                let ts_ptr_const =
                    i64_t_local.const_int(off.scheduler_timestamp_ptr, false);
                let ts_ptr = bf_b
                    .build_int_to_ptr(ts_ptr_const, ptr_t, "ts_ptr")
                    .ok()?;
                let ts_val = bf_b
                    .build_load(i64_t_local, ts_ptr, "ts_val")
                    .ok()?
                    .into_int_value();
                let new_ts = bf_b
                    .build_int_add(ts_val, k_i64, "ts_new")
                    .ok()?;
                bf_b.build_store(ts_ptr, new_ts).ok()?;
                bf_b.build_return(None).ok()?;
                bf_ref
            } else {
                let fetch_only_fn = self.fetch_only_thumb_fn.unwrap();
                let fo_name = format!("rba_aot_fetch_only_{}", id);
                let fo_ref = module.add_function(&fo_name, fo_sig, None);
                self.engine.add_global_mapping(&fo_ref, fetch_only_fn as usize);
                fo_ref
            };
            (Some(chosen_fo_ref), Some(off))
        } else {
            (None, None)
        };

        // Phase-4P-B: register the pipeline_restore extern when baked
        // path is active, so the for-loop can emit a single block-exit
        // call to set pipeline[0]/[1] for the next block.
        let pipe_restore_ref = if baked_cycles_active {
            let pf = self.pipeline_restore_thumb_fn.unwrap();
            let pr_sig = self.context.void_type()
                .fn_type(&[ptr_t.into(), i32_t.into()], false);
            let pr_name = format!("rba_aot_pipe_restore_{}", id);
            let pr_ref = module.add_function(&pr_name, pr_sig, None);
            self.engine.add_global_mapping(&pr_ref, pf as usize);
            Some(pr_ref)
        } else {
            None
        };

        // Phase-4 F6 helpers: bus.load_32 + idle_cycle externs.
        // load_32 is F6-only; idle_cycle is shared with F4 shifts (which
        // also need a per-iter idle cycle, charged via the bus path
        // exactly as F6 does). Per F4 MUL rejection (commit 58e054b),
        // inline `*ts_ptr += 1` is suspect — bus extern is the safe path.
        let load_32_ref = if (f6_inline_env || f14_inline_env || f15_inline_env) && self.load_32_fn.is_some() {
            let load_32_fn = self.load_32_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let load_32_sig = i32_t.fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let load_32_name = format!("rba_aot_load_32_{}", id);
            let l32_ref = module.add_function(&load_32_name, load_32_sig, None);
            self.engine.add_global_mapping(&l32_ref, load_32_fn as usize);
            Some(l32_ref)
        } else {
            None
        };
        // F6 + F4_SHIFT + F11 LDR + F10 (LDRH variant has 1I post-load
        // idle) all need idle_cycle.
        let idle_cycle_ref = if (f6_inline_env || f4_shift_inline_env
            || f4_mul_inline_env
            || f11_ldr_inline_env || f10_inline_env || f9_inline_env
            || f7_inline_env || f8_inline_env || f14_inline_env
            || f15_inline_env)
            && self.idle_cycle_fn.is_some()
        {
            let idle_fn = self.idle_cycle_fn.unwrap();
            let idle_sig = self.context.void_type().fn_type(&[ptr_t.into()], false);
            let idle_name = format!("rba_aot_idle_{}", id);
            let idle_ref = module.add_function(&idle_name, idle_sig, None);
            self.engine.add_global_mapping(&idle_ref, idle_fn as usize);
            Some(idle_ref)
        } else {
            None
        };

        // Phase-4 F11 STR helper: bus.store_32 extern.
        let store_32_ref = if (f11_str_inline_env || f9_inline_env || f7_inline_env || f14_inline_env || f15_inline_env) && self.store_32_fn.is_some() {
            let store_32_fn = self.store_32_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let store_32_sig = self.context.void_type().fn_type(
                &[ptr_t.into(), i32_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let store_32_name = format!("rba_aot_store_32_{}", id);
            let s32_ref = module.add_function(&store_32_name, store_32_sig, None);
            self.engine.add_global_mapping(&s32_ref, store_32_fn as usize);
            Some(s32_ref)
        } else {
            None
        };

        // Phase-4 F11 LDR helper: ldr_word extern. Same signature as
        // load_32 (cpu_ctx, addr, access_byte) -> u32 — but the wrapper
        // routes through `cpu.aot_ldr_word` which does I14 misaligned-
        // LDR ROR + cpsr.C side effect when `addr & 3 != 0`.
        let ldr_word_ref = if (f11_ldr_inline_env || f9_inline_env || f7_inline_env) && self.ldr_word_fn.is_some() {
            let ldr_word_fn = self.ldr_word_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let ldr_word_sig = i32_t.fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let ldr_word_name = format!("rba_aot_ldr_word_{}", id);
            let lw_ref = module.add_function(&ldr_word_name, ldr_word_sig, None);
            self.engine.add_global_mapping(&lw_ref, ldr_word_fn as usize);
            Some(lw_ref)
        } else {
            None
        };

        // Phase-4 F10 helpers: ldr_half + store_16 externs.
        let ldr_half_ref = if (f10_inline_env || f8_inline_env) && self.ldr_half_fn.is_some() {
            let ldr_half_fn = self.ldr_half_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let ldr_half_sig = i32_t.fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let ldr_half_name = format!("rba_aot_ldr_half_{}", id);
            let lh_ref = module.add_function(&ldr_half_name, ldr_half_sig, None);
            self.engine.add_global_mapping(&lh_ref, ldr_half_fn as usize);
            Some(lh_ref)
        } else {
            None
        };
        let store_16_ref = if (f10_inline_env || f8_inline_env) && self.store_16_fn.is_some() {
            let store_16_fn = self.store_16_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let i16_t = self.context.i16_type();
            let store_16_sig = self.context.void_type().fn_type(
                &[ptr_t.into(), i32_t.into(), i16_t.into(), i8_t_local.into()],
                false,
            );
            let store_16_name = format!("rba_aot_store_16_{}", id);
            let s16_ref = module.add_function(&store_16_name, store_16_sig, None);
            self.engine.add_global_mapping(&s16_ref, store_16_fn as usize);
            Some(s16_ref)
        } else {
            None
        };

        // Phase-4 F9 helpers: load_8 + store_8 externs (LDRB/STRB).
        let load_8_ref = if (f9_inline_env || f7_inline_env || f8_inline_env) && self.load_8_fn.is_some() {
            let load_8_fn = self.load_8_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let load_8_sig = i8_t_local.fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let load_8_name = format!("rba_aot_load_8_{}", id);
            let l8_ref = module.add_function(&load_8_name, load_8_sig, None);
            self.engine.add_global_mapping(&l8_ref, load_8_fn as usize);
            Some(l8_ref)
        } else {
            None
        };
        let store_8_ref = if (f9_inline_env || f7_inline_env) && self.store_8_fn.is_some() {
            let store_8_fn = self.store_8_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let store_8_sig = self.context.void_type().fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into(), i8_t_local.into()],
                false,
            );
            let store_8_name = format!("rba_aot_store_8_{}", id);
            let s8_ref = module.add_function(&store_8_name, store_8_sig, None);
            self.engine.add_global_mapping(&s8_ref, store_8_fn as usize);
            Some(s8_ref)
        } else {
            None
        };

        // Phase-4 F8 helper: ldr_sign_half extern (LDSH, handles
        // misaligned via sign-extended byte load).
        let ldr_sign_half_ref = if f8_inline_env && self.ldr_sign_half_fn.is_some() {
            let ldr_sign_half_fn = self.ldr_sign_half_fn.unwrap();
            let i8_t_local = self.context.i8_type();
            let ldr_sign_half_sig = i32_t.fn_type(
                &[ptr_t.into(), i32_t.into(), i8_t_local.into()],
                false,
            );
            let ldr_sign_half_name = format!("rba_aot_ldr_sign_half_{}", id);
            let lsh_ref = module.add_function(&ldr_sign_half_name, ldr_sign_half_sig, None);
            self.engine.add_global_mapping(&lsh_ref, ldr_sign_half_fn as usize);
            Some(lsh_ref)
        } else {
            None
        };

        // Block fn.
        let block_sig = i32_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);
        let block_name = format!("aot_pi_blk_{}", id);
        let block_fn = module.add_function(&block_name, block_sig, None);
        let entry = self.context.append_basic_block(block_fn, "entry");
        let exit_blk = self.context.append_basic_block(block_fn, "exit");
        let abort_blk = self.context.append_basic_block(block_fn, "abort");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let cpu_ctx = block_fn.get_nth_param(0).unwrap().into_pointer_value();

        for (k, &opcode) in opcodes.iter().enumerate() {
            // K=2 abort cadence: check before iters with k odd && k != 0.
            if k != 0 && (k & 1) == 1 {
                let acall = builder
                    .build_call(abort_ref, &[cpu_ctx.into()], "abort_res")
                    .ok()?;
                let abort_v = acall.try_as_basic_value().unwrap_basic().into_int_value();
                let zero = i32_t.const_int(0, false);
                let nonzero = builder
                    .build_int_compare(IntPredicate::NE, abort_v, zero, "abort_nz")
                    .ok()?;
                let after_abort = self.context.append_basic_block(block_fn, "after_abort");
                builder
                    .build_conditional_branch(nonzero, abort_blk, after_abort)
                    .ok()?;
                builder.position_at_end(after_abort);
            }

            // Compute fetch_addr at compile time (constant).
            let exec_addr = entry_pc.wrapping_add((2 * k) as u32);
            let fetch_addr = exec_addr.wrapping_add(4);
            let fa = i32_t.const_int(fetch_addr as u64, false);
            let insn = i32_t.const_int(opcode as u64, false);

            // Phase-4 inline IR for F1 MoveShiftedReg (LSL/LSR/ASR
            // Rd, Rs, #imm5). Encoding: 000_oo_IIIII_SSS_DDD where
            // oo ∈ {LSL=0, LSR=1, ASR=2}; oo=0b11 is F2 AddSub.
            // Effects (matches arm7tdmi/src/alu.rs::{lsl,lsr,asr}
            // immediate=true):
            //   LSL #0:  result = Rs;            carry preserved
            //   LSL #n:  result = Rs<<n;         carry = (Rs >> (32-n)) & 1
            //   LSR #0:  result = 0;             carry = Rs >> 31
            //   LSR #n:  result = Rs>>n logical; carry = (Rs >> (n-1)) & 1
            //   ASR #0:  result = ashr 31;       carry = Rs >> 31
            //   ASR #n:  result = ashr n;        carry = (Rs >> (n-1)) & 1
            //   gpr[Rd] = result; cpsr.N = result bit 31; cpsr.Z = result==0;
            //   cpsr.C = carry; cpsr.V preserved.
            //   pc = fetch_addr + 2; nfa = Seq.
            //
            // No inline cycle accumulation — fetch_only call charges
            // cycles via the bus path (always reads current cycle_luts).
            // See compiler.rs F3 IR commentary for the WAITCNT bug
            // story that scuttled the inline cycle-accounting attempt.
            let f1_top3 = (opcode >> 13) & 0x7;
            let f1_op_bits = ((opcode >> 11) & 0x3) as u32;
            if inline_enabled && f1_inline_env && f1_top3 == 0b000 && f1_op_bits != 0b11 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let imm = ((opcode >> 6) & 0x1f) as u32;
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rs].
                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f1_gpr_rs_ptr").ok()?
                };
                let rs_val = builder.build_load(i32_t, gpr_rs_ptr, "f1_rs_val").ok()?
                    .into_int_value();

                // Load old cpsr (used for LSL #0 carry preservation +
                // final cpsr update).
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f1_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f1_cpsr_old").ok()?
                    .into_int_value();

                let zero_i32 = i32_t.const_int(0, false);
                // Compute (result, carry) — all constants from the opcode.
                let (result, carry_i32) = match (f1_op_bits, imm) {
                    (0, 0) => {
                        // LSL #0: result = Rs; carry = (cpsr_old >> 29) & 1.
                        let c_shift = builder
                            .build_right_shift(cpsr_old, i32_t.const_int(29, false), false, "f1_c_old_s")
                            .ok()?;
                        let c_old = builder
                            .build_and(c_shift, i32_t.const_int(1, false), "f1_c_old")
                            .ok()?;
                        (rs_val, c_old)
                    }
                    (0, n) => {
                        let result = builder
                            .build_left_shift(rs_val, i32_t.const_int(n as u64, false), "f1_lsl_res")
                            .ok()?;
                        let c_shift = builder
                            .build_right_shift(rs_val, i32_t.const_int((32 - n) as u64, false), false, "f1_lsl_cs")
                            .ok()?;
                        let carry = builder
                            .build_and(c_shift, i32_t.const_int(1, false), "f1_lsl_c")
                            .ok()?;
                        (result, carry)
                    }
                    (1, 0) => {
                        let carry = builder
                            .build_right_shift(rs_val, i32_t.const_int(31, false), false, "f1_lsr32_c")
                            .ok()?;
                        (zero_i32, carry)
                    }
                    (1, n) => {
                        let result = builder
                            .build_right_shift(rs_val, i32_t.const_int(n as u64, false), false, "f1_lsr_res")
                            .ok()?;
                        let c_shift = builder
                            .build_right_shift(rs_val, i32_t.const_int((n - 1) as u64, false), false, "f1_lsr_cs")
                            .ok()?;
                        let carry = builder
                            .build_and(c_shift, i32_t.const_int(1, false), "f1_lsr_c")
                            .ok()?;
                        (result, carry)
                    }
                    (2, 0) => {
                        let result = builder
                            .build_right_shift(rs_val, i32_t.const_int(31, false), true, "f1_asr32_res")
                            .ok()?;
                        let carry = builder
                            .build_right_shift(rs_val, i32_t.const_int(31, false), false, "f1_asr32_c")
                            .ok()?;
                        (result, carry)
                    }
                    (2, n) => {
                        let result = builder
                            .build_right_shift(rs_val, i32_t.const_int(n as u64, false), true, "f1_asr_res")
                            .ok()?;
                        let c_shift = builder
                            .build_right_shift(rs_val, i32_t.const_int((n - 1) as u64, false), false, "f1_asr_cs")
                            .ok()?;
                        let carry = builder
                            .build_and(c_shift, i32_t.const_int(1, false), "f1_asr_c")
                            .ok()?;
                        (result, carry)
                    }
                    _ => unreachable!(),
                };

                // gpr[Rd] = result.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f1_gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, result).ok()?;

                // cpsr update: clear N|Z|C, set N from result bit 31,
                // Z from (result==0), C from carry. V untouched.
                let nzc_clear = i32_t.const_int(0x1fff_ffff, false);
                let cleared = builder.build_and(cpsr_old, nzc_clear, "f1_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f1_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, zero_i32, "f1_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f1_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f1_z_bit")
                    .ok()?;
                let c_bit = builder
                    .build_left_shift(carry_i32, i32_t.const_int(29, false), "f1_c_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f1_cpsr_n").ok()?;
                let cpsr_nz = builder.build_or(cpsr_n, z_bit, "f1_cpsr_nz").ok()?;
                let cpsr_new = builder.build_or(cpsr_nz, c_bit, "f1_cpsr_new").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f1_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq (= 1).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f1_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC → always AdvancePC).
                let cont_blk = self.context.append_basic_block(block_fn, "f1_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F2 AddSub (ADD/SUB Rd, Rs, Rn or imm3).
            // Encoding: 00011_I_S_NNN_SSS_DDD; bits 15:11 = 0b00011.
            //   I (bit 10): 0 = reg (NNN = Rn), 1 = imm3 (NNN = imm value)
            //   S (bit 9):  0 = ADD, 1 = SUB
            //   NNN (bits 8:6): Rn or imm3
            //   SSS (bits 5:3): Rs
            //   DDD (bits 2:0): Rd
            //
            // Effects (mirrors arm7tdmi/src/cpu.rs aot_thumb_step F2 path
            // and arm7tdmi/src/alu.rs::{alu_add_flags, alu_sub_flags}):
            //   a = gpr[Rs]; b = imm3 (constant) or gpr[Rn] (runtime).
            //   ADD: result = a + b
            //        carry    = unsigned overflow = (result < a)
            //        overflow = ((result ^ a) & (result ^ b)) bit 31
            //   SUB: result = a - b
            //        carry    = no-borrow = (a >= b unsigned)
            //        overflow = ((a ^ b) & (a ^ result)) bit 31
            //   gpr[Rd] = result
            //   cpsr: N from result bit 31, Z from result==0, C from
            //         carry, V from overflow.  pc = fetch_addr + 2; nfa = Seq.
            let f2_top5 = (opcode >> 11) & 0x1f;
            if inline_enabled && f2_inline_env && f2_top5 == 0b00011 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let imm_flag = (opcode >> 10) & 0x1 != 0;
                let sub = (opcode >> 9) & 0x1 != 0;
                let rn_or_imm = ((opcode >> 6) & 0x7) as u32;
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load a = gpr[Rs].
                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f2_gpr_rs_ptr").ok()?
                };
                let a = builder.build_load(i32_t, gpr_rs_ptr, "f2_a").ok()?
                    .into_int_value();

                // b = imm3 const | gpr[Rn] runtime.
                let b = if imm_flag {
                    i32_t.const_int(rn_or_imm as u64, false)
                } else {
                    let gpr_rn_off = (off.gpr + rn_or_imm * 4) as u64;
                    let gpr_rn_off_v = i32_t.const_int(gpr_rn_off, false);
                    let gpr_rn_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rn_off_v], "f2_gpr_rn_ptr").ok()?
                    };
                    builder.build_load(i32_t, gpr_rn_ptr, "f2_b").ok()?
                        .into_int_value()
                };

                let (result, carry_i32, ovf_i32) = if sub {
                    // result = a - b
                    let result = builder.build_int_sub(a, b, "f2_sub_res").ok()?;
                    // carry (no-borrow) = (a >= b) unsigned
                    let carry_cmp = builder
                        .build_int_compare(IntPredicate::UGE, a, b, "f2_sub_c_cmp")
                        .ok()?;
                    let carry_i32 = builder.build_int_z_extend(carry_cmp, i32_t, "f2_sub_c").ok()?;
                    // overflow = ((a ^ b) & (a ^ result)) bit 31
                    let ab_xor = builder.build_xor(a, b, "f2_sub_ab_xor").ok()?;
                    let ar_xor = builder.build_xor(a, result, "f2_sub_ar_xor").ok()?;
                    let ovf_and = builder.build_and(ab_xor, ar_xor, "f2_sub_ovf_and").ok()?;
                    let ovf_shift = builder
                        .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f2_sub_ovf_s")
                        .ok()?;
                    let ovf_i32 = builder
                        .build_and(ovf_shift, i32_t.const_int(1, false), "f2_sub_ovf")
                        .ok()?;
                    (result, carry_i32, ovf_i32)
                } else {
                    // result = a + b
                    let result = builder.build_int_add(a, b, "f2_add_res").ok()?;
                    // carry = unsigned overflow = (result < a)
                    let carry_cmp = builder
                        .build_int_compare(IntPredicate::ULT, result, a, "f2_add_c_cmp")
                        .ok()?;
                    let carry_i32 = builder.build_int_z_extend(carry_cmp, i32_t, "f2_add_c").ok()?;
                    // overflow = ((result ^ a) & (result ^ b)) bit 31
                    let ra_xor = builder.build_xor(result, a, "f2_add_ra_xor").ok()?;
                    let rb_xor = builder.build_xor(result, b, "f2_add_rb_xor").ok()?;
                    let ovf_and = builder.build_and(ra_xor, rb_xor, "f2_add_ovf_and").ok()?;
                    let ovf_shift = builder
                        .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f2_add_ovf_s")
                        .ok()?;
                    let ovf_i32 = builder
                        .build_and(ovf_shift, i32_t.const_int(1, false), "f2_add_ovf")
                        .ok()?;
                    (result, carry_i32, ovf_i32)
                };

                // gpr[Rd] = result.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f2_gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, result).ok()?;

                // cpsr: clear N|Z|C|V, set from result/carry/ovf.
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f2_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f2_cpsr_old").ok()?
                    .into_int_value();
                let nzcv_clear = i32_t.const_int(0x0fff_ffff, false); // clear bits 31..28
                let cleared = builder.build_and(cpsr_old, nzcv_clear, "f2_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f2_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, i32_t.const_int(0, false), "f2_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f2_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f2_z_bit")
                    .ok()?;
                let c_bit = builder
                    .build_left_shift(carry_i32, i32_t.const_int(29, false), "f2_c_bit")
                    .ok()?;
                let v_bit = builder
                    .build_left_shift(ovf_i32, i32_t.const_int(28, false), "f2_v_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f2_cpsr_n").ok()?;
                let cpsr_nz = builder.build_or(cpsr_n, z_bit, "f2_cpsr_nz").ok()?;
                let cpsr_nzc = builder.build_or(cpsr_nz, c_bit, "f2_cpsr_nzc").ok()?;
                let cpsr_new = builder.build_or(cpsr_nzc, v_bit, "f2_cpsr_new").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f2_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f2_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f2_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F4 ALU shift sub-ops:
            // LSL(2), LSR(3), ASR(4), ROR(7).
            // Encoding: 010000_OOOO_SSS_DDD; mask 0xfc00 == 0x4000.
            // Effects (mirrors arm7tdmi/src/cpu.rs F4 path + alu.rs
            // shift_by_register / barrel_shift_op with immediate=false):
            //   val    = gpr[Rd]
            //   amount = gpr[Rs] & 0xff   (8-bit, runtime — can't fold)
            //   carry  = old cpsr.C       (preserved when amount==0)
            //   then bus.idle_cycle().
            //   alu_update_flags(result, arithmetic=false, c=carry, v=old V):
            //     N = (result < 0 signed); Z = (result == 0); C = carry; V preserved.
            //   gpr[Rd] = result          (no setting-flags-no-writeback for shifts).
            //   pc = fetch_addr + 2; nfa = Seq.
            //
            // Per-amount-range CFG (per sub-op) avoids LLVM poison from
            // `shl/lshr/ashr i32, X` where X >= 32 (UB at LLVM level —
            // `select` would propagate poison into the live result).
            //
            // LSL ranges (immediate=false):
            //   amount==0   : result=val,         carry=old_C
            //   1..=31      : result=val<<n,      carry=(val>>(32-n))&1
            //   ==32        : result=0,           carry=val&1
            //   >32         : result=0,           carry=0
            //
            // LSR ranges (immediate=false):
            //   amount==0   : result=val,         carry=old_C
            //   1..=31      : result=val>>n lgcl, carry=(val>>(n-1))&1
            //   ==32        : result=0,           carry=val>>31
            //   >32         : result=0,           carry=0
            //
            // ASR ranges (immediate=false):
            //   amount==0   : result=val,         carry=old_C
            //   1..=31      : result=val>>n arth, carry=(val>>(n-1))&1
            //   >=32        : result=sext(val),   carry=val>>31
            //
            // ROR ranges (immediate=false, rrx=true):
            //   amount==0   : result=val,         carry=old_C
            //         (rrx not reached: rrx only fires when immediate=true)
            //   amount>0    : m = amount % 32
            //                  if m==0: result=val            carry=val>>31
            //                  else   : result=rotate_right(val, m); carry=result>>31
            let f4s_top6 = (opcode >> 10) & 0x3f;
            let f4s_op = (opcode >> 6) & 0xf;
            let f4_is_shift = matches!(f4s_op, 2 | 3 | 4 | 7);
            if inline_enabled
                && f4_shift_inline_env
                && idle_cycle_ref.is_some()
                && f4s_top6 == 0b010000
                && f4_is_shift
            {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let idle = idle_cycle_ref.unwrap();
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting (fetch) + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load val = gpr[Rd], amount_full = gpr[Rs], cpsr_old.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f4s_rd_ptr").ok()?
                };
                let val = builder.build_load(i32_t, gpr_rd_ptr, "f4s_val").ok()?
                    .into_int_value();

                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f4s_rs_ptr").ok()?
                };
                let amount_full = builder.build_load(i32_t, gpr_rs_ptr, "f4s_rs_val").ok()?
                    .into_int_value();
                // amount = amount_full & 0xff (per shift_by_register).
                let amount = builder
                    .build_and(amount_full, i32_t.const_int(0xff, false), "f4s_amount")
                    .ok()?;

                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f4s_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f4s_cpsr_old").ok()?
                    .into_int_value();
                // c_old = (cpsr_old >> 29) & 1
                let c_old_shift = builder
                    .build_right_shift(cpsr_old, i32_t.const_int(29, false), false, "f4s_c_old_s")
                    .ok()?;
                let c_old = builder
                    .build_and(c_old_shift, i32_t.const_int(1, false), "f4s_c_old")
                    .ok()?;

                let zero_i32 = i32_t.const_int(0, false);
                let one_i32 = i32_t.const_int(1, false);

                // Per-sub-op CFG. Each sub-op builds basic blocks and
                // phi-merges (result, carry) into a join block before
                // the shared writeback / cpsr / idle_cycle code.
                let join_blk = self.context.append_basic_block(block_fn, "f4s_join");

                let (result, carry_i32) = match f4s_op {
                    2 => {
                        // LSL by amount.
                        // is_zero block:    result=val,    carry=c_old
                        // is_lt32 block:    result=val<<n, carry=(val>>(32-n))&1
                        // is_eq32 block:    result=0,      carry=val&1
                        // is_gt32 block:    result=0,      carry=0
                        let is_zero_blk = self.context.append_basic_block(block_fn, "f4s_lsl_z");
                        let nz_blk = self.context.append_basic_block(block_fn, "f4s_lsl_nz");
                        let is_lt32_blk = self.context.append_basic_block(block_fn, "f4s_lsl_lt");
                        let ge32_blk = self.context.append_basic_block(block_fn, "f4s_lsl_ge");
                        let is_eq32_blk = self.context.append_basic_block(block_fn, "f4s_lsl_eq");
                        let is_gt32_blk = self.context.append_basic_block(block_fn, "f4s_lsl_gt");

                        // amount == 0 ?
                        let is_zero = builder
                            .build_int_compare(IntPredicate::EQ, amount, zero_i32, "f4s_lsl_isz")
                            .ok()?;
                        builder.build_conditional_branch(is_zero, is_zero_blk, nz_blk).ok()?;

                        // is_zero_blk: result = val, carry = c_old (only branch to join).
                        builder.position_at_end(is_zero_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // nz_blk: amount < 32 ?
                        builder.position_at_end(nz_blk);
                        let is_lt32 = builder
                            .build_int_compare(IntPredicate::ULT, amount, i32_t.const_int(32, false), "f4s_lsl_lt32")
                            .ok()?;
                        builder.build_conditional_branch(is_lt32, is_lt32_blk, ge32_blk).ok()?;

                        // is_lt32_blk: result = val << amount; carry = (val >> (32 - amount)) & 1
                        builder.position_at_end(is_lt32_blk);
                        let lt_res = builder.build_left_shift(val, amount, "f4s_lsl_lt_res").ok()?;
                        let neg_amount = builder
                            .build_int_sub(i32_t.const_int(32, false), amount, "f4s_lsl_lt_neg")
                            .ok()?;
                        let cs = builder
                            .build_right_shift(val, neg_amount, false, "f4s_lsl_lt_cs")
                            .ok()?;
                        let lt_carry = builder.build_and(cs, one_i32, "f4s_lsl_lt_c").ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // ge32_blk: amount == 32 ?
                        builder.position_at_end(ge32_blk);
                        let is_eq32 = builder
                            .build_int_compare(IntPredicate::EQ, amount, i32_t.const_int(32, false), "f4s_lsl_eq32")
                            .ok()?;
                        builder.build_conditional_branch(is_eq32, is_eq32_blk, is_gt32_blk).ok()?;

                        // is_eq32_blk: result = 0; carry = val & 1
                        builder.position_at_end(is_eq32_blk);
                        let eq_carry = builder.build_and(val, one_i32, "f4s_lsl_eq_c").ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // is_gt32_blk: result = 0; carry = 0
                        builder.position_at_end(is_gt32_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // join: phi result, phi carry
                        builder.position_at_end(join_blk);
                        let res_phi = builder.build_phi(i32_t, "f4s_lsl_res").ok()?;
                        res_phi.add_incoming(&[
                            (&val, is_zero_blk),
                            (&lt_res, is_lt32_blk),
                            (&zero_i32, is_eq32_blk),
                            (&zero_i32, is_gt32_blk),
                        ]);
                        let c_phi = builder.build_phi(i32_t, "f4s_lsl_c").ok()?;
                        c_phi.add_incoming(&[
                            (&c_old, is_zero_blk),
                            (&lt_carry, is_lt32_blk),
                            (&eq_carry, is_eq32_blk),
                            (&zero_i32, is_gt32_blk),
                        ]);
                        (res_phi.as_basic_value().into_int_value(), c_phi.as_basic_value().into_int_value())
                    }
                    3 => {
                        // LSR by amount (immediate=false).
                        let is_zero_blk = self.context.append_basic_block(block_fn, "f4s_lsr_z");
                        let nz_blk = self.context.append_basic_block(block_fn, "f4s_lsr_nz");
                        let is_lt32_blk = self.context.append_basic_block(block_fn, "f4s_lsr_lt");
                        let ge32_blk = self.context.append_basic_block(block_fn, "f4s_lsr_ge");
                        let is_eq32_blk = self.context.append_basic_block(block_fn, "f4s_lsr_eq");
                        let is_gt32_blk = self.context.append_basic_block(block_fn, "f4s_lsr_gt");

                        let is_zero = builder
                            .build_int_compare(IntPredicate::EQ, amount, zero_i32, "f4s_lsr_isz")
                            .ok()?;
                        builder.build_conditional_branch(is_zero, is_zero_blk, nz_blk).ok()?;

                        builder.position_at_end(is_zero_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(nz_blk);
                        let is_lt32 = builder
                            .build_int_compare(IntPredicate::ULT, amount, i32_t.const_int(32, false), "f4s_lsr_lt32")
                            .ok()?;
                        builder.build_conditional_branch(is_lt32, is_lt32_blk, ge32_blk).ok()?;

                        // is_lt32_blk: result = val >> amount logical; carry = (val >> (amount-1)) & 1
                        builder.position_at_end(is_lt32_blk);
                        let lt_res = builder
                            .build_right_shift(val, amount, false, "f4s_lsr_lt_res")
                            .ok()?;
                        let am_m1 = builder
                            .build_int_sub(amount, one_i32, "f4s_lsr_lt_am1")
                            .ok()?;
                        let cs = builder
                            .build_right_shift(val, am_m1, false, "f4s_lsr_lt_cs")
                            .ok()?;
                        let lt_carry = builder.build_and(cs, one_i32, "f4s_lsr_lt_c").ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(ge32_blk);
                        let is_eq32 = builder
                            .build_int_compare(IntPredicate::EQ, amount, i32_t.const_int(32, false), "f4s_lsr_eq32")
                            .ok()?;
                        builder.build_conditional_branch(is_eq32, is_eq32_blk, is_gt32_blk).ok()?;

                        // is_eq32_blk: result = 0; carry = val >> 31
                        builder.position_at_end(is_eq32_blk);
                        let eq_carry = builder
                            .build_right_shift(val, i32_t.const_int(31, false), false, "f4s_lsr_eq_c")
                            .ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // is_gt32_blk: result = 0; carry = 0
                        builder.position_at_end(is_gt32_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(join_blk);
                        let res_phi = builder.build_phi(i32_t, "f4s_lsr_res").ok()?;
                        res_phi.add_incoming(&[
                            (&val, is_zero_blk),
                            (&lt_res, is_lt32_blk),
                            (&zero_i32, is_eq32_blk),
                            (&zero_i32, is_gt32_blk),
                        ]);
                        let c_phi = builder.build_phi(i32_t, "f4s_lsr_c").ok()?;
                        c_phi.add_incoming(&[
                            (&c_old, is_zero_blk),
                            (&lt_carry, is_lt32_blk),
                            (&eq_carry, is_eq32_blk),
                            (&zero_i32, is_gt32_blk),
                        ]);
                        (res_phi.as_basic_value().into_int_value(), c_phi.as_basic_value().into_int_value())
                    }
                    4 => {
                        // ASR by amount (immediate=false).
                        // amount==0: result=val,   carry=c_old
                        // 1..=31:    result=ashr,  carry=(val>>(amount-1))&1
                        // >=32:      result=sext,  carry=val>>31
                        //   (sext: 0 or 0xFFFFFFFF — same as ashr 31)
                        let is_zero_blk = self.context.append_basic_block(block_fn, "f4s_asr_z");
                        let nz_blk = self.context.append_basic_block(block_fn, "f4s_asr_nz");
                        let is_lt32_blk = self.context.append_basic_block(block_fn, "f4s_asr_lt");
                        let is_ge32_blk = self.context.append_basic_block(block_fn, "f4s_asr_ge");

                        let is_zero = builder
                            .build_int_compare(IntPredicate::EQ, amount, zero_i32, "f4s_asr_isz")
                            .ok()?;
                        builder.build_conditional_branch(is_zero, is_zero_blk, nz_blk).ok()?;

                        builder.position_at_end(is_zero_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(nz_blk);
                        let is_lt32 = builder
                            .build_int_compare(IntPredicate::ULT, amount, i32_t.const_int(32, false), "f4s_asr_lt32")
                            .ok()?;
                        builder.build_conditional_branch(is_lt32, is_lt32_blk, is_ge32_blk).ok()?;

                        // is_lt32_blk: result = val ashr amount; carry = (val >> (amount-1)) & 1
                        builder.position_at_end(is_lt32_blk);
                        let lt_res = builder
                            .build_right_shift(val, amount, true, "f4s_asr_lt_res")
                            .ok()?;
                        let am_m1 = builder
                            .build_int_sub(amount, one_i32, "f4s_asr_lt_am1")
                            .ok()?;
                        let cs = builder
                            .build_right_shift(val, am_m1, false, "f4s_asr_lt_cs")
                            .ok()?;
                        let lt_carry = builder.build_and(cs, one_i32, "f4s_asr_lt_c").ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // is_ge32_blk: result = val ashr 31 (sign-extend); carry = val>>31 & 1
                        builder.position_at_end(is_ge32_blk);
                        let ge_res = builder
                            .build_right_shift(val, i32_t.const_int(31, false), true, "f4s_asr_ge_res")
                            .ok()?;
                        let ge_c_shift = builder
                            .build_right_shift(val, i32_t.const_int(31, false), false, "f4s_asr_ge_cs")
                            .ok()?;
                        let ge_carry = builder
                            .build_and(ge_c_shift, one_i32, "f4s_asr_ge_c")
                            .ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(join_blk);
                        let res_phi = builder.build_phi(i32_t, "f4s_asr_res").ok()?;
                        res_phi.add_incoming(&[
                            (&val, is_zero_blk),
                            (&lt_res, is_lt32_blk),
                            (&ge_res, is_ge32_blk),
                        ]);
                        let c_phi = builder.build_phi(i32_t, "f4s_asr_c").ok()?;
                        c_phi.add_incoming(&[
                            (&c_old, is_zero_blk),
                            (&lt_carry, is_lt32_blk),
                            (&ge_carry, is_ge32_blk),
                        ]);
                        (res_phi.as_basic_value().into_int_value(), c_phi.as_basic_value().into_int_value())
                    }
                    7 => {
                        // ROR by amount (immediate=false, rrx=true). With
                        // immediate=false the rrx path (amount==0 special-
                        // case) is never reached: scalar's `match amount {
                        // 0 => if immediate & rrx ... else val }`
                        // resolves to `val` since immediate=false.
                        // amount==0: result=val,                 carry=c_old
                        // amount>0:  m = amount % 32
                        //   m==0: result=val,                    carry=val>>31
                        //   else: result=rotate_right(val, m),   carry=result>>31
                        let is_zero_blk = self.context.append_basic_block(block_fn, "f4s_ror_z");
                        let nz_blk = self.context.append_basic_block(block_fn, "f4s_ror_nz");
                        let m_zero_blk = self.context.append_basic_block(block_fn, "f4s_ror_m0");
                        let m_nz_blk = self.context.append_basic_block(block_fn, "f4s_ror_mnz");

                        let is_zero = builder
                            .build_int_compare(IntPredicate::EQ, amount, zero_i32, "f4s_ror_isz")
                            .ok()?;
                        builder.build_conditional_branch(is_zero, is_zero_blk, nz_blk).ok()?;

                        builder.position_at_end(is_zero_blk);
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // nz_blk: m = amount & 0x1f  (mod 32). m==0 ?
                        builder.position_at_end(nz_blk);
                        let m = builder
                            .build_and(amount, i32_t.const_int(0x1f, false), "f4s_ror_m")
                            .ok()?;
                        let m_is_zero = builder
                            .build_int_compare(IntPredicate::EQ, m, zero_i32, "f4s_ror_misz")
                            .ok()?;
                        builder.build_conditional_branch(m_is_zero, m_zero_blk, m_nz_blk).ok()?;

                        // m_zero_blk: result = val; carry = val >> 31 & 1
                        builder.position_at_end(m_zero_blk);
                        let m0_c_shift = builder
                            .build_right_shift(val, i32_t.const_int(31, false), false, "f4s_ror_m0_cs")
                            .ok()?;
                        let m0_carry = builder
                            .build_and(m0_c_shift, one_i32, "f4s_ror_m0_c")
                            .ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        // m_nz_blk: result = (val >> m) | (val << (32 - m));
                        //           carry  = result >> 31 & 1.
                        // Both shifts have m in 1..=31 so neither poisons.
                        builder.position_at_end(m_nz_blk);
                        let lo = builder
                            .build_right_shift(val, m, false, "f4s_ror_lo")
                            .ok()?;
                        let neg_m = builder
                            .build_int_sub(i32_t.const_int(32, false), m, "f4s_ror_negm")
                            .ok()?;
                        let hi = builder
                            .build_left_shift(val, neg_m, "f4s_ror_hi")
                            .ok()?;
                        let mnz_res = builder
                            .build_or(lo, hi, "f4s_ror_mnz_res")
                            .ok()?;
                        let mnz_c_shift = builder
                            .build_right_shift(mnz_res, i32_t.const_int(31, false), false, "f4s_ror_mnz_cs")
                            .ok()?;
                        let mnz_carry = builder
                            .build_and(mnz_c_shift, one_i32, "f4s_ror_mnz_c")
                            .ok()?;
                        builder.build_unconditional_branch(join_blk).ok()?;

                        builder.position_at_end(join_blk);
                        let res_phi = builder.build_phi(i32_t, "f4s_ror_res").ok()?;
                        res_phi.add_incoming(&[
                            (&val, is_zero_blk),
                            (&val, m_zero_blk),
                            (&mnz_res, m_nz_blk),
                        ]);
                        let c_phi = builder.build_phi(i32_t, "f4s_ror_c").ok()?;
                        c_phi.add_incoming(&[
                            (&c_old, is_zero_blk),
                            (&m0_carry, m_zero_blk),
                            (&mnz_carry, m_nz_blk),
                        ]);
                        (res_phi.as_basic_value().into_int_value(), c_phi.as_basic_value().into_int_value())
                    }
                    _ => unreachable!(),
                };

                // bus.idle_cycle() — matches scalar's `self.idle_cycle()`.
                builder.build_call(idle, &[cpu_ctx.into()], "f4s_idle").ok()?;

                // gpr[Rd] = result. Always writeback (no setting-flags-no-writeback for shifts).
                builder.build_store(gpr_rd_ptr, result).ok()?;

                // cpsr: clear N|Z|C, preserve V; set N from result bit 31,
                // Z from (result==0), C from carry. arithmetic=false → V untouched.
                let nzc_clear = i32_t.const_int(0x1fff_ffff, false);
                let cleared = builder.build_and(cpsr_old, nzc_clear, "f4s_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f4s_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, zero_i32, "f4s_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f4s_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f4s_z_bit")
                    .ok()?;
                let c_bit = builder
                    .build_left_shift(carry_i32, i32_t.const_int(29, false), "f4s_c_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f4s_cpsr_n").ok()?;
                let cpsr_nz = builder.build_or(cpsr_n, z_bit, "f4s_cpsr_nz").ok()?;
                let cpsr_new = builder.build_or(cpsr_nz, c_bit, "f4s_cpsr_new").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f4s_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f4s_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f4s_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F4 ALU logical sub-ops only:
            // AND(0), EOR(1), TST(8), ORR(12), BIC(14), MVN(15).
            // Encoding: 010000_OOOO_SSS_DDD; mask 0xfc00 == 0x4000.
            // Effects (mirrors arm7tdmi/src/cpu.rs F4 path; logical
            // ops use arithmetic=false so C+V are preserved):
            //   src = gpr[Rs]; dst = gpr[Rd]
            //   AND: result = dst & src         (writeback)
            //   EOR: result = dst ^ src         (writeback)
            //   TST: result = dst & src         (no writeback)
            //   ORR: result = dst | src         (writeback)
            //   BIC: result = dst & ~src        (writeback)
            //   MVN: result = ~src              (writeback)
            //   cpsr: N from result bit 31, Z from result==0, C+V preserved.
            //   pc = fetch_addr + 2; nfa = Seq.
            // ADC/SBC/NEG/CMP/CMN/shifts/MUL not in this commit (different
            // flag/cycle semantics).
            let f4_top6 = (opcode >> 10) & 0x3f;
            let f4_op = (opcode >> 6) & 0xf;
            let f4_logical = matches!(f4_op, 0 | 1 | 8 | 12 | 14 | 15);
            if inline_enabled && f4_log_inline_env && f4_top6 == 0b010000 && f4_logical {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rs].
                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f4l_rs_ptr").ok()?
                };
                let src = builder.build_load(i32_t, gpr_rs_ptr, "f4l_src").ok()?
                    .into_int_value();

                // gpr[Rd] ptr (load only when needed; some ops don't use dst).
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f4l_rd_ptr").ok()?
                };

                // MVN doesn't read dst; the others do.
                let need_dst = f4_op != 15;
                let dst_opt = if need_dst {
                    Some(builder.build_load(i32_t, gpr_rd_ptr, "f4l_dst").ok()?
                        .into_int_value())
                } else {
                    None
                };

                let result = match f4_op {
                    0 | 8 => builder.build_and(dst_opt.unwrap(), src, "f4l_and").ok()?,
                    1 => builder.build_xor(dst_opt.unwrap(), src, "f4l_eor").ok()?,
                    12 => builder.build_or(dst_opt.unwrap(), src, "f4l_orr").ok()?,
                    14 => {
                        let not_src = builder.build_not(src, "f4l_not_src").ok()?;
                        builder.build_and(dst_opt.unwrap(), not_src, "f4l_bic").ok()?
                    }
                    15 => builder.build_not(src, "f4l_mvn").ok()?,
                    _ => unreachable!(),
                };

                // Writeback unless TST (op=8).
                if f4_op != 8 {
                    builder.build_store(gpr_rd_ptr, result).ok()?;
                }

                // cpsr: clear N|Z (preserve C+V), set N+Z from result.
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f4l_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f4l_cpsr_old").ok()?
                    .into_int_value();
                let nz_clear = i32_t.const_int(0x3fff_ffff, false); // clear bits 31, 30
                let cleared = builder.build_and(cpsr_old, nz_clear, "f4l_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f4l_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, i32_t.const_int(0, false), "f4l_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f4l_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f4l_z_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f4l_cpsr_n").ok()?;
                let cpsr_new = builder.build_or(cpsr_n, z_bit, "f4l_cpsr_nz").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f4l_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f4l_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f4l_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F4 ALU arithmetic sub-ops:
            // ADC(5), SBC(6), NEG(9), CMP(10), CMN(11).
            // Encoding: 010000_OOOO_SSS_DDD; mask 0xfc00 == 0x4000.
            // Effects (mirrors arm7tdmi/src/cpu.rs F4 path + alu helpers):
            //   src = gpr[Rs]; dst = gpr[Rd] (only ADC/SBC/CMP need dst)
            //   ADC: 64-bit (dst + src + cin); carry = bit 32; ovf = !(a^b)&(b^res) bit31
            //   SBC: same as ADC with b' = ~src
            //   NEG: a=0, b=src, SUB-style: res=-src; carry = (src==0); ovf = (a^b)&(a^res)
            //   CMP: a=dst, b=src, SUB-style; no writeback
            //   CMN: a=dst, b=src, ADD-style; no writeback
            //   cpsr: clear N|Z|C|V; set N from result bit 31, Z from
            //         (result==0), C from carry, V from overflow.
            let f4a_top6 = (opcode >> 10) & 0x3f;
            let f4a_op = (opcode >> 6) & 0xf;
            let f4_arith = matches!(f4a_op, 5 | 6 | 9 | 10 | 11);
            if inline_enabled && f4_arith_inline_env && f4a_top6 == 0b010000 && f4_arith {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load src = gpr[Rs] always.
                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f4a_rs_ptr").ok()?
                };
                let src = builder.build_load(i32_t, gpr_rs_ptr, "f4a_src").ok()?
                    .into_int_value();

                // gpr[Rd] ptr (load dst when needed: ADC/SBC/CMP/CMN; NEG uses 0).
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f4a_rd_ptr").ok()?
                };
                let zero_i32 = i32_t.const_int(0, false);
                let dst = if f4a_op == 9 {
                    zero_i32
                } else {
                    builder.build_load(i32_t, gpr_rd_ptr, "f4a_dst").ok()?
                        .into_int_value()
                };

                // For ADC/SBC: read carry-in from cpsr bit 29.
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f4a_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f4a_cpsr_old").ok()?
                    .into_int_value();

                let i64_t = self.context.i64_type();
                let (result, carry_i32, ovf_i32) = match f4a_op {
                    5 | 6 => {
                        // ADC / SBC: 64-bit add of (a, b_or_notb, cin).
                        let b_op = if f4a_op == 5 {
                            src
                        } else {
                            // SBC: b' = ~src
                            builder.build_not(src, "f4a_sbc_notb").ok()?
                        };
                        // a, b zero-ext to i64.
                        let a64 = builder.build_int_z_extend(dst, i64_t, "f4a_a64").ok()?;
                        let b64 = builder.build_int_z_extend(b_op, i64_t, "f4a_b64").ok()?;
                        // c_in = (cpsr_old >> 29) & 1, zero-ext to i64.
                        let c_shift = builder
                            .build_right_shift(cpsr_old, i32_t.const_int(29, false), false, "f4a_cin_s")
                            .ok()?;
                        let c_in_i32 = builder
                            .build_and(c_shift, i32_t.const_int(1, false), "f4a_cin_i32")
                            .ok()?;
                        let c_in = builder.build_int_z_extend(c_in_i32, i64_t, "f4a_cin").ok()?;
                        let sum = builder.build_int_add(a64, b64, "f4a_sum_ab").ok()?;
                        let sum = builder.build_int_add(sum, c_in, "f4a_sum_abc").ok()?;
                        // result = low 32.
                        let result = builder.build_int_truncate(sum, i32_t, "f4a_res").ok()?;
                        // carry = (sum >> 32) bit 0.
                        let carry_shift = builder
                            .build_right_shift(sum, i64_t.const_int(32, false), false, "f4a_c_s64")
                            .ok()?;
                        let carry_trunc = builder.build_int_truncate(carry_shift, i32_t, "f4a_c_tr").ok()?;
                        let carry_i32 = builder
                            .build_and(carry_trunc, i32_t.const_int(1, false), "f4a_c_i32")
                            .ok()?;
                        // overflow = (!(a ^ b) & (b ^ result)) bit 31.
                        let ab_xor = builder.build_xor(dst, b_op, "f4a_ab_xor").ok()?;
                        let ab_xor_not = builder.build_not(ab_xor, "f4a_ab_nxor").ok()?;
                        let br_xor = builder.build_xor(b_op, result, "f4a_br_xor").ok()?;
                        let ovf_and = builder.build_and(ab_xor_not, br_xor, "f4a_ovf_and").ok()?;
                        let ovf_shift = builder
                            .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f4a_ovf_s")
                            .ok()?;
                        let ovf_i32 = builder
                            .build_and(ovf_shift, i32_t.const_int(1, false), "f4a_ovf")
                            .ok()?;
                        (result, carry_i32, ovf_i32)
                    }
                    9 | 10 => {
                        // NEG (a=0, b=src), CMP (a=dst, b=src) — both SUB-style.
                        let a = dst;
                        let b = src;
                        let result = builder.build_int_sub(a, b, "f4a_sub_res").ok()?;
                        let carry_cmp = builder
                            .build_int_compare(IntPredicate::UGE, a, b, "f4a_sub_c_cmp")
                            .ok()?;
                        let carry_i32 = builder.build_int_z_extend(carry_cmp, i32_t, "f4a_sub_c").ok()?;
                        let ab_xor = builder.build_xor(a, b, "f4a_sub_ab").ok()?;
                        let ar_xor = builder.build_xor(a, result, "f4a_sub_ar").ok()?;
                        let ovf_and = builder.build_and(ab_xor, ar_xor, "f4a_sub_ovf_and").ok()?;
                        let ovf_shift = builder
                            .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f4a_sub_ovf_s")
                            .ok()?;
                        let ovf_i32 = builder
                            .build_and(ovf_shift, i32_t.const_int(1, false), "f4a_sub_ovf")
                            .ok()?;
                        (result, carry_i32, ovf_i32)
                    }
                    11 => {
                        // CMN: ADD-style (no writeback).
                        let a = dst;
                        let b = src;
                        let result = builder.build_int_add(a, b, "f4a_add_res").ok()?;
                        let carry_cmp = builder
                            .build_int_compare(IntPredicate::ULT, result, a, "f4a_add_c_cmp")
                            .ok()?;
                        let carry_i32 = builder.build_int_z_extend(carry_cmp, i32_t, "f4a_add_c").ok()?;
                        let ra_xor = builder.build_xor(result, a, "f4a_add_ra").ok()?;
                        let rb_xor = builder.build_xor(result, b, "f4a_add_rb").ok()?;
                        let ovf_and = builder.build_and(ra_xor, rb_xor, "f4a_add_ovf_and").ok()?;
                        let ovf_shift = builder
                            .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f4a_add_ovf_s")
                            .ok()?;
                        let ovf_i32 = builder
                            .build_and(ovf_shift, i32_t.const_int(1, false), "f4a_add_ovf")
                            .ok()?;
                        (result, carry_i32, ovf_i32)
                    }
                    _ => unreachable!(),
                };

                // Writeback unless CMP(10) or CMN(11).
                if !matches!(f4a_op, 10 | 11) {
                    builder.build_store(gpr_rd_ptr, result).ok()?;
                }

                // cpsr: clear N|Z|C|V, set from result/carry/ovf.
                let nzcv_clear = i32_t.const_int(0x0fff_ffff, false);
                let cleared = builder.build_and(cpsr_old, nzcv_clear, "f4a_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f4a_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, zero_i32, "f4a_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f4a_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f4a_z_bit")
                    .ok()?;
                let c_bit = builder
                    .build_left_shift(carry_i32, i32_t.const_int(29, false), "f4a_c_bit")
                    .ok()?;
                let v_bit = builder
                    .build_left_shift(ovf_i32, i32_t.const_int(28, false), "f4a_v_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f4a_cpsr_n").ok()?;
                let cpsr_nz = builder.build_or(cpsr_n, z_bit, "f4a_cpsr_nz").ok()?;
                let cpsr_nzc = builder.build_or(cpsr_nz, c_bit, "f4a_cpsr_nzc").ok()?;
                let cpsr_new = builder.build_or(cpsr_nzc, v_bit, "f4a_cpsr_new").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f4a_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f4a_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f4a_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F4 MUL (op=0b1101).
            // Encoding: 010000_1101_SSS_DDD; mask 0xffc0 == 0x4340.
            // Effects (mirrors arm7tdmi/src/cpu.rs aot_thumb_step F4 MUL):
            //   src = gpr[Rs]; dst = gpr[Rd]
            //   m = get_required_multipiler_array_cycles(src):
            //       if (src & 0xff)       == src → 1
            //       elif (src & 0xffff)   == src → 2
            //       elif (src & 0xffffff) == src → 3
            //       else                         → 4
            //   for k in 0..m: bus.idle_cycle()         (BEFORE mul)
            //   result = dst.wrapping_mul(src)
            //   gpr[Rd] = result
            //   cpsr: clear N|Z|C|V; set N from result bit 31, Z from
            //         (result==0); C=0, V=0  (alu_update_flags(_, false, false, false))
            //   pc = fetch_addr + 2; nfa = Seq.
            //
            // Cycle-accounting strategy (PER COMMIT 58e054b POSTMORTEM):
            // The original F4 MUL attempt did `*ts_ptr += m` inline. That
            // produced 3 extra MK divs at sw=64KB. Hypothesis: scalar
            // bus.idle_cycle() pumps the scheduler `m` times (one tick
            // each), so any scheduler event whose deadline falls within
            // [ts0, ts0+m] fires AT the matching tick. The inline bulk
            // add jumps ts straight from ts0 → ts0+m, so the event fires
            // once at ts0+m instead of at its real deadline. fb_hash
            // diverges at frames where MUL straddles a scheduler deadline.
            // Fix: replicate scalar exactly — call aot_idle_cycle_for
            // up to 4 times, gated by m. Bus path is known-correct (it's
            // what scalar uses).
            let f4m_top6 = (opcode >> 10) & 0x3f;
            let f4m_op = (opcode >> 6) & 0xf;
            if inline_enabled
                && f4_mul_inline_env
                && idle_cycle_ref.is_some()
                && f4m_top6 == 0b010000
                && f4m_op == 0b1101
            {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let idle = idle_cycle_ref.unwrap();
                let rs = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting (fetch) + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load src = gpr[Rs], dst = gpr[Rd].
                let gpr_rs_off = (off.gpr + rs * 4) as u64;
                let gpr_rs_off_v = i32_t.const_int(gpr_rs_off, false);
                let gpr_rs_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rs_off_v], "f4m_rs_ptr").ok()?
                };
                let src = builder.build_load(i32_t, gpr_rs_ptr, "f4m_src").ok()?
                    .into_int_value();

                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f4m_rd_ptr").ok()?
                };
                let dst = builder.build_load(i32_t, gpr_rd_ptr, "f4m_dst").ok()?
                    .into_int_value();

                // Compute m via three bit-mask checks. Scalar formula
                // (cpu.rs::get_required_multipiler_array_cycles):
                //   if rs & 0xff       == rs → 1
                //   elif rs & 0xffff   == rs → 2
                //   elif rs & 0xffffff == rs → 3
                //   else                     → 4
                // Equivalent: count how many of (src&~0xff != 0,
                // src&~0xffff != 0, src&~0xffffff != 0) are true; m = 1+count.
                // We just chain idle_cycle calls via conditional branches.
                //
                // Always 1 idle (m >= 1 is unconditional).
                // Branch on (src & 0xffffff00) != 0 → idle 2nd time.
                // Branch on (src & 0xffff0000) != 0 → idle 3rd time.
                // Branch on (src & 0xff000000) != 0 → idle 4th time.
                // Each ladder rung is a conditional br + cont label.

                let zero_i32 = i32_t.const_int(0, false);

                // First idle (always).
                builder.build_call(idle, &[cpu_ctx.into()], "f4m_idle1").ok()?;

                // Rung 2: m >= 2 ⇔ src & 0xffffff00 != 0.
                let mask_ge2 = builder
                    .build_and(src, i32_t.const_int(0xffff_ff00u64, false), "f4m_m2_msk")
                    .ok()?;
                let is_ge2 = builder
                    .build_int_compare(IntPredicate::NE, mask_ge2, zero_i32, "f4m_m2_cmp")
                    .ok()?;
                let m2_blk = self.context.append_basic_block(block_fn, "f4m_m2");
                let after_m2 = self.context.append_basic_block(block_fn, "f4m_a2");
                builder.build_conditional_branch(is_ge2, m2_blk, after_m2).ok()?;
                builder.position_at_end(m2_blk);
                builder.build_call(idle, &[cpu_ctx.into()], "f4m_idle2").ok()?;
                builder.build_unconditional_branch(after_m2).ok()?;
                builder.position_at_end(after_m2);

                // Rung 3: m >= 3 ⇔ src & 0xffff0000 != 0.
                let mask_ge3 = builder
                    .build_and(src, i32_t.const_int(0xffff_0000u64, false), "f4m_m3_msk")
                    .ok()?;
                let is_ge3 = builder
                    .build_int_compare(IntPredicate::NE, mask_ge3, zero_i32, "f4m_m3_cmp")
                    .ok()?;
                let m3_blk = self.context.append_basic_block(block_fn, "f4m_m3");
                let after_m3 = self.context.append_basic_block(block_fn, "f4m_a3");
                builder.build_conditional_branch(is_ge3, m3_blk, after_m3).ok()?;
                builder.position_at_end(m3_blk);
                builder.build_call(idle, &[cpu_ctx.into()], "f4m_idle3").ok()?;
                builder.build_unconditional_branch(after_m3).ok()?;
                builder.position_at_end(after_m3);

                // Rung 4: m >= 4 ⇔ src & 0xff000000 != 0.
                let mask_ge4 = builder
                    .build_and(src, i32_t.const_int(0xff00_0000u64, false), "f4m_m4_msk")
                    .ok()?;
                let is_ge4 = builder
                    .build_int_compare(IntPredicate::NE, mask_ge4, zero_i32, "f4m_m4_cmp")
                    .ok()?;
                let m4_blk = self.context.append_basic_block(block_fn, "f4m_m4");
                let after_m4 = self.context.append_basic_block(block_fn, "f4m_a4");
                builder.build_conditional_branch(is_ge4, m4_blk, after_m4).ok()?;
                builder.position_at_end(m4_blk);
                builder.build_call(idle, &[cpu_ctx.into()], "f4m_idle4").ok()?;
                builder.build_unconditional_branch(after_m4).ok()?;
                builder.position_at_end(after_m4);

                // result = dst * src (LLVM mul wraps i32 by definition).
                let result = builder.build_int_mul(dst, src, "f4m_res").ok()?;

                // gpr[Rd] = result. (MUL is not a setting-flags-no-writeback op.)
                builder.build_store(gpr_rd_ptr, result).ok()?;

                // cpsr: clear N|Z|C|V (bits 31..28), set N from result
                // bit 31, Z from (result == 0). C=0, V=0 — alu_update_flags
                // is called with arithmetic=false, c=false, v=false; the
                // helper unconditionally writes ALL FOUR flags.
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f4m_cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f4m_cpsr_old").ok()?
                    .into_int_value();
                let nzcv_clear = i32_t.const_int(0x0fff_ffff, false);
                let cleared = builder.build_and(cpsr_old, nzcv_clear, "f4m_cpsr_cl").ok()?;
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "f4m_n_bit")
                    .ok()?;
                let z_cmp = builder
                    .build_int_compare(IntPredicate::EQ, result, zero_i32, "f4m_z_cmp")
                    .ok()?;
                let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f4m_z_ext").ok()?;
                let z_bit = builder
                    .build_left_shift(z_ext, i32_t.const_int(30, false), "f4m_z_bit")
                    .ok()?;
                let cpsr_n = builder.build_or(cleared, n_bit, "f4m_cpsr_n").ok()?;
                let cpsr_new = builder.build_or(cpsr_n, z_bit, "f4m_cpsr_nz").ok()?;
                builder.build_store(cpsr_ptr, cpsr_new).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f4m_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f4m_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f4m_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F5 high-reg ADD/CMP/MOV (AdvancePC subset).
            // Encoding: 010001_OO_H1_H2_SSS_DDD; mask 0xfc00 == 0x4400.
            //   OO ∈ {ADD=0, CMP=1, MOV=2, BX=3}.
            //   dst_reg = if h1 then rd_low+8 else rd_low.
            //   src_reg = if h2 then rs_low+8 else rs_low.
            //   ADD/MOV with dst_reg == 15 → PipelineFlushed (block terminator).
            //   BX (op=3) → always PipelineFlushed.
            // We inline ONLY the AdvancePC subset:
            //   - ADD with dst_reg != 15
            //   - CMP (no writeback; full flag update via alu_sub_flags)
            //   - MOV with dst_reg != 15
            // Block terminators (BX, ADD/MOV with R15) fall through to the
            // step trampoline path. At AOT compile time we know op/h1/rd_low
            // so the inline emit fires only for the safe subset.
            // src_reg == 15: get_reg(15) returns pipeline-head pc = fetch_addr.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F5 path.
            if inline_enabled
                && f5_inline_env
                && (opcode & 0xfc00) == 0x4400
            {
                let op = ((opcode >> 8) & 0x3) as u32;
                let h1 = (opcode >> 7) & 0x1;
                let h2 = (opcode >> 6) & 0x1;
                let rs_low = ((opcode >> 3) & 0x7) as u32;
                let rd_low = (opcode & 0x7) as u32;
                let dst_reg = if h1 == 1 { rd_low + 8 } else { rd_low };
                let src_reg = if h2 == 1 { rs_low + 8 } else { rs_low };
                let is_add_or_mov = op == 0 || op == 2;
                let dst_is_pc = dst_reg == 15;
                // Skip block-terminator cases — let step trampoline handle.
                if op != 3 && !(is_add_or_mov && dst_is_pc) {
                    let off = offsets.unwrap();
                    let fo = fetch_only_ref.unwrap();

                    // fetch_only.
                    builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                    // Read op2 = gpr[src_reg]. If src_reg==15, baked fetch_addr.
                    let op2 = if src_reg == 15 {
                        i32_t.const_int(fetch_addr as u64, false)
                    } else {
                        let gpr_src_off = (off.gpr + src_reg * 4) as u64;
                        let gpr_src_off_v = i32_t.const_int(gpr_src_off, false);
                        let gpr_src_ptr = unsafe {
                            builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_src_off_v], "f5_src_ptr").ok()?
                        };
                        builder.build_load(i32_t, gpr_src_ptr, "f5_op2").ok()?
                            .into_int_value()
                    };

                    // For ADD or CMP we also need op1 = gpr[dst_reg]. MOV
                    // doesn't read op1.
                    let op1 = if op == 0 || op == 1 {
                        if dst_reg == 15 {
                            // CMP can have dst_reg=15 (uses pc); ADD never
                            // reaches here with dst_reg=15 (skipped above).
                            Some(i32_t.const_int(fetch_addr as u64, false))
                        } else {
                            let gpr_dst_off = (off.gpr + dst_reg * 4) as u64;
                            let gpr_dst_off_v = i32_t.const_int(gpr_dst_off, false);
                            let gpr_dst_ptr = unsafe {
                                builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_dst_off_v], "f5_dst_ptr").ok()?
                            };
                            Some(builder.build_load(i32_t, gpr_dst_ptr, "f5_op1").ok()?
                                .into_int_value())
                        }
                    } else {
                        None
                    };

                    match op {
                        0 => {
                            // ADD: gpr[dst_reg] = op1 + op2 (no flag update).
                            // (dst_reg != 15 here.)
                            let result = builder.build_int_add(op1.unwrap(), op2, "f5_add").ok()?;
                            let gpr_dst_off = (off.gpr + dst_reg * 4) as u64;
                            let gpr_dst_off_v = i32_t.const_int(gpr_dst_off, false);
                            let gpr_dst_ptr = unsafe {
                                builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_dst_off_v], "f5_dst_w_ptr").ok()?
                            };
                            builder.build_store(gpr_dst_ptr, result).ok()?;
                        }
                        1 => {
                            // CMP: full N|Z|C|V flag update from op1 - op2.
                            // No writeback. Mirrors F2 SUB / F4_ARITH CMP.
                            let result = builder.build_int_sub(op1.unwrap(), op2, "f5_cmp_res").ok()?;
                            // carry = op1 >= op2 (unsigned)
                            let carry_cmp = builder
                                .build_int_compare(IntPredicate::UGE, op1.unwrap(), op2, "f5_cmp_c")
                                .ok()?;
                            let carry_i32 = builder.build_int_z_extend(carry_cmp, i32_t, "f5_cmp_ci").ok()?;
                            // overflow = ((op1^op2) & (op1^result)) bit 31
                            let ab_xor = builder.build_xor(op1.unwrap(), op2, "f5_cmp_ab").ok()?;
                            let ar_xor = builder.build_xor(op1.unwrap(), result, "f5_cmp_ar").ok()?;
                            let ovf_and = builder.build_and(ab_xor, ar_xor, "f5_cmp_ovf_and").ok()?;
                            let ovf_shift = builder
                                .build_right_shift(ovf_and, i32_t.const_int(31, false), false, "f5_cmp_ovf_s")
                                .ok()?;
                            let ovf_i32 = builder
                                .build_and(ovf_shift, i32_t.const_int(1, false), "f5_cmp_ovf")
                                .ok()?;
                            // cpsr update: clear N|Z|C|V, set from result/carry/ovf.
                            let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                            let cpsr_ptr = unsafe {
                                builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "f5_cpsr_ptr").ok()?
                            };
                            let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "f5_cpsr_old").ok()?
                                .into_int_value();
                            let nzcv_clear = i32_t.const_int(0x0fff_ffff, false);
                            let cleared = builder.build_and(cpsr_old, nzcv_clear, "f5_cpsr_cl").ok()?;
                            let n_bit = builder
                                .build_and(result, i32_t.const_int(0x8000_0000, false), "f5_n_bit")
                                .ok()?;
                            let z_cmp = builder
                                .build_int_compare(IntPredicate::EQ, result, i32_t.const_int(0, false), "f5_z_cmp")
                                .ok()?;
                            let z_ext = builder.build_int_z_extend(z_cmp, i32_t, "f5_z_ext").ok()?;
                            let z_bit = builder
                                .build_left_shift(z_ext, i32_t.const_int(30, false), "f5_z_bit")
                                .ok()?;
                            let c_bit = builder
                                .build_left_shift(carry_i32, i32_t.const_int(29, false), "f5_c_bit")
                                .ok()?;
                            let v_bit = builder
                                .build_left_shift(ovf_i32, i32_t.const_int(28, false), "f5_v_bit")
                                .ok()?;
                            let cpsr_n = builder.build_or(cleared, n_bit, "f5_cpsr_n").ok()?;
                            let cpsr_nz = builder.build_or(cpsr_n, z_bit, "f5_cpsr_nz").ok()?;
                            let cpsr_nzc = builder.build_or(cpsr_nz, c_bit, "f5_cpsr_nzc").ok()?;
                            let cpsr_new = builder.build_or(cpsr_nzc, v_bit, "f5_cpsr_new").ok()?;
                            builder.build_store(cpsr_ptr, cpsr_new).ok()?;
                        }
                        2 => {
                            // MOV: gpr[dst_reg] = op2.  (dst_reg != 15 here.)
                            let gpr_dst_off = (off.gpr + dst_reg * 4) as u64;
                            let gpr_dst_off_v = i32_t.const_int(gpr_dst_off, false);
                            let gpr_dst_ptr = unsafe {
                                builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_dst_off_v], "f5_mov_dst_ptr").ok()?
                            };
                            builder.build_store(gpr_dst_ptr, op2).ok()?;
                        }
                        _ => unreachable!(),
                    }

                    // pc = fetch_addr + 2; nfa = Seq.
                    let pc_off_v = i32_t.const_int(off.pc as u64, false);
                    let pc_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f5_pc_ptr").ok()?
                    };
                    builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f5_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                    let cont_blk = self.context.append_basic_block(block_fn, "f5_cont");
                    builder.build_unconditional_branch(cont_blk).ok()?;
                    builder.position_at_end(cont_blk);
                    continue;
                }
                // else: fall through to step trampoline.
            }

            // Phase-4 inline IR for F6 LDR PC-relative (literal pool).
            // Encoding: 01001_DDD_IIIIIIII; mask 0xf800 == 0x4800.
            //   imm = (insn & 0xff) << 2  (constant, word-scaled)
            //   addr = (fetch_addr & ~3) + imm
            //          (cpu.pc at handler entry = fetch_addr; (pc & ~3) is
            //           always 4-aligned, so addr is always 4-aligned —
            //           never triggers the I14 misaligned-LDR ROR path)
            //   gpr[Rd] = bus.load_32(addr, NonSeq)
            //   bus.idle_cycle()    (1S+1N+1I per execution)
            //   pc = fetch_addr + 2; nfa = NonSeq (= 0)  ← different from
            //   most formats which set nfa = Seq.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F6 path.
            if inline_enabled
                && f6_inline_env
                && load_32_ref.is_some()
                && idle_cycle_ref.is_some()
                && (opcode & 0xf800) == 0x4800
            {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let l32 = load_32_ref.unwrap();
                let idle = idle_cycle_ref.unwrap();
                let rd = ((opcode >> 8) & 0x7) as u32;
                let imm = ((opcode & 0xff) as u32) << 2;
                // addr is constant at AOT compile time.
                let addr_const = (fetch_addr & !0b11).wrapping_add(imm);

                // Cycle accounting (fetch) + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // val = bus.load_32(addr_const, NonSeq).  access byte = 0.
                let addr_v = i32_t.const_int(addr_const as u64, false);
                let access_v = i8_t.const_int(0, false);
                let lcall = builder
                    .build_call(l32, &[cpu_ctx.into(), addr_v.into(), access_v.into()], "f6_load")
                    .ok()?;
                let val = lcall.try_as_basic_value().unwrap_basic().into_int_value();

                // bus.idle_cycle().
                builder.build_call(idle, &[cpu_ctx.into()], "f6_idle").ok()?;

                // gpr[Rd] = val.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f6_gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, val).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f6_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = NonSeq (= 0).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f6_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f6_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F11 LDR SP-relative (word).
            // Encoding: 1001_1_DDD_IIIIIIII; mask 0xf800 == 0x9800.
            //   imm = (insn & 0xff) << 2  (constant, word-scaled)
            //   addr = gpr[SP] + imm  (runtime base, constant offset)
            //   data = ldr_word(addr, NonSeq)
            //          (extern handles I14 misaligned-LDR ROR + cpsr.C)
            //   bus.idle_cycle()    (1S+1N+1I cycle profile)
            //   gpr[Rd] = data
            //   pc = fetch_addr + 2; nfa = Seq (= 1)  ← differs from STR
            //   cpsr.C touched only inside ldr_word when addr & 3 != 0,
            //   per I14. IR caller doesn't separately update flags.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F11 LDR path.
            // Both F11 LDR (mask 0x9800) and F11 STR (mask 0x9000) share
            // top4 = 0x9 but the masks are disjoint so order doesn't
            // matter; LDR placed first to mirror typical ld-first reading.
            if inline_enabled
                && f11_ldr_inline_env
                && ldr_word_ref.is_some()
                && idle_cycle_ref.is_some()
                && (opcode & 0xf800) == 0x9800
            {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let lw = ldr_word_ref.unwrap();
                let idle = idle_cycle_ref.unwrap();
                let rd = ((opcode >> 8) & 0x7) as u32;
                let imm = ((opcode & 0xff) as u32) << 2;

                // Cycle accounting (fetch) + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[SP] (REG_SP = 13).
                let sp_off = (off.gpr + 13 * 4) as u64;
                let sp_off_v = i32_t.const_int(sp_off, false);
                let sp_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[sp_off_v], "f11l_sp_ptr").ok()?
                };
                let sp_val = builder.build_load(i32_t, sp_ptr, "f11l_sp_val").ok()?
                    .into_int_value();

                // addr = SP + imm.
                let addr = builder
                    .build_int_add(sp_val, i32_t.const_int(imm as u64, false), "f11l_addr")
                    .ok()?;

                // data = ldr_word(addr, NonSeq=0). Extern handles I14 ROR
                // + cpsr.C side effect when addr & 3 != 0.
                let access_v = i8_t.const_int(0, false);
                let lcall = builder.build_call(
                    lw,
                    &[cpu_ctx.into(), addr.into(), access_v.into()],
                    "f11l_load",
                ).ok()?;
                let data = lcall.try_as_basic_value().unwrap_basic().into_int_value();

                // bus.idle_cycle() — match scalar order (ldr_word, idle,
                // gpr store).
                builder.build_call(idle, &[cpu_ctx.into()], "f11l_idle").ok()?;

                // gpr[Rd] = data.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f11l_gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, data).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f11l_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq (= 1) — LDR sets Seq, STR sets NonSeq.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f11l_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f11l_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F10 LDRH/STRH imm5*2-offset.
            // Encoding: 1000_L_IIIII_BBB_DDD; mask 0xf000 == 0x8000.
            //   L (bit 11): 0 = STRH, 1 = LDRH
            //   offset = imm5 << 1  (constant, halfword-scaled)
            //   addr = gpr[Rb] + offset (runtime + constant)
            //   LDRH: gpr[Rd] = ldr_half(addr, NonSeq); idle_cycle; nfa = Seq.
            //         (extern handles misaligned-LDRH ROR + cpsr.C)
            //   STRH: store_16(addr & ~1, gpr[Rd] as u16, NonSeq); nfa = NonSeq.
            //   pc = fetch_addr + 2; no flag updates (LDRH may set cpsr.C
            //   internally on misaligned).
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F10 path.
            if inline_enabled
                && f10_inline_env
                && (opcode & 0xf000) == 0x8000
            {
                let load = (opcode >> 11) & 0x1 != 0;
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let imm5 = ((opcode >> 6) & 0x1f) as u32;
                let rb = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;
                let offset_const = imm5 << 1;

                // Cycle accounting + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rb] (base).
                let gpr_rb_off = (off.gpr + rb * 4) as u64;
                let gpr_rb_off_v = i32_t.const_int(gpr_rb_off, false);
                let gpr_rb_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rb_off_v], "f10_rb_ptr").ok()?
                };
                let base = builder.build_load(i32_t, gpr_rb_ptr, "f10_base").ok()?
                    .into_int_value();

                // addr = base + offset_const.
                let addr = builder
                    .build_int_add(base, i32_t.const_int(offset_const as u64, false), "f10_addr")
                    .ok()?;

                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f10_rd_ptr").ok()?
                };

                if load {
                    // LDRH: data = ldr_half(addr, NonSeq).
                    let lh = ldr_half_ref?;
                    let idle = idle_cycle_ref?;
                    let access_v = i8_t.const_int(0, false);
                    let lcall = builder.build_call(
                        lh,
                        &[cpu_ctx.into(), addr.into(), access_v.into()],
                        "f10_load",
                    ).ok()?;
                    let data = lcall.try_as_basic_value().unwrap_basic().into_int_value();

                    // bus.idle_cycle().
                    builder.build_call(idle, &[cpu_ctx.into()], "f10_idle").ok()?;

                    // gpr[Rd] = data.
                    builder.build_store(gpr_rd_ptr, data).ok()?;

                    // nfa = Seq (= 1).
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f10_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;
                } else {
                    // STRH: store_16(addr & ~1, gpr[Rd] as u16, NonSeq).
                    let s16 = store_16_ref?;
                    let val = builder.build_load(i32_t, gpr_rd_ptr, "f10_val").ok()?
                        .into_int_value();
                    let i16_t = self.context.i16_type();
                    let val16 = builder.build_int_truncate(val, i16_t, "f10_val16").ok()?;
                    let access_v = i8_t.const_int(0, false);
                    builder.build_call(
                        s16,
                        &[cpu_ctx.into(), addr.into(), val16.into(), access_v.into()],
                        "f10_store",
                    ).ok()?;

                    // nfa = NonSeq (= 0).
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f10_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;
                }

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f10_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // Continue.
                let cont_blk = self.context.append_basic_block(block_fn, "f10_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F9 LDR/STR imm5-offset (byte/word).
            // Encoding: 011_BL_IIIII_BBB_DDD; mask 0xe000 == 0x6000.
            //   B=bit 12 (1 → byte), L=bit 11 (1 → load).
            //   word ops: offset = imm5 << 2.
            //   byte ops: offset = imm5.
            //   addr = gpr[Rb] + offset (runtime base, constant offset).
            //   LDR  word: gpr[Rd] = ldr_word(addr, NonSeq) (I14 ROR + cpsr.C);
            //              idle_cycle; nfa = Seq.
            //   LDRB byte: gpr[Rd] = load_8(addr, NonSeq) zero-extended;
            //              idle_cycle; nfa = Seq.
            //   STR  word: store_32(addr & ~3, gpr[Rd], NonSeq); nfa = NonSeq.
            //   STRB byte: store_8(addr, gpr[Rd] as u8, NonSeq); nfa = NonSeq.
            //   pc = fetch_addr + 2; no flag updates (LDR may set cpsr.C
            //   internally on misaligned).
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F9 path.
            if inline_enabled
                && f9_inline_env
                && (opcode & 0xe000) == 0x6000
            {
                let byte_op = (opcode >> 12) & 0x1 != 0;
                let load = (opcode >> 11) & 0x1 != 0;
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let imm5 = ((opcode >> 6) & 0x1f) as u32;
                let rb = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;
                let offset_const = if byte_op { imm5 } else { imm5 << 2 };

                // Cycle accounting + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rb] (base).
                let gpr_rb_off = (off.gpr + rb * 4) as u64;
                let gpr_rb_off_v = i32_t.const_int(gpr_rb_off, false);
                let gpr_rb_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rb_off_v], "f9_rb_ptr").ok()?
                };
                let base = builder.build_load(i32_t, gpr_rb_ptr, "f9_base").ok()?
                    .into_int_value();

                // addr = base + offset_const.
                let addr = builder
                    .build_int_add(base, i32_t.const_int(offset_const as u64, false), "f9_addr")
                    .ok()?;

                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f9_rd_ptr").ok()?
                };

                if load {
                    let idle = idle_cycle_ref?;
                    let access_v = i8_t.const_int(0, false);
                    let data_i32 = if byte_op {
                        let l8 = load_8_ref?;
                        let lcall = builder.build_call(
                            l8,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f9_loadb",
                        ).ok()?;
                        let data_i8 = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                        // Zero-extend to i32.
                        builder.build_int_z_extend(data_i8, i32_t, "f9_zext").ok()?
                    } else {
                        let lw = ldr_word_ref?;
                        let lcall = builder.build_call(
                            lw,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f9_loadw",
                        ).ok()?;
                        lcall.try_as_basic_value().unwrap_basic().into_int_value()
                    };

                    // bus.idle_cycle().
                    builder.build_call(idle, &[cpu_ctx.into()], "f9_idle").ok()?;

                    // gpr[Rd] = data.
                    builder.build_store(gpr_rd_ptr, data_i32).ok()?;

                    // nfa = Seq (= 1).
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f9_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;
                } else {
                    let val_i32 = builder.build_load(i32_t, gpr_rd_ptr, "f9_val").ok()?
                        .into_int_value();
                    let access_v = i8_t.const_int(0, false);
                    if byte_op {
                        let s8 = store_8_ref?;
                        let val_i8 = builder.build_int_truncate(val_i32, i8_t, "f9_val8").ok()?;
                        builder.build_call(
                            s8,
                            &[cpu_ctx.into(), addr.into(), val_i8.into(), access_v.into()],
                            "f9_storeb",
                        ).ok()?;
                    } else {
                        let s32 = store_32_ref?;
                        builder.build_call(
                            s32,
                            &[cpu_ctx.into(), addr.into(), val_i32.into(), access_v.into()],
                            "f9_storew",
                        ).ok()?;
                    }

                    // nfa = NonSeq (= 0).
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f9_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;
                }

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f9_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // Continue.
                let cont_blk = self.context.append_basic_block(block_fn, "f9_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F7 LDR/STR reg-offset (byte/word).
            // Encoding: 0101_LB_0_RRR_BBB_DDD; mask 0xf200 == 0x5000.
            //   L=bit 11 (load/store), B=bit 10 (byte/word).
            //   Ro=bits 8:6 (reg offset), Rb=bits 5:3, Rd=bits 2:0.
            //   addr = gpr[Rb] + gpr[Ro]  (both runtime).
            //   Same memory-op pattern as F9 but with runtime offset
            //   instead of constant imm5-derived offset.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F7 path.
            if inline_enabled
                && f7_inline_env
                && (opcode & 0xf200) == 0x5000
            {
                let load = (opcode >> 11) & 0x1 != 0;
                let byte_op = (opcode >> 10) & 0x1 != 0;
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let ro = ((opcode >> 6) & 0x7) as u32;
                let rb = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rb] (base) and gpr[Ro] (offset).
                let gpr_rb_off = (off.gpr + rb * 4) as u64;
                let gpr_rb_off_v = i32_t.const_int(gpr_rb_off, false);
                let gpr_rb_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rb_off_v], "f7_rb_ptr").ok()?
                };
                let base = builder.build_load(i32_t, gpr_rb_ptr, "f7_base").ok()?
                    .into_int_value();

                let gpr_ro_off = (off.gpr + ro * 4) as u64;
                let gpr_ro_off_v = i32_t.const_int(gpr_ro_off, false);
                let gpr_ro_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_ro_off_v], "f7_ro_ptr").ok()?
                };
                let offset_v = builder.build_load(i32_t, gpr_ro_ptr, "f7_off").ok()?
                    .into_int_value();

                // addr = base + offset.
                let addr = builder.build_int_add(base, offset_v, "f7_addr").ok()?;

                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f7_rd_ptr").ok()?
                };

                if load {
                    let idle = idle_cycle_ref?;
                    let access_v = i8_t.const_int(0, false);
                    let data_i32 = if byte_op {
                        let l8 = load_8_ref?;
                        let lcall = builder.build_call(
                            l8,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f7_loadb",
                        ).ok()?;
                        let data_i8 = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                        builder.build_int_z_extend(data_i8, i32_t, "f7_zext").ok()?
                    } else {
                        let lw = ldr_word_ref?;
                        let lcall = builder.build_call(
                            lw,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f7_loadw",
                        ).ok()?;
                        lcall.try_as_basic_value().unwrap_basic().into_int_value()
                    };

                    builder.build_call(idle, &[cpu_ctx.into()], "f7_idle").ok()?;
                    builder.build_store(gpr_rd_ptr, data_i32).ok()?;

                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f7_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;
                } else {
                    let val_i32 = builder.build_load(i32_t, gpr_rd_ptr, "f7_val").ok()?
                        .into_int_value();
                    let access_v = i8_t.const_int(0, false);
                    if byte_op {
                        let s8 = store_8_ref?;
                        let val_i8 = builder.build_int_truncate(val_i32, i8_t, "f7_val8").ok()?;
                        builder.build_call(
                            s8,
                            &[cpu_ctx.into(), addr.into(), val_i8.into(), access_v.into()],
                            "f7_storeb",
                        ).ok()?;
                    } else {
                        let s32 = store_32_ref?;
                        builder.build_call(
                            s32,
                            &[cpu_ctx.into(), addr.into(), val_i32.into(), access_v.into()],
                            "f7_storew",
                        ).ok()?;
                    }

                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f7_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;
                }

                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f7_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                let cont_blk = self.context.append_basic_block(block_fn, "f7_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F8 LDR/STR sign-extended/halfword reg-offset.
            // Encoding: 0101_HS_1_OOO_BBB_DDD; mask 0xf200 == 0x5200.
            //   H=bit 11, S=bit 10. (S,H) selects:
            //     (0,0): STRH  — store_16(addr & ~1, val u16, NonSeq)
            //     (0,1): LDRH  — ldr_half(addr) [misaligned ROR + cpsr.C]; idle
            //     (1,0): LDSB  — load_8 sign-extended i8→i32; idle
            //     (1,1): LDSH  — ldr_sign_half(addr) [misaligned: i8 sign-ext]; idle
            //   addr = gpr[Rb] + gpr[Ro] (both runtime).
            //   nfa = NonSeq for ALL F8 sub-cases (note: differs from F7
            //   which sets Seq for LDR variants — per scalar comment
            //   "Always returns AdvancePC(NonSeq)").
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F8 path.
            if inline_enabled
                && f8_inline_env
                && (opcode & 0xf200) == 0x5200
            {
                let halfword = (opcode >> 11) & 0x1 != 0;
                let sign_extend = (opcode >> 10) & 0x1 != 0;
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let ro = ((opcode >> 6) & 0x7) as u32;
                let rb = ((opcode >> 3) & 0x7) as u32;
                let rd = (opcode & 0x7) as u32;

                // Cycle accounting + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[Rb], gpr[Ro].
                let gpr_rb_off = (off.gpr + rb * 4) as u64;
                let gpr_rb_off_v = i32_t.const_int(gpr_rb_off, false);
                let gpr_rb_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rb_off_v], "f8_rb_ptr").ok()?
                };
                let base = builder.build_load(i32_t, gpr_rb_ptr, "f8_base").ok()?
                    .into_int_value();

                let gpr_ro_off = (off.gpr + ro * 4) as u64;
                let gpr_ro_off_v = i32_t.const_int(gpr_ro_off, false);
                let gpr_ro_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_ro_off_v], "f8_ro_ptr").ok()?
                };
                let offset_v = builder.build_load(i32_t, gpr_ro_ptr, "f8_off").ok()?
                    .into_int_value();
                let addr = builder.build_int_add(base, offset_v, "f8_addr").ok()?;

                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f8_rd_ptr").ok()?
                };

                let access_v = i8_t.const_int(0, false);
                match (sign_extend, halfword) {
                    (false, false) => {
                        // STRH: store_16(addr & ~1, val as u16, NonSeq).
                        let s16 = store_16_ref?;
                        let val_i32 = builder.build_load(i32_t, gpr_rd_ptr, "f8_val").ok()?
                            .into_int_value();
                        let i16_t = self.context.i16_type();
                        let val16 = builder.build_int_truncate(val_i32, i16_t, "f8_val16").ok()?;
                        builder.build_call(
                            s16,
                            &[cpu_ctx.into(), addr.into(), val16.into(), access_v.into()],
                            "f8_strh",
                        ).ok()?;
                    }
                    (false, true) => {
                        // LDRH: data = ldr_half(addr); idle; gpr[Rd] = data.
                        let lh = ldr_half_ref?;
                        let idle = idle_cycle_ref?;
                        let lcall = builder.build_call(
                            lh,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f8_ldrh",
                        ).ok()?;
                        let data = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                        builder.build_call(idle, &[cpu_ctx.into()], "f8_idle").ok()?;
                        builder.build_store(gpr_rd_ptr, data).ok()?;
                    }
                    (true, false) => {
                        // LDSB: data = sign_extend(load_8(addr) as i8 to i32); idle; gpr[Rd] = data.
                        let l8 = load_8_ref?;
                        let idle = idle_cycle_ref?;
                        let lcall = builder.build_call(
                            l8,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f8_ldsb",
                        ).ok()?;
                        let data_i8 = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                        // Sign-extend i8 → i32.
                        let data_i32 = builder
                            .build_int_s_extend(data_i8, i32_t, "f8_sext")
                            .ok()?;
                        builder.build_call(idle, &[cpu_ctx.into()], "f8_idle").ok()?;
                        builder.build_store(gpr_rd_ptr, data_i32).ok()?;
                    }
                    (true, true) => {
                        // LDSH: data = ldr_sign_half(addr); idle; gpr[Rd] = data.
                        let lsh = ldr_sign_half_ref?;
                        let idle = idle_cycle_ref?;
                        let lcall = builder.build_call(
                            lsh,
                            &[cpu_ctx.into(), addr.into(), access_v.into()],
                            "f8_ldsh",
                        ).ok()?;
                        let data = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                        builder.build_call(idle, &[cpu_ctx.into()], "f8_idle").ok()?;
                        builder.build_store(gpr_rd_ptr, data).ok()?;
                    }
                }

                // pc = fetch_addr + 2; nfa = NonSeq for ALL F8 sub-cases.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f8_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f8_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;

                let cont_blk = self.context.append_basic_block(block_fn, "f8_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F11 STR SP-relative (word).
            // Encoding: 1001_0_DDD_IIIIIIII; mask 0xf800 == 0x9000.
            //   imm = (insn & 0xff) << 2  (constant, word-scaled)
            //   addr = gpr[SP] + imm  (runtime base, constant offset)
            //   bus.store_32(addr & ~3, gpr[Rd], NonSeq)
            //                (matches scalar's store_aligned_32)
            //   pc = fetch_addr + 2; nfa = NonSeq (= 0)
            //   No flag updates.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F11 STR path.
            if inline_enabled
                && f11_str_inline_env
                && store_32_ref.is_some()
                && (opcode & 0xf800) == 0x9000
            {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let s32 = store_32_ref.unwrap();
                let rd = ((opcode >> 8) & 0x7) as u32;
                let imm = ((opcode & 0xff) as u32) << 2;

                // Cycle accounting (fetch) + pipeline shift via fetch_only.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[SP] (REG_SP = 13).
                let sp_off = (off.gpr + 13 * 4) as u64;
                let sp_off_v = i32_t.const_int(sp_off, false);
                let sp_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[sp_off_v], "f11s_sp_ptr").ok()?
                };
                let sp_val = builder.build_load(i32_t, sp_ptr, "f11s_sp_val").ok()?
                    .into_int_value();

                // addr = SP + imm.
                let addr = builder
                    .build_int_add(sp_val, i32_t.const_int(imm as u64, false), "f11s_addr")
                    .ok()?;

                // val = gpr[Rd].
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f11s_gpr_rd_ptr").ok()?
                };
                let val = builder.build_load(i32_t, gpr_rd_ptr, "f11s_val").ok()?
                    .into_int_value();

                // bus.store_32(addr, val, NonSeq=0). Wrapper does addr & ~3.
                let access_v = i8_t.const_int(0, false);
                builder.build_call(
                    s32,
                    &[cpu_ctx.into(), addr.into(), val.into(), access_v.into()],
                    "f11s_store",
                ).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f11s_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = NonSeq (= 0).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f11s_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC).
                let cont_blk = self.context.append_basic_block(block_fn, "f11s_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F19 hi (BL pair high half).
            // Encoding: 11110_OOOOOOOOOOO; top5 == 0b11110.
            // Effects (mirrors arm7tdmi/src/cpu.rs aot_thumb_step F19 hi path):
            //   off = sign-extend ((insn & 0x7ff) << 12) — constant at AOT time
            //   gpr[REG_LR=14] = (fetch_addr + off) wrapping — constant
            //   pc = fetch_addr + 2; nfa = Seq.
            //   No flag updates; no PipelineFlushed (F19 lo terminates).
            // Smallest-possible inline IR: 1 const store to gpr[14] + pc/nfa
            // updates. Useful for the stacking-pattern hazard test — does
            // even minimal-IR-per-opcode addition cause MK regression?
            if inline_enabled && f19hi_inline_env && (opcode >> 11) == 0b11110 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                // off bake: ((insn & 0x7ff) << 21) as i32 >> 9
                let off_imm = ((((opcode & 0x7ff) as u32) << 21) as i32 >> 9) as u32;
                let lr_const = fetch_addr.wrapping_add(off_imm);

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // gpr[14] = lr_const.
                let gpr_lr_off = (off.gpr + 14 * 4) as u64;
                let gpr_lr_off_v = i32_t.const_int(gpr_lr_off, false);
                let gpr_lr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_lr_off_v], "f19h_lr_ptr").ok()?
                };
                builder.build_store(gpr_lr_ptr, i32_t.const_int(lr_const as u64, false)).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f19h_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq (= 1).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f19h_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue.
                let cont_blk = self.context.append_basic_block(block_fn, "f19h_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F12 LoadAddress (ADD Rd, [PC|SP], #imm8).
            // Encoding: 1010_S_DDD_IIIIIIII; mask 0xf000 == 0xa000.
            //   S (bit 11): 0 = PC-relative, 1 = SP-relative.
            //   imm = (insn & 0xff) << 2 (word-scaled, constant at AOT time).
            //   if S=0: Rd = ((exec_addr) & ~2) + 4 + imm   (constant at AOT)
            //   if S=1: Rd = gpr[SP] + imm                  (runtime)
            //   no flag updates; pc = fetch_addr + 2; nfa = Seq.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F12 path
            // (which mirrors thumb/exec.rs::exec_thumb_load_address).
            if inline_enabled && f12_inline_env && (opcode & 0xf000) == 0xa000 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let sp_flag = (opcode >> 11) & 0x1 != 0;
                let rd = ((opcode >> 8) & 0x7) as u32;
                let imm = ((opcode & 0xff) as u32) << 2;

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                let val = if sp_flag {
                    // Load gpr[SP] (REG_SP = 13).
                    let sp_off = (off.gpr + 13 * 4) as u64;
                    let sp_off_v = i32_t.const_int(sp_off, false);
                    let sp_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[sp_off_v], "f12_sp_ptr").ok()?
                    };
                    let sp_val = builder.build_load(i32_t, sp_ptr, "f12_sp_val").ok()?
                        .into_int_value();
                    builder
                        .build_int_add(sp_val, i32_t.const_int(imm as u64, false), "f12_val")
                        .ok()?
                } else {
                    // PC-relative: val = (exec_addr & !2) + 4 + imm.
                    let exec_addr_aligned = exec_addr & !0b10;
                    let val_const = exec_addr_aligned.wrapping_add(4).wrapping_add(imm);
                    i32_t.const_int(val_const as u64, false)
                };

                // gpr[Rd] = val.
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "f12_gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, val).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f12_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq (= 1).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f12_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue (Rd is 3 bits, can't be PC → always AdvancePC).
                let cont_blk = self.context.append_basic_block(block_fn, "f12_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F13 AddSp (ADD/SUB SP, #imm7<<2).
            // Encoding: 10110000_S_IIIIIII; mask 0xff00 == 0xb000.
            //   S (bit 7): 0 = ADD, 1 = SUB.
            //   offset = (insn & 0x7f) << 2 (constant at AOT time).
            //   gpr[SP] = sp +/- offset (wrapping).
            //   no flag updates; pc = fetch_addr + 2; nfa = Seq.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F13 path.
            if inline_enabled && f13_inline_env && (opcode & 0xff00) == 0xb000 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let sub = (opcode >> 7) & 0x1 != 0;
                let offset = ((opcode & 0x7f) as u32) << 2;

                // Cycle accounting + pipeline shift.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Load gpr[SP] (REG_SP = 13).
                let sp_off = (off.gpr + 13 * 4) as u64;
                let sp_off_v = i32_t.const_int(sp_off, false);
                let sp_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[sp_off_v], "f13_sp_ptr").ok()?
                };
                let sp_val = builder.build_load(i32_t, sp_ptr, "f13_sp_val").ok()?
                    .into_int_value();

                let off_const = i32_t.const_int(offset as u64, false);
                let new_sp = if sub {
                    builder.build_int_sub(sp_val, off_const, "f13_sp_sub").ok()?
                } else {
                    builder.build_int_add(sp_val, off_const, "f13_sp_add").ok()?
                };
                builder.build_store(sp_ptr, new_sp).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f13_pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // nfa = Seq (= 1).
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f13_nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue.
                let cont_blk = self.context.append_basic_block(block_fn, "f13_cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Phase-4 inline IR for F14 PUSH/POP.
            // Encoding: 1011_L_10_R_RRRRRRRR; mask 0xf600 == 0xb400.
            //   L=bit 11 (0=PUSH, 1=POP), R=bit 8 (PUSH→include LR, POP→include PC).
            //   rlist=bits 7:0 (R0..R7).
            //
            // PUSH (any rlist): inline IR. SP -= 4*N, stores in high-to-low
            //   register order. flag_r=1 pushes LR first (top of stack).
            //   First store NonSeq, subsequent Seq.
            //   nfa = NonSeq; pc = fetch_addr + 2.
            //
            // POP without PC (flag_r=0): inline IR. Loads in low-to-high
            //   register order. SP += 4*popcount.
            //   First load NonSeq, subsequent Seq. idle_cycle. nfa = NonSeq.
            //
            // POP with PC (flag_r=1): block terminator (pipeline flush) —
            //   fall through to step trampoline.
            //
            // rlist is constant at AOT compile time, so the per-register
            // loop is unrolled in the IR-emit Rust code.
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F14 path.
            if inline_enabled
                && f14_inline_env
                && (opcode & 0xf600) == 0xb400
            {
                let pop = (opcode >> 11) & 0x1 != 0;
                let flag_r = (opcode >> 8) & 0x1 != 0;
                let rlist = (opcode & 0xff) as u8;
                // Skip POP-with-PC (block terminator).
                if !(pop && flag_r) {
                    let off = offsets.unwrap();
                    let fo = fetch_only_ref.unwrap();

                    // fetch_only.
                    builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                    // Load init_sp = gpr[REG_SP=13].
                    let sp_off = (off.gpr + 13 * 4) as u64;
                    let sp_off_v = i32_t.const_int(sp_off, false);
                    let sp_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[sp_off_v], "f14_sp_ptr").ok()?
                    };
                    let init_sp = builder.build_load(i32_t, sp_ptr, "f14_init_sp").ok()?
                        .into_int_value();

                    let neg3_mask = i32_t.const_int(!3u32 as u64, false);

                    if pop {
                        // POP without PC. Loads low-to-high.
                        let l32 = load_32_ref?;
                        let mut access_byte: u8 = 0; // first NonSeq, rest Seq
                        let mut offset: u32 = 0;
                        for r in 0..8u32 {
                            if (rlist >> r) & 1 != 0 {
                                let addr_unaligned = if offset == 0 {
                                    init_sp
                                } else {
                                    builder.build_int_add(
                                        init_sp,
                                        i32_t.const_int(offset as u64, false),
                                        "f14_pop_addr_u",
                                    ).ok()?
                                };
                                let addr = builder.build_and(addr_unaligned, neg3_mask, "f14_pop_addr").ok()?;
                                let access_v = i8_t.const_int(access_byte as u64, false);
                                let lcall = builder.build_call(
                                    l32,
                                    &[cpu_ctx.into(), addr.into(), access_v.into()],
                                    "f14_pop_load",
                                ).ok()?;
                                let val = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                                let gpr_r_off = (off.gpr + r * 4) as u64;
                                let gpr_r_off_v = i32_t.const_int(gpr_r_off, false);
                                let gpr_r_ptr = unsafe {
                                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_r_off_v], "f14_pop_r_ptr").ok()?
                                };
                                builder.build_store(gpr_r_ptr, val).ok()?;
                                access_byte = 1;
                                offset = offset.wrapping_add(4);
                            }
                        }
                        // gpr[SP] += offset.
                        let new_sp = builder.build_int_add(init_sp, i32_t.const_int(offset as u64, false), "f14_pop_new_sp").ok()?;
                        builder.build_store(sp_ptr, new_sp).ok()?;

                        // idle_cycle.
                        let idle = idle_cycle_ref?;
                        builder.build_call(idle, &[cpu_ctx.into()], "f14_pop_idle").ok()?;
                    } else {
                        // PUSH. flag_r first (LR at top of stack), then high-to-low.
                        let s32 = store_32_ref?;
                        let mut access_byte: u8 = 0;
                        let mut offset: i32 = 0;

                        let do_store = |builder: &inkwell::builder::Builder<'_>,
                                        offset: i32,
                                        access_byte: u8,
                                        reg_idx: u32|
                         -> Option<()> {
                            let addr_unaligned = builder.build_int_add(
                                init_sp,
                                i32_t.const_int(offset as i64 as u64, false),
                                "f14_push_addr_u",
                            ).ok()?;
                            let addr = builder.build_and(addr_unaligned, neg3_mask, "f14_push_addr").ok()?;
                            let gpr_r_off = (off.gpr + reg_idx * 4) as u64;
                            let gpr_r_off_v = i32_t.const_int(gpr_r_off, false);
                            let gpr_r_ptr = unsafe {
                                builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_r_off_v], "f14_push_r_ptr").ok()?
                            };
                            let val = builder.build_load(i32_t, gpr_r_ptr, "f14_push_val").ok()?
                                .into_int_value();
                            let access_v = i8_t.const_int(access_byte as u64, false);
                            builder.build_call(
                                s32,
                                &[cpu_ctx.into(), addr.into(), val.into(), access_v.into()],
                                "f14_push_store",
                            ).ok()?;
                            Some(())
                        };

                        if flag_r {
                            offset -= 4;
                            do_store(&builder, offset, access_byte, 14)?;
                            access_byte = 1;
                        }
                        for r in (0..8u32).rev() {
                            if (rlist >> r) & 1 != 0 {
                                offset -= 4;
                                do_store(&builder, offset, access_byte, r)?;
                                access_byte = 1;
                            }
                        }
                        // gpr[SP] = init_sp + offset (offset is negative).
                        let new_sp = builder.build_int_add(init_sp, i32_t.const_int(offset as i64 as u64, false), "f14_push_new_sp").ok()?;
                        builder.build_store(sp_ptr, new_sp).ok()?;
                    }

                    // pc = fetch_addr + 2; nfa = NonSeq.
                    let pc_off_v = i32_t.const_int(off.pc as u64, false);
                    let pc_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f14_pc_ptr").ok()?
                    };
                    builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f14_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;

                    let cont_blk = self.context.append_basic_block(block_fn, "f14_cont");
                    builder.build_unconditional_branch(cont_blk).ok()?;
                    builder.position_at_end(cont_blk);
                    continue;
                }
                // POP-with-PC: fall through to step trampoline.
            }

            // Phase-4 inline IR for F15 LDMIA/STMIA Rb!, {rlist}.
            // Encoding: 1100_L_BBB_RRRRRRRR; mask 0xf000 == 0xc000.
            //   L=bit 11 (1=LDM, 0=STM), B=bits 10:8 (Rb), rlist=bits 7:0.
            //
            // Empty rlist (rlist == 0) hits the GBATEK ARMv4 quirk —
            // LDM-empty is a block terminator (loads PC, pipeline-flush).
            // We don't inline the empty case at all; it falls through to
            // the step trampoline. Since rlist is constant at AOT compile
            // time the check is a constant.
            //
            // Non-empty: unroll the per-register loop in IR-emit Rust.
            //   addr = gpr[Rb] & !3, align_preserve = gpr[Rb] & 3.
            //   First access NonSeq, subsequent Seq.
            //   LDM: idle_cycle after loop. Writeback (addr_after +
            //     align_preserve) only if Rb is NOT in rlist.
            //   STM: no idle cycle. Writeback always (addr_after +
            //     align_preserve). For STM with Rb in rlist:
            //       value-stored-at-Rb = if Rb is the FIRST set bit then
            //         init_addr (masked) else init_addr + (popcount-1)*4.
            //   nfa = NonSeq; pc = fetch_addr + 2.
            //
            // Mirrors arm7tdmi/src/cpu.rs aot_thumb_step F15 path.
            if inline_enabled
                && f15_inline_env
                && (opcode & 0xf000) == 0xc000
            {
                let load = (opcode >> 11) & 0x1 != 0;
                let rb = ((opcode >> 8) & 0x7) as u32;
                let rlist = (opcode & 0xff) as u8;
                // Empty rlist: fall through to step trampoline (handles
                // the GBATEK quirk including LDM-empty pipeline flush).
                if rlist != 0 {
                    let off = offsets.unwrap();
                    let fo = fetch_only_ref.unwrap();

                    // fetch_only.
                    builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                    // Load init Rb.
                    let rb_off = (off.gpr + rb * 4) as u64;
                    let rb_off_v = i32_t.const_int(rb_off, false);
                    let rb_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[rb_off_v], "f15_rb_ptr").ok()?
                    };
                    let init_rb = builder.build_load(i32_t, rb_ptr, "f15_init_rb").ok()?
                        .into_int_value();

                    let mask3 = i32_t.const_int(3u32 as u64, false);
                    let neg3_mask = i32_t.const_int(!3u32 as u64, false);

                    // align_preserve = init_rb & 3.
                    let align_preserve = builder.build_and(init_rb, mask3, "f15_align_pres").ok()?;
                    // init_addr = init_rb & !3.
                    let init_addr = builder.build_and(init_rb, neg3_mask, "f15_init_addr").ok()?;

                    let popcount: u32 = (rlist.count_ones()) as u32;
                    let rb_in_rlist = (rlist >> rb) & 1 != 0;

                    if load {
                        // LDM. Loads in low-to-high.
                        let l32 = load_32_ref?;
                        let mut access_byte: u8 = 0; // first NonSeq, rest Seq
                        let mut iter_off: u32 = 0;
                        for r in 0..8u32 {
                            if (rlist >> r) & 1 != 0 {
                                let addr = if iter_off == 0 {
                                    init_addr
                                } else {
                                    builder.build_int_add(
                                        init_addr,
                                        i32_t.const_int(iter_off as u64, false),
                                        "f15_ldm_addr",
                                    ).ok()?
                                };
                                let access_v = i8_t.const_int(access_byte as u64, false);
                                let lcall = builder.build_call(
                                    l32,
                                    &[cpu_ctx.into(), addr.into(), access_v.into()],
                                    "f15_ldm_load",
                                ).ok()?;
                                let val = lcall.try_as_basic_value().unwrap_basic().into_int_value();
                                let gpr_r_off = (off.gpr + r * 4) as u64;
                                let gpr_r_off_v = i32_t.const_int(gpr_r_off, false);
                                let gpr_r_ptr = unsafe {
                                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_r_off_v], "f15_ldm_r_ptr").ok()?
                                };
                                builder.build_store(gpr_r_ptr, val).ok()?;
                                access_byte = 1;
                                iter_off = iter_off.wrapping_add(4);
                            }
                        }
                        // idle_cycle.
                        let idle = idle_cycle_ref?;
                        builder.build_call(idle, &[cpu_ctx.into()], "f15_ldm_idle").ok()?;
                        // Writeback only if Rb is NOT in rlist.
                        if !rb_in_rlist {
                            // gpr[Rb] = (init_addr + popcount*4) + align_preserve.
                            let final_addr = builder.build_int_add(
                                init_addr,
                                i32_t.const_int((popcount * 4) as u64, false),
                                "f15_ldm_final_addr",
                            ).ok()?;
                            let new_rb = builder.build_int_add(
                                final_addr,
                                align_preserve,
                                "f15_ldm_new_rb",
                            ).ok()?;
                            builder.build_store(rb_ptr, new_rb).ok()?;
                        }
                    } else {
                        // STM. Stores in low-to-high.
                        let s32 = store_32_ref?;
                        let mut access_byte: u8 = 0;
                        let mut iter_off: u32 = 0;
                        // For STM-with-Rb-in-rlist, count = popcount - 1
                        // (matches scalar `let count = popcount - 1`).
                        let count: u32 = popcount.wrapping_sub(1);
                        // Find first-set bit index in rlist (lowest bit).
                        let first_set = rlist.trailing_zeros();

                        for r in 0..8u32 {
                            if (rlist >> r) & 1 != 0 {
                                let addr = if iter_off == 0 {
                                    init_addr
                                } else {
                                    builder.build_int_add(
                                        init_addr,
                                        i32_t.const_int(iter_off as u64, false),
                                        "f15_stm_addr",
                                    ).ok()?
                                };
                                // Determine value to store.
                                let val = if r != rb {
                                    // gpr[r].
                                    let gpr_r_off = (off.gpr + r * 4) as u64;
                                    let gpr_r_off_v = i32_t.const_int(gpr_r_off, false);
                                    let gpr_r_ptr = unsafe {
                                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_r_off_v], "f15_stm_r_ptr").ok()?
                                    };
                                    builder.build_load(i32_t, gpr_r_ptr, "f15_stm_val")
                                        .ok()?.into_int_value()
                                } else {
                                    // r == rb. If this is the first set
                                    // bit, store init_addr; else store
                                    // init_addr + count*4.
                                    if r == first_set {
                                        init_addr
                                    } else {
                                        builder.build_int_add(
                                            init_addr,
                                            i32_t.const_int((count * 4) as u64, false),
                                            "f15_stm_rb_addrn",
                                        ).ok()?
                                    }
                                };
                                let access_v = i8_t.const_int(access_byte as u64, false);
                                builder.build_call(
                                    s32,
                                    &[cpu_ctx.into(), addr.into(), val.into(), access_v.into()],
                                    "f15_stm_store",
                                ).ok()?;
                                access_byte = 1;
                                iter_off = iter_off.wrapping_add(4);
                            }
                        }
                        // STM always writes back: gpr[Rb] = init_addr + popcount*4 + align_preserve.
                        let final_addr = builder.build_int_add(
                            init_addr,
                            i32_t.const_int((popcount * 4) as u64, false),
                            "f15_stm_final_addr",
                        ).ok()?;
                        let new_rb = builder.build_int_add(
                            final_addr,
                            align_preserve,
                            "f15_stm_new_rb",
                        ).ok()?;
                        builder.build_store(rb_ptr, new_rb).ok()?;
                    }

                    // pc = fetch_addr + 2; nfa = NonSeq.
                    let pc_off_v = i32_t.const_int(off.pc as u64, false);
                    let pc_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "f15_pc_ptr").ok()?
                    };
                    builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;
                    let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                    let nfa_ptr = unsafe {
                        builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "f15_nfa_ptr").ok()?
                    };
                    builder.build_store(nfa_ptr, i8_t.const_int(0, false)).ok()?;

                    let cont_blk = self.context.append_basic_block(block_fn, "f15_cont");
                    builder.build_unconditional_branch(cont_blk).ok()?;
                    builder.position_at_end(cont_blk);
                    continue;
                }
                // Empty rlist: fall through to step trampoline.
            }

            // Phase-4 inline IR for F3 MOV imm8 (top5=00100, op=00).
            // Encoding: 00100_RRR_IIIIIIII. Effects:
            //   gpr[Rd] = imm8 (zero-extended)
            //   cpsr.N = 0 (always — imm8 is positive)
            //   cpsr.Z = (imm8 == 0)
            //   cpsr.C, .V unchanged
            //   pc = fetch_addr + 2
            //   next_fetch_access = Seq (= 0)
            // Cycle accounting via fetch-only trampoline (load_16 +
            // pipeline shift) — same per-iter cost as scalar.
            // Phase 4 step 1: F3 MOV imm8 inline IR.
            // Only fires when inline_enabled (AOT_INLINE_F3=1 + the
            // fetch_only/offsets infrastructure registered). Currently
            // a slight perf regression vs the trampoline path; kept for
            // correctness verification + future iteration.
            let f3_top5 = (opcode >> 11) & 0x1f;
            let f3_op = (opcode >> 11) & 0x3;
            if inline_enabled && f3_inline_env && f3_top5 == 0b00100 && f3_op == 0 {
                let off = offsets.unwrap();
                let fo = fetch_only_ref.unwrap();
                let rd = ((opcode >> 8) & 0x7) as u32;
                let imm = (opcode & 0xff) as u32;

                // Cycle accounting + pipeline shift via the fetch-only
                // extern. Note: at commit afb1ebd we tried inline cycle
                // accumulation here (`*sched_ts_ptr += baked_K_cycles`)
                // but it baked WAITCNT-derived cycle constants at AOT
                // compile time; PE writes WAITCNT during BIOS boot,
                // which silently invalidates the constants and produces
                // ~12 hash-divs at sw=64KB. Per I7 the proper fix is
                // a synchronous recompile on WAITCNT write — not
                // implemented yet. Until then, charge cycles via the
                // bus path (always reads current cycle_luts) by having
                // fetch_only call load_16 instead of read_16_no_cycles.
                // See aot_thumb_fetch_only in arm7tdmi/src/cpu.rs.
                builder.build_call(fo, &[cpu_ctx.into(), fa.into()], "").ok()?;

                // Store imm at gpr[rd] (gpr is u32 array; offset = gpr_off + rd*4).
                let gpr_rd_off = (off.gpr + rd * 4) as u64;
                let gpr_rd_off_v = i32_t.const_int(gpr_rd_off, false);
                let gpr_rd_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[gpr_rd_off_v], "gpr_rd_ptr").ok()?
                };
                builder.build_store(gpr_rd_ptr, i32_t.const_int(imm as u64, false)).ok()?;

                // cpsr = (cpsr_old & ~(N|Z)) | (Z if imm==0). N=0 always for imm8.
                let cpsr_off_v = i32_t.const_int(off.cpsr as u64, false);
                let cpsr_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[cpsr_off_v], "cpsr_ptr").ok()?
                };
                let cpsr_old = builder.build_load(i32_t, cpsr_ptr, "cpsr_old").ok()?
                    .into_int_value();
                // Mask: clear bits N (0x80000000) and Z (0x40000000).
                let nz_mask_inv = i32_t.const_int(0x3fff_ffff, false);
                let cleared = builder.build_and(cpsr_old, nz_mask_inv, "cpsr_cleared").ok()?;
                // Set Z if imm == 0 (constant at compile time).
                let z_bit = if imm == 0 { 0x4000_0000u32 } else { 0 };
                let final_cpsr = builder
                    .build_or(cleared, i32_t.const_int(z_bit as u64, false), "cpsr_new").ok()?;
                builder.build_store(cpsr_ptr, final_cpsr).ok()?;

                // pc = fetch_addr + 2.
                let pc_off_v = i32_t.const_int(off.pc as u64, false);
                let pc_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[pc_off_v], "pc_ptr").ok()?
                };
                builder.build_store(pc_ptr, i32_t.const_int(fetch_addr.wrapping_add(2) as u64, false)).ok()?;

                // next_fetch_access = Seq (= 1, MemoryAccess enum). u8.
                let nfa_off_v = i32_t.const_int(off.next_fetch_access as u64, false);
                let nfa_ptr = unsafe {
                    builder.build_in_bounds_gep(i8_t, cpu_ctx, &[nfa_off_v], "nfa_ptr").ok()?
                };
                builder.build_store(nfa_ptr, i8_t.const_int(1, false)).ok()?;

                // Continue to next iter (no PipelineFlushed branch — F3 MOV is always AdvancePC).
                let cont_blk = self.context.append_basic_block(block_fn, "cont");
                builder.build_unconditional_branch(cont_blk).ok()?;
                builder.position_at_end(cont_blk);
                continue;
            }

            // Fallback: per-iter step trampoline.
            let scall = builder
                .build_call(step_ref, &[cpu_ctx.into(), fa.into(), insn.into()], "step_res")
                .ok()?;
            let step_v = scall.try_as_basic_value().unwrap_basic().into_int_value();
            // Branch on PipelineFlushed (returned 1) → exit_blk.
            let zero = i32_t.const_int(0, false);
            let flushed = builder
                .build_int_compare(IntPredicate::NE, step_v, zero, "flushed")
                .ok()?;
            let cont_blk = self.context.append_basic_block(block_fn, "cont");
            builder
                .build_conditional_branch(flushed, exit_blk, cont_blk)
                .ok()?;
            builder.position_at_end(cont_blk);
        }
        // Phase-4P-B: emit one block-exit pipeline_restore call when
        // the baked-cycle path is active. The per-opcode helper skipped
        // pipeline shifts; we make the next block's pipeline correct
        // here. exec_first = entry_pc + 2*N is the address of the next
        // block's first opcode (= current cpu.pc - 4 since the last
        // opcode's AdvancePC set pc = fetch_addr_{N-1} + 2 = entry_pc +
        // 2*N + 4 = exec_first + 4).
        // Only fires on the fall-through path; PipelineFlushed branches
        // skip this (the dispatcher reload_pipeline16 at the new target
        // overwrites whatever pipeline state we have).
        if baked_cycles_active {
            if let Some(pr) = pipe_restore_ref {
                let exec_first =
                    entry_pc.wrapping_add(2u32.wrapping_mul(opcodes.len() as u32));
                let exec_first_v = i32_t.const_int(exec_first as u64, false);
                builder
                    .build_call(
                        pr,
                        &[cpu_ctx.into(), exec_first_v.into()],
                        "",
                    )
                    .ok()?;
            }
        }
        // Fell through all opcodes — go to exit.
        builder.build_unconditional_branch(exit_blk).ok()?;

        // exit_blk: return 0 (per phase-0 ABI; handler updates cpu.pc
        // on PipelineFlushed before reaching here).
        builder.position_at_end(exit_blk);
        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .ok()?;

        // abort_blk: return 0b10.
        builder.position_at_end(abort_blk);
        builder
            .build_return(Some(&i32_t.const_int(0b10, false)))
            .ok()?;

        self.engine.add_module(&module).ok()?;
        let raw = self.engine.get_function_address(&block_name).ok()?;
        Some(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Phase-0 placeholder block emit. Creates an LLVM fn matching
    /// the `CompiledFn` ABI (`extern "C" fn(cpu_ctx, pc_out) -> u32`)
    /// that calls the registered Thumb replay trampoline with the
    /// supplied opcodes_ptr/len/entry_pc baked in as constants.
    ///
    /// `opcodes_ptr` must point to a stable memory location that
    /// outlives the AotTable (caller arena: `AotTable.thumb_arena`).
    pub fn emit_placeholder_thumb_block(
        &mut self,
        opcodes_ptr: *const u32,
        opcodes_len: u32,
        entry_pc: u32,
    ) -> Option<CompiledFn> {
        use inkwell::AddressSpace;

        let replay_fn = self.replay_thumb_fn?;

        let module = self.context.create_module("aot_blk");
        let i32_t = self.context.i32_type();
        let i64_t = self.context.i64_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        // Import the replay trampoline as
        //   extern "C" fn(cpu_ctx: ptr, opcodes: ptr, len: i32, pc: i32) -> i32
        let trampoline_sig = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), i32_t.into(), i32_t.into()],
            false,
        );
        let trampoline_ref =
            module.add_function("rba_aot_replay_thumb", trampoline_sig, None);
        self.engine
            .add_global_mapping(&trampoline_ref, replay_fn as usize);

        // Block fn: extern "C" fn(*mut u8 cpu_ctx, *mut u32 pc_out) -> u32
        let block_sig = i32_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);
        self.next_id += 1;
        let block_name = format!("aot_blk_{}", self.next_id);
        let block_fn = module.add_function(&block_name, block_sig, None);
        let entry = self.context.append_basic_block(block_fn, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let cpu_ctx = block_fn.get_nth_param(0).unwrap().into_pointer_value();
        // pc_out unused for phase-0 placeholder (handler updates cpu.pc
        // directly on PipelineFlushed; the dispatcher reads cpu.pc).

        // Bake opcodes_ptr as a constant ptr-sized integer cast to ptr.
        let opcodes_ptr_const =
            i64_t.const_int(opcodes_ptr as u64, false);
        let opcodes_ptr_v = builder
            .build_int_to_ptr(opcodes_ptr_const, ptr_t, "opc_ptr")
            .ok()?;
        let len_v = i32_t.const_int(opcodes_len as u64, false);
        let entry_pc_v = i32_t.const_int(entry_pc as u64, false);

        // ret = call rba_aot_replay_thumb(cpu_ctx, opcodes_ptr, len, entry_pc)
        let call = builder
            .build_call(
                trampoline_ref,
                &[
                    cpu_ctx.into(),
                    opcodes_ptr_v.into(),
                    len_v.into(),
                    entry_pc_v.into(),
                ],
                "ret",
            )
            .ok()?;
        let ret = call
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        builder.build_return(Some(&ret)).ok()?;

        // Add module + look up the JITed fn pointer.
        self.engine.add_module(&module).ok()?;
        let raw = self.engine.get_function_address(&block_name).ok()?;
        Some(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Phase-8 ARM placeholder block emit. Mirrors
    /// `emit_placeholder_thumb_block` but calls the ARM whole-block
    /// trampoline. The trampoline does the per-iter ARM step dispatch
    /// (32-bit fetch, ARM_LUT cond check + handler).
    ///
    /// `opcodes_ptr` points to the ARM opcode arena (32-bit u32s).
    /// Each opcode is one full ARM instruction word.
    pub fn emit_placeholder_arm_block(
        &mut self,
        opcodes_ptr: *const u32,
        opcodes_len: u32,
        entry_pc: u32,
    ) -> Option<CompiledFn> {
        use inkwell::AddressSpace;

        let replay_fn = self.replay_arm_fn?;

        let module = self.context.create_module("aot_arm_blk");
        let i32_t = self.context.i32_type();
        let i64_t = self.context.i64_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let trampoline_sig = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), i32_t.into(), i32_t.into()],
            false,
        );
        let trampoline_ref =
            module.add_function("rba_aot_replay_arm", trampoline_sig, None);
        self.engine
            .add_global_mapping(&trampoline_ref, replay_fn as usize);

        let block_sig = i32_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);
        self.next_id += 1;
        let block_name = format!("aot_arm_blk_{}", self.next_id);
        let block_fn = module.add_function(&block_name, block_sig, None);
        let entry = self.context.append_basic_block(block_fn, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let cpu_ctx = block_fn.get_nth_param(0).unwrap().into_pointer_value();

        let opcodes_ptr_const = i64_t.const_int(opcodes_ptr as u64, false);
        let opcodes_ptr_v = builder
            .build_int_to_ptr(opcodes_ptr_const, ptr_t, "opc_ptr").ok()?;
        let len_v = i32_t.const_int(opcodes_len as u64, false);
        let entry_pc_v = i32_t.const_int(entry_pc as u64, false);

        let call = builder
            .build_call(
                trampoline_ref,
                &[
                    cpu_ctx.into(),
                    opcodes_ptr_v.into(),
                    len_v.into(),
                    entry_pc_v.into(),
                ],
                "ret",
            ).ok()?;
        let ret = call.try_as_basic_value().unwrap_basic().into_int_value();
        builder.build_return(Some(&ret)).ok()?;

        self.engine.add_module(&module).ok()?;
        let raw = self.engine.get_function_address(&block_name).ok()?;
        Some(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// A0 pre-flight ABI sanity test. Compile a function that returns
    /// 42, JIT-execute it, return the result. If this returns 42 then
    /// the inkwell + LLVM 18 + libpolly-prefer-dynamic toolchain is
    /// wired correctly on this host. Anything else means env is broken
    /// and there's no point starting phase 1.
    #[cfg(test)]
    pub fn compile_constant_42(&mut self) -> Result<u32, String> {
        let module = self.context.create_module("hello_42");
        let i32_t = self.context.i32_type();
        let fn_ty = i32_t.fn_type(&[], false);
        self.next_id += 1;
        let name = format!("hello_42_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);
        let val = i32_t.const_int(42, false);
        builder
            .build_return(Some(&val))
            .map_err(|e| format!("build_return: {}", e))?;
        self.engine
            .add_module(&module)
            .map_err(|_| "add_module failed".to_string())?;
        let f: JitFunction<unsafe extern "C" fn() -> u32> = unsafe {
            self.engine
                .get_function(&name)
                .map_err(|e| format!("get_function: {}", e))?
        };
        Ok(unsafe { f.call() })
    }

    /// A5 compile-time budget probe. Compiles N placeholder block-shaped
    /// functions in the same module, returns wall time in ms. Each
    /// function is a 4-instruction sequence (gpr load + add + store +
    /// return) — simulates the per-block IR shape phase 0 will emit.
    /// Used to extrapolate compile time per pokeemerald-sized ROM and
    /// decide serial vs parallel.
    #[cfg(test)]
    pub fn compile_n_placeholder_blocks(&mut self, n: usize) -> Result<u128, String> {
        use inkwell::AddressSpace;
        let module = self.context.create_module("budget_probe");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());
        // canonical block ABI per I8: (cpu_ctx, pc_out) -> u32
        let fn_ty = i32_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        let start = std::time::Instant::now();

        for i in 0..n {
            let name = format!("blk_{}", i);
            let func = module.add_function(&name, fn_ty, None);
            let entry = self.context.append_basic_block(func, "entry");
            let builder = self.context.create_builder();
            builder.position_at_end(entry);

            // simulate 4 instructions: 4 gpr loads/adds/stores
            let cpu_ctx = func.get_nth_param(0).unwrap().into_pointer_value();
            for k in 0..4 {
                let off = i32_t.const_int((k * 4) as u64, false);
                let addr = unsafe {
                    builder
                        .build_in_bounds_gep(self.context.i8_type(), cpu_ctx, &[off], "p")
                        .map_err(|e| format!("gep: {}", e))?
                };
                let val = builder
                    .build_load(i32_t, addr, "v")
                    .map_err(|e| format!("load: {}", e))?
                    .into_int_value();
                let inc = builder
                    .build_int_add(val, i32_t.const_int(1, false), "i")
                    .map_err(|e| format!("add: {}", e))?;
                builder
                    .build_store(addr, inc)
                    .map_err(|e| format!("store: {}", e))?;
            }
            builder
                .build_return(Some(&i32_t.const_int(0, false)))
                .map_err(|e| format!("ret: {}", e))?;
        }

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module failed".to_string())?;

        // force codegen by looking up one function
        let _ = self
            .engine
            .get_function_address("blk_0")
            .map_err(|e| format!("get_function_address: {}", e))?;

        Ok(start.elapsed().as_millis())
    }
}

/// Phase-4P-B: returns true when this Thumb opcode would be handled
/// by an inline format in `emit_per_instr_thumb_block` given the
/// current AOT_INLINE_F* env-var settings. Used to decide whether
/// the whole block can use the baked-cycle fetch path: that requires
/// EVERY mid-block opcode to be inline (so no step_thumb fallback
/// runs and breaks the assumed-no-pipeline-shift contract).
///
/// Format dispatch order mirrors the if-chain in
/// `emit_per_instr_thumb_block`'s for-loop body. Block terminators
/// (F16 Bcc-taken, F17 SWI, F18 B, F19 lo) are handled by the AOT
/// scan at block boundaries — they don't appear mid-block here.
#[allow(clippy::too_many_arguments)]
fn thumb_opcode_in_inline_path(
    opcode: u32,
    f1: bool, f2: bool, f3: bool,
    f4_log: bool, f4_arith: bool, f4_shift: bool, f4_mul: bool,
    f5: bool, f6: bool, f7: bool, f8: bool,
    f9: bool, f10: bool,
    f11_str: bool, f11_ldr: bool,
    f12: bool, f13: bool, f14: bool, f15: bool,
    f19hi: bool,
) -> bool {
    let op = opcode as u16;
    let top3 = (op >> 13) & 0x7;
    let top4 = (op >> 12) & 0xf;
    let top5 = (op >> 11) & 0x1f;
    let top6 = (op >> 10) & 0x3f;
    let top8 = (op >> 8) & 0xff;

    if f1 && top3 == 0b000 && ((op >> 11) & 0x3) != 0b11 { return true; }
    if f2 && top5 == 0b00011 { return true; }
    if f3 && top3 == 0b001 { return true; }
    if top6 == 0b010000 {
        let f4_op = (op >> 6) & 0xf;
        let log = matches!(f4_op, 0b0000 | 0b0001 | 0b1000 | 0b1100 | 0b1110 | 0b1111);
        let arith = matches!(f4_op, 0b0101 | 0b0110 | 0b1001 | 0b1010 | 0b1011);
        let shift = matches!(f4_op, 0b0010 | 0b0011 | 0b0100 | 0b0111);
        let mul = f4_op == 0b1101;
        if (log && f4_log) || (arith && f4_arith) || (shift && f4_shift) || (mul && f4_mul) {
            return true;
        }
    }
    if f5 && top6 == 0b010001 { return true; }
    if f6 && top5 == 0b01001 { return true; }
    if f7 && top4 == 0b0101 && ((op >> 9) & 1) == 0 { return true; }
    if f8 && top4 == 0b0101 && ((op >> 9) & 1) == 1 { return true; }
    if f9 && top3 == 0b011 { return true; }
    if f10 && top4 == 0b1000 { return true; }
    if top4 == 0b1001 {
        let l = (op >> 11) & 1;
        if l == 0 && f11_str { return true; }
        if l == 1 && f11_ldr { return true; }
    }
    if f12 && top4 == 0b1010 { return true; }
    if f13 && top8 == 0b10110000 { return true; }
    if f14 && top4 == 0b1011 && ((op >> 9) & 0x3) == 0b10 { return true; }
    if f15 && top4 == 0b1100 { return true; }
    if f19hi && top5 == 0b11110 { return true; }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A0: confirm the LLVM 18 + inkwell toolchain links and runs an
    /// emitted function on this host. If this test fails, do NOT
    /// proceed with phase 1 — the env is broken.
    #[test]
    fn a0_jit_executes_constant_function() {
        let mut compiler = LlvmCompiler::new().expect("LlvmCompiler::new failed");
        let result = compiler
            .compile_constant_42()
            .expect("compile_constant_42 failed");
        assert_eq!(result, 42, "JIT-emitted constant must return 42");
    }

    /// A5: compile 1000 placeholder blocks in one module, print
    /// total wall time. Run with `cargo test -p arm7tdmi-aot
    /// a5_compile_budget -- --nocapture` to see the number.
    /// Pokeemerald has ~100k blocks, so projected wall time is
    /// 100x what this prints.
    #[test]
    fn a5_compile_budget() {
        let mut compiler = LlvmCompiler::new().expect("LlvmCompiler::new failed");
        let n = 5000;
        let elapsed_ms = compiler
            .compile_n_placeholder_blocks(n)
            .expect("compile_n_placeholder_blocks failed");
        let per_block_us = (elapsed_ms as f64 / n as f64) * 1000.0;
        let projected_pe_ms = (per_block_us * 100_000.0) / 1000.0;
        let projected_mk_ms = (per_block_us * 30_000.0) / 1000.0;
        println!(
            "A5 budget: compiled {} placeholder blocks in {} ms ({:.1} us/block); \
             projected PE (~100k blocks) = {:.0} ms, MK (~30k blocks) = {:.0} ms",
            n, elapsed_ms, per_block_us, projected_pe_ms, projected_mk_ms,
        );
        // No hard assertion — this is a measurement, not a gate.
        // The findings doc records the result and decides parallel
        // vs serial.
    }
}
