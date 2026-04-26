//! LLVM JIT compiler scaffold. Owns the leaked `Context` and the
//! `ExecutionEngine` for the lifetime of the AOT pass. Modules built
//! per-block (or per-page in parallel mode) are added here.

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
#[cfg(test)]
use inkwell::execution_engine::JitFunction;

use crate::replay::{AotAbortFn, AotFetchOnlyFn, AotReplayFn, AotStepFn};
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
        // Also gated on AOT_INLINE_F3=1 (read once, not per-opcode!).
        let f3_inline_env = std::env::var("AOT_INLINE_F3").map(|v| v == "1").unwrap_or(false);
        let inline_enabled = self.cpu_offsets.is_some()
            && self.fetch_only_thumb_fn.is_some()
            && f3_inline_env;
        let (fetch_only_ref, offsets) = if inline_enabled {
            let fetch_only_fn = self.fetch_only_thumb_fn.unwrap();
            let off = self.cpu_offsets.unwrap();
            let fo_sig = self.context.void_type()
                .fn_type(&[ptr_t.into(), i32_t.into()], false);
            let fo_name = format!("rba_aot_fetch_only_{}", id);
            let fo_ref = module.add_function(&fo_name, fo_sig, None);
            self.engine.add_global_mapping(&fo_ref, fetch_only_fn as usize);
            (Some(fo_ref), Some(off))
        } else {
            (None, None)
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
            if inline_enabled && f3_top5 == 0b00100 && f3_op == 0 {
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
