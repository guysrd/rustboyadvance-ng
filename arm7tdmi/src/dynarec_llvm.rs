//! LLVM-via-inkwell JIT backend for the ARM7TDMI dynarec.
//!
//! This module is the replacement for the Cranelift-based `dynarec` module.
//! Goal: per-instruction codegen quality at rustc-LTO parity (LLVM is rustc's
//! own backend), in exchange for slower compile time. Compile-time cost is
//! mitigated later by lazy compilation of hot blocks only.
//!
//! Migration phases:
//!   1. Scaffolding + hello-world JIT (this commit). Proves inkwell+LLVM 18
//!      links and runs an emitted function.
//!   2. Thumb body emit functions, one format at a time, mirroring the
//!      existing Cranelift versions in `crate::dynarec`. SDL-tested per
//!      shape.
//!   3. ARM compile path.
//!   4. Tail emits + chain mechanism + abort handling.
//!   5. Wire into `crate::cache` dispatch, remove Cranelift.

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
#[cfg(test)]
use inkwell::execution_engine::JitFunction;

/// Thumb shift kind for F1 (LSL/LSR/ASR). Mirrors the type in
/// `crate::dynarec` (Cranelift backend) so dynarec_llvm doesn't have
/// to pull in that whole module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
}

/// Thumb F2 right-hand operand: 3-bit immediate (always non-negative,
/// simplifies V flag formula) or a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thumb2Operand {
    Imm3(u8),
    Reg(u8),
}

/// Thumb format 5 high-register ops. MOV/ADD don't update flags
/// (a Thumb quirk); only CMP does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thumb5Op {
    Mov,
    Add,
    Cmp,
}

/// Thumb format 4 logical / compare ops. All write flags; only some
/// write back to Rd. Most preserve C, V; CMP/CMN/NEG compute full NZCV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thumb4Op {
    And,
    Eor,
    Orr,
    Bic,
    Mvn,
    Tst,
    Cmp,
    Cmn,
    Neg,
}

/// Function signature shared by every compiled block. Mirrors the Cranelift
/// shape (`*mut u32 gpr, *mut u32 cpsr, *mut u32 pc_out, *mut u8 cpu_ctx -> u32`).
/// Return value is the same `took` bit-encoding the dispatcher already
/// understands:
///   bit 0 = branch fired (pc_out populated, dispatcher reloads pipeline).
///   bit 1 = mid-block abort (state already saved, dispatcher yields).
pub type CompiledFn =
    unsafe extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32;

/// Memory access trampoline: `(*mut u8 cpu_ctx, u32 addr) -> u32`.
/// Same shape as the Cranelift backend's `BusLoad32Fn`. The compiled
/// LLVM block calls these directly via inkwell's `add_global_mapping`
/// to bridge into the bus implementation owned by the host.
pub type BusLoadFn = unsafe extern "C" fn(*mut u8, u32) -> u32;

/// Memory STORE trampoline: `(*mut u8 cpu_ctx, u32 addr, u32 value)`.
pub type BusStoreFn = unsafe extern "C" fn(*mut u8, u32, u32);

/// Set of bus trampolines the LLVM JIT can link against. Caller fills
/// these from `crate::dynarec::trampolines::for_cpu::<I>()` (or test
/// stubs) and passes the struct to `LlvmCompiler::register_trampolines`
/// once at compiler construction. The compile fns then refer to the
/// trampolines by name and inkwell maps the names to these pointers
/// at JIT link time.
#[derive(Clone, Copy, Default)]
pub struct LlvmBusTrampolines {
    /// Word load WITHOUT the +1I idle (used for fetches and PUSH/POP
    /// in the cranelift backend; same here).
    pub load_32: Option<BusLoadFn>,
    /// Word load WITH the +1I idle (LDR family).
    pub load_with_idle_32: Option<BusLoadFn>,
    /// Byte load with +1I idle (LDRB).
    pub load_with_idle_8: Option<BusLoadFn>,
    /// Word store, no +1I.
    pub store_32: Option<BusStoreFn>,
    /// Byte store, no +1I.
    pub store_8: Option<BusStoreFn>,
    /// Halfword load with +1I idle (LDRH).
    pub load_with_idle_16: Option<BusLoadFn>,
    /// Signed halfword load with +1I idle (LDSH — sign-extend).
    pub load_with_idle_sh: Option<BusLoadFn>,
    /// Halfword store, no +1I.
    pub store_16: Option<BusStoreFn>,
}

/// LLVM JIT compiler. One per CPU instance; freed on Drop, which releases
/// the generated machine-code pages back to the OS.
///
/// Lifetime layout matches inkwell's borrow rules: the `Context` outlives
/// everything; the `Module` is single-use per compile cycle and consumed by
/// the `ExecutionEngine` when adding it. We hold one ExecutionEngine and
/// add new modules to it as we compile blocks.
pub struct LlvmCompiler {
    context: &'static Context,
    engine: ExecutionEngine<'static>,
    /// Counter for generating unique function names within the module.
    /// Each compiled block gets `dynarec_block_<n>`.
    next_id: u64,
    /// Bus trampoline pointers, registered once via `register_trampolines`.
    /// Compiled blocks reference these by name (extern function declarations
    /// in the LLVM module); inkwell's `add_global_mapping` resolves the
    /// names to the function pointers stored here.
    trampolines: LlvmBusTrampolines,
}

impl LlvmCompiler {
    /// Build a new LLVM-backed JIT compiler. Initializes the native target
    /// and creates an empty module + execution engine.
    ///
    /// The `Context` is leaked to `'static` so the ExecutionEngine and the
    /// Modules it owns can borrow from it for the compiler's lifetime. CPU
    /// drop releases the engine + JIT-allocated code; the leaked Context
    /// is small and stays for the process lifetime, which matches typical
    /// emulator usage (one CPU per process).
    pub fn new() -> Result<Self, String> {
        // SAFETY: leaking a Context is the standard inkwell idiom for
        // multi-module JIT scenarios where modules need a stable parent.
        let context: &'static Context = Box::leak(Box::new(Context::create()));
        let module = context.create_module("dynarec_llvm");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| format!("create_jit_execution_engine: {}", e))?;
        Ok(Self {
            context,
            engine,
            next_id: 0,
            trampolines: LlvmBusTrampolines::default(),
        })
    }

    /// Register bus trampolines so compiled blocks can call into the
    /// host's memory implementation. Call ONCE after `new`, before
    /// any compile_* method that emits memory ops. Idempotent; the
    /// last call wins.
    pub fn register_trampolines(&mut self, t: LlvmBusTrampolines) {
        self.trampolines = t;
    }

    /// Helper: declare an extern function in `module` with the
    /// `(*mut u8, u32) -> u32` shape and bind it to `fn_ptr` via the
    /// execution engine's global mapping. Returns the FunctionValue
    /// so the builder can `build_call` it.
    fn import_load_trampoline<'a>(
        &self,
        module: &'a inkwell::module::Module<'static>,
        name: &str,
        fn_ptr: BusLoadFn,
    ) -> inkwell::values::FunctionValue<'a> {
        use inkwell::AddressSpace;
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());
        let fn_ty = i32_t.fn_type(&[ptr_t.into(), i32_t.into()], false);
        let func = module.add_function(name, fn_ty, None);
        // Bind the symbol name to the actual host function pointer.
        // SAFETY: BusLoadFn has the matching ABI; inkwell will hand
        // the pointer to the JIT's symbol resolver.
        self.engine.add_global_mapping(&func, fn_ptr as usize);
        func
    }

    /// Same as `import_load_trampoline` but for STORE-shape
    /// `(*mut u8, u32, u32)` trampolines (return type is void).
    fn import_store_trampoline<'a>(
        &self,
        module: &'a inkwell::module::Module<'static>,
        name: &str,
        fn_ptr: BusStoreFn,
    ) -> inkwell::values::FunctionValue<'a> {
        use inkwell::AddressSpace;
        let i32_t = self.context.i32_type();
        let void_t = self.context.void_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());
        let fn_ty = void_t.fn_type(&[ptr_t.into(), i32_t.into(), i32_t.into()], false);
        let func = module.add_function(name, fn_ty, None);
        self.engine.add_global_mapping(&func, fn_ptr as usize);
        func
    }

    /// Hello-world: compile and run a function that returns 42. Used in
    /// the scaffold-validation test to prove the LLVM toolchain links and
    /// executes generated code on this host.
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

    /// Emit a single-instruction compiled block for Thumb format 3 ADD
    /// imm8 (`ADD Rd, #imm8`). Exercises the full cpsr round-trip
    /// (load *cpsr_ptr → compute NZCV from rd+imm8 result → store back)
    /// plus gpr read+write. This is the pattern every flag-setting
    /// Thumb ALU format will follow.
    ///
    /// For Thumb3 ADD/SUB/CMP with imm8 ∈ [0, 255] the V-flag formula
    /// is the simplified one (sign bit of imm8 is always 0):
    ///   ADD V = (~rd & result) >> 31
    /// (matches the Cranelift codegen at `emit_thumb_format3` ADD path.)
    pub fn compile_thumb_format3_add(
        &mut self,
        rd: u8,
        imm8: u8,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );

        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        // Load gpr[rd]
        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep rd: {}", e))?
        };
        let rd_val = builder
            .build_load(i32_t, rd_addr, "rd_val")
            .map_err(|e| format!("load rd: {}", e))?
            .into_int_value();

        // result = rd + imm8
        let imm_val = i32_t.const_int(imm8 as u64, false);
        let result = builder
            .build_int_add(rd_val, imm_val, "result")
            .map_err(|e| format!("iadd: {}", e))?;

        // Store back gpr[rd] = result
        builder
            .build_store(rd_addr, result)
            .map_err(|e| format!("store rd: {}", e))?;

        // Flags
        // N = result >> 31 (top bit), shifted into bit 31 already
        let n_bit = builder
            .build_and(
                result,
                i32_t.const_int(0x8000_0000, false),
                "n_bit",
            )
            .map_err(|e| format!("and N: {}", e))?;
        // Z = (result == 0) << 30
        let zero_const = i32_t.const_int(0, false);
        let z_bool = builder
            .build_int_compare(IntPredicate::EQ, result, zero_const, "z_bool")
            .map_err(|e| format!("icmp Z: {}", e))?;
        let z_u32 = builder
            .build_int_z_extend(z_bool, i32_t, "z_u32")
            .map_err(|e| format!("zext Z: {}", e))?;
        let z_shifted = builder
            .build_left_shift(z_u32, i32_t.const_int(30, false), "z_shifted")
            .map_err(|e| format!("shl Z: {}", e))?;
        // C = (result < rd) (unsigned wrap) << 29
        let c_bool = builder
            .build_int_compare(IntPredicate::ULT, result, rd_val, "c_bool")
            .map_err(|e| format!("icmp C: {}", e))?;
        let c_u32 = builder
            .build_int_z_extend(c_bool, i32_t, "c_u32")
            .map_err(|e| format!("zext C: {}", e))?;
        let c_shifted = builder
            .build_left_shift(c_u32, i32_t.const_int(29, false), "c_shifted")
            .map_err(|e| format!("shl C: {}", e))?;
        // V = (~rd & result) >> 31 << 28 = (~rd & result & 0x80000000) >> 3
        let not_rd = builder
            .build_not(rd_val, "not_rd")
            .map_err(|e| format!("not: {}", e))?;
        let v_bits = builder
            .build_and(not_rd, result, "v_bits")
            .map_err(|e| format!("and V: {}", e))?;
        let v_top = builder
            .build_and(
                v_bits,
                i32_t.const_int(0x8000_0000, false),
                "v_top",
            )
            .map_err(|e| format!("and Vtop: {}", e))?;
        let v_shifted = builder
            .build_right_shift(v_top, i32_t.const_int(3, false), false, "v_shifted")
            .map_err(|e| format!("lshr V: {}", e))?;

        // Pack: cpsr = (cpsr & 0x0FFFFFFF) | (N | Z | C | V)
        let cpsr_old = builder
            .build_load(i32_t, cpsr_ptr, "cpsr_old")
            .map_err(|e| format!("load cpsr: {}", e))?
            .into_int_value();
        let cpsr_cleared = builder
            .build_and(
                cpsr_old,
                i32_t.const_int(0x0FFF_FFFF, false),
                "cpsr_cleared",
            )
            .map_err(|e| format!("and cpsr: {}", e))?;
        let nz = builder
            .build_or(n_bit, z_shifted, "nz")
            .map_err(|e| format!("or NZ: {}", e))?;
        let cv = builder
            .build_or(c_shifted, v_shifted, "cv")
            .map_err(|e| format!("or CV: {}", e))?;
        let flags = builder
            .build_or(nz, cv, "flags")
            .map_err(|e| format!("or flags: {}", e))?;
        let cpsr_new = builder
            .build_or(cpsr_cleared, flags, "cpsr_new")
            .map_err(|e| format!("or cpsr_new: {}", e))?;
        builder
            .build_store(cpsr_ptr, cpsr_new)
            .map_err(|e| format!("store cpsr: {}", e))?;

        // return 0 (no branch)
        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module failed".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F10 LDRH/STRH (`LDRH/STRH Rd, [Rb, #imm5*2]`).
    /// addr = gpr[rb] + imm5*2 (halfword stride).
    /// LDRH uses load_with_idle_16; STRH uses store_16.
    /// Note: this implementation handles the aligned case. Misaligned
    /// LDRH (addr & 1) requires CPSR.C update via ROR — the cranelift
    /// backend has special handling at `crate::dynarec::emit_thumb_format10`.
    /// Mirrored in this file's TODO list; needed for full SDL parity.
    pub fn compile_thumb_format10(
        &mut self,
        rd: u8,
        rb: u8,
        imm5: u8,
        load: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let (load_ref, store_ref) = if load {
            let fp = self
                .trampolines
                .load_with_idle_16
                .ok_or_else(|| "load_with_idle_16 not registered".to_string())?;
            (
                Some(self.import_load_trampoline(&module, "rba_load_idle_16", fp)),
                None,
            )
        } else {
            let fp = self
                .trampolines
                .store_16
                .ok_or_else(|| "store_16 not registered".to_string())?;
            (
                None,
                Some(self.import_store_trampoline(&module, "rba_store_16", fp)),
            )
        };

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpu_ctx = func.get_nth_param(3).unwrap().into_pointer_value();

        let rb_idx = i32_t.const_int(rb as u64, false);
        let rb_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rb_idx], "rb_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rb_val = builder
            .build_load(i32_t, rb_addr, "rb_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();
        let off = i32_t.const_int((imm5 as u64) * 2, false);
        let addr_val = builder
            .build_int_add(rb_val, off, "addr")
            .map_err(|e| format!("iadd: {}", e))?;

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };

        if load {
            let call = builder
                .build_call(
                    load_ref.unwrap(),
                    &[cpu_ctx.into(), addr_val.into()],
                    "ldrh_val",
                )
                .map_err(|e| format!("call: {}", e))?;
            let val = call.try_as_basic_value().unwrap_basic().into_int_value();
            // LDRH zero-extends to u32; mask just in case the trampoline
            // returns garbage upper bits.
            let masked = builder
                .build_and(val, i32_t.const_int(0xFFFF, false), "halfword_mask")
                .map_err(|e| format!("and: {}", e))?;
            builder
                .build_store(rd_addr, masked)
                .map_err(|e| format!("store: {}", e))?;
        } else {
            let rd_val = builder
                .build_load(i32_t, rd_addr, "rd_val")
                .map_err(|e| format!("load: {}", e))?
                .into_int_value();
            let store_val = builder
                .build_and(rd_val, i32_t.const_int(0xFFFF, false), "halfword_val")
                .map_err(|e| format!("and: {}", e))?;
            builder
                .build_call(
                    store_ref.unwrap(),
                    &[cpu_ctx.into(), addr_val.into(), store_val.into()],
                    "",
                )
                .map_err(|e| format!("call: {}", e))?;
        }

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F11 LDR/STR SP-relative (`LDR/STR Rd, [SP, #imm8*4]`).
    /// Same shape as F9 but rb is implicit (always R13/SP) and offset
    /// is imm8*4 (always word-sized).
    pub fn compile_thumb_format11(
        &mut self,
        rd: u8,
        imm8: u8,
        load: bool,
    ) -> Result<CompiledFn, String> {
        // SP is always R13. F11 is always word-sized (no byte variant).
        self.compile_thumb_format9(rd, /*rb=SP*/ 13, (imm8 as u32) * 4, load, false)
    }

    /// Emit a Thumb F12 load-address (`ADD Rd, PC, #imm8*4` or
    /// `ADD Rd, SP, #imm8*4` based on `from_sp`). Pure ALU — no
    /// memory access. Used for getting address of literals or
    /// stack-allocated values.
    pub fn compile_thumb_format12(
        &mut self,
        rd: u8,
        imm8: u8,
        instr_pc: u32,
        from_sp: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();

        let result = if from_sp {
            // ADD Rd, SP, #imm8*4: gpr[13] + imm8*4
            let sp_idx = i32_t.const_int(13, false);
            let sp_addr = unsafe {
                builder
                    .build_in_bounds_gep(i32_t, gpr_ptr, &[sp_idx], "sp_addr")
                    .map_err(|e| format!("gep: {}", e))?
            };
            let sp_val = builder
                .build_load(i32_t, sp_addr, "sp_val")
                .map_err(|e| format!("load: {}", e))?
                .into_int_value();
            let off = i32_t.const_int((imm8 as u64) * 4, false);
            builder
                .build_int_add(sp_val, off, "result")
                .map_err(|e| format!("iadd: {}", e))?
        } else {
            // ADD Rd, PC, #imm8*4: pc-folded constant.
            // Thumb pipeline: PC = instr_pc + 4 during decode, & ~3 alignment.
            let addr = ((instr_pc.wrapping_add(4)) & !3u32).wrapping_add((imm8 as u32) * 4);
            i32_t.const_int(addr as u64, false)
        };

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        builder
            .build_store(rd_addr, result)
            .map_err(|e| format!("store: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F13 ADD/SUB SP-imm (`ADD/SUB SP, #imm7*4`).
    /// Adjusts the stack pointer by ±imm7*4. No flag updates.
    pub fn compile_thumb_format13(
        &mut self,
        imm7: u8,
        sub: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();

        let sp_idx = i32_t.const_int(13, false);
        let sp_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[sp_idx], "sp_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let sp_val = builder
            .build_load(i32_t, sp_addr, "sp_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();
        let off = i32_t.const_int((imm7 as u64) * 4, false);
        let result = if sub {
            builder
                .build_int_sub(sp_val, off, "sp_new")
                .map_err(|e| format!("isub: {}", e))?
        } else {
            builder
                .build_int_add(sp_val, off, "sp_new")
                .map_err(|e| format!("iadd: {}", e))?
        };
        builder
            .build_store(sp_addr, result)
            .map_err(|e| format!("store: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F9 LDR/STR with imm5 offset (`LDR/STR Rd, [Rb, #imm5*4]`
    /// for word, `LDRB/STRB Rd, [Rb, #imm5]` for byte).
    /// addr = gpr[rb] + offset (offset = imm5*4 for word, imm5 for byte).
    /// LDR pays +1I via load_with_idle_*; STR uses no-idle store_*.
    /// No flag updates.
    pub fn compile_thumb_format9(
        &mut self,
        rd: u8,
        rb: u8,
        offset: u32,
        load: bool,
        byte: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        // Pick & import the right trampoline.
        let (load_ref, store_ref) = if load {
            let name = if byte { "rba_load_idle_8" } else { "rba_load_idle_32" };
            let fp = if byte {
                self.trampolines
                    .load_with_idle_8
                    .ok_or_else(|| "load_with_idle_8 not registered".to_string())?
            } else {
                self.trampolines
                    .load_with_idle_32
                    .ok_or_else(|| "load_with_idle_32 not registered".to_string())?
            };
            (
                Some(self.import_load_trampoline(&module, name, fp)),
                None,
            )
        } else {
            let name = if byte { "rba_store_8" } else { "rba_store_32" };
            let fp = if byte {
                self.trampolines
                    .store_8
                    .ok_or_else(|| "store_8 not registered".to_string())?
            } else {
                self.trampolines
                    .store_32
                    .ok_or_else(|| "store_32 not registered".to_string())?
            };
            (
                None,
                Some(self.import_store_trampoline(&module, name, fp)),
            )
        };

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpu_ctx = func.get_nth_param(3).unwrap().into_pointer_value();

        // addr = gpr[rb] + offset
        let rb_idx = i32_t.const_int(rb as u64, false);
        let rb_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rb_idx], "rb_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rb_val = builder
            .build_load(i32_t, rb_addr, "rb_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();
        let offset_const = i32_t.const_int(offset as u64, false);
        let addr_val = builder
            .build_int_add(rb_val, offset_const, "addr")
            .map_err(|e| format!("iadd: {}", e))?;

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };

        if load {
            let call = builder
                .build_call(
                    load_ref.unwrap(),
                    &[cpu_ctx.into(), addr_val.into()],
                    "ldr_val",
                )
                .map_err(|e| format!("call: {}", e))?;
            let val = call
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            // For LDRB the trampoline returns zero-extended u32 already
            // — store as-is. (The Cranelift backend masks to 0xFF for
            // safety; mirror that.)
            let final_val = if byte {
                builder
                    .build_and(val, i32_t.const_int(0xFF, false), "byte_mask")
                    .map_err(|e| format!("and: {}", e))?
            } else {
                val
            };
            builder
                .build_store(rd_addr, final_val)
                .map_err(|e| format!("store: {}", e))?;
        } else {
            let rd_val = builder
                .build_load(i32_t, rd_addr, "rd_val")
                .map_err(|e| format!("load: {}", e))?
                .into_int_value();
            let store_val = if byte {
                builder
                    .build_and(rd_val, i32_t.const_int(0xFF, false), "byte_val")
                    .map_err(|e| format!("and: {}", e))?
            } else {
                rd_val
            };
            builder
                .build_call(
                    store_ref.unwrap(),
                    &[cpu_ctx.into(), addr_val.into(), store_val.into()],
                    "",
                )
                .map_err(|e| format!("call: {}", e))?;
        }

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F6 LDR PC-relative (`LDR Rd, [PC, #imm8 * 4]`).
    /// Address is computed at codegen time from `instr_pc`:
    ///   addr = ((instr_pc + 4) & ~3) + imm8 * 4
    /// (PC + 4 because the Thumb pipeline-head convention leaves PC
    /// pointing two instrs ahead during decode; & ~3 aligns.)
    /// Calls `load_with_idle_32` (charges +1I per LDR semantics) and
    /// stores the result to gpr[rd]. No flag updates.
    pub fn compile_thumb_format6(
        &mut self,
        rd: u8,
        imm8: u8,
        instr_pc: u32,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        let load_fn = self
            .trampolines
            .load_with_idle_32
            .ok_or_else(|| "load_with_idle_32 trampoline not registered".to_string())?;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let load_ref = self.import_load_trampoline(&module, "rba_load_idle_32", load_fn);

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpu_ctx = func.get_nth_param(3).unwrap().into_pointer_value();

        // addr = ((instr_pc + 4) & ~3) + imm8 * 4 — fully constant.
        let addr_const =
            ((instr_pc.wrapping_add(4)) & !3u32).wrapping_add((imm8 as u32) * 4);
        let addr_val = i32_t.const_int(addr_const as u64, false);

        // Call the LDR trampoline.
        let call = builder
            .build_call(
                load_ref,
                &[cpu_ctx.into(), addr_val.into()],
                "ldr_val",
            )
            .map_err(|e| format!("call: {}", e))?;
        let val = call.try_as_basic_value().unwrap_basic().into_int_value();

        // Store to gpr[rd].
        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        builder
            .build_store(rd_addr, val)
            .map_err(|e| format!("store: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F5 non-branch high-register op. MOV/ADD just do
    /// the value math (NO flag update — Thumb high-reg quirk). CMP
    /// updates full NZCV like F4 CMP.
    ///
    /// Note: rd and rs are 4-bit register selectors (0..15). Caller
    /// is responsible for ensuring rd != 15 (R15 = PC, branch handled
    /// elsewhere as a tail).
    pub fn compile_thumb_format5(
        &mut self,
        op: Thumb5Op,
        rd: u8,
        rs: u8,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        debug_assert!(rd < 15, "F5 with rd=15 should be a branch tail");

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rd_val = builder
            .build_load(i32_t, rd_addr, "rd_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let rs_idx = i32_t.const_int(rs as u64, false);
        let rs_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rs_idx], "rs_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rs_val = builder
            .build_load(i32_t, rs_addr, "rs_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        match op {
            Thumb5Op::Mov => {
                builder
                    .build_store(rd_addr, rs_val)
                    .map_err(|e| format!("store: {}", e))?;
            }
            Thumb5Op::Add => {
                let result = builder
                    .build_int_add(rd_val, rs_val, "result")
                    .map_err(|e| format!("iadd: {}", e))?;
                builder
                    .build_store(rd_addr, result)
                    .map_err(|e| format!("store: {}", e))?;
            }
            Thumb5Op::Cmp => {
                // Full NZCV from rd - rs, no writeback.
                let result = builder
                    .build_int_sub(rd_val, rs_val, "result")
                    .map_err(|e| format!("isub: {}", e))?;
                let zero_const = i32_t.const_int(0, false);
                let n_bit = builder
                    .build_and(result, i32_t.const_int(0x8000_0000, false), "n")
                    .map_err(|e| format!("and: {}", e))?;
                let z_bool = builder
                    .build_int_compare(IntPredicate::EQ, result, zero_const, "z")
                    .map_err(|e| format!("icmp: {}", e))?;
                let z_u32 = builder
                    .build_int_z_extend(z_bool, i32_t, "z_u32")
                    .map_err(|e| format!("zext: {}", e))?;
                let z_shifted = builder
                    .build_left_shift(z_u32, i32_t.const_int(30, false), "z_sh")
                    .map_err(|e| format!("shl: {}", e))?;
                let c_bool = builder
                    .build_int_compare(IntPredicate::UGE, rd_val, rs_val, "c")
                    .map_err(|e| format!("icmp: {}", e))?;
                let c_u32 = builder
                    .build_int_z_extend(c_bool, i32_t, "c_u32")
                    .map_err(|e| format!("zext: {}", e))?;
                let c_shifted = builder
                    .build_left_shift(c_u32, i32_t.const_int(29, false), "c_sh")
                    .map_err(|e| format!("shl: {}", e))?;
                // V (sub): (rd ^ rs) & (rd ^ result) >> 31
                let xor1 = builder
                    .build_xor(rd_val, rs_val, "x1")
                    .map_err(|e| format!("xor: {}", e))?;
                let xor2 = builder
                    .build_xor(rd_val, result, "x2")
                    .map_err(|e| format!("xor: {}", e))?;
                let v_bits = builder
                    .build_and(xor1, xor2, "vb")
                    .map_err(|e| format!("and: {}", e))?;
                let v_top = builder
                    .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "vt")
                    .map_err(|e| format!("and: {}", e))?;
                let v_shifted = builder
                    .build_right_shift(v_top, i32_t.const_int(3, false), false, "vs")
                    .map_err(|e| format!("lshr: {}", e))?;
                let cpsr_old = builder
                    .build_load(i32_t, cpsr_ptr, "cpsr_old")
                    .map_err(|e| format!("load: {}", e))?
                    .into_int_value();
                let cpsr_cleared = builder
                    .build_and(cpsr_old, i32_t.const_int(0x0FFF_FFFF, false), "cl")
                    .map_err(|e| format!("and: {}", e))?;
                let nz = builder
                    .build_or(n_bit, z_shifted, "nz")
                    .map_err(|e| format!("or: {}", e))?;
                let cv = builder
                    .build_or(c_shifted, v_shifted, "cv")
                    .map_err(|e| format!("or: {}", e))?;
                let flags = builder
                    .build_or(nz, cv, "flags")
                    .map_err(|e| format!("or: {}", e))?;
                let cpsr_new = builder
                    .build_or(cpsr_cleared, flags, "cpsr_new")
                    .map_err(|e| format!("or: {}", e))?;
                builder
                    .build_store(cpsr_ptr, cpsr_new)
                    .map_err(|e| format!("store: {}", e))?;
            }
        }

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F4 logical / compare op. Per Thumb4Op:
    ///   And/Eor/Orr/Bic/Mvn:  writeback Rd, NZ flags, preserve C/V.
    ///   Tst:                    no writeback, NZ flags, preserve C/V.
    ///   Cmp:                    no writeback, full NZCV from rd - rs.
    ///   Cmn:                    no writeback, full NZCV from rd + rs.
    ///   Neg:                    writeback Rd = -rs, full NZCV.
    ///
    /// `rd` is the dest/lhs (loaded as left operand), `rs` is the
    /// source. For Mvn the lhs is unused (rd loaded then ignored,
    /// keeps the load uniform).
    pub fn compile_thumb_format4(
        &mut self,
        op: Thumb4Op,
        rd: u8,
        rs: u8,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rd_val = builder
            .build_load(i32_t, rd_addr, "rd_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let rs_idx = i32_t.const_int(rs as u64, false);
        let rs_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rs_idx], "rs_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rs_val = builder
            .build_load(i32_t, rs_addr, "rs_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let zero_const = i32_t.const_int(0, false);
        let writeback = matches!(
            op,
            Thumb4Op::And
                | Thumb4Op::Eor
                | Thumb4Op::Orr
                | Thumb4Op::Bic
                | Thumb4Op::Mvn
                | Thumb4Op::Neg
        );
        let full_nzcv = matches!(op, Thumb4Op::Cmp | Thumb4Op::Cmn | Thumb4Op::Neg);

        // Compute the result
        let result = match op {
            Thumb4Op::And | Thumb4Op::Tst => builder
                .build_and(rd_val, rs_val, "result")
                .map_err(|e| format!("and: {}", e))?,
            Thumb4Op::Eor => builder
                .build_xor(rd_val, rs_val, "result")
                .map_err(|e| format!("xor: {}", e))?,
            Thumb4Op::Orr => builder
                .build_or(rd_val, rs_val, "result")
                .map_err(|e| format!("or: {}", e))?,
            Thumb4Op::Bic => {
                let not_rs = builder
                    .build_not(rs_val, "not_rs")
                    .map_err(|e| format!("not: {}", e))?;
                builder
                    .build_and(rd_val, not_rs, "result")
                    .map_err(|e| format!("and: {}", e))?
            }
            Thumb4Op::Mvn => builder
                .build_not(rs_val, "result")
                .map_err(|e| format!("not: {}", e))?,
            Thumb4Op::Cmp => builder
                .build_int_sub(rd_val, rs_val, "result")
                .map_err(|e| format!("isub: {}", e))?,
            Thumb4Op::Cmn => builder
                .build_int_add(rd_val, rs_val, "result")
                .map_err(|e| format!("iadd: {}", e))?,
            Thumb4Op::Neg => builder
                .build_int_sub(zero_const, rs_val, "result")
                .map_err(|e| format!("isub: {}", e))?,
        };

        if writeback {
            builder
                .build_store(rd_addr, result)
                .map_err(|e| format!("store rd: {}", e))?;
        }

        // N, Z always come from result.
        let n_bit = builder
            .build_and(result, i32_t.const_int(0x8000_0000, false), "n")
            .map_err(|e| format!("and: {}", e))?;
        let z_bool = builder
            .build_int_compare(IntPredicate::EQ, result, zero_const, "z")
            .map_err(|e| format!("icmp: {}", e))?;
        let z_u32 = builder
            .build_int_z_extend(z_bool, i32_t, "z_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let z_shifted = builder
            .build_left_shift(z_u32, i32_t.const_int(30, false), "z_sh")
            .map_err(|e| format!("shl: {}", e))?;

        // C, V handling depends on op category.
        let cpsr_old = builder
            .build_load(i32_t, cpsr_ptr, "cpsr_old")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let (c_shifted, v_shifted) = if full_nzcv {
            // Full NZCV (Cmp/Cmn/Neg). Neg has rn=0 for flag computation.
            let (lhs, rhs, is_sub) = match op {
                Thumb4Op::Cmp => (rd_val, rs_val, true),
                Thumb4Op::Cmn => (rd_val, rs_val, false),
                Thumb4Op::Neg => (zero_const, rs_val, true),
                _ => unreachable!(),
            };
            let c_bool = if is_sub {
                builder
                    .build_int_compare(IntPredicate::UGE, lhs, rhs, "c")
                    .map_err(|e| format!("icmp: {}", e))?
            } else {
                builder
                    .build_int_compare(IntPredicate::ULT, result, lhs, "c")
                    .map_err(|e| format!("icmp: {}", e))?
            };
            let c_u32 = builder
                .build_int_z_extend(c_bool, i32_t, "c_u32")
                .map_err(|e| format!("zext: {}", e))?;
            let c_sh = builder
                .build_left_shift(c_u32, i32_t.const_int(29, false), "c_sh")
                .map_err(|e| format!("shl: {}", e))?;
            // V (sub):  (lhs ^ rhs) & (lhs ^ result) >> 31
            // V (add):  ~(lhs ^ rhs) & (lhs ^ result) >> 31
            let xor1 = builder
                .build_xor(lhs, rhs, "x1")
                .map_err(|e| format!("xor: {}", e))?;
            let xor2 = builder
                .build_xor(lhs, result, "x2")
                .map_err(|e| format!("xor: {}", e))?;
            let v_bits = if is_sub {
                builder
                    .build_and(xor1, xor2, "vb")
                    .map_err(|e| format!("and: {}", e))?
            } else {
                let not_xor1 = builder
                    .build_not(xor1, "nx1")
                    .map_err(|e| format!("not: {}", e))?;
                builder
                    .build_and(not_xor1, xor2, "vb")
                    .map_err(|e| format!("and: {}", e))?
            };
            let v_top = builder
                .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "vt")
                .map_err(|e| format!("and: {}", e))?;
            let v_sh = builder
                .build_right_shift(v_top, i32_t.const_int(3, false), false, "vs")
                .map_err(|e| format!("lshr: {}", e))?;
            (c_sh, v_sh)
        } else {
            // Logical: preserve C and V from cpsr_old.
            let c_pres = builder
                .build_and(cpsr_old, i32_t.const_int(0x2000_0000, false), "c_pr")
                .map_err(|e| format!("and: {}", e))?;
            let v_pres = builder
                .build_and(cpsr_old, i32_t.const_int(0x1000_0000, false), "v_pr")
                .map_err(|e| format!("and: {}", e))?;
            (c_pres, v_pres)
        };

        let cpsr_cleared = builder
            .build_and(cpsr_old, i32_t.const_int(0x0FFF_FFFF, false), "cpsr_cl")
            .map_err(|e| format!("and: {}", e))?;
        let nz = builder
            .build_or(n_bit, z_shifted, "nz")
            .map_err(|e| format!("or: {}", e))?;
        let cv = builder
            .build_or(c_shifted, v_shifted, "cv")
            .map_err(|e| format!("or: {}", e))?;
        let flags = builder
            .build_or(nz, cv, "flags")
            .map_err(|e| format!("or: {}", e))?;
        let cpsr_new = builder
            .build_or(cpsr_cleared, flags, "cpsr_new")
            .map_err(|e| format!("or: {}", e))?;
        builder
            .build_store(cpsr_ptr, cpsr_new)
            .map_err(|e| format!("store: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F2 ADD/SUB reg or imm3 (`ADD/SUB Rd, Rs, op2`).
    /// Always writes full NZCV (Thumb2 always has S-bit semantics).
    /// For `Imm3` the sign bit of imm is always 0, so V can use the
    /// simplified formula (same as F3 ADD/SUB):
    ///   ADD V = (~rs & result) >> 31
    ///   SUB V = (rs & ~result) >> 31
    /// For `Reg`, the general formula:
    ///   ADD V = (~(rs ^ rn) & (rs ^ result)) >> 31
    ///   SUB V = ((rs ^ rn) & (rs ^ result)) >> 31
    pub fn compile_thumb_format2(
        &mut self,
        rd: u8,
        rs: u8,
        operand: Thumb2Operand,
        sub: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        // Load Rs.
        let rs_idx = i32_t.const_int(rs as u64, false);
        let rs_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rs_idx], "rs_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rs_val = builder
            .build_load(i32_t, rs_addr, "rs_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        // Load operand (imm3 const or gpr[rn]).
        let (rhs, is_imm3) = match operand {
            Thumb2Operand::Imm3(v) => (i32_t.const_int(v as u64, false), true),
            Thumb2Operand::Reg(rn) => {
                let rn_idx = i32_t.const_int(rn as u64, false);
                let rn_addr = unsafe {
                    builder
                        .build_in_bounds_gep(i32_t, gpr_ptr, &[rn_idx], "rn_addr")
                        .map_err(|e| format!("gep: {}", e))?
                };
                let v = builder
                    .build_load(i32_t, rn_addr, "rn_val")
                    .map_err(|e| format!("load: {}", e))?
                    .into_int_value();
                (v, false)
            }
        };

        let result = if sub {
            builder
                .build_int_sub(rs_val, rhs, "result")
                .map_err(|e| format!("isub: {}", e))?
        } else {
            builder
                .build_int_add(rs_val, rhs, "result")
                .map_err(|e| format!("iadd: {}", e))?
        };

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        builder
            .build_store(rd_addr, result)
            .map_err(|e| format!("store rd: {}", e))?;

        // Flags
        let n_bit = builder
            .build_and(result, i32_t.const_int(0x8000_0000, false), "n")
            .map_err(|e| format!("and: {}", e))?;
        let zero_const = i32_t.const_int(0, false);
        let z_bool = builder
            .build_int_compare(IntPredicate::EQ, result, zero_const, "z")
            .map_err(|e| format!("icmp: {}", e))?;
        let z_u32 = builder
            .build_int_z_extend(z_bool, i32_t, "z_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let z_shifted = builder
            .build_left_shift(z_u32, i32_t.const_int(30, false), "z_sh")
            .map_err(|e| format!("shl: {}", e))?;

        let (c_bool, v_shifted) = if sub {
            // C = rs >= rhs (unsigned no-borrow)
            let c = builder
                .build_int_compare(IntPredicate::UGE, rs_val, rhs, "c")
                .map_err(|e| format!("icmp: {}", e))?;
            // V (sub):
            //   imm3:  V = (rs & ~result) >> 31
            //   reg:   V = ((rs ^ rn) & (rs ^ result)) >> 31
            let v_top = if is_imm3 {
                let not_result = builder
                    .build_not(result, "not_result")
                    .map_err(|e| format!("not: {}", e))?;
                let v_bits = builder
                    .build_and(rs_val, not_result, "v_bits")
                    .map_err(|e| format!("and: {}", e))?;
                builder
                    .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "v_top")
                    .map_err(|e| format!("and: {}", e))?
            } else {
                let xor1 = builder
                    .build_xor(rs_val, rhs, "xor_rs_rhs")
                    .map_err(|e| format!("xor: {}", e))?;
                let xor2 = builder
                    .build_xor(rs_val, result, "xor_rs_res")
                    .map_err(|e| format!("xor: {}", e))?;
                let v_bits = builder
                    .build_and(xor1, xor2, "v_bits")
                    .map_err(|e| format!("and: {}", e))?;
                builder
                    .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "v_top")
                    .map_err(|e| format!("and: {}", e))?
            };
            let v = builder
                .build_right_shift(v_top, i32_t.const_int(3, false), false, "v_sh")
                .map_err(|e| format!("lshr: {}", e))?;
            (c, v)
        } else {
            // ADD: C = result < rs (unsigned wrap)
            let c = builder
                .build_int_compare(IntPredicate::ULT, result, rs_val, "c")
                .map_err(|e| format!("icmp: {}", e))?;
            // V (add):
            //   imm3:  V = (~rs & result) >> 31
            //   reg:   V = (~(rs ^ rn) & (rs ^ result)) >> 31
            let v_top = if is_imm3 {
                let not_rs = builder
                    .build_not(rs_val, "not_rs")
                    .map_err(|e| format!("not: {}", e))?;
                let v_bits = builder
                    .build_and(not_rs, result, "v_bits")
                    .map_err(|e| format!("and: {}", e))?;
                builder
                    .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "v_top")
                    .map_err(|e| format!("and: {}", e))?
            } else {
                let xor1 = builder
                    .build_xor(rs_val, rhs, "xor_rs_rhs")
                    .map_err(|e| format!("xor: {}", e))?;
                let not_xor1 = builder
                    .build_not(xor1, "not_xor1")
                    .map_err(|e| format!("not: {}", e))?;
                let xor2 = builder
                    .build_xor(rs_val, result, "xor_rs_res")
                    .map_err(|e| format!("xor: {}", e))?;
                let v_bits = builder
                    .build_and(not_xor1, xor2, "v_bits")
                    .map_err(|e| format!("and: {}", e))?;
                builder
                    .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "v_top")
                    .map_err(|e| format!("and: {}", e))?
            };
            let v = builder
                .build_right_shift(v_top, i32_t.const_int(3, false), false, "v_sh")
                .map_err(|e| format!("lshr: {}", e))?;
            (c, v)
        };
        let c_u32 = builder
            .build_int_z_extend(c_bool, i32_t, "c_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let c_shifted = builder
            .build_left_shift(c_u32, i32_t.const_int(29, false), "c_sh")
            .map_err(|e| format!("shl: {}", e))?;

        // Pack
        let cpsr_old = builder
            .build_load(i32_t, cpsr_ptr, "cpsr_old")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();
        let cpsr_cleared = builder
            .build_and(cpsr_old, i32_t.const_int(0x0FFF_FFFF, false), "cpsr_cl")
            .map_err(|e| format!("and: {}", e))?;
        let nz = builder
            .build_or(n_bit, z_shifted, "nz")
            .map_err(|e| format!("or: {}", e))?;
        let cv = builder
            .build_or(c_shifted, v_shifted, "cv")
            .map_err(|e| format!("or: {}", e))?;
        let flags = builder
            .build_or(nz, cv, "flags")
            .map_err(|e| format!("or: {}", e))?;
        let cpsr_new = builder
            .build_or(cpsr_cleared, flags, "cpsr_new")
            .map_err(|e| format!("or: {}", e))?;
        builder
            .build_store(cpsr_ptr, cpsr_new)
            .map_err(|e| format!("store: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F1 shift-by-imm5 (`LSL/LSR/ASR Rd, Rs, #imm5`).
    /// Writes N, Z, and shifter-out C; preserves V.
    ///
    /// ARM7TDMI barrel-shifter quirks (mirrors `crate::dynarec::emit_thumb_format1`):
    ///   LSL #0:  result = Rs, C preserved (read from cpsr).
    ///   LSR #0 → LSR #32: result = 0, C = bit 31 of Rs.
    ///   ASR #0 → ASR #32: result = sign-extended Rs, C = bit 31 of Rs.
    ///   LSL #n (1..31): result = Rs << n, C = bit (32 - n) of Rs.
    ///   LSR #n (1..31): result = Rs >> n, C = bit (n - 1) of Rs.
    ///   ASR #n (1..31): result = (i32)Rs >> n, C = bit (n - 1) of Rs.
    pub fn compile_thumb_format1_shift(
        &mut self,
        kind: ShiftKind,
        rd: u8,
        rs: u8,
        imm5: u8,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        let rs_idx = i32_t.const_int(rs as u64, false);
        let rs_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rs_idx], "rs_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rs_val = builder
            .build_load(i32_t, rs_addr, "rs_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let one_const = i32_t.const_int(1, false);

        // Preserved-C bit pulled from cpsr_var when the shift quirk
        // calls for it (LSL #0 only). Computed eagerly; LLVM DCE drops
        // it if unused.
        let cpsr_old = builder
            .build_load(i32_t, cpsr_ptr, "cpsr_old")
            .map_err(|e| format!("load cpsr: {}", e))?
            .into_int_value();
        let preserved_c_top = builder
            .build_and(cpsr_old, i32_t.const_int(0x2000_0000, false), "c_old_top")
            .map_err(|e| format!("and: {}", e))?;
        let preserved_c = builder
            .build_right_shift(
                preserved_c_top,
                i32_t.const_int(29, false),
                false,
                "c_old",
            )
            .map_err(|e| format!("lshr: {}", e))?;

        let (result, new_c) = match (kind, imm5) {
            (ShiftKind::Lsl, 0) => (rs_val, preserved_c),
            (ShiftKind::Lsl, n) => {
                let r = builder
                    .build_left_shift(
                        rs_val,
                        i32_t.const_int(n as u64, false),
                        "lsl_r",
                    )
                    .map_err(|e| format!("lsl: {}", e))?;
                let c_shift = i32_t.const_int(32 - n as u64, false);
                let c_raw = builder
                    .build_right_shift(rs_val, c_shift, false, "c_raw")
                    .map_err(|e| format!("lshr: {}", e))?;
                let c = builder
                    .build_and(c_raw, one_const, "c_bit")
                    .map_err(|e| format!("and: {}", e))?;
                (r, c)
            }
            (ShiftKind::Lsr, 0) => {
                let r = i32_t.const_int(0, false);
                let c_raw = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int(31, false),
                        false,
                        "c_raw",
                    )
                    .map_err(|e| format!("lshr: {}", e))?;
                let c = builder
                    .build_and(c_raw, one_const, "c_bit")
                    .map_err(|e| format!("and: {}", e))?;
                (r, c)
            }
            (ShiftKind::Lsr, n) => {
                let r = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int(n as u64, false),
                        false,
                        "lsr_r",
                    )
                    .map_err(|e| format!("lshr: {}", e))?;
                let c_raw = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int((n - 1) as u64, false),
                        false,
                        "c_raw",
                    )
                    .map_err(|e| format!("lshr: {}", e))?;
                let c = builder
                    .build_and(c_raw, one_const, "c_bit")
                    .map_err(|e| format!("and: {}", e))?;
                (r, c)
            }
            (ShiftKind::Asr, 0) => {
                let r = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int(31, false),
                        true,
                        "asr_r",
                    )
                    .map_err(|e| format!("ashr: {}", e))?;
                let c_raw = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int(31, false),
                        false,
                        "c_raw",
                    )
                    .map_err(|e| format!("lshr: {}", e))?;
                let c = builder
                    .build_and(c_raw, one_const, "c_bit")
                    .map_err(|e| format!("and: {}", e))?;
                (r, c)
            }
            (ShiftKind::Asr, n) => {
                let r = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int(n as u64, false),
                        true,
                        "asr_r",
                    )
                    .map_err(|e| format!("ashr: {}", e))?;
                let c_raw = builder
                    .build_right_shift(
                        rs_val,
                        i32_t.const_int((n - 1) as u64, false),
                        false,
                        "c_raw",
                    )
                    .map_err(|e| format!("lshr: {}", e))?;
                let c = builder
                    .build_and(c_raw, one_const, "c_bit")
                    .map_err(|e| format!("and: {}", e))?;
                (r, c)
            }
        };

        // Store result back to gpr[rd]
        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        builder
            .build_store(rd_addr, result)
            .map_err(|e| format!("store rd: {}", e))?;

        // Flags: N, Z from result; C is new_c (already at bit 0); V preserved.
        let n_bit = builder
            .build_and(result, i32_t.const_int(0x8000_0000, false), "n")
            .map_err(|e| format!("and: {}", e))?;
        let zero_const = i32_t.const_int(0, false);
        let z_bool = builder
            .build_int_compare(IntPredicate::EQ, result, zero_const, "z")
            .map_err(|e| format!("icmp: {}", e))?;
        let z_u32 = builder
            .build_int_z_extend(z_bool, i32_t, "z_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let z_shifted = builder
            .build_left_shift(z_u32, i32_t.const_int(30, false), "z_sh")
            .map_err(|e| format!("shl: {}", e))?;
        let c_shifted = builder
            .build_left_shift(new_c, i32_t.const_int(29, false), "c_sh")
            .map_err(|e| format!("shl: {}", e))?;
        let v_preserved = builder
            .build_and(cpsr_old, i32_t.const_int(0x1000_0000, false), "v_pres")
            .map_err(|e| format!("and v: {}", e))?;

        // cpsr_new = (cpsr_old & 0x0FFFFFFF) | N | Z | C | V_preserved
        let cpsr_cleared = builder
            .build_and(cpsr_old, i32_t.const_int(0x0FFF_FFFF, false), "cpsr_cl")
            .map_err(|e| format!("and: {}", e))?;
        let nz = builder
            .build_or(n_bit, z_shifted, "nz")
            .map_err(|e| format!("or: {}", e))?;
        let cv = builder
            .build_or(c_shifted, v_preserved, "cv")
            .map_err(|e| format!("or: {}", e))?;
        let flags = builder
            .build_or(nz, cv, "flags")
            .map_err(|e| format!("or: {}", e))?;
        let cpsr_new = builder
            .build_or(cpsr_cleared, flags, "cpsr_new")
            .map_err(|e| format!("or: {}", e))?;
        builder
            .build_store(cpsr_ptr, cpsr_new)
            .map_err(|e| format!("store cpsr: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a Thumb F3 SUB imm8 (`SUB Rd, #imm8`). Same shape as ADD
    /// but `result = rd - imm8` and the C/V flags differ:
    ///   C = (rd >= imm8)        unsigned no-borrow
    ///   V = (rd & ~result) >> 31  imm8 sign bit always 0
    pub fn compile_thumb_format3_sub(
        &mut self,
        rd: u8,
        imm8: u8,
    ) -> Result<CompiledFn, String> {
        self.compile_thumb_format3_sub_or_cmp(rd, imm8, /*writeback*/ true)
    }

    /// Emit a Thumb F3 CMP imm8 (`CMP Rd, #imm8`). Same as SUB but no
    /// writeback to Rd. Just sets NZCV from rd - imm8.
    pub fn compile_thumb_format3_cmp(
        &mut self,
        rd: u8,
        imm8: u8,
    ) -> Result<CompiledFn, String> {
        self.compile_thumb_format3_sub_or_cmp(rd, imm8, /*writeback*/ false)
    }

    fn compile_thumb_format3_sub_or_cmp(
        &mut self,
        rd: u8,
        imm8: u8,
        writeback: bool,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;
        use inkwell::IntPredicate;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());

        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );
        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let cpsr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let rd_val = builder
            .build_load(i32_t, rd_addr, "rd_val")
            .map_err(|e| format!("load: {}", e))?
            .into_int_value();

        let imm_val = i32_t.const_int(imm8 as u64, false);
        let result = builder
            .build_int_sub(rd_val, imm_val, "result")
            .map_err(|e| format!("isub: {}", e))?;

        if writeback {
            builder
                .build_store(rd_addr, result)
                .map_err(|e| format!("store rd: {}", e))?;
        }

        // N from top bit, Z from result == 0
        let n_bit = builder
            .build_and(result, i32_t.const_int(0x8000_0000, false), "n")
            .map_err(|e| format!("and N: {}", e))?;
        let zero_const = i32_t.const_int(0, false);
        let z_bool = builder
            .build_int_compare(IntPredicate::EQ, result, zero_const, "z")
            .map_err(|e| format!("icmp Z: {}", e))?;
        let z_u32 = builder
            .build_int_z_extend(z_bool, i32_t, "z_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let z_shifted = builder
            .build_left_shift(z_u32, i32_t.const_int(30, false), "z_sh")
            .map_err(|e| format!("shl: {}", e))?;
        // C = rd >= imm8 (unsigned no-borrow)
        let c_bool = builder
            .build_int_compare(IntPredicate::UGE, rd_val, imm_val, "c")
            .map_err(|e| format!("icmp C: {}", e))?;
        let c_u32 = builder
            .build_int_z_extend(c_bool, i32_t, "c_u32")
            .map_err(|e| format!("zext: {}", e))?;
        let c_shifted = builder
            .build_left_shift(c_u32, i32_t.const_int(29, false), "c_sh")
            .map_err(|e| format!("shl: {}", e))?;
        // V = rd & ~result & 0x8000_0000 >> 3
        let not_result = builder
            .build_not(result, "not_result")
            .map_err(|e| format!("not: {}", e))?;
        let v_bits = builder
            .build_and(rd_val, not_result, "v_bits")
            .map_err(|e| format!("and V: {}", e))?;
        let v_top = builder
            .build_and(v_bits, i32_t.const_int(0x8000_0000, false), "v_top")
            .map_err(|e| format!("and Vtop: {}", e))?;
        let v_shifted = builder
            .build_right_shift(v_top, i32_t.const_int(3, false), false, "v_sh")
            .map_err(|e| format!("lshr: {}", e))?;

        let cpsr_old = builder
            .build_load(i32_t, cpsr_ptr, "cpsr_old")
            .map_err(|e| format!("load cpsr: {}", e))?
            .into_int_value();
        let cpsr_cleared = builder
            .build_and(cpsr_old, i32_t.const_int(0x0FFF_FFFF, false), "cpsr_cl")
            .map_err(|e| format!("and cpsr: {}", e))?;
        let nz = builder
            .build_or(n_bit, z_shifted, "nz")
            .map_err(|e| format!("or NZ: {}", e))?;
        let cv = builder
            .build_or(c_shifted, v_shifted, "cv")
            .map_err(|e| format!("or CV: {}", e))?;
        let flags = builder
            .build_or(nz, cv, "flags")
            .map_err(|e| format!("or flags: {}", e))?;
        let cpsr_new = builder
            .build_or(cpsr_cleared, flags, "cpsr_new")
            .map_err(|e| format!("or cpsr_new: {}", e))?;
        builder
            .build_store(cpsr_ptr, cpsr_new)
            .map_err(|e| format!("store cpsr: {}", e))?;

        builder
            .build_return(Some(&i32_t.const_int(0, false)))
            .map_err(|e| format!("ret: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }

    /// Emit a single-instruction compiled block for Thumb format 3 MOV
    /// imm8 (`MOV Rd, #imm8`). This is the simplest non-trivial Thumb
    /// shape: imm8 constant goes into `gpr[rd]`, no flag work, return 0
    /// (no branch fired, dispatcher continues).
    ///
    /// First real Thumb format wired up in the LLVM backend. Mirrors
    /// `crate::dynarec::emit_thumb_format3` (Cranelift) for the MOV path,
    /// but emits LLVM IR via inkwell.
    pub fn compile_thumb_format3_mov(
        &mut self,
        rd: u8,
        imm8: u8,
    ) -> Result<CompiledFn, String> {
        use inkwell::AddressSpace;

        let module = self.context.create_module("thumb_block");
        let i32_t = self.context.i32_type();
        let i8_t = self.context.i8_type();
        let ptr_t = self.context.ptr_type(AddressSpace::default());
        let _ = i8_t;

        // (*mut u32 gpr, *mut u32 cpsr, *mut u32 pc_out, *mut u8 cpu_ctx) -> u32
        let fn_ty = i32_t.fn_type(
            &[ptr_t.into(), ptr_t.into(), ptr_t.into(), ptr_t.into()],
            false,
        );

        self.next_id += 1;
        let name = format!("dynarec_block_{}", self.next_id);
        let func = module.add_function(&name, fn_ty, None);
        let entry = self.context.append_basic_block(func, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let gpr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();

        // gpr[rd] = imm8. GEP to gpr_ptr + rd (i32 indexing — element-typed
        // GEP) then store the constant.
        let rd_idx = i32_t.const_int(rd as u64, false);
        let rd_addr = unsafe {
            builder
                .build_in_bounds_gep(i32_t, gpr_ptr, &[rd_idx], "rd_addr")
                .map_err(|e| format!("gep: {}", e))?
        };
        let imm_val = i32_t.const_int(imm8 as u64, false);
        builder
            .build_store(rd_addr, imm_val)
            .map_err(|e| format!("store: {}", e))?;

        // return 0 (took = 0, no branch, no abort — dispatcher continues
        // at cpu.pc which was already advanced by the per-iter fetch).
        // This compiled-fn shape doesn't fetch yet (no per-iter loop) —
        // fetch + cpsr handling come in subsequent migration commits.
        let zero = i32_t.const_int(0, false);
        builder
            .build_return(Some(&zero))
            .map_err(|e| format!("build_return: {}", e))?;

        self.engine
            .add_module(&module)
            .map_err(|_| "add_module failed".to_string())?;
        let raw = self
            .engine
            .get_function_address(&name)
            .map_err(|e| format!("get_function_address: {}", e))?;
        Ok(unsafe { std::mem::transmute::<usize, CompiledFn>(raw) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scaffold-validation test: confirms that LLVM 18 + inkwell link
    /// against this host, and a trivially-compiled function can be
    /// dispatched via the JIT execution engine.
    #[test]
    fn jit_executes_constant_function() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let result = compiler.compile_constant_42().expect("compile");
        assert_eq!(result, 42);
    }

    /// Thumb F3 ADD imm8 with full NZCV flag handling. Validates the
    /// cpsr round-trip pattern (load → modify → store) that every
    /// flag-setting Thumb format will use.
    #[test]
    fn thumb_f3_add_imm_writes_rd_and_flags() {
        let mut compiler = LlvmCompiler::new().expect("new");
        // ADD r3, #5: r3 starts at 100, expected r3 = 105.
        // Flags: N=0, Z=0, C=0 (no overflow), V=0 (no signed overflow).
        let func = compiler
            .compile_thumb_format3_add(3, 5)
            .expect("compile ADD r3, #5");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[3] = 100;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut cpu_ctx: u8 = 0;
        let took = unsafe {
            func(
                gpr.as_mut_ptr(),
                &mut cpsr,
                &mut pc_out,
                &mut cpu_ctx as *mut u8,
            )
        };
        assert_eq!(took, 0);
        assert_eq!(gpr[3], 105);
        // No flags set (positive non-zero result, no overflow).
        assert_eq!(cpsr & 0xF000_0000, 0, "NZCV should be zero");
    }

    /// Thumb F3 ADD imm8 with overflow into Z flag.
    #[test]
    fn thumb_f3_add_imm_z_flag() {
        let mut compiler = LlvmCompiler::new().expect("new");
        // ADD r0, #1: r0 starts at 0xFFFF_FFFF.
        // Result wraps to 0 → Z=1, C=1 (unsigned wrap). N=0. V=0
        // (sign-bit pattern: 1 + 0 = 0; ~rd=0 & result=0 → V=0).
        let func = compiler
            .compile_thumb_format3_add(0, 1)
            .expect("compile ADD r0, #1");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0xFFFF_FFFF;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut cpu_ctx: u8 = 0;
        unsafe {
            func(
                gpr.as_mut_ptr(),
                &mut cpsr,
                &mut pc_out,
                &mut cpu_ctx as *mut u8,
            );
        }
        assert_eq!(gpr[0], 0);
        // Z=1 (bit 30), C=1 (bit 29). N=0, V=0.
        assert_eq!(cpsr & 0xF000_0000, 0x6000_0000,
            "Z + C should be set: cpsr = 0x{:08x}", cpsr);
    }

    /// F10 LDRH: stub returns 0xAABB_CCDD; verify only low 16 bits
    /// land in gpr[rd].
    #[test]
    fn thumb_f10_ldrh_zero_extends() {
        unsafe extern "C" fn fake_ldrh(_ctx: *mut u8, _addr: u32) -> u32 {
            0xAABB_CCDD
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            load_with_idle_16: Some(fake_ldrh),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format10(0, 1, 4, /*load*/ true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xCCDD);
    }

    /// F10 STRH: stub captures (addr, val); verify val is masked to 16 bits.
    #[test]
    fn thumb_f10_strh_captures_halfword() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CAPTURED_ADDR: AtomicU32 = AtomicU32::new(0);
        static CAPTURED_VAL: AtomicU32 = AtomicU32::new(0);
        unsafe extern "C" fn fake_strh(_ctx: *mut u8, addr: u32, val: u32) {
            CAPTURED_ADDR.store(addr, Ordering::Relaxed);
            CAPTURED_VAL.store(val, Ordering::Relaxed);
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            store_16: Some(fake_strh),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format10(2, 5, 8, /*load*/ false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[5] = 0x0300_2000;
        gpr[2] = 0x1234_5678;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(CAPTURED_ADDR.load(Ordering::Relaxed), 0x0300_2010);
        assert_eq!(CAPTURED_VAL.load(Ordering::Relaxed), 0x5678);
    }

    /// F11 LDR SP-rel: thin wrapper over F9 with rb=SP, offset=imm8*4.
    #[test]
    fn thumb_f11_ldr_sp_rel() {
        unsafe extern "C" fn fake_load(_ctx: *mut u8, addr: u32) -> u32 {
            addr // echo
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            load_with_idle_32: Some(fake_load),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format11(2, /*imm8*/ 5, /*load*/ true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[13] = 0x0300_7E00; // SP
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // addr = 0x0300_7E00 + 5*4 = 0x0300_7E14, echoed.
        assert_eq!(gpr[2], 0x0300_7E14);
    }

    /// F12 ADD Rd, PC, #imm8*4: pure ALU, addr-of-PC.
    #[test]
    fn thumb_f12_add_pc() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format12(0, /*imm8*/ 4, /*instr_pc*/ 0x0800_0100, /*from_sp*/ false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // (0x0800_0100 + 4) & ~3 = 0x0800_0104, + 4*4 = 0x0800_0114.
        assert_eq!(gpr[0], 0x0800_0114);
    }

    /// F12 ADD Rd, SP, #imm8*4: gpr[rd] = SP + imm8*4.
    #[test]
    fn thumb_f12_add_sp() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format12(0, 5, 0, /*from_sp*/ true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[13] = 0x0300_8000;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // 0x0300_8000 + 20 = 0x0300_8014.
        assert_eq!(gpr[0], 0x0300_8014);
    }

    /// F13 SUB SP, #imm7*4: shrinks the stack.
    #[test]
    fn thumb_f13_sub_sp() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format13(/*imm7*/ 4, /*sub*/ true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[13] = 0x0300_8000;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // SP -= 4*4 = 16. 0x0300_8000 - 16 = 0x0300_7FF0.
        assert_eq!(gpr[13], 0x0300_7FF0);
    }

    /// F9 LDR word from a known address. Stub trampoline returns the
    /// addr it was called with so the test can verify the address
    /// computation (gpr[rb] + offset).
    #[test]
    fn thumb_f9_ldr_word_addr_compute() {
        unsafe extern "C" fn fake_load(_ctx: *mut u8, addr: u32) -> u32 {
            // Echo the addr so test can verify gpr[rb] + offset.
            addr
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            load_with_idle_32: Some(fake_load),
            ..Default::default()
        });
        // LDR r3, [r4, #0x10]: addr = gpr[4] + 0x10.
        let func = compiler
            .compile_thumb_format9(3, 4, 0x10, /*load*/ true, /*byte*/ false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[4] = 0x0200_1000;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // addr = 0x0200_1000 + 0x10 = 0x0200_1010, echoed back into gpr[3].
        assert_eq!(gpr[3], 0x0200_1010);
    }

    /// F9 STR word: stub captures (addr, value) it was called with.
    #[test]
    fn thumb_f9_str_word_captures_addr_value() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CAPTURED_ADDR: AtomicU32 = AtomicU32::new(0);
        static CAPTURED_VAL: AtomicU32 = AtomicU32::new(0);
        unsafe extern "C" fn fake_store(_ctx: *mut u8, addr: u32, val: u32) {
            CAPTURED_ADDR.store(addr, Ordering::Relaxed);
            CAPTURED_VAL.store(val, Ordering::Relaxed);
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            store_32: Some(fake_store),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format9(2, 5, 0x20, /*load*/ false, /*byte*/ false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[5] = 0x0300_0000;
        gpr[2] = 0xDEAD_BEEF;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(CAPTURED_ADDR.load(Ordering::Relaxed), 0x0300_0020);
        assert_eq!(CAPTURED_VAL.load(Ordering::Relaxed), 0xDEAD_BEEF);
    }

    /// F9 LDRB: byte load, value masked to 0xFF.
    #[test]
    fn thumb_f9_ldrb_masks_byte() {
        unsafe extern "C" fn fake_loadb(_ctx: *mut u8, _addr: u32) -> u32 {
            0xAABB_CCDD // upper bits should be masked off
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            load_with_idle_8: Some(fake_loadb),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format9(0, 1, 0, /*load*/ true, /*byte*/ true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xDD); // only low byte
    }

    /// F6 LDR PC-relative: install a stub trampoline that returns a
    /// known sentinel, run a compiled `LDR r3, [PC, #4]` block, assert
    /// gpr[3] holds the sentinel.
    #[test]
    fn thumb_f6_ldr_pc_rel() {
        // Stub trampoline — ignores ctx + addr, returns 0xCAFEBABE.
        unsafe extern "C" fn fake_ldr(_ctx: *mut u8, _addr: u32) -> u32 {
            0xCAFEBABE
        }
        let mut compiler = LlvmCompiler::new().expect("new");
        compiler.register_trampolines(LlvmBusTrampolines {
            load_with_idle_32: Some(fake_ldr),
            ..Default::default()
        });
        let func = compiler
            .compile_thumb_format6(3, /*imm8*/ 1, /*instr_pc*/ 0x0800_0100)
            .expect("compile LDR PC-rel");
        let mut gpr: [u32; 15] = [0; 15];
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[3], 0xCAFEBABE);
    }

    /// F5 MOV: r8 ← r2 (high reg dest, low reg src). No flag update.
    #[test]
    fn thumb_f5_mov_high_reg() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format5(Thumb5Op::Mov, 8, 2)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[2] = 0xCAFEBABE;
        // Pre-set NZCV to verify they're untouched by F5 MOV.
        let mut cpsr: u32 = 0xF000_0000;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[8], 0xCAFEBABE);
        // Flags untouched.
        assert_eq!(cpsr, 0xF000_0000);
    }

    /// F5 ADD: r10 += r0, no flag update.
    #[test]
    fn thumb_f5_add_high_reg() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format5(Thumb5Op::Add, 10, 0)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[10] = 100;
        gpr[0] = 50;
        let mut cpsr: u32 = 0xF000_0000;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[10], 150);
        // Flags untouched.
        assert_eq!(cpsr, 0xF000_0000);
    }

    /// F5 CMP: full NZCV.
    #[test]
    fn thumb_f5_cmp_high_reg_less() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format5(Thumb5Op::Cmp, 8, 2)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[8] = 5;
        gpr[2] = 10;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // r8 unchanged. 5 - 10 = -5: N=1, C=0 (borrow), Z=0, V=0.
        assert_eq!(gpr[8], 5);
        assert_eq!(cpsr & 0xF000_0000, 0x8000_0000);
    }

    /// F4 AND: r0 = 0xFF00, r1 = 0x0FF0, AND r0, r0, r1 → r0 = 0x0F00.
    /// Logical, NZ updated, C/V preserved.
    #[test]
    fn thumb_f4_and() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::And, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0xFF00;
        gpr[1] = 0x0FF0;
        // Pre-set C+V in cpsr to verify they're preserved.
        let mut cpsr: u32 = 0x3000_0000;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0x0F00);
        // C, V preserved (bits 29, 28). N=0, Z=0.
        assert_eq!(cpsr & 0xF000_0000, 0x3000_0000);
    }

    /// F4 EOR: bitwise XOR.
    #[test]
    fn thumb_f4_eor() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Eor, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0xAAAA;
        gpr[1] = 0x5555;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xFFFF);
    }

    /// F4 BIC (bit clear): r0 & ~r1.
    #[test]
    fn thumb_f4_bic() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Bic, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0xFFFF;
        gpr[1] = 0x00F0;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xFF0F);
    }

    /// F4 MVN: bitwise not. rs's source value.
    #[test]
    fn thumb_f4_mvn() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Mvn, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[1] = 0x0000_FFFF;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xFFFF_0000);
        // N=1 (top bit set after invert).
        assert_eq!(cpsr & 0x8000_0000, 0x8000_0000);
    }

    /// F4 TST: AND but no writeback. Sets flags from result.
    #[test]
    fn thumb_f4_tst_zero() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Tst, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0xFF00;
        gpr[1] = 0x00FF;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // r0 unchanged (no writeback).
        assert_eq!(gpr[0], 0xFF00);
        // Z=1 (AND result is 0).
        assert_eq!(cpsr & 0x4000_0000, 0x4000_0000);
    }

    /// F4 NEG: 0 - rs, full NZCV.
    #[test]
    fn thumb_f4_neg() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Neg, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[1] = 5;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], (-5_i32) as u32);
        // N=1 (negative result), C=0 (0 < 5 unsigned, borrow).
        assert_eq!(cpsr & 0x8000_0000, 0x8000_0000);
        assert_eq!(cpsr & 0x2000_0000, 0);
    }

    /// F4 CMP: rd - rs without writeback.
    #[test]
    fn thumb_f4_cmp_equal() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format4(Thumb4Op::Cmp, 0, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 42;
        gpr[1] = 42;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        // r0 unchanged. Z=1, C=1.
        assert_eq!(gpr[0], 42);
        assert_eq!(cpsr & 0xF000_0000, 0x6000_0000);
    }

    /// F2 ADD reg: r0 = 100, r1 = 50, ADD r2, r0, r1 → r2=150, no flags.
    #[test]
    fn thumb_f2_add_reg() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format2(2, 0, Thumb2Operand::Reg(1), false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 100;
        gpr[1] = 50;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[2], 150);
        assert_eq!(cpsr & 0xF000_0000, 0);
    }

    /// F2 SUB imm3: r5 = 10, SUB r6, r5, #3 → r6=7, C=1, no other flags.
    #[test]
    fn thumb_f2_sub_imm3() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format2(6, 5, Thumb2Operand::Imm3(3), true)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[5] = 10;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[6], 7);
        // C=1 (rs >= imm3, so no borrow). N=0, Z=0, V=0.
        assert_eq!(cpsr & 0xF000_0000, 0x2000_0000);
    }

    /// F2 ADD reg with signed overflow: 0x7FFFFFFF + 1 = 0x80000000.
    /// V flag set (positive + positive = negative).
    #[test]
    fn thumb_f2_add_reg_signed_overflow() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format2(2, 0, Thumb2Operand::Reg(1), false)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[0] = 0x7FFF_FFFF;
        gpr[1] = 1;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[2], 0x8000_0000);
        // N=1 (top bit), V=1 (signed overflow). Z=0, C=0.
        assert_eq!(cpsr & 0xF000_0000, 0x9000_0000);
    }

    /// F1 LSL: r1 = 0x12345678, LSL r0, r1, #4 → r0 = 0x23456780,
    /// C = bit 28 of r1 = 1 (after shifting up by 4, that bit fell off).
    #[test]
    fn thumb_f1_lsl_imm() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format1_shift(ShiftKind::Lsl, 0, 1, 4)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[1] = 0x1234_5678;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0x2345_6780);
        // bit (32 - 4) = bit 28 of 0x1234_5678 → 0x1 → C set.
        assert_eq!(
            cpsr & 0x2000_0000,
            0x2000_0000,
            "C bit should be set: cpsr = 0x{:08x}",
            cpsr
        );
    }

    /// F1 LSR by 1, simple bit drop.
    #[test]
    fn thumb_f1_lsr_imm() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format1_shift(ShiftKind::Lsr, 0, 1, 1)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[1] = 0x0000_0003;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 1);
        // bit 0 of original = 1 → C = 1.
        assert_eq!(cpsr & 0x2000_0000, 0x2000_0000);
    }

    /// F1 ASR sign-extends. r1 = 0xF000_0000, ASR r0, r1, #4 →
    /// r0 = 0xFF00_0000.
    #[test]
    fn thumb_f1_asr_imm_sign_extend() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format1_shift(ShiftKind::Asr, 0, 1, 4)
            .expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[1] = 0xF000_0000;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[0], 0xFF00_0000);
        // N = 1 (top bit of result).
        assert_eq!(cpsr & 0x8000_0000, 0x8000_0000);
    }

    /// F3 SUB imm: r4 = 100, SUB r4, #5 → r4=95, C=1 (no borrow), N=0,
    /// Z=0, V=0.
    #[test]
    fn thumb_f3_sub_imm_no_borrow() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler.compile_thumb_format3_sub(4, 5).expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[4] = 100;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[4], 95);
        // C=1, others 0.
        assert_eq!(cpsr & 0xF000_0000, 0x2000_0000);
    }

    /// F3 CMP imm: like SUB but no writeback. r4=100, CMP r4, #100 →
    /// r4 unchanged, Z=1, C=1 (rd >= imm8 holds when equal).
    #[test]
    fn thumb_f3_cmp_imm_equal() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler.compile_thumb_format3_cmp(4, 100).expect("compile");
        let mut gpr: [u32; 15] = [0; 15];
        gpr[4] = 100;
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut ctx: u8 = 0;
        unsafe {
            func(gpr.as_mut_ptr(), &mut cpsr, &mut pc_out, &mut ctx as *mut u8);
        }
        assert_eq!(gpr[4], 100, "CMP shouldnt write rd");
        // Z=1 (bit 30), C=1 (bit 29), N=0, V=0.
        assert_eq!(cpsr & 0xF000_0000, 0x6000_0000);
    }

    /// First Thumb format end-to-end via LLVM: F3 MOV Rd, #imm8.
    /// Compile a block consisting of one `MOV r5, #42` instruction,
    /// run it against a real gpr array, assert gpr[5] = 42, and
    /// confirm took=0 (no branch).
    #[test]
    fn thumb_f3_mov_imm_writes_rd() {
        let mut compiler = LlvmCompiler::new().expect("new");
        let func = compiler
            .compile_thumb_format3_mov(5, 42)
            .expect("compile MOV r5, #42");

        let mut gpr: [u32; 15] = [0; 15];
        let mut cpsr: u32 = 0;
        let mut pc_out: u32 = 0;
        let mut cpu_ctx: u8 = 0;
        let took = unsafe {
            func(
                gpr.as_mut_ptr(),
                &mut cpsr,
                &mut pc_out,
                &mut cpu_ctx as *mut u8,
            )
        };
        assert_eq!(took, 0, "MOV imm shouldn't set took bits");
        assert_eq!(gpr[5], 42, "gpr[5] should hold the imm8");
        // Other gprs untouched.
        for i in 0..15 {
            if i == 5 {
                continue;
            }
            assert_eq!(gpr[i], 0, "gpr[{}] should be unchanged", i);
        }
    }
}
