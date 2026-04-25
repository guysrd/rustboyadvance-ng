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

/// Function signature shared by every compiled block. Mirrors the Cranelift
/// shape (`*mut u32 gpr, *mut u32 cpsr, *mut u32 pc_out, *mut u8 cpu_ctx -> u32`).
/// Return value is the same `took` bit-encoding the dispatcher already
/// understands:
///   bit 0 = branch fired (pc_out populated, dispatcher reloads pipeline).
///   bit 1 = mid-block abort (state already saved, dispatcher yields).
pub type CompiledFn =
    unsafe extern "C" fn(*mut u32, *mut u32, *mut u32, *mut u8) -> u32;

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
        })
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
}
