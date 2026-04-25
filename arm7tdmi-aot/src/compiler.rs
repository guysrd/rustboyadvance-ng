//! LLVM JIT compiler scaffold. Owns the leaked `Context` and the
//! `ExecutionEngine` for the lifetime of the AOT pass. Modules built
//! per-block (or per-page in parallel mode) are added here.

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
#[cfg(test)]
use inkwell::execution_engine::JitFunction;

/// Compiler handle. The `Context` is leaked to `'static` so the
/// `ExecutionEngine` and modules can borrow it for the process
/// lifetime — same idiom the JIT branch used.
pub struct LlvmCompiler {
    context: &'static Context,
    engine: ExecutionEngine<'static>,
    next_id: u64,
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
        })
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
}
