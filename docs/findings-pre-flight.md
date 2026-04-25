# A0: pre-flight ABI sanity — PASSED

Date: 2026-04-25
Branch: aot-apr25
Commit: pending (will land with arm7tdmi-aot scaffold)

## What

Spec from `docs/aot-llvm-program.md` deliverable A0: compile a
constant-returning fn through inkwell, JIT-execute it, assert it
returns 42. Ports `compile_constant_42` from the JIT branch
(shape-opt/apr22 `arm7tdmi/src/dynarec.rs:472`).

## Result

```
running 1 test
test compiler::tests::a0_jit_executes_constant_function ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
```

The toolchain is wired correctly:
- `LLVM_SYS_181_PREFIX=/usr/lib/llvm-18` resolves.
- `inkwell = "0.9"` with `features = ["llvm18-1-prefer-dynamic"]`
  links against `libLLVM-18.so.1` instead of static-linking, sidesteps
  the missing `libPolly.a` on Ubuntu 24.04.
- Native target initialization, JIT execution engine creation, module
  add, function lookup, JitFunction call — all fine.

## Notes / lessons re-confirmed

- `LLVM_SYS_181_PREFIX` is mandatory; without it `llvm-sys` falls
  back to whatever `llvm-config` is in PATH, which on this host is
  not LLVM 18.
- The `bin/llvm-config-18` binary doesn't exist at
  `/usr/lib/llvm-18/bin/`; only `bin/llvm-config` does. `llvm-sys`
  appears to find it via the prefix lookup. Our `build.rs` accepts
  both names.
- `arm7tdmi-aot/build.rs` enforces presence with a clear error
  message instead of letting `llvm-sys` fail with the cryptic
  "could not find native static library X". Per I20 graceful
  degradation.
- The leaked `&'static Context` pattern works as in the JIT branch.
  No drop-related crashes.

## Phase 0 status

A0 done. Moving on to A1 (handler-pipeline-read audit).
