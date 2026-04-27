// Build-time guard: enforce LLVM 18 presence with a clean error.
// llvm-sys's own error is opaque ("could not find native static library
// X") so we pre-check and emit a useful message.

fn main() {
    let prefix = match std::env::var("LLVM_SYS_181_PREFIX") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "ERROR: arm7tdmi-aot requires LLVM 18.\n  set LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 (Ubuntu 24.04 path)\n  or build the workspace without arm7tdmi-aot if you dont have it"
            );
            std::process::exit(1);
        }
    };
    let prefix_path = std::path::Path::new(&prefix);
    // /usr/lib/llvm-18/bin/llvm-config (no -18 suffix on the symlink target)
    let config = prefix_path.join("bin").join("llvm-config");
    let config_18 = prefix_path.join("bin").join("llvm-config-18");
    if !config.exists() && !config_18.exists() {
        eprintln!(
            "ERROR: LLVM 18 not found at {}.\n  expected one of:\n    {}\n    {}\n  install llvm-18 (apt install llvm-18-dev) or fix LLVM_SYS_181_PREFIX",
            prefix,
            config.display(),
            config_18.display()
        );
        std::process::exit(1);
    }
    println!("cargo:rerun-if-env-changed=LLVM_SYS_181_PREFIX");
}
