//! Dump Cranelift-generated asm for a realistic compiled Thumb block.
//! Lets us inspect the codegen shape and see where cycles go.
use arm7tdmi::dynarec::dump::dump_thumb_unified_block;

fn main() {
    // Realistic 4-instr Thumb block from pokeemerald: MOV/ADD/LDR/ADD.
    // Equivalent of a hot getter: load field, add constant, store.
    // 0x2005 = MOV R0, #5 (F3)
    // 0x1c40 = ADD R0, R0, #1 (F2)
    // 0x6800 = LDR R0, [R0] (F9)
    // 0x1832 = ADD R2, R6, R0 (F2)
    let opcodes = [0x2005u16, 0x1c40, 0x6800, 0x1832];
    println!("=== 4-instr fall-through block ===");
    match dump_thumb_unified_block(&opcodes, 0x0800_0000) {
        Ok(s) => print!("{s}"),
        Err(e) => eprintln!("err: {e}"),
    }
}
