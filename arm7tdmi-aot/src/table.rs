//! Two-level PC->fn lookup table (per I18 — inlined dispatcher
//! lookup).
//!
//! Top-level: 65536-entry boxed slice indexed by `pc >> 16`. Each
//! entry is `Option<Box<Leaf>>` — leaves allocated lazily as the
//! AOT scan discovers blocks in their 64KB address page. The
//! 65536-entry top covers the full 32-bit GBA address space; using
//! only 8 bits of the top would collide BIOS (0x00000000) with ROM
//! (0x08000000) since both have bits 16-23 = 0.
//!
//! Leaf: 32768-entry dense array indexed by `(pc >> 1) & 0x7fff`.
//! Slot type is `Option<CompiledFn>`. Covers bits 1-15 of pc within
//! a 64KB page, Thumb-aligned (the index encodes Thumb pc & ARM pc
//! disambiguation via PC alignment).
//!
//! Memory profile: top is 65536 × 8 bytes = 512KB upfront. Each
//! allocated leaf is 32768 × 8 = 256KB. Pokeemerald (16MB ROM ≈
//! 256 ROM pages) uses ~256 leaves = 64MB. Acceptable.
//!
//! Lookup is `#[inline(always)]` and does:
//!   1. pc >> 16            → top index (16-bit)
//!   2. null-check the leaf box
//!   3. (pc >> 1) & 0x7fff  → leaf index
//!   4. null-check the slot
//!   5. return Option<CompiledFn>

/// AOT-compiled block fn signature (per I8). Single-pointer ABI to
/// avoid noalias issues between gpr_ptr and cpu_ctx.
///
/// Args:
///   `cpu_ctx`  — `&mut Arm7tdmiCore<I>` cast to `*mut u8`.
///   `pc_out`   — output u32 written when return = 0b01.
///
/// Return:
///   0     → fall-through to next block.
///   0b01  → branch fired; target in `*pc_out` with Thumb bit in bit 0.
///   0b10  → mid-block abort; dispatcher yields.
pub type CompiledFn = unsafe extern "C" fn(cpu_ctx: *mut u8, pc_out: *mut u32) -> u32;

const TOP_LEVEL_SIZE: usize = 65536;
const LEAF_SIZE: usize = 32768;

type Leaf = [Option<CompiledFn>; LEAF_SIZE];

pub struct AotTable {
    /// 65536 leaves, lazily allocated. Heap-boxed because the array
    /// itself is 512KB — too big to keep on the stack. `Box<Leaf>`
    /// keeps the leaf allocation stable so we can hand out raw
    /// pointers via the I18 inlined lookup.
    pages: Box<[Option<Box<Leaf>>]>,
    /// Phase-8 ARM lookup table. Same shape as `pages` but separate
    /// to disambiguate ARM-mode blocks from Thumb-mode blocks at the
    /// same pc. Dispatcher selects the right table based on
    /// cpu.cpsr.state().
    arm_pages: Box<[Option<Box<Leaf>>]>,
    /// Arena of per-block opcode buffers. Each phase-0 placeholder
    /// block has a stable raw ptr into this arena baked into its
    /// LLVM IR (the trampoline call argument). Box<[u32]> keeps
    /// each buffer's address fixed for the AotTable's lifetime.
    /// Drop ordering: AotTable owns this; when AotTable drops, the
    /// associated ExecutionEngine should already have been dropped
    /// (if any) so no in-flight calls reference these buffers.
    pub(crate) thumb_opcode_arena: Vec<Box<[u32]>>,
    /// Phase-8 ARM-block opcode arena. Separate from thumb arena so
    /// ARM blocks don't accidentally share buffers (different
    /// instruction widths anyway).
    pub(crate) arm_opcode_arena: Vec<Box<[u32]>>,
    /// Compiled-block count, maintained as inserts happen so we
    /// don't have to walk all 65536 leaves to report it.
    pub(crate) compiled_count: usize,
    /// Phase-8 ARM-compiled-block count.
    pub(crate) arm_compiled_count: usize,
    /// Phase-4P-B: WAITCNT generation counter pointer. Set by SDL
    /// frontend (or other host) when enabling baked-cycle paths. Lookup
    /// reads this pointer at every dispatch and short-circuits to None
    /// (auto-disable AOT) when the live counter doesn't match
    /// `aot_gen_baked`. `null` = no gen check (legacy / disabled).
    pub(crate) aot_gen_counter_ptr: *const u32,
    /// Phase-4P-B: counter value captured at AOT compile time.
    pub(crate) aot_gen_baked: u32,
}

// SAFETY: AotTable is owned by the bus side per I12; the
// `aot_gen_counter_ptr` points into the SAME bus's SysBus struct so
// they share lifetime. No threads cross.
unsafe impl Send for AotTable {}
unsafe impl Sync for AotTable {}

impl AotTable {
    pub fn new() -> Self {
        // Vec → Boxed slice of length 65536, all None.
        let v: Vec<Option<Box<Leaf>>> = (0..TOP_LEVEL_SIZE).map(|_| None).collect();
        let v_arm: Vec<Option<Box<Leaf>>> = (0..TOP_LEVEL_SIZE).map(|_| None).collect();
        Self {
            pages: v.into_boxed_slice(),
            arm_pages: v_arm.into_boxed_slice(),
            thumb_opcode_arena: Vec::new(),
            arm_opcode_arena: Vec::new(),
            compiled_count: 0,
            arm_compiled_count: 0,
            aot_gen_counter_ptr: std::ptr::null(),
            aot_gen_baked: 0,
        }
    }

    /// Phase-4P-B: install gen-check pointer + baked value. Caller is
    /// the SDL frontend (or other host) — pointer must outlive the
    /// table; baked is captured at AOT compile time. After this call,
    /// `aot_lookup` will return None whenever the live counter doesn't
    /// match — auto-disabling AOT for stale-WAITCNT blocks.
    pub fn set_gen_check(&mut self, gen_ptr: *const u32, baked: u32) {
        self.aot_gen_counter_ptr = gen_ptr;
        self.aot_gen_baked = baked;
    }

    /// Insert a compiled fn for entry-PC `pc`. Thumb / ARM mode
    /// disambiguation is via PC alignment (Thumb pc bit 0 = 0 after
    /// strip; ARM pc bit 1 = 0). The index hashes via `pc >> 1` so
    /// adjacent Thumb pcs land in adjacent slots.
    pub fn insert(&mut self, pc: u32, f: CompiledFn) {
        let top = (pc >> 16) as usize;
        let leaf_idx = ((pc >> 1) & 0x7fff) as usize;
        let leaf = self.pages[top].get_or_insert_with(|| {
            // 32768 None slots.
            Box::new(std::array::from_fn(|_| None))
        });
        if leaf[leaf_idx].is_none() {
            self.compiled_count += 1;
        }
        leaf[leaf_idx] = Some(f);
    }

    /// Add an opcode buffer to the arena and return a stable raw
    /// pointer. Caller bakes the pointer into the LLVM IR for a
    /// per-block trampoline call.
    pub fn intern_opcodes(&mut self, opcodes: &[u32]) -> *const u32 {
        let boxed: Box<[u32]> = opcodes.to_vec().into_boxed_slice();
        let ptr = boxed.as_ptr();
        self.thumb_opcode_arena.push(boxed);
        ptr
    }

    /// Phase-8: insert an ARM CompiledFn at lookup_pc. Separate from
    /// `insert` (Thumb) so ARM blocks don't collide with Thumb blocks
    /// at the same pc.
    pub fn insert_arm(&mut self, pc: u32, f: CompiledFn) {
        let top = (pc >> 16) as usize;
        let leaf_idx = ((pc >> 1) & 0x7fff) as usize;
        let leaf = self.arm_pages[top].get_or_insert_with(|| {
            Box::new(std::array::from_fn(|_| None))
        });
        if leaf[leaf_idx].is_none() {
            self.arm_compiled_count += 1;
        }
        leaf[leaf_idx] = Some(f);
    }

    /// Phase-8: ARM opcode buffer arena. Separate from thumb arena.
    pub fn intern_arm_opcodes(&mut self, opcodes: &[u32]) -> *const u32 {
        let boxed: Box<[u32]> = opcodes.to_vec().into_boxed_slice();
        let ptr = boxed.as_ptr();
        self.arm_opcode_arena.push(boxed);
        ptr
    }

    pub fn arm_block_count(&self) -> usize {
        self.arm_compiled_count
    }

    /// Diagnostic: total compiled-block count. Maintained as
    /// inserts happen so this is O(1).
    pub fn block_count(&self) -> usize {
        self.compiled_count
    }

    /// Diagnostic: number of allocated pages (sparsity indicator).
    pub fn page_count(&self) -> usize {
        self.pages.iter().filter(|p| p.is_some()).count()
    }
}

impl Default for AotTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Inlined lookup hot path (per I18). Called from the dispatcher's
/// inner loop. Don't add nontrivial logic here — every cycle counts.
#[inline(always)]
pub fn aot_lookup(table: &AotTable, pc: u32) -> Option<CompiledFn> {
    // Phase-4P-B: optional WAITCNT gen check. If a counter pointer is
    // installed and the live value diverged from the baked value, the
    // entire AOT table reads as empty — dispatcher falls through to
    // scalar (correct, just slower). Pointer-null = no check (legacy).
    if !table.aot_gen_counter_ptr.is_null() {
        let live = unsafe { *table.aot_gen_counter_ptr };
        if live != table.aot_gen_baked {
            return None;
        }
    }
    let top = (pc >> 16) as usize;
    // SAFETY: top fits in 16 bits, pages.len() == 65536, so the index
    // is always in bounds. Use get_unchecked to drop the bounds check
    // from the hot path.
    let page = unsafe { table.pages.get_unchecked(top) };
    let leaf = page.as_ref()?;
    let leaf_idx = ((pc >> 1) & 0x7fff) as usize;
    leaf[leaf_idx]
}

/// Phase-8: ARM-mode lookup. Same shape as `aot_lookup` but indexes
/// the parallel `arm_pages` table to disambiguate ARM blocks from
/// Thumb blocks at the same pc.
#[inline(always)]
pub fn aot_lookup_arm(table: &AotTable, pc: u32) -> Option<CompiledFn> {
    if !table.aot_gen_counter_ptr.is_null() {
        let live = unsafe { *table.aot_gen_counter_ptr };
        if live != table.aot_gen_baked {
            return None;
        }
    }
    let top = (pc >> 16) as usize;
    let page = unsafe { table.arm_pages.get_unchecked(top) };
    let leaf = page.as_ref()?;
    let leaf_idx = ((pc >> 1) & 0x7fff) as usize;
    leaf[leaf_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Stub fn for table tests — never actually called.
    unsafe extern "C" fn stub(_ctx: *mut u8, _pc_out: *mut u32) -> u32 {
        0
    }

    #[test]
    fn empty_table_lookup_misses() {
        let table = AotTable::new();
        assert!(aot_lookup(&table, 0x0800_0000).is_none());
        assert!(aot_lookup(&table, 0x0).is_none());
        assert!(aot_lookup(&table, 0xFFFF_FFFE).is_none());
    }

    #[test]
    fn insert_and_lookup() {
        let mut table = AotTable::new();
        table.insert(0x0800_1000, stub);
        assert!(aot_lookup(&table, 0x0800_1000).is_some());
        // miss on neighboring PC
        assert!(aot_lookup(&table, 0x0800_1002).is_none());
        assert!(aot_lookup(&table, 0x0800_0FFE).is_none());
    }

    #[test]
    fn separate_pages() {
        let mut table = AotTable::new();
        table.insert(0x0000_0000, stub);   // BIOS page
        table.insert(0x0800_0000, stub);   // ROM page
        assert!(aot_lookup(&table, 0x0000_0000).is_some());
        assert!(aot_lookup(&table, 0x0800_0000).is_some());
        assert_eq!(table.page_count(), 2);
        assert_eq!(table.block_count(), 2);
    }

    #[test]
    fn many_inserts_in_one_page() {
        let mut table = AotTable::new();
        for i in 0..100 {
            table.insert(0x0800_0000 + i * 2, stub);
        }
        for i in 0..100 {
            assert!(aot_lookup(&table, 0x0800_0000 + i * 2).is_some());
        }
        assert_eq!(table.page_count(), 1);
        assert_eq!(table.block_count(), 100);
    }
}
