# A6: differential test infra port

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A6: port the
per-instruction differential test framework from the deleted
`dynarec_pattern_differential.rs`.

The framework runs the SAME opcode sequence through both:
- The scalar handler (source of truth).
- The AOT-emitted IR.

…and asserts byte-equality across `gpr`, `cpsr`, `pc`, and emulated
cycles. Phase 1+ commits add per-format tests (one per emit fn).

## Why we need a new bus, not SimpleMemory

`SimpleMemory` (in `arm7tdmi/src/simple_memory.rs`) has:
- `load_*` / `store_*` that DON'T increment any cycle counter.
- `idle_cycle` is a no-op.

The AOT path's whole point is precise cycle accounting (`add_cycles`
inlined per memory op). With SimpleMemory, AOT-emitted cycle counts
would have nothing to compare against — both paths would
report 0 cycles, which is not a useful diff.

**We need a `DiffBus`** that:
- Backs memory in a `Vec<u8>`.
- Maintains a `cycles: u64` counter.
- Increments cycles on every `load_*` / `store_*` by a fixed cost
  per access width (16-bit = 1 cycle, 32-bit = 2 cycles for the
  test scenario — the actual cost matrix isn't load-bearing for
  the diff because both sides see the same).
- Returns `false` from `cached_block_should_abort` and
  `take_block_cache_dirty` (no scheduler events in the test world).

## Implementation: arm7tdmi-aot/src/diff.rs

```rust
use arm7tdmi::memory::{Addr, BusIO, MemoryAccess, MemoryInterface};

pub struct DiffBus {
    data: Box<[u8]>,
    pub cycles: u64,
}

impl DiffBus {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0; capacity].into_boxed_slice(), cycles: 0 }
    }
    pub fn load_program(&mut self, program: &[u8]) {
        self.data[..program.len()].copy_from_slice(program);
    }
}

impl MemoryInterface for DiffBus {
    fn load_8(&mut self, addr: u32, _: MemoryAccess) -> u8 {
        self.cycles += 1;
        self.read_8(addr)
    }
    fn load_16(&mut self, addr: u32, _: MemoryAccess) -> u16 {
        self.cycles += 1;
        self.read_16(addr & !1)
    }
    fn load_32(&mut self, addr: u32, _: MemoryAccess) -> u32 {
        self.cycles += 2;
        self.read_32(addr & !3)
    }
    fn store_8(&mut self, addr: u32, val: u8, _: MemoryAccess) {
        self.cycles += 1;
        self.write_8(addr, val);
    }
    fn store_16(&mut self, addr: u32, val: u16, _: MemoryAccess) {
        self.cycles += 1;
        self.write_16(addr & !1, val);
    }
    fn store_32(&mut self, addr: u32, val: u32, _: MemoryAccess) {
        self.cycles += 2;
        self.write_32(addr & !3, val);
    }
    fn idle_cycle(&mut self) { self.cycles += 1; }

    // No scheduler events in the test world.
    fn cached_block_should_abort(&self) -> bool { false }
    fn take_block_cache_dirty(&mut self) -> bool { false }
    // ... other MemoryInterface methods stubbed similarly ...
}

impl BusIO for DiffBus {
    fn read_8(&mut self, addr: Addr) -> u8 { ... }
    fn write_8(&mut self, addr: Addr, value: u8) { ... }
    // ... read_16/32 + write_16/32 from data buffer ...
}
```

## Test framework shape

```rust
pub fn diff_thumb(opcode: u16, initial_gpr: [u32; 15], initial_cpsr: u32) {
    let scalar = run_scalar_thumb(opcode, initial_gpr, initial_cpsr);
    let aot = run_aot_thumb(opcode, initial_gpr, initial_cpsr);
    assert_eq!(scalar.gpr, aot.gpr,    "gpr mismatch on {:#06x}", opcode);
    assert_eq!(scalar.cpsr, aot.cpsr,  "cpsr mismatch on {:#06x}", opcode);
    assert_eq!(scalar.pc, aot.pc,      "pc mismatch on {:#06x}", opcode);
    assert_eq!(scalar.cycles, aot.cycles, "cycles mismatch on {:#06x}", opcode);
}

struct CpuSnapshot {
    gpr: [u32; 15],
    cpsr: u32,
    pc: u32,
    cycles: u64,
}

fn run_scalar_thumb(op: u16, gpr: [u32; 15], cpsr: u32) -> CpuSnapshot {
    let bus = Shared::new(DiffBus::new(0x1000));
    let mut cpu = Arm7tdmiCore::new(bus);
    cpu.gpr = gpr;
    // Set cpsr.
    // Drive the THUMB_LUT handler directly.
    let info = &Arm7tdmiCore::<DiffBus>::THUMB_LUT[(op as usize >> 6) & 0x3FF];
    (info.handler_fn)(&mut cpu, op);
    CpuSnapshot { gpr: cpu.gpr, cpsr: cpu.cpsr.get(), pc: cpu.pc,
                   cycles: cpu.bus.cycles }
}

fn run_aot_thumb(op: u16, gpr: [u32; 15], cpsr: u32) -> CpuSnapshot {
    // For phase 0, AOT side is a placeholder that just runs scalar
    // (so the test trivially passes). Phase 1+ replaces this with
    // actual AOT-emitted IR for the format being tested.
    run_scalar_thumb(op, gpr, cpsr)
}
```

## Per-format test invocation pattern (phase 1+)

```rust
#[test]
fn aot_diff_thumb_f3_mov_imm8_basic() {
    diff_thumb(0x2005, [0; 15], 0);   // MOV r0, #5
}

#[test]
fn aot_diff_thumb_f3_mov_imm8_zero() {
    diff_thumb(0x2000, [0; 15], 0);   // MOV r0, #0 — Z flag should set
}

#[test]
fn aot_diff_thumb_f3_mov_imm8_clobber_n() {
    let mut gpr = [0; 15];
    gpr[0] = 0xDEADBEEF;
    let mut cpsr = 0x80000000;        // N flag preset
    diff_thumb(0x2042, gpr, cpsr);    // MOV r0, #66 → should clear N
}
```

Coverage targets (spec, not all required at phase 1):
- Every AOT-supported format.
- Per format: zero-result, sign-bit-result, carry-out, overflow,
  register-clash (rd == rn), misaligned-load (where applicable).

## Phase 0 deliverable

`arm7tdmi-aot/src/diff.rs` lands with the phase-0 scaffold commit
containing:
- `DiffBus` impl.
- `CpuSnapshot` struct.
- `run_scalar_thumb` and a placeholder `run_aot_thumb` (calls scalar
  for now).
- `diff_thumb` driver.
- Smoke test confirming the framework runs end-to-end (trivially
  passes since aot-side calls scalar).

Phase 1's first commit replaces the placeholder with real AOT-side
emit-and-execute for whatever format phase 1 inlines, and adds the
real-coverage tests for that format.

## Phase 0 status

A6 done. On to A7 (instruction-format distribution profile).
