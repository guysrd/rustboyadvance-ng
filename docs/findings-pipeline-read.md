# A1: handler-pipeline-read audit

Date: 2026-04-25
Branch: aot-apr25

## What

Spec from `docs/aot-llvm-program.md` deliverable A1: grep every Thumb
and ARM handler for `cpu.pipeline[`. Decide per handler whether the
inline-fetch optimization (skip `read_16`, keep `add_cycles`) is safe.

The optimization rests on: if no handler reads `pipeline[0]` or
`pipeline[1]` for execution data, AOT can elide the actual memory
load and only inline `add_cycles` for the scheduler advance.

## Method

```
grep -rn "self\.pipeline\|cpu\.pipeline" arm7tdmi/src/
grep -n "pipeline" arm7tdmi/src/arm/mod.rs arm7tdmi/src/arm/exec.rs arm7tdmi/src/thumb/exec.rs
```

## Result

**No Thumb handler reads `pipeline[]`. No ARM handler reads
`pipeline[]`.** Handlers only call `reload_pipeline16()` /
`reload_pipeline32()` after branches and mode-flips (BX / BL / B /
SWI / mode change). Those calls write pipeline, not read.

Inventory of pipeline mentions inside handlers:

| File | Line | Context | Reads pipeline data? |
|------|------|---------|----------------------|
| `arm/exec.rs:33` | reset path → reload_pipeline32 | write only |
| `arm/exec.rs:42-47` | BX → reload_pipeline16/32 | write only |
| `arm/exec.rs:264-265` | mode flip in MSR → reload_pipeline | write only |
| `arm/exec.rs:337` | B/BL → reload_pipeline32 | write only |
| `arm/exec.rs:442` | LDM with R15 in rlist → reload | write only |
| `arm/exec.rs:551` | LDR PC → reload_pipeline32 | write only |
| `arm/exec.rs:600` | data-proc PC dest → reload_pipeline32 | write only |
| `thumb/exec.rs:181,194` | F5 high-reg PC ops → reload_pipeline16 | write only |
| `thumb/exec.rs:433` | F12 ADD PC → reload_pipeline16 | write only |
| `thumb/exec.rs:504` | F18 B → reload_pipeline16 | write only |
| `thumb/exec.rs:529` | F19 BL pair → reload_pipeline16 | write only |
| `thumb/exec.rs:537` | SWI → exception (implies reload) | write only |
| `thumb/exec.rs:546` | F16 Bcc taken → reload_pipeline16 | write only |
| `thumb/exec.rs:562` | F5 BX → reload_pipeline16 | write only |

All other `pipeline[]` references are in `cpu.rs` only, in:
- `reload_pipeline16/32` (write).
- `replay_cached_block` (read pipeline[0], shift) — this is the
  **dispatcher**, NOT a handler. AOT replaces this dispatcher path.
- `single_step` (similar dispatcher pattern).
- `record_new_block` (similar).
- `from_saved_state` / `Clone::clone` / `restore_state` (state
  copy).

## Implication for AOT

**Inline-fetch optimization is safe for all formats.** Every Thumb
format and every ARM class can use:

```
; instead of: cpu.bus.load_16(fetch_addr, access)
add_cycles_inline(scheduler, fetch_addr, access, 16)
; pipeline state is irrelevant since no handler reads it
```

Cycle accounting still must reflect the fetch (scheduler advance
must happen) — only the actual memory read is elided.

The pipeline shift (pipeline[0] = pipeline[1], pipeline[1] = fetched)
is also unnecessary in steady-state AOT — but we still need to keep
pipeline VALID so that scalar can take over on AOT-miss with the
right state. The `replay_cached_block` scalar path reads pipeline[0]
on each iter to get `insn`. So:

- **In the AOT block**: don't update pipeline (handlers don't read,
  scalar isn't running here).
- **At AOT block exit (returning to dispatcher)**: ensure pipeline
  is in a state consistent with what scalar would expect at the
  new PC. The simplest invariant: at any AOT exit, scalar's
  `reload_pipeline*` is called once before resuming. This forces
  a fresh fetch for the next 2 instrs and re-establishes the
  scalar-dispatcher pipeline invariant. Trade: 2 extra fetches
  per AOT-to-scalar transition. Cheap (rare event in steady
  state — only on abort / branch / mode-flip exits).

This pipeline-bookkeeping detail extends I6: the AOT block does NOT
keep `cpu.pipeline[]` consistent with execution; on exit, the
dispatcher reloads pipeline before scalar takes over.

## Phase 0 status

A1 done. On to A2 (IO-region map).
