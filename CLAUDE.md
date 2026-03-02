# VOLK — Claude Context

## What is VOLK?

VOLK (Vector-Optimized Library of Kernels) is a sub-project of GNU Radio. It provides
hand-written SIMD implementations of common DSP math operations. At runtime, VOLK's
dispatcher selects the best available proto-kernel for the current CPU.

- Kernels live in `kernels/volk/volk_<type_sig>_<operation>.h`
- Code generation scripts live in `gen/` (Python, reads `archs.xml` and `machines.xml`)
- ISA feature flags (`LV_HAVE_AVX2`, etc.) are defined by the build system from `gen/archs.xml`

## Kernel file structure

Each kernel header contains **two include-guarded sections**. The sections represent
a contract with the caller about pointer alignment:

- **`_u_` (unaligned)**: Pointers **may be unaligned**. Implementations must tolerate
  arbitrary alignment. This section comes **first** in the file.
- **`_a_` (aligned)**: The caller **guarantees** pointers are aligned (16 B for SSE,
  32 B for AVX, 64 B for AVX-512). Implementations may use aligned loads/stores.

```
#ifndef INCLUDED_volk_<name>_u_H   ← unaligned section (first)
#define INCLUDED_volk_<name>_u_H
...
#endif

#ifndef INCLUDED_volk_<name>_a_H   ← aligned section (second)
#define INCLUDED_volk_<name>_a_H
...
#endif
```

The `_a_` / `_u_` distinction only applies to **x86 SSE/AVX/AVX-512**, where aligned
loads (`_mm_load_ps`, etc.) fault on misaligned addresses. ARM NEON, RISC-V RVV, ORC,
and scalar C handle any alignment natively, so they need only one implementation and
belong in the **unaligned section** (available to all callers regardless of alignment).

### Function naming convention

| Suffix | Meaning |
|---|---|
| `_a_avx2` | aligned-pointer variant, AVX2 ISA |
| `_u_avx2` | unaligned-pointer variant, AVX2 ISA |
| `_generic` | scalar fallback (unaligned section, no ISA prefix) |
| `_neon` | ARM NEON (unaligned section; no `_a_`/`_u_` prefix) |
| `_rvv` | RISC-V Vector (unaligned section; no `_a_`/`_u_` prefix) |

### ISA ordering convention

Implementations are listed in order of increasing complexity within each section.
This is a **readability convention** — the dispatcher selects by ISA bitmask value
(from `archs.xml`), not by file position.

- **Unaligned section (`_u_`)**: `GENERIC`, `SSE`, `SSE2`, `SSE3`, `SSE4_1`, `AVX`,
  `AVX2`, `AVX512F`, `AVX512BW`, `AVX512VBMI`, `NEON`, `NEONV8`, `RVV`, `RVVSEG`, `ORC`
- **Aligned section (`_a_`)**: `SSE`, `SSE2`, `SSE3`, `SSE4_1`, `AVX`, `AVX2`,
  `AVX512F`, `AVX512BW`, `AVX512VBMI`

### ISA include headers

| ISA guard | Header |
|---|---|
| `LV_HAVE_SSE` | `<xmmintrin.h>` |
| `LV_HAVE_SSE2` | `<emmintrin.h>` |
| `LV_HAVE_SSE3` | `<pmmintrin.h>` (implicitly includes emmintrin.h) |
| `LV_HAVE_SSE4_1` | `<smmintrin.h>` |
| `LV_HAVE_AVX2`, `LV_HAVE_AVX512F`, `LV_HAVE_AVX512BW`, `LV_HAVE_AVX512VBMI` | `<immintrin.h>` |
| `LV_HAVE_NEON`, `LV_HAVE_NEONV8` | `<arm_neon.h>` |
| `LV_HAVE_RVV`, `LV_HAVE_RVVSEG` | `<riscv_vector.h>` |

### machines.xml ISA hierarchy (x86)

From `gen/machines.xml`, the main x86 machine chain is:
```
generic → sse → sse2 → sse3 → ssse3 → sse4_1 → sse4_2 (+popcount) → avx → fma → avx2
                                                                      → avx512f → avx512bw → avx512dq
                                                                                → avx512cd
                                                                      → avx512f → avx512bw (+avx512vl, +f16c)
                                                                                → avx512vbmi → avx512vnni
                                                                                             → avx512vbmi2
                          → sse4_a (+popcount)  [AMD branch]
```
`popcount` is enabled from `sse4_2` onward. `fma` first appears at `avx2`.
`f16c` first appears at `avx512vl` (and all machines above it).
`LV_HAVE_*` enum values are assigned in `archs.xml` order — the dispatcher picks
the implementation whose dependency bitmask is numerically highest.

### Documentation block

Each kernel file starts with a Doxygen block. The `\b Example` section must include:
allocation with `volk_malloc(..., volk_get_alignment())`, data initialization, the kernel
call with all correct arguments, and `volk_free()` for each buffer. See
[volk_16ic_deinterleave_16i_x2.h](kernels/volk/volk_16ic_deinterleave_16i_x2.h)
lines 33–47 as the canonical example.

---

## Coding rules

### Section and alignment correctness

- `_a_` variants must use aligned load/store intrinsics (e.g. `_mm_load_ps`,
  `_mm256_load_si256`, `_mm512_store_ps`). `_u_` variants must use unaligned
  variants (e.g. `_mm_loadu_ps`, `_mm256_loadu_si256`, `_mm512_storeu_ps`).
  AVX-512 aligned requires **64-byte** alignment.
- Every `_a_` variant needs a `_u_` twin differing only by `loadu`/`storeu`.
- The function name prefix (`_a_` or `_u_`) and load/store alignment must match the
  section it appears in. The type suffix (`_ps`, `_si128`, etc.) depends on the
  kernel's data type — use `_ps`/`_pd` for float/double kernels and `_si128`/`_si256`/
  `_si512` for integer kernels.

### ISA guard correctness

- `LV_HAVE_FMA` is a **separate flag** from `LV_HAVE_AVX2` — do not use
  `_mm256_fmadd_ps` inside a bare `LV_HAVE_AVX2` block without also checking
  `LV_HAVE_FMA`.
- AVX-512F vs AVX-512BW vs AVX-512VBMI: widening conversions
  (`_mm512_cvtepi16_epi32`) are **AVX-512F**; operations *on* 16-bit elements within
  512-bit registers (`_mm512_permutexvar_epi16`, `_mm512_srai_epi16`,
  `_mm512_add_epi16`) require **AVX-512BW**; byte-granularity cross-lane permutations
  (`_mm512_permutexvar_epi8`, `_mm512_permutex2var_epi8`) require **AVX-512VBMI**.
- AVX-512VBMI is only useful for **byte-level permutation** kernels (deinterleave,
  channel extraction, format conversion involving sub-32-bit elements). Kernels that
  operate on float/int32 elements (dot products, magnitudes, multiply-accumulate)
  gain nothing from VBMI — their critical paths are arithmetic, not data rearrangement.
- `#endif` comments must match their `#ifdef` exactly — including the full kernel
  name and direction (e.g. `16ic_convert_32fc` vs `32fc_convert_16ic`).

### Code correctness

- Cast `const T*` input pointers as `(const U*)`, not `(U*)`, when reinterpreting.
  Tail-loop derived pointers must also preserve `const`.
- Use `-1` not `0xFFFFFFFF` in `_mm*_set_epi32` — `0xFFFFFFFF` exceeds `INT_MAX`
  and its conversion to `int` is implementation-defined.
- In RVV kernels, write `size_t n = (size_t)num_points * 2` to avoid `unsigned int`
  overflow on 32-bit targets.
- Doxygen `\b Example` blocks must include allocations, correct argument order,
  data initialization, and `volk_free()` for each buffer.
- Use load + convert intrinsics (`_mm_load_si128` → `_mm_cvtepi16_epi32` →
  `_mm_cvtepi32_ps`), not `_mm_set_ps` from individually cast scalars (compiles to
  scalar `cvtsi2ss`/`insertps` sequences).
- Use `1.0f / scalar`, not `1.0 / scalar`, when computing float reciprocals — the
  double literal silently promotes `scalar` to `double`, performs a double-precision
  division, then narrows back to `float`. The Scaling Convention section already
  prescribes `1.0f`; treat `1.0` here as a bug.
- NEONV7 rsqrt zero guard: `_vinvsqrtq_f32(0.0f)` returns `+∞`, so
  `0 * ∞ = NaN` when computing magnitude as `mag_sq * rsqrt(mag_sq)`. Guard
  zero-magnitude samples with `vcgtq_f32` + `vbslq_f32`:
  ```c
  const uint32x4_t nonzero = vcgtq_f32(mag_sq, vdupq_n_f32(0.0f));
  const float32x4_t mag = vmulq_f32(mag_sq, _vinvsqrtq_f32(mag_sq));
  result = vbslq_f32(nonzero, mag, vdupq_n_f32(0.0f));
  ```
  `vbslq_f32` selects `mag` where `nonzero` is true, `0.0f` elsewhere. This pattern
  is not needed under `LV_HAVE_NEONV8` where `vsqrtq_f32` handles zero correctly.

---

## Performance guidelines

### FMA accumulator count for reduction kernels

FMA has 4-cycle latency and 0.5-cycle throughput on Skylake, so 2 accumulators leave
the pipeline ~4× under-utilised. Use at least **4 independent `dotProdVal` registers**
for AVX/AVX-512 reduction loops. General rule: `min_accumulators ≥ latency / throughput`
(= 8 for Skylake FMA; 4 is a practical minimum that recovers most headroom).

### Loop unrolling and prefetch

For memory-bandwidth-bound kernels, process 2× SIMD widths per loop iteration and
prefetch ahead with `__VOLK_PREFETCH`. Typical distances:

| ISA | Floats/iter (2× unroll) | Prefetch distance |
|---|---|---|
| SSE | 8 | `ptr + 16` (64 B = 1 cache line) |
| AVX | 16 | `ptr + 16` (64 B = 1 cache line) |
| AVX-512 | 32 | `ptr + 32` (128 B = 2 cache lines) |

**Exception — purely sequential stride kernels (e.g. dot products):** Intel's hardware
stream prefetcher detects constant-stride sequential reads automatically within a few
iterations. For kernels that simply walk two or more arrays linearly (no scatter/gather,
no indirect addressing), software prefetch adds execution-unit overhead with no benefit
and can measurably reduce throughput. Omit `__VOLK_PREFETCH` for these kernels.

### Tap duplication (complex × real kernels)

To expand packed real taps `[t0,t1,t2,t3]` into paired form `[t0,t0,t1,t1]` /
`[t2,t2,t3,t3]`, load once and unpack with itself:
```c
__m128 x = _mm_load_ps(bPtr);
b0 = _mm_unpacklo_ps(x, x);  // [t0,t0,t1,t1]
b1 = _mm_unpackhi_ps(x, x);  // [t2,t2,t3,t3]
```
Do **not** load the same address twice into separate registers — it wastes a load-port
op. Also note that `_mm_moveldup_ps` / `_mm_movehdup_ps` give a *different* pattern
(`[t0,t0,t2,t2]` / `[t1,t1,t3,t3]`) and cannot substitute here.

### NEON multiply-accumulate

Under `LV_HAVE_NEON` (ARMv7+), prefer `vmlaq_f32(acc, a, b)` over separate
`vmulq_f32` + `vaddq_f32` — single instruction, fewer micro-ops. `vfmaq_f32` (fused,
avoids double-rounding) is only available under `LV_HAVE_NEONV8`.

### Scaling convention

Always precompute `invScalarF = 1.0f / scalar` and **multiply** — never divide in the
hot path or tail loop. Declare a scalar `float invScalarF` even when a vector
`__m256 invScalar` is also used, so both the main loop and tail loop can share it.

### Loop tail pattern

```c
number = nPoints * stride;
magnitudeVectorPtr = &magnitudeVector[number];   // explicit reset (redundant but clear)
complexVectorPtr   = (const int16_t*)&complexVector[number];
for (; number < num_points; number++) {
    float real = (float)(*complexVectorPtr++) * invScalarF;
    float imag = (float)(*complexVectorPtr++) * invScalarF;
    *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
}
```

### Avoid `_mm_hadd_ps` and `_mm256_hadd_ps`

Both 128-bit `_mm_hadd_ps` (SSE3) and 256-bit `_mm256_hadd_ps` (AVX) have 2-cycle
reciprocal throughput on Intel vs 1 cycle for shuffle+add. Prefer deinterleave +
`_mm*_add_ps`. SSE3 variants that use `_mm_hadd_ps` can measure *slower* than SSE2
because the higher-latency hadd more than offsets any reduction in instruction count —
so `LV_HAVE_SSE3` implementations relying on hadd may be outperformed by `LV_HAVE_SSE2`
and should be removed rather than kept. Also remember that **both** `_mm256_shuffle_ps`
and `_mm256_shuffle_epi8` operate within 128-bit lanes on AVX2 and need a cross-lane
permute to fix the artifact: `_mm256_permutevar8x32_ps` (floats) or
`_mm256_permutevar8x32_epi32` (integers).

---

## SIMD pattern reference

Detailed SIMD patterns for 16ic and complex float kernels (int16→float conversion,
deinterleaving, channel extraction, narrowing) are in
[docs/simd_patterns.md](docs/simd_patterns.md). Read that file when implementing or
modifying kernel SIMD code.

Key patterns covered:
- Int16 → float conversion (SSE2 through AVX-512F)
- Extracting high byte of int16 complex to int8 (SSE2 through AVX-512VBMI)
- Extracting one/both int16 channels from interleaved complex (SSSE3 through AVX-512VBMI)
- Deinterleaving complex float re/im (SSE through AVX-512BW)
- Narrowing float/int32 → int16 output (AVX2, AVX-512F)

---

## Session workflow

Work is organized into three phases, designed to be tackled in order. Each phase
produces uniform, self-contained changes that are easy to review.

### Phase 1 — Bug fixes

Fix correctness issues in existing implementations. These are mechanical and
high-priority:

- Wrong `#endif` comment (doesn't match its `#ifdef`)
- Single `_H` include guard instead of the required `_u_H` / `_a_H` pair
- `_a_` section appearing before `_u_` section (unaligned section must come first)
- `_mm256_loadu_si256` in an `_a_` variant (should be aligned load)
- `_mm_store_ps` in a `_u_` variant (should be unaligned store)
- Missing `const` on casted input pointers
- `0xFFFFFFFF` in `_mm*_set_epi32` (should be `-1`)
- `/ scalar` in tail loops (should be `* invScalarF`)
- Incorrect Doxygen `\b Example` (wrong args, missing allocations/frees)
- `_mm256_fmadd_ps` under bare `LV_HAVE_AVX2` without `LV_HAVE_FMA` guard

### Phase 2 — Missing `_u_` twins

Many kernels have `_a_sse2`, `_a_avx2`, etc. but lack the corresponding `_u_`
variant. For each missing twin:

1. Copy the `_a_` implementation into the `_u_` section
2. Rename `_a_` to `_u_` in the function name
3. Replace aligned loads/stores with unaligned equivalents
4. Update the `#ifdef` / `#endif` guard comments

### Phase 3 — Missing `_a_` twins

Many kernels have `_u_sse2`, `_u_avx2`, etc. but lack the corresponding `_a_`
variant. For each missing twin:

1. Copy the `_u_` implementation into the `_a_` section
2. Rename `_u_` to `_a_` in the function name
3. Replace unaligned loads/stores with aligned equivalents
4. Update the `#ifdef` / `#endif` guard comments

### Phase 4 — New ISA implementations

Add higher-ISA implementations (AVX2, AVX-512F, AVX-512BW, AVX-512VBMI) to kernels
that only have SSE or generic. Refer to the SIMD pattern reference above for common
idioms. AVX-512VBMI is most beneficial for byte-level data rearrangement kernels
(deinterleave, extract) where `_mm512_permutex2var_epi8` replaces multi-step
shuffle+permute pipelines.

### Per-kernel checklist

When working on a kernel, follow this order:

1.  **Build and benchmark** — build in **Release mode** (`-DCMAKE_BUILD_TYPE=Release`),
    then run `volk_profile -T 10 -R <kernel_name>` to record the performance prior to
    making changes.  Please store this output in "log/<kernel_name>-before.txt".
2.  This output should contain two ISAs with improvement factors labeled: "Best
    aligned arch" & "Best unaligned arch".  If the best arch is generic, list the
    improvement factor as "< 1.0" (The output won't contain a factor). Please add a
    row to the file "KernelImprovements.md" containing: <kernel_name>, date, time,
    best a_ISA Before, a_improve Before, best u_ISA before, u_improve Before, and
    a series of columns with annotations for which ISAs the kernel has been implemented.
    If a row in the file already exists for the current <kernel_name>, please update
    the entries. If this kernel has ISAs that other kernels in the file don't have,
    please create blank entries for those rows in the new ISA column.
3.  **Identify** all existing issues (bugs, missing twins, missing ISAs)
4.  **Fix Doxygen** — clarify unclear prose, add missing documentation, fix
    incorrect `\b Example` code (wrong args, missing allocations/frees)
5.  **Improve `_generic` readability** — simplify if the logic is hard to follow,
    but do not change its numerical behavior (it is the correctness reference)
6.  **Fix bugs** (Phase 1) — Please correct any identified issues.
7.  **Add missing `_u_` twins** (Phase 2) if any `_a_` variants lack them
8.  **Add missing `_a_` twins** (Phase 3) if any `_u_` variants lack them
9.  **Verify ISA ordering** — implementations within each section should follow
    the ordering in "ISA ordering convention" above
10. **Add new ISA implementations** (Phase 4) Are there other implementations (e.g.
    sse, sse2, sse4.1, avx, avx2, avx512, etc) which could be more efficient on
    certain hardware?
11. **Build and test** — build in **Release mode** (`-DCMAKE_BUILD_TYPE=Release`),
    then run `volk_profile -R <kernel_name>` to confirm correctness and get meaningful
    benchmark numbers. Debug builds lack `-O3` auto-vectorization, which causes the
    `_generic` variant to appear far slower than it really is and makes hand-written
    SIMD look disproportionately fast — misleading for both correctness assessment and
    optimization decisions.
12. **Fix bugs, Second Pass** Do you see any other errors in the implementation?
13. **Optimize** Do you see any areas of improvement?
14. **Add new ISA, Second Pass** Are there other implementations (e.g. sse, sse2,
    sse4.1, avx, avx2, avx512) which could be more efficient on certain hardware?
15. **Final benchmark** — build in **Release mode** (`-DCMAKE_BUILD_TYPE=Release`),
    then run `volk_profile -T 10 -R <kernel_name>` to record the performance after
    making changes.  Please store this output in "log/<kernel_name>-after.txt".
16. This output should contain two ISAs with times labeled: "Best aligned arch"
    & "Best unaligned arch".  Please append to the row with the corresponding
    <kernel_name> in "KernelImprovements.md" with: best a_ISA After, a_improve After,
    best u_ISA After, and u_improve After.  Please order them such that:

    "best a_ISA Before", "best a_improve Before", "best a_ISA After", "best a_improve After",
    "best u_ISA Before", "best u_improve Before", "best u_ISA After", "best u_improve After"

    Please remove the word "best " from the column headings.  I included it to be easier
    for you to interpret the meaning. Please format this table as a markdown table
    (not a CSV).  Thanks!

---

## What NOT to change

- **ORC implementations** — auto-generated from `.orc` files; do not edit
- **`__restrict__`** — do not add unless explicitly requested
- **Unrelated code** — when fixing a kernel, do not refactor surrounding code,
  add comments to unchanged functions, or modify other kernels in the same session
