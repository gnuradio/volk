# SIMD Pattern Reference (16ic and Complex Float Kernels)

Read this file when implementing or modifying VOLK kernel SIMD code. These are
proven patterns extracted from working implementations.

## `_mm512_permutex2var_epi8` index encoding (AVX-512VBMI)

This intrinsic selects bytes from the concatenation of two 512-bit source registers.
Each byte in the index vector:
- **Bit 6** selects the source register (0 = first operand, 1 = second operand)
- **Bits [5:0]** select the byte position (0–63) within that source

Valid index range: 0–127. Indices 0–63 read from source 1, 64–127 from source 2.

---

## Int16 → float conversion patterns (SSE/AVX)

**SSE2** (no `_mm_cvtepi16_epi32`):
```c
__m128i sign = _mm_srai_epi16(raw, 15);          // sign mask: 0x0000 or 0xFFFF
__m128i lo32 = _mm_unpacklo_epi16(raw, sign);    // [I0,Q0,I1,Q1] as int32
__m128i hi32 = _mm_unpackhi_epi16(raw, sign);    // [I2,Q2,I3,Q3] as int32
```

**SSE4.1** (cleaner):
```c
__m128i lo32 = _mm_cvtepi16_epi32(raw);
__m128i hi32 = _mm_cvtepi16_epi32(_mm_srli_si128(raw, 8));
```

**AVX (without AVX2)** — AVX always implies SSE4.1, so `_mm_cvtepi16_epi32` is
available. Load 8× int16 into a 128-bit register, convert each 64-bit half separately,
then combine into a 256-bit result:
```c
__m128i cplx = _mm_load_si128((const __m128i*)ptr);         // 8× int16
__m128  lo_f = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(cplx));   // lower 4× int16 → float
__m128  hi_f = _mm_cvtepi32_ps(
    _mm_cvtepi16_epi32(_mm_unpackhi_epi64(cplx, cplx)));    // upper 4× int16 → float
__m256  out  = _mm256_set_m128(hi_f, lo_f);                 // 8× float
```
Note: `_mm_cvtepi16_epi32` always reads only the **lower 64 bits** of its input.
Use `_mm_unpackhi_epi64(reg, reg)` (or `_mm_srli_si128(reg, 8)`) to move the upper
half to the lower lane before the second call.

**AVX2**:
```c
__m128i half = _mm256_extracti128_si256(reg256, 0 or 1);
__m256i wide = _mm256_cvtepi16_epi32(half);   // 8 x int16 → 8 x int32
```

**AVX512F**:
```c
__m256i lo = _mm512_castsi512_si256(raw);             // lower 256 bits
__m256i hi = _mm512_extracti64x4_epi64(raw, 1);      // upper 256 bits
__m512i wide = _mm512_cvtepi16_epi32(half);           // 16 x int16 → 16 x int32
```

---

## Extracting the high byte of int16 complex samples to int8

For kernels outputting `(int8_t)(I[i] >> 8)`: within each `lv_16sc_t` =
`[I_lo, I_hi, Q_lo, Q_hi]`, the target byte is at offsets 1, 5, 9, 13 within
each 128-bit register (4 complex samples per register).

**SSE2**: isolate `I_hi` with AND, shift to byte 0 of each int32, narrow:
```c
const __m128i iMask = _mm_set1_epi32(0x0000FF00);
v = _mm_srli_epi32(_mm_and_si128(v, iMask), 8);
// then packs_epi32 (pair of registers) → packs result via packus_epi16 → uint8
// (bit pattern of I_hi is already the correct int8 two's-complement value)
```

**SSSE3 / AVX2 (optimal — 4-mask direct placement)**: Use 4 `pshufb` masks, each
routing `I_hi` from one 128-bit load directly to its output byte position, then OR
all 4. Eliminates `srai_epi16` and `packs_epi16` (7 ops vs 9):
```c
// iMask1 routes bytes 1,5,9,13 → output positions 0,1,2,3; zeros elsewhere
// iMask2 → positions 4,5,6,7; iMask3 → 8..11; iMask4 → 12..15
const __m128i iMask1 = _mm_set_epi8(
    0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 13,9,5,1);
// store OR of all four shuffled registers directly
```
For **AVX2**, replicate each mask across both 128-bit lanes. After ORing all four,
32-bit output groups are interleaved between lanes; restore sequential order with
`_mm256_permutevar8x32_epi32(..., _mm256_set_epi32(7,3,6,2,5,1,4,0))` (8 ops vs 12).

**AVX512BW**: `_mm512_shuffle_epi8` + OR + `_mm512_permutexvar_epi64` collects 32
I-values as int16; then `_mm512_srai_epi16(v, 8)` + `_mm512_cvtepi16_epi8(v)`
narrows to a `__m256i` of 32 int8 values.

**AVX512VBMI** (optimal — single cross-source byte permute): Two 512-bit loads +
one `_mm512_permutex2var_epi8` produces 32 int8 results directly. The index vector
picks byte 1 (I_hi) from each 4-byte complex sample across both inputs:
```c
const __m512i iHiIdx = _mm512_set_epi8(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      // unused upper 256 bits
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125,121,117,113, 109,105,101,97, 93,89,85,81, 77,73,69,65,  // I_hi from source 2
    61,57,53,49, 45,41,37,33, 29,25,21,17, 13,9,5,1);           // I_hi from source 1
__m256i result = _mm512_castsi512_si256(
    _mm512_permutex2var_epi8(complexVal1, iHiIdx, complexVal2));
```
3 ops total (2 loads + 1 permute + 1 cast) vs BW's 9 ops. Measured **17% faster**
than AVX-512BW on deinterleave_real_8i.

---

## Extracting one int16 channel from interleaved complex int16

For kernels that output only I (or Q) from `[I0,Q0,I1,Q1,...]` into a flat int16 buffer:

**SSSE3** (8 samples/iter — two 128-bit loads):
```c
// mask1: routes bytes 0,1,4,5,8,9,12,13 (I-values) to low 8 bytes, zeros high 8 bytes
__m128i mask1 = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 13,12,9,8,5,4,1,0);
// mask2: routes I-values to high 8 bytes, zeros low 8 bytes
__m128i mask2 = _mm_set_epi8(13,12,9,8,5,4,1,0, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80);
iOutputVal = _mm_or_si128(_mm_shuffle_epi8(v1, mask1), _mm_shuffle_epi8(v2, mask2));
```

**AVX2** (16 samples/iter — two 256-bit loads): replicate the SSSE3 masks across both
128-bit lanes of a 256-bit register. After `_mm256_or_si256`, fix lane interleaving:
```c
iOutputVal = _mm256_permute4x64_epi64(_mm256_or_si256(...), 0xd8);
```

**AVX-512F (16 samples/iter — single 512-bit load, float output only)**: When the
output is `float`, treat each `lv_16sc_t` as `int32` and sign-extend I in-place — no
shuffle required:
```c
const int32_t* p = (const int32_t*)complexVectorPtr; // each int32 = one complex sample
__m512i iq  = _mm512_loadu_si512((const void*)p);     // 16 × [I,Q] as int32
__m512i i32 = _mm512_srai_epi32(_mm512_slli_epi32(iq, 16), 16); // sign-extend I to 32b
__m512  f   = _mm512_cvtepi32_ps(i32);                // 16 × float, ready to scale
```
Shift left 16 moves I to the upper half of each int32 (discarding Q); arithmetic shift
right 16 sign-extends it back. All intrinsics are **AVX-512F** — no BW required.

**AVX512BW** (32 samples/iter — two 512-bit loads): replicate masks across all four
128-bit lanes of a 512-bit register. After `_mm512_or_si512`, the 64-bit qwords are in
order `[I0..3, I16..19, I4..7, I20..23, I8..11, I24..27, I12..15, I28..31]`; fix with:
```c
__m512i permIdx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
iOutputVal = _mm512_permutexvar_epi64(permIdx, _mm512_or_si512(v1, v2));
```

**AVX512VBMI** (32 samples/iter — two 512-bit loads, optimal): Single
`_mm512_permutex2var_epi8` replaces the entire BW shuffle+OR+permute chain. The
index vector picks bytes 0,1 (I channel) from each 4-byte complex sample:
```c
const __m512i iIdx = _mm512_set_epi8(
    125,124, 121,120, 117,116, 113,112, 109,108, 105,104, 101,100, 97,96,
    93,92, 89,88, 85,84, 81,80, 77,76, 73,72, 69,68, 65,64,
    61,60, 57,56, 53,52, 49,48, 45,44, 41,40, 37,36, 33,32,
    29,28, 25,24, 21,20, 17,16, 13,12, 9,8, 5,4, 1,0);
_mm512_store_si512(iBufferPtr,
    _mm512_permutex2var_epi8(complexVal1, iIdx, complexVal2));
```
4 ops (2 loads + 1 permute + 1 store) vs BW's 7 ops. For Q channel, offset indices
by +2 (bytes 2,3 of each sample).

---

## Deinterleaving both int16 channels simultaneously (two-load patterns)

For kernels that output *both* I and Q into separate flat int16 buffers, 32 complex
samples per iteration using two 512-bit loads.

### AVX-512BW

After `_mm512_shuffle_epi8` with the 16-byte IQ-separation mask replicated across all
four 128-bit lanes, the 64-bit chunks of each 512-bit register holding 16 complex
samples are: `[I0-3, Q0-3, I4-7, Q4-7, I8-11, Q8-11, I12-15, Q12-15]`.

`_mm512_permutex2var_epi64` gathers from two source registers in one instruction
(indices 0–7 select from the first operand, 8–15 from the second):
```c
const __m512i iIdx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0); // even chunks → I
const __m512i qIdx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1); // odd chunks  → Q
__m512i iOut = _mm512_permutex2var_epi64(shuffled1, iIdx, shuffled2); // all 32 I values
__m512i qOut = _mm512_permutex2var_epi64(shuffled1, qIdx, shuffled2); // all 32 Q values
```
Note: `_mm512_permutex2var_epi64` is **AVX-512F** (not BW), so it is available inside
an `LV_HAVE_AVX512BW` block without any additional guard.

### AVX-512VBMI

Same task, but `_mm512_permutex2var_epi8` operates at byte granularity — no
pre-shuffle step needed. Two calls (one with I indices, one with Q indices) replace
the entire BW pipeline of 2 shuffles + 2 `permutex2var_epi64`:
```c
const __m512i iIdx = _mm512_set_epi8(          // bytes 0,1 of each 4-byte sample
    125,124, 121,120, 117,116, 113,112, 109,108, 105,104, 101,100, 97,96,
    93,92, 89,88, 85,84, 81,80, 77,76, 73,72, 69,68, 65,64,
    61,60, 57,56, 53,52, 49,48, 45,44, 41,40, 37,36, 33,32,
    29,28, 25,24, 21,20, 17,16, 13,12, 9,8, 5,4, 1,0);
const __m512i qIdx = _mm512_set_epi8(          // bytes 2,3 of each 4-byte sample
    127,126, 123,122, 119,118, 115,114, 111,110, 107,106, 103,102, 99,98,
    95,94, 91,90, 87,86, 83,82, 79,78, 75,74, 71,70, 67,66,
    63,62, 59,58, 55,54, 51,50, 47,46, 43,42, 39,38, 35,34,
    31,30, 27,26, 23,22, 19,18, 15,14, 11,10, 7,6, 3,2);
__m512i iOut = _mm512_permutex2var_epi8(complexVal1, iIdx, complexVal2);
__m512i qOut = _mm512_permutex2var_epi8(complexVal1, qIdx, complexVal2);
```
6 ops total (2 loads + 2 permutes + 2 stores) vs BW's 8 ops (2 loads + 2 shuffles
+ 2 permutes + 2 stores).

---

## Deinterleaving complex float (re/im separation)

After converting interleaved `[I0,Q0,I1,Q1,...]` floats:

**SSE (128-bit)** — `cplxValue1=[I0,Q0,I1,Q1]`, `cplxValue2=[I2,Q2,I3,Q3]`:
```c
__m128 re = _mm_shuffle_ps(cplxValue1, cplxValue2, 0x88); // [I0,I1,I2,I3]
__m128 im = _mm_shuffle_ps(cplxValue1, cplxValue2, 0xdd); // [Q0,Q1,Q2,Q3]
// 0x88 = _MM_SHUFFLE(2,0,2,0),  0xdd = _MM_SHUFFLE(3,1,3,1)
```

**AVX2 (256-bit)** — `_mm256_shuffle_ps` operates within 128-bit lanes; needs a
`_mm256_permutevar8x32_ps` to fix the lane-crossing artifact:
```c
__m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
__m256 re = _mm256_permutevar8x32_ps(
    _mm256_shuffle_ps(cplxValue1, cplxValue2, 0x88), idx);
__m256 im = _mm256_permutevar8x32_ps(
    _mm256_shuffle_ps(cplxValue1, cplxValue2, 0xdd), idx);
```

**AVX512F** — use `permutex2var_ps` across two 512-bit float registers:
```c
// fltA=[I0,Q0,...,I7,Q7], fltB=[I8,Q8,...,I15,Q15]
// Bit 4 of each index selects source (0=fltA, 1=fltB); bits [3:0] select element
const __m512i idx_re = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
const __m512i idx_im = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);
__m512 re = _mm512_permutex2var_ps(fltA, idx_re, fltB);
__m512 im = _mm512_permutex2var_ps(fltA, idx_im, fltB);
```

**AVX512BW** — deinterleave at int16 level first (two fully independent chains):
```c
// Selects even (I) or odd (Q) int16 elements into lower 256 bits; upper are don't-cares
const __m512i idx_re = _mm512_set_epi16(0,0,...,0, 30,28,26,...,2,0);
const __m512i idx_im = _mm512_set_epi16(0,0,...,0, 31,29,27,...,3,1);
__m256i i16_re = _mm512_castsi512_si256(_mm512_permutexvar_epi16(idx_re, raw));
__m256i i16_im = _mm512_castsi512_si256(_mm512_permutexvar_epi16(idx_im, raw));
// then _mm512_cvtepi16_epi32 each → two independent float chains
```

---

## Narrowing float/int32 → int16 output (AVX2 / AVX-512F)

For kernels that compute a float result per sample and write `int16_t` output:

**AVX2** (8 samples → 8 × int16, stored to 16 bytes):
```c
__m256i int_result = _mm256_cvtps_epi32(float_result);      // 8 × float → 8 × int32
__m256i packed     = _mm256_packs_epi32(int_result, int_result);  // 16 × int16, duplicated
// lanes are [m0..3,m0..3 | m4..7,m4..7]; collect sequential values into low 128 bits:
packed = _mm256_permute4x64_epi64(packed, 0x08);             // 0x08 = qwords [0,2,0,0]
_mm_storeu_si128((__m128i*)outPtr, _mm256_castsi256_si128(packed));
```

**AVX-512F** (16 samples → 16 × int16, stored to 32 bytes):
`_mm512_cvtsepi32_epi16` converts 16 × int32 → 16 × int16 with **signed saturation**,
returning `__m256i` directly — no packing step needed:
```c
__m512i int_result  = _mm512_cvtps_epi32(float_result);      // 16 × float → 16 × int32
__m256i i16_result  = _mm512_cvtsepi32_epi16(int_result);    // saturating narrow → __m256i
_mm256_storeu_si256((__m256i*)outPtr, i16_result);
```
`_mm512_cvtsepi32_epi16` is **AVX-512F** (not BW). It clamps values outside
`[INT16_MIN, INT16_MAX]` rather than wrapping, which is the correct behavior when the
float computation can slightly exceed the int16 range due to rounding.
