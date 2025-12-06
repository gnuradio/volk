/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold AVX intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX_INTRINSICS_H_
#include <immintrin.h>

/*
 * Approximate arctan(x) via polynomial expansion
 * on the interval [-1, 1]
 *
 * Maximum relative error ~6.5e-7
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arctan_poly_avx(const __m256 x)
{
    const __m256 a1 = _mm256_set1_ps(+0x1.ffffeap-1f);
    const __m256 a3 = _mm256_set1_ps(-0x1.55437p-2f);
    const __m256 a5 = _mm256_set1_ps(+0x1.972be6p-3f);
    const __m256 a7 = _mm256_set1_ps(-0x1.1436ap-3f);
    const __m256 a9 = _mm256_set1_ps(+0x1.5785aap-4f);
    const __m256 a11 = _mm256_set1_ps(-0x1.2f3004p-5f);
    const __m256 a13 = _mm256_set1_ps(+0x1.01a37cp-7f);

    const __m256 x_times_x = _mm256_mul_ps(x, x);
    __m256 arctan;
    arctan = a13;
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a11);
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a9);
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a7);
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a5);
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a3);
    arctan = _mm256_mul_ps(x_times_x, arctan);
    arctan = _mm256_add_ps(arctan, a1);
    arctan = _mm256_mul_ps(x, arctan);

    return arctan;
}

/*
 * Approximate arcsin(x) via polynomial expansion
 * P(u) such that asin(x) = x * P(x^2) on |x| <= 0.5
 *
 * Maximum relative error ~1.5e-6
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arcsin_poly_avx(const __m256 x)
{
    const __m256 c0 = _mm256_set1_ps(0x1.ffffcep-1f);
    const __m256 c1 = _mm256_set1_ps(0x1.55b648p-3f);
    const __m256 c2 = _mm256_set1_ps(0x1.24d192p-4f);
    const __m256 c3 = _mm256_set1_ps(0x1.0a788p-4f);

    const __m256 u = _mm256_mul_ps(x, x);
    __m256 p = c3;
    p = _mm256_mul_ps(u, p);
    p = _mm256_add_ps(p, c2);
    p = _mm256_mul_ps(u, p);
    p = _mm256_add_ps(p, c1);
    p = _mm256_mul_ps(u, p);
    p = _mm256_add_ps(p, c0);

    return _mm256_mul_ps(x, p);
}

static inline __m256 _mm256_complexmul_ps(__m256 x, __m256 y)
{
    __m256 yl, yh, tmp1, tmp2;
    yl = _mm256_moveldup_ps(y);        // Load yl with cr,cr,dr,dr ...
    yh = _mm256_movehdup_ps(y);        // Load yh with ci,ci,di,di ...
    tmp1 = _mm256_mul_ps(x, yl);       // tmp1 = ar*cr,ai*cr,br*dr,bi*dr ...
    x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br ...
    tmp2 = _mm256_mul_ps(x, yh);       // tmp2 = ai*ci,ar*ci,bi*di,br*di

    // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di
    return _mm256_addsub_ps(tmp1, tmp2);
}

static inline __m256 _mm256_conjugate_ps(__m256 x)
{
    const __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);
    return _mm256_xor_ps(x, conjugator); // conjugate y
}

static inline __m256 _mm256_complexconjugatemul_ps(const __m256 x, const __m256 y)
{
    const __m256 nswap = _mm256_permute_ps(x, 0xb1);
    const __m256 dreal = _mm256_moveldup_ps(y);
    const __m256 dimag = _mm256_movehdup_ps(y);

    const __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);
    const __m256 dimagconj = _mm256_xor_ps(dimag, conjugator);
    const __m256 multreal = _mm256_mul_ps(x, dreal);
    const __m256 multimag = _mm256_mul_ps(nswap, dimagconj);
    return _mm256_add_ps(multreal, multimag);
}

static inline __m256 _mm256_normalize_ps(__m256 val)
{
    __m256 tmp1 = _mm256_mul_ps(val, val);
    tmp1 = _mm256_hadd_ps(tmp1, tmp1);
    tmp1 = _mm256_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(3, 1, 2, 0)); // equals 0xD8
    tmp1 = _mm256_sqrt_ps(tmp1);
    return _mm256_div_ps(val, tmp1);
}

static inline __m256 _mm256_magnitudesquared_ps(__m256 cplxValue1, __m256 cplxValue2)
{
    __m256 complex1, complex2;
    cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values
    complex1 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x20);
    complex2 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x31);
    return _mm256_hadd_ps(complex1, complex2); // Add the I2 and Q2 values
}

static inline __m256 _mm256_magnitude_ps(__m256 cplxValue1, __m256 cplxValue2)
{
    return _mm256_sqrt_ps(_mm256_magnitudesquared_ps(cplxValue1, cplxValue2));
}

static inline __m256 _mm256_scaled_norm_dist_ps(const __m256 symbols0,
                                                const __m256 symbols1,
                                                const __m256 points0,
                                                const __m256 points1,
                                                const __m256 scalar)
{
    /*
     * Calculate: |y - x|^2 * SNR_lin
     * Consider 'symbolsX' and 'pointsX' to be complex float
     * 'symbolsX' are 'y' and 'pointsX' are 'x'
     */
    const __m256 diff0 = _mm256_sub_ps(symbols0, points0);
    const __m256 diff1 = _mm256_sub_ps(symbols1, points1);
    const __m256 norms = _mm256_magnitudesquared_ps(diff0, diff1);
    return _mm256_mul_ps(norms, scalar);
}

static inline __m256 _mm256_polar_sign_mask(__m128i fbits)
{
    __m256 sign_mask_dummy = _mm256_setzero_ps();
    const __m128i zeros = _mm_set1_epi8(0x00);
    const __m128i sign_extract = _mm_set1_epi8(0x80);
    const __m128i shuffle_mask0 = _mm_setr_epi8(0xff,
                                                0xff,
                                                0xff,
                                                0x00,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x01,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x02,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x03);
    const __m128i shuffle_mask1 = _mm_setr_epi8(0xff,
                                                0xff,
                                                0xff,
                                                0x04,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x05,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x06,
                                                0xff,
                                                0xff,
                                                0xff,
                                                0x07);

    fbits = _mm_cmpgt_epi8(fbits, zeros);
    fbits = _mm_and_si128(fbits, sign_extract);
    __m128i sign_bits0 = _mm_shuffle_epi8(fbits, shuffle_mask0);
    __m128i sign_bits1 = _mm_shuffle_epi8(fbits, shuffle_mask1);

    __m256 sign_mask =
        _mm256_insertf128_ps(sign_mask_dummy, _mm_castsi128_ps(sign_bits0), 0x0);
    return _mm256_insertf128_ps(sign_mask, _mm_castsi128_ps(sign_bits1), 0x1);
    // This is the desired function call. Though it seems to be missing in GCC.
    // Compare: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
    // return _mm256_set_m128(_mm_castsi128_ps(sign_bits1),
    // _mm_castsi128_ps(sign_bits0));
}

static inline void
_mm256_polar_deinterleave(__m256* llr0, __m256* llr1, __m256 src0, __m256 src1)
{
    // deinterleave values
    __m256 part0 = _mm256_permute2f128_ps(src0, src1, 0x20);
    __m256 part1 = _mm256_permute2f128_ps(src0, src1, 0x31);
    *llr0 = _mm256_shuffle_ps(part0, part1, 0x88);
    *llr1 = _mm256_shuffle_ps(part0, part1, 0xdd);
}

static inline __m256 _mm256_polar_minsum_llrs(__m256 src0, __m256 src1)
{
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 abs_mask =
        _mm256_andnot_ps(sign_mask, _mm256_castsi256_ps(_mm256_set1_epi8(0xff)));

    __m256 llr0, llr1;
    _mm256_polar_deinterleave(&llr0, &llr1, src0, src1);

    // calculate result
    __m256 sign =
        _mm256_xor_ps(_mm256_and_ps(llr0, sign_mask), _mm256_and_ps(llr1, sign_mask));
    __m256 dst =
        _mm256_min_ps(_mm256_and_ps(llr0, abs_mask), _mm256_and_ps(llr1, abs_mask));
    return _mm256_or_ps(dst, sign);
}

static inline __m256 _mm256_polar_fsign_add_llrs(__m256 src0, __m256 src1, __m128i fbits)
{
    // prepare sign mask for correct +-
    __m256 sign_mask = _mm256_polar_sign_mask(fbits);

    __m256 llr0, llr1;
    _mm256_polar_deinterleave(&llr0, &llr1, src0, src1);

    // calculate result
    llr0 = _mm256_xor_ps(llr0, sign_mask);
    __m256 dst = _mm256_add_ps(llr0, llr1);
    return dst;
}

static inline __m256 _mm256_accumulate_square_sum_ps(
    __m256 sq_acc, __m256 acc, __m256 val, __m256 rec, __m256 aux)
{
    aux = _mm256_mul_ps(aux, val);
    aux = _mm256_sub_ps(aux, acc);
    aux = _mm256_mul_ps(aux, aux);
    aux = _mm256_mul_ps(aux, rec);
    return _mm256_add_ps(sq_acc, aux);
}

#endif /* INCLUDE_VOLK_VOLK_AVX_INTRINSICS_H_ */
