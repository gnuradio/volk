/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_conjugate_dot_prod_32fc
 *
 * \b Overview
 *
 * This block computes the conjugate dot product (or inner product)
 * between two vectors, the \p input and \p taps vectors. Given a set
 * of \p num_points taps, the result is the sum of products between
 * the input vector and the conjugate of the taps. The result is a
 * single value stored in the \p result address and is returned as a
 * complex float.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_conjugate_dot_prod_32fc(lv_32fc_t* result, const lv_32fc_t* input,
 * const lv_32fc_t* taps, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of complex floats.
 * \li taps:  complex float taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a complex float value to hold the dot product result.
 *
 * \b Example
 * \code
 * unsigned int N = 1000;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_32fc_t* a = (lv_32fc_t*) volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* b = (lv_32fc_t*) volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * for (int i = 0; i < N; ++i) {
 *   a[i] = lv_cmake(.50f, .50f);
 *   b[i] = lv_cmake(.50f, .75f);
 * }
 *
 * lv_32fc_t e = (float) N * a[0] * lv_conj(b[0]); // When a and b constant
 * lv_32fc_t res;
 *
 * volk_32fc_x2_conjugate_dot_prod_32fc(&res, a, b, N);
 *
 * printf("Expected: %8.2f%+8.2fi\n", lv_real(e), lv_imag(e));
 * printf("Result:   %8.2f%+8.2fi\n", lv_real(res), lv_imag(res));
 *
 * volk_free(a);
 * volk_free(b);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H
#define INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H


#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_generic(lv_32fc_t* result,
                                                                const lv_32fc_t* input,
                                                                const lv_32fc_t* taps,
                                                                unsigned int num_points)
{
    lv_32fc_t res = lv_cmake(0.f, 0.f);
    for (unsigned int i = 0; i < num_points; ++i) {
        res += (*input++) * lv_conj((*taps++));
    }
    *result = res;
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_block(lv_32fc_t* result,
                                                              const lv_32fc_t* input,
                                                              const lv_32fc_t* taps,
                                                              unsigned int num_points)
{

    const unsigned int num_bytes = num_points * 8;

    float* res = (float*)result;
    float* in = (float*)input;
    float* tp = (float*)taps;
    unsigned int n_2_ccomplex_blocks = num_bytes >> 4;

    float sum0[2] = { 0, 0 };
    float sum1[2] = { 0, 0 };
    unsigned int i = 0;

    for (i = 0; i < n_2_ccomplex_blocks; ++i) {
        sum0[0] += in[0] * tp[0] + in[1] * tp[1];
        sum0[1] += (-in[0] * tp[1]) + in[1] * tp[0];
        sum1[0] += in[2] * tp[2] + in[3] * tp[3];
        sum1[1] += (-in[2] * tp[3]) + in[3] * tp[2];

        in += 4;
        tp += 4;
    }

    res[0] = sum0[0] + sum1[0];
    res[1] = sum0[1] + sum1[1];

    if (num_bytes >> 3 & 1) {
        *result += input[(num_bytes >> 3) - 1] * lv_conj(taps[(num_bytes >> 3) - 1]);
    }
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_u_avx(lv_32fc_t* result,
                                                              const lv_32fc_t* input,
                                                              const lv_32fc_t* taps,
                                                              unsigned int num_points)
{
    // Partial sums for indices i, i+1, i+2 and i+3.
    __m256 sum_a_mult_b_real = _mm256_setzero_ps();
    __m256 sum_a_mult_b_imag = _mm256_setzero_ps();

    for (long unsigned i = 0; i < (num_points & ~3u); i += 4) {
        /* Four complex elements a time are processed.
         * (ar + j⋅ai)*conj(br + j⋅bi) =
         * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
         */

        /* Load input and taps, split and duplicate real und imaginary parts of taps.
         * a: | ai,i+3 | ar,i+3 | … | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
         * b: | bi,i+3 | br,i+3 | … | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
         * b_real: | br,i+3 | br,i+3 | … | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
         * b_imag: | bi,i+3 | bi,i+3 | … | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
         */
        __m256 a = _mm256_loadu_ps((const float*)&input[i]);
        __m256 b = _mm256_loadu_ps((const float*)&taps[i]);
        __m256 b_real = _mm256_moveldup_ps(b);
        __m256 b_imag = _mm256_movehdup_ps(b);

        // Add | ai⋅br,i+3 | ar⋅br,i+3 | … | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
        sum_a_mult_b_real = _mm256_add_ps(sum_a_mult_b_real, _mm256_mul_ps(a, b_real));
        // Add | ai⋅bi,i+3 | −ar⋅bi,i+3 | … | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
        sum_a_mult_b_imag = _mm256_addsub_ps(sum_a_mult_b_imag, _mm256_mul_ps(a, b_imag));
    }

    // Swap position of −ar⋅bi and ai⋅bi.
    sum_a_mult_b_imag = _mm256_permute_ps(sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
    // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains four such partial sums.
    __m256 sum = _mm256_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
    /* Sum the four partial sums: Add high half of vector sum to the low one, i.e.
     * s1 + s3 and s0 + s2 …
     */
    sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x01));
    // … and now (s0 + s2) + (s1 + s3)
    sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    // Store result.
    __m128 lower = _mm256_extractf128_ps(sum, 0);
    _mm_storel_pi((__m64*)result, lower);

    // Handle the last elements if num_points mod 4 is bigger than 0.
    for (long unsigned i = num_points & ~3u; i < num_points; ++i) {
        *result += lv_cmake(lv_creal(input[i]) * lv_creal(taps[i]) +
                                lv_cimag(input[i]) * lv_cimag(taps[i]),
                            lv_cimag(input[i]) * lv_creal(taps[i]) -
                                lv_creal(input[i]) * lv_cimag(taps[i]));
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_u_sse3(lv_32fc_t* result,
                                                               const lv_32fc_t* input,
                                                               const lv_32fc_t* taps,
                                                               unsigned int num_points)
{
    // Partial sums for indices i and i+1.
    __m128 sum_a_mult_b_real = _mm_setzero_ps();
    __m128 sum_a_mult_b_imag = _mm_setzero_ps();

    for (long unsigned i = 0; i < (num_points & ~1u); i += 2) {
        /* Two complex elements a time are processed.
         * (ar + j⋅ai)*conj(br + j⋅bi) =
         * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
         */

        /* Load input and taps, split and duplicate real und imaginary parts of taps.
         * a: | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
         * b: | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
         * b_real: | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
         * b_imag: | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
         */
        __m128 a = _mm_loadu_ps((const float*)&input[i]);
        __m128 b = _mm_loadu_ps((const float*)&taps[i]);
        __m128 b_real = _mm_moveldup_ps(b);
        __m128 b_imag = _mm_movehdup_ps(b);

        // Add | ai⋅br,i+1 | ar⋅br,i+1 | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
        sum_a_mult_b_real = _mm_add_ps(sum_a_mult_b_real, _mm_mul_ps(a, b_real));
        // Add | ai⋅bi,i+1 | −ar⋅bi,i+1 | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
        sum_a_mult_b_imag = _mm_addsub_ps(sum_a_mult_b_imag, _mm_mul_ps(a, b_imag));
    }

    // Swap position of −ar⋅bi and ai⋅bi.
    sum_a_mult_b_imag =
        _mm_shuffle_ps(sum_a_mult_b_imag, sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
    // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains two such partial sums.
    __m128 sum = _mm_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
    // Sum the two partial sums.
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
    // Store result.
    _mm_storel_pi((__m64*)result, sum);

    // Handle the last element if num_points mod 2 is 1.
    if (num_points & 1u) {
        *result += lv_cmake(
            lv_creal(input[num_points - 1]) * lv_creal(taps[num_points - 1]) +
                lv_cimag(input[num_points - 1]) * lv_cimag(taps[num_points - 1]),
            lv_cimag(input[num_points - 1]) * lv_creal(taps[num_points - 1]) -
                lv_creal(input[num_points - 1]) * lv_cimag(taps[num_points - 1]));
    }
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32fc_x2_conjugate_dot_prod_32fc_neon(lv_32fc_t* result,
                                                             const lv_32fc_t* input,
                                                             const lv_32fc_t* taps,
                                                             unsigned int num_points)
{

    unsigned int quarter_points = num_points / 4;
    unsigned int number;

    lv_32fc_t* a_ptr = (lv_32fc_t*)taps;
    lv_32fc_t* b_ptr = (lv_32fc_t*)input;
    // for 2-lane vectors, 1st lane holds the real part,
    // 2nd lane holds the imaginary part
    float32x4x2_t a_val, b_val, accumulator;
    float32x4x2_t tmp_imag;
    accumulator.val[0] = vdupq_n_f32(0);
    accumulator.val[1] = vdupq_n_f32(0);

    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((float*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
        b_val = vld2q_f32((float*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
        __VOLK_PREFETCH(a_ptr + 8);
        __VOLK_PREFETCH(b_ptr + 8);

        // do the first multiply
        tmp_imag.val[1] = vmulq_f32(a_val.val[1], b_val.val[0]);
        tmp_imag.val[0] = vmulq_f32(a_val.val[0], b_val.val[0]);

        // use multiply accumulate/subtract to get result
        tmp_imag.val[1] = vmlsq_f32(tmp_imag.val[1], a_val.val[0], b_val.val[1]);
        tmp_imag.val[0] = vmlaq_f32(tmp_imag.val[0], a_val.val[1], b_val.val[1]);

        accumulator.val[0] = vaddq_f32(accumulator.val[0], tmp_imag.val[0]);
        accumulator.val[1] = vaddq_f32(accumulator.val[1], tmp_imag.val[1]);

        // increment pointers
        a_ptr += 4;
        b_ptr += 4;
    }
    lv_32fc_t accum_result[4];
    vst2q_f32((float*)accum_result, accumulator);
    *result = accum_result[0] + accum_result[1] + accum_result[2] + accum_result[3];

    // tail case
    for (number = quarter_points * 4; number < num_points; ++number) {
        *result += (*a_ptr++) * lv_conj(*b_ptr++);
    }
    *result = lv_conj(*result);
}
#endif /*LV_HAVE_NEON*/

#endif /*INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H*/

#ifndef INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H
#define INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H

#include <stdio.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_avx(lv_32fc_t* result,
                                                              const lv_32fc_t* input,
                                                              const lv_32fc_t* taps,
                                                              unsigned int num_points)
{
    // Partial sums for indices i, i+1, i+2 and i+3.
    __m256 sum_a_mult_b_real = _mm256_setzero_ps();
    __m256 sum_a_mult_b_imag = _mm256_setzero_ps();

    for (long unsigned i = 0; i < (num_points & ~3u); i += 4) {
        /* Four complex elements a time are processed.
         * (ar + j⋅ai)*conj(br + j⋅bi) =
         * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
         */

        /* Load input and taps, split and duplicate real und imaginary parts of taps.
         * a: | ai,i+3 | ar,i+3 | … | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
         * b: | bi,i+3 | br,i+3 | … | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
         * b_real: | br,i+3 | br,i+3 | … | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
         * b_imag: | bi,i+3 | bi,i+3 | … | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
         */
        __m256 a = _mm256_load_ps((const float*)&input[i]);
        __m256 b = _mm256_load_ps((const float*)&taps[i]);
        __m256 b_real = _mm256_moveldup_ps(b);
        __m256 b_imag = _mm256_movehdup_ps(b);

        // Add | ai⋅br,i+3 | ar⋅br,i+3 | … | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
        sum_a_mult_b_real = _mm256_add_ps(sum_a_mult_b_real, _mm256_mul_ps(a, b_real));
        // Add | ai⋅bi,i+3 | −ar⋅bi,i+3 | … | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
        sum_a_mult_b_imag = _mm256_addsub_ps(sum_a_mult_b_imag, _mm256_mul_ps(a, b_imag));
    }

    // Swap position of −ar⋅bi and ai⋅bi.
    sum_a_mult_b_imag = _mm256_permute_ps(sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
    // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains four such partial sums.
    __m256 sum = _mm256_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
    /* Sum the four partial sums: Add high half of vector sum to the low one, i.e.
     * s1 + s3 and s0 + s2 …
     */
    sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x01));
    // … and now (s0 + s2) + (s1 + s3)
    sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    // Store result.
    __m128 lower = _mm256_extractf128_ps(sum, 0);
    _mm_storel_pi((__m64*)result, lower);

    // Handle the last elements if num_points mod 4 is bigger than 0.
    for (long unsigned i = num_points & ~3u; i < num_points; ++i) {
        *result += lv_cmake(lv_creal(input[i]) * lv_creal(taps[i]) +
                                lv_cimag(input[i]) * lv_cimag(taps[i]),
                            lv_cimag(input[i]) * lv_creal(taps[i]) -
                                lv_creal(input[i]) * lv_cimag(taps[i]));
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_sse3(lv_32fc_t* result,
                                                               const lv_32fc_t* input,
                                                               const lv_32fc_t* taps,
                                                               unsigned int num_points)
{
    // Partial sums for indices i and i+1.
    __m128 sum_a_mult_b_real = _mm_setzero_ps();
    __m128 sum_a_mult_b_imag = _mm_setzero_ps();

    for (long unsigned i = 0; i < (num_points & ~1u); i += 2) {
        /* Two complex elements a time are processed.
         * (ar + j⋅ai)*conj(br + j⋅bi) =
         * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
         */

        /* Load input and taps, split and duplicate real und imaginary parts of taps.
         * a: | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
         * b: | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
         * b_real: | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
         * b_imag: | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
         */
        __m128 a = _mm_load_ps((const float*)&input[i]);
        __m128 b = _mm_load_ps((const float*)&taps[i]);
        __m128 b_real = _mm_moveldup_ps(b);
        __m128 b_imag = _mm_movehdup_ps(b);

        // Add | ai⋅br,i+1 | ar⋅br,i+1 | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
        sum_a_mult_b_real = _mm_add_ps(sum_a_mult_b_real, _mm_mul_ps(a, b_real));
        // Add | ai⋅bi,i+1 | −ar⋅bi,i+1 | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
        sum_a_mult_b_imag = _mm_addsub_ps(sum_a_mult_b_imag, _mm_mul_ps(a, b_imag));
    }

    // Swap position of −ar⋅bi and ai⋅bi.
    sum_a_mult_b_imag =
        _mm_shuffle_ps(sum_a_mult_b_imag, sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
    // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains two such partial sums.
    __m128 sum = _mm_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
    // Sum the two partial sums.
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
    // Store result.
    _mm_storel_pi((__m64*)result, sum);

    // Handle the last element if num_points mod 2 is 1.
    if (num_points & 1u) {
        *result += lv_cmake(
            lv_creal(input[num_points - 1]) * lv_creal(taps[num_points - 1]) +
                lv_cimag(input[num_points - 1]) * lv_cimag(taps[num_points - 1]),
            lv_cimag(input[num_points - 1]) * lv_creal(taps[num_points - 1]) -
                lv_creal(input[num_points - 1]) * lv_cimag(taps[num_points - 1]));
    }
}

#endif /*LV_HAVE_SSE3*/


#endif /*INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H*/
