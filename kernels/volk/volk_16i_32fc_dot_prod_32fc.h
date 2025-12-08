/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_32fc_dot_prod_32fc
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and will be complex.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_32fc_dot_prod_32fc(lv_32fc_t* result, const short* input, const lv_32fc_t
 * * taps, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of shorts.
 * \li taps:  complex taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a complex value to hold the dot product result.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_16i_32fc_dot_prod_32fc();
 *
 * \endcode
 */

#ifndef INCLUDED_volk_16i_32fc_dot_prod_32fc_H
#define INCLUDED_volk_16i_32fc_dot_prod_32fc_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_16i_32fc_dot_prod_32fc_generic(lv_32fc_t* result,
                                                       const short* input,
                                                       const lv_32fc_t* taps,
                                                       unsigned int num_points)
{

    static const int N_UNROLL = 4;

    lv_32fc_t acc0 = 0;
    lv_32fc_t acc1 = 0;
    lv_32fc_t acc2 = 0;
    lv_32fc_t acc3 = 0;

    unsigned i = 0;
    unsigned n = (num_points / N_UNROLL) * N_UNROLL;

    for (i = 0; i < n; i += N_UNROLL) {
        acc0 += taps[i + 0] * (float)input[i + 0];
        acc1 += taps[i + 1] * (float)input[i + 1];
        acc2 += taps[i + 2] * (float)input[i + 2];
        acc3 += taps[i + 3] * (float)input[i + 3];
    }

    for (; i < num_points; i++) {
        acc0 += taps[i] * (float)input[i];
    }

    *result = acc0 + acc1 + acc2 + acc3;
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_16i_32fc_dot_prod_32fc_neon(lv_32fc_t* result,
                                                    const short* input,
                                                    const lv_32fc_t* taps,
                                                    unsigned int num_points)
{

    unsigned ii;
    unsigned quarter_points = num_points / 4;
    lv_32fc_t* tapsPtr = (lv_32fc_t*)taps;
    short* inputPtr = (short*)input;
    lv_32fc_t accumulator_vec[4];

    float32x4x2_t tapsVal, accumulator_val;
    int16x4_t input16;
    int32x4_t input32;
    float32x4_t input_float, prod_re, prod_im;

    accumulator_val.val[0] = vdupq_n_f32(0.0);
    accumulator_val.val[1] = vdupq_n_f32(0.0);

    for (ii = 0; ii < quarter_points; ++ii) {
        tapsVal = vld2q_f32((float*)tapsPtr);
        input16 = vld1_s16(inputPtr);
        // widen 16-bit int to 32-bit int
        input32 = vmovl_s16(input16);
        // convert 32-bit int to float with scale
        input_float = vcvtq_f32_s32(input32);

        prod_re = vmulq_f32(input_float, tapsVal.val[0]);
        prod_im = vmulq_f32(input_float, tapsVal.val[1]);

        accumulator_val.val[0] = vaddq_f32(prod_re, accumulator_val.val[0]);
        accumulator_val.val[1] = vaddq_f32(prod_im, accumulator_val.val[1]);

        tapsPtr += 4;
        inputPtr += 4;
    }
    vst2q_f32((float*)accumulator_vec, accumulator_val);
    accumulator_vec[0] += accumulator_vec[1];
    accumulator_vec[2] += accumulator_vec[3];
    accumulator_vec[0] += accumulator_vec[2];

    for (ii = quarter_points * 4; ii < num_points; ++ii) {
        accumulator_vec[0] += *(tapsPtr++) * (float)(*(inputPtr++));
    }

    *result = accumulator_vec[0];
}

#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_32fc_dot_prod_32fc_neonv8(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const short* inputPtr = input;
    const lv_32fc_t* tapsPtr = taps;

    /* Use 2 independent real/imag accumulators for FMA pipelining */
    float32x4_t real_acc0 = vdupq_n_f32(0);
    float32x4_t imag_acc0 = vdupq_n_f32(0);
    float32x4_t real_acc1 = vdupq_n_f32(0);
    float32x4_t imag_acc1 = vdupq_n_f32(0);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        /* Load 8 int16 values and convert to float */
        int16x8_t input16 = vld1q_s16(inputPtr);
        float32x4_t input_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input16)));
        float32x4_t input_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input16)));

        /* Load 8 complex taps deinterleaved */
        float32x4x2_t taps0 = vld2q_f32((const float*)tapsPtr);
        float32x4x2_t taps1 = vld2q_f32((const float*)(tapsPtr + 4));
        __VOLK_PREFETCH(inputPtr + 16);
        __VOLK_PREFETCH(tapsPtr + 16);

        /* FMA: acc += input * taps */
        real_acc0 = vfmaq_f32(real_acc0, input_lo, taps0.val[0]);
        imag_acc0 = vfmaq_f32(imag_acc0, input_lo, taps0.val[1]);
        real_acc1 = vfmaq_f32(real_acc1, input_hi, taps1.val[0]);
        imag_acc1 = vfmaq_f32(imag_acc1, input_hi, taps1.val[1]);

        inputPtr += 8;
        tapsPtr += 8;
    }

    /* Combine accumulators */
    real_acc0 = vaddq_f32(real_acc0, real_acc1);
    imag_acc0 = vaddq_f32(imag_acc0, imag_acc1);

    /* Horizontal sum */
    float real_sum = vaddvq_f32(real_acc0);
    float imag_sum = vaddvq_f32(imag_acc0);

    lv_32fc_t returnValue = lv_cmake(real_sum, imag_sum);

    /* Handle remainder */
    const float* bPtr = (const float*)tapsPtr;
    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        returnValue += lv_cmake(inputPtr[0] * bPtr[0], inputPtr[0] * bPtr[1]);
        inputPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}
#endif /*LV_HAVE_NEONV8*/

#if LV_HAVE_SSE && LV_HAVE_MMX

static inline void volk_16i_32fc_dot_prod_32fc_u_sse(lv_32fc_t* result,
                                                     const short* input,
                                                     const lv_32fc_t* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m64 m0, m1;
    __m128 f0, f1, f2, f3;
    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        m0 = _mm_set_pi16(*(aPtr + 3), *(aPtr + 2), *(aPtr + 1), *(aPtr + 0));
        m1 = _mm_set_pi16(*(aPtr + 7), *(aPtr + 6), *(aPtr + 5), *(aPtr + 4));
        f0 = _mm_cvtpi16_ps(m0);
        f1 = _mm_cvtpi16_ps(m0);
        f2 = _mm_cvtpi16_ps(m1);
        f3 = _mm_cvtpi16_ps(m1);

        a0Val = _mm_unpacklo_ps(f0, f1);
        a1Val = _mm_unpackhi_ps(f0, f1);
        a2Val = _mm_unpacklo_ps(f2, f3);
        a3Val = _mm_unpackhi_ps(f2, f3);

        b0Val = _mm_loadu_ps(bPtr);
        b1Val = _mm_loadu_ps(bPtr + 4);
        b2Val = _mm_loadu_ps(bPtr + 8);
        b3Val = _mm_loadu_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    _mm_empty(); // clear the mmx technology state

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE && LV_HAVE_MMX*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_16i_32fc_dot_prod_32fc_u_avx2_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_loadu_si128((__m128i const*)aPtr);
        m1 = _mm_loadu_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2 && lV_HAVE_FMA*/


#ifdef LV_HAVE_AVX2

static inline void volk_16i_32fc_dot_prod_32fc_u_avx2(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_loadu_si128((__m128i const*)aPtr);
        m1 = _mm_loadu_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2*/


#if LV_HAVE_SSE && LV_HAVE_MMX


static inline void volk_16i_32fc_dot_prod_32fc_a_sse(lv_32fc_t* result,
                                                     const short* input,
                                                     const lv_32fc_t* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m64 m0, m1;
    __m128 f0, f1, f2, f3;
    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        m0 = _mm_set_pi16(*(aPtr + 3), *(aPtr + 2), *(aPtr + 1), *(aPtr + 0));
        m1 = _mm_set_pi16(*(aPtr + 7), *(aPtr + 6), *(aPtr + 5), *(aPtr + 4));
        f0 = _mm_cvtpi16_ps(m0);
        f1 = _mm_cvtpi16_ps(m0);
        f2 = _mm_cvtpi16_ps(m1);
        f3 = _mm_cvtpi16_ps(m1);

        a0Val = _mm_unpacklo_ps(f0, f1);
        a1Val = _mm_unpackhi_ps(f0, f1);
        a2Val = _mm_unpacklo_ps(f2, f3);
        a3Val = _mm_unpackhi_ps(f2, f3);

        b0Val = _mm_load_ps(bPtr);
        b1Val = _mm_load_ps(bPtr + 4);
        b2Val = _mm_load_ps(bPtr + 8);
        b3Val = _mm_load_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    _mm_empty(); // clear the mmx technology state

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE && LV_HAVE_MMX*/

#ifdef LV_HAVE_AVX2

static inline void volk_16i_32fc_dot_prod_32fc_a_avx2(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_load_si128((__m128i const*)aPtr);
        m1 = _mm_load_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}


#endif /*LV_HAVE_AVX2*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_16i_32fc_dot_prod_32fc_a_avx2_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_load_si128((__m128i const*)aPtr);
        m1 = _mm_load_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}


#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_16i_32fc_dot_prod_32fc_rvv(lv_32fc_t* result,
                                                   const short* input,
                                                   const lv_32fc_t* taps,
                                                   unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)taps, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v =
            __riscv_vfwcvt_f(__riscv_vle16_v_i16m2((const int16_t*)input, vl), vl);
        vsumr = __riscv_vfmacc_tu(vsumr, vr, v, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vi, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_16i_32fc_dot_prod_32fc_rvvseg(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)taps, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v =
            __riscv_vfwcvt_f(__riscv_vle16_v_i16m2((const int16_t*)input, vl), vl);
        vsumr = __riscv_vfmacc_tu(vsumr, vr, v, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vi, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_16i_32fc_dot_prod_32fc_H*/
