/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_sin_32f
 *
 * \b Overview
 *
 * Computes the sine of the input vector and stores the results in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_sin_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The output vector.
 *
 * \b Example
 * Calculate sin(theta) for several common angles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.000;
 *   in[1] = 0.524;
 *   in[2] = 0.786;
 *   in[3] = 1.047;
 *   in[4] = 1.571;
 *   in[5] = 1.571;
 *   in[6] = 2.094;
 *   in[7] = 2.356;
 *   in[8] = 2.618;
 *   in[9] = 3.142;
 *
 *   volk_32f_sin_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("sin(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_sin_32f_a_H
#define INCLUDED_volk_32f_sin_32f_a_H
#ifdef LV_HAVE_AVX512F

#include <immintrin.h>
static inline void volk_32f_sin_32f_a_avx512f(float* sinVector,
                                              const float* inVector,
                                              unsigned int num_points)
{
    float* sinPtr = sinVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, r, m4pi, pio4A, pio4B, pio4C, cp1, cp2, cp3, cp4, cp5, ffours, ftwos,
        fones;
    __m512 sine, cosine;
    __m512i q, zeros, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239544735162542821171882678754627704620361328125);
    pio4A = _mm512_set1_ps(0.7853981554508209228515625);
    pio4B = _mm512_set1_ps(0.794662735614792836713604629039764404296875e-8);
    pio4C = _mm512_set1_ps(0.306161699786838294306516483068750264552437361480769e-16);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    zeros = _mm512_setzero_epi32();
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.08333333333333333);
    cp3 = _mm512_set1_ps(0.002777777777777778);
    cp4 = _mm512_set1_ps(4.96031746031746e-05);
    cp5 = _mm512_set1_ps(5.511463844797178e-07);
    __mmask16 condition1, condition2, ltZero;

    for (; number < sixteenPoints; number++) {
        aVal = _mm512_load_ps(inPtr);
        // s = fabs(aVal)
        s = (__m512)(_mm512_and_si512((__m512i)(aVal), _mm512_set1_epi32(0x7fffffff)));

        // q = (int) (s * (4/pi)), floor(aVal / (pi/4))
        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        // r = q + q&1, q indicates quadrant, r gives
        r = _mm512_cvtepi32_ps(_mm512_add_epi32(q, _mm512_and_si512(q, ones)));

        s = _mm512_fnmadd_ps(r, pio4A, s);
        s = _mm512_fnmadd_ps(r, pio4B, s);
        s = _mm512_fnmadd_ps(r, pio4C, s);

        s = _mm512_div_ps(
            s,
            _mm512_set1_ps(8.0f)); // The constant is 2^N, for 3 times argument reduction
        s = _mm512_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++)
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        condition1 = _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos), zeros);
        ltZero = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        condition2 = _mm512_kxor(
            _mm512_cmpneq_epi32_mask(_mm512_and_epi32(q, fours), zeros), ltZero);

        sine = _mm512_mask_blend_ps(condition1, sine, cosine);
        sine = _mm512_mask_mul_ps(sine, condition2, sine, _mm512_set1_ps(-1.f));
        _mm512_store_ps(sinPtr, sine);
        inPtr += 16;
        sinPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr++);
    }
}
#endif
#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_sin_32f_a_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, condition1, condition2;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4A, s);
        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4B, s);

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_fmadd_ps(
                _mm256_fmsub_ps(
                    _mm256_fmadd_ps(_mm256_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        // Need this condition only for cos
        // condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q,
        // twos), fours)), fzeroes);

        sine =
            _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(cosine, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        _mm256_store_ps(bPtr, sine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = sin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_sin_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, condition1, condition2;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4B));

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4),
                                              s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        // Need this condition only for cos
        // condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q,
        // twos), fours)), fzeroes);

        sine =
            _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(cosine, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        _mm256_store_ps(bPtr, sine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = sin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_sin_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, condition1, condition2;
    __m128i q, r, ones, twos, fours;

    m4pi = _mm_set1_ps(1.273239545);
    pio4A = _mm_set1_ps(0.78515625);
    pio4B = _mm_set1_ps(0.241876e-3);
    ffours = _mm_set1_ps(4.0);
    ftwos = _mm_set1_ps(2.0);
    fones = _mm_set1_ps(1.0);
    fzeroes = _mm_setzero_ps();
    ones = _mm_set1_epi32(1);
    twos = _mm_set1_epi32(2);
    fours = _mm_set1_epi32(4);

    cp1 = _mm_set1_ps(1.0);
    cp2 = _mm_set1_ps(0.83333333e-1);
    cp3 = _mm_set1_ps(0.2777778e-2);
    cp4 = _mm_set1_ps(0.49603e-4);
    cp5 = _mm_set1_ps(0.551e-6);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        s = _mm_sub_ps(aVal,
                       _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
        q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
        r = _mm_add_epi32(q, _mm_and_si128(q, ones));

        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4A));
        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4B));

        s = _mm_div_ps(
            s, _mm_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm_mul_ps(
            _mm_add_ps(
                _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_mul_ps(
                            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s),
                                       cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
        }
        s = _mm_div_ps(s, ftwos);

        sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
        cosine = _mm_sub_ps(fones, s);

        condition1 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, ones), twos)), fzeroes);
        condition2 = _mm_cmpneq_ps(
            _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(q, fours)), fzeroes),
            _mm_cmplt_ps(aVal, fzeroes));
        // Need this condition only for cos
        // condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q,
        // twos), fours)), fzeroes);

        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(cosine, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        _mm_store_ps(bPtr, sine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = sinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */


#endif /* INCLUDED_volk_32f_sin_32f_a_H */

#ifndef INCLUDED_volk_32f_sin_32f_u_H
#define INCLUDED_volk_32f_sin_32f_u_H

#ifdef LV_HAVE_AVX512F

#include <immintrin.h>
static inline void volk_32f_sin_32f_u_avx512f(float* sinVector,
                                              const float* inVector,
                                              unsigned int num_points)
{
    float* sinPtr = sinVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, r, m4pi, pio4A, pio4B, pio4C, cp1, cp2, cp3, cp4, cp5, ffours, ftwos,
        fones;
    __m512 sine, cosine;
    __m512i q, zeros, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239544735162542821171882678754627704620361328125);
    pio4A = _mm512_set1_ps(0.7853981554508209228515625);
    pio4B = _mm512_set1_ps(0.794662735614792836713604629039764404296875e-8);
    pio4C = _mm512_set1_ps(0.306161699786838294306516483068750264552437361480769e-16);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    zeros = _mm512_setzero_epi32();
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.08333333333333333);
    cp3 = _mm512_set1_ps(0.002777777777777778);
    cp4 = _mm512_set1_ps(4.96031746031746e-05);
    cp5 = _mm512_set1_ps(5.511463844797178e-07);
    __mmask16 condition1, condition2, ltZero;

    for (; number < sixteenPoints; number++) {
        aVal = _mm512_loadu_ps(inPtr);
        // s = fabs(aVal)
        s = (__m512)(_mm512_and_si512((__m512i)(aVal), _mm512_set1_epi32(0x7fffffff)));

        // q = (int) (s * (4/pi)), floor(aVal / (pi/4))
        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        // r = q + q&1, q indicates quadrant, r gives
        r = _mm512_cvtepi32_ps(_mm512_add_epi32(q, _mm512_and_si512(q, ones)));

        s = _mm512_fnmadd_ps(r, pio4A, s);
        s = _mm512_fnmadd_ps(r, pio4B, s);
        s = _mm512_fnmadd_ps(r, pio4C, s);

        s = _mm512_div_ps(
            s,
            _mm512_set1_ps(8.0f)); // The constant is 2^N, for 3 times argument reduction
        s = _mm512_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++)
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        condition1 = _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos), zeros);
        ltZero = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        condition2 = _mm512_kxor(
            _mm512_cmpneq_epi32_mask(_mm512_and_epi32(q, fours), zeros), ltZero);

        sine = _mm512_mask_blend_ps(condition1, sine, cosine);
        sine = _mm512_mask_mul_ps(sine, condition2, sine, _mm512_set1_ps(-1.f));
        _mm512_storeu_ps(sinPtr, sine);
        inPtr += 16;
        sinPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr++);
    }
}
#endif

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_sin_32f_u_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, condition1, condition2;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4A, s);
        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4B, s);

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_fmadd_ps(
                _mm256_fmsub_ps(
                    _mm256_fmadd_ps(_mm256_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        // Need this condition only for cos
        // condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q,
        // twos), fours)), fzeroes);

        sine =
            _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(cosine, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        _mm256_storeu_ps(bPtr, sine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = sin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_sin_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, condition1, condition2;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4B));

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4),
                                              s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        // Need this condition only for cos
        // condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q,
        // twos), fours)), fzeroes);

        sine =
            _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(cosine, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        _mm256_storeu_ps(bPtr, sine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = sin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_sin_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, condition1, condition2;
    __m128i q, r, ones, twos, fours;

    m4pi = _mm_set1_ps(1.273239545);
    pio4A = _mm_set1_ps(0.78515625);
    pio4B = _mm_set1_ps(0.241876e-3);
    ffours = _mm_set1_ps(4.0);
    ftwos = _mm_set1_ps(2.0);
    fones = _mm_set1_ps(1.0);
    fzeroes = _mm_setzero_ps();
    ones = _mm_set1_epi32(1);
    twos = _mm_set1_epi32(2);
    fours = _mm_set1_epi32(4);

    cp1 = _mm_set1_ps(1.0);
    cp2 = _mm_set1_ps(0.83333333e-1);
    cp3 = _mm_set1_ps(0.2777778e-2);
    cp4 = _mm_set1_ps(0.49603e-4);
    cp5 = _mm_set1_ps(0.551e-6);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        s = _mm_sub_ps(aVal,
                       _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
        q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
        r = _mm_add_epi32(q, _mm_and_si128(q, ones));

        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4A));
        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4B));

        s = _mm_div_ps(
            s, _mm_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm_mul_ps(
            _mm_add_ps(
                _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_mul_ps(
                            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s),
                                       cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
        }
        s = _mm_div_ps(s, ftwos);

        sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
        cosine = _mm_sub_ps(fones, s);

        condition1 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, ones), twos)), fzeroes);
        condition2 = _mm_cmpneq_ps(
            _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(q, fours)), fzeroes),
            _mm_cmplt_ps(aVal, fzeroes));

        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(cosine, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        _mm_storeu_ps(bPtr, sine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = sinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_sin_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = sinf(*aPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_sin_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    float* bVectorPtr = bVector;
    const float* aVectorPtr = aVector;

    float32x4_t b_vec;
    float32x4_t a_vec;

    for (number = 0; number < quarter_points; number++) {
        a_vec = vld1q_f32(aVectorPtr);
        // Prefetch next one, speeds things up
        __VOLK_PREFETCH(aVectorPtr + 4);
        b_vec = _vsinq_f32(a_vec);
        vst1q_f32(bVectorPtr, b_vec);
        // move pointers ahead
        bVectorPtr += 4;
        aVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        *bVectorPtr++ = sinf(*aVectorPtr++);
    }
}

#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_32f_sin_32f_u_H */
