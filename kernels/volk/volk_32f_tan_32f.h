/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_tan_32f
 *
 * \b Overview
 *
 * Computes the tangent of each element of the aVector.
 *
 * b[i] = tan(a[i])
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_tan_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The buffer of points.
 * \li num_points: The number of values in input buffer.
 *
 * \b Outputs
 * \li bVector: The output buffer.
 *
 * \b Example
 * Calculate tan(theta) for common angles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.000;
 *   in[1] = 0.524;
 *   in[2] = 0.785;
 *   in[3] = 1.047;
 *   in[4] = 1.571  ;
 *   in[5] = 1.571  ;
 *   in[6] = -1.047;
 *   in[7] = -0.785;
 *   in[8] = -0.524;
 *   in[9] = -0.000;
 *
 *   volk_32f_tan_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("tan(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_tan_32f_a_H
#define INCLUDED_volk_32f_tan_32f_a_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tan(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tan(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_tan_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, twos), fours)), fzeroes);

        __m128 temp = cosine;
        cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1));
        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(temp, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        cosine = _mm_sub_ps(
            cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3));
        tangent = _mm_div_ps(sine, cosine);
        _mm_store_ps(bPtr, tangent);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */


#endif /* INCLUDED_volk_32f_tan_32f_a_H */

#ifndef INCLUDED_volk_32f_tan_32f_u_H
#define INCLUDED_volk_32f_tan_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tan(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tan(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_tan_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, tangent, condition1, condition2, condition3;
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
        condition3 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, twos), fours)), fzeroes);

        __m128 temp = cosine;
        cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1));
        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(temp, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        cosine = _mm_sub_ps(
            cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3));
        tangent = _mm_div_ps(sine, cosine);
        _mm_storeu_ps(bPtr, tangent);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tan_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_tan_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
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
        b_vec = _vtanq_f32(a_vec);
        vst1q_f32(bVectorPtr, b_vec);
        // move pointers ahead
        bVectorPtr += 4;
        aVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        *bVectorPtr++ = tanf(*aVectorPtr++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_tan_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    float* bVectorPtr = bVector;
    const float* aVectorPtr = aVector;

    for (number = 0; number < quarter_points; number++) {
        float32x4_t a_vec = vld1q_f32(aVectorPtr);
        // Use sincos, then native division for tan = sin/cos
        const float32x4x2_t sincos = _vsincosq_f32(a_vec);
        float32x4_t b_vec = vdivq_f32(sincos.val[0], sincos.val[1]);
        vst1q_f32(bVectorPtr, b_vec);
        bVectorPtr += 4;
        aVectorPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *bVectorPtr++ = tanf(*aVectorPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_tan_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t c4oPi = __riscv_vfmv_v_f_f32m2(1.2732395f, vlmax);
    const vfloat32m2_t cPio4a = __riscv_vfmv_v_f_f32m2(0.7853982f, vlmax);
    const vfloat32m2_t cPio4b = __riscv_vfmv_v_f_f32m2(7.946627e-09f, vlmax);
    const vfloat32m2_t cPio4c = __riscv_vfmv_v_f_f32m2(3.061617e-17f, vlmax);

    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf4 = __riscv_vfmv_v_f_f32m2(4.0f, vlmax);

    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(0.0833333333f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(0.0027777778f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(4.9603175e-05f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(5.5114638e-07f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t s = __riscv_vfabs(v, vl);
        vint32m2_t q = __riscv_vfcvt_x(__riscv_vfmul(s, c4oPi, vl), vl);
        vfloat32m2_t r = __riscv_vfcvt_f(__riscv_vadd(q, __riscv_vand(q, 1, vl), vl), vl);

        s = __riscv_vfnmsac(s, cPio4a, r, vl);
        s = __riscv_vfnmsac(s, cPio4b, r, vl);
        s = __riscv_vfnmsac(s, cPio4c, r, vl);

        s = __riscv_vfmul(s, 1 / 8.0f, vl);
        s = __riscv_vfmul(s, s, vl);
        vfloat32m2_t t = s;
        s = __riscv_vfmsub(s, c5, c4, vl);
        s = __riscv_vfmadd(s, t, c3, vl);
        s = __riscv_vfmsub(s, t, c2, vl);
        s = __riscv_vfmadd(s, t, cf1, vl);
        s = __riscv_vfmul(s, t, vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, 1 / 2.0f, vl);

        vfloat32m2_t sine =
            __riscv_vfsqrt(__riscv_vfmul(__riscv_vfrsub(s, 2.0f, vl), s, vl), vl);
        vfloat32m2_t cosine = __riscv_vfsub(cf1, s, vl);

        vbool16_t m1 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 1, vl), 2, vl), 0, vl);
        vbool16_t m2 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 2, vl), 4, vl), 0, vl);
        vbool16_t m3 = __riscv_vmxor(__riscv_vmslt(__riscv_vreinterpret_i32m2(v), 0, vl),
                                     __riscv_vmsne(__riscv_vand(q, 4, vl), 0, vl),
                                     vl);

        vfloat32m2_t sine0 = sine;
        sine = __riscv_vmerge(sine, cosine, m1, vl);
        sine = __riscv_vfneg_mu(m3, sine, sine, vl);

        cosine = __riscv_vmerge(cosine, sine0, m1, vl);
        cosine = __riscv_vfneg_mu(m2, cosine, cosine, vl);

        __riscv_vse32(bVector, __riscv_vfdiv(sine, cosine, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_tan_32f_u_H */
