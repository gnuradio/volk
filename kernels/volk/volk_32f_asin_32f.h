/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_asin_32f
 *
 * \b Overview
 *
 * Computes arcsine of input vector and stores results in output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_asin_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The vector where results will be stored.
 *
 * \b Example
 * \code
 * Calculate common angles around the top half of the unit circle.
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0;
 *   in[1] = 0.5;
 *   in[2] = std::sqrt(2.f)/2.f;
 *   in[3] = std::sqrt(3.f)/2.f;
 *   in[4] = in[5] = 1;
 *   for(unsigned int ii = 6; ii < N; ++ii){
 *       in[ii] = - in[N-ii-1];
 *   }
 *
 *   volk_32f_asin_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("asin(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

/* This is the number of terms of Taylor series to evaluate, increase this for more
 * accuracy*/
#define ASIN_TERMS 2

#ifndef INCLUDED_volk_32f_asin_32f_a_H
#define INCLUDED_volk_32f_asin_32f_a_H

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_asin_32f_a_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int sixteenthPoints = num_points / 16;
    int i, j;

    __m512 aVal, pio2, x, y, z, arcsine;
    __m512 fzeroes, fones, ftwos, ffours;
    __mmask16 condition;

    pio2 = _mm512_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm512_setzero_ps();
    fones = _mm512_set1_ps(1.0);
    ftwos = _mm512_set1_ps(2.0);
    ffours = _mm512_set1_ps(4.0);

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);
        aVal =
            _mm512_mul_ps(aVal,
                          _mm512_rsqrt14_ps(_mm512_mul_ps(_mm512_add_ps(fones, aVal),
                                                          _mm512_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm512_cmp_ps_mask(z, fzeroes, _CMP_LT_OS);
        z = _mm512_mask_sub_ps(z, condition, z, _mm512_mul_ps(z, ftwos));
        condition = _mm512_cmp_ps_mask(z, fones, _CMP_LT_OS);
        x = _mm512_mask_add_ps(z, condition, z, _mm512_sub_ps(_mm512_rcp14_ps(z), z));

        for (i = 0; i < 2; i++) {
            x = _mm512_add_ps(x, _mm512_sqrt_ps(_mm512_fmadd_ps(x, x, fones)));
        }
        x = _mm512_rcp14_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm512_fmadd_ps(
                y, _mm512_mul_ps(x, x), _mm512_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm512_mul_ps(y, _mm512_mul_ps(x, ffours));
        condition = _mm512_cmp_ps_mask(z, fones, _CMP_GT_OS);

        y = _mm512_mask_add_ps(y, condition, y, _mm512_fnmadd_ps(y, ftwos, pio2));
        arcsine = y;
        condition = _mm512_cmp_ps_mask(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm512_mask_sub_ps(
            arcsine, condition, arcsine, _mm512_mul_ps(arcsine, ftwos));

        _mm512_store_ps(bPtr, arcsine);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX512F for aligned */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_a_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(z,
                          _mm256_and_ps(_mm256_sub_ps(_mm256_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        }
        x = _mm256_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_store_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_asin_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(z,
                          _mm256_and_ps(_mm256_sub_ps(_mm256_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        }
        x = _mm256_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_store_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_asin_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, pio2, x, y, z, arcsine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        aVal = _mm_div_ps(
            aVal,
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        }
        x = _mm_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arcsine = _mm_sub_ps(arcsine, _mm_and_ps(_mm_mul_ps(arcsine, ftwos), condition));

        _mm_store_ps(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_asin_32f_a_H */

#ifndef INCLUDED_volk_32f_asin_32f_u_H
#define INCLUDED_volk_32f_asin_32f_u_H

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_asin_32f_u_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int sixteenthPoints = num_points / 16;
    int i, j;

    __m512 aVal, pio2, x, y, z, arcsine;
    __m512 fzeroes, fones, ftwos, ffours;
    __mmask16 condition;

    pio2 = _mm512_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm512_setzero_ps();
    fones = _mm512_set1_ps(1.0);
    ftwos = _mm512_set1_ps(2.0);
    ffours = _mm512_set1_ps(4.0);

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);
        aVal =
            _mm512_mul_ps(aVal,
                          _mm512_rsqrt14_ps(_mm512_mul_ps(_mm512_add_ps(fones, aVal),
                                                          _mm512_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm512_cmp_ps_mask(z, fzeroes, _CMP_LT_OS);
        z = _mm512_mask_sub_ps(z, condition, z, _mm512_mul_ps(z, ftwos));
        condition = _mm512_cmp_ps_mask(z, fones, _CMP_LT_OS);
        x = _mm512_mask_add_ps(z, condition, z, _mm512_sub_ps(_mm512_rcp14_ps(z), z));

        for (i = 0; i < 2; i++) {
            x = _mm512_add_ps(x, _mm512_sqrt_ps(_mm512_fmadd_ps(x, x, fones)));
        }
        x = _mm512_rcp14_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm512_fmadd_ps(
                y, _mm512_mul_ps(x, x), _mm512_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm512_mul_ps(y, _mm512_mul_ps(x, ffours));
        condition = _mm512_cmp_ps_mask(z, fones, _CMP_GT_OS);

        y = _mm512_mask_add_ps(y, condition, y, _mm512_fnmadd_ps(y, ftwos, pio2));
        arcsine = y;
        condition = _mm512_cmp_ps_mask(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm512_mask_sub_ps(
            arcsine, condition, arcsine, _mm512_mul_ps(arcsine, ftwos));

        _mm512_storeu_ps(bPtr, arcsine);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX512F for unaligned */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_u_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(z,
                          _mm256_and_ps(_mm256_sub_ps(_mm256_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        }
        x = _mm256_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_storeu_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_asin_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(z,
                          _mm256_and_ps(_mm256_sub_ps(_mm256_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        }
        x = _mm256_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_storeu_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for unaligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_asin_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, pio2, x, y, z, arcsine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        aVal = _mm_div_ps(
            aVal,
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_rcp_ps(z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        }
        x = _mm_rcp_ps(x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arcsine = _mm_sub_ps(arcsine, _mm_and_ps(_mm_mul_ps(arcsine, ftwos), condition));

        _mm_storeu_ps(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_asin_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_asin_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    float32x4_t aVal, x, y, z, arcsine;
    uint32x4_t condition;
    const float32x4_t fzeroes = vdupq_n_f32(0.0f);
    const float32x4_t fones = vdupq_n_f32(1.0f);
    const float32x4_t ftwos = vdupq_n_f32(2.0f);
    const float32x4_t ffours = vdupq_n_f32(4.0f);
    const float32x4_t pio2 = vdupq_n_f32(3.14159265358979323846f / 2.0f);

    for (; number < quarterPoints; number++) {
        aVal = vld1q_f32(aPtr);

        // Compute x / sqrt((1+x)*(1-x)) using reciprocal estimate
        // For |x| > 1, this produces NaN (matching generic behavior)
        float32x4_t one_plus = vaddq_f32(fones, aVal);
        float32x4_t one_minus = vsubq_f32(fones, aVal);
        float32x4_t sqrt_arg = vmulq_f32(one_plus, one_minus);

        // Newton-Raphson sqrt approximation
        float32x4_t sqrt_est = vrsqrteq_f32(sqrt_arg);
        sqrt_est =
            vmulq_f32(sqrt_est, vrsqrtsq_f32(vmulq_f32(sqrt_arg, sqrt_est), sqrt_est));
        float32x4_t sqrt_val = vmulq_f32(sqrt_arg, sqrt_est);

        // Reciprocal of sqrt_val
        float32x4_t recip = vrecpeq_f32(sqrt_val);
        recip = vmulq_f32(recip, vrecpsq_f32(sqrt_val, recip));
        float32x4_t tanVal = vmulq_f32(aVal, recip);

        z = tanVal;
        // z = abs(z)
        condition = vcltq_f32(z, fzeroes);
        z = vbslq_f32(condition, vnegq_f32(z), z);

        // x = 1/z if z < 1, else x = z (matching SSE logic)
        condition = vcltq_f32(z, fones);
        float32x4_t z_recip = vrecpeq_f32(z);
        z_recip = vmulq_f32(z_recip, vrecpsq_f32(z, z_recip));
        x = vbslq_f32(condition, z_recip, z);

        // Two iterations: x = x + sqrt(1 + x*x)
        // Note: For very large x (approaching infinity), the NR rsqrt iteration produces
        // NaN due to inf*0 in vrsqrtsq. Use approximation sqrt(1+x²) ≈ x for large x.
        const float32x4_t large_threshold = vdupq_n_f32(1e10f);
        for (i = 0; i < 2; i++) {
            float32x4_t xx = vmulq_f32(x, x);
            float32x4_t sum = vaddq_f32(fones, xx);
            uint32x4_t is_large = vcgtq_f32(x, large_threshold);
            float32x4_t sqrt_sum_est = vrsqrteq_f32(sum);
            sqrt_sum_est = vmulq_f32(
                sqrt_sum_est, vrsqrtsq_f32(vmulq_f32(sum, sqrt_sum_est), sqrt_sum_est));
            float32x4_t sqrt_sum = vmulq_f32(sum, sqrt_sum_est);
            sqrt_sum = vbslq_f32(is_large, x, sqrt_sum);
            x = vaddq_f32(x, sqrt_sum);
        }

        // x = 1/x
        float32x4_t x_recip = vrecpeq_f32(x);
        x_recip = vmulq_f32(x_recip, vrecpsq_f32(x, x_recip));
        x = x_recip;

        // Taylor series
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            float coeff = (j % 2 == 0) ? 1.0f / (2 * j + 1) : -1.0f / (2 * j + 1);
            y = vaddq_f32(vmulq_f32(y, vmulq_f32(x, x)), vdupq_n_f32(coeff));
        }

        y = vmulq_f32(y, vmulq_f32(x, ffours));

        // Adjust if z > 1: y = y + (pio2 - 2*y)
        condition = vcgtq_f32(z, fones);
        float32x4_t y_adj = vsubq_f32(pio2, vmulq_f32(y, ftwos));
        y = vbslq_f32(condition, vaddq_f32(y, y_adj), y);

        arcsine = y;

        // If tanVal < 0, arcsine = -arcsine
        condition = vcltq_f32(tanVal, fzeroes);
        arcsine = vbslq_f32(condition, vnegq_f32(arcsine), arcsine);

        vst1q_f32(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_asin_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    float32x4_t aVal, x, y, z, arcsine;
    const float32x4_t fzeroes = vdupq_n_f32(0.0f);
    const float32x4_t fones = vdupq_n_f32(1.0f);
    const float32x4_t ftwos = vdupq_n_f32(2.0f);
    const float32x4_t ffours = vdupq_n_f32(4.0f);
    const float32x4_t pio2 = vdupq_n_f32(3.14159265358979323846f / 2.0f);

    for (; number < quarterPoints; number++) {
        aVal = vld1q_f32(aPtr);

        // Compute x / sqrt((1+x)*(1-x))
        // For |x| > 1, this produces NaN (matching generic behavior)
        float32x4_t one_plus = vaddq_f32(fones, aVal);
        float32x4_t one_minus = vsubq_f32(fones, aVal);
        float32x4_t sqrt_val = vsqrtq_f32(vmulq_f32(one_plus, one_minus));
        float32x4_t tanVal = vdivq_f32(aVal, sqrt_val);

        z = tanVal;
        // z = abs(z)
        z = vabsq_f32(z);

        // x = 1/z if z < 1, else x = z (matching SSE logic)
        uint32x4_t z_lt_one = vcltq_f32(z, fones);
        float32x4_t z_recip = vdivq_f32(fones, z);
        x = vbslq_f32(z_lt_one, z_recip, z);

        // Two iterations: x = x + sqrt(1 + x*x)
        for (i = 0; i < 2; i++) {
            x = vaddq_f32(x, vsqrtq_f32(vfmaq_f32(fones, x, x)));
        }

        // x = 1/x
        x = vdivq_f32(fones, x);

        // Taylor series
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            float coeff = (j % 2 == 0) ? 1.0f / (2 * j + 1) : -1.0f / (2 * j + 1);
            y = vfmaq_f32(vdupq_n_f32(coeff), y, vmulq_f32(x, x));
        }

        y = vmulq_f32(y, vmulq_f32(x, ffours));

        // Adjust if z > 1: y = y + (pio2 - 2*y)
        uint32x4_t z_gt_one = vcgtq_f32(z, fones);
        float32x4_t y_adj = vfmsq_f32(pio2, y, ftwos);
        y = vbslq_f32(z_gt_one, vaddq_f32(y, y_adj), y);

        arcsine = y;

        // If tanVal < 0, arcsine = -arcsine
        uint32x4_t tanVal_neg = vcltq_f32(tanVal, fzeroes);
        arcsine = vbslq_f32(tanVal_neg, vnegq_f32(arcsine), arcsine);

        vst1q_f32(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void
volk_32f_asin_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t cpio2 = __riscv_vfmv_v_f_f32m2(1.5707964f, vlmax);
    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf2 = __riscv_vfmv_v_f_f32m2(2.0f, vlmax);
    const vfloat32m2_t cf4 = __riscv_vfmv_v_f_f32m2(4.0f, vlmax);

#if ASIN_TERMS == 2
    const vfloat32m2_t cfm1o3 = __riscv_vfmv_v_f_f32m2(-1 / 3.0f, vlmax);
#elif ASIN_TERMS == 3
    const vfloat32m2_t cf1o5 = __riscv_vfmv_v_f_f32m2(1 / 5.0f, vlmax);
#elif ASIN_TERMS == 4
    const vfloat32m2_t cfm1o7 = __riscv_vfmv_v_f_f32m2(-1 / 7.0f, vlmax);
#endif

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        // Compute 1 - v^2 = (1+v)*(1-v) for better numerical stability
        // For asin: a = v / sqrt(1 - v^2)  (inverse of acos)
        vfloat32m2_t one_minus_v_sq =
            __riscv_vfmul(__riscv_vfadd(cf1, v, vl), __riscv_vfsub(cf1, v, vl), vl);
        vfloat32m2_t a = __riscv_vfdiv(v, __riscv_vfsqrt(one_minus_v_sq, vl), vl);
        vfloat32m2_t z = __riscv_vfabs(a, vl);
        vfloat32m2_t x = __riscv_vfdiv_mu(__riscv_vmflt(z, cf1, vl), z, cf1, z, vl);
        x = __riscv_vfadd(x, __riscv_vfsqrt(__riscv_vfmadd(x, x, cf1, vl), vl), vl);
        x = __riscv_vfadd(x, __riscv_vfsqrt(__riscv_vfmadd(x, x, cf1, vl), vl), vl);
        x = __riscv_vfdiv(cf1, x, vl);
        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);

#if ASIN_TERMS < 1
        vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(0, vl);
#elif ASIN_TERMS == 1
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 2
        vfloat32m2_t y = cfm1o3;
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 3
        vfloat32m2_t y = cf1o5;
        y = __riscv_vfmadd(y, xx, cfm1o3, vl);
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 4
        vfloat32m2_t y = cfm1o7;
        y = __riscv_vfmadd(y, xx, cf1o5, vl);
        y = __riscv_vfmadd(y, xx, cfm1o3, vl);
        y = __riscv_vfmadd(y, xx, cf1, vl);
#else
#error "ASIN_TERMS > 4 not supported by volk_32f_asin_32f_rvv"
#endif
        y = __riscv_vfmul(y, __riscv_vfmul(x, cf4, vl), vl);
        y = __riscv_vfadd_mu(
            __riscv_vmfgt(z, cf1, vl), y, y, __riscv_vfnmsub(y, cf2, cpio2, vl), vl);

        vfloat32m2_t asine;
        asine = __riscv_vfneg_mu(RISCV_VMFLTZ(32m2, a, vl), y, y, vl);

        __riscv_vse32(bVector, asine, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_asin_32f_u_H */
