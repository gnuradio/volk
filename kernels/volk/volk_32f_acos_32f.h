/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_acos_32f
 *
 * \b Overview
 *
 * Computes arccosine of the input vector and stores results in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_acos_32f(float* bVector, const float* aVector, unsigned int num_points)
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
 * Calculate common angles around the top half of the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 1;
 *   in[1] = std::sqrt(3.f)/2.f;
 *   in[2] = std::sqrt(2.f)/2.f;
 *   in[3] = 0.5;
 *   in[4] = in[5] = 0;
 *   for(unsigned int ii = 6; ii < N; ++ii){
 *       in[ii] = - in[N-ii-1];
 *   }
 *
 *   volk_32f_acos_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("acos(%1.3f) = %1.3f\n", in[ii], out[ii]);
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
#define ACOS_TERMS 2

#ifndef INCLUDED_volk_32f_acos_32f_a_H
#define INCLUDED_volk_32f_acos_32f_a_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_acos_32f_a_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, d, pi, pio2, x, y, z, arccosine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm256_set1_ps(3.14159265358979323846);
    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        d = aVal;
        aVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))),
                             aVal);
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arccosine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_sub_ps(
            arccosine, _mm256_and_ps(_mm256_mul_ps(arccosine, ftwos), condition));
        condition = _mm256_cmp_ps(d, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_add_ps(arccosine, _mm256_and_ps(pi, condition));

        _mm256_store_ps(bPtr, arccosine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = acos(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_acos_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, d, pi, pio2, x, y, z, arccosine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm256_set1_ps(3.14159265358979323846);
    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        d = aVal;
        aVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))),
                             aVal);
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arccosine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_sub_ps(
            arccosine, _mm256_and_ps(_mm256_mul_ps(arccosine, ftwos), condition));
        condition = _mm256_cmp_ps(d, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_add_ps(arccosine, _mm256_and_ps(pi, condition));

        _mm256_store_ps(bPtr, arccosine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = acos(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_acos_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, d, pi, pio2, x, y, z, arccosine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm_set1_ps(3.14159265358979323846);
    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        d = aVal;
        aVal = _mm_div_ps(
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))),
            aVal);
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        x = _mm_div_ps(fones, x);
        y = fzeroes;
        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arccosine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arccosine =
            _mm_sub_ps(arccosine, _mm_and_ps(_mm_mul_ps(arccosine, ftwos), condition));
        condition = _mm_cmplt_ps(d, fzeroes);
        arccosine = _mm_add_ps(arccosine, _mm_and_ps(pi, condition));

        _mm_store_ps(bPtr, arccosine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = acosf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_acos_32f_a_H */


#ifndef INCLUDED_volk_32f_acos_32f_u_H
#define INCLUDED_volk_32f_acos_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_acos_32f_u_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, d, pi, pio2, x, y, z, arccosine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm256_set1_ps(3.14159265358979323846);
    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        d = aVal;
        aVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))),
                             aVal);
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arccosine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_sub_ps(
            arccosine, _mm256_and_ps(_mm256_mul_ps(arccosine, ftwos), condition));
        condition = _mm256_cmp_ps(d, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_add_ps(arccosine, _mm256_and_ps(pi, condition));

        _mm256_storeu_ps(bPtr, arccosine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = acos(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_acos_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, d, pi, pio2, x, y, z, arccosine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm256_set1_ps(3.14159265358979323846);
    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        d = aVal;
        aVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))),
                             aVal);
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arccosine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_sub_ps(
            arccosine, _mm256_and_ps(_mm256_mul_ps(arccosine, ftwos), condition));
        condition = _mm256_cmp_ps(d, fzeroes, _CMP_LT_OS);
        arccosine = _mm256_add_ps(arccosine, _mm256_and_ps(pi, condition));

        _mm256_storeu_ps(bPtr, arccosine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = acos(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_acos_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, d, pi, pio2, x, y, z, arccosine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pi = _mm_set1_ps(3.14159265358979323846);
    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        d = aVal;
        aVal = _mm_div_ps(
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))),
            aVal);
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++)
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        x = _mm_div_ps(fones, x);
        y = fzeroes;

        for (j = ACOS_TERMS - 1; j >= 0; j--)
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arccosine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arccosine =
            _mm_sub_ps(arccosine, _mm_and_ps(_mm_mul_ps(arccosine, ftwos), condition));
        condition = _mm_cmplt_ps(d, fzeroes);
        arccosine = _mm_add_ps(arccosine, _mm_and_ps(pi, condition));

        _mm_storeu_ps(bPtr, arccosine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = acosf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_acos_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = acosf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_acos_32f_u_H */
