/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_subtract_32f
 *
 * \b Overview
 *
 * Subtracts values in bVector from values in aVector.
 *
 * c[i] = a[i] - b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_subtract_32f(float* cVector, const float* aVector, const float*
 * bVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The initial vector.
 * \li bVector: The vector to be subtracted.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li complexVector: The output vector.
 *
 * \b Example
 * Subtract and increasing vector from a decreasing vector.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* decreasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *       decreasing[ii] = 10.f - (float)ii;
 *   }
 *
 *   volk_32f_x2_subtract_32f(out, increasing, decreasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(decreasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_subtract_32f_a_H
#define INCLUDED_volk_32f_x2_subtract_32f_a_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_subtract_32f_generic(float* cVector,
                                                    const float* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        *cVector++ = (*aVector++) - (*bVector++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_subtract_32f_a_avx512f(float* cVector,
                                                      const float* aVector,
                                                      const float* bVector,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aVector);
        __m512 bVal = _mm512_load_ps(bVector);

        __m512 cVal = _mm512_sub_ps(aVal, bVal);

        _mm512_store_ps(cVector, cVal); // Store the results back into the C container

        aVector += 16;
        bVector += 16;
        cVector += 16;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_subtract_32f_a_avx(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aVector);
        __m256 bVal = _mm256_load_ps(bVector);

        __m256 cVal = _mm256_sub_ps(aVal, bVal);

        _mm256_store_ps(cVector, cVal); // Store the results back into the C container

        aVector += 8;
        bVector += 8;
        cVector += 8;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_subtract_32f_a_sse(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aVector);
        __m128 bVal = _mm_load_ps(bVector);

        __m128 cVal = _mm_sub_ps(aVal, bVal);

        _mm_store_ps(cVector, cVal); // Store the results back into the C container

        aVector += 4;
        bVector += 4;
        cVector += 4;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_subtract_32f_neon(float* cVector,
                                                 const float* aVector,
                                                 const float* bVector,
                                                 unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    for (unsigned int number = 0; number < quarterPoints; number++) {
        float32x4_t a_vec = vld1q_f32(aVector);
        float32x4_t b_vec = vld1q_f32(bVector);

        float32x4_t c_vec = vsubq_f32(a_vec, b_vec);

        vst1q_f32(cVector, c_vec);

        aVector += 4;
        bVector += 4;
        cVector += 4;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_ORC
extern void volk_32f_x2_subtract_32f_a_orc_impl(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                int num_points);

static inline void volk_32f_x2_subtract_32f_u_orc(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    volk_32f_x2_subtract_32f_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32f_x2_subtract_32f_a_H */


#ifndef INCLUDED_volk_32f_x2_subtract_32f_u_H
#define INCLUDED_volk_32f_x2_subtract_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_subtract_32f_u_avx512f(float* cVector,
                                                      const float* aVector,
                                                      const float* bVector,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aVector);
        __m512 bVal = _mm512_loadu_ps(bVector);

        __m512 cVal = _mm512_sub_ps(aVal, bVal);

        _mm512_storeu_ps(cVector, cVal); // Store the results back into the C container

        aVector += 16;
        bVector += 16;
        cVector += 16;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_subtract_32f_u_avx(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aVector);
        __m256 bVal = _mm256_loadu_ps(bVector);

        __m256 cVal = _mm256_sub_ps(aVal, bVal);

        _mm256_storeu_ps(cVector, cVal); // Store the results back into the C container

        aVector += 8;
        bVector += 8;
        cVector += 8;
    }

    volk_32f_x2_subtract_32f_generic(
        cVector, aVector, bVector, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_x2_subtract_32f_u_H */
