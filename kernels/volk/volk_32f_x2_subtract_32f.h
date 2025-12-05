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


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_subtract_32f_neonv8(float* cVector,
                                                   const float* aVector,
                                                   const float* bVector,
                                                   unsigned int num_points)
{
    unsigned int n = num_points;
    float* c = cVector;
    const float* a = aVector;
    const float* b = bVector;

    /* Process 8 floats per iteration (2x unroll) */
    while (n >= 8) {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b);
        float32x4_t b1 = vld1q_f32(b + 4);
        __VOLK_PREFETCH(a + 16);
        __VOLK_PREFETCH(b + 16);

        vst1q_f32(c, vsubq_f32(a0, b0));
        vst1q_f32(c + 4, vsubq_f32(a1, b1));

        a += 8;
        b += 8;
        c += 8;
        n -= 8;
    }

    /* Process remaining 4 floats */
    if (n >= 4) {
        vst1q_f32(c, vsubq_f32(vld1q_f32(a), vld1q_f32(b)));
        a += 4;
        b += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = *a++ - *b++;
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */


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

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_subtract_32f_rvv(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(aVector, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(bVector, vl);
        __riscv_vse32(cVector, __riscv_vfsub(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_x2_subtract_32f_u_H */
