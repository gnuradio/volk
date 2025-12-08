/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64f_x2_multiply_64f
 *
 * \b Overview
 *
 * Multiplies two input double-precision floating point vectors together.
 *
 * c[i] = a[i] * b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_x2_multiply_64f(float* cVector, const float* aVector, const float*
 * bVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: First input vector.
 * \li bVector: Second input vector.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * Multiply elements of an increasing vector by those of a decreasing vector.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   double* increasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* decreasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* out = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *       decreasing[ii] = 10.f - (float)ii;
 *   }
 *
 *   volk_64f_x2_multiply_64f(out, increasing, decreasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2F\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(decreasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_64f_x2_multiply_64f_H
#define INCLUDED_volk_64f_x2_multiply_64f_H

#include <inttypes.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_64f_x2_multiply_64f_generic(double* cVector,
                                                    const double* aVector,
                                                    const double* bVector,
                                                    unsigned int num_points)
{
    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64f_x2_multiply_64f_neonv8(double* cVector,
                                                   const double* aVector,
                                                   const double* bVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    for (; number < quarter_points; number++) {
        float64x2_t aVal0 = vld1q_f64(aPtr);
        float64x2_t aVal1 = vld1q_f64(aPtr + 2);
        float64x2_t bVal0 = vld1q_f64(bPtr);
        float64x2_t bVal1 = vld1q_f64(bPtr + 2);
        __VOLK_PREFETCH(aPtr + 4);
        __VOLK_PREFETCH(bPtr + 4);

        float64x2_t cVal0 = vmulq_f64(aVal0, bVal0);
        float64x2_t cVal1 = vmulq_f64(aVal1, bVal1);

        vst1q_f64(cPtr, cVal0);
        vst1q_f64(cPtr + 2, cVal1);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */


/*
 * Unaligned versions
 */

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_64f_x2_multiply_64f_u_sse2(double* cVector,
                                                   const double* aVector,
                                                   const double* bVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int half_points = num_points / 2;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m128d aVal, bVal, cVal;
    for (; number < half_points; number++) {
        aVal = _mm_loadu_pd(aPtr);
        bVal = _mm_loadu_pd(bPtr);

        cVal = _mm_mul_pd(aVal, bVal);

        _mm_storeu_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = half_points * 2;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_64f_x2_multiply_64f_u_avx(double* cVector,
                                                  const double* aVector,
                                                  const double* bVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarter_points; number++) {

        aVal = _mm256_loadu_pd(aPtr);
        bVal = _mm256_loadu_pd(bPtr);

        cVal = _mm256_mul_pd(aVal, bVal);

        _mm256_storeu_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

/*
 * Aligned versions
 */

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_64f_x2_multiply_64f_a_sse2(double* cVector,
                                                   const double* aVector,
                                                   const double* bVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int half_points = num_points / 2;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m128d aVal, bVal, cVal;
    for (; number < half_points; number++) {
        aVal = _mm_load_pd(aPtr);
        bVal = _mm_load_pd(bPtr);

        cVal = _mm_mul_pd(aVal, bVal);

        _mm_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = half_points * 2;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_64f_x2_multiply_64f_a_avx(double* cVector,
                                                  const double* aVector,
                                                  const double* bVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarter_points; number++) {

        aVal = _mm256_load_pd(aPtr);
        bVal = _mm256_load_pd(bPtr);

        cVal = _mm256_mul_pd(aVal, bVal);

        _mm256_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_64f_x2_multiply_64f_rvv(double* cVector,
                                                const double* aVector,
                                                const double* bVector,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t va = __riscv_vle64_v_f64m8(aVector, vl);
        vfloat64m8_t vb = __riscv_vle64_v_f64m8(bVector, vl);
        __riscv_vse64(cVector, __riscv_vfmul(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_64f_x2_multiply_64f_u_H */
