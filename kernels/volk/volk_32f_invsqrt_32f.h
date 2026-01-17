/* -*- c++ -*- */
/*
 * Copyright 2013, 2014 Free Software Foundation, Inc.
 * Copyright 2026 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_invsqrt_32f
 *
 * \b Overview
 *
 * Computes the inverse square root of the input vector and stores
 * result in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_invsqrt_32f(float* cVector, const float* aVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li aVector: the input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = 1.0 / (float)(ii*ii);
 *   }
 *
 *   volk_32f_invsqrt_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_invsqrt_32f_a_H
#define INCLUDED_volk_32f_invsqrt_32f_a_H

#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_invsqrt_32f_generic(float* cVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        cVector[number] = 1.0f / sqrtf(aVector[number]);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32f_invsqrt_32f_a_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);
        __m256 cVal = _mm256_rsqrt_nr_ps(aVal);
        _mm256_store_ps(cPtr, cVal);
        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_invsqrt_32f_a_avx512f(float* cVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);
        __m512 cVal = _mm512_rsqrt_nr_ps(aVal);
        _mm512_store_ps(cPtr, cVal);
        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void
volk_32f_invsqrt_32f_a_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aPtr);
        __m128 cVal = _mm_rsqrt_nr_ps(aVal);
        _mm_store_ps(cPtr, cVal);
        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_invsqrt_32f_neon(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number;
    const unsigned int quarter_points = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (number = 0; number < quarter_points; ++number) {
        float32x4_t a_val = vld1q_f32(aPtr);
        float32x4_t c_val = _vinvsqrtq_f32(a_val);
        vst1q_f32(cPtr, c_val);
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_invsqrt_32f_neonv8(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number;
    const unsigned int quarter_points = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    // Process 4 elements at a time (1 vector)
    for (number = 0; number < quarter_points; ++number) {
        float32x4_t a = vld1q_f32(aPtr);
        float32x4_t x = vrsqrteq_f32(a);

        // Two Newton-Raphson iterations for float32 accuracy
        x = vmulq_f32(x, vrsqrtsq_f32(vmulq_f32(a, x), x));
        x = vmulq_f32(x, vrsqrtsq_f32(vmulq_f32(a, x), x));

        vst1q_f32(cPtr, x);
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void
volk_32f_invsqrt_32f_u_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_loadu_ps(aPtr);
        __m128 cVal = _mm_rsqrt_nr_ps(aVal);
        _mm_storeu_ps(cPtr, cVal);
        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32f_invsqrt_32f_u_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);
        __m256 cVal = _mm256_rsqrt_nr_ps(aVal);
        _mm256_storeu_ps(cPtr, cVal);
        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_invsqrt_32f_u_avx512f(float* cVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);
        __m512 cVal = _mm512_rsqrt_nr_ps(aVal);
        _mm512_storeu_ps(cPtr, cVal);
        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_invsqrt_32f_rvv(float* cVector, const float* aVector, unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t a = __riscv_vle32_v_f32m8(aVector, vl);
        vfloat32m8_t half = __riscv_vfmv_v_f_f32m8(0.5f, vl);
        vfloat32m8_t three_halfs = __riscv_vfmv_v_f_f32m8(1.5f, vl);
        // Initial estimate (~7-bit precision)
        vfloat32m8_t x = __riscv_vfrsqrt7(a, vl);
        // Two Newton-Raphson iterations: x = x * (1.5 - 0.5 * a * x * x)
        vfloat32m8_t half_a = __riscv_vfmul(half, a, vl);
        x = __riscv_vfmul(
            x, __riscv_vfnmsac(three_halfs, half_a, __riscv_vfmul(x, x, vl), vl), vl);
        x = __riscv_vfmul(
            x, __riscv_vfnmsac(three_halfs, half_a, __riscv_vfmul(x, x, vl), vl), vl);
        __riscv_vse32(cVector, x, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_invsqrt_32f_a_H */
