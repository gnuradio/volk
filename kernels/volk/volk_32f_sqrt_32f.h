/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_sqrt_32f
 *
 * \b Overview
 *
 * Computes the square root of the input vector and stores the results
 * in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_sqrt_32f(float* cVector, const float* aVector, unsigned int num_points)
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
 * \code
    int N = 10;
    unsigned int alignment = volk_get_alignment();
    float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
    float* out = (float*)volk_malloc(sizeof(float)*N, alignment);

    for(unsigned int ii = 0; ii < N; ++ii){
        in[ii] = (float)(ii*ii);
    }

    volk_32f_sqrt_32f(out, in, N);

    for(unsigned int ii = 0; ii < N; ++ii){
        printf("out(%i) = %f\n", ii, out[ii]);
    }

    volk_free(in);
    volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_sqrt_32f_a_H
#define INCLUDED_volk_32f_sqrt_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_sqrt_32f_generic(float* cVector, const float* aVector, unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_sqrt_32f_a_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, cVal;
    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);

        cVal = _mm_sqrt_ps(aVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE */

#if LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_a_avx512(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m512 aVal, cVal;
    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);
        cVal = _mm512_sqrt_ps(aVal);
        _mm512_store_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_a_avx2(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        cVal = _mm256_sqrt_ps(aVal);
        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_a_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);

        cVal = _mm256_sqrt_ps(aVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_sqrt_32f_neon(float* cVector, const float* aVector, unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    float32x4_t in_vec, out_vec;

    for (number = 0; number < quarter_points; number++) {
        in_vec = vld1q_f32(aPtr);
        // note that armv8 has vsqrt_f32 which will be much better
        out_vec = vrecpeq_f32(vrsqrteq_f32(in_vec));
        vst1q_f32(cPtr, out_vec);
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_sqrt_32f_neonv8(float* cVector, const float* aVector, unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    for (number = 0; number < quarter_points; number++) {
        float32x4_t in_vec = vld1q_f32(aPtr);
        float32x4_t out_vec = vsqrtq_f32(in_vec);
        vst1q_f32(cPtr, out_vec);
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_sqrt_32f_a_H */

#ifndef INCLUDED_volk_32f_sqrt_32f_u_H
#define INCLUDED_volk_32f_sqrt_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_sqrt_32f_u_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, cVal;
    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        cVal = _mm_sqrt_ps(aVal);
        _mm_storeu_ps(cPtr, cVal);

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE */


#if LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_u_avx512(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m512 aVal, cVal;
    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);
        cVal = _mm512_sqrt_ps(aVal);
        _mm512_storeu_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_u_avx2(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        cVal = _mm256_sqrt_ps(aVal);
        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_sqrt_32f_u_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);

        cVal = _mm256_sqrt_ps(aVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = sqrtf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_sqrt_32f_rvv(float* cVector, const float* aVector, unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(aVector, vl);
        __riscv_vse32(cVector, __riscv_vfsqrt(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_sqrt_32f_u_H */
