/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_32f_multiply_32fc
 *
 * \b Overview
 *
 * Multiplies a complex vector by a floating point vector and returns
 * the complex result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_32f_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * float* bVector, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of complex floats.
 * \li bVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector complex floats.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32fc_32f_multiply_32fc();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_32f_multiply_32fc_a_H
#define INCLUDED_volk_32fc_32f_multiply_32fc_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_32f_multiply_32fc_a_avx(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float* bVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal1, aVal2, bVal, bVal1, bVal2, cVal1, cVal2;

    __m256i permute_mask = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);

    for (; number < eighthPoints; number++) {

        aVal1 = _mm256_load_ps((float*)aPtr);
        aPtr += 4;

        aVal2 = _mm256_load_ps((float*)aPtr);
        aPtr += 4;

        bVal = _mm256_load_ps(bPtr); // b0|b1|b2|b3|b4|b5|b6|b7
        bPtr += 8;

        bVal1 = _mm256_permute2f128_ps(bVal, bVal, 0x00); // b0|b1|b2|b3|b0|b1|b2|b3
        bVal2 = _mm256_permute2f128_ps(bVal, bVal, 0x11); // b4|b5|b6|b7|b4|b5|b6|b7

        bVal1 = _mm256_permutevar_ps(bVal1, permute_mask); // b0|b0|b1|b1|b2|b2|b3|b3
        bVal2 = _mm256_permutevar_ps(bVal2, permute_mask); // b4|b4|b5|b5|b6|b6|b7|b7

        cVal1 = _mm256_mul_ps(aVal1, bVal1);
        cVal2 = _mm256_mul_ps(aVal2, bVal2);

        _mm256_store_ps((float*)cPtr,
                        cVal1); // Store the results back into the C container
        cPtr += 4;

        _mm256_store_ps((float*)cPtr,
                        cVal2); // Store the results back into the C container
        cPtr += 4;
    }

    number = eighthPoints * 8;
    for (; number < num_points; ++number) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_32f_multiply_32fc_a_sse(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float* bVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;

    __m128 aVal1, aVal2, bVal, bVal1, bVal2, cVal;
    for (; number < quarterPoints; number++) {

        aVal1 = _mm_load_ps((const float*)aPtr);
        aPtr += 2;

        aVal2 = _mm_load_ps((const float*)aPtr);
        aPtr += 2;

        bVal = _mm_load_ps(bPtr);
        bPtr += 4;

        bVal1 = _mm_shuffle_ps(bVal, bVal, _MM_SHUFFLE(1, 1, 0, 0));
        bVal2 = _mm_shuffle_ps(bVal, bVal, _MM_SHUFFLE(3, 3, 2, 2));

        cVal = _mm_mul_ps(aVal1, bVal1);

        _mm_store_ps((float*)cPtr, cVal); // Store the results back into the C container
        cPtr += 2;

        cVal = _mm_mul_ps(aVal2, bVal2);

        _mm_store_ps((float*)cPtr, cVal); // Store the results back into the C container

        cPtr += 2;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr);
        bPtr++;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_32f_multiply_32fc_generic(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const float* bVector,
                                                       unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_32f_multiply_32fc_neon(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    float32x4x2_t inputVector, outputVector;
    float32x4_t tapsVector;
    for (number = 0; number < quarter_points; number++) {
        inputVector = vld2q_f32((float*)aPtr);
        tapsVector = vld1q_f32(bPtr);

        outputVector.val[0] = vmulq_f32(inputVector.val[0], tapsVector);
        outputVector.val[1] = vmulq_f32(inputVector.val[1], tapsVector);

        vst2q_f32((float*)cPtr, outputVector);
        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_32f_multiply_32fc_neonv8(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const float* bVector,
                                                      unsigned int num_points)
{
    unsigned int n = num_points;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const float* b = bVector;

    /* Process 8 complex numbers per iteration (2x unroll) */
    while (n >= 8) {
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4x2_t a1 = vld2q_f32((const float*)(a + 4));
        float32x4_t b0 = vld1q_f32(b);
        float32x4_t b1 = vld1q_f32(b + 4);
        __VOLK_PREFETCH(a + 8);
        __VOLK_PREFETCH(b + 8);

        /* Complex × real: just scale both real and imag parts */
        float32x4x2_t c0, c1;
        c0.val[0] = vmulq_f32(a0.val[0], b0);
        c0.val[1] = vmulq_f32(a0.val[1], b0);
        c1.val[0] = vmulq_f32(a1.val[0], b1);
        c1.val[1] = vmulq_f32(a1.val[1], b1);

        vst2q_f32((float*)c, c0);
        vst2q_f32((float*)(c + 4), c1);

        a += 8;
        b += 8;
        c += 8;
        n -= 8;
    }

    /* Process remaining 4 */
    if (n >= 4) {
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4_t b0 = vld1q_f32(b);
        float32x4x2_t c0;
        c0.val[0] = vmulq_f32(a0.val[0], b0);
        c0.val[1] = vmulq_f32(a0.val[1], b0);
        vst2q_f32((float*)c, c0);
        a += 4;
        b += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = (*a++) * (*b++);
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_ORC

extern void volk_32fc_32f_multiply_32fc_a_orc_impl(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   const float* bVector,
                                                   int num_points);

static inline void volk_32fc_32f_multiply_32fc_u_orc(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float* bVector,
                                                     unsigned int num_points)
{
    volk_32fc_32f_multiply_32fc_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_32f_multiply_32fc_rvv(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   const float* bVector,
                                                   unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, cVector += vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m8_t vc = __riscv_vle32_v_f32m8((const float*)aVector, vl * 2);
        vuint32m4_t v = __riscv_vle32_v_u32m4((const uint32_t*)bVector, vl);
        vfloat32m8_t vf = __riscv_vreinterpret_f32m8(__riscv_vreinterpret_u32m8(
            __riscv_vwmaccu(__riscv_vwaddu_vv(v, v, vl), 0xFFFFFFFF, v, vl)));
        __riscv_vse32((float*)cVector, __riscv_vfmul(vc, vf, vl * 2), vl * 2);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_32f_multiply_32fc_a_H */
