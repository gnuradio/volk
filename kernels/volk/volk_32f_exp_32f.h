/* -*- c++ -*- */
/*
 * Copyright 2015-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* SIMD (SSE4) implementation of exp
   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

/*!
 * \page volk_32f_exp_32f
 *
 * \b Overview
 *
 * Computes exponential of input vector and stores results in output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_exp_32f(float* bVector, const float* aVector, unsigned int num_points)
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
 *   volk_32f_exp_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("exp(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_exp_32f_a_H
#define INCLUDED_volk_32f_exp_32f_a_H

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_exp_32f_a_sse2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Declare variables and constants
    __m128 aVal, bVal, tmp, fx, mask, pow2n, z, y;
    __m128 one, exp_hi, exp_lo, log2EF, half, exp_C1, exp_C2;
    __m128 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m128i emm0, pi32_0x7f;

    one = _mm_set1_ps(1.0);
    exp_hi = _mm_set1_ps(88.3762626647949);
    exp_lo = _mm_set1_ps(-88.3762626647949);
    log2EF = _mm_set1_ps(1.44269504088896341);
    half = _mm_set1_ps(0.5);
    exp_C1 = _mm_set1_ps(0.693359375);
    exp_C2 = _mm_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm_set1_epi32(0x7f);

    exp_p0 = _mm_set1_ps(1.9875691500e-4);
    exp_p1 = _mm_set1_ps(1.3981999507e-3);
    exp_p2 = _mm_set1_ps(8.3334519073e-3);
    exp_p3 = _mm_set1_ps(4.1665795894e-2);
    exp_p4 = _mm_set1_ps(1.6666665459e-1);
    exp_p5 = _mm_set1_ps(5.0000001201e-1);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        tmp = _mm_setzero_ps();

        aVal = _mm_max_ps(_mm_min_ps(aVal, exp_hi), exp_lo);

        /* express exp(x) as exp(g + n*log(2)) */
        fx = _mm_add_ps(_mm_mul_ps(aVal, log2EF), half);

        emm0 = _mm_cvttps_epi32(fx);
        tmp = _mm_cvtepi32_ps(emm0);

        mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        z = _mm_mul_ps(fx, exp_C2);
        aVal = _mm_sub_ps(_mm_sub_ps(aVal, tmp), z);
        z = _mm_mul_ps(aVal, aVal);

        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, aVal), exp_p1), aVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), aVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, aVal), exp_p4), aVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), aVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm_castsi128_ps(emm0);
        bVal = _mm_mul_ps(y, pow2n);

        _mm_store_ps(bPtr, bVal);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE2 for aligned */


#endif /* INCLUDED_volk_32f_exp_32f_a_H */

#ifndef INCLUDED_volk_32f_exp_32f_u_H
#define INCLUDED_volk_32f_exp_32f_u_H

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_exp_32f_u_sse2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Declare variables and constants
    __m128 aVal, bVal, tmp, fx, mask, pow2n, z, y;
    __m128 one, exp_hi, exp_lo, log2EF, half, exp_C1, exp_C2;
    __m128 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m128i emm0, pi32_0x7f;

    one = _mm_set1_ps(1.0);
    exp_hi = _mm_set1_ps(88.3762626647949);
    exp_lo = _mm_set1_ps(-88.3762626647949);
    log2EF = _mm_set1_ps(1.44269504088896341);
    half = _mm_set1_ps(0.5);
    exp_C1 = _mm_set1_ps(0.693359375);
    exp_C2 = _mm_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm_set1_epi32(0x7f);

    exp_p0 = _mm_set1_ps(1.9875691500e-4);
    exp_p1 = _mm_set1_ps(1.3981999507e-3);
    exp_p2 = _mm_set1_ps(8.3334519073e-3);
    exp_p3 = _mm_set1_ps(4.1665795894e-2);
    exp_p4 = _mm_set1_ps(1.6666665459e-1);
    exp_p5 = _mm_set1_ps(5.0000001201e-1);


    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        tmp = _mm_setzero_ps();

        aVal = _mm_max_ps(_mm_min_ps(aVal, exp_hi), exp_lo);

        /* express exp(x) as exp(g + n*log(2)) */
        fx = _mm_add_ps(_mm_mul_ps(aVal, log2EF), half);

        emm0 = _mm_cvttps_epi32(fx);
        tmp = _mm_cvtepi32_ps(emm0);

        mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        z = _mm_mul_ps(fx, exp_C2);
        aVal = _mm_sub_ps(_mm_sub_ps(aVal, tmp), z);
        z = _mm_mul_ps(aVal, aVal);

        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, aVal), exp_p1), aVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), aVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, aVal), exp_p4), aVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), aVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm_castsi128_ps(emm0);
        bVal = _mm_mul_ps(y, pow2n);

        _mm_storeu_ps(bPtr, bVal);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE2 for unaligned */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_exp_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_exp_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Constants
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);

    float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    for (; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(aPtr);

        // Clamp to valid range
        aVal = vmaxq_f32(vminq_f32(aVal, exp_hi), exp_lo);

        // express exp(x) as exp(g + n*log(2))
        float32x4_t fx = vmlaq_f32(half, aVal, log2EF);

        // Floor function
        int32x4_t emm0 = vcvtq_s32_f32(fx);
        float32x4_t tmp = vcvtq_f32_s32(emm0);

        // If tmp > fx, subtract 1 (floor correction)
        uint32x4_t mask = vcgtq_f32(tmp, fx);
        float32x4_t mask_one = vbslq_f32(mask, one, vdupq_n_f32(0.0f));
        fx = vsubq_f32(tmp, mask_one);

        // Reduce x
        tmp = vmulq_f32(fx, exp_C1);
        float32x4_t z = vmulq_f32(fx, exp_C2);
        aVal = vsubq_f32(vsubq_f32(aVal, tmp), z);
        z = vmulq_f32(aVal, aVal);

        // Polynomial approximation
        float32x4_t y = vmlaq_f32(exp_p1, exp_p0, aVal);
        y = vmulq_f32(y, aVal);
        y = vaddq_f32(y, exp_p2);
        y = vmulq_f32(y, aVal);
        y = vaddq_f32(y, exp_p3);
        y = vmlaq_f32(exp_p4, y, aVal);
        y = vmulq_f32(y, aVal);
        y = vaddq_f32(y, exp_p5);
        y = vmlaq_f32(aVal, y, z);
        y = vaddq_f32(y, one);

        // Build 2^n
        emm0 = vcvtq_s32_f32(fx);
        emm0 = vaddq_s32(emm0, pi32_0x7f);
        emm0 = vshlq_n_s32(emm0, 23);
        float32x4_t pow2n = vreinterpretq_f32_s32(emm0);

        float32x4_t bVal = vmulq_f32(y, pow2n);
        vst1q_f32(bPtr, bVal);

        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_exp_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);

    float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(aPtr + 16);

        float32x4_t aVal0 = vld1q_f32(aPtr);
        float32x4_t aVal1 = vld1q_f32(aPtr + 4);

        // Clamp to valid range
        aVal0 = vmaxq_f32(vminq_f32(aVal0, exp_hi), exp_lo);
        aVal1 = vmaxq_f32(vminq_f32(aVal1, exp_hi), exp_lo);

        // express exp(x) as exp(g + n*log(2))
        float32x4_t fx0 = vfmaq_f32(half, aVal0, log2EF);
        float32x4_t fx1 = vfmaq_f32(half, aVal1, log2EF);

        // Floor function
        int32x4_t emm0_0 = vcvtq_s32_f32(fx0);
        int32x4_t emm0_1 = vcvtq_s32_f32(fx1);
        float32x4_t tmp0 = vcvtq_f32_s32(emm0_0);
        float32x4_t tmp1 = vcvtq_f32_s32(emm0_1);

        // If tmp > fx, subtract 1 (floor correction)
        uint32x4_t mask0 = vcgtq_f32(tmp0, fx0);
        uint32x4_t mask1 = vcgtq_f32(tmp1, fx1);
        float32x4_t mask_one0 = vbslq_f32(mask0, one, vdupq_n_f32(0.0f));
        float32x4_t mask_one1 = vbslq_f32(mask1, one, vdupq_n_f32(0.0f));
        fx0 = vsubq_f32(tmp0, mask_one0);
        fx1 = vsubq_f32(tmp1, mask_one1);

        // Reduce x
        tmp0 = vmulq_f32(fx0, exp_C1);
        tmp1 = vmulq_f32(fx1, exp_C1);
        float32x4_t z0 = vmulq_f32(fx0, exp_C2);
        float32x4_t z1 = vmulq_f32(fx1, exp_C2);
        aVal0 = vsubq_f32(vsubq_f32(aVal0, tmp0), z0);
        aVal1 = vsubq_f32(vsubq_f32(aVal1, tmp1), z1);
        z0 = vmulq_f32(aVal0, aVal0);
        z1 = vmulq_f32(aVal1, aVal1);

        // Polynomial approximation using FMA
        float32x4_t y0 = vfmaq_f32(exp_p1, exp_p0, aVal0);
        float32x4_t y1 = vfmaq_f32(exp_p1, exp_p0, aVal1);
        y0 = vmulq_f32(y0, aVal0);
        y1 = vmulq_f32(y1, aVal1);
        y0 = vaddq_f32(y0, exp_p2);
        y1 = vaddq_f32(y1, exp_p2);
        y0 = vmulq_f32(y0, aVal0);
        y1 = vmulq_f32(y1, aVal1);
        y0 = vaddq_f32(y0, exp_p3);
        y1 = vaddq_f32(y1, exp_p3);
        y0 = vfmaq_f32(exp_p4, y0, aVal0);
        y1 = vfmaq_f32(exp_p4, y1, aVal1);
        y0 = vmulq_f32(y0, aVal0);
        y1 = vmulq_f32(y1, aVal1);
        y0 = vaddq_f32(y0, exp_p5);
        y1 = vaddq_f32(y1, exp_p5);
        y0 = vfmaq_f32(aVal0, y0, z0);
        y1 = vfmaq_f32(aVal1, y1, z1);
        y0 = vaddq_f32(y0, one);
        y1 = vaddq_f32(y1, one);

        // Build 2^n
        emm0_0 = vcvtq_s32_f32(fx0);
        emm0_1 = vcvtq_s32_f32(fx1);
        emm0_0 = vaddq_s32(emm0_0, pi32_0x7f);
        emm0_1 = vaddq_s32(emm0_1, pi32_0x7f);
        emm0_0 = vshlq_n_s32(emm0_0, 23);
        emm0_1 = vshlq_n_s32(emm0_1, 23);
        float32x4_t pow2n0 = vreinterpretq_f32_s32(emm0_0);
        float32x4_t pow2n1 = vreinterpretq_f32_s32(emm0_1);

        float32x4_t bVal0 = vmulq_f32(y0, pow2n0);
        float32x4_t bVal1 = vmulq_f32(y1, pow2n1);
        vst1q_f32(bPtr, bVal0);
        vst1q_f32(bPtr + 4, bVal1);

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_exp_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t exp_hi = __riscv_vfmv_v_f_f32m2(88.376259f, vlmax);
    const vfloat32m2_t exp_lo = __riscv_vfmv_v_f_f32m2(-88.376259f, vlmax);
    const vfloat32m2_t log2EF = __riscv_vfmv_v_f_f32m2(1.442695f, vlmax);
    const vfloat32m2_t exp_C1 = __riscv_vfmv_v_f_f32m2(-0.6933594f, vlmax);
    const vfloat32m2_t exp_C2 = __riscv_vfmv_v_f_f32m2(0.000212194f, vlmax);
    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf1o2 = __riscv_vfmv_v_f_f32m2(0.5f, vlmax);

    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(1.9875691500e-4, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(1.3981999507e-3, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(8.3334519073e-3, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(4.1665795894e-2, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(1.6666665459e-1, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(5.0000001201e-1, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        v = __riscv_vfmin(v, exp_hi, vl);
        v = __riscv_vfmax(v, exp_lo, vl);
        vfloat32m2_t fx = __riscv_vfmadd(v, log2EF, cf1o2, vl);

        vfloat32m2_t rtz = __riscv_vfcvt_f(__riscv_vfcvt_rtz_x(fx, vl), vl);
        fx = __riscv_vfsub_mu(__riscv_vmfgt(rtz, fx, vl), rtz, rtz, cf1, vl);
        v = __riscv_vfmacc(v, fx, exp_C1, vl);
        v = __riscv_vfmacc(v, fx, exp_C2, vl);
        vfloat32m2_t vv = __riscv_vfmul(v, v, vl);

        vfloat32m2_t y = c0;
        y = __riscv_vfmadd(y, v, c1, vl);
        y = __riscv_vfmadd(y, v, c2, vl);
        y = __riscv_vfmadd(y, v, c3, vl);
        y = __riscv_vfmadd(y, v, c4, vl);
        y = __riscv_vfmadd(y, v, c5, vl);
        y = __riscv_vfmadd(y, vv, v, vl);
        y = __riscv_vfadd(y, cf1, vl);

        vfloat32m2_t pow2n = __riscv_vreinterpret_f32m2(
            __riscv_vsll(__riscv_vadd(__riscv_vfcvt_rtz_x(fx, vl), 0x7f, vl), 23, vl));

        __riscv_vse32(bVector, __riscv_vfmul(y, pow2n, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_exp_32f_u_H */
