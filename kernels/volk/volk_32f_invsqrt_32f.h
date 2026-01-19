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
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_invsqrt_32f_generic(float* cVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        cVector[number] = sqrtf(1.0f / aVector[number]);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_invsqrt_32f_recip_sqrt(float* cVector,
                                                   const float* aVector,
                                                   unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        cVector[number] = 1.0f / sqrtf(aVector[number]);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_invsqrt_32f_Q_rsqrt(float* cVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    // The famous fast inverse square root from Quake III Arena
    union {
        float f;
        uint32_t u;
    } conv, result;

    for (unsigned int number = 0; number < num_points; number++) {
        float a = aVector[number];
        float xhalf = 0.5f * a;
        conv.f = a;
        uint32_t input_bits = conv.u;        // Save original bits for edge case detection
        conv.u = 0x5f3759df - (conv.u >> 1); // The magic (note: use unsigned for shift)
        float x = conv.f;
        x = x * (1.5f - xhalf * x * x); // Newton-Raphson iteration 1
        x = x * (1.5f - xhalf * x * x);
        x = x * (1.5f - xhalf * x * x);

        // Branchless special case handling
        result.f = x;
        uint32_t is_positive = (uint32_t)(-(int32_t)(a > 0.0f));
        uint32_t is_zero = (uint32_t)(-(int32_t)(input_bits == 0x00000000));
        uint32_t is_inf = (uint32_t)(-(int32_t)(input_bits == 0x7F800000));
        uint32_t is_normal_pos = is_positive & ~is_inf;
        result.u = (result.u & is_normal_pos) | // Normal positive: keep result
                   (0x7F800000u & is_zero) |    // +0 → +Inf
                   (0x7FC00000u & ~is_positive & ~is_zero); // Negative/NaN → NaN
        // Note: +Inf → 0 is handled implicitly (all terms are 0 when is_inf)
        cVector[number] = result.f;
    }
}
#endif /* LV_HAVE_GENERIC */

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


#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32f_invsqrt_32f_a_avx2(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);
        __m256 cVal = _mm256_rsqrt_nr_avx2(aVal);
        _mm256_store_ps(cPtr, cVal);
        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */


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

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32f_invsqrt_32f_u_avx2(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);
        __m256 cVal = _mm256_rsqrt_nr_avx2(aVal);
        _mm256_storeu_ps(cPtr, cVal);
        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */

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
        float32x4_t x0 = vrsqrteq_f32(a); // +Inf for +0, 0 for +Inf

        // Two Newton-Raphson iterations for float32 accuracy
        float32x4_t x1 = vmulq_f32(x0, vrsqrtsq_f32(vmulq_f32(a, x0), x0));
        x1 = vmulq_f32(x1, vrsqrtsq_f32(vmulq_f32(a, x1), x1));

        // For +0 and +Inf inputs, x0 is correct but NR produces NaN due to Inf*0
        // Blend: use x0 where a == +0 or a == +Inf, else use x1
        uint32x4_t a_bits = vreinterpretq_u32_f32(a);
        uint32x4_t zero_mask = vceqq_u32(a_bits, vdupq_n_u32(0x00000000));
        uint32x4_t inf_mask = vceqq_u32(a_bits, vdupq_n_u32(0x7F800000));
        uint32x4_t special_mask = vorrq_u32(zero_mask, inf_mask);

        vst1q_f32(cPtr, vbslq_f32(special_mask, x0, x1));
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = 1.0f / sqrtf(*aPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

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

        // Initial estimate (~7-bit precision): +Inf for +0, 0 for +Inf
        vfloat32m8_t x0 = __riscv_vfrsqrt7(a, vl);

        // Two Newton-Raphson iterations: x = x * (1.5 - 0.5 * a * x * x)
        vfloat32m8_t half_a = __riscv_vfmul(half, a, vl);
        vfloat32m8_t x1 = __riscv_vfmul(
            x0, __riscv_vfnmsac(three_halfs, half_a, __riscv_vfmul(x0, x0, vl), vl), vl);
        x1 = __riscv_vfmul(
            x1, __riscv_vfnmsac(three_halfs, half_a, __riscv_vfmul(x1, x1, vl), vl), vl);

        // For +0 and +Inf inputs, x0 is correct but NR produces NaN due to Inf*0
        // Blend: use x0 where a == +0 or a == +Inf, else use x1
        vuint32m8_t a_bits = __riscv_vreinterpret_v_f32m8_u32m8(a);
        vbool4_t zero_mask = __riscv_vmseq_vx_u32m8_b4(a_bits, 0x00000000, vl);
        vbool4_t inf_mask = __riscv_vmseq_vx_u32m8_b4(a_bits, 0x7F800000, vl);
        vbool4_t special_mask = __riscv_vmor_mm_b4(zero_mask, inf_mask, vl);
        vfloat32m8_t result = __riscv_vmerge_vvm_f32m8(x1, x0, special_mask, vl);

        __riscv_vse32(cVector, result, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_invsqrt_32f_a_H */
