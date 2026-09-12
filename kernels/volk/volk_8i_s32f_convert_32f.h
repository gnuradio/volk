/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8i_s32f_convert_32f
 *
 * \b Overview
 *
 * Convert the input vector of 8-bit chars to a vector of floats. The
 * floats are then divided by the scalar factor.  shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_s32f_convert_32f(float* outputVector, const int8_t* inputVector, const
 * float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 8-bit chars.
 * \li scalar: the scaling factor used to divide the results of the conversion.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li outputVector: The output 16-bit shorts.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8i_s32f_convert_32f();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_s32f_convert_32f_u_H
#define INCLUDED_volk_8i_s32f_convert_32f_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_8i_s32f_convert_32f_generic(float* outputVector,
                                                    const int8_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const float invScalar = 1.0f / scalar;

    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = (float)(*inputVector++) * invScalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_u_avx2(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m256 invScalar = _mm256_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_loadu_si128((const __m128i*)inputVector);
        const __m256i lower = _mm256_cvtepi8_epi32(input);
        const __m256i upper = _mm256_cvtepi8_epi32(_mm_srli_si128(input, 8));

        _mm256_storeu_ps(outputVector,
                         _mm256_mul_ps(_mm256_cvtepi32_ps(lower), invScalar));
        _mm256_storeu_ps(outputVector + 8,
                         _mm256_mul_ps(_mm256_cvtepi32_ps(upper), invScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_u_avx512(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m512 invScalar = _mm512_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_loadu_si128((const __m128i*)inputVector);
        const __m512i integers = _mm512_cvtepi8_epi32(input);
        _mm512_storeu_ps(outputVector,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(integers), invScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_s32f_convert_32f_u_sse4_1(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m128 invScalar = _mm_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_loadu_si128((const __m128i*)inputVector);
        _mm_storeu_ps(outputVector,
                      _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(input)), invScalar));
        _mm_storeu_ps(
            outputVector + 4,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 4))),
                       invScalar));
        _mm_storeu_ps(
            outputVector + 8,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 8))),
                       invScalar));
        _mm_storeu_ps(
            outputVector + 12,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 12))),
                       invScalar));
        inputVector += 16;
        outputVector += 16;
    }
    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE4_1 */


#endif /* INCLUDED_VOLK_8s_CONVERT_32f_UNALIGNED8_H */

#ifndef INCLUDED_volk_8i_s32f_convert_32f_a_H
#define INCLUDED_volk_8i_s32f_convert_32f_a_H

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_a_avx2(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m256 invScalar = _mm256_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_load_si128((const __m128i*)inputVector);
        const __m256i lower = _mm256_cvtepi8_epi32(input);
        const __m256i upper = _mm256_cvtepi8_epi32(_mm_srli_si128(input, 8));

        _mm256_store_ps(outputVector,
                        _mm256_mul_ps(_mm256_cvtepi32_ps(lower), invScalar));
        _mm256_store_ps(outputVector + 8,
                        _mm256_mul_ps(_mm256_cvtepi32_ps(upper), invScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_a_avx512(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m512 invScalar = _mm512_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_load_si128((const __m128i*)inputVector);
        const __m512i integers = _mm512_cvtepi8_epi32(input);
        _mm512_store_ps(outputVector,
                        _mm512_mul_ps(_mm512_cvtepi32_ps(integers), invScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_s32f_convert_32f_a_sse4_1(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m128 invScalar = _mm_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_load_si128((const __m128i*)inputVector);
        _mm_store_ps(outputVector,
                     _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(input)), invScalar));
        _mm_store_ps(
            outputVector + 4,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 4))),
                       invScalar));
        _mm_store_ps(
            outputVector + 8,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 8))),
                       invScalar));
        _mm_store_ps(
            outputVector + 12,
            _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(input, 12))),
                       invScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8i_s32f_convert_32f_neon(float* outputVector,
                                                 const int8_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    const float iScalar = 1.0f / scalar;
    const float32x4_t qiScalar = vdupq_n_f32(iScalar);
    const unsigned int sixteenthPoints = num_points / 16;
    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const int8x16_t input = vld1q_s8(inputVector);
        const int16x8_t lower = vmovl_s8(vget_low_s8(input));
        const int16x8_t upper = vmovl_s8(vget_high_s8(input));

        vst1q_f32(outputVector,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower))), qiScalar));
        vst1q_f32(outputVector + 4,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower))), qiScalar));
        vst1q_f32(outputVector + 8,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(upper))), qiScalar));
        vst1q_f32(outputVector + 12,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(upper))), qiScalar));
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_8i_s32f_convert_32f_neonv8(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const float iScalar = 1.0f / scalar;
    const float32x4_t qiScalar = vdupq_n_f32(iScalar);
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const int8x16_t input0 = vld1q_s8(inputVector);
        const int8x16_t input1 = vld1q_s8(inputVector + 16);
        __VOLK_PREFETCH(inputVector + 64);

        /* Widen int8 -> int16 -> int32 -> float */
        const int16x8_t lower0 = vmovl_s8(vget_low_s8(input0));
        const int16x8_t upper0 = vmovl_s8(vget_high_s8(input0));
        const int16x8_t lower1 = vmovl_s8(vget_low_s8(input1));
        const int16x8_t upper1 = vmovl_s8(vget_high_s8(input1));

        vst1q_f32(outputVector,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower0))), qiScalar));
        vst1q_f32(outputVector + 4,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower0))), qiScalar));
        vst1q_f32(outputVector + 8,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(upper0))), qiScalar));
        vst1q_f32(outputVector + 12,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(upper0))), qiScalar));
        vst1q_f32(outputVector + 16,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower1))), qiScalar));
        vst1q_f32(outputVector + 20,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower1))), qiScalar));
        vst1q_f32(outputVector + 24,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(upper1))), qiScalar));
        vst1q_f32(outputVector + 28,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(upper1))), qiScalar));

        inputVector += 32;
        outputVector += 32;
    }

    volk_8i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_ORC
extern void volk_8i_s32f_convert_32f_a_orc_impl(float* outputVector,
                                                const int8_t* inputVector,
                                                const float scalar,
                                                int num_points);

static inline void volk_8i_s32f_convert_32f_u_orc(float* outputVector,
                                                  const int8_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    float invscalar = 1.0 / scalar;
    volk_8i_s32f_convert_32f_a_orc_impl(outputVector, inputVector, invscalar, num_points);
}
#endif /* LV_HAVE_ORC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8i_s32f_convert_32f_rvv(float* outputVector,
                                                const int8_t* inputVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e8m2(n);
        vint16m4_t v = __riscv_vsext_vf2(__riscv_vle8_v_i8m2(inputVector, vl), vl);
        __riscv_vse32(
            outputVector, __riscv_vfmul(__riscv_vfwcvt_f(v, vl), 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_VOLK_8s_CONVERT_32f_ALIGNED8_H */
