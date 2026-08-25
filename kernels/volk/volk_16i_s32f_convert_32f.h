/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts 16-bit shorts to scaled 32-bit floating point values.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_s32f_convert_32f(float* outputVector, const int16_t* inputVector, const
 * float scalar, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 16-bit shorts.
 * \li scalar: The value divided against each point in the output buffer.
 * \li num_points: The number of complex data points.
 *
 * \b Outputs
 * \li outputVector: The output vector of 8-bit chars.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16i_s32f_convert_32f();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_s32f_convert_32f_u_H
#define INCLUDED_volk_16i_s32f_convert_32f_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_s32f_convert_32f_generic(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = ((float)(*inputVector++)) / scalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 invScalar = _mm256_set1_ps(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;

        const __m256i inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        const __m256 ret = _mm256_mul_ps(_mm256_cvtepi32_ps(inputVal2), invScalar);

        _mm256_storeu_ps(outputVector, ret);
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512 invScalar = _mm512_set1_ps(1.0 / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m256i inputVal = _mm256_loadu_si256((__m256i*)inputVector);
        inputVector += 16;

        const __m512i inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        const __m512 ret = _mm512_mul_ps(_mm512_cvtepi32_ps(inputVal2), invScalar);

        _mm512_storeu_ps(outputVector, ret);
        outputVector += 16;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;

        // Shift the input data to the right by 64 bits ( 8 bytes )
        const __m128i inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        const __m128i converted1 = _mm_cvtepi16_epi32(inputVal);
        const __m128i converted2 = _mm_cvtepi16_epi32(inputVal2);

        const __m128 first = _mm_mul_ps(_mm_cvtepi32_ps(converted1), invScalar);
        const __m128 second = _mm_mul_ps(_mm_cvtepi32_ps(converted2), invScalar);
        const __m256 output = _mm256_set_m128(second, first);

        _mm256_storeu_ps(outputVector, output);
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;

        // Shift the input data to the right by 64 bits ( 8 bytes )
        const __m128i inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        const __m128i converted1 = _mm_cvtepi16_epi32(inputVal);
        const __m128i converted2 = _mm_cvtepi16_epi32(inputVal2);

        const __m128 first = _mm_mul_ps(_mm_cvtepi32_ps(converted1), invScalar);
        const __m128 second = _mm_mul_ps(_mm_cvtepi32_ps(converted2), invScalar);
        _mm_storeu_ps(outputVector, first);
        outputVector += 4;

        _mm_storeu_ps(outputVector, second);
        outputVector += 4;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 ret = _mm_mul_ps(_mm_set_ps((float)(inputVector[3]),
                                                 (float)(inputVector[2]),
                                                 (float)(inputVector[1]),
                                                 (float)(inputVector[0])),
                                      invScalar);

        _mm_storeu_ps(outputVector, ret);

        inputVector += 4;
        outputVector += 4;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_s32f_convert_32f_neon(float* outputVector,
                                                  const int16_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    const float32x4_t inv_scale = vdupq_n_f32(1.0 / scalar);

    // the generic disassembles to a 128-bit load
    // and duplicates every instruction to operate on 64-bits
    // at a time. This is only possible with lanes, which is faster
    // than just doing a vld1_s16, but still slower.
    for (unsigned int number = 0; number < eighth_points; ++number) {
        const int16x4x2_t input16 = vld2_s16(inputVector);
        // widen 16-bit int to 32-bit int
        const int32x4_t input32_0 = vmovl_s16(input16.val[0]);
        const int32x4_t input32_1 = vmovl_s16(input16.val[1]);
        // convert 32-bit int to float with scale
        const float32x4_t input_float_0 = vcvtq_f32_s32(input32_0);
        const float32x4_t input_float_1 = vcvtq_f32_s32(input32_1);
        const float32x4x2_t output_float = { vmulq_f32(input_float_0, inv_scale),
                                             vmulq_f32(input_float_1, inv_scale) };
        vst2q_f32(outputVector, output_float);
        inputVector += 8;
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighth_points * 8);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_s32f_convert_32f_neonv8(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const float32x4_t inv_scale = vdupq_n_f32(1.0f / scalar);

    /* Process 8 int16 values per iteration using 64-bit loads */
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; ++number) {
        const int16x4_t v0 = vld1_s16(inputVector);
        const int16x4_t v1 = vld1_s16(inputVector + 4);
        __VOLK_PREFETCH(inputVector + 16);

        /* Widen int16 to int32, convert to float, scale */
        const float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(v0)), inv_scale);
        const float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(v1)), inv_scale);

        vst1q_f32(outputVector, f0);
        vst1q_f32(outputVector + 4, f1);

        inputVector += 8;
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_16i_s32f_convert_32f_u_H */
#ifndef INCLUDED_volk_16i_s32f_convert_32f_a_H
#define INCLUDED_volk_16i_s32f_convert_32f_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 invScalar = _mm256_set1_ps(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_load_si128((__m128i*)inputVector);
        inputVector += 8;

        const __m256i inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        const __m256 ret = _mm256_mul_ps(_mm256_cvtepi32_ps(inputVal2), invScalar);

        _mm256_store_ps(outputVector, ret);
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512 invScalar = _mm512_set1_ps(1.0 / scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m256i inputVal = _mm256_load_si256((__m256i*)inputVector);
        inputVector += 16;

        const __m512i inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        const __m512 ret = _mm512_mul_ps(_mm512_cvtepi32_ps(inputVal2), invScalar);

        _mm512_store_ps(outputVector, ret);
        outputVector += 16;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_load_si128((__m128i*)inputVector);
        inputVector += 8;

        // Shift the input data to the right by 64 bits ( 8 bytes )
        const __m128i inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        const __m128i converted1 = _mm_cvtepi16_epi32(inputVal);
        const __m128i converted2 = _mm_cvtepi16_epi32(inputVal2);

        const __m128 first = _mm_mul_ps(_mm_cvtepi32_ps(converted1), invScalar);
        const __m128 second = _mm_mul_ps(_mm_cvtepi32_ps(converted2), invScalar);
        const __m256 output = _mm256_set_m128(second, first);

        _mm256_store_ps(outputVector, output);
        outputVector += 8;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128i inputVal = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;

        // Shift the input data to the right by 64 bits ( 8 bytes )
        const __m128i inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        const __m128i converted1 = _mm_cvtepi16_epi32(inputVal);
        const __m128i converted2 = _mm_cvtepi16_epi32(inputVal2);

        const __m128 first = _mm_mul_ps(_mm_cvtepi32_ps(converted1), invScalar);
        const __m128 second = _mm_mul_ps(_mm_cvtepi32_ps(converted2), invScalar);
        _mm_storeu_ps(outputVector, first);
        outputVector += 4;

        _mm_storeu_ps(outputVector, second);
        outputVector += 4;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 ret = _mm_mul_ps(_mm_set_ps((float)(inputVector[3]),
                                                 (float)(inputVector[2]),
                                                 (float)(inputVector[1]),
                                                 (float)(inputVector[0])),
                                      invScalar);

        _mm_storeu_ps(outputVector, ret);

        inputVector += 4;
        outputVector += 4;
    }

    volk_16i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_s32f_convert_32f_rvv(float* outputVector,
                                                 const int16_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t remaining = num_points;
    for (size_t vl; remaining > 0;
         remaining -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e16m4(remaining);
        const vfloat32m8_t v =
            __riscv_vfwcvt_f(__riscv_vle16_v_i16m4(inputVector, vl), vl);
        __riscv_vse32(outputVector, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16i_s32f_convert_32f_a_H */
