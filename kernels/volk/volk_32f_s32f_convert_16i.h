/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_convert_16i
 *
 * \b Overview
 *
 * Converts a floating point number to a 16-bit short after applying a
 * scaling factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_convert_16i(int16_t* outputVector, const float* inputVector, const
 * float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: the input vector of floats.
 * \li scalar: The value multiplied against each point in the input buffer.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector.
 *
 * \b Example
 * Convert floats from [-1,1] to 16-bit integers with a scale of 5 to maintain smallest
 * delta int N = 10; unsigned int alignment = volk_get_alignment(); float* increasing =
 * (float*)volk_malloc(sizeof(float)*N, alignment); int16_t* out =
 * (int16_t*)volk_malloc(sizeof(int16_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   float scale = 5.f;
 *
 *   volk_32f_s32f_convert_32i(out, increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_convert_16i_u_H
#define INCLUDED_volk_32f_s32f_convert_16i_u_H

#include <inttypes.h>
#include <limits.h>
#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_convert_16i_generic(int16_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const float min_val = SHRT_MIN;
    const float max_val = SHRT_MAX;

    for (unsigned int number = 0; number < num_points; ++number) {
        float value = inputVector[number] * scalar;
        if (value > max_val)
            value = max_val;
        else if (value < min_val)
            value = min_val;
        outputVector[number] = (int16_t)rintf(value);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_u_avx2(int16_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m256 inputVal1 = _mm256_loadu_ps(inputVector);
        const __m256 inputVal2 = _mm256_loadu_ps(inputVector + 8);

        const __m256 ret1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal1 = _mm256_cvtps_epi32(ret1);
        const __m256i intInputVal2 = _mm256_cvtps_epi32(ret2);

        const __m256i packed = _mm256_packs_epi32(intInputVal1, intInputVal2);
        const __m256i output = _mm256_permute4x64_epi64(packed, 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_u_avx512(int16_t* outputVector,
                                                      const float* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps((float)SHRT_MIN);
    const __m512 vmax_val = _mm512_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m512 inputVal = _mm512_loadu_ps(inputVector);

        const __m512 ret = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal, vScalar), vmax_val), vmin_val);
        const __m256i intInputVal = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret));

        _mm256_storeu_si256((__m256i*)outputVector, intInputVal);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_u_avx(int16_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m256 inputVal = _mm256_loadu_ps(inputVector);

        const __m256 ret = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal = _mm256_cvtps_epi32(ret);

        const __m128i intInputVal1 = _mm256_extractf128_si256(intInputVal, 0);
        const __m128i intInputVal2 = _mm256_extractf128_si256(intInputVal, 1);
        const __m128i output = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_storeu_si128((__m128i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_16i_u_sse2(int16_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const __m128 vScalar = _mm_set_ps1(scalar);
    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128 inputVal1 = _mm_loadu_ps(inputVector);
        const __m128 inputVal2 = _mm_loadu_ps(inputVector + 4);

        const __m128 ret1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m128 ret2 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m128i intInputVal1 = _mm_cvtps_epi32(ret1);
        const __m128i intInputVal2 = _mm_cvtps_epi32(ret2);

        const __m128i output = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_storeu_si128((__m128i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_16i_u_sse(int16_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    const __m128 vScalar = _mm_set_ps1(scalar);
    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];
    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 input = _mm_loadu_ps(inputVector);

        const __m128 ret =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(input, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        outputVector[0] = (int16_t)rintf(outputFloatBuffer[0]);
        outputVector[1] = (int16_t)rintf(outputFloatBuffer[1]);
        outputVector[2] = (int16_t)rintf(outputFloatBuffer[2]);
        outputVector[3] = (int16_t)rintf(outputFloatBuffer[3]);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_convert_16i_u_H */
#ifndef INCLUDED_volk_32f_s32f_convert_16i_a_H
#define INCLUDED_volk_32f_s32f_convert_16i_a_H

#include <inttypes.h>
#include <math.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_a_avx2(int16_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m256 inputVal1 = _mm256_load_ps(inputVector);
        const __m256 inputVal2 = _mm256_load_ps(inputVector + 8);

        const __m256 ret1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal1 = _mm256_cvtps_epi32(ret1);
        const __m256i intInputVal2 = _mm256_cvtps_epi32(ret2);

        const __m256i packed = _mm256_packs_epi32(intInputVal1, intInputVal2);
        const __m256i output = _mm256_permute4x64_epi64(packed, 0b11011000);

        _mm256_store_si256((__m256i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_a_avx512(int16_t* outputVector,
                                                      const float* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps((float)SHRT_MIN);
    const __m512 vmax_val = _mm512_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m512 inputVal = _mm512_load_ps(inputVector);

        const __m512 ret = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal, vScalar), vmax_val), vmin_val);
        const __m256i intInputVal = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret));

        _mm256_store_si256((__m256i*)outputVector, intInputVal);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_16i_a_avx(int16_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m256 inputVal = _mm256_load_ps(inputVector);

        const __m256 ret = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal = _mm256_cvtps_epi32(ret);

        const __m128i intInputVal1 = _mm256_extractf128_si256(intInputVal, 0);
        const __m128i intInputVal2 = _mm256_extractf128_si256(intInputVal, 1);
        const __m128i output = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_store_si128((__m128i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_16i_a_sse2(int16_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const __m128 vScalar = _mm_set_ps1(scalar);
    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const __m128 inputVal1 = _mm_load_ps(inputVector);
        const __m128 inputVal2 = _mm_load_ps(inputVector + 4);

        const __m128 ret1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m128 ret2 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m128i intInputVal1 = _mm_cvtps_epi32(ret1);
        const __m128i intInputVal2 = _mm_cvtps_epi32(ret2);

        const __m128i output = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_store_si128((__m128i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_16i_a_sse(int16_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    const __m128 vScalar = _mm_set_ps1(scalar);
    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];
    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 input = _mm_load_ps(inputVector);

        const __m128 ret =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(input, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        outputVector[0] = (int16_t)rintf(outputFloatBuffer[0]);
        outputVector[1] = (int16_t)rintf(outputFloatBuffer[1]);
        outputVector[2] = (int16_t)rintf(outputFloatBuffer[2]);
        outputVector[3] = (int16_t)rintf(outputFloatBuffer[3]);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32f_s32f_convert_16i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_16i_neon(int16_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = inputVector;
    int16_t* outputVectorPtr = outputVector;

    float min_val = SHRT_MIN;
    float max_val = SHRT_MAX;
    float r;

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < eighthPoints; number++) {
        float32x4_t inputVal1 = vld1q_f32(inputVectorPtr);
        float32x4_t inputVal2 = vld1q_f32(inputVectorPtr + 4);
        inputVectorPtr += 8;

        // Scale and clip
        float32x4_t ret1 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal1, vScalar), vmax_val), vmin_val);
        float32x4_t ret2 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal2, vScalar), vmax_val), vmin_val);

        // Round to nearest: add copysign(0.5, x) before truncating
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t neg_half = vdupq_n_f32(-0.5f);
        float32x4_t zero = vdupq_n_f32(0.0f);
        uint32x4_t neg1 = vcltq_f32(ret1, zero);
        uint32x4_t neg2 = vcltq_f32(ret2, zero);
        ret1 = vaddq_f32(ret1, vbslq_f32(neg1, neg_half, half));
        ret2 = vaddq_f32(ret2, vbslq_f32(neg2, neg_half, half));

        // Convert to int32 (truncates towards zero, but we pre-rounded)
        int32x4_t intVal1 = vcvtq_s32_f32(ret1);
        int32x4_t intVal2 = vcvtq_s32_f32(ret2);

        // Narrow to int16 with saturation
        int16x4_t narrow1 = vqmovn_s32(intVal1);
        int16x4_t narrow2 = vqmovn_s32(intVal2);
        int16x8_t result = vcombine_s16(narrow1, narrow2);

        vst1q_s16(outputVectorPtr, result);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int16_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_16i_neonv8(int16_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* inputVectorPtr = inputVector;
    int16_t* outputVectorPtr = outputVector;

    float min_val = SHRT_MIN;
    float max_val = SHRT_MAX;
    float r;

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < sixteenthPoints; number++) {
        float32x4_t inputVal0 = vld1q_f32(inputVectorPtr);
        float32x4_t inputVal1 = vld1q_f32(inputVectorPtr + 4);
        float32x4_t inputVal2 = vld1q_f32(inputVectorPtr + 8);
        float32x4_t inputVal3 = vld1q_f32(inputVectorPtr + 12);
        __VOLK_PREFETCH(inputVectorPtr + 16);
        inputVectorPtr += 16;

        // Scale and clip
        float32x4_t ret0 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal0, vScalar), vmax_val), vmin_val);
        float32x4_t ret1 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal1, vScalar), vmax_val), vmin_val);
        float32x4_t ret2 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal2, vScalar), vmax_val), vmin_val);
        float32x4_t ret3 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal3, vScalar), vmax_val), vmin_val);

        // Convert to int32 using round-to-nearest (ARMv8)
        int32x4_t intVal0 = vcvtnq_s32_f32(ret0);
        int32x4_t intVal1 = vcvtnq_s32_f32(ret1);
        int32x4_t intVal2 = vcvtnq_s32_f32(ret2);
        int32x4_t intVal3 = vcvtnq_s32_f32(ret3);

        // Narrow to int16 with saturation
        int16x4_t narrow0 = vqmovn_s32(intVal0);
        int16x4_t narrow1 = vqmovn_s32(intVal1);
        int16x4_t narrow2 = vqmovn_s32(intVal2);
        int16x4_t narrow3 = vqmovn_s32(intVal3);
        int16x8_t result0 = vcombine_s16(narrow0, narrow1);
        int16x8_t result1 = vcombine_s16(narrow2, narrow3);

        vst1q_s16(outputVectorPtr, result0);
        vst1q_s16(outputVectorPtr + 8, result1);
        outputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int16_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_convert_16i_rvv(int16_t* outputVector,
                                                 const float* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        v = __riscv_vfmul(v, scalar, vl);
        __riscv_vse16(outputVector, __riscv_vfncvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_convert_16i_a_H */
