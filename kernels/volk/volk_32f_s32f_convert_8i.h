/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_convert_8i
 *
 * \b Overview
 *
 * Converts a floating point number to a 8-bit int after applying a
 * scaling factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_convert_8i(int8_t* outputVector, const float* inputVector, const
 float scalar, unsigned int num_points)
 * \endcode
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
 * Convert floats from [-1,1] to 8-bit integers with a scale of 5 to maintain smallest
 delta
 *  int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   int8_t* out = (int8_t*)volk_malloc(sizeof(int8_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   // With float -> 8 bit ints be careful of scaling

 *   float scale = 5.1f;
 *
 *   volk_32f_s32f_convert_8i(out, increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_convert_8i_u_H
#define INCLUDED_volk_32f_s32f_convert_8i_u_H

#include <inttypes.h>
#include <math.h>

static inline void volk_32f_s32f_convert_8i_single(int8_t* out, const float in)
{
    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    if (in > max_val) {
        *out = (int8_t)(max_val);
    } else if (in < min_val) {
        *out = (int8_t)(min_val);
    } else {
        *out = (int8_t)(rintf(in));
    }
}

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_convert_8i_generic(int8_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    for (unsigned number = 0; number < num_points; ++number) {
        volk_32f_s32f_convert_8i_single(outputVector++, *inputVector++ * scalar);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_8i_u_avx2(int8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScalar = _mm256_set1_ps(scalar);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m256 inputVal1 = _mm256_loadu_ps(inputVector);
        const __m256 inputVal2 = _mm256_loadu_ps(inputVector + 8);
        const __m256 inputVal3 = _mm256_loadu_ps(inputVector + 16);
        const __m256 inputVal4 = _mm256_loadu_ps(inputVector + 24);

        const __m256 ret1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);
        const __m256 ret3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal3, vScalar), vmax_val), vmin_val);
        const __m256 ret4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal4, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal1 = _mm256_cvtps_epi32(ret1);
        const __m256i intInputVal2 = _mm256_cvtps_epi32(ret2);
        const __m256i intInputVal3 = _mm256_cvtps_epi32(ret3);
        const __m256i intInputVal4 = _mm256_cvtps_epi32(ret4);

        const __m256i packed1 = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(intInputVal1, intInputVal2), 0b11011000);
        const __m256i packed2 = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(intInputVal3, intInputVal4), 0b11011000);

        const __m256i intInputVal =
            _mm256_permute4x64_epi64(_mm256_packs_epi16(packed1, packed2), 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVector, intInputVal);
        inputVector += 32;
        outputVector += 32;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_8i_u_avx512(int8_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m512 inputVal1 = _mm512_loadu_ps(inputVector);
        const __m512 inputVal2 = _mm512_loadu_ps(inputVector + 16);

        const __m512 ret1 = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m512 ret2 = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m512i intInputVal1 = _mm512_cvtps_epi32(ret1);
        const __m512i intInputVal2 = _mm512_cvtps_epi32(ret2);

        // Pack int32 -> int16 -> int8
        const __m128i packed_result1 = _mm512_cvtsepi32_epi8(intInputVal1);
        _mm_storeu_si128((__m128i*)outputVector, packed_result1);

        const __m128i packed_result2 = _mm512_cvtsepi32_epi8(intInputVal2);
        _mm_storeu_si128((__m128i*)outputVector + 1, packed_result2);
        inputVector += 32;
        outputVector += 32;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_8i_u_sse2(int8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128 inputVal1 = _mm_loadu_ps(inputVector);
        const __m128 inputVal2 = _mm_loadu_ps(inputVector + 4);
        const __m128 inputVal3 = _mm_loadu_ps(inputVector + 8);
        const __m128 inputVal4 = _mm_loadu_ps(inputVector + 12);

        const __m128 ret1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m128 ret2 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);
        const __m128 ret3 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal3, vScalar), vmax_val), vmin_val);
        const __m128 ret4 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal4, vScalar), vmax_val), vmin_val);

        const __m128i intInputVal1 = _mm_cvtps_epi32(ret1);
        const __m128i intInputVal2 = _mm_cvtps_epi32(ret2);
        const __m128i intInputVal3 = _mm_cvtps_epi32(ret3);
        const __m128i intInputVal4 = _mm_cvtps_epi32(ret4);

        const __m128i packed1 = _mm_packs_epi32(intInputVal1, intInputVal2);
        const __m128i packed2 = _mm_packs_epi32(intInputVal3, intInputVal4);

        const __m128i output = _mm_packs_epi16(packed1, packed2);

        _mm_storeu_si128((__m128i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_8i_u_sse(int8_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scalar);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 input = _mm_loadu_ps(inputVector);

        const __m128 ret =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(input, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; inner_loop++) {
            outputVector[inner_loop] = (int8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
        inputVector += 4;
        outputVector += 4;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_convert_8i_u_H */
#ifndef INCLUDED_volk_32f_s32f_convert_8i_a_H
#define INCLUDED_volk_32f_s32f_convert_8i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_8i_a_avx2(int8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScalar = _mm256_set1_ps(scalar);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m256 inputVal1 = _mm256_load_ps(inputVector);
        const __m256 inputVal2 = _mm256_load_ps(inputVector + 8);
        const __m256 inputVal3 = _mm256_load_ps(inputVector + 16);
        const __m256 inputVal4 = _mm256_load_ps(inputVector + 24);

        const __m256 ret1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);
        const __m256 ret3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal3, vScalar), vmax_val), vmin_val);
        const __m256 ret4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal4, vScalar), vmax_val), vmin_val);

        const __m256i intInputVal1 = _mm256_cvtps_epi32(ret1);
        const __m256i intInputVal2 = _mm256_cvtps_epi32(ret2);
        const __m256i intInputVal3 = _mm256_cvtps_epi32(ret3);
        const __m256i intInputVal4 = _mm256_cvtps_epi32(ret4);

        const __m256i packed1 = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(intInputVal1, intInputVal2), 0b11011000);
        const __m256i packed2 = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(intInputVal3, intInputVal4), 0b11011000);

        const __m256i intInputVal =
            _mm256_permute4x64_epi64(_mm256_packs_epi16(packed1, packed2), 0b11011000);

        _mm256_store_si256((__m256i*)outputVector, intInputVal);
        inputVector += 32;
        outputVector += 32;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_8i_a_avx512(int8_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m512 inputVal1 = _mm512_load_ps(inputVector);
        const __m512 inputVal2 = _mm512_load_ps(inputVector + 16);

        const __m512 ret1 = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m512 ret2 = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);

        const __m512i intInputVal1 = _mm512_cvtps_epi32(ret1);
        const __m512i intInputVal2 = _mm512_cvtps_epi32(ret2);

        // Pack int32 -> int16 -> int8
        const __m128i packed_result1 = _mm512_cvtsepi32_epi8(intInputVal1);
        _mm_store_si128((__m128i*)outputVector, packed_result1);

        const __m128i packed_result2 = _mm512_cvtsepi32_epi8(intInputVal2);
        _mm_store_si128((__m128i*)(outputVector + 16), packed_result2);
        inputVector += 32;
        outputVector += 32;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_8i_a_sse2(int8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scalar);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128 inputVal1 = _mm_load_ps(inputVector);
        const __m128 inputVal2 = _mm_load_ps(inputVector + 4);
        const __m128 inputVal3 = _mm_load_ps(inputVector + 8);
        const __m128 inputVal4 = _mm_load_ps(inputVector + 12);

        const __m128 ret1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        const __m128 ret2 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal2, vScalar), vmax_val), vmin_val);
        const __m128 ret3 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal3, vScalar), vmax_val), vmin_val);
        const __m128 ret4 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal4, vScalar), vmax_val), vmin_val);

        const __m128i intInputVal1 = _mm_cvtps_epi32(ret1);
        const __m128i intInputVal2 = _mm_cvtps_epi32(ret2);
        const __m128i intInputVal3 = _mm_cvtps_epi32(ret3);
        const __m128i intInputVal4 = _mm_cvtps_epi32(ret4);

        const __m128i packed1 = _mm_packs_epi32(intInputVal1, intInputVal2);
        const __m128i packed2 = _mm_packs_epi32(intInputVal3, intInputVal4);

        const __m128i output = _mm_packs_epi16(packed1, packed2);

        _mm_store_si128((__m128i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_8i_a_sse(int8_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scalar);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128 input = _mm_load_ps(inputVector);

        const __m128 ret =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(input, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; inner_loop++) {
            outputVector[inner_loop] = (int8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
        inputVector += 4;
        outputVector += 4;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_8i_neon(int8_t* outputVector,
                                                 const float* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;

    const float32x4_t vScalar = vdupq_n_f32(scalar);
    const float32x4_t vmin_val = vdupq_n_f32(min_val);
    const float32x4_t vmax_val = vdupq_n_f32(max_val);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t neg_half = vdupq_n_f32(-0.5f);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const float32x4_t inputVal0 = vld1q_f32(inputVector);
        const float32x4_t inputVal1 = vld1q_f32(inputVector + 4);
        const float32x4_t inputVal2 = vld1q_f32(inputVector + 8);
        const float32x4_t inputVal3 = vld1q_f32(inputVector + 12);

        // Scale and clip
        float32x4_t ret0 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal0, vScalar), vmax_val), vmin_val);
        float32x4_t ret1 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal1, vScalar), vmax_val), vmin_val);
        float32x4_t ret2 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal2, vScalar), vmax_val), vmin_val);
        float32x4_t ret3 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal3, vScalar), vmax_val), vmin_val);

        // Round to nearest: add copysign(0.5, x) before truncating
        const uint32x4_t neg0 = vcltq_f32(ret0, zero);
        const uint32x4_t neg1 = vcltq_f32(ret1, zero);
        const uint32x4_t neg2 = vcltq_f32(ret2, zero);
        const uint32x4_t neg3 = vcltq_f32(ret3, zero);
        ret0 = vaddq_f32(ret0, vbslq_f32(neg0, neg_half, half));
        ret1 = vaddq_f32(ret1, vbslq_f32(neg1, neg_half, half));
        ret2 = vaddq_f32(ret2, vbslq_f32(neg2, neg_half, half));
        ret3 = vaddq_f32(ret3, vbslq_f32(neg3, neg_half, half));

        // Convert to int32 (truncates towards zero, but we pre-rounded)
        const int32x4_t intVal0 = vcvtq_s32_f32(ret0);
        const int32x4_t intVal1 = vcvtq_s32_f32(ret1);
        const int32x4_t intVal2 = vcvtq_s32_f32(ret2);
        const int32x4_t intVal3 = vcvtq_s32_f32(ret3);

        // Narrow to int16 with saturation
        const int16x4_t narrow16_0 = vqmovn_s32(intVal0);
        const int16x4_t narrow16_1 = vqmovn_s32(intVal1);
        const int16x4_t narrow16_2 = vqmovn_s32(intVal2);
        const int16x4_t narrow16_3 = vqmovn_s32(intVal3);
        const int16x8_t wide16_0 = vcombine_s16(narrow16_0, narrow16_1);
        const int16x8_t wide16_1 = vcombine_s16(narrow16_2, narrow16_3);

        // Narrow to int8 with saturation
        const int8x8_t narrow8_0 = vqmovn_s16(wide16_0);
        const int8x8_t narrow8_1 = vqmovn_s16(wide16_1);
        const int8x16_t result = vcombine_s8(narrow8_0, narrow8_1);

        vst1q_s8(outputVector, result);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_8i_neonv8(int8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = INT8_MIN;
    const float max_val = INT8_MAX;

    const float32x4_t vScalar = vdupq_n_f32(scalar);
    const float32x4_t vmin_val = vdupq_n_f32(min_val);
    const float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const float32x4_t inputVal0 = vld1q_f32(inputVector);
        const float32x4_t inputVal1 = vld1q_f32(inputVector + 4);
        const float32x4_t inputVal2 = vld1q_f32(inputVector + 8);
        const float32x4_t inputVal3 = vld1q_f32(inputVector + 12);
        const float32x4_t inputVal4 = vld1q_f32(inputVector + 16);
        const float32x4_t inputVal5 = vld1q_f32(inputVector + 20);
        const float32x4_t inputVal6 = vld1q_f32(inputVector + 24);
        const float32x4_t inputVal7 = vld1q_f32(inputVector + 28);
        __VOLK_PREFETCH(inputVector + 32);

        // Scale and clip
        float32x4_t ret0 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal0, vScalar), vmax_val), vmin_val);
        float32x4_t ret1 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal1, vScalar), vmax_val), vmin_val);
        float32x4_t ret2 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal2, vScalar), vmax_val), vmin_val);
        float32x4_t ret3 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal3, vScalar), vmax_val), vmin_val);
        float32x4_t ret4 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal4, vScalar), vmax_val), vmin_val);
        float32x4_t ret5 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal5, vScalar), vmax_val), vmin_val);
        float32x4_t ret6 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal6, vScalar), vmax_val), vmin_val);
        float32x4_t ret7 =
            vmaxq_f32(vminq_f32(vmulq_f32(inputVal7, vScalar), vmax_val), vmin_val);

        // Convert to int32 using round-to-nearest (ARMv8)
        const int32x4_t intVal0 = vcvtnq_s32_f32(ret0);
        const int32x4_t intVal1 = vcvtnq_s32_f32(ret1);
        const int32x4_t intVal2 = vcvtnq_s32_f32(ret2);
        const int32x4_t intVal3 = vcvtnq_s32_f32(ret3);
        const int32x4_t intVal4 = vcvtnq_s32_f32(ret4);
        const int32x4_t intVal5 = vcvtnq_s32_f32(ret5);
        const int32x4_t intVal6 = vcvtnq_s32_f32(ret6);
        const int32x4_t intVal7 = vcvtnq_s32_f32(ret7);

        // Narrow to int16 with saturation
        const int16x4_t narrow16_0 = vqmovn_s32(intVal0);
        const int16x4_t narrow16_1 = vqmovn_s32(intVal1);
        const int16x4_t narrow16_2 = vqmovn_s32(intVal2);
        const int16x4_t narrow16_3 = vqmovn_s32(intVal3);
        const int16x4_t narrow16_4 = vqmovn_s32(intVal4);
        const int16x4_t narrow16_5 = vqmovn_s32(intVal5);
        const int16x4_t narrow16_6 = vqmovn_s32(intVal6);
        const int16x4_t narrow16_7 = vqmovn_s32(intVal7);

        const int16x8_t wide16_0 = vcombine_s16(narrow16_0, narrow16_1);
        const int16x8_t wide16_1 = vcombine_s16(narrow16_2, narrow16_3);
        const int16x8_t wide16_2 = vcombine_s16(narrow16_4, narrow16_5);
        const int16x8_t wide16_3 = vcombine_s16(narrow16_6, narrow16_7);

        // Narrow to int8 with saturation
        const int8x8_t narrow8_0 = vqmovn_s16(wide16_0);
        const int8x8_t narrow8_1 = vqmovn_s16(wide16_1);
        const int8x8_t narrow8_2 = vqmovn_s16(wide16_2);
        const int8x8_t narrow8_3 = vqmovn_s16(wide16_3);

        const int8x16_t result0 = vcombine_s8(narrow8_0, narrow8_1);
        const int8x16_t result1 = vcombine_s8(narrow8_2, narrow8_3);

        vst1q_s8(outputVector, result0);
        vst1q_s8(outputVector + 16, result1);
        inputVector += 32;
        outputVector += 32;
    }

    volk_32f_s32f_convert_8i_generic(
        outputVector, inputVector, scalar, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_convert_8i_rvv(int8_t* outputVector,
                                                const float* inputVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        vint16m4_t vi = __riscv_vfncvt_x(__riscv_vfmul(v, scalar, vl), vl);
        __riscv_vse8(outputVector, __riscv_vnclip(vi, 0, 0, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_convert_8i_a_H */
