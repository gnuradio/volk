/* -*- c++ -*- */
/*
 * Copyright 2023 Daniel Estevez <daniel@destevez.net>
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_x2_convert_8u
 *
 * \b Overview
 *
 * Converts a floating point number to an 8-bit unsigned int after applying a
 * multiplicative scaling factor and an additive bias.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_x2_convert_8u(uint8_t* outputVector, const float* inputVector,
 const float scale, const float bias, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: the input vector of floats.
 * \li scale: The value multiplied against each point in the input buffer.
 * \li bias: The value added to each multiplication by the scale.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector.
 *
 * \b Example
 * Convert floats from [-1,1] to 8-bit unsigend integers with a scale of 128 and a bias of
 128
 *  int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   uint8_t* out = (uint8_t*)volk_malloc(sizeof(uint8_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   float scale = 128.0f;
 *   float bias = 128.0f;
 *
 *   volk_32f_s32f_x2_convert_8u(out, increasing, scale, bias, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_x2_convert_8u_u_H
#define INCLUDED_volk_32f_s32f_x2_convert_8u_u_H

#include <inttypes.h>

static inline void volk_32f_s32f_x2_convert_8u_single(uint8_t* out, const float in)
{
    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    if (in > max_val) {
        *out = (uint8_t)(max_val);
    } else if (in < min_val) {
        *out = (uint8_t)(min_val);
    } else {
        *out = (uint8_t)(rintf(in));
    }
}


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_x2_convert_8u_generic(uint8_t* outputVector,
                                                       const float* inputVector,
                                                       const float scale,
                                                       const float bias,
                                                       unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        volk_32f_s32f_x2_convert_8u_single(outputVector++, *inputVector++ * scale + bias);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_avx512f(uint8_t* outputVector,
                                                         const float* inputVector,
                                                         const float scale,
                                                         const float bias,
                                                         unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    const __m512 vScale = _mm512_set1_ps(scale);
    const __m512 vBias = _mm512_set1_ps(bias);
    const __m512 vmin_val = _mm512_setzero_ps();
    const __m512 vmax_val = _mm512_set1_ps(UINT8_MAX);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m512 inputVal1 = _mm512_loadu_ps(inputVector);
        const __m512 inputVal2 = _mm512_loadu_ps(inputVector + 16);
        inputVector += 32;
        const __m512 scaledVal1 = _mm512_max_ps(
            _mm512_min_ps(_mm512_fmadd_ps(inputVal1, vScale, vBias), vmax_val), vmin_val);
        const __m512 scaledVal2 = _mm512_max_ps(
            _mm512_min_ps(_mm512_fmadd_ps(inputVal2, vScale, vBias), vmax_val), vmin_val);
        const __m128i output1 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(scaledVal1));
        const __m128i output2 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(scaledVal2));
        _mm_storeu_si128((__m128i*)outputVector, output1);
        _mm_storeu_si128((__m128i*)(outputVector + 16), output2);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX512F */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_avx2_fma(uint8_t* outputVector,
                                                          const float* inputVector,
                                                          const float scale,
                                                          const float bias,
                                                          unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        __m256 inputVal1 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal2 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal3 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal4 = _mm256_loadu_ps(inputVector);
        inputVector += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal1, vScale, vBias), vmax_val), vmin_val);
        inputVal2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal2, vScale, vBias), vmax_val), vmin_val);
        inputVal3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal3, vScale, vBias), vmax_val), vmin_val);
        inputVal4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal4, vScale, vBias), vmax_val), vmin_val);

        __m256i intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        __m256i intInputVal2 = _mm256_cvtps_epi32(inputVal2);
        __m256i intInputVal3 = _mm256_cvtps_epi32(inputVal3);
        __m256i intInputVal4 = _mm256_cvtps_epi32(inputVal4);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);
        intInputVal3 = _mm256_packs_epi32(intInputVal3, intInputVal4);
        intInputVal3 = _mm256_permute4x64_epi64(intInputVal3, 0b11011000);

        intInputVal1 = _mm256_packus_epi16(intInputVal1, intInputVal3);
        const __m256i intInputVal = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVector, intInputVal);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_avx2(uint8_t* outputVector,
                                                      const float* inputVector,
                                                      const float scale,
                                                      const float bias,
                                                      unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        __m256 inputVal1 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal2 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal3 = _mm256_loadu_ps(inputVector);
        inputVector += 8;
        __m256 inputVal4 = _mm256_loadu_ps(inputVector);
        inputVector += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal1, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal2, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal3, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal4, vScale), vBias),
                          vmax_val),
            vmin_val);

        __m256i intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        __m256i intInputVal2 = _mm256_cvtps_epi32(inputVal2);
        __m256i intInputVal3 = _mm256_cvtps_epi32(inputVal3);
        __m256i intInputVal4 = _mm256_cvtps_epi32(inputVal4);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);
        intInputVal3 = _mm256_packs_epi32(intInputVal3, intInputVal4);
        intInputVal3 = _mm256_permute4x64_epi64(intInputVal3, 0b11011000);

        intInputVal1 = _mm256_packus_epi16(intInputVal1, intInputVal3);
        const __m256i intInputVal = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVector, intInputVal);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_sse2(uint8_t* outputVector,
                                                      const float* inputVector,
                                                      const float scale,
                                                      const float bias,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        __m128 inputVal1 = _mm_loadu_ps(inputVector);
        inputVector += 4;
        __m128 inputVal2 = _mm_loadu_ps(inputVector);
        inputVector += 4;
        __m128 inputVal3 = _mm_loadu_ps(inputVector);
        inputVector += 4;
        __m128 inputVal4 = _mm_loadu_ps(inputVector);
        inputVector += 4;

        inputVal1 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal1, vScale), vBias), vmax_val),
            vmin_val);
        inputVal2 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal2, vScale), vBias), vmax_val),
            vmin_val);
        inputVal3 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal3, vScale), vBias), vmax_val),
            vmin_val);
        inputVal4 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal4, vScale), vBias), vmax_val),
            vmin_val);

        __m128i intInputVal1 = _mm_cvtps_epi32(inputVal1);
        __m128i intInputVal2 = _mm_cvtps_epi32(inputVal2);
        __m128i intInputVal3 = _mm_cvtps_epi32(inputVal3);
        __m128i intInputVal4 = _mm_cvtps_epi32(inputVal4);

        intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);
        intInputVal3 = _mm_packs_epi32(intInputVal3, intInputVal4);

        intInputVal1 = _mm_packus_epi16(intInputVal1, intInputVal3);

        _mm_storeu_si128((__m128i*)outputVector, intInputVal1);
        outputVector += 16;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - sixteenthPoints * 16);
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_sse(uint8_t* outputVector,
                                                     const float* inputVector,
                                                     const float scale,
                                                     const float bias,
                                                     unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        __m128 ret = _mm_loadu_ps(inputVector);
        inputVector += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_add_ps(_mm_mul_ps(ret, vScale), vBias), vmax_val),
                         vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; ++inner_loop) {
            *outputVector++ = (uint8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_x2_convert_8u_u_H */
#ifndef INCLUDED_volk_32f_s32f_x2_convert_8u_a_H
#define INCLUDED_volk_32f_s32f_x2_convert_8u_a_H

#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_avx512f(uint8_t* outputVector,
                                                         const float* inputVector,
                                                         const float scale,
                                                         const float bias,
                                                         unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    const __m512 vScale = _mm512_set1_ps(scale);
    const __m512 vBias = _mm512_set1_ps(bias);
    const __m512 vmin_val = _mm512_setzero_ps();
    const __m512 vmax_val = _mm512_set1_ps(UINT8_MAX);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m512 inputVal1 = _mm512_load_ps(inputVector);
        const __m512 inputVal2 = _mm512_load_ps(inputVector + 16);
        inputVector += 32;
        const __m512 scaledVal1 = _mm512_max_ps(
            _mm512_min_ps(_mm512_fmadd_ps(inputVal1, vScale, vBias), vmax_val), vmin_val);
        const __m512 scaledVal2 = _mm512_max_ps(
            _mm512_min_ps(_mm512_fmadd_ps(inputVal2, vScale, vBias), vmax_val), vmin_val);
        const __m128i output1 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(scaledVal1));
        const __m128i output2 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(scaledVal2));
        _mm_store_si128((__m128i*)outputVector, output1);
        _mm_store_si128((__m128i*)(outputVector + 16), output2);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_avx2_fma(uint8_t* outputVector,
                                                          const float* inputVector,
                                                          const float scale,
                                                          const float bias,
                                                          unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        __m256 inputVal1 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal2 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal3 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal4 = _mm256_load_ps(inputVector);
        inputVector += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal1, vScale, vBias), vmax_val), vmin_val);
        inputVal2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal2, vScale, vBias), vmax_val), vmin_val);
        inputVal3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal3, vScale, vBias), vmax_val), vmin_val);
        inputVal4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_fmadd_ps(inputVal4, vScale, vBias), vmax_val), vmin_val);

        __m256i intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        __m256i intInputVal2 = _mm256_cvtps_epi32(inputVal2);
        __m256i intInputVal3 = _mm256_cvtps_epi32(inputVal3);
        __m256i intInputVal4 = _mm256_cvtps_epi32(inputVal4);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);
        intInputVal3 = _mm256_packs_epi32(intInputVal3, intInputVal4);
        intInputVal3 = _mm256_permute4x64_epi64(intInputVal3, 0b11011000);

        intInputVal1 = _mm256_packus_epi16(intInputVal1, intInputVal3);
        const __m256i intInputVal = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);

        _mm256_store_si256((__m256i*)outputVector, intInputVal);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_avx2(uint8_t* outputVector,
                                                      const float* inputVector,
                                                      const float scale,
                                                      const float bias,
                                                      unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        __m256 inputVal1 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal2 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal3 = _mm256_load_ps(inputVector);
        inputVector += 8;
        __m256 inputVal4 = _mm256_load_ps(inputVector);
        inputVector += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal1, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal2 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal2, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal3 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal3, vScale), vBias),
                          vmax_val),
            vmin_val);
        inputVal4 = _mm256_max_ps(
            _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(inputVal4, vScale), vBias),
                          vmax_val),
            vmin_val);

        __m256i intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        __m256i intInputVal2 = _mm256_cvtps_epi32(inputVal2);
        __m256i intInputVal3 = _mm256_cvtps_epi32(inputVal3);
        __m256i intInputVal4 = _mm256_cvtps_epi32(inputVal4);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);
        intInputVal3 = _mm256_packs_epi32(intInputVal3, intInputVal4);
        intInputVal3 = _mm256_permute4x64_epi64(intInputVal3, 0b11011000);

        intInputVal1 = _mm256_packus_epi16(intInputVal1, intInputVal3);
        const __m256i intInputVal = _mm256_permute4x64_epi64(intInputVal1, 0b11011000);

        _mm256_store_si256((__m256i*)outputVector, intInputVal);
        outputVector += 32;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - thirtysecondPoints * 32);
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_sse2(uint8_t* outputVector,
                                                      const float* inputVector,
                                                      const float scale,
                                                      const float bias,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        __m128 inputVal1 = _mm_load_ps(inputVector);
        inputVector += 4;
        __m128 inputVal2 = _mm_load_ps(inputVector);
        inputVector += 4;
        __m128 inputVal3 = _mm_load_ps(inputVector);
        inputVector += 4;
        __m128 inputVal4 = _mm_load_ps(inputVector);
        inputVector += 4;

        inputVal1 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal1, vScale), vBias), vmax_val),
            vmin_val);
        inputVal2 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal2, vScale), vBias), vmax_val),
            vmin_val);
        inputVal3 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal3, vScale), vBias), vmax_val),
            vmin_val);
        inputVal4 = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(inputVal4, vScale), vBias), vmax_val),
            vmin_val);

        __m128i intInputVal1 = _mm_cvtps_epi32(inputVal1);
        __m128i intInputVal2 = _mm_cvtps_epi32(inputVal2);
        __m128i intInputVal3 = _mm_cvtps_epi32(inputVal3);
        __m128i intInputVal4 = _mm_cvtps_epi32(inputVal4);

        intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);
        intInputVal3 = _mm_packs_epi32(intInputVal3, intInputVal4);

        intInputVal1 = _mm_packus_epi16(intInputVal1, intInputVal3);

        _mm_store_si128((__m128i*)outputVector, intInputVal1);
        outputVector += 16;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_sse(uint8_t* outputVector,
                                                     const float* inputVector,
                                                     const float scale,
                                                     const float bias,
                                                     unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        __m128 ret = _mm_load_ps(inputVector);
        inputVector += 4;

        ret = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(ret, vScalar), vBias), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; ++inner_loop) {
            *outputVector++ = (uint8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_x2_convert_8u_neon(uint8_t* outputVector,
                                                    const float* inputVector,
                                                    const float scale,
                                                    const float bias,
                                                    unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;

    float32x4_t vScale = vdupq_n_f32(scale);
    float32x4_t vBias = vdupq_n_f32(bias);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);
    const float32x4_t half = vdupq_n_f32(0.5f);

    for (unsigned int number = 0; number < sixteenth_points; ++number) {
        float32x4_t inputVal0 = vld1q_f32(inputVector);
        float32x4_t inputVal1 = vld1q_f32(inputVector + 4);
        float32x4_t inputVal2 = vld1q_f32(inputVector + 8);
        float32x4_t inputVal3 = vld1q_f32(inputVector + 12);
        inputVector += 16;

        inputVal0 = vmlaq_f32(vBias, inputVal0, vScale);
        inputVal1 = vmlaq_f32(vBias, inputVal1, vScale);
        inputVal2 = vmlaq_f32(vBias, inputVal2, vScale);
        inputVal3 = vmlaq_f32(vBias, inputVal3, vScale);

        inputVal0 = vmaxq_f32(vminq_f32(inputVal0, vmax_val), vmin_val);
        inputVal1 = vmaxq_f32(vminq_f32(inputVal1, vmax_val), vmin_val);
        inputVal2 = vmaxq_f32(vminq_f32(inputVal2, vmax_val), vmin_val);
        inputVal3 = vmaxq_f32(vminq_f32(inputVal3, vmax_val), vmin_val);

        uint32x4_t intVal0 = vcvtq_u32_f32(vaddq_f32(inputVal0, half));
        uint32x4_t intVal1 = vcvtq_u32_f32(vaddq_f32(inputVal1, half));
        uint32x4_t intVal2 = vcvtq_u32_f32(vaddq_f32(inputVal2, half));
        uint32x4_t intVal3 = vcvtq_u32_f32(vaddq_f32(inputVal3, half));

        uint16x4_t shortVal0 = vqmovn_u32(intVal0);
        uint16x4_t shortVal1 = vqmovn_u32(intVal1);
        uint16x4_t shortVal2 = vqmovn_u32(intVal2);
        uint16x4_t shortVal3 = vqmovn_u32(intVal3);

        uint16x8_t shortVal01 = vcombine_u16(shortVal0, shortVal1);
        uint16x8_t shortVal23 = vcombine_u16(shortVal2, shortVal3);

        uint8x8_t byteVal01 = vqmovn_u16(shortVal01);
        uint8x8_t byteVal23 = vqmovn_u16(shortVal23);

        vst1_u8(outputVector, byteVal01);
        vst1_u8(outputVector + 8, byteVal23);
        outputVector += 16;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - sixteenth_points * 16);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_x2_convert_8u_neonv8(uint8_t* outputVector,
                                                      const float* inputVector,
                                                      const float scale,
                                                      const float bias,
                                                      unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;

    float32x4_t vScale = vdupq_n_f32(scale);
    float32x4_t vBias = vdupq_n_f32(bias);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (unsigned int number = 0; number < sixteenth_points; ++number) {
        float32x4_t inputVal0 = vld1q_f32(inputVector);
        float32x4_t inputVal1 = vld1q_f32(inputVector + 4);
        float32x4_t inputVal2 = vld1q_f32(inputVector + 8);
        float32x4_t inputVal3 = vld1q_f32(inputVector + 12);
        __VOLK_PREFETCH(inputVector + 16);
        inputVector += 16;

        inputVal0 = vfmaq_f32(vBias, inputVal0, vScale);
        inputVal1 = vfmaq_f32(vBias, inputVal1, vScale);
        inputVal2 = vfmaq_f32(vBias, inputVal2, vScale);
        inputVal3 = vfmaq_f32(vBias, inputVal3, vScale);

        inputVal0 = vmaxq_f32(vminq_f32(inputVal0, vmax_val), vmin_val);
        inputVal1 = vmaxq_f32(vminq_f32(inputVal1, vmax_val), vmin_val);
        inputVal2 = vmaxq_f32(vminq_f32(inputVal2, vmax_val), vmin_val);
        inputVal3 = vmaxq_f32(vminq_f32(inputVal3, vmax_val), vmin_val);

        uint32x4_t intVal0 = vcvtaq_u32_f32(inputVal0);
        uint32x4_t intVal1 = vcvtaq_u32_f32(inputVal1);
        uint32x4_t intVal2 = vcvtaq_u32_f32(inputVal2);
        uint32x4_t intVal3 = vcvtaq_u32_f32(inputVal3);

        uint16x4_t shortVal0 = vqmovn_u32(intVal0);
        uint16x4_t shortVal1 = vqmovn_u32(intVal1);
        uint16x4_t shortVal2 = vqmovn_u32(intVal2);
        uint16x4_t shortVal3 = vqmovn_u32(intVal3);

        uint16x8_t shortVal01 = vcombine_u16(shortVal0, shortVal1);
        uint16x8_t shortVal23 = vcombine_u16(shortVal2, shortVal3);

        uint8x8_t byteVal01 = vqmovn_u16(shortVal01);
        uint8x8_t byteVal23 = vqmovn_u16(shortVal23);

        vst1_u8(outputVector, byteVal01);
        vst1_u8(outputVector + 8, byteVal23);
        outputVector += 16;
    }

    volk_32f_s32f_x2_convert_8u_generic(
        outputVector, inputVector, scale, bias, num_points - sixteenth_points * 16);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_x2_convert_8u_rvv(uint8_t* outputVector,
                                                   const float* inputVector,
                                                   const float scale,
                                                   const float bias,
                                                   unsigned int num_points)
{
    vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(bias, __riscv_vsetvlmax_e32m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        vuint16m4_t vi = __riscv_vfncvt_xu(__riscv_vfmadd_vf_f32m8(v, scale, vb, vl), vl);
        __riscv_vse8(outputVector, __riscv_vnclipu(vi, 0, 0, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_x2_convert_8u_a_H */
