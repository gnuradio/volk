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
    const float* inputVectorPtr = inputVector;

    for (unsigned int number = 0; number < num_points; number++) {
        const float r = *inputVectorPtr++ * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
}

#endif /* LV_HAVE_GENERIC */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_u_avx2_fma(uint8_t* outputVector,
                                                          const float* inputVector,
                                                          const float scale,
                                                          const float bias,
                                                          unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        __m256 inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal2 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal3 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal4 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;

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

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        __m256 inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal2 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal3 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal4 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;

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

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m128 inputVal1 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal2 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal3 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal4 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

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

        _mm_storeu_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 ret = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_add_ps(_mm_mul_ps(ret, vScale), vBias), vmax_val),
                         vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; inner_loop++) {
            *outputVectorPtr++ = (uint8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
}

#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_x2_convert_8u_u_H */
#ifndef INCLUDED_volk_32f_s32f_x2_convert_8u_a_H
#define INCLUDED_volk_32f_s32f_x2_convert_8u_a_H

#include <inttypes.h>
#include <volk/volk_common.h>

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_s32f_x2_convert_8u_a_avx2_fma(uint8_t* outputVector,
                                                          const float* inputVector,
                                                          const float scale,
                                                          const float bias,
                                                          unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        __m256 inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal2 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal3 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal4 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;

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

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);

    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vBias = _mm256_set1_ps(bias);

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        __m256 inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal2 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal3 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __m256 inputVal4 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;

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

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScale = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m128 inputVal1 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal2 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal3 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __m128 inputVal4 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

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

        _mm_store_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
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

    const float* inputVectorPtr = (const float*)inputVector;
    uint8_t* outputVectorPtr = outputVector;

    const float min_val = 0.0f;
    const float max_val = UINT8_MAX;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    const __m128 vScalar = _mm_set_ps1(scale);
    const __m128 vBias = _mm_set_ps1(bias);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 ret = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(
            _mm_min_ps(_mm_add_ps(_mm_mul_ps(ret, vScalar), vBias), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        for (size_t inner_loop = 0; inner_loop < 4; inner_loop++) {
            *outputVectorPtr++ = (uint8_t)(rintf(outputFloatBuffer[inner_loop]));
        }
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        const float r = inputVector[number] * scale + bias;
        volk_32f_s32f_x2_convert_8u_single(&outputVector[number], r);
    }
}

#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_x2_convert_8u_a_H */
