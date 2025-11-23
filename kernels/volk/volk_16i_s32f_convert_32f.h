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
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((__m128i*)inputPtr);

        // Convert
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    __m512 invScalar = _mm512_set1_ps(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m256i inputVal;
    __m512i inputVal2;
    __m512 ret;

    for (; number < sixteenthPoints; number++) {

        // Load 16 int16 values
        inputVal = _mm256_loadu_si256((__m256i*)inputPtr);

        // Convert int16 → int32 → float
        inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        ret = _mm512_cvtepi32_ps(inputVal2);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal, inputVal2;
    __m128 ret;
    __m256 output;
    __m256 dummy = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        // inputVal = _mm_loadu_si128((__m128i*)inputPtr);
        inputVal = _mm_loadu_si128((__m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(dummy, ret, 0);

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(output, ret, 1);

        _mm256_storeu_ps(outputVectorPtr, output);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal;
    __m128i inputVal2;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((__m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        ret = _mm_set_ps((float)(inputPtr[3]),
                         (float)(inputPtr[2]),
                         (float)(inputPtr[1]),
                         (float)(inputPtr[0]));

        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        inputPtr += 4;
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) / scalar;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_s32f_convert_32f_generic(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const int16_t* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) / scalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_s32f_convert_32f_neon(float* outputVector,
                                                  const int16_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    float* outputPtr = outputVector;
    const int16_t* inputPtr = inputVector;
    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;

    int16x4x2_t input16;
    int32x4_t input32_0, input32_1;
    float32x4_t input_float_0, input_float_1;
    float32x4x2_t output_float;
    float32x4_t inv_scale;

    inv_scale = vdupq_n_f32(1.0 / scalar);

    // the generic disassembles to a 128-bit load
    // and duplicates every instruction to operate on 64-bits
    // at a time. This is only possible with lanes, which is faster
    // than just doing a vld1_s16, but still slower.
    for (number = 0; number < eighth_points; number++) {
        input16 = vld2_s16(inputPtr);
        // widen 16-bit int to 32-bit int
        input32_0 = vmovl_s16(input16.val[0]);
        input32_1 = vmovl_s16(input16.val[1]);
        // convert 32-bit int to float with scale
        input_float_0 = vcvtq_f32_s32(input32_0);
        input_float_1 = vcvtq_f32_s32(input32_1);
        output_float.val[0] = vmulq_f32(input_float_0, inv_scale);
        output_float.val[1] = vmulq_f32(input_float_1, inv_scale);
        vst2q_f32(outputPtr, output_float);
        inputPtr += 8;
        outputPtr += 8;
    }

    for (number = eighth_points * 8; number < num_points; number++) {
        *outputPtr++ = ((float)(*inputPtr++)) / scalar;
    }
}
#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_16i_s32f_convert_32f_u_H */
#ifndef INCLUDED_volk_16i_s32f_convert_32f_a_H
#define INCLUDED_volk_16i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_load_si128((__m128i*)inputPtr);

        // Convert
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    __m512 invScalar = _mm512_set1_ps(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m256i inputVal;
    __m512i inputVal2;
    __m512 ret;

    for (; number < sixteenthPoints; number++) {

        // Load 16 int16 values
        inputVal = _mm256_load_si256((__m256i*)inputPtr);

        // Convert int16 → int32 → float
        inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        ret = _mm512_cvtepi32_ps(inputVal2);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal, inputVal2;
    __m128 ret;
    __m256 output;
    __m256 dummy = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        // inputVal = _mm_loadu_si128((__m128i*)inputPtr);
        inputVal = _mm_load_si128((__m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(dummy, ret, 0);

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(output, ret, 1);

        _mm256_store_ps(outputVectorPtr, output);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal;
    __m128i inputVal2;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((__m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) / scalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        ret = _mm_set_ps((float)(inputPtr[3]),
                         (float)(inputPtr[2]),
                         (float)(inputPtr[1]),
                         (float)(inputPtr[0]));

        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        inputPtr += 4;
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) / scalar;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_s32f_convert_32f_rvv(float* outputVector,
                                                 const int16_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t v = __riscv_vfwcvt_f(__riscv_vle16_v_i16m4(inputVector, vl), vl);
        __riscv_vse32(outputVector, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16i_s32f_convert_32f_a_H */
