/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_convert_32i
 *
 * \b Overview
 *
 * Converts a floating point number to a 32-bit integer after applying a
 * scaling factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_convert_32i(int32_t* outputVector, const float* inputVector, const
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
 * Convert floats from [-1,1] to integers with a scale of 5 to maintain smallest delta
 * \code
 *  int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   int32_t* out = (int32_t*)volk_malloc(sizeof(int32_t)*N, alignment);
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

#ifndef INCLUDED_volk_32f_s32f_convert_32i_u_H
#define INCLUDED_volk_32f_s32f_convert_32i_u_H

#include <inttypes.h>
#include <limits.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_convert_32i_generic(int32_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const float MAXIMUM = (float)INT32_MAX;
    const float MINIMUM = (float)INT32_MIN;
    const float FLOATING_NUMERIC_PRECISION_DIFFERENCE = 128.0f;
    const float LARGEST_FLOAT_INT = MAXIMUM - FLOATING_NUMERIC_PRECISION_DIFFERENCE;

    for (unsigned int number = 0; number < num_points; number++) {
        const float in = *inputVector++;
        const float rounded = rintf(fmaxf(in * scalar, MINIMUM));
        const int s = rounded > LARGEST_FLOAT_INT ? INT32_MAX : (int32_t)rounded;
        *outputVector++ = s;
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx512(int32_t* outputVector,
                                                      const float* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512i INTEGER_MAX = _mm512_set1_epi32(INT32_MAX);
    const __m512 LARGEST_FLOAT_INT = _mm512_set1_ps(2147483520.0f);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 input = _mm512_loadu_ps(inputVector);
        inputVector += 16;

        __m512 scaled = _mm512_mul_ps(input, vScalar);
        __mmask16 valid_mask = _mm512_cmp_ps_mask(scaled, LARGEST_FLOAT_INT, _CMP_LT_OQ);
        __m512i result =
            _mm512_mask_cvt_roundps_epi32(INTEGER_MAX,
                                          valid_mask,
                                          scaled,
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm512_storeu_si512(outputVector, result);
        outputVector += 16;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256i INTEGER_MAX = _mm256_set1_epi32(INT32_MAX);
    const __m256 LARGEST_FLOAT_INT = _mm256_set1_ps(2147483520.0f);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 input = _mm256_loadu_ps(inputVector);
        inputVector += 8;

        __m256 rounded = _mm256_round_ps(_mm256_mul_ps(input, vScalar),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 overflow_mask = _mm256_cmp_ps(rounded, LARGEST_FLOAT_INT, _CMP_GE_OQ);
        __m256i converted = _mm256_cvtps_epi32(rounded);
        __m256i result =
            _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(converted),
                                                 _mm256_castsi256_ps(INTEGER_MAX),
                                                 overflow_mask));

        _mm256_storeu_si256((__m256i*)outputVector, result);
        outputVector += 8;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float MINIMUM = INT32_MIN;
    const float MAXIMUM = INT32_MAX;
    const __m128 VMIN = _mm_set_ps1(MINIMUM);
    const __m128 VMAX = _mm_set_ps1(MAXIMUM);
    const __m128i VMAX_INT32 = _mm_set1_epi32(INT32_MAX);

    const __m128 vScalar = _mm_set_ps1(scalar);
    for (unsigned int number = 0; number < quarterPoints; number++) {
        const __m128 input = _mm_loadu_ps(inputVector);
        inputVector += 4;

        const __m128 scaled = _mm_mul_ps(input, vScalar);
        const __m128 overflow = _mm_cmpge_ps(scaled, VMAX);
        const __m128 clamped = _mm_max_ps(_mm_min_ps(scaled, VMAX), VMIN);
        const __m128i converted = _mm_cvtps_epi32(clamped);
        const __m128i overflow_mask = _mm_castps_si128(overflow);
        const __m128i output = _mm_or_si128(_mm_andnot_si128(overflow_mask, converted),
                                            _mm_and_si128(overflow_mask, VMAX_INT32));

        _mm_storeu_si128((__m128i*)outputVector, output);
        outputVector += 4;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float MINIMUM = INT32_MIN;
    const float MAXIMUM = INT32_MAX;
    const float FLOATING_NUMERIC_PRECISION_DIFFERENCE = 128.0f;
    const float LARGEST_FLOAT_INT = MAXIMUM - FLOATING_NUMERIC_PRECISION_DIFFERENCE;

    const __m128 VMIN = _mm_set_ps1(MINIMUM);
    const __m128 VMAX = _mm_set_ps1(MAXIMUM);

    const __m128 vScalar = _mm_set_ps1(scalar);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 ret = _mm_loadu_ps(inputVector);
        inputVector += 4;

        ret = _mm_mul_ps(ret, vScalar);
        ret = _mm_max_ps(_mm_min_ps(ret, VMAX), VMIN);

        _mm_store_ps(outputFloatBuffer, ret);
        for (unsigned int index = 0; index < 4; ++index) {
            *outputVector++ = outputFloatBuffer[index] > LARGEST_FLOAT_INT
                                  ? INT32_MAX
                                  : (int32_t)rintf(outputFloatBuffer[index]);
        }
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_volk_32f_s32f_convert_32i_u_H */
#ifndef INCLUDED_volk_32f_s32f_convert_32i_a_H
#define INCLUDED_volk_32f_s32f_convert_32i_a_H

#include <inttypes.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx512(int32_t* outputVector,
                                                      const float* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512i INTEGER_MAX = _mm512_set1_epi32(INT32_MAX);
    const __m512 LARGEST_FLOAT_INT = _mm512_set1_ps(2147483520.0f);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 input = _mm512_load_ps(inputVector);
        inputVector += 16;

        __m512 scaled = _mm512_mul_ps(input, vScalar);

        __mmask16 valid_mask = _mm512_cmp_ps_mask(scaled, LARGEST_FLOAT_INT, _CMP_LT_OQ);

        __m512i result =
            _mm512_mask_cvt_roundps_epi32(INTEGER_MAX,
                                          valid_mask,
                                          scaled,
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm512_store_si512(outputVector, result);
        outputVector += 16;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const __m256i INTEGER_MAX = _mm256_set1_epi32(INT32_MAX);
    const __m256 LARGEST_FLOAT_INT = _mm256_set1_ps(2147483520.0f);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 input = _mm256_load_ps(inputVector);
        inputVector += 8;

        __m256 rounded = _mm256_round_ps(_mm256_mul_ps(input, vScalar),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m256 overflow_mask = _mm256_cmp_ps(rounded, LARGEST_FLOAT_INT, _CMP_GE_OQ);

        __m256i converted = _mm256_cvtps_epi32(rounded);

        __m256i result =
            _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(converted),
                                                 _mm256_castsi256_ps(INTEGER_MAX),
                                                 overflow_mask));

        _mm256_store_si256((__m256i*)outputVector, result);
        outputVector += 8;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float MINIMUM = INT32_MIN;
    const float MAXIMUM = INT32_MAX;
    const __m128 VMIN = _mm_set_ps1(MINIMUM);
    const __m128 VMAX = _mm_set_ps1(MAXIMUM);
    const __m128i VMAX_INT32 = _mm_set1_epi32(INT32_MAX);

    const __m128 vScalar = _mm_set_ps1(scalar);
    for (unsigned int number = 0; number < quarterPoints; number++) {
        const __m128 input = _mm_load_ps(inputVector);
        inputVector += 4;

        const __m128 scaled = _mm_mul_ps(input, vScalar);
        const __m128 overflow = _mm_cmpge_ps(scaled, VMAX);
        const __m128 clamped = _mm_max_ps(_mm_min_ps(scaled, VMAX), VMIN);
        const __m128i converted = _mm_cvtps_epi32(clamped);
        const __m128i overflow_mask = _mm_castps_si128(overflow);
        const __m128i output = _mm_or_si128(_mm_andnot_si128(overflow_mask, converted),
                                            _mm_and_si128(overflow_mask, VMAX_INT32));

        _mm_store_si128((__m128i*)outputVector, output);
        outputVector += 4;
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float MINIMUM = INT32_MIN;
    const float MAXIMUM = INT32_MAX;
    const float FLOATING_NUMERIC_PRECISION_DIFFERENCE = 128.0f;
    const float LARGEST_FLOAT_INT = MAXIMUM - FLOATING_NUMERIC_PRECISION_DIFFERENCE;

    const __m128 VMIN = _mm_set_ps1(MINIMUM);
    const __m128 VMAX = _mm_set_ps1(MAXIMUM);

    const __m128 vScalar = _mm_set_ps1(scalar);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 ret = _mm_load_ps(inputVector);
        inputVector += 4;

        ret = _mm_mul_ps(ret, vScalar);
        ret = _mm_max_ps(_mm_min_ps(ret, VMAX), VMIN);

        _mm_store_ps(outputFloatBuffer, ret);
        for (unsigned int index = 0; index < 4; ++index) {
            *outputVector++ = outputFloatBuffer[index] > LARGEST_FLOAT_INT
                                  ? INT32_MAX
                                  : (int32_t)rintf(outputFloatBuffer[index]);
        }
    }

    volk_32f_s32f_convert_32i_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neon(int32_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t neg_half = vdupq_n_f32(-0.5f);
    float32x4_t zero = vdupq_n_f32(0.0f);

    for (; number < quarter_points; number++) {
        float32x4_t inputVal = vld1q_f32(inputPtr);
        inputVal = vmulq_f32(inputVal, vScalar);
        inputVal = vmaxq_f32(vminq_f32(inputVal, vmax_val), vmin_val);
        // Round to nearest: add copysign(0.5, x) before truncating
        uint32x4_t neg = vcltq_f32(inputVal, zero);
        inputVal = vaddq_f32(inputVal, vbslq_f32(neg, neg_half, half));
        int32x4_t intVal = vcvtq_s32_f32(inputVal);
        vst1q_s32(outputPtr, intVal);
        inputPtr += 4;
        outputPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neonv8(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < eighth_points; number++) {
        float32x4_t inputVal0 = vld1q_f32(inputPtr);
        float32x4_t inputVal1 = vld1q_f32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);

        inputVal0 = vmulq_f32(inputVal0, vScalar);
        inputVal1 = vmulq_f32(inputVal1, vScalar);
        inputVal0 = vmaxq_f32(vminq_f32(inputVal0, vmax_val), vmin_val);
        inputVal1 = vmaxq_f32(vminq_f32(inputVal1, vmax_val), vmin_val);

        int32x4_t intVal0 = vcvtnq_s32_f32(inputVal0);
        int32x4_t intVal1 = vcvtnq_s32_f32(inputVal1);

        vst1q_s32(outputPtr, intVal0);
        vst1q_s32(outputPtr + 4, intVal1);
        inputPtr += 8;
        outputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_convert_32i_rvv(int32_t* outputVector,
                                                 const float* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        v = __riscv_vfmul(v, scalar, vl);
        __riscv_vse32(outputVector, __riscv_vfcvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_convert_32i_a_H */
