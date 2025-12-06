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
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_u_avx2(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m256 ret;
    __m128i inputVal128;
    __m256i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal128 = _mm_loadu_si128((__m128i*)inputVectorPtr);

        interimVal = _mm256_cvtepi8_epi32(inputVal128);
        ret = _mm256_cvtepi32_ps(interimVal);
        ret = _mm256_mul_ps(ret, invScalar);
        _mm256_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 8;

        inputVal128 = _mm_srli_si128(inputVal128, 8);
        interimVal = _mm256_cvtepi8_epi32(inputVal128);
        ret = _mm256_cvtepi32_ps(interimVal);
        ret = _mm256_mul_ps(ret, invScalar);
        _mm256_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 8;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_u_avx512(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m512 ret;
    __m128i inputVal128;
    __m512i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal128 = _mm_loadu_si128((__m128i*)inputVectorPtr);

        interimVal = _mm512_cvtepi8_epi32(inputVal128);
        ret = _mm512_cvtepi32_ps(interimVal);
        ret = _mm512_mul_ps(ret, invScalar);
        _mm512_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 16;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_s32f_convert_32f_u_sse4_1(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m128 ret;
    __m128i inputVal;
    __m128i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_loadu_si128((__m128i*)inputVectorPtr);

        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_GENERIC

static inline void volk_8i_s32f_convert_32f_generic(float* outputVector,
                                                    const int8_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;
    unsigned int number = 0;
    const float iScalar = 1.0 / scalar;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_VOLK_8s_CONVERT_32f_UNALIGNED8_H */

#ifndef INCLUDED_volk_8i_s32f_convert_32f_a_H
#define INCLUDED_volk_8i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_a_avx2(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m256 ret;
    __m128i inputVal128;
    __m256i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal128 = _mm_load_si128((__m128i*)inputVectorPtr);

        interimVal = _mm256_cvtepi8_epi32(inputVal128);
        ret = _mm256_cvtepi32_ps(interimVal);
        ret = _mm256_mul_ps(ret, invScalar);
        _mm256_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 8;

        inputVal128 = _mm_srli_si128(inputVal128, 8);
        interimVal = _mm256_cvtepi8_epi32(inputVal128);
        ret = _mm256_cvtepi32_ps(interimVal);
        ret = _mm256_mul_ps(ret, invScalar);
        _mm256_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 8;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8i_s32f_convert_32f_a_avx512(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m512 ret;
    __m128i inputVal128;
    __m512i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal128 = _mm_load_si128((__m128i*)inputVectorPtr);

        interimVal = _mm512_cvtepi8_epi32(inputVal128);
        ret = _mm512_cvtepi32_ps(interimVal);
        ret = _mm512_mul_ps(ret, invScalar);
        _mm512_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 16;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_s32f_convert_32f_a_sse4_1(float* outputVector,
                                                     const int8_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* inputVectorPtr = inputVector;
    __m128 ret;
    __m128i inputVal;
    __m128i interimVal;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_load_si128((__m128i*)inputVectorPtr);

        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVal = _mm_srli_si128(inputVal, 4);
        interimVal = _mm_cvtepi8_epi32(inputVal);
        ret = _mm_cvtepi32_ps(interimVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        inputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * iScalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8i_s32f_convert_32f_neon(float* outputVector,
                                                 const int8_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;

    const float iScalar = 1.0 / scalar;
    const float32x4_t qiScalar = vdupq_n_f32(iScalar);

    int8x16_t inputVal;

    int16x8_t lower;
    int16x8_t higher;

    float32x4_t outputFloat;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    for (; number < sixteenthPoints; number++) {
        inputVal = vld1q_s8(inputVectorPtr);
        inputVectorPtr += 16;

        lower = vmovl_s8(vget_low_s8(inputVal));
        higher = vmovl_s8(vget_high_s8(inputVal));

        outputFloat = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower))), qiScalar);
        vst1q_f32(outputVectorPtr, outputFloat);
        outputVectorPtr += 4;

        outputFloat = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower))), qiScalar);
        vst1q_f32(outputVectorPtr, outputFloat);
        outputVectorPtr += 4;

        outputFloat = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(higher))), qiScalar);
        vst1q_f32(outputVectorPtr, outputFloat);
        outputVectorPtr += 4;

        outputFloat =
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(higher))), qiScalar);
        vst1q_f32(outputVectorPtr, outputFloat);
        outputVectorPtr += 4;
    }
    for (number = sixteenthPoints * 16; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_8i_s32f_convert_32f_neonv8(float* outputVector,
                                                   const int8_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;
    const float iScalar = 1.0f / scalar;
    const float32x4_t qiScalar = vdupq_n_f32(iScalar);
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        int8x16_t in0 = vld1q_s8(inputVectorPtr);
        int8x16_t in1 = vld1q_s8(inputVectorPtr + 16);
        __VOLK_PREFETCH(inputVectorPtr + 64);

        /* Widen int8 -> int16 -> int32 -> float */
        int16x8_t lo0 = vmovl_s8(vget_low_s8(in0));
        int16x8_t hi0 = vmovl_s8(vget_high_s8(in0));
        int16x8_t lo1 = vmovl_s8(vget_low_s8(in1));
        int16x8_t hi1 = vmovl_s8(vget_high_s8(in1));

        vst1q_f32(outputVectorPtr,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))), qiScalar));
        vst1q_f32(outputVectorPtr + 4,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), qiScalar));
        vst1q_f32(outputVectorPtr + 8,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))), qiScalar));
        vst1q_f32(outputVectorPtr + 12,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), qiScalar));
        vst1q_f32(outputVectorPtr + 16,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))), qiScalar));
        vst1q_f32(outputVectorPtr + 20,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), qiScalar));
        vst1q_f32(outputVectorPtr + 24,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))), qiScalar));
        vst1q_f32(outputVectorPtr + 28,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), qiScalar));

        inputVectorPtr += 32;
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
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
