/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8i_convert_16i
 *
 * \b Overview
 *
 * Convert the input vector of 8-bit chars to a vector of 16-bit
 * shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_convert_16i(int16_t* outputVector, const int8_t* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 8-bit chars.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li outputVector: The output 16-bit shorts.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8i_convert_16i();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_convert_16i_u_H
#define INCLUDED_volk_8i_convert_16i_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_8i_convert_16i_generic(int16_t* outputVector,
                                               const int8_t* inputVector,
                                               unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = ((int16_t)(*inputVector++)) * 256;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_convert_16i_u_avx2(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_loadu_si128((const __m128i*)inputVector);
        const __m256i output = _mm256_slli_epi16(_mm256_cvtepi8_epi16(input), 8);
        _mm256_storeu_si256((__m256i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8i_convert_16i_u_avx512bw(int16_t* outputVector,
                                                  const int8_t* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m256i input = _mm256_loadu_si256((const __m256i*)inputVector);
        const __m512i output = _mm512_slli_epi16(_mm512_cvtepi8_epi16(input), 8);
        _mm512_storeu_si512((__m512i*)outputVector, output);
        inputVector += 32;
        outputVector += 32;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_convert_16i_u_sse4_1(int16_t* outputVector,
                                                const int8_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_loadu_si128((const __m128i*)inputVector);
        const __m128i output0 = _mm_slli_epi16(_mm_cvtepi8_epi16(input), 8);
        const __m128i output1 =
            _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(input, 8)), 8);
        _mm_storeu_si128((__m128i*)outputVector, output0);
        _mm_storeu_si128((__m128i*)(outputVector + 8), output1);
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE4_1 */

#endif /* INCLUDED_VOLK_8s_CONVERT_16s_UNALIGNED8_H */


#ifndef INCLUDED_volk_8i_convert_16i_a_H
#define INCLUDED_volk_8i_convert_16i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_convert_16i_a_avx2(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_load_si128((const __m128i*)inputVector);
        const __m256i output = _mm256_slli_epi16(_mm256_cvtepi8_epi16(input), 8);
        _mm256_store_si256((__m256i*)outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8i_convert_16i_a_avx512bw(int16_t* outputVector,
                                                  const int8_t* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const __m256i input = _mm256_load_si256((const __m256i*)inputVector);
        const __m512i output = _mm512_slli_epi16(_mm512_cvtepi8_epi16(input), 8);
        _mm512_store_si512((__m512i*)outputVector, output);
        inputVector += 32;
        outputVector += 32;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_convert_16i_a_sse4_1(int16_t* outputVector,
                                                const int8_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const __m128i input = _mm_load_si128((const __m128i*)inputVector);
        const __m128i output0 = _mm_slli_epi16(_mm_cvtepi8_epi16(input), 8);
        const __m128i output1 =
            _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(input, 8)), 8);
        _mm_store_si128((__m128i*)outputVector, output0);
        _mm_store_si128((__m128i*)(outputVector + 8), output1);
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8i_convert_16i_neon(int16_t* outputVector,
                                            const int8_t* inputVector,
                                            unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    // NEON doesn't have a concept of 8 bit registers, so we are really
    // dealing with the low half of 16-bit registers. Since this requires
    // a move instruction we likely do better with ASM here.
    for (unsigned int number = 0; number < eighth_points; ++number) {
        const int8x8_t input = vld1_s8(inputVector);
        const int16x8_t output = vshlq_n_s16(vmovl_s8(input), 8);
        vst1q_s16(outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - eighth_points * 8);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_8i_convert_16i_neonv8(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {
        const int8x16_t input = vld1q_s8(inputVector);
        __VOLK_PREFETCH(inputVector + 32);
        const int16x8_t output0 = vshll_n_s8(vget_low_s8(input), 8);
        const int16x8_t output1 = vshll_n_s8(vget_high_s8(input), 8);
        vst1q_s16(outputVector, output0);
        vst1q_s16(outputVector + 8, output1);
        inputVector += 16;
        outputVector += 16;
    }

    volk_8i_convert_16i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_ORC
extern void volk_8i_convert_16i_a_orc_impl(int16_t* outputVector,
                                           const int8_t* inputVector,
                                           int num_points);

static inline void volk_8i_convert_16i_u_orc(int16_t* outputVector,
                                             const int8_t* inputVector,
                                             unsigned int num_points)
{
    volk_8i_convert_16i_a_orc_impl(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_ORC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8i_convert_16i_rvv(int16_t* outputVector,
                                           const int8_t* inputVector,
                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e8m4(n);
        vint16m8_t v = __riscv_vsext_vf2(__riscv_vle8_v_i8m4(inputVector, vl), vl);
        __riscv_vse16(outputVector, __riscv_vsll(v, 8, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_VOLK_8s_CONVERT_16s_ALIGNED8_H */
