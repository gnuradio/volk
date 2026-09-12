/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_convert_32fc
 *
 * \b Overview
 *
 * Converts a complex vector of 16-bits integer each component
 * into a complex vector of 32-bits float each component.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_convert_32fc(lv_32fc_t* outputVector, const lv_16sc_t* inputVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector:  The complex 16-bit integer input data buffer.
 * \li num_points:   The number of data values to be converted.
 *
 * \b Outputs
 * \li outputVector: pointer to a vector holding the converted vector.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * unsigned int alignment = volk_get_alignment();
 * lv_16sc_t* input  = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t)*N, alignment);
 * lv_32fc_t* output  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 * volk_16ic_convert_32f(output, input, N);
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */


#ifndef INCLUDED_volk_16ic_convert_32fc_a_H
#define INCLUDED_volk_16ic_convert_32fc_a_H

#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_convert_32fc_generic(lv_32fc_t* outputVector,
                                                  const lv_16sc_t* inputVector,
                                                  unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        outputVector[number] = lv_cmake((float)lv_creal(inputVector[number]),
                                        (float)lv_cimag(inputVector[number]));
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 4;
    for (unsigned int number = 0; number < avx_iters; ++number) {
        const __m128i cplxValue = _mm_load_si128((__m128i*)inputVector);
        __VOLK_PREFETCH((const int16_t*)inputVector + 16);
        const __m256i outValInt = _mm256_cvtepi16_epi32(cplxValue);
        const __m256 outVal = _mm256_cvtepi32_ps(outValInt);
        _mm256_store_ps((float*)outputVector, outVal);
        inputVector += 4;
        outputVector += 4;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - avx_iters * 4);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx512(lv_32fc_t* outputVector,
                                                   const lv_16sc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;
    for (unsigned int number = 0; number < avx512_iters; ++number) {
        const __m256i cplxValue = _mm256_load_si256((__m256i*)inputVector);
        __VOLK_PREFETCH((const int16_t*)inputVector + 32);
        const __m512i outValInt = _mm512_cvtepi16_epi32(cplxValue);
        const __m512 outVal = _mm512_cvtepi32_ps(outValInt);
        _mm512_store_ps((float*)outputVector, outVal);
        inputVector += 8;
        outputVector += 8;
    }

    volk_16ic_convert_32fc_generic(
        outputVector, inputVector, num_points - avx512_iters * 8);
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16ic_convert_32fc_a_sse4_1(lv_32fc_t* outputVector,
                                                   const lv_16sc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;
    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128i cplxValue = _mm_load_si128((__m128i*)inputVector);
        inputVector += 4;

        const __m128i firstHalf = _mm_cvtepi16_epi32(cplxValue);
        const __m128 firstResult = _mm_cvtepi32_ps(firstHalf);
        _mm_store_ps((float*)outputVector, firstResult);
        outputVector += 2;

        const __m128i shiftedValue = _mm_srli_si128(cplxValue, 8);
        const __m128i secondHalf = _mm_cvtepi16_epi32(shiftedValue);
        const __m128 secondResult = _mm_cvtepi32_ps(secondHalf);
        _mm_store_ps((float*)outputVector, secondResult);
        outputVector += 2;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 4);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_a_sse2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;
    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128 a = _mm_set_ps((float)lv_cimag(inputVector[1]),
                                    (float)lv_creal(inputVector[1]),
                                    (float)lv_cimag(inputVector[0]),
                                    (float)lv_creal(inputVector[0]));
        _mm_store_ps((float*)outputVector, a);
        inputVector += 2;
        outputVector += 2;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 2);
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx(lv_32fc_t* outputVector,
                                                const lv_16sc_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m256 a = _mm256_set_ps((float)lv_cimag(inputVector[3]),
                                       (float)lv_creal(inputVector[3]),
                                       (float)lv_cimag(inputVector[2]),
                                       (float)lv_creal(inputVector[2]),
                                       (float)lv_cimag(inputVector[1]),
                                       (float)lv_creal(inputVector[1]),
                                       (float)lv_cimag(inputVector[0]),
                                       (float)lv_creal(inputVector[0]));
        _mm256_store_ps((float*)outputVector, a);
        inputVector += 4;
        outputVector += 4;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 4);
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_convert_32fc_neon(lv_32fc_t* outputVector,
                                               const lv_16sc_t* inputVector,
                                               unsigned int num_points)
{
    const unsigned int neon_iters = num_points / 8;
    for (unsigned int number = 0; number < neon_iters; ++number) {
        const int16x4_t v0 = vld1_s16((const int16_t*)inputVector);
        const int16x4_t v1 = vld1_s16((const int16_t*)inputVector + 4);
        const int16x4_t v2 = vld1_s16((const int16_t*)inputVector + 8);
        const int16x4_t v3 = vld1_s16((const int16_t*)inputVector + 12);
        __VOLK_PREFETCH((const int16_t*)inputVector + 32);

        vst1q_f32((float*)outputVector, vcvtq_f32_s32(vmovl_s16(v0)));
        vst1q_f32((float*)outputVector + 4, vcvtq_f32_s32(vmovl_s16(v1)));
        vst1q_f32((float*)outputVector + 8, vcvtq_f32_s32(vmovl_s16(v2)));
        vst1q_f32((float*)outputVector + 12, vcvtq_f32_s32(vmovl_s16(v3)));

        inputVector += 8;
        outputVector += 8;
    }

    volk_16ic_convert_32fc_generic(
        outputVector, inputVector, num_points - neon_iters * 8);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_convert_32fc_neonv8(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int neon_iters = num_points / 8;
    for (unsigned int number = 0; number < neon_iters; ++number) {
        const int16x4_t v0 = vld1_s16((const int16_t*)inputVector);
        const int16x4_t v1 = vld1_s16((const int16_t*)inputVector + 4);
        const int16x4_t v2 = vld1_s16((const int16_t*)inputVector + 8);
        const int16x4_t v3 = vld1_s16((const int16_t*)inputVector + 12);
        __VOLK_PREFETCH((const int16_t*)inputVector + 32);

        vst1q_f32((float*)outputVector, vcvtq_f32_s32(vmovl_s16(v0)));
        vst1q_f32((float*)outputVector + 4, vcvtq_f32_s32(vmovl_s16(v1)));
        vst1q_f32((float*)outputVector + 8, vcvtq_f32_s32(vmovl_s16(v2)));
        vst1q_f32((float*)outputVector + 12, vcvtq_f32_s32(vmovl_s16(v3)));

        inputVector += 8;
        outputVector += 8;
    }

    volk_16ic_convert_32fc_generic(
        outputVector, inputVector, num_points - neon_iters * 8);
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32fc_convert_16ic_a_H */

#ifndef INCLUDED_volk_16ic_convert_32fc_u_H
#define INCLUDED_volk_16ic_convert_32fc_u_H

#include <volk/volk_complex.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 4;
    for (unsigned int number = 0; number < avx_iters; ++number) {
        const __m128i cplxValue = _mm_loadu_si128((__m128i*)inputVector);
        __VOLK_PREFETCH((const int16_t*)inputVector + 16);
        const __m256i outValInt = _mm256_cvtepi16_epi32(cplxValue);
        const __m256 outVal = _mm256_cvtepi32_ps(outValInt);
        _mm256_storeu_ps((float*)outputVector, outVal);
        inputVector += 4;
        outputVector += 4;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - avx_iters * 4);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx512(lv_32fc_t* outputVector,
                                                   const lv_16sc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;
    for (unsigned int number = 0; number < avx512_iters; ++number) {
        const __m256i cplxValue = _mm256_loadu_si256((__m256i*)inputVector);
        __VOLK_PREFETCH((const int16_t*)inputVector + 32);

        const __m512i outValInt = _mm512_cvtepi16_epi32(cplxValue);
        const __m512 outVal = _mm512_cvtepi32_ps(outValInt);
        _mm512_storeu_ps((float*)outputVector, outVal);
        inputVector += 8;
        outputVector += 8;
    }

    volk_16ic_convert_32fc_generic(
        outputVector, inputVector, num_points - avx512_iters * 8);
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE4_1

static inline void volk_16ic_convert_32fc_u_sse4_1(lv_32fc_t* outputVector,
                                                   const lv_16sc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;
    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128i cplxValue = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 4;

        const __m128i firstHalf = _mm_cvtepi16_epi32(cplxValue);
        const __m128 firstResult = _mm_cvtepi32_ps(firstHalf);
        _mm_storeu_ps((float*)outputVector, firstResult);
        outputVector += 2;

        const __m128i shiftedValue = _mm_srli_si128(cplxValue, 8);
        const __m128i secondHalf = _mm_cvtepi16_epi32(shiftedValue);
        const __m128 secondResult = _mm_cvtepi32_ps(secondHalf);
        _mm_storeu_ps((float*)outputVector, secondResult);
        outputVector += 2;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 4);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_u_sse2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;
    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128 a = _mm_set_ps((float)lv_cimag(inputVector[1]),
                                    (float)lv_creal(inputVector[1]),
                                    (float)lv_cimag(inputVector[0]),
                                    (float)lv_creal(inputVector[0]));
        _mm_storeu_ps((float*)outputVector, a);
        inputVector += 2;
        outputVector += 2;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 2);
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx(lv_32fc_t* outputVector,
                                                const lv_16sc_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;
    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m256 a = _mm256_set_ps((float)lv_cimag(inputVector[3]),
                                       (float)lv_creal(inputVector[3]),
                                       (float)lv_cimag(inputVector[2]),
                                       (float)lv_creal(inputVector[2]),
                                       (float)lv_cimag(inputVector[1]),
                                       (float)lv_creal(inputVector[1]),
                                       (float)lv_cimag(inputVector[0]),
                                       (float)lv_creal(inputVector[0]));
        _mm256_storeu_ps((float*)outputVector, a);
        inputVector += 4;
        outputVector += 4;
    }

    volk_16ic_convert_32fc_generic(outputVector, inputVector, num_points - sse_iters * 4);
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_convert_32fc_rvv(lv_32fc_t* outputVector,
                                              const lv_16sc_t* inputVector,
                                              unsigned int num_points)
{
    const size_t num_values = num_points * 2;
    for (size_t number = 0; number < num_values;) {
        const size_t vector_length = __riscv_vsetvl_e16m4(num_values - number);
        const vint16m4_t input =
            __riscv_vle16_v_i16m4((const int16_t*)inputVector + number, vector_length);
        const vfloat32m8_t output = __riscv_vfwcvt_f(input, vector_length);
        __riscv_vse32((float*)outputVector + number, output, vector_length);
        number += vector_length;
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_convert_16ic_u_H */
