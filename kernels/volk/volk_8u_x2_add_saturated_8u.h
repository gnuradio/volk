/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8u_x2_add_saturated_8u
 *
 * \b Overview
 *
 * Adds two uint8_t vectors element-wise with saturation. Results are clamped
 * to the range [0, 255] to prevent overflow wraparound.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_x2_add_saturated_8u(uint8_t* outVector, const uint8_t* inVectorA, const
 * uint8_t* inVectorB, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inVectorA: First input vector.
 * \li inVectorB: Second input vector.
 * \li num_points: Vector length.
 *
 * \b Outputs
 * \li outVector: Saturated sum output.
 *
 * \b Example
 * \code
 *   unsigned int N = 8;
 *   unsigned int align = volk_get_alignment();
 *   uint8_t* a = (uint8_t*)volk_malloc(N, align);
 *   uint8_t* b = (uint8_t*)volk_malloc(N, align);
 *   uint8_t* result = (uint8_t*)volk_malloc(N, align);
 *
 *   // Values that will cause saturation
 *   a[0] = 200; b[0] = 100; // 300 -> saturates to 255
 *   a[1] = 50;  b[1] = 30;  // 80 -> no saturation
 *
 *   volk_8u_x2_add_saturated_8u(result, a, b, N);
 *   // result[0] == 255, result[1] == 80
 *
 *   volk_free(a);
 *   volk_free(b);
 *   volk_free(result);
 * \endcode
 */

#ifndef INCLUDED_volk_8u_x2_add_saturated_8u_u_H
#define INCLUDED_volk_8u_x2_add_saturated_8u_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_8u_x2_add_saturated_8u_generic(uint8_t* outVector,
                                                       const uint8_t* inVectorA,
                                                       const uint8_t* inVectorB,
                                                       unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        uint8_t sum = inVectorA[i] + inVectorB[i];
        outVector[i] = sum | -(uint8_t)(sum < inVectorA[i]);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_8u_x2_add_saturated_8u_u_sse2(uint8_t* outVector,
                                                      const uint8_t* inVectorA,
                                                      const uint8_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m128i a = _mm_loadu_si128((const __m128i*)(inVectorA + 16 * number));
        __m128i b = _mm_loadu_si128((const __m128i*)(inVectorB + 16 * number));
        __m128i result = _mm_adds_epu8(a, b);
        _mm_storeu_si128((__m128i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_add_saturated_8u_u_avx2(uint8_t* outVector,
                                                      const uint8_t* inVectorA,
                                                      const uint8_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(inVectorA + 32 * number));
        __m256i b = _mm256_loadu_si256((const __m256i*)(inVectorB + 32 * number));
        __m256i result = _mm256_adds_epu8(a, b);
        _mm256_storeu_si256((__m256i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8u_x2_add_saturated_8u_u_avx512bw(uint8_t* outVector,
                                                          const uint8_t* inVectorA,
                                                          const uint8_t* inVectorB,
                                                          unsigned int num_points)
{
    const unsigned int sixtyfourthPoints = num_points / 64;
    unsigned int number = 0;

    for (; number < sixtyfourthPoints; number++) {
        __m512i a = _mm512_loadu_si512((const __m512i*)(inVectorA + 64 * number));
        __m512i b = _mm512_loadu_si512((const __m512i*)(inVectorB + 64 * number));
        __m512i result = _mm512_adds_epu8(a, b);
        _mm512_storeu_si512((__m512i*)(outVector + 64 * number), result);
    }

    for (number = sixtyfourthPoints * 64; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_8u_x2_add_saturated_8u_u_H */


#ifndef INCLUDED_volk_8u_x2_add_saturated_8u_a_H
#define INCLUDED_volk_8u_x2_add_saturated_8u_a_H

#include <inttypes.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_8u_x2_add_saturated_8u_a_sse2(uint8_t* outVector,
                                                      const uint8_t* inVectorA,
                                                      const uint8_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m128i a = _mm_load_si128((const __m128i*)(inVectorA + 16 * number));
        __m128i b = _mm_load_si128((const __m128i*)(inVectorB + 16 * number));
        __m128i result = _mm_adds_epu8(a, b);
        _mm_store_si128((__m128i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_add_saturated_8u_a_avx2(uint8_t* outVector,
                                                      const uint8_t* inVectorA,
                                                      const uint8_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m256i a = _mm256_load_si256((const __m256i*)(inVectorA + 32 * number));
        __m256i b = _mm256_load_si256((const __m256i*)(inVectorB + 32 * number));
        __m256i result = _mm256_adds_epu8(a, b);
        _mm256_store_si256((__m256i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8u_x2_add_saturated_8u_a_avx512bw(uint8_t* outVector,
                                                          const uint8_t* inVectorA,
                                                          const uint8_t* inVectorB,
                                                          unsigned int num_points)
{
    const unsigned int sixtyfourthPoints = num_points / 64;
    unsigned int number = 0;

    for (; number < sixtyfourthPoints; number++) {
        __m512i a = _mm512_load_si512((const __m512i*)(inVectorA + 64 * number));
        __m512i b = _mm512_load_si512((const __m512i*)(inVectorB + 64 * number));
        __m512i result = _mm512_adds_epu8(a, b);
        _mm512_store_si512((__m512i*)(outVector + 64 * number), result);
    }

    for (number = sixtyfourthPoints * 64; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8u_x2_add_saturated_8u_neon(uint8_t* outVector,
                                                    const uint8_t* inVectorA,
                                                    const uint8_t* inVectorB,
                                                    unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        uint8x16_t a = vld1q_u8(inVectorA + 16 * number);
        uint8x16_t b = vld1q_u8(inVectorB + 16 * number);
        vst1q_u8(outVector + 16 * number, vqaddq_u8(a, b));
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint16_t sum = (uint16_t)inVectorA[number] + (uint16_t)inVectorB[number];
        if (sum > 255)
            sum = 255;
        outVector[number] = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_common.h>

static inline void volk_8u_x2_add_saturated_8u_neonv8(uint8_t* outVector,
                                                      const uint8_t* inVectorA,
                                                      const uint8_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __VOLK_PREFETCH(inVectorA + 64);
        __VOLK_PREFETCH(inVectorB + 64);
        uint8x16_t a0 = vld1q_u8(inVectorA);
        uint8x16_t b0 = vld1q_u8(inVectorB);
        uint8x16_t a1 = vld1q_u8(inVectorA + 16);
        uint8x16_t b1 = vld1q_u8(inVectorB + 16);
        vst1q_u8(outVector, vqaddq_u8(a0, b0));
        vst1q_u8(outVector + 16, vqaddq_u8(a1, b1));
        inVectorA += 32;
        inVectorB += 32;
        outVector += 32;
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        uint16_t sum = (uint16_t)(*inVectorA++) + (uint16_t)(*inVectorB++);
        if (sum > 255)
            sum = 255;
        *outVector++ = (uint8_t)sum;
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8u_x2_add_saturated_8u_rvv(uint8_t* outVector,
                                                   const uint8_t* inVectorA,
                                                   const uint8_t* inVectorB,
                                                   unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inVectorA += vl, inVectorB += vl, outVector += vl) {
        vl = __riscv_vsetvl_e8m8(n);
        vuint8m8_t a = __riscv_vle8_v_u8m8(inVectorA, vl);
        vuint8m8_t b = __riscv_vle8_v_u8m8(inVectorB, vl);
        __riscv_vse8(outVector, __riscv_vsaddu(a, b, vl), vl);
    }
}

#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_8u_x2_add_saturated_8u_a_H */
