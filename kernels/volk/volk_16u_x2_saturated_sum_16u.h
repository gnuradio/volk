/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16u_x2_saturated_sum_16u
 *
 * \b Overview
 *
 * Adds two uint16_t vectors element-wise with saturation. Results are clamped
 * to the range [0, 65535] to prevent overflow wraparound.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16u_x2_saturated_sum_16u(uint16_t* outVector, const uint16_t* inVectorA,
 * const uint16_t* inVectorB, unsigned int num_points) \endcode
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
 *   uint16_t* a = (uint16_t*)volk_malloc(N * sizeof(uint16_t), align);
 *   uint16_t* b = (uint16_t*)volk_malloc(N * sizeof(uint16_t), align);
 *   uint16_t* result = (uint16_t*)volk_malloc(N * sizeof(uint16_t), align);
 *
 *   // Values that will cause saturation
 *   a[0] = 60000; b[0] = 10000; // 70000 -> saturates to 65535
 *   a[1] = 1000;  b[1] = 2000;  // 3000 -> no saturation
 *
 *   volk_16u_x2_saturated_sum_16u(result, a, b, N);
 *   // result[0] == 65535, result[1] == 3000
 *
 *   volk_free(a);
 *   volk_free(b);
 *   volk_free(result);
 * \endcode
 */

#ifndef INCLUDED_volk_16u_x2_saturated_sum_16u_u_H
#define INCLUDED_volk_16u_x2_saturated_sum_16u_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16u_x2_saturated_sum_16u_generic(uint16_t* outVector,
                                                         const uint16_t* inVectorA,
                                                         const uint16_t* inVectorB,
                                                         unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        uint32_t sum = (uint32_t)inVectorA[i] + (uint32_t)inVectorB[i];
        if (sum > 65535)
            sum = 65535;
        outVector[i] = (uint16_t)sum;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_u_sse2(uint16_t* outVector,
                                                        const uint16_t* inVectorA,
                                                        const uint16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        __m128i a = _mm_loadu_si128((const __m128i*)(inVectorA + 8 * number));
        __m128i b = _mm_loadu_si128((const __m128i*)(inVectorB + 8 * number));
        __m128i result = _mm_adds_epu16(a, b);
        _mm_storeu_si128((__m128i*)(outVector + 8 * number), result);
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_u_avx2(uint16_t* outVector,
                                                        const uint16_t* inVectorA,
                                                        const uint16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(inVectorA + 16 * number));
        __m256i b = _mm256_loadu_si256((const __m256i*)(inVectorB + 16 * number));
        __m256i result = _mm256_adds_epu16(a, b);
        _mm256_storeu_si256((__m256i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_u_avx512bw(uint16_t* outVector,
                                                            const uint16_t* inVectorA,
                                                            const uint16_t* inVectorB,
                                                            unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m512i a = _mm512_loadu_si512((const __m512i*)(inVectorA + 32 * number));
        __m512i b = _mm512_loadu_si512((const __m512i*)(inVectorB + 32 * number));
        __m512i result = _mm512_adds_epu16(a, b);
        _mm512_storeu_si512((__m512i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_16u_x2_saturated_sum_16u_u_H */


#ifndef INCLUDED_volk_16u_x2_saturated_sum_16u_a_H
#define INCLUDED_volk_16u_x2_saturated_sum_16u_a_H

#include <inttypes.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_a_sse2(uint16_t* outVector,
                                                        const uint16_t* inVectorA,
                                                        const uint16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        __m128i a = _mm_load_si128((const __m128i*)(inVectorA + 8 * number));
        __m128i b = _mm_load_si128((const __m128i*)(inVectorB + 8 * number));
        __m128i result = _mm_adds_epu16(a, b);
        _mm_store_si128((__m128i*)(outVector + 8 * number), result);
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_a_avx2(uint16_t* outVector,
                                                        const uint16_t* inVectorA,
                                                        const uint16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m256i a = _mm256_load_si256((const __m256i*)(inVectorA + 16 * number));
        __m256i b = _mm256_load_si256((const __m256i*)(inVectorB + 16 * number));
        __m256i result = _mm256_adds_epu16(a, b);
        _mm256_store_si256((__m256i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16u_x2_saturated_sum_16u_a_avx512bw(uint16_t* outVector,
                                                            const uint16_t* inVectorA,
                                                            const uint16_t* inVectorB,
                                                            unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m512i a = _mm512_load_si512((const __m512i*)(inVectorA + 32 * number));
        __m512i b = _mm512_load_si512((const __m512i*)(inVectorB + 32 * number));
        __m512i result = _mm512_adds_epu16(a, b);
        _mm512_store_si512((__m512i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16u_x2_saturated_sum_16u_neon(uint16_t* outVector,
                                                      const uint16_t* inVectorA,
                                                      const uint16_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        uint16x8_t a = vld1q_u16(inVectorA + 8 * number);
        uint16x8_t b = vld1q_u16(inVectorB + 8 * number);
        vst1q_u16(outVector + 8 * number, vqaddq_u16(a, b));
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        uint32_t sum = (uint32_t)inVectorA[number] + (uint32_t)inVectorB[number];
        if (sum > 65535)
            sum = 65535;
        outVector[number] = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_common.h>

static inline void volk_16u_x2_saturated_sum_16u_neonv8(uint16_t* outVector,
                                                        const uint16_t* inVectorA,
                                                        const uint16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __VOLK_PREFETCH(inVectorA + 32);
        __VOLK_PREFETCH(inVectorB + 32);
        uint16x8_t a0 = vld1q_u16(inVectorA);
        uint16x8_t b0 = vld1q_u16(inVectorB);
        uint16x8_t a1 = vld1q_u16(inVectorA + 8);
        uint16x8_t b1 = vld1q_u16(inVectorB + 8);
        vst1q_u16(outVector, vqaddq_u16(a0, b0));
        vst1q_u16(outVector + 8, vqaddq_u16(a1, b1));
        inVectorA += 16;
        inVectorB += 16;
        outVector += 16;
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        uint32_t sum = (uint32_t)(*inVectorA++) + (uint32_t)(*inVectorB++);
        if (sum > 65535)
            sum = 65535;
        *outVector++ = (uint16_t)sum;
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16u_x2_saturated_sum_16u_rvv(uint16_t* outVector,
                                                     const uint16_t* inVectorA,
                                                     const uint16_t* inVectorB,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inVectorA += vl, inVectorB += vl, outVector += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vuint16m8_t a = __riscv_vle16_v_u16m8(inVectorA, vl);
        vuint16m8_t b = __riscv_vle16_v_u16m8(inVectorB, vl);
        __riscv_vse16(outVector, __riscv_vsaddu(a, b, vl), vl);
    }
}

#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_16u_x2_saturated_sum_16u_a_H */
