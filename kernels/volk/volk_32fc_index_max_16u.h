/* -*- c++ -*- */
/*
 * Copyright 2012, 2014-2016, 2018-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_index_max_16u
 *
 * \b Overview
 *
 * Returns Argmax_i mag(x[i]). Finds and returns the index which contains the
 * maximum magnitude for complex points in the given vector.
 *
 * Note that num_points is a uint32_t, but the return value is
 * uint16_t. Providing a vector larger than the max of a uint16_t
 * (65536) would miss anything outside of this boundary. The kernel
 * will check the length of num_points and cap it to this max value,
 * anyways.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_index_max_16u(uint16_t* target, const lv_32fc_t* src0, uint32_t
 * num_points) \endcode
 *
 * \b Inputs
 * \li src0: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li target: The index of the point with maximum magnitude.
 *
 * \b Example
 * Calculate the index of the maximum value of \f$x^2 + x\f$ for points around
 * the unit circle.
 * \code
 *   int N = 10;
 *   uint32_t alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   uint16_t* max = (uint16_t*)volk_malloc(sizeof(uint16_t), alignment);
 *
 *   for(uint32_t ii = 0; ii < N/2; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii] = in[ii] * in[ii] + in[ii];
 *       in[N-ii] = lv_cmake(real, imag);
 *       in[N-ii] = in[N-ii] * in[N-ii] + in[N-ii];
 *   }
 *
 *   volk_32fc_index_max_16u(max, in, N);
 *
 *   printf("index of max value = %u\n",  *max);
 *
 *   volk_free(in);
 *   volk_free(max);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_index_max_16u_a_H
#define INCLUDED_volk_32fc_index_max_16u_a_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_max_16u_a_avx2_variant_0(uint16_t* target,
                                                            const lv_32fc_t* src0,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_max_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 max_values = _mm256_setzero_ps();
    __m256i max_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_load_ps((float*)src0);
        __m256 in1 = _mm256_load_ps((float*)(src0 + 4));
        vector_32fc_index_max_variant0(
            in0, in1, &max_values, &max_indices, &current_indices, indices_increment);
        src0 += 8;
    }

    // determine maximum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float max_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t max_indices_buffer[8];
    _mm256_store_ps(max_values_buffer, max_values);
    _mm256_store_si256((__m256i*)max_indices_buffer, max_indices);

    float max = 0.f;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (max_values_buffer[i] > max) {
            max = max_values_buffer[i];
            index = max_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*src0) * lv_creal(*src0) + lv_cimag(*src0) * lv_cimag(*src0);
        if (abs_squared > max) {
            max = abs_squared;
            index = i;
        }
        ++src0;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_max_16u_a_avx2_variant_1(uint16_t* target,
                                                            const lv_32fc_t* src0,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_max_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 max_values = _mm256_setzero_ps();
    __m256i max_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_load_ps((float*)src0);
        __m256 in1 = _mm256_load_ps((float*)(src0 + 4));
        vector_32fc_index_max_variant1(
            in0, in1, &max_values, &max_indices, &current_indices, indices_increment);
        src0 += 8;
    }

    // determine maximum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float max_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t max_indices_buffer[8];
    _mm256_store_ps(max_values_buffer, max_values);
    _mm256_store_si256((__m256i*)max_indices_buffer, max_indices);

    float max = 0.f;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (max_values_buffer[i] > max) {
            max = max_values_buffer[i];
            index = max_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*src0) * lv_creal(*src0) + lv_cimag(*src0) * lv_cimag(*src0);
        if (abs_squared > max) {
            max = abs_squared;
            index = i;
        }
        ++src0;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_index_max_16u_a_sse3(uint16_t* target,
                                                  const lv_32fc_t* src0,
                                                  uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
    const uint32_t num_bytes = num_points * 8;

    union bit128 holderf;
    union bit128 holderi;
    float sq_dist = 0.0;

    union bit128 xmm5, xmm4;
    __m128 xmm1, xmm2, xmm3;
    __m128i xmm8, xmm11, xmm12, xmm9, xmm10;

    xmm5.int_vec = _mm_setzero_si128();
    xmm4.int_vec = _mm_setzero_si128();
    holderf.int_vec = _mm_setzero_si128();
    holderi.int_vec = _mm_setzero_si128();

    int bound = num_bytes >> 5;
    int i = 0;

    xmm8 = _mm_setr_epi32(0, 1, 2, 3);
    xmm9 = _mm_setzero_si128();
    xmm10 = _mm_setr_epi32(4, 4, 4, 4);
    xmm3 = _mm_setzero_ps();

    for (; i < bound; ++i) {
        xmm1 = _mm_load_ps((float*)src0);
        xmm2 = _mm_load_ps((float*)&src0[2]);

        src0 += 4;

        xmm1 = _mm_mul_ps(xmm1, xmm1);
        xmm2 = _mm_mul_ps(xmm2, xmm2);

        xmm1 = _mm_hadd_ps(xmm1, xmm2);

        xmm3 = _mm_max_ps(xmm1, xmm3);

        xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);

        xmm8 = _mm_add_epi32(xmm8, xmm10);
    }

    if (num_bytes >> 4 & 1) {
        xmm2 = _mm_load_ps((float*)src0);

        xmm1 = _mm_movelh_ps(bit128_p(&xmm8)->float_vec, bit128_p(&xmm8)->float_vec);
        xmm8 = bit128_p(&xmm1)->int_vec;

        xmm2 = _mm_mul_ps(xmm2, xmm2);

        src0 += 2;

        xmm1 = _mm_hadd_ps(xmm2, xmm2);

        xmm3 = _mm_max_ps(xmm1, xmm3);

        xmm10 = _mm_setr_epi32(2, 2, 2, 2);

        xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);

        xmm8 = _mm_add_epi32(xmm8, xmm10);
    }

    if (num_bytes >> 3 & 1) {
        sq_dist =
            lv_creal(src0[0]) * lv_creal(src0[0]) + lv_cimag(src0[0]) * lv_cimag(src0[0]);

        xmm2 = _mm_load1_ps(&sq_dist);

        xmm1 = xmm3;

        xmm3 = _mm_max_ss(xmm3, xmm2);

        xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm8 = _mm_shuffle_epi32(xmm8, 0x00);

        xmm11 = _mm_and_si128(xmm8, xmm4.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm5.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);
    }

    _mm_store_ps((float*)&(holderf.f), xmm3);
    _mm_store_si128(&(holderi.int_vec), xmm9);

    target[0] = holderi.i[0];
    sq_dist = holderf.f[0];
    target[0] = (holderf.f[1] > sq_dist) ? holderi.i[1] : target[0];
    sq_dist = (holderf.f[1] > sq_dist) ? holderf.f[1] : sq_dist;
    target[0] = (holderf.f[2] > sq_dist) ? holderi.i[2] : target[0];
    sq_dist = (holderf.f[2] > sq_dist) ? holderf.f[2] : sq_dist;
    target[0] = (holderf.f[3] > sq_dist) ? holderi.i[3] : target[0];
    sq_dist = (holderf.f[3] > sq_dist) ? holderf.f[3] : sq_dist;
}

#endif /*LV_HAVE_SSE3*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <float.h>
#include <limits.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32fc_index_max_16u_neon(uint16_t* target, const lv_32fc_t* src0, uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    if (num_points == 0)
        return;

    const uint32_t quarter_points = num_points / 4;
    const lv_32fc_t* inputPtr = src0;

    // Use integer indices directly
    uint32x4_t vec_indices = { 0, 1, 2, 3 };
    const uint32x4_t vec_incr = vdupq_n_u32(4);

    float32x4_t vec_max = vdupq_n_f32(0.0f);
    uint32x4_t vec_max_idx = vdupq_n_u32(0);

    for (uint32_t i = 0; i < quarter_points; i++) {
        // Load and deinterleave complex values
        float32x4x2_t cplx = vld2q_f32((const float*)inputPtr);
        inputPtr += 4;

        // Magnitude squared: re*re + im*im
        float32x4_t mag2 =
            vmlaq_f32(vmulq_f32(cplx.val[0], cplx.val[0]), cplx.val[1], cplx.val[1]);

        // Compare BEFORE max update to know which lanes change
        uint32x4_t gt_mask = vcgtq_f32(mag2, vec_max);
        vec_max_idx = vbslq_u32(gt_mask, vec_indices, vec_max_idx);

        // vmaxq_f32 is single-cycle, no dependency on comparison result
        vec_max = vmaxq_f32(mag2, vec_max);

        vec_indices = vaddq_u32(vec_indices, vec_incr);
    }

    // Scalar reduction
    __VOLK_ATTR_ALIGNED(16) float max_buf[4];
    __VOLK_ATTR_ALIGNED(16) uint32_t idx_buf[4];
    vst1q_f32(max_buf, vec_max);
    vst1q_u32(idx_buf, vec_max_idx);

    float max_val = max_buf[0];
    uint32_t result_idx = idx_buf[0];
    for (int i = 1; i < 4; i++) {
        if (max_buf[i] > max_val) {
            max_val = max_buf[i];
            result_idx = idx_buf[i];
        } else if (max_buf[i] == max_val && idx_buf[i] < result_idx) {
            result_idx = idx_buf[i];
        }
    }

    // Handle tail
    for (uint32_t i = quarter_points * 4; i < num_points; i++) {
        float re = lv_creal(src0[i]);
        float im = lv_cimag(src0[i]);
        float mag2 = re * re + im * im;
        if (mag2 > max_val) {
            max_val = mag2;
            result_idx = i;
        }
    }

    *target = (uint16_t)result_idx;
}

#endif /*LV_HAVE_NEON*/


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <float.h>
#include <limits.h>

static inline void volk_32fc_index_max_16u_neonv8(uint16_t* target,
                                                  const lv_32fc_t* src0,
                                                  uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    if (num_points == 0)
        return;

    const uint32_t quarter_points = num_points / 4;
    const lv_32fc_t* inputPtr = src0;

    // Use integer indices directly (no float conversion overhead)
    uint32x4_t vec_indices = { 0, 1, 2, 3 };
    const uint32x4_t vec_incr = vdupq_n_u32(4);

    float32x4_t vec_max = vdupq_n_f32(0.0f);
    uint32x4_t vec_max_idx = vdupq_n_u32(0);

    for (uint32_t i = 0; i < quarter_points; i++) {
        // Load and deinterleave complex values
        float32x4x2_t cplx = vld2q_f32((const float*)inputPtr);
        inputPtr += 4;

        // Magnitude squared using FMA: re*re + im*im
        float32x4_t mag2 =
            vfmaq_f32(vmulq_f32(cplx.val[0], cplx.val[0]), cplx.val[1], cplx.val[1]);

        // Compare BEFORE max update to know which lanes change
        uint32x4_t gt_mask = vcgtq_f32(mag2, vec_max);
        vec_max_idx = vbslq_u32(gt_mask, vec_indices, vec_max_idx);

        // vmaxq_f32 is single-cycle, no dependency on comparison result
        vec_max = vmaxq_f32(mag2, vec_max);

        vec_indices = vaddq_u32(vec_indices, vec_incr);
    }

    // ARMv8 horizontal reduction
    float max_val = vmaxvq_f32(vec_max);
    uint32x4_t max_mask = vceqq_f32(vec_max, vdupq_n_f32(max_val));
    uint32x4_t idx_masked = vbslq_u32(max_mask, vec_max_idx, vdupq_n_u32(UINT32_MAX));
    uint32_t result_idx = vminvq_u32(idx_masked);

    // Handle tail
    for (uint32_t i = quarter_points * 4; i < num_points; i++) {
        float re = lv_creal(src0[i]);
        float im = lv_cimag(src0[i]);
        float mag2 = re * re + im * im;
        if (mag2 > max_val) {
            max_val = mag2;
            result_idx = i;
        }
    }

    *target = (uint16_t)result_idx;
}

#endif /*LV_HAVE_NEONV8*/


#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_index_max_16u_generic(uint16_t* target,
                                                   const lv_32fc_t* src0,
                                                   uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const uint32_t num_bytes = num_points * 8;

    float sq_dist = 0.0;
    float max = 0.0;
    uint16_t index = 0;

    uint32_t i = 0;

    for (; i < (num_bytes >> 3); ++i) {
        sq_dist =
            lv_creal(src0[i]) * lv_creal(src0[i]) + lv_cimag(src0[i]) * lv_cimag(src0[i]);

        if (sq_dist > max) {
            index = i;
            max = sq_dist;
        }
    }
    target[0] = index;
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <limits.h>

static inline void volk_32fc_index_max_16u_a_avx512f(uint16_t* target,
                                                     const lv_32fc_t* src0,
                                                     uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const lv_32fc_t* src0Ptr = src0;
    const uint32_t sixteenthPoints = num_points / 16;

    // Index ordering after shuffle: [0,1,8,9, 2,3,10,11, 4,5,12,13, 6,7,14,15]
    __m512 currentIndexes =
        _mm512_setr_ps(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    const __m512 indexIncrement = _mm512_set1_ps(16);

    __m512 maxValues = _mm512_setzero_ps();
    __m512 maxIndices = _mm512_setzero_ps();

    for (uint32_t number = 0; number < sixteenthPoints; number++) {
        // Load 16 complex values (32 floats)
        __m512 in0 = _mm512_load_ps((const float*)src0Ptr);
        __m512 in1 = _mm512_load_ps((const float*)(src0Ptr + 8));
        src0Ptr += 16;

        // Square all values
        in0 = _mm512_mul_ps(in0, in0);
        in1 = _mm512_mul_ps(in1, in1);

        // Add adjacent pairs (re² + im²) using within-lane shuffle
        // 0xB1 = _MM_SHUFFLE(2,3,0,1) swaps adjacent elements
        __m512 sw0 = _mm512_shuffle_ps(in0, in0, 0xB1);
        __m512 sw1 = _mm512_shuffle_ps(in1, in1, 0xB1);
        __m512 sum0 = _mm512_add_ps(in0, sw0);
        __m512 sum1 = _mm512_add_ps(in1, sw1);

        // Compact: pick elements 0,2 from sum0 and sum1 per 128-bit lane
        // 0x88 = _MM_SHUFFLE(2,0,2,0)
        __m512 mag_sq = _mm512_shuffle_ps(sum0, sum1, 0x88);

        // Compare and update maximums
        __mmask16 cmpMask = _mm512_cmp_ps_mask(mag_sq, maxValues, _CMP_GT_OS);
        maxIndices = _mm512_mask_blend_ps(cmpMask, maxIndices, currentIndexes);
        maxValues = _mm512_max_ps(mag_sq, maxValues);

        currentIndexes = _mm512_add_ps(currentIndexes, indexIncrement);
    }

    // Reduce 16 values to find maximum
    __VOLK_ATTR_ALIGNED(64) float maxValuesBuffer[16];
    __VOLK_ATTR_ALIGNED(64) float maxIndexesBuffer[16];
    _mm512_store_ps(maxValuesBuffer, maxValues);
    _mm512_store_ps(maxIndexesBuffer, maxIndices);

    float max = 0.0f;
    uint32_t index = 0;
    for (uint32_t i = 0; i < 16; i++) {
        if (maxValuesBuffer[i] > max) {
            max = maxValuesBuffer[i];
            index = (uint32_t)maxIndexesBuffer[i];
        } else if (maxValuesBuffer[i] == max) {
            if ((uint32_t)maxIndexesBuffer[i] < index)
                index = (uint32_t)maxIndexesBuffer[i];
        }
    }

    // Handle tail
    for (uint32_t number = sixteenthPoints * 16; number < num_points; number++) {
        const float re = lv_creal(*src0Ptr);
        const float im = lv_cimag(*src0Ptr);
        const float sq_dist = re * re + im * im;
        if (sq_dist > max) {
            max = sq_dist;
            index = number;
        }
        src0Ptr++;
    }
    *target = (uint16_t)index;
}

#endif /*LV_HAVE_AVX512F*/

#endif /*INCLUDED_volk_32fc_index_max_16u_a_H*/

#ifndef INCLUDED_volk_32fc_index_max_16u_u_H
#define INCLUDED_volk_32fc_index_max_16u_u_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_max_16u_u_avx2_variant_0(uint16_t* target,
                                                            const lv_32fc_t* src0,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_max_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 max_values = _mm256_setzero_ps();
    __m256i max_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_loadu_ps((float*)src0);
        __m256 in1 = _mm256_loadu_ps((float*)(src0 + 4));
        vector_32fc_index_max_variant0(
            in0, in1, &max_values, &max_indices, &current_indices, indices_increment);
        src0 += 8;
    }

    // determine maximum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float max_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t max_indices_buffer[8];
    _mm256_store_ps(max_values_buffer, max_values);
    _mm256_store_si256((__m256i*)max_indices_buffer, max_indices);

    float max = 0.f;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (max_values_buffer[i] > max) {
            max = max_values_buffer[i];
            index = max_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*src0) * lv_creal(*src0) + lv_cimag(*src0) * lv_cimag(*src0);
        if (abs_squared > max) {
            max = abs_squared;
            index = i;
        }
        ++src0;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_max_16u_u_avx2_variant_1(uint16_t* target,
                                                            const lv_32fc_t* src0,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_max_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 max_values = _mm256_setzero_ps();
    __m256i max_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_loadu_ps((float*)src0);
        __m256 in1 = _mm256_loadu_ps((float*)(src0 + 4));
        vector_32fc_index_max_variant1(
            in0, in1, &max_values, &max_indices, &current_indices, indices_increment);
        src0 += 8;
    }

    // determine maximum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float max_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t max_indices_buffer[8];
    _mm256_store_ps(max_values_buffer, max_values);
    _mm256_store_si256((__m256i*)max_indices_buffer, max_indices);

    float max = 0.f;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (max_values_buffer[i] > max) {
            max = max_values_buffer[i];
            index = max_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*src0) * lv_creal(*src0) + lv_cimag(*src0) * lv_cimag(*src0);
        if (abs_squared > max) {
            max = abs_squared;
            index = i;
        }
        ++src0;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <limits.h>

static inline void volk_32fc_index_max_16u_u_avx512f(uint16_t* target,
                                                     const lv_32fc_t* src0,
                                                     uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const lv_32fc_t* src0Ptr = src0;
    const uint32_t sixteenthPoints = num_points / 16;

    // Index ordering after shuffle: [0,1,8,9, 2,3,10,11, 4,5,12,13, 6,7,14,15]
    __m512 currentIndexes =
        _mm512_setr_ps(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    const __m512 indexIncrement = _mm512_set1_ps(16);

    __m512 maxValues = _mm512_setzero_ps();
    __m512 maxIndices = _mm512_setzero_ps();

    for (uint32_t number = 0; number < sixteenthPoints; number++) {
        // Load 16 complex values (32 floats)
        __m512 in0 = _mm512_loadu_ps((const float*)src0Ptr);
        __m512 in1 = _mm512_loadu_ps((const float*)(src0Ptr + 8));
        src0Ptr += 16;

        // Square all values
        in0 = _mm512_mul_ps(in0, in0);
        in1 = _mm512_mul_ps(in1, in1);

        // Add adjacent pairs (re² + im²) using within-lane shuffle
        // 0xB1 = _MM_SHUFFLE(2,3,0,1) swaps adjacent elements
        __m512 sw0 = _mm512_shuffle_ps(in0, in0, 0xB1);
        __m512 sw1 = _mm512_shuffle_ps(in1, in1, 0xB1);
        __m512 sum0 = _mm512_add_ps(in0, sw0);
        __m512 sum1 = _mm512_add_ps(in1, sw1);

        // Compact: pick elements 0,2 from sum0 and sum1 per 128-bit lane
        // 0x88 = _MM_SHUFFLE(2,0,2,0)
        __m512 mag_sq = _mm512_shuffle_ps(sum0, sum1, 0x88);

        // Compare and update maximums
        __mmask16 cmpMask = _mm512_cmp_ps_mask(mag_sq, maxValues, _CMP_GT_OS);
        maxIndices = _mm512_mask_blend_ps(cmpMask, maxIndices, currentIndexes);
        maxValues = _mm512_max_ps(mag_sq, maxValues);

        currentIndexes = _mm512_add_ps(currentIndexes, indexIncrement);
    }

    // Reduce 16 values to find maximum
    __VOLK_ATTR_ALIGNED(64) float maxValuesBuffer[16];
    __VOLK_ATTR_ALIGNED(64) float maxIndexesBuffer[16];
    _mm512_store_ps(maxValuesBuffer, maxValues);
    _mm512_store_ps(maxIndexesBuffer, maxIndices);

    float max = 0.0f;
    uint32_t index = 0;
    for (uint32_t i = 0; i < 16; i++) {
        if (maxValuesBuffer[i] > max) {
            max = maxValuesBuffer[i];
            index = (uint32_t)maxIndexesBuffer[i];
        } else if (maxValuesBuffer[i] == max) {
            if ((uint32_t)maxIndexesBuffer[i] < index)
                index = (uint32_t)maxIndexesBuffer[i];
        }
    }

    // Handle tail
    for (uint32_t number = sixteenthPoints * 16; number < num_points; number++) {
        const float re = lv_creal(*src0Ptr);
        const float im = lv_cimag(*src0Ptr);
        const float sq_dist = re * re + im * im;
        if (sq_dist > max) {
            max = sq_dist;
            index = number;
        }
        src0Ptr++;
    }
    *target = (uint16_t)index;
}

#endif /*LV_HAVE_AVX512F*/

#ifdef LV_HAVE_RVV
#include <float.h>
#include <riscv_vector.h>

static inline void
volk_32fc_index_max_16u_rvv(uint16_t* target, const lv_32fc_t* src0, uint32_t num_points)
{
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vuint16m2_t vmaxi = __riscv_vmv_v_x_u16m2(0, __riscv_vsetvlmax_e16m2());
    vuint16m2_t vidx = __riscv_vid_v_u16m2(__riscv_vsetvlmax_e16m2());
    size_t n = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
    for (size_t vl; n > 0; n -= vl, src0 += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)src0, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vr, vr, vl), vi, vi, vl);
        vbool8_t m = __riscv_vmflt(vmax, v, vl);
        vmax = __riscv_vfmax_tu(vmax, vmax, v, vl);
        vmaxi = __riscv_vmerge_tu(vmaxi, vmaxi, vidx, m, vl);
        vidx = __riscv_vadd(vidx, vl, __riscv_vsetvlmax_e16m4());
    }
    size_t vl = __riscv_vsetvlmax_e32m4();
    float max = __riscv_vfmv_f(__riscv_vfredmax(RISCV_SHRINK4(vfmax, f, 32, vmax),
                                                __riscv_vfmv_v_f_f32m1(0, 1),
                                                __riscv_vsetvlmax_e32m1()));
    // Find minimum index among lanes with max value
    // Note: mask type mismatch (vbool8_t vs vbool4_t) prevents using vmerge,
    // so we use scalar comparison
    __attribute__((aligned(32))) float values[128];
    __attribute__((aligned(32))) uint16_t indices[128];
    __riscv_vse32(values, vmax, vl);
    __riscv_vse16(indices, vmaxi, vl);
    uint16_t min_idx = UINT16_MAX;
    for (size_t i = 0; i < vl; i++) {
        if (values[i] == max && indices[i] < min_idx) {
            min_idx = indices[i];
        }
    }
    *target = min_idx;
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <float.h>
#include <riscv_vector.h>

static inline void volk_32fc_index_max_16u_rvvseg(uint16_t* target,
                                                  const lv_32fc_t* src0,
                                                  uint32_t num_points)
{
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vuint16m2_t vmaxi = __riscv_vmv_v_x_u16m2(0, __riscv_vsetvlmax_e16m2());
    vuint16m2_t vidx = __riscv_vid_v_u16m2(__riscv_vsetvlmax_e16m2());
    size_t n = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
    for (size_t vl; n > 0; n -= vl, src0 += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)src0, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0), vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vr, vr, vl), vi, vi, vl);
        vbool8_t m = __riscv_vmflt(vmax, v, vl);
        vmax = __riscv_vfmax_tu(vmax, vmax, v, vl);
        vmaxi = __riscv_vmerge_tu(vmaxi, vmaxi, vidx, m, vl);
        vidx = __riscv_vadd(vidx, vl, __riscv_vsetvlmax_e16m4());
    }
    size_t vl = __riscv_vsetvlmax_e32m4();
    float max = __riscv_vfmv_f(__riscv_vfredmax(RISCV_SHRINK4(vfmax, f, 32, vmax),
                                                __riscv_vfmv_v_f_f32m1(0, 1),
                                                __riscv_vsetvlmax_e32m1()));
    // Find minimum index among lanes with max value
    __attribute__((aligned(32))) float values[128];
    __attribute__((aligned(32))) uint16_t indices[128];
    __riscv_vse32(values, vmax, vl);
    __riscv_vse16(indices, vmaxi, vl);
    uint16_t min_idx = UINT16_MAX;
    for (size_t i = 0; i < vl; i++) {
        if (values[i] == max && indices[i] < min_idx) {
            min_idx = indices[i];
        }
    }
    *target = min_idx;
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_32fc_index_max_16u_u_H*/
