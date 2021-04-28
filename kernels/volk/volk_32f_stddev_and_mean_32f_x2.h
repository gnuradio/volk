/* -*- c++ -*- */
/*
 * Copyright 2012, 2014, 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/*!
 * \page volk_32f_stddev_and_mean_32f_x2
 *
 * \b Overview
 *
 * Computes the standard deviation and mean of the input buffer by means of
 * Youngs and Cramer's Algorithm
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_stddev_and_mean_32f_x2(float* stddev, float* mean, const float*
 * inputBuffer, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputBuffer: The buffer of points.
 * \li num_points The number of values in input buffer.
 *
 * \b Outputs
 * \li stddev: The calculated standard deviation.
 * \li mean: The mean of the input buffer.
 *
 * \b Example
 * Generate random numbers with c++11's normal distribution and estimate the mean and
 * standard deviation \code int N = 1000; unsigned int alignment = volk_get_alignment();
 *   float* rand_numbers = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* mean = (float*)volk_malloc(sizeof(float), alignment);
 *   float* stddev = (float*)volk_malloc(sizeof(float), alignment);
 *
 *   // Use a normal generator with 0 mean, stddev 1
 *   std::default_random_engine generator;
 *   std::normal_distribution<float> distribution(0,1000);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       rand_numbers[ii] =  distribution(generator);
 *   }
 *
 *   volk_32f_stddev_and_mean_32f_x2(stddev, mean, rand_numbers, N);
 *
 *   printf("std. dev. = %f\n", *stddev);
 *   printf("mean = %f\n", *mean);
 *
 *   volk_free(rand_numbers);
 *   volk_free(mean);
 *   volk_free(stddev);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H
#define INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H

#include <inttypes.h>
#include <math.h>
#include <volk/volk_common.h>

// Youngs and Cramer's Algorithm for calculating std and mean
//   Using the methods discussed here:
//   https://doi.org/10.1145/3221269.3223036
#ifdef LV_HAVE_GENERIC

static inline void volk_32f_stddev_and_mean_32f_x2_generic(float* stddev,
                                                           float* mean,
                                                           const float* inputBuffer,
                                                           unsigned int num_points)
{
    const float* in_ptr = inputBuffer;
    if (num_points == 0) {
        return;
    } else if (num_points == 1) {
        *stddev = 0.f;
        *mean = (*in_ptr);
        return;
    }

    float Sum[2];
    float SquareSum[2] = { 0.f, 0.f };
    Sum[0] = (*in_ptr++);
    Sum[1] = (*in_ptr++);

    uint32_t half_points = num_points / 2;

    for (uint32_t number = 1; number < half_points; number++) {
        float Val0 = (*in_ptr++);
        float Val1 = (*in_ptr++);
        float n = (float)number;
        float n_plus_one = n + 1.f;
        float r = 1.f / (n * n_plus_one);

        Sum[0] += Val0;
        Sum[1] += Val1;

        SquareSum[0] += r * powf(n_plus_one * Val0 - Sum[0], 2);
        SquareSum[1] += r * powf(n_plus_one * Val1 - Sum[1], 2);
    }

    SquareSum[0] += SquareSum[1] + .5f / half_points * pow(Sum[0] - Sum[1], 2);
    Sum[0] += Sum[1];

    uint32_t points_done = half_points * 2;

    for (; points_done < num_points; points_done++) {
        float Val = (*in_ptr++);
        float n = (float)points_done;
        float n_plus_one = n + 1.f;
        float r = 1.f / (n * n_plus_one);
        Sum[0] += Val;
        SquareSum[0] += r * powf(n_plus_one * Val - Sum[0], 2);
    }
    *stddev = sqrtf(SquareSum[0] / num_points);
    *mean = Sum[0] / num_points;
}
#endif /* LV_HAVE_GENERIC */

static inline float update_square_sum_1_val(const float SquareSum,
                                            const float Sum,
                                            const uint32_t len,
                                            const float val)
{
    // Updates a sum of squares calculated over len values with the value val
    float n = (float)len;
    float n_plus_one = n + 1.f;
    return SquareSum +
           1.f / (n * n_plus_one) * (n_plus_one * val - Sum) * (n_plus_one * val - Sum);
}

static inline float add_square_sums(const float SquareSum0,
                                    const float Sum0,
                                    const float SquareSum1,
                                    const float Sum1,
                                    const uint32_t len)
{
    // Add two sums of squares calculated over the same number of values, len
    float n = (float)len;
    return SquareSum0 + SquareSum1 + .5f / n * (Sum0 - Sum1) * (Sum0 - Sum1);
}

static inline void accrue_result(float* PartialSquareSums,
                                 float* PartialSums,
                                 const uint32_t NumberOfPartitions,
                                 const uint32_t PartitionLen)
{
    // Add all partial sums and square sums into the first element of the arrays
    uint32_t accumulators = NumberOfPartitions;
    uint32_t stages = 0;
    uint32_t offset = 1;
    uint32_t partition_len = PartitionLen;

    while (accumulators >>= 1) {
        stages++;
    } // Integer log2
    accumulators = NumberOfPartitions;

    for (uint32_t s = 0; s < stages; s++) {
        accumulators /= 2;
        uint32_t idx = 0;
        for (uint32_t a = 0; a < accumulators; a++) {
            PartialSquareSums[idx] = add_square_sums(PartialSquareSums[idx],
                                                     PartialSums[idx],
                                                     PartialSquareSums[idx + offset],
                                                     PartialSums[idx + offset],
                                                     partition_len);
            PartialSums[idx] += PartialSums[idx + offset];
            idx += 2 * offset;
        }
        offset *= 2;
        partition_len *= 2;
    }
}

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_32f_stddev_and_mean_32f_x2_neon(float* stddev,
                                                        float* mean,
                                                        const float* inputBuffer,
                                                        unsigned int num_points)
{
    if (num_points < 8) {
        volk_32f_stddev_and_mean_32f_x2_generic(stddev, mean, inputBuffer, num_points);
        return;
    }

    const float* in_ptr = inputBuffer;

    __VOLK_ATTR_ALIGNED(16) float SumLocal[8] = { 0.f };
    __VOLK_ATTR_ALIGNED(16) float SquareSumLocal[8] = { 0.f };

    const uint32_t eigth_points = num_points / 8;

    float32x4_t Sum0, Sum1;

    Sum0 = vld1q_f32((const float32_t*)in_ptr);
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    Sum1 = vld1q_f32((const float32_t*)in_ptr);
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    float32x4_t SquareSum0 = { 0.f };
    float32x4_t SquareSum1 = { 0.f };

    float32x4_t Values0, Values1;
    float32x4_t Aux0, Aux1;
    float32x4_t Reciprocal;

    for (uint32_t number = 1; number < eigth_points; number++) {
        Values0 = vld1q_f32(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        Values1 = vld1q_f32(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        float n = (float)number;
        float n_plus_one = n + 1.f;
        Reciprocal = vdupq_n_f32(1.f / (n * n_plus_one));

        Sum0 = vaddq_f32(Sum0, Values0);
        Aux0 = vdupq_n_f32(n_plus_one);
        SquareSum0 =
            _neon_accumulate_square_sum_f32(SquareSum0, Sum0, Values0, Reciprocal, Aux0);

        Sum1 = vaddq_f32(Sum1, Values1);
        Aux1 = vdupq_n_f32(n_plus_one);
        SquareSum1 =
            _neon_accumulate_square_sum_f32(SquareSum1, Sum1, Values1, Reciprocal, Aux1);
    }

    vst1q_f32(&SumLocal[0], Sum0);
    vst1q_f32(&SumLocal[4], Sum1);
    vst1q_f32(&SquareSumLocal[0], SquareSum0);
    vst1q_f32(&SquareSumLocal[4], SquareSum1);

    accrue_result(SquareSumLocal, SumLocal, 8, eigth_points);

    uint32_t points_done = eigth_points * 8;

    for (; points_done < num_points; points_done++) {
        float val = (*in_ptr++);
        SumLocal[0] += val;
        SquareSumLocal[0] =
            update_square_sum_1_val(SquareSumLocal[0], SumLocal[0], points_done, val);
    }

    *stddev = sqrtf(SquareSumLocal[0] / num_points);
    *mean = SumLocal[0] / num_points;
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32f_stddev_and_mean_32f_x2_u_sse(float* stddev,
                                                         float* mean,
                                                         const float* inputBuffer,
                                                         unsigned int num_points)
{
    if (num_points < 8) {
        volk_32f_stddev_and_mean_32f_x2_generic(stddev, mean, inputBuffer, num_points);
        return;
    }

    const float* in_ptr = inputBuffer;

    __VOLK_ATTR_ALIGNED(16) float SumLocal[8] = { 0.f };
    __VOLK_ATTR_ALIGNED(16) float SquareSumLocal[8] = { 0.f };


    const uint32_t eigth_points = num_points / 8;

    __m128 Sum0 = _mm_loadu_ps(in_ptr);
    in_ptr += 4;
    __m128 Sum1 = _mm_loadu_ps(in_ptr);
    in_ptr += 4;
    __m128 SquareSum0 = _mm_setzero_ps();
    __m128 SquareSum1 = _mm_setzero_ps();
    __m128 Values0, Values1;
    __m128 Aux0, Aux1;
    __m128 Reciprocal;

    for (uint32_t number = 1; number < eigth_points; number++) {
        Values0 = _mm_loadu_ps(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        Values1 = _mm_loadu_ps(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        float n = (float)number;
        float n_plus_one = n + 1.f;
        Reciprocal = _mm_set_ps1(1.f / (n * n_plus_one));

        Sum0 = _mm_add_ps(Sum0, Values0);
        Aux0 = _mm_set_ps1(n_plus_one);
        SquareSum0 =
            _mm_accumulate_square_sum_ps(SquareSum0, Sum0, Values0, Reciprocal, Aux0);

        Sum1 = _mm_add_ps(Sum1, Values1);
        Aux1 = _mm_set_ps1(n_plus_one);
        SquareSum1 =
            _mm_accumulate_square_sum_ps(SquareSum1, Sum1, Values1, Reciprocal, Aux1);
    }

    _mm_store_ps(&SumLocal[0], Sum0);
    _mm_store_ps(&SumLocal[4], Sum1);
    _mm_store_ps(&SquareSumLocal[0], SquareSum0);
    _mm_store_ps(&SquareSumLocal[4], SquareSum1);

    accrue_result(SquareSumLocal, SumLocal, 8, eigth_points);

    uint32_t points_done = eigth_points * 8;

    for (; points_done < num_points; points_done++) {
        float val = (*in_ptr++);
        SumLocal[0] += val;
        SquareSumLocal[0] =
            update_square_sum_1_val(SquareSumLocal[0], SumLocal[0], points_done, val);
    }

    *stddev = sqrtf(SquareSumLocal[0] / num_points);
    *mean = SumLocal[0] / num_points;
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32f_stddev_and_mean_32f_x2_u_avx(float* stddev,
                                                         float* mean,
                                                         const float* inputBuffer,
                                                         unsigned int num_points)
{
    if (num_points < 16) {
        volk_32f_stddev_and_mean_32f_x2_generic(stddev, mean, inputBuffer, num_points);
        return;
    }

    const float* in_ptr = inputBuffer;

    __VOLK_ATTR_ALIGNED(32) float SumLocal[16] = { 0.f };
    __VOLK_ATTR_ALIGNED(32) float SquareSumLocal[16] = { 0.f };

    const unsigned int sixteenth_points = num_points / 16;

    __m256 Sum0 = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;
    __m256 Sum1 = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;

    __m256 SquareSum0 = _mm256_setzero_ps();
    __m256 SquareSum1 = _mm256_setzero_ps();
    __m256 Values0, Values1;
    __m256 Aux0, Aux1;
    __m256 Reciprocal;

    for (uint32_t number = 1; number < sixteenth_points; number++) {
        Values0 = _mm256_loadu_ps(in_ptr);
        in_ptr += 8;
        __VOLK_PREFETCH(in_ptr + 8);

        Values1 = _mm256_loadu_ps(in_ptr);
        in_ptr += 8;
        __VOLK_PREFETCH(in_ptr + 8);

        float n = (float)number;
        float n_plus_one = n + 1.f;

        Reciprocal = _mm256_set1_ps(1.f / (n * n_plus_one));

        Sum0 = _mm256_add_ps(Sum0, Values0);
        Aux0 = _mm256_set1_ps(n_plus_one);
        SquareSum0 =
            _mm256_accumulate_square_sum_ps(SquareSum0, Sum0, Values0, Reciprocal, Aux0);

        Sum1 = _mm256_add_ps(Sum1, Values1);
        Aux1 = _mm256_set1_ps(n_plus_one);
        SquareSum1 =
            _mm256_accumulate_square_sum_ps(SquareSum1, Sum1, Values1, Reciprocal, Aux1);
    }

    _mm256_store_ps(&SumLocal[0], Sum0);
    _mm256_store_ps(&SumLocal[8], Sum1);
    _mm256_store_ps(&SquareSumLocal[0], SquareSum0);
    _mm256_store_ps(&SquareSumLocal[8], SquareSum1);

    accrue_result(SquareSumLocal, SumLocal, 16, sixteenth_points);

    uint32_t points_done = sixteenth_points * 16;

    for (; points_done < num_points; points_done++) {
        float val = (*in_ptr++);
        SumLocal[0] += val;
        SquareSumLocal[0] =
            update_square_sum_1_val(SquareSumLocal[0], SumLocal[0], points_done, val);
    }

    *stddev = sqrtf(SquareSumLocal[0] / num_points);
    *mean = SumLocal[0] / num_points;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_stddev_and_mean_32f_x2_a_sse(float* stddev,
                                                         float* mean,
                                                         const float* inputBuffer,
                                                         unsigned int num_points)
{
    if (num_points < 8) {
        volk_32f_stddev_and_mean_32f_x2_generic(stddev, mean, inputBuffer, num_points);
        return;
    }

    const float* in_ptr = inputBuffer;

    __VOLK_ATTR_ALIGNED(16) float SumLocal[8] = { 0.f };
    __VOLK_ATTR_ALIGNED(16) float SquareSumLocal[8] = { 0.f };


    const uint32_t eigth_points = num_points / 8;

    __m128 Sum0 = _mm_load_ps(in_ptr);
    in_ptr += 4;
    __m128 Sum1 = _mm_load_ps(in_ptr);
    in_ptr += 4;
    __m128 SquareSum0 = _mm_setzero_ps();
    __m128 SquareSum1 = _mm_setzero_ps();
    __m128 Values0, Values1;
    __m128 Aux0, Aux1;
    __m128 Reciprocal;

    for (uint32_t number = 1; number < eigth_points; number++) {
        Values0 = _mm_loadu_ps(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        Values1 = _mm_loadu_ps(in_ptr);
        in_ptr += 4;
        __VOLK_PREFETCH(in_ptr + 4);

        float n = (float)number;
        float n_plus_one = n + 1.f;
        Reciprocal = _mm_set_ps1(1.f / (n * n_plus_one));

        Sum0 = _mm_add_ps(Sum0, Values0);
        Aux0 = _mm_set_ps1(n_plus_one);
        SquareSum0 =
            _mm_accumulate_square_sum_ps(SquareSum0, Sum0, Values0, Reciprocal, Aux0);

        Sum1 = _mm_add_ps(Sum1, Values1);
        Aux1 = _mm_set_ps1(n_plus_one);
        SquareSum1 =
            _mm_accumulate_square_sum_ps(SquareSum1, Sum1, Values1, Reciprocal, Aux1);
    }

    _mm_store_ps(&SumLocal[0], Sum0);
    _mm_store_ps(&SumLocal[4], Sum1);
    _mm_store_ps(&SquareSumLocal[0], SquareSum0);
    _mm_store_ps(&SquareSumLocal[4], SquareSum1);

    accrue_result(SquareSumLocal, SumLocal, 8, eigth_points);

    uint32_t points_done = eigth_points * 8;

    for (; points_done < num_points; points_done++) {
        float val = (*in_ptr++);
        SumLocal[0] += val;
        SquareSumLocal[0] =
            update_square_sum_1_val(SquareSumLocal[0], SumLocal[0], points_done, val);
    }

    *stddev = sqrtf(SquareSumLocal[0] / num_points);
    *mean = SumLocal[0] / num_points;
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_stddev_and_mean_32f_x2_a_avx(float* stddev,
                                                         float* mean,
                                                         const float* inputBuffer,
                                                         unsigned int num_points)
{
    if (num_points < 16) {
        volk_32f_stddev_and_mean_32f_x2_generic(stddev, mean, inputBuffer, num_points);
        return;
    }

    const float* in_ptr = inputBuffer;

    __VOLK_ATTR_ALIGNED(32) float SumLocal[16] = { 0.f };
    __VOLK_ATTR_ALIGNED(32) float SquareSumLocal[16] = { 0.f };

    const unsigned int sixteenth_points = num_points / 16;

    __m256 Sum0 = _mm256_load_ps(in_ptr);
    in_ptr += 8;
    __m256 Sum1 = _mm256_load_ps(in_ptr);
    in_ptr += 8;

    __m256 SquareSum0 = _mm256_setzero_ps();
    __m256 SquareSum1 = _mm256_setzero_ps();
    __m256 Values0, Values1;
    __m256 Aux0, Aux1;
    __m256 Reciprocal;

    for (uint32_t number = 1; number < sixteenth_points; number++) {
        Values0 = _mm256_loadu_ps(in_ptr);
        in_ptr += 8;
        __VOLK_PREFETCH(in_ptr + 8);

        Values1 = _mm256_loadu_ps(in_ptr);
        in_ptr += 8;
        __VOLK_PREFETCH(in_ptr + 8);

        float n = (float)number;
        float n_plus_one = n + 1.f;

        Reciprocal = _mm256_set1_ps(1.f / (n * n_plus_one));

        Sum0 = _mm256_add_ps(Sum0, Values0);
        Aux0 = _mm256_set1_ps(n_plus_one);
        SquareSum0 =
            _mm256_accumulate_square_sum_ps(SquareSum0, Sum0, Values0, Reciprocal, Aux0);

        Sum1 = _mm256_add_ps(Sum1, Values1);
        Aux1 = _mm256_set1_ps(n_plus_one);
        SquareSum1 =
            _mm256_accumulate_square_sum_ps(SquareSum1, Sum1, Values1, Reciprocal, Aux1);
    }

    _mm256_store_ps(&SumLocal[0], Sum0);
    _mm256_store_ps(&SumLocal[8], Sum1);
    _mm256_store_ps(&SquareSumLocal[0], SquareSum0);
    _mm256_store_ps(&SquareSumLocal[8], SquareSum1);

    accrue_result(SquareSumLocal, SumLocal, 16, sixteenth_points);

    uint32_t points_done = sixteenth_points * 16;

    for (; points_done < num_points; points_done++) {
        float val = (*in_ptr++);
        SumLocal[0] += val;
        SquareSumLocal[0] =
            update_square_sum_1_val(SquareSumLocal[0], SumLocal[0], points_done, val);
    }

    *stddev = sqrtf(SquareSumLocal[0] / num_points);
    *mean = SumLocal[0] / num_points;
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */