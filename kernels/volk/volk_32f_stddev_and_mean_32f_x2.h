/* -*- c++ -*- */
/*
 * Copyright 2012, 2014, 2019 Free Software Foundation, Inc.
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
 * void volk_32f_stddev_and_mean_32f_x2(float* stddev, float* mean, const float* inputBuffer, unsigned int num_points)
 * \endcode
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
 * Generate random numbers with c++11's normal distribution and estimate the mean and standard deviation
 * \code
 *   int N = 1000;
 *   unsigned int alignment = volk_get_alignment();
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

#include <volk/volk_common.h>
#include <inttypes.h>
#include <math.h>

// Youngs and Cramer's Algorithm for calculating std and mean
//   Using the methods discussed here:
//   https://doi.org/10.1145/3221269.3223036


static inline float
update_square_sum_1_val(const float SquareSum, const float Sum, const size_t len, const float val){
  // Updates a sum of squares calculated over len values with the value val
  float n = (float) len;
  return SquareSum + 1.f/( n * (n + 1.f) ) * ( n*val - Sum ) * ( n*val - Sum );
}

static inline float
add_square_sums(const float SquareSum0, const float Sum0, 
                const float SquareSum1, const float Sum1, const size_t len){
  // Add two sums of squares calculated over the same number of values, len
  float n = (float) len;
  return SquareSum0 + SquareSum1 + .5f / n * ( Sum0 - Sum1 )*( Sum0 - Sum1 );
}

static inline void
accrue_result( float* PartialSquareSums, float* PartialSums, 
               const size_t NumberOfPartitions, const size_t PartitionLen) {
  // Add all partial sums and square sums into the first element of the arays
  size_t accumulators = NumberOfPartitions;
  size_t stages = 0;
  size_t offset = 1;
  size_t partition_len = PartitionLen;

  while (accumulators >>= 1) { stages++; } // Integer log2
  accumulators = NumberOfPartitions;

  for (size_t s = 0; s < stages; s++ ) {
    accumulators /= 2;
    size_t idx = 0;
    for (size_t a = 0; a < accumulators; a++)  {
      PartialSquareSums[idx] = add_square_sums(PartialSquareSums[idx], PartialSums[idx], 
                               PartialSquareSums[idx + offset], PartialSums[idx + offset], 
                               partition_len);
      PartialSums[idx] += PartialSums[idx + offset];
      idx += 2*offset;
    }
    offset *= 2;
    partition_len *= 2;
  }
}

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_stddev_and_mean_32f_x2_generic(float* stddev, float* mean,
                                                      const float* inputBuffer,
                                                      unsigned int num_points)
{
  if (num_points == 0) { return; }

  const float* in_ptr = inputBuffer;

  float Sum = (*in_ptr++);
  float SquareSum = 0.f;
  uint32_t number = 1;

  for (; number < num_points; number++) {
    float val = (*in_ptr++);
    float n = (float) number;
    float np1 = n + 1.f;
    Sum += val;
    SquareSum += 1.f/( n * np1 ) * powf( np1 * val - Sum , 2);
  }
  *stddev = sqrtf( SquareSum / num_points );
  *mean   = Sum / num_points;
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_neon(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  if (num_points == 0) { return; }
  if (num_points < 8) {  
       volk_32f_stddev_and_mean_32f_x2_generic(stddev,  mean, inputBuffer, num_points);
       return;
  }

  const float* in_ptr = inputBuffer;

  __VOLK_ATTR_ALIGNED(16) float PSumsLoc[8] = {0.f};      // Store partial results from
  __VOLK_ATTR_ALIGNED(16) float PSquareSumLoc[8] = {0.f}; // accumulators


  const unsigned int eigth_points = num_points / 8;

  float32x4_t T0_acc, T1_acc;
  T0_acc = vld1q_f32((const float32_t*) in_ptr);
  in_ptr += 4;
  __VOLK_PREFETCH(in_ptr + 4);

  T1_acc = vld1q_f32((const float32_t*) in_ptr);
  in_ptr += 4;
  __VOLK_PREFETCH(in_ptr + 4);

  float32x4_t S0_acc = {0.f, 0.f, 0.f, 0.f};
  float32x4_t S1_acc = {0.f, 0.f, 0.f, 0.f};

  float32x4_t v0_reg, v1_reg;
  float32x4_t x0_reg, x1_reg;
  float32x4_t f_reg;

  for(size_t number = 1; number < eigth_points; number++) {
    v0_reg = vld1q_f32( in_ptr );
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    v1_reg = vld1q_f32( in_ptr );
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = vdupq_n_f32(  1.f/( n*np1 ) );

    T0_acc = vaddq_f32(T0_acc, v0_reg);
    x0_reg = vdupq_n_f32(np1);
    x0_reg = vmulq_f32(x0_reg, v0_reg);
    x0_reg = vsubq_f32(x0_reg, T0_acc);
    x0_reg = vmulq_f32(x0_reg, x0_reg);
    S0_acc = vfmaq_f32(S0_acc, x0_reg, f_reg);

    T1_acc = vaddq_f32(T1_acc, v1_reg);
    x1_reg = vdupq_n_f32(np1);
    x1_reg = vmulq_f32(x1_reg, v1_reg);
    x1_reg = vsubq_f32(x1_reg, T1_acc);
    x1_reg = vmulq_f32(x1_reg, x1_reg);
    S1_acc = vfmaq_f32(S1_acc, x1_reg, f_reg);
  }

  vst1q_f32(&PSumsLoc[0], T0_acc);
  vst1q_f32(&PSumsLoc[4], T1_acc);

  vst1q_f32(&PSquareSumLoc[0], S0_acc);
  vst1q_f32(&PSquareSumLoc[4], S1_acc);

  accrue_result( PSquareSumLoc, PSumsLoc, 8, eigth_points);

  size_t points_done = eigth_points*8;

  for (; points_done < num_points; points_done++) {
    float val = (*in_ptr);
    PSquareSumLoc[0] = update_square_sum_1_val(PSquareSumLoc[0], PSumsLoc[0], points_done, val);
    PSumsLoc[0]     += val;
    in_ptr++;
  }

  *stddev = sqrtf( PSquareSumLoc[0] / num_points );
  *mean   = PSumsLoc[0] / num_points;
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_sse(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  if (num_points == 0) { return; }
  if (num_points < 8) {  
       volk_32f_stddev_and_mean_32f_x2_generic(stddev,  mean, inputBuffer, num_points);
       return;
  }

  const float* in_ptr = inputBuffer;

  __VOLK_ATTR_ALIGNED(16) float T[8] = {0.f};
  __VOLK_ATTR_ALIGNED(16) float S[8] = {0.f};


  const unsigned int eigth_points = num_points / 8;

  __m128 T0_acc = _mm_loadu_ps(in_ptr);
  in_ptr += 4;
  __m128 T1_acc = _mm_loadu_ps(in_ptr);
  in_ptr += 4;
  __m128 S0_acc = _mm_setzero_ps();
  __m128 S1_acc = _mm_setzero_ps();
  __m128 v0_reg, v1_reg;
  __m128 x0_reg, x1_reg;
  __m128 f_reg;


  for(size_t number = 1; number < eigth_points; number++) {
    v0_reg = _mm_loadu_ps(in_ptr);
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    v1_reg = _mm_loadu_ps(in_ptr);
    in_ptr += 4;    
    __VOLK_PREFETCH(in_ptr + 4);

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T0_acc = _mm_add_ps(T0_acc, v0_reg);

    x0_reg = _mm_set_ps1(np1);
    x0_reg = _mm_mul_ps(x0_reg, v0_reg);
    x0_reg = _mm_sub_ps(x0_reg, T0_acc);
    x0_reg = _mm_mul_ps(x0_reg, x0_reg);
    x0_reg = _mm_mul_ps(x0_reg, f_reg);
    S0_acc = _mm_add_ps(S0_acc, x0_reg);

    T1_acc = _mm_add_ps(T1_acc, v1_reg);

    x1_reg = _mm_set_ps1(np1);
    x1_reg = _mm_mul_ps(x1_reg, v1_reg);
    x1_reg = _mm_sub_ps(x1_reg, T1_acc);
    x1_reg = _mm_mul_ps(x1_reg, x1_reg);
    x1_reg = _mm_mul_ps(x1_reg, f_reg);
    S1_acc = _mm_add_ps(S1_acc, x1_reg);
  }

  _mm_store_ps(&T[0], T0_acc);
  _mm_store_ps(&T[4], T1_acc);
  _mm_store_ps(&S[0], S0_acc);
  _mm_store_ps(&S[4], S1_acc);

  accrue_result( S, T, 8, eigth_points);

  size_t points_done = eigth_points*8;

  for (; points_done < num_points; points_done++) {
    float val = (*in_ptr);
    S[0] = update_square_sum_1_val(S[0], T[0], points_done, val);
    T[0] += val;
    in_ptr++;
  }

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_avx(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  if (num_points == 0) { return; }
  if (num_points < 16) {  
       volk_32f_stddev_and_mean_32f_x2_generic(stddev,  mean, inputBuffer, num_points);
       return;
  }

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(32) float T[16] = {0.f};
  __VOLK_ATTR_ALIGNED(32) float S[16] = {0.f};
  
  const unsigned int sixteenth_points = num_points / 16;

  __m256 T0_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;
  __m256 T1_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;

  __m256 S0_acc = _mm256_setzero_ps();
  __m256 S1_acc = _mm256_setzero_ps();
  __m256 v0_reg, v1_reg;
  __m256 x0_reg, x1_reg;
  __m256 f_reg;

  for(;number < sixteenth_points; number++) {
    v0_reg = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;
    __VOLK_PREFETCH(in_ptr + 8);

    v1_reg = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;
    __VOLK_PREFETCH(in_ptr + 8);


    float n   = (float) number;
    float np1 = n + 1.f;

    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    x0_reg = _mm256_set1_ps(np1);
    
    T0_acc = _mm256_add_ps(T0_acc, v0_reg);
    
    x0_reg = _mm256_mul_ps(x0_reg, v0_reg);
    x0_reg = _mm256_sub_ps(x0_reg, T0_acc);
    x0_reg = _mm256_mul_ps(x0_reg, x0_reg);
    x0_reg = _mm256_mul_ps(x0_reg, f_reg);
    S0_acc = _mm256_add_ps(S0_acc, x0_reg);

    T1_acc = _mm256_add_ps(T1_acc, v1_reg);

    x1_reg = _mm256_set1_ps(np1);
    x1_reg = _mm256_mul_ps(x1_reg, v1_reg);
    x1_reg = _mm256_sub_ps(x1_reg, T1_acc);
    x1_reg = _mm256_mul_ps(x1_reg, x1_reg);
    x1_reg = _mm256_mul_ps(x1_reg, f_reg);
    S1_acc = _mm256_add_ps(S1_acc, x1_reg);
  }  

  _mm256_store_ps(&T[0], T0_acc);
  _mm256_store_ps(&T[8], T1_acc);
  _mm256_store_ps(&S[0], S0_acc);  
  _mm256_store_ps(&S[8], S1_acc);  

  accrue_result(S, T, 16, sixteenth_points);

  number = sixteenth_points*16;

  for (; number < num_points; number++) {
    S[0] = update_square_sum_1_val(S[0], T[0], number, *in_ptr);
    T[0] += (*in_ptr++);
  }

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_sse(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  if (num_points == 0) { return; }
  if (num_points < 8) {  
       volk_32f_stddev_and_mean_32f_x2_generic(stddev,  mean, inputBuffer, num_points);
       return;
  }

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(16) float T[8] = {0.f};
  __VOLK_ATTR_ALIGNED(16) float S[8] = {0.f};


  const unsigned int eigth_points = num_points / 8;

  __m128 T0_acc = _mm_load_ps(in_ptr);
  in_ptr += 4;
  __m128 T1_acc = _mm_load_ps(in_ptr);
  in_ptr += 4;
  __m128 S0_acc = _mm_setzero_ps();
  __m128 S1_acc = _mm_setzero_ps();
  __m128 v0_reg, v1_reg;
  __m128 x0_reg, x1_reg;
  __m128 f_reg;


  for(;number < eigth_points; number++) {
    v0_reg = _mm_load_ps(in_ptr);
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    v1_reg = _mm_load_ps(in_ptr);
    in_ptr += 4;    
    __VOLK_PREFETCH(in_ptr + 4);

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T0_acc = _mm_add_ps(T0_acc, v0_reg);

    x0_reg = _mm_set_ps1(np1);
    x0_reg = _mm_mul_ps(x0_reg, v0_reg);
    x0_reg = _mm_sub_ps(x0_reg, T0_acc);
    x0_reg = _mm_mul_ps(x0_reg, x0_reg);
    x0_reg = _mm_mul_ps(x0_reg, f_reg);
    S0_acc = _mm_add_ps(S0_acc, x0_reg);

    T1_acc = _mm_add_ps(T1_acc, v1_reg);

    x1_reg = _mm_set_ps1(np1);
    x1_reg = _mm_mul_ps(x1_reg, v1_reg);
    x1_reg = _mm_sub_ps(x1_reg, T1_acc);
    x1_reg = _mm_mul_ps(x1_reg, x1_reg);
    x1_reg = _mm_mul_ps(x1_reg, f_reg);
    S1_acc = _mm_add_ps(S1_acc, x1_reg);
  }

  _mm_store_ps(&T[0], T0_acc);
  _mm_store_ps(&T[4], T1_acc);
  _mm_store_ps(&S[0], S0_acc);
  _mm_store_ps(&S[4], S1_acc);

  accrue_result( S, T, 8, eigth_points);

  number = eigth_points*8;

  for (; number < num_points; number++) {
    S[0] = update_square_sum_1_val(S[0], T[0], number, *in_ptr);
    T[0] += (*in_ptr++);
  }

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_avx(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  if (num_points == 0) { return; }

  const float* in_ptr = inputBuffer;
  if (num_points < 16) {  
       volk_32f_stddev_and_mean_32f_x2_generic(stddev,  mean, inputBuffer, num_points);
       return;
  }

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(32) float T[16] = {0.f};
  __VOLK_ATTR_ALIGNED(32) float S[16] = {0.f};


  const unsigned int sixteenth_points = num_points / 16;

  __m256 T0_acc = _mm256_load_ps(in_ptr);
  in_ptr += 8;
  __m256 T1_acc = _mm256_load_ps(in_ptr);
  in_ptr += 8;

  __m256 S0_acc = _mm256_setzero_ps();
  __m256 S1_acc = _mm256_setzero_ps();
  __m256 v0_reg, v1_reg;
  __m256 x0_reg, x1_reg;
  __m256 f_reg;

  for(;number < sixteenth_points; number++) {
    v0_reg = _mm256_load_ps(in_ptr);
    in_ptr += 8;
    __VOLK_PREFETCH(in_ptr + 8);

    v1_reg = _mm256_load_ps(in_ptr);
    in_ptr += 8;
    __VOLK_PREFETCH(in_ptr + 8);

    float n   = (float) number;
    float np1 = n + 1.f;

    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    x0_reg = _mm256_set1_ps(np1);
    
    T0_acc = _mm256_add_ps(T0_acc, v0_reg);
    
    x0_reg = _mm256_mul_ps(x0_reg, v0_reg);
    x0_reg = _mm256_sub_ps(x0_reg, T0_acc);
    x0_reg = _mm256_mul_ps(x0_reg, x0_reg);
    x0_reg = _mm256_mul_ps(x0_reg, f_reg);
    S0_acc = _mm256_add_ps(S0_acc, x0_reg);

    T1_acc = _mm256_add_ps(T1_acc, v1_reg);

    x1_reg = _mm256_set1_ps(np1);
    x1_reg = _mm256_mul_ps(x1_reg, v1_reg);
    x1_reg = _mm256_sub_ps(x1_reg, T1_acc);
    x1_reg = _mm256_mul_ps(x1_reg, x1_reg);
    x1_reg = _mm256_mul_ps(x1_reg, f_reg);
    S1_acc = _mm256_add_ps(S1_acc, x1_reg);
  }  

  _mm256_store_ps(&T[0], T0_acc);
  _mm256_store_ps(&T[8], T1_acc);
  _mm256_store_ps(&S[0], S0_acc);  
  _mm256_store_ps(&S[8], S1_acc);  

  accrue_result(S, T, 16, sixteenth_points);

  number = sixteenth_points*16;

  for (; number < num_points; number++) {
    S[0] = update_square_sum_1_val(S[0], T[0], number, *in_ptr);
    T[0] += (*in_ptr++);
  }

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_AVX */


#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */