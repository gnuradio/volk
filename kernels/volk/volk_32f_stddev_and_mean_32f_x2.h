/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
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
 * Computes the standard deviation and mean of the input buffer.
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
 *   std::normal_distribution<float> distribution(0,1);
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


static inline void
update_square_sum_1_val(float* S, const float* T, const uint32_t* N, const float* val){
  float n = (float) (*N);
  (*S) += 1.f/( n*(n + 1.f) ) * ( n*(*val) - (*T) ) * ( n*(*val) - (*T) );
}

static inline void
square_add(float* S,  const float* T0, const float* S1, const float* T1, const uint32_t* N){
  float n = (float) (*N);
  (*S) += (*S1);
  (*S) += .5f/n*( (*T0) - (*T1) )*( (*T0) - (*T1) );
}

static inline void
accrue_result( float* S, float* T, const uint32_t N_accumulators, const uint32_t N_partition) {
  uint32_t accumulators = N_accumulators;
  uint32_t stages = 0;
  uint32_t m = 1;
  uint32_t partition_size = N_partition;

  while (accumulators >>= 1) { stages++; }
  accumulators = N_accumulators;

  for (uint32_t s = 0; s < stages; s++ ) {
    accumulators /= 2;
    uint32_t idx = 0;
    for (uint32_t a = 0; a < accumulators; a++)  {
      square_add( &S[idx] , &T[idx] , &S[idx+m], &T[idx+m], &partition_size);
      T[idx] += T[idx+m];
      idx += 2*m;
    }
    m *= 2;
    partition_size *= 2;
  }
}

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_stddev_and_mean_32f_x2_generic(float* stddev, float* mean,
                                                      const float* inputBuffer,
                                                      unsigned int num_points)
{
  // Youngs and Cramer's Algorithm for calculating std and mean
  //   T is the running sum of values
  //   S is the running square sum of values
  if (num_points == 0) { return; }

  const float* in_ptr = inputBuffer;

  float T = (*in_ptr++);
  float S = 0.f;
  uint32_t number = 1;

  for (; number < num_points; number++) {
    float v = (*in_ptr++);
    float n = (float) number;
    float np1 = n + 1.f;
    T += v;
    S += 1.f/( n*np1 )*powf( np1*v - T , 2);
  }

  *stddev = sqrtf( S / num_points );
  *mean   = T / num_points;
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

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(16) float T[8] = {0.f};
  __VOLK_ATTR_ALIGNED(16) float S[8] = {0.f};

  if (num_points < 8) {   
    T[0] = (*in_ptr++);
    goto FINALIZE; }

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

  for(;number < eigth_points; number++) {
    v0_reg = vld1q_f32( in_ptr );
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    v1_reg = vld1q_f32( in_ptr );
    in_ptr += 4;
    __VOLK_PREFETCH(in_ptr + 4);

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = vdupq_n_f32(  1.f/( n*np1 ) );

    T0_acc = vaddq_f32(T0_acc, v0_reg); // T = T + v  |
    x0_reg = vdupq_n_f32(np1);          // x = n + 1  | n+1
    x0_reg = vmulq_f32(x0_reg, v0_reg); // x = x * v  | (n+1)*v
    x0_reg = vsubq_f32(x0_reg, T0_acc); // x = x - T  | (n+1)*v - T
    x0_reg = vmulq_f32(x0_reg, x0_reg); // x = x * x  | ( (n+1)*v - T )**2
    S0_acc = vfmaq_f32(S0_acc, x0_reg, f_reg); // S = S + inv(n*(n+1)*x

    T1_acc = vaddq_f32(T1_acc, v1_reg);
    x1_reg = vdupq_n_f32(np1);
    x1_reg = vmulq_f32(x1_reg, v1_reg);
    x1_reg = vsubq_f32(x1_reg, T1_acc);
    x1_reg = vmulq_f32(x1_reg, x1_reg);
    S1_acc = vfmaq_f32(S1_acc, x1_reg, f_reg);
  }

  vst1q_f32(&T[0], T0_acc);
  vst1q_f32(&T[4], T1_acc);

  vst1q_f32(&S[0], S0_acc);
  vst1q_f32(&S[4], S1_acc);

  accrue_result( S, T, 8, eigth_points);

  number = eigth_points*8;

  FINALIZE:
  for (; number < num_points; number++) {
    update_square_sum_1_val(&S[0], &T[0], &number, in_ptr);
    T[0] += (*in_ptr++);
  }

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
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

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(16) float T[8] = {0.f};
  __VOLK_ATTR_ALIGNED(16) float S[8] = {0.f};

  if (num_points < 8) {   
    T[0] = (*in_ptr++);
    goto FINALIZE; }

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


  for(;number < eigth_points; number++) {
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

  number = eigth_points*8;

  FINALIZE:
  for (; number < num_points; number++) {
    update_square_sum_1_val(&S[0], &T[0], &number, in_ptr);
    T[0] += (*in_ptr++);
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

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(32) float T[16] = {0.f};
  __VOLK_ATTR_ALIGNED(32) float S[16] = {0.f};
  
  if (num_points < 16) {   
    T[0] = (*in_ptr++);
    goto FINALIZE; }

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

  FINALIZE:
  for (; number < num_points; number++) {
    update_square_sum_1_val(&S[0], &T[0], &number, in_ptr);
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

  const float* in_ptr = inputBuffer;

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(16) float T[8] = {0.f};
  __VOLK_ATTR_ALIGNED(16) float S[8] = {0.f};

  if (num_points < 8) {   
    T[0] = (*in_ptr++);
    goto FINALIZE; }

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

  FINALIZE:
  for (; number < num_points; number++) {
    update_square_sum_1_val(&S[0], &T[0], &number, in_ptr);
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

  unsigned int number = 1;

  __VOLK_ATTR_ALIGNED(32) float T[16] = {0.f};
  __VOLK_ATTR_ALIGNED(32) float S[16] = {0.f};

  if (num_points < 16) {   
    T[0] = (*in_ptr++);
    goto FINALIZE; }

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

  FINALIZE:
  for (; number < num_points; number++) {
    update_square_sum_1_val(&S[0], &T[0], &number, in_ptr);
    T[0] += (*in_ptr++);
  }    

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_AVX */


#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */