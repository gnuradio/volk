/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <math.h>

/*
static uint32_t 
local_log2(uint32_t val) {
  if      (val ==  8) { return 3; }
  else if (val ==  4) { return 2; }
  else if (val ==  2) { return 1; }
  else if (val == 16) { return 4; }
  else if (val == 32) { return 5; }
  else if (val == 64) { return 6; }
  return 0;
}
*/

static inline uint32_t
local_log2(uint32_t val){
  uint32_t ret = 0;

  while (val >>= 1) {
    ret++;
  }
  return ret;
}


static inline void
update_square_sum_1_val(float* S, const float* T, const uint32_t* N, const float* val){
  float n = (float) (*N);
  (*S) += 1.f/( n*(n + 1.f) ) * ( n*(*val) - (*T) ) * ( n*(*val) - (*T) );
  return;
}

static inline void
update_square_sum_equal_N(float* S, const float* S0, const float* T0, 
                          const float* S1, const float* T1, const uint32_t* N){
  float n = (float) (*N);
  (*S)  = (*S0);  
  (*S) += (*S1);
  (*S) += .5f/n*( (*T0) - (*T1) )*( (*T0) - (*T1) );
  return;
}

static inline void
local_sqaure_add(float* S,  const float* T0, const float* S1, const float* T1, const uint32_t* N){
  float n = (float) (*N);
  (*S) += (*S1);
  (*S) += .5f/n*( (*T0) - (*T1) )*( (*T0) - (*T1) );
  return;
}

static inline void
accrue_square_sum( float* S, float* T, const uint32_t N_accumulators, const uint32_t N_partition){
  // Accrue pairwise
  uint32_t stages = local_log2(N_accumulators);
  uint32_t accs   = N_accumulators;
  uint32_t m = 1;
  uint32_t partition_size = N_partition;

  for (uint32_t s = 0; s < stages; s++ ) {
    accs /= 2;
    uint32_t idx = 0;
    for (uint32_t a = 0; a < accs; a++)  {
      local_sqaure_add( &S[idx] , &T[idx] , &S[idx+m], &T[idx+m], &partition_size);
      idx += 2*m;
    }
    m *= 2;
    partition_size *= 2;
  }
  return;
}

static inline void
accrue_sum( float* T, const uint32_t N_accumulators) {
  for (uint32_t i = 1; i < N_accumulators; i++)
  {
    T[0] += T[i];
  }
  return;
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
  const float* in_ptr = inputBuffer;

  float T = (*in_ptr++);
  float S = 0.f;
  uint32_t number = 1;

  for (; number < num_points; number++) {
    float v = (*in_ptr++);
    float n = (float) number;
    T += v;
    S += 1.f/( n*(n + 1.f) )*( (n + 1.f)*v - T )*( (n + 1.f)*v - T ); 
  }

  *stddev = sqrtf( S / num_points );
  *mean   = T / num_points;
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_sse(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int qtr_points = num_points / 4;
  const unsigned int half_points = 2 * qtr_points;

  __VOLK_ATTR_ALIGNED(16) float T[4];
  __VOLK_ATTR_ALIGNED(16) float S[4];

  __m128 T_acc = _mm_load_ps(in_ptr);
  in_ptr += 4;

  __m128 S_acc = _mm_setzero_ps();
  __m128 v_reg;
  __m128 x_reg;
  __m128 f_reg;

  for(;number < qtr_points; number++) {
    v_reg = _mm_load_ps(in_ptr);
    in_ptr += 4;

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T_acc = _mm_add_ps(T_acc, v_reg);

    x_reg = _mm_set_ps1(np1);
    x_reg = _mm_mul_ps(x_reg, v_reg);
    x_reg = _mm_sub_ps(x_reg, T_acc);
    x_reg = _mm_mul_ps(x_reg, x_reg);
    x_reg = _mm_mul_ps(x_reg, f_reg);
    S_acc = _mm_add_ps(S_acc, x_reg);
  }

  _mm_store_ps(T, T_acc);
  _mm_store_ps(S, S_acc);

  float T01, T23, T_tot;
  float S01 = 0.f, S23 = 0.f, S_tot = 0.f;

  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T_tot = T01 + T23;

  update_square_sum_equal_N( &S01,   &S[0], &T[0], &S[1], &T[1],  &qtr_points);
  update_square_sum_equal_N( &S23,   &S[2], &T[2], &S[3], &T[3],  &qtr_points);
  update_square_sum_equal_N( &S_tot,  &S01,  &T01,  &S23,  &T23, &half_points);

  number = qtr_points*4;

  for (; number < num_points; number++) {
    update_square_sum_1_val(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr++);    
  }

  *stddev = sqrtf( S_tot / num_points );
  *mean   = T_tot / num_points;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_sse(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int qtr_points = num_points / 4;
  const unsigned int half_points = 2 * qtr_points;

  __VOLK_ATTR_ALIGNED(16) float T[4];
  __VOLK_ATTR_ALIGNED(16) float S[4];

  __m128 T_acc = _mm_loadu_ps(in_ptr);
  in_ptr += 4;

  __m128 S_acc = _mm_setzero_ps();
  __m128 v_reg;
  __m128 x_reg;
  __m128 f_reg;


  for(;number < qtr_points; number++) {
    v_reg = _mm_loadu_ps(in_ptr);
    in_ptr += 4;

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T_acc = _mm_add_ps(T_acc, v_reg);

    x_reg = _mm_set_ps1(np1);
    x_reg = _mm_mul_ps(x_reg, v_reg);
    x_reg = _mm_sub_ps(x_reg, T_acc);
    x_reg = _mm_mul_ps(x_reg, x_reg);
    x_reg = _mm_mul_ps(x_reg, f_reg);
    S_acc = _mm_add_ps(S_acc, x_reg);
  }

  _mm_store_ps(T, T_acc);
  _mm_store_ps(S, S_acc);


  float T01, T23, T_tot;
  float S01 = 0.f, S23 = 0.f, S_tot = 0.f;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T_tot = T01 + T23;

  update_square_sum_equal_N(&S01,   &S[0], &T[0], &S[1], &T[1],  &qtr_points);
  update_square_sum_equal_N(&S23,   &S[2], &T[2], &S[3], &T[3],  &qtr_points);
  update_square_sum_equal_N(&S_tot,  &S01,  &T01,  &S23,  &T23, &half_points);

  S[0] = S_tot;
  T[0] = T_tot;

  /*
  accrue_square_sum( S, T, 4, qtr_points);
  accrue_sum( T , 4);
  */

  number = qtr_points*4;

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
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int eigth_points = num_points / 8;
  const unsigned int qtr_points = 2 * eigth_points;
  const unsigned int half_points = 2 * qtr_points;

  __VOLK_ATTR_ALIGNED(32) float T[8];
  __VOLK_ATTR_ALIGNED(32) float S[8];

  __m256 T_acc = _mm256_load_ps(in_ptr);
  in_ptr += 8;

  __m256 S_acc = _mm256_setzero_ps();
  __m256 v_reg;
  __m256 x_reg;
  __m256 f_reg;

  for(;number < eigth_points; number++) {
    v_reg = _mm256_load_ps(in_ptr);
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;
    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    
    T_acc = _mm256_add_ps(T_acc, v_reg);

    x_reg = _mm256_set1_ps(np1);
    x_reg = _mm256_mul_ps(x_reg, v_reg);
    x_reg = _mm256_sub_ps(x_reg, T_acc);
    x_reg = _mm256_mul_ps(x_reg, x_reg);
    x_reg = _mm256_mul_ps(x_reg, f_reg);
    S_acc = _mm256_add_ps(S_acc, x_reg);
  }  

  _mm256_store_ps(T, T_acc);
  _mm256_store_ps(S, S_acc);  

  float T01, T23, T45, T67, T0123, T4567, T_tot;
  float S01 = 0.f, S23 = 0.f, S45 = 0.f, S67 = 0.f, S0123 = 0.f, S4567 = 0.f, S_tot = 0.f;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[0], &T[0], &S[1], &T[1], &eigth_points);
  update_square_sum_equal_N(&S23, &S[2], &T[2], &S[3], &T[3], &eigth_points);
  update_square_sum_equal_N(&S45, &S[4], &T[4], &S[5], &T[5], &eigth_points);
  update_square_sum_equal_N(&S67, &S[6], &T[6], &S[7], &T[7], &eigth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &qtr_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &qtr_points);  

  update_square_sum_equal_N(&S_tot, &S0123 , &T0123 , &S4567 , &T4567 , &half_points);  

  number = eigth_points*8;

  for (; number < num_points; number++) {
    update_square_sum_1_val(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr++);
  }    

  *stddev = sqrtf( S_tot / num_points );
  *mean   = T_tot / num_points;
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_avx_2(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int sixteenth_points = num_points / 16;
  const unsigned int eigth_points     = 2 * sixteenth_points;
  const unsigned int qtr_points       = 2 * eigth_points;
  const unsigned int half_points      = 2 * qtr_points;

  __VOLK_ATTR_ALIGNED(32) float T[16];
  __VOLK_ATTR_ALIGNED(32) float S[16];

  __m256 T0_acc = _mm256_load_ps(in_ptr);
  in_ptr += 8;
  __m256 T1_acc = _mm256_load_ps(in_ptr);
  in_ptr += 8;

  __m256 S0_acc = _mm256_setzero_ps();
  __m256 S1_acc = _mm256_setzero_ps();
  __m256 v0_reg, v1_reg;
  __m256 x0_reg, x1_reg;
  __m256 f0_reg, f1_reg;

  for(;number < sixteenth_points; number++) {
    v0_reg = _mm256_load_ps(in_ptr);
    in_ptr += 8;

    v1_reg = _mm256_load_ps(in_ptr);
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;

    f0_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    f1_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    
    T0_acc = _mm256_add_ps(T0_acc, v0_reg);
    T1_acc = _mm256_add_ps(T1_acc, v1_reg);

    x0_reg = _mm256_set1_ps(np1);
    x1_reg = _mm256_set1_ps(np1);

    x0_reg = _mm256_mul_ps(x0_reg, v0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, v1_reg);

    x0_reg = _mm256_sub_ps(x0_reg, T0_acc);
    x1_reg = _mm256_sub_ps(x1_reg, T1_acc);

    x0_reg = _mm256_mul_ps(x0_reg, x0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, x1_reg);

    x0_reg = _mm256_mul_ps(x0_reg, f0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, f1_reg);

    S0_acc = _mm256_add_ps(S0_acc, x0_reg);
    S1_acc = _mm256_add_ps(S1_acc, x1_reg);
  }  

  _mm256_store_ps(&T[0], T0_acc);
  _mm256_store_ps(&T[8], T1_acc);
  _mm256_store_ps(&S[0], S0_acc);  
  _mm256_store_ps(&S[8], S1_acc);  

  float T01, T23, T45, T67, T0123, T4567, T_tot0;
  float S01 = 0.f, S23 = 0.f, S45 = 0.f, S67 = 0.f, S0123 = 0.f, S4567 = 0.f, S_tot0 = 0.f;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot0 = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[0], &T[0], &S[1], &T[1], &sixteenth_points);
  update_square_sum_equal_N(&S23, &S[2], &T[2], &S[3], &T[3], &sixteenth_points);
  update_square_sum_equal_N(&S45, &S[4], &T[4], &S[5], &T[5], &sixteenth_points);
  update_square_sum_equal_N(&S67, &S[6], &T[6], &S[7], &T[7], &sixteenth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &eigth_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &eigth_points);  

  update_square_sum_equal_N(&S_tot0, &S0123 , &T0123 , &S4567 , &T4567 , &qtr_points);  

  float S_tot1 = 0.f;
  S01 = 0.f; S23 = 0.f; S45 = 0.f; S67 = 0.f; S0123 = 0.f; S4567 = 0.f;

  T01 = T[8] + T[9];
  T23 = T[10] + T[11];
  T45 = T[12] + T[13];
  T67 = T[14] + T[15];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  float T_tot1 = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[8], &T[8], &S[9], &T[9], &sixteenth_points);
  update_square_sum_equal_N(&S23, &S[10], &T[10], &S[11], &T[11], &sixteenth_points);
  update_square_sum_equal_N(&S45, &S[12], &T[12], &S[13], &T[13], &sixteenth_points);
  update_square_sum_equal_N(&S67, &S[14], &T[14], &S[15], &T[15], &sixteenth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &eigth_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &eigth_points);    

  update_square_sum_equal_N(&S_tot1, &S0123 , &T0123 , &S4567 , &T4567 , &qtr_points); 

  float S_tot = 0.f;
  float T_tot = 0.f;
  T_tot  = T_tot0 + T_tot1;

  update_square_sum_equal_N(&S_tot, &S_tot0 , &T_tot0 , &S_tot1 , &T_tot1 , &half_points);

  number = sixteenth_points*16;

  for (; number < num_points; number++) {
    update_square_sum_1_val(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr++);
  }    

  *stddev = sqrtf( S_tot / num_points );
  *mean   = T_tot / num_points;
}
#endif /* LV_HAVE_AVX */



#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_avx(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int eigth_points = num_points / 8;
  //const unsigned int qtr_points = 2 * eigth_points;
  //const unsigned int half_points = 2 * qtr_points;  

  __VOLK_ATTR_ALIGNED(32) float T[8];
  __VOLK_ATTR_ALIGNED(32) float S[8];

  __m256 T_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;

  __m256 S_acc = _mm256_setzero_ps();
  __m256 v_reg;
  __m256 x_reg;
  __m256 f_reg;

  for(;number < eigth_points; number++) {
    v_reg = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;
    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );    
    
    T_acc = _mm256_add_ps(T_acc, v_reg);

    x_reg = _mm256_set1_ps(np1);
    x_reg = _mm256_mul_ps(x_reg, v_reg);
    x_reg = _mm256_sub_ps(x_reg, T_acc);
    x_reg = _mm256_mul_ps(x_reg, x_reg);
    x_reg = _mm256_mul_ps(x_reg, f_reg);
    S_acc = _mm256_add_ps(S_acc, x_reg);
  }

  _mm256_store_ps(T, T_acc);
  _mm256_store_ps(S, S_acc);  

  /*

  float T01, T23, T45, T67, T0123, T4567, T_tot;
  float S01 = 0.f, S23 = 0.f, S45 = 0.f, S67 = 0.f, S0123 = 0.f, S4567 = 0.f, S_tot = 0.f;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[0], &T[0], &S[1], &T[1], &eigth_points);
  update_square_sum_equal_N(&S23, &S[2], &T[2], &S[3], &T[3], &eigth_points);
  update_square_sum_equal_N(&S45, &S[4], &T[4], &S[5], &T[5], &eigth_points);
  update_square_sum_equal_N(&S67, &S[6], &T[6], &S[7], &T[7], &eigth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &qtr_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &qtr_points);  

  update_square_sum_equal_N(&S_tot, &S0123 , &T0123 , &S4567 , &T4567 , &half_points);

  */
  accrue_square_sum( S, T, 8, eigth_points);
  accrue_sum( T, 8);

  /*

  T[0] += T[1];
  T[0] += T[2];
  T[0] += T[3];
  T[0] += T[4];
  T[0] += T[5];
  T[0] += T[6];
  T[0] += T[7];
  */

  number = eigth_points*8;

  for (; number < num_points; number++) {
    update_square_sum_1_val( &S[0], &T[0], &number, in_ptr );
    T[0] += (*in_ptr++);
  }    

  *stddev = sqrtf( S[0] / num_points );
  *mean   = T[0] / num_points;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_avx_2(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  const float* in_ptr = inputBuffer;

  unsigned int number = 1;
  const unsigned int sixteenth_points = num_points / 16;
  const unsigned int eigth_points     = 2 * sixteenth_points;
  const unsigned int qtr_points       = 2 * eigth_points;
  const unsigned int half_points      = 2 * qtr_points;

  __VOLK_ATTR_ALIGNED(32) float T[16];
  __VOLK_ATTR_ALIGNED(32) float S[16];

  __m256 T0_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;
  __m256 T1_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;

  __m256 S0_acc = _mm256_setzero_ps();
  __m256 S1_acc = _mm256_setzero_ps();
  __m256 v0_reg, v1_reg;
  __m256 x0_reg, x1_reg;
  __m256 f0_reg, f1_reg;

  for(;number < sixteenth_points; number++) {
    v0_reg = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;

    v1_reg = _mm256_loadu_ps(in_ptr);
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;

    f0_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    f1_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    
    T0_acc = _mm256_add_ps(T0_acc, v0_reg);
    T1_acc = _mm256_add_ps(T1_acc, v1_reg);

    x0_reg = _mm256_set1_ps(np1);
    x1_reg = _mm256_set1_ps(np1);

    x0_reg = _mm256_mul_ps(x0_reg, v0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, v1_reg);

    x0_reg = _mm256_sub_ps(x0_reg, T0_acc);
    x1_reg = _mm256_sub_ps(x1_reg, T1_acc);

    x0_reg = _mm256_mul_ps(x0_reg, x0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, x1_reg);

    x0_reg = _mm256_mul_ps(x0_reg, f0_reg);
    x1_reg = _mm256_mul_ps(x1_reg, f1_reg);

    S0_acc = _mm256_add_ps(S0_acc, x0_reg);
    S1_acc = _mm256_add_ps(S1_acc, x1_reg);
  }  

  _mm256_store_ps(&T[0], T0_acc);
  _mm256_store_ps(&T[8], T1_acc);
  _mm256_store_ps(&S[0], S0_acc);  
  _mm256_store_ps(&S[8], S1_acc);  

  float T01, T23, T45, T67, T0123, T4567, T_tot0;
  float S01 = 0.f, S23 = 0.f, S45 = 0.f, S67 = 0.f, S0123 = 0.f, S4567 = 0.f, S_tot0 = 0.f;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot0 = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[0], &T[0], &S[1], &T[1], &sixteenth_points);
  update_square_sum_equal_N(&S23, &S[2], &T[2], &S[3], &T[3], &sixteenth_points);
  update_square_sum_equal_N(&S45, &S[4], &T[4], &S[5], &T[5], &sixteenth_points);
  update_square_sum_equal_N(&S67, &S[6], &T[6], &S[7], &T[7], &sixteenth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &eigth_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &eigth_points);  

  update_square_sum_equal_N(&S_tot0, &S0123 , &T0123 , &S4567 , &T4567 , &qtr_points);  

  float S_tot1 = 0.f;
  S01 = 0.f; S23 = 0.f; S45 = 0.f; S67 = 0.f; S0123 = 0.f; S4567 = 0.f;

  T01 = T[8] + T[9];
  T23 = T[10] + T[11];
  T45 = T[12] + T[13];
  T67 = T[14] + T[15];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  float T_tot1 = T0123 + T4567;

  update_square_sum_equal_N(&S01, &S[8], &T[8], &S[9], &T[9], &sixteenth_points);
  update_square_sum_equal_N(&S23, &S[10], &T[10], &S[11], &T[11], &sixteenth_points);
  update_square_sum_equal_N(&S45, &S[12], &T[12], &S[13], &T[13], &sixteenth_points);
  update_square_sum_equal_N(&S67, &S[14], &T[14], &S[15], &T[15], &sixteenth_points);

  update_square_sum_equal_N(&S0123, &S01 , &T01 , &S23 , &T23 , &eigth_points);  
  update_square_sum_equal_N(&S4567, &S45 , &T45 , &S67 , &T67 , &eigth_points);    

  update_square_sum_equal_N(&S_tot1, &S0123 , &T0123 , &S4567 , &T4567 , &qtr_points); 

  float S_tot = 0.f;
  float T_tot = 0.f;
  T_tot  = T_tot0 + T_tot1;

  update_square_sum_equal_N(&S_tot, &S_tot0 , &T_tot0 , &S_tot1 , &T_tot1 , &half_points);

  number = sixteenth_points*16;

  for (; number < num_points; number++) {
    update_square_sum_1_val(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr++);
  }    

  *stddev = sqrtf( S_tot / num_points );
  *mean   = T_tot / num_points;
}
#endif /* LV_HAVE_AVX */


#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */