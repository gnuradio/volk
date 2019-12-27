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

static inline float
square_sum_update_1(float* S_tot, float* T_tot, uint32_t* N, const float* val){
  float ret = 0.f;
  float N_f = (float) (*N);
  ret += (*S_tot);
  //ret += 1.f/((*N)*((*N)+1))*( (*N)*(*val) - (*T_tot) )*( (*N)*(*val) - (*T_tot) );
  ret += 1.f/(N_f*(N_f+1.f))*( N_f*(*val) - (*T_tot) )*( N_f*(*val) - (*T_tot) );
  return ret;
}

static inline float
square_sum_update_equal_N(float* S0, float* T0, float* S1, float* T1, uint32_t N){
  float ret = 0.f;
  ret += (*S0);
  ret += (*S1);
  ret += 1.f/(2.f*N)*( (*T0) - (*T1) )*( (*T0) - (*T1) );
  return ret;
}

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_stddev_and_mean_32f_x2_generic(float* stddev, float* mean,
                                                      const float* inputBuffer,
                                                      unsigned int num_points)
{
  // Youngs and Cramer's Algorithm for calculating std and mean  
  const float* in_ptr = inputBuffer;

  float T = (*in_ptr++);
  float S = 0.f;
  uint32_t number = 1;

  for (; number < num_points; number++)
  {
    float v = (*in_ptr++);
    float n = (float) number;
    T += v;
    S += 1.f/( n*(n + 1.f) )*( (n + 1.f)*v - T )*( (n + 1.f)*v - T ); 
  }

  *mean = T/num_points;
  *stddev  = sqrtf( S/num_points );
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

  __VOLK_ATTR_ALIGNED(16) float T[4];
  __VOLK_ATTR_ALIGNED(16) float S[4];

  __m128 T_acc = _mm_load_ps(in_ptr);
  __m128 S_acc = _mm_setzero_ps();
  __m128 v_reg;
  __m128 x_reg;
  __m128 f_reg;

  in_ptr += 4; // First load into T_accu

  for(;number < qtr_points; number++) {
    v_reg = _mm_load_ps(in_ptr);        // v <- x0 x1 x2 x3
    in_ptr += 4;

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T_acc = _mm_add_ps(T_acc, v_reg);   // T += v

    x_reg = _mm_set_ps1(np1);           // x  = number+1
    x_reg = _mm_mul_ps(x_reg, v_reg);   // x  = (number+1)*v
    x_reg = _mm_sub_ps(x_reg, T_acc);   // x  = (number+1)*v - T_acc
    x_reg = _mm_mul_ps(x_reg, x_reg);   // x  = ((number+1)*v - T_acc)**2
    x_reg = _mm_mul_ps(x_reg, f_reg);   // x  = 1/(n*n(n+1))*((number+1)*v - T_acc)**2
    S_acc = _mm_add_ps(S_acc, x_reg);   // S += x
  }

  _mm_store_ps(T, T_acc);
  _mm_store_ps(S, S_acc);

  float T01, T23, T_tot;
  float S01, S23, S_tot;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T_tot = T01 + T23;

  S01   = square_sum_update_equal_N(&S[0], &T[0], &S[1], &T[1], qtr_points);
  S23   = square_sum_update_equal_N(&S[2], &T[2], &S[3], &T[3], qtr_points);
  S_tot = square_sum_update_equal_N(&S01 , &T01 , &S23 , &T23 , 2*qtr_points);

  number = qtr_points*4;
  for (; number < num_points; number++)
  {
    S_tot = square_sum_update_1(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr);    
    in_ptr++;
  }

  *stddev = sqrtf( S_tot/num_points );
  *mean = T_tot/num_points;
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

  __VOLK_ATTR_ALIGNED(16) float T[4];
  __VOLK_ATTR_ALIGNED(16) float S[4];

  __m128 T_acc = _mm_load_ps(in_ptr);
  __m128 S_acc = _mm_setzero_ps();
  __m128 v_reg;
  __m128 x_reg;
  __m128 f_reg;

  in_ptr += 4; // First load into T_accu

  for(;number < qtr_points; number++) {
    v_reg = _mm_load_ps(in_ptr);        // v <- x0 x1 x2 x3
    in_ptr += 4;

    float n   = (float) number;
    float np1 = n + 1.f;
    f_reg = _mm_set_ps1(  1.f/( n*np1 ) );
    
    T_acc = _mm_add_ps(T_acc, v_reg);   // T += v

    x_reg = _mm_set_ps1(np1);           // x  = number+1
    x_reg = _mm_mul_ps(x_reg, v_reg);   // x  = (number+1)*v
    x_reg = _mm_sub_ps(x_reg, T_acc);   // x  = (number+1)*v - T_acc
    x_reg = _mm_mul_ps(x_reg, x_reg);   // x  = ((number+1)*v - T_acc)**2
    x_reg = _mm_mul_ps(x_reg, f_reg);   // x  = 1/(n*n(n+1))*((number+1)*v - T_acc)**2
    S_acc = _mm_add_ps(S_acc, x_reg);   // S += x
  }

  _mm_store_ps(T, T_acc);
  _mm_store_ps(S, S_acc);

  float T01, T23, T_tot;
  float S01, S23, S_tot;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T_tot = T01 + T23;

  S01   = square_sum_update_equal_N(&S[0], &T[0], &S[1], &T[1], qtr_points);
  S23   = square_sum_update_equal_N(&S[2], &T[2], &S[3], &T[3], qtr_points);
  S_tot = square_sum_update_equal_N(&S01 , &T01 , &S23 , &T23 , 2*qtr_points);

  number = qtr_points*4;
  for (; number < num_points; number++)
  {
    S_tot = square_sum_update_1(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr);    
    in_ptr++;
  }

  *stddev = sqrtf( S_tot/num_points );
  *mean = T_tot/num_points;
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

  __VOLK_ATTR_ALIGNED(32) float T[8];
  __VOLK_ATTR_ALIGNED(32) float S[8];

  __m256 T_acc = _mm256_load_ps(in_ptr);
  __m256 S_acc = _mm256_setzero_ps();
  __m256 v_reg;
  __m256 x_reg;
  __m256 f_reg;

  in_ptr += 8; // First load into T_accu  

  for(;number < eigth_points; number++) {
    v_reg = _mm256_load_ps(in_ptr);        // v <- x0 x1 x2 x3
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;
    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );
    
    T_acc = _mm256_add_ps(T_acc, v_reg);   // T += v

    x_reg = _mm256_set1_ps(np1);           // x  = number+1
    x_reg = _mm256_mul_ps(x_reg, v_reg);   // x  = (number+1)*v
    x_reg = _mm256_sub_ps(x_reg, T_acc);   // x  = (number+1)*v - T_acc
    x_reg = _mm256_mul_ps(x_reg, x_reg);   // x  = ((number+1)*v - T_acc)**2
    x_reg = _mm256_mul_ps(x_reg, f_reg);   // x  = 1/(n*n(n+1))*((number+1)*v - T_acc)**2
    S_acc = _mm256_add_ps(S_acc, x_reg);   // S += x
  }  

  _mm256_store_ps(T, T_acc);
  _mm256_store_ps(S, S_acc);  

  float T01, T23, T45, T67, T0123, T4567, T_tot;
  float S01, S23, S45, S67, S0123, S4567, S_tot;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot = T0123 + T4567;

  S01   = square_sum_update_equal_N(&S[0], &T[0], &S[1], &T[1], eigth_points);
  S23   = square_sum_update_equal_N(&S[2], &T[2], &S[3], &T[3], eigth_points);
  S45   = square_sum_update_equal_N(&S[4], &T[4], &S[5], &T[5], eigth_points);
  S67   = square_sum_update_equal_N(&S[6], &T[6], &S[7], &T[7], eigth_points);

  S0123 = square_sum_update_equal_N(&S01 , &T01 , &S23 , &T23 , 2*eigth_points);  
  S4567 = square_sum_update_equal_N(&S45 , &T45 , &S67 , &T67 , 2*eigth_points);  

  S_tot = square_sum_update_equal_N(&S0123 , &T0123 , &S4567 , &T4567 , 4*eigth_points);  

  number = eigth_points*8;
  for (; number < num_points; number++)
  {
    S_tot = square_sum_update_1(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr);    
    in_ptr++;
  }    

  *stddev = sqrtf( S_tot/num_points );
  *mean = T_tot/num_points;
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

  __VOLK_ATTR_ALIGNED(32) float T[8];
  __VOLK_ATTR_ALIGNED(32) float S[8];

  __m256 T_acc = _mm256_loadu_ps(in_ptr);
  in_ptr += 8;
  __m256 S_acc = _mm256_setzero_ps();
  __m256 v_reg;
  __m256 x_reg;
  __m256 f_reg;

  for(;number < eigth_points; number++) {
    v_reg = _mm256_loadu_ps(in_ptr);        // v <- x0 x1 x2 x3
    in_ptr += 8;

    float n   = (float) number;
    float np1 = number + 1.f;
    f_reg = _mm256_set1_ps(  1.f/( n*np1 ) );    
    
    T_acc = _mm256_add_ps(T_acc, v_reg);   // T += v

    x_reg = _mm256_set1_ps(np1);           // x  = number+1
    x_reg = _mm256_mul_ps(x_reg, v_reg);   // x  = (number+1)*v
    x_reg = _mm256_sub_ps(x_reg, T_acc);   // x  = (number+1)*v - T_acc
    x_reg = _mm256_mul_ps(x_reg, x_reg);   // x  = ((number+1)*v - T_acc)**2
    x_reg = _mm256_mul_ps(x_reg, f_reg);   // x  = 1/(n*n(n+1))*((number+1)*v - T_acc)**2
    S_acc = _mm256_add_ps(S_acc, x_reg);   // S += x
  }  

  _mm256_store_ps(T, T_acc);
  _mm256_store_ps(S, S_acc);  

  float T01, T23, T45, T67, T0123, T4567, T_tot;
  float S01, S23, S45, S67, S0123, S4567, S_tot;
  
  T01 = T[0] + T[1];
  T23 = T[2] + T[3];
  T45 = T[4] + T[5];
  T67 = T[6] + T[7];
  T0123 = T01 + T23;
  T4567 = T45 + T67;
  T_tot = T0123 + T4567;

  S01   = square_sum_update_equal_N(&S[0], &T[0], &S[1], &T[1], eigth_points);
  S23   = square_sum_update_equal_N(&S[2], &T[2], &S[3], &T[3], eigth_points);
  S45   = square_sum_update_equal_N(&S[4], &T[4], &S[5], &T[5], eigth_points);
  S67   = square_sum_update_equal_N(&S[6], &T[6], &S[7], &T[7], eigth_points);

  S0123 = square_sum_update_equal_N(&S01 , &T01 , &S23 , &T23 , 2*eigth_points);  
  S4567 = square_sum_update_equal_N(&S45 , &T45 , &S67 , &T67 , 2*eigth_points);  

  S_tot = square_sum_update_equal_N(&S0123 , &T0123 , &S4567 , &T4567 , 4*eigth_points);  

  number = eigth_points*8;
  for (; number < num_points; number++)
  {
    S_tot = square_sum_update_1(&S_tot, &T_tot, &number, in_ptr);
    T_tot += (*in_ptr);    
    in_ptr++;
  }    

  *stddev = sqrtf( S_tot/num_points );
  *mean = T_tot/num_points;
}
#endif /* LV_HAVE_AVX */


#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */