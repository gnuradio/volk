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
#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_avx(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  float stdDev = 0;
  float newMean = 0;
  if(num_points > 0){
    unsigned int number = 0;
    const unsigned int thirtySecondthPoints = num_points / 32;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float meanBuffer[8];
    __VOLK_ATTR_ALIGNED(32) float squareBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 squareAccumulator = _mm256_setzero_ps();
    __m256 aVal1, aVal2, aVal3, aVal4;
    __m256 cVal1, cVal2, cVal3, cVal4;
    for(;number < thirtySecondthPoints; number++) {
      aVal1 = _mm256_load_ps(aPtr); aPtr += 8;
      cVal1 = _mm256_dp_ps(aVal1, aVal1, 0xF1);
      accumulator = _mm256_add_ps(accumulator, aVal1);  // accumulator += x

      aVal2 = _mm256_load_ps(aPtr); aPtr += 8;
      cVal2 = _mm256_dp_ps(aVal2, aVal2, 0xF2);
      accumulator = _mm256_add_ps(accumulator, aVal2);  // accumulator += x

      aVal3 = _mm256_load_ps(aPtr); aPtr += 8;
      cVal3 = _mm256_dp_ps(aVal3, aVal3, 0xF4);
      accumulator = _mm256_add_ps(accumulator, aVal3);  // accumulator += x

      aVal4 = _mm256_load_ps(aPtr); aPtr += 8;
      cVal4 = _mm256_dp_ps(aVal4, aVal4, 0xF8);
      accumulator = _mm256_add_ps(accumulator, aVal4);  // accumulator += x

      cVal1 = _mm256_or_ps(cVal1, cVal2);
      cVal3 = _mm256_or_ps(cVal3, cVal4);
      cVal1 = _mm256_or_ps(cVal1, cVal3);

      squareAccumulator = _mm256_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
    }
    _mm256_store_ps(meanBuffer,accumulator); // Store the results back into the C container
    _mm256_store_ps(squareBuffer,squareAccumulator); // Store the results back into the C container
    newMean = meanBuffer[0];
    newMean += meanBuffer[1];
    newMean += meanBuffer[2];
    newMean += meanBuffer[3];
    newMean += meanBuffer[4];
    newMean += meanBuffer[5];
    newMean += meanBuffer[6];
    newMean += meanBuffer[7];
    stdDev = squareBuffer[0];
    stdDev += squareBuffer[1];
    stdDev += squareBuffer[2];
    stdDev += squareBuffer[3];
    stdDev += squareBuffer[4];
    stdDev += squareBuffer[5];
    stdDev += squareBuffer[6];
    stdDev += squareBuffer[7];

    number = thirtySecondthPoints * 32;
    for(;number < num_points; number++){
      stdDev += (*aPtr) * (*aPtr);
      newMean += *aPtr++;
    }
    newMean /= num_points;
    stdDev /= num_points;
    stdDev -= (newMean * newMean);
    stdDev = sqrtf(stdDev);
  }
  *stddev = stdDev;
  *mean = newMean;

}
#endif /* LV_HAVE_AVX */


/*
#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_u_avx(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  float stdDev = 0;
  float newMean = 0;
  if(num_points > 0){
    unsigned int number = 0;
    const unsigned int thirtySecondthPoints = num_points / 32;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float meanBuffer[8];
    __VOLK_ATTR_ALIGNED(32) float squareBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 squareAccumulator = _mm256_setzero_ps();
    __m256 aVal1, aVal2, aVal3, aVal4;
    __m256 cVal1, cVal2, cVal3, cVal4;
    for(;number < thirtySecondthPoints; number++) {
      aVal1 = _mm256_loadu_ps(aPtr); aPtr += 8;
      cVal1 = _mm256_dp_ps(aVal1, aVal1, 0xF1);
      accumulator = _mm256_add_ps(accumulator, aVal1);  // accumulator += x

      aVal2 = _mm256_loadu_ps(aPtr); aPtr += 8;
      cVal2 = _mm256_dp_ps(aVal2, aVal2, 0xF2);
      accumulator = _mm256_add_ps(accumulator, aVal2);  // accumulator += x

      aVal3 = _mm256_loadu_ps(aPtr); aPtr += 8;
      cVal3 = _mm256_dp_ps(aVal3, aVal3, 0xF4);
      accumulator = _mm256_add_ps(accumulator, aVal3);  // accumulator += x

      aVal4 = _mm256_loadu_ps(aPtr); aPtr += 8;
      cVal4 = _mm256_dp_ps(aVal4, aVal4, 0xF8);
      accumulator = _mm256_add_ps(accumulator, aVal4);  // accumulator += x

      cVal1 = _mm256_or_ps(cVal1, cVal2);
      cVal3 = _mm256_or_ps(cVal3, cVal4);
      cVal1 = _mm256_or_ps(cVal1, cVal3);

      squareAccumulator = _mm256_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
    }
    _mm256_store_ps(meanBuffer,accumulator); // Store the results back into the C container
    _mm256_store_ps(squareBuffer,squareAccumulator); // Store the results back into the C container
    newMean = meanBuffer[0];
    newMean += meanBuffer[1];
    newMean += meanBuffer[2];
    newMean += meanBuffer[3];
    newMean += meanBuffer[4];
    newMean += meanBuffer[5];
    newMean += meanBuffer[6];
    newMean += meanBuffer[7];
    stdDev = squareBuffer[0];
    stdDev += squareBuffer[1];
    stdDev += squareBuffer[2];
    stdDev += squareBuffer[3];
    stdDev += squareBuffer[4];
    stdDev += squareBuffer[5];
    stdDev += squareBuffer[6];
    stdDev += squareBuffer[7];

    number = thirtySecondthPoints * 32;
    for(;number < num_points; number++){
      stdDev += (*aPtr) * (*aPtr);
      newMean += *aPtr++;
    }
    newMean /= num_points;
    stdDev /= num_points;
    stdDev -= (newMean * newMean);
    stdDev = sqrtf(stdDev);
  }
  *stddev = stdDev;
  *mean = newMean;

}
#endif /* LV_HAVE_AVX */

/*
#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
static inline void
volk_32f_stddev_and_mean_32f_x2_a_sse4_1(float* stddev, float* mean,
                                         const float* inputBuffer,
                                         unsigned int num_points)
{
  float returnValue = 0;
  float newMean = 0;
  if(num_points > 0){
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float meanBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float squareBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 squareAccumulator = _mm_setzero_ps();
    __m128 aVal1, aVal2, aVal3, aVal4;
    __m128 cVal1, cVal2, cVal3, cVal4;
    for(;number < sixteenthPoints; number++) {
      aVal1 = _mm_load_ps(aPtr); aPtr += 4;
      cVal1 = _mm_dp_ps(aVal1, aVal1, 0xF1);
      accumulator = _mm_add_ps(accumulator, aVal1);  // accumulator += x

      aVal2 = _mm_load_ps(aPtr); aPtr += 4;
      cVal2 = _mm_dp_ps(aVal2, aVal2, 0xF2);
      accumulator = _mm_add_ps(accumulator, aVal2);  // accumulator += x

      aVal3 = _mm_load_ps(aPtr); aPtr += 4;
      cVal3 = _mm_dp_ps(aVal3, aVal3, 0xF4);
      accumulator = _mm_add_ps(accumulator, aVal3);  // accumulator += x

      aVal4 = _mm_load_ps(aPtr); aPtr += 4;
      cVal4 = _mm_dp_ps(aVal4, aVal4, 0xF8);
      accumulator = _mm_add_ps(accumulator, aVal4);  // accumulator += x

      cVal1 = _mm_or_ps(cVal1, cVal2);
      cVal3 = _mm_or_ps(cVal3, cVal4);
      cVal1 = _mm_or_ps(cVal1, cVal3);

      squareAccumulator = _mm_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
    }
    _mm_store_ps(meanBuffer,accumulator); // Store the results back into the C container
    _mm_store_ps(squareBuffer,squareAccumulator); // Store the results back into the C container
    newMean = meanBuffer[0];
    newMean += meanBuffer[1];
    newMean += meanBuffer[2];
    newMean += meanBuffer[3];
    returnValue = squareBuffer[0];
    returnValue += squareBuffer[1];
    returnValue += squareBuffer[2];
    returnValue += squareBuffer[3];

    number = sixteenthPoints * 16;
    for(;number < num_points; number++){
      returnValue += (*aPtr) * (*aPtr);
      newMean += *aPtr++;
    }
    newMean /= num_points;
    returnValue /= num_points;
    returnValue -= (newMean * newMean);
    returnValue = sqrtf(returnValue);
  }
  *stddev = returnValue;
  *mean = newMean;
}
#endif /* LV_HAVE_SSE4_1 */

/*
#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_stddev_and_mean_32f_x2_a_sse(float* stddev, float* mean,
                                      const float* inputBuffer,
                                      unsigned int num_points)
{
  float returnValue = 0;
  float newMean = 0;
  if(num_points > 0){
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float meanBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float squareBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 squareAccumulator = _mm_setzero_ps();
    __m128 aVal = _mm_setzero_ps();
    for(;number < quarterPoints; number++) {
      aVal = _mm_load_ps(aPtr);                     // aVal = x
      accumulator = _mm_add_ps(accumulator, aVal);  // accumulator += x
      aVal = _mm_mul_ps(aVal, aVal);                // squareAccumulator += x^2
      squareAccumulator = _mm_add_ps(squareAccumulator, aVal);
      aPtr += 4;
    }
    _mm_store_ps(meanBuffer,accumulator); // Store the results back into the C container
    _mm_store_ps(squareBuffer,squareAccumulator); // Store the results back into the C container
    newMean = meanBuffer[0];
    newMean += meanBuffer[1];
    newMean += meanBuffer[2];
    newMean += meanBuffer[3];
    returnValue = squareBuffer[0];
    returnValue += squareBuffer[1];
    returnValue += squareBuffer[2];
    returnValue += squareBuffer[3];

    number = quarterPoints * 4;
    for(;number < num_points; number++){
      returnValue += (*aPtr) * (*aPtr);
      newMean += *aPtr++;
    }
    newMean /= num_points;
    returnValue /= num_points;
    returnValue -= (newMean * newMean);
    returnValue = sqrtf(returnValue);
  }
  *stddev = returnValue;
  *mean = newMean;
}
#endif /* LV_HAVE_SSE */



/*
#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_stddev_and_mean_32f_x2_generic_welford(float* stddev, float* mean,
                                                  const float* inputBuffer,
                                                  unsigned int num_points)
{
  // Welford's Algorithm for calculating std and mean
  const float* in_ptr = inputBuffer;
  float T = (*in_ptr++);
  float S = 0.f;
  uint32_t number = 1;

  for (;number < num_points; number++)
  {
    float T_old = T;
    float v = (*in_ptr++);
    T += (v - T)/( number + 1);
    S += (v - T)*( v - T_old );
  }

  *mean = T;
  *stddev = sqrtf( S/num_points );
}
#endif /* LV_HAVE_GENERIC */


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
    T += v;
    S += 1.f/( number*(number + 1) )*( (number+1)*v - T )*( (number+1)*v - T ); 
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

    float np1 = number + 1.f;
    f_reg = _mm_set_ps1(  1.f/( number*np1 ) );
    
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

  float T_tot = (T[0] + T[1]) + (T[2] + T[3]);
  float S01 = S[0] + S[1] + 1.f/(2*qtr_points)*( T[0] - T[1] )*( T[0] - T[1] );
  float S23 = S[2] + S[3] + 1.f/(2*qtr_points)*( T[2] - T[3] )*( T[2] - T[3] );
  float S_tot = S01 + S23 + 1.f/(4*qtr_points)*( (T[0]+T[1]) - (T[2]+T[3]) )*( (T[0]+T[1]) - (T[2]+T[3]) );  

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
    
    float np1 = number + 1.f;
    f_reg = _mm256_set1_ps(  1.f/( number*np1 ) );
    
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

  float T_tot = ((T[0] + T[1]) + (T[2] + T[3])) + ((T[4] + T[5]) + (T[6] + T[7]));
  float S01 = S[0] + S[1] + 1.f/(2*eigth_points)*( T[0] - T[1] )*( T[0] - T[1] );
  float S23 = S[2] + S[3] + 1.f/(2*eigth_points)*( T[2] - T[3] )*( T[2] - T[3] );
  float S45 = S[4] + S[5] + 1.f/(2*eigth_points)*( T[4] - T[5] )*( T[4] - T[5] );
  float S67 = S[6] + S[7] + 1.f/(2*eigth_points)*( T[6] - T[7] )*( T[6] - T[7] );

  float S0123 = S01 + S23 + 
    1.f/(4*eigth_points)*( (T[0]+T[1]) - (T[2]+T[3]) )*( (T[0]+T[1]) - (T[2]+T[3]) );
  float S4567 = S45 + S67 + 
    1.f/(4*eigth_points)*( (T[4]+T[5]) - (T[6]+T[7]) )*( (T[4]+T[5]) - (T[6]+T[7]) );  

  float S_tot = S0123 + S4567 + 1.f/num_points*
    ( (T[0]+T[1]+T[2]+T[3]) - (T[4]+T[5]+T[6]+T[7]) )*( (T[0]+T[1]+T[2]+T[3]) - (T[4]+T[5]+T[6]+T[7]) );

  *stddev = sqrtf( S_tot/num_points );
  *mean = T_tot/num_points;
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_stddev_and_mean_32f_x2_a_H */