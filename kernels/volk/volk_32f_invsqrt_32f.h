/* -*- c++ -*- */
/*
 * Copyright 2013, 2014 Free Software Foundation, Inc.
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
 * \page volk_32f_invsqrt_32f
 *
 * \b Overview
 *
 * Computes the inverse square root of the input vector and stores
 * result in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_invsqrt_32f(float* cVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: the input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = 1.0 / (float)(ii*ii);
 *   }
 *
 *   volk_32f_invsqrt_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_invsqrt_32f_a_H
#define INCLUDED_volk_32f_invsqrt_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static inline float
Q_rsqrt(float number)
{
  float x2;
  const float threehalfs = 1.5F;
  union f32_to_i32 {
    int32_t i;
    float f;
  } u;

  x2 = number * 0.5F;
  u.f = number;
  u.i = 0x5f3759df - ( u.i >> 1 );                   // what the fuck?
  u.f = u.f * ( threehalfs - ( x2 * u.f * u.f ) );   // 1st iteration
  //u.f  = u.f * ( threehalfs - ( x2 * u.f * u.f ) );   // 2nd iteration, this can be removed

  return u.f;
}

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_invsqrt_32f_a_avx(float* cVector, const float* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  __m256 aVal, cVal;
  for (; number < eighthPoints; number++) {
    aVal = _mm256_load_ps(aPtr);
    cVal = _mm256_rsqrt_ps(aVal);
    _mm256_store_ps(cPtr, cVal);
    aPtr += 8;
    cPtr += 8;
  }

  number = eighthPoints * 8;
  for(;number < num_points; number++)
    *cPtr++ = Q_rsqrt(*aPtr++);

}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_invsqrt_32f_a_sse(float* cVector, const float* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* cPtr = cVector;
  const float* aPtr = aVector;

  __m128 aVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm_load_ps(aPtr);

    cVal = _mm_rsqrt_ps(aVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++) {
    *cPtr++ = Q_rsqrt(*aPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_invsqrt_32f_neon(float* cVector, const float* aVector, unsigned int num_points)
{
  unsigned int number;
  const unsigned int quarter_points = num_points / 4;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  float32x4_t a_val, c_val;
  for (number = 0; number < quarter_points; ++number) {
    a_val = vld1q_f32(aPtr);
    c_val = vrsqrteq_f32(a_val);
    vst1q_f32(cPtr, c_val);
    aPtr += 4;
    cPtr += 4;
  }

  for(number=quarter_points * 4;number < num_points; number++)
    *cPtr++ = Q_rsqrt(*aPtr++);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_invsqrt_32f_generic(float* cVector, const float* aVector, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;
  for(number = 0; number < num_points; number++) {
    *cPtr++ = Q_rsqrt(*aPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_invsqrt_32f_u_avx(float* cVector, const float* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  __m256 aVal, cVal;
  for (; number < eighthPoints; number++) {
    aVal = _mm256_loadu_ps(aPtr);
    cVal = _mm256_rsqrt_ps(aVal);
    _mm256_storeu_ps(cPtr, cVal);
    aPtr += 8;
    cPtr += 8;
  }

  number = eighthPoints * 8;
  for(;number < num_points; number++)
    *cPtr++ = Q_rsqrt(*aPtr++);

}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_invsqrt_32f_a_H */
