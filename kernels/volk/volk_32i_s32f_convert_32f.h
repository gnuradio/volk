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
 * \page volk_32i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts the samples in the inputVector from 32-bit integers into
 * floating point values and then divides them by the input scalar.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_s32f_convert_32f(float* outputVector, const int32_t* inputVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of 32-bit integers.
 * \li scalar: The value that the output is divided by after being converted to a float.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li complexVector: The output vector of floats.
 *
 * \b Example
 * Convert full-range integers to floats in range [0,1].
 * \code
 *   int N = 1<<8;
 *   unsigned int alignment = volk_get_alignment();
 *
 *   int32_t* x = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   float* z = (float*)volk_malloc(N*sizeof(float), alignment);
 *   float scale = (float)N;
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       x[ii] = ii;
 *   }
 *
 *   volk_32i_s32f_convert_32f(z, x, scale, N);
 *
 *   volk_free(x);
 *   volk_free(z);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_s32f_convert_32f_u_H
#define INCLUDED_volk_32i_s32f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32i_s32f_convert_32f_u_sse2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m128i inputVal;
  __m128 ret;

  for(;number < quarterPoints; number++){
    // Load the 4 values
    inputVal = _mm_loadu_si128((__m128i*)inputPtr);

    ret = _mm_cvtepi32_ps(inputVal);
    ret = _mm_mul_ps(ret, invScalar);

    _mm_storeu_ps(outputVectorPtr, ret);

    outputVectorPtr += 4;
    inputPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_s32f_convert_32f_generic(float* outputVector, const int32_t* inputVector,
                                  const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int32_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32i_s32f_convert_32f_u_H */



#ifndef INCLUDED_volk_32i_s32f_convert_32f_a_H
#define INCLUDED_volk_32i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32i_s32f_convert_32f_a_sse2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m128i inputVal;
  __m128 ret;

  for(;number < quarterPoints; number++){
    // Load the 4 values
    inputVal = _mm_load_si128((__m128i*)inputPtr);

    ret = _mm_cvtepi32_ps(inputVal);
    ret = _mm_mul_ps(ret, invScalar);

    _mm_store_ps(outputVectorPtr, ret);

    outputVectorPtr += 4;
    inputPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_s32f_convert_32f_a_generic(float* outputVector, const int32_t* inputVector,
                                    const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int32_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32i_s32f_convert_32f_a_H */
